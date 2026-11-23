// Copyright © 2026 Apple Inc.
// Quantized expert kernels adapted from mlx-swift-lm PR #821 by aleroot.

import Foundation
import MLX
import MLXNN

/// The current parameters of a supported expert projection, without cached weight views.
struct Inkling2FusedExpertTensors {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let globalScale: MLXArray?
    let mode: QuantizationMode
    let bits: Int
    let groupSize: Int

    init?(_ projection: SwitchLinear, inputDimensions: Int) {
        guard
            type(of: projection) == QuantizedSwitchLinear.self
                || type(of: projection) == Inkling2ScaledSwitchLinear.self,
            let quantized = projection as? QuantizedSwitchLinear,
            !projection.training, projection.bias == nil,
            projection.weight.ndim == 3, projection.weight.dtype == .uint32,
            quantized.scales.ndim == 3,
            [4, 8].contains(quantized.bits), quantized.groupSize > 0,
            quantized.groupSize % 16 == 0, inputDimensions % 16 == 0,
            inputDimensions % quantized.groupSize == 0,
            projection.weight.dim(2) * 32 == inputDimensions * quantized.bits,
            quantized.scales.shape == [
                projection.weight.dim(0), projection.weight.dim(1),
                inputDimensions / quantized.groupSize,
            ]
        else { return nil }
        switch (quantized.mode, quantized.bits, quantized.groupSize) {
        case (.affine, _, _):
            guard let biases = quantized.biases, biases.shape == quantized.scales.shape,
                [.float16, .bfloat16, .float32].contains(quantized.scales.dtype),
                biases.dtype == quantized.scales.dtype
            else { return nil }
        case (.nvfp4, 4, 16), (.mxfp4, 4, 32), (.mxfp8, 8, 32):
            guard quantized.biases == nil, quantized.scales.dtype == .uint8 else { return nil }
        default:
            return nil
        }
        let global = (projection as? Inkling2ScaledSwitchLinear)?.globalScale
        if let global {
            guard global.dtype == .float32, global.shape == [projection.weight.dim(0)] else {
                return nil
            }
        }
        weight = projection.weight
        scales = quantized.scales
        biases = quantized.biases
        globalScale = global
        mode = quantized.mode
        bits = quantized.bits
        groupSize = quantized.groupSize
    }

    fileprivate var inputs: [MLXArray] {
        [weight, scales, biases ?? MLXArray([Float(0)]), globalScale ?? MLXArray([Float(1)])]
    }

    fileprivate func template(_ dtype: DType) -> [(String, any KernelTemplateArg)] {
        [
            ("T", dtype), ("BITS", bits), ("GS", groupSize),
            ("AFFINE", mode == .affine), ("HAS_GS", globalScale != nil),
        ]
    }
}

extension Inkling2SparseMoE {
    /// One-token expert projections, weighted sum, convolution and residual in two kernels.
    /// Padding and masks must be excluded by the caller. Other shapes retain the general path.
    func fusedExpertDecode(
        _ x: MLXArray, residual: MLXArray, state: MLXArray?, convolutionWeight: MLXArray,
        keepWindow: Bool
    ) -> (MLXArray, MLXArray)? {
        guard inkling2UsesKernels(), !training, !router.training, !experts.training,
            shardingGroup == nil,
            x.ndim == 3, x.dim(1) == 1, x.dim(0) > 0,
            [.bfloat16, .float16].contains(x.dtype), residual.shape == x.shape,
            residual.dtype == .float32, let state, state.dtype == .float32,
            state.shape == [x.dim(0), 3, x.dim(2)],
            convolutionWeight.shape == [x.dim(2), 4, 1],
            [.bfloat16, .float16, .float32].contains(convolutionWeight.dtype),
            router.topK > 0, router.topK <= 32, x.dim(0) * router.topK < 64,
            x.dim(2) % 2 == 0,
            let gateUp = Inkling2FusedExpertTensors(
                experts.gateUpProjection, inputDimensions: x.dim(2)),
            gateUp.weight.dim(1) % 8 == 0,
            let down = Inkling2FusedExpertTensors(
                experts.downProjection, inputDimensions: gateUp.weight.dim(1) / 2),
            down.weight.dim(0) == gateUp.weight.dim(0), down.weight.dim(1) == x.dim(2)
        else { return nil }

        let (batch, dimensions) = (x.dim(0), x.dim(2))
        let (indices, weights) = router(x)
        let routes = indices.asType(.uint32).reshaped(batch, router.topK)
        let activations = inkling2FusedExpertGateUp(
            x.reshaped(batch, dimensions), experts: gateUp, indices: routes,
            softcaps: MLXArray([experts.gateCap, experts.upCap]))
        let (hidden, nextState) = inkling2FusedExpertDown(
            activations, experts: down, indices: routes,
            weights: weights.asType(.float32).reshaped(batch, router.topK),
            residual: residual.reshaped(batch, dimensions), state: state,
            convolutionWeight: convolutionWeight)
        let outputState =
            keepWindow
            ? concatenated([state, nextState[0..., 2 ..< 3, 0...]], axis: 1) : nextState
        return (hidden.reshaped(x.shape), outputState)
    }
}

/// The softcapped activation of alternating gate/up rows, `[B, K, I]`.
func inkling2FusedExpertGateUp(
    _ x: MLXArray, experts: Inkling2FusedExpertTensors, indices: MLXArray,
    softcaps: MLXArray
) -> MLXArray {
    let (rows, k) = (indices.dim(0), indices.dim(1))
    let (dimensions, channels) = (x.dim(1), experts.weight.dim(1) / 2)
    return Inkling2FusedExpertKernels.shared.gateUp(
        [x] + experts.inputs + [indices, softcaps],
        template: experts.template(x.dtype) + [
            ("DIN", dimensions), ("I", channels), ("KE", k), ("NSG", 4),
        ],
        grid: (channels * 32, rows * k, 1), threadGroup: (128, 1, 1),
        outputShapes: [[rows, k, channels]], outputDTypes: [x.dtype])[0]
}

/// Expert down projections and the unchanged convolution arithmetic, with float32 state.
func inkling2FusedExpertDown(
    _ activations: MLXArray, experts: Inkling2FusedExpertTensors, indices: MLXArray,
    weights: MLXArray, residual: MLXArray, state: MLXArray, convolutionWeight: MLXArray
) -> (MLXArray, MLXArray) {
    let (rows, k, inner) = (activations.dim(0), activations.dim(1), activations.dim(2))
    let dimensions = experts.weight.dim(1)
    let output = Inkling2FusedExpertKernels.shared.down(
        [activations] + experts.inputs + [
            indices, weights, residual, state, convolutionWeight, MLXArray([sqrt(Float(5))]),
        ],
        template: experts.template(activations.dtype) + [
            ("W", convolutionWeight.dtype), ("K", inner), ("D", dimensions),
            ("KE", k), ("ROWS", 2),
        ],
        grid: (dimensions / 2 * k * 32, rows, 1), threadGroup: (k * 32, 1, 1),
        outputShapes: [[rows, dimensions], [rows, 3, dimensions]],
        outputDTypes: [activations.dtype, .float32])
    return (output[0], output[1])
}

private final class Inkling2FusedExpertKernels: Sendable {
    static let shared = Inkling2FusedExpertKernels()
    let gateUp: MLXFast.MLXFastKernel
    let down: MLXFast.MLXFastKernel

    private init() {
        let experts = ["w", "scales", "biases", "global_scale", "indices"]
        gateUp = MLXFast.metalKernel(
            name: "inkling2_fused_expert_gate_up", inputNames: ["x"] + experts + ["caps"],
            outputNames: ["act"], source: fusedExpertGateUpSource, header: fusedExpertHeader)
        down = MLXFast.metalKernel(
            name: "inkling2_fused_expert_down",
            inputNames: ["act"] + experts + [
                "weights", "residual", "state", "conv_weight", "divisor",
            ],
            outputNames: ["hidden", "state_out"], source: fusedExpertDownSource,
            header: fusedExpertHeader)
    }
}

private let fusedExpertHeader = """
    template <int GS, bool AFFINE, typename S>
    inline float inkling2_group_scale(S s) {
        if (AFFINE) {
            return float(s);
        }
        const uint8_t b = uint8_t(s);
        if (GS == 16) {
            const ushort v = b & 127;
            const ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (ushort(b & 128) << 8);
            return float(as_type<half>(u)) * 256.0f;
        }
        return as_type<float>(b == 0 ? 0x400000u : (uint(b) << 23));
    }

    // The dot product of 16 weights with x, before the group scale.
    template <int BITS, bool AFFINE>
    inline float inkling2_unit_dot(const device uint8_t* w, thread const float* x) {
        float s = 0.0f;
        if (BITS == 4) {
            const uint2 q = *(const device uint2*)w;
            for (int i = 0; i < 8; i++) {
                const uint n = (i < 4 ? q.x : q.y) >> ((i % 4) * 8);
                if (AFFINE) {
                    s += x[2 * i] * float(n & 15) + x[2 * i + 1] * float((n >> 4) & 15);
                } else {
                    const half lo = as_type<half>(ushort(((n & 8) << 12) | ((n & 7) << 9)));
                    const half hi = as_type<half>(ushort(((n & 128) << 8) | ((n & 112) << 5)));
                    s += x[2 * i] * float(lo) + x[2 * i + 1] * float(hi);
                }
            }
            return AFFINE ? s : s * 16384.0f;
        }
        const uint4 q = *(const device uint4*)w;
        for (int i = 0; i < 16; i++) {
            const uint b = (q[i / 4] >> ((i % 4) * 8)) & 255;
            if (AFFINE) {
                s += x[i] * float(b);
            } else {
                const ushort v = b & 127;
                const ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (ushort(b & 128) << 8);
                s += x[i] * float(as_type<half>(u));
            }
        }
        return AFFINE ? s : s * 256.0f;
    }

    // Each lane's share of the dot products of R consecutive rows with x.
    template <int BITS, int GS, bool AFFINE, int K, int R, typename T, typename S, typename B>
    inline void inkling2_rows_dot(
        const device uint8_t* w, const device S* scales, B biases, const device T* x, uint lane,
        thread float* acc) {
        for (int u = lane; u < K / 16; u += 32) {
            float xv[16];
            float xs = 0.0f;
            for (int i = 0; i < 16; i++) {
                xv[i] = float(x[u * 16 + i]);
                xs += xv[i];
            }
            const int g = u * 16 / GS;
            for (int r = 0; r < R; r++) {
                const float dot = inkling2_unit_dot<BITS, AFFINE>(w + (r * K + u * 16) * BITS / 8, xv);
                acc[r] += inkling2_group_scale<GS, AFFINE>(scales[r * (K / GS) + g]) * dot;
                if (AFFINE) {
                    acc[r] += float(biases[r * (K / GS) + g]) * xs;
                }
            }
        }
    }
    """

private let fusedExpertGateUpSource = """
        const uint lane = thread_index_in_simdgroup;
        const int bj = threadgroup_position_in_grid.y;
        const int c = threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup;
        const uint e = indices[bj];
        const size_t row = size_t(e) * 2 * I + 2 * c;
        float acc[2] = {0.0f, 0.0f};
        inkling2_rows_dot<BITS, GS, AFFINE, DIN, 2>(
            (const device uint8_t*)w + row * (DIN * BITS / 8), scales + row * (DIN / GS),
            biases + (AFFINE ? row * (DIN / GS) : 0), x + size_t(bj / KE) * DIN, lane, acc);
        const float g = simd_sum(acc[0]);
        const float u = simd_sum(acc[1]);
        if (lane == 0) {
            const float s = HAS_GS ? global_scale[e] / (448.0f * 6.0f) : 1.0f;
            // Match the separate matmul and global-scale multiply, including both casts.
            const float rounded_scale = float(T(s));
            const float gate = HAS_GS ? float(T(float(T(g)) * rounded_scale)) : float(T(g));
            const float up = HAS_GS ? float(T(float(T(u)) * rounded_scale)) : float(T(u));
            const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gate)));
            const float sig = gate < 0 ? y : 1.0f - y;
            const float gc = metal::precise::tanh(gate / caps[0]) * caps[0];
            const float uc = metal::precise::tanh(up / caps[1]) * caps[1];
            act[size_t(bj) * I + c] = T(gc * sig * uc);
        }
    """

private let fusedExpertEpilogueTail = """
        const T mb = T(m);
        const size_t sb = size_t(row) * 3 * D + d;
        float a = 0.0f;
        for (int tap = 0; tap < 4; tap++) {
            const float v = tap < 3 ? state[sb + size_t(tap) * D] : float(mb);
            a += float(W(v)) * float(conv_weight[size_t(d) * 4 + tap]);
        }
        const float conv = float(W(a));
        const T y = T((conv + float(mb)) / divisor[0]);
        hidden[size_t(row) * D + d] = T(residual[size_t(row) * D + d] + float(y));
        state_out[sb] = state[sb + D];
        state_out[sb + D] = state[sb + 2 * D];
        state_out[sb + 2 * D] = float(mb);
    """

private let fusedExpertDownSource =
    """
        const uint lane = thread_index_in_simdgroup;
        const uint j = simdgroup_index_in_threadgroup;
        const int row = threadgroup_position_in_grid.y;
        const int d0 = threadgroup_position_in_grid.x * ROWS;
        const uint e = indices[row * KE + j];
        const float s = HAS_GS ? global_scale[e] / (448.0f * 6.0f) : 1.0f;
        threadgroup float outputs[KE * ROWS];
        for (int r = 0; r < ROWS; r++) {
            const size_t n = size_t(e) * D + d0 + r;
            float acc[1] = {0.0f};
            inkling2_rows_dot<BITS, GS, AFFINE, K, 1>(
                (const device uint8_t*)w + n * (K * BITS / 8), scales + n * (K / GS),
                biases + (AFFINE ? n * (K / GS) : 0), act + size_t(row * KE + j) * K, lane, acc);
            const float o = simd_sum(acc[0]);
            if (lane == 0) {
                const float rounded_scale = float(T(s));
                outputs[j * ROWS + r] = HAS_GS
                    ? float(T(float(T(o)) * rounded_scale)) : float(T(o));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = thread_position_in_threadgroup.x;
        if (t >= ROWS) {
            return;
        }
        const int d = d0 + t;
        float m = 0.0f;
        for (int k = 0; k < KE; k++) {
            const float product = outputs[k * ROWS + t] * weights[row * KE + k];
            m = k == 0 ? product : m + product;
        }
    """ + fusedExpertEpilogueTail
