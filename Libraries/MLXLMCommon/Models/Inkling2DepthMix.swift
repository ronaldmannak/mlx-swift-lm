// Copyright © 2026 Apple Inc.
// Depth-mix fusion adapted from mlx-swift-lm PR #821 by aleroot.

import Foundation
import MLX

struct Inkling2FusedDepthMixResult {
    let residual: MLXArray
    let normed: MLXArray
    /// The unweighted float32 depth readout, `[B, 1, Q, D]`.
    let readout: MLXArray
    /// The normalized, weighted readout in the requested output dtype, when supplied.
    let normalizedReadout: MLXArray?
    let state: MLXArray?
}

/// One unpadded decode position. The caller excludes training and padded/masked execution.
/// Unit RMS uses its existing 1e-6 epsilon; `eps` belongs to the optional weighted norm.
func inkling2FusedDepthMix(
    blocks: MLXArray, base: MLXArray?,
    convolution: (x: MLXArray, state: MLXArray, weight: MLXArray)?,
    query: MLXArray, normWeight: MLXArray?, eps: Float, outputType: DType = .bfloat16
) -> Inkling2FusedDepthMixResult? {
    guard inkling2UsesKernels(), blocks.ndim == 4, blocks.dtype == .float32,
        blocks.dim(2) == 1, (0 ... 16).contains(blocks.dim(0)),
        (1 ... 8).contains(blocks.dim(1)), (1 ... 4096).contains(blocks.dim(3)),
        [1, 2].contains(query.ndim), query.dim(-1) == blocks.dim(3),
        [.float32, .bfloat16, .float16].contains(query.dtype),
        eps.isFinite, eps > 0, [.float32, .bfloat16, .float16].contains(outputType),
        base != nil || convolution != nil
    else { return nil }
    let (k, batch, dimensions) = (blocks.dim(0), blocks.dim(1), blocks.dim(3))
    let queries = query.ndim == 1 ? 1 : query.dim(0)
    guard (1 ... 3).contains(queries) else { return nil }
    let shape = [batch, 1, dimensions]
    if let base {
        guard base.shape == shape, [.float32, .bfloat16, .float16].contains(base.dtype) else {
            return nil
        }
    }
    if let convolution {
        guard convolution.x.shape == shape,
            [.float32, .bfloat16, .float16].contains(convolution.x.dtype),
            convolution.state.shape == [batch, 3, dimensions], convolution.state.dtype == .float32,
            convolution.weight.shape == [dimensions, 4, 1],
            [.float32, .bfloat16, .float16].contains(convolution.weight.dtype)
        else { return nil }
    }
    if let normWeight {
        guard [.float32, .bfloat16].contains(normWeight.dtype),
            normWeight.shape == [dimensions] || normWeight.shape == [1, dimensions]
                || normWeight.shape == [queries, dimensions]
        else { return nil }
    }
    let dummy = MLXArray([Float(0)])
    let depthThreads = min(1024, max(32, dimensions / 8 / 32 * 32))
    let rmsThreads = ((dimensions + 3) / 4 + 31) / 32 * 32
    let threads = max(depthThreads, rmsThreads)
    let output = Inkling2DepthMixKernels.shared.mix(
        [
            blocks, base ?? dummy, convolution?.x ?? dummy, convolution?.state ?? dummy,
            convolution?.weight ?? dummy, query.asType(.float32).reshaped(queries, dimensions),
            normWeight?.reshaped(-1, dimensions) ?? dummy,
            MLXArray([pow(Float(dimensions), -0.5), eps, sqrt(Float(5)), Float(dimensions), 1e-6]),
        ],
        template: [
            ("T", convolution?.x.dtype ?? DType.float32),
            ("W", convolution?.weight.dtype ?? DType.float32),
            ("NT", outputType),
            ("D", dimensions), ("K", k), ("Q", queries), ("DTG", depthThreads),
            ("NW", (normWeight?.size ?? dimensions) / dimensions),
            ("HAS_BASE", base != nil), ("HAS_CONV", convolution != nil),
            ("NORM_OUT", normWeight != nil),
        ],
        grid: (batch * threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [
            shape, shape, [batch, 1, queries, dimensions],
            normWeight != nil ? [batch, 1, queries, dimensions] : [1],
            convolution != nil ? [batch, 3, dimensions] : [1],
        ],
        outputDTypes: [.float32, .float32, .float32, outputType, .float32])
    return Inkling2FusedDepthMixResult(
        residual: output[0], normed: output[1], readout: output[2],
        normalizedReadout: normWeight != nil ? output[3] : nil,
        state: convolution != nil ? output[4] : nil)
}

private final class Inkling2DepthMixKernels: Sendable {
    static let shared = Inkling2DepthMixKernels()
    let mix: MLXFast.MLXFastKernel

    private init() {
        mix = MLXFast.metalKernel(
            name: "inkling2_fused_depth_mix",
            inputNames: [
                "blocks", "base", "x", "state", "conv_weight", "queries", "norm_weight", "params",
            ],
            outputNames: ["residual_out", "normed_out", "readout", "normalized", "state_out"],
            source: inkling2DepthMixSource)
    }
}

private let inkling2DepthMixSource = """
    constexpr int N = K + 1;
    constexpr int DNR = DTG / 32;
    const uint row = threadgroup_position_in_grid.x;
    const uint rows = threadgroups_per_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t base_row = size_t(row) * D;
    threadgroup float rms_sums[Q][32];
    threadgroup float rms_inverse[Q];
    threadgroup float depth_sums[N][DNR];
    threadgroup float depth_weights[N * Q];

    // Native rms_single_row: four contiguous values per lane and two SIMD reductions.
    float residuals[4];
    float square = 0.0f;
    for (int i = 0; i < 4; i++) {
        const int d = int(tid) * 4 + i;
        float r = 0.0f;
        if (d < D) {
            if (HAS_BASE) r = float(base[base_row + d]);
            if (HAS_CONV) {
                const size_t sb = size_t(row) * 3 * D + d;
                const float xv = float(x[base_row + d]);
                float a = 0.0f;
                for (int tap = 0; tap < 4; tap++) {
                    const float v = tap < 3 ? state[sb + size_t(tap) * D] : xv;
                    a += float(W(v)) * float(conv_weight[size_t(d) * 4 + tap]);
                }
                r += float(T((float(W(a)) + xv) / params[2]));
                state_out[sb] = state[sb + D];
                state_out[sb + D] = state[sb + 2 * D];
                state_out[sb + 2 * D] = xv;
            }
            residual_out[base_row + d] = r;
        }
        residuals[i] = r;
        square += r * r;
    }
    square = simd_sum(square);
    if (sg == 0) rms_sums[0][lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) rms_sums[0][sg] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        square = simd_sum(rms_sums[0][lane]);
        // Xcode's native RMS shader uses fast-math reciprocal multiplication.
        if (lane == 0) {
            const float mean = square * metal::fast::divide(1.0f, params[3]) + params[4];
            rms_inverse[0] = metal::precise::rsqrt(mean);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rms_inverse[0];
    for (int i = 0; i < 4; i++) {
        const int d = int(tid) * 4 + i;
        if (d < D) normed_out[base_row + d] = residuals[i] * inv;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Preserve the existing depth kernel's lane assignment and accumulation order.
    // One query at a time bounds registers when RMS needs 1024 threads at D4096.
    for (int j = 0; j < Q; j++) {
        if (tid < DTG) {
            float acc[N];
            for (int i = 0; i < N; i++) acc[i] = 0.0f;
            for (int d = tid; d < D; d += DTG) {
                const float qv = queries[j * D + d];
                for (int i = 0; i < K; i++) {
                    acc[i] += blocks[(size_t(i) * rows + row) * D + d] * qv;
                }
                acc[K] += normed_out[base_row + d] * qv;
            }
            for (int i = 0; i < N; i++) {
                const float value = simd_sum(acc[i]);
                if (lane == 0) depth_sums[i][sg] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (int i = 0; i < N; i++) {
                float total = 0.0f;
                for (int s = 0; s < DNR; s++) total += depth_sums[i][s];
                depth_weights[i * Q + j] = total * params[0];
            }
            float maximum = -INFINITY;
            for (int i = 0; i < N; i++) maximum = max(maximum, depth_weights[i * Q + j]);
            float total = 0.0f;
            for (int i = 0; i < N; i++) {
                depth_weights[i * Q + j] = metal::exp(depth_weights[i * Q + j] - maximum);
                total += depth_weights[i * Q + j];
            }
            for (int i = 0; i < N; i++) depth_weights[i * Q + j] /= total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < DTG) {
        for (int d = tid; d < D; d += DTG) {
            for (int j = 0; j < Q; j++) {
                float output = 0.0f;
                for (int i = 0; i < K; i++) {
                    output += depth_weights[i * Q + j]
                        * blocks[(size_t(i) * rows + row) * D + d];
                }
                output += depth_weights[K * Q + j] * normed_out[base_row + d];
                readout[(size_t(row) * Q + j) * D + d] = output;
            }
        }
    }
    if (!NORM_OUT) return;
    threadgroup_barrier(mem_flags::mem_device);

    // The optional weighted RMS uses the same native F32 reduction on each query.
    float readouts[Q * 4];
    float squares[Q];
    for (int j = 0; j < Q; j++) {
        squares[j] = 0.0f;
        for (int i = 0; i < 4; i++) {
            const int d = int(tid) * 4 + i;
            const float value = d < D ? readout[(size_t(row) * Q + j) * D + d] : 0.0f;
            readouts[j * 4 + i] = value;
            squares[j] += value * value;
        }
        squares[j] = simd_sum(squares[j]);
        if (sg == 0) rms_sums[j][lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        for (int j = 0; j < Q; j++) rms_sums[j][sg] = squares[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int j = 0; j < Q; j++) {
            const float total = simd_sum(rms_sums[j][lane]);
            if (lane == 0) {
                const float mean = total * metal::fast::divide(1.0f, params[3]) + params[1];
                rms_inverse[j] = metal::precise::rsqrt(mean);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < Q; j++) {
        for (int i = 0; i < 4; i++) {
            const int d = int(tid) * 4 + i;
            if (d < D) {
                const float value = readouts[j * 4 + i] * rms_inverse[j];
                normalized[(size_t(row) * Q + j) * D + d] =
                    NT(value * float(norm_weight[(NW == 1 ? 0 : j) * D + d]));
            }
        }
    }
    """
