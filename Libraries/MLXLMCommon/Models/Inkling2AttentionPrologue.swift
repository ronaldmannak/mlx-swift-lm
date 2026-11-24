// Copyright © 2026 Apple Inc.

// Adapted from aleroot/mlx-swift-lm PR #821, ba8b1af3 (MIT). Preserve the original
// float32 convolution state, convolution casts and scalar relative-clip reduction.

import MLX
import MLXNN

private let inkling2AttentionPrologueSource = """
        constexpr int E = DH / 32;
        constexpr int KV = HKV * DH;
        const int unit = threadgroup_position_in_grid.x;
        const int row = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        if (unit < HQ) {
            const int h = unit;
            float states[E];
            float total = 0.0f;
            for (int e = 0; e < E; e++) {
                const int d = lane * E + e;
                states[e] = float(q[row * q_strides[0] + (h * DH + d) * q_strides[2]])
                    * float(q_scale[h * q_scale_strides[0] + d * q_scale_strides[1]]);
                const float square = states[e] * states[e];
                total = square + total;
            }
            total = simd_sum(0.0f + total);
            const float bound = params[0] / (metal::precise::sqrt(total) + 1e-6f);
            const float factor = metal::isnan(bound) || bound < 1.0f ? bound : 1.0f;
            for (int e = 0; e < E; e++) {
                q_out[(size_t(row) * HQ + h) * DH + lane * E + e] = T(states[e] * factor);
            }
            return;
        }
        if (unit < 2 * HQ) {
            const int h = unit - HQ;
            float total = 0.0f;
            for (int d = 0; d < DR; d++) {
                const float state = float(r[row * r_strides[0] + (h * DR + d) * r_strides[2]])
                    * float(r_scale[h * r_scale_strides[0] + d * r_scale_strides[1]]);
                const float square = state * state;
                total = square + total;
            }
            const float bound = params[2] / (metal::precise::sqrt(total) + 1e-6f);
            const float factor = metal::isnan(bound) || bound < 1.0f ? bound : 1.0f;
            if (lane < DR) {
                const float state = float(r[row * r_strides[0] + (h * DR + lane) * r_strides[2]])
                    * float(r_scale[h * r_scale_strides[0] + lane * r_scale_strides[1]]);
                r_out[(size_t(row) * HQ + h) * DR + lane] = T(state * factor);
            }
            return;
        }
        const bool is_key = unit < 2 * HQ + HKV;
        const int h = unit - 2 * HQ - (is_key ? 0 : HKV);
        float convolved[E];
        float total = 0.0f;
        for (int e = 0; e < E; e++) {
            const int d = lane * E + e;
            const int c = h * DH + d;
            const float x = is_key
                ? float(k[row * k_strides[0] + c * k_strides[2]])
                : float(v[row * v_strides[0] + c * v_strides[2]]);
            float previous[3];
            for (int tap = 0; tap < 3; tap++) {
                previous[tap] = is_key
                    ? k_state[row * k_state_strides[0] + tap * k_state_strides[1]
                        + c * k_state_strides[2]]
                    : v_state[row * v_state_strides[0] + tap * v_state_strides[1]
                        + c * v_state_strides[2]];
            }
            float accumulated = 0.0f;
            for (int tap = 0; tap < 4; tap++) {
                if (DYNAMIC) {
                    const float source = tap == 0 ? x : previous[3 - tap];
                    const float weight = is_key
                        ? float(k_weight[tap * k_weight_strides[0] + c * k_weight_strides[1]])
                        : float(v_weight[tap * v_weight_strides[0] + c * v_weight_strides[1]]);
                    const float adjustment = is_key
                        ? k_taps[row * k_taps_strides[0] + (h * 4 + tap) * k_taps_strides[2]]
                        : v_taps[row * v_taps_strides[0] + (h * 4 + tap) * v_taps_strides[2]];
                    const float product = source * (weight + adjustment);
                    accumulated = product + accumulated;
                } else {
                    const float source = tap == 3 ? x : previous[tap];
                    const float rounded = is_key ? float(KW(source)) : float(VW(source));
                    const float weight = is_key
                        ? float(k_weight[c * k_weight_strides[0] + tap * k_weight_strides[1]])
                        : float(v_weight[c * v_weight_strides[0] + tap * v_weight_strides[1]]);
                    accumulated += rounded * weight;
                }
            }
            const float convolution = DYNAMIC ? accumulated
                : (is_key ? float(KW(accumulated)) : float(VW(accumulated)));
            const float y = float(T((convolution + x) / params[4]));
            convolved[e] = y;
            const float square = y * y;
            total = square + total;
            device float* next = is_key ? k_state_out : v_state_out;
            const size_t at = size_t(row) * 3 * KV + c;
            next[at] = previous[1];
            next[at + KV] = previous[2];
            next[at + 2 * KV] = x;
        }
        total = simd_sum(0.0f + total);
        float factor;
        if (is_key) {
            const float bound = params[1] / (metal::precise::sqrt(total) + 1e-6f);
            factor = metal::isnan(bound) || bound < 1.0f ? bound : 1.0f;
        } else {
            factor = metal::precise::rsqrt(total / DH + params[3]);
        }
        device T* output = is_key ? k_out : v_out;
        for (int e = 0; e < E; e++) {
            output[(size_t(row) * HKV + h) * DH + lane * E + e] = T(convolved[e] * factor);
        }
    """

private final class Inkling2AttentionPrologueKernel: Sendable {
    static let shared = Inkling2AttentionPrologueKernel()

    let kernel = MLXFast.metalKernel(
        name: "inkling2_attention_prologue_float32_state",
        inputNames: [
            "q", "r", "k", "v", "k_taps", "v_taps", "k_weight", "v_weight",
            "k_state", "v_state", "q_scale", "r_scale", "params",
        ],
        outputNames: ["q_out", "r_out", "k_out", "v_out", "k_state_out", "v_state_out"],
        source: inkling2AttentionPrologueSource, ensureRowContiguous: false)
}

extension Inkling2Attention {
    /// One decode token through the existing projections, then fused convolution and norms.
    /// Other paths keep their existing state, training and rewind behavior.
    func fusedDecodePrologue(
        query: MLXArray, key: MLXArray, value: MLXArray,
        states: (MLXArray?, MLXArray?), mask: MLXArray?, lengths: MLXArray?,
        keepWindows: Bool = false
    ) -> (
        queries: MLXArray, relative: MLXArray, keys: MLXArray, values: MLXArray,
        gate: MLXArray, sinks: MLXArray, keyState: MLXArray, valueState: MLXArray
    )? {
        guard inkling2UsesKernels(), !training, !keepWindows, mask == nil, lengths == nil,
            query.ndim == 3, query.dim(0) > 0, query.dim(1) == 1,
            key.shape == query.shape, value.shape == query.shape,
            [query, key, value].allSatisfy({ $0.dtype == .bfloat16 }),
            headDim == 128, heads > 0, kvHeads > 0,
            queryClip.scale.shape == [heads, headDim], relativeClip.scale.ndim == 2,
            relativeClip.scale.dim(0) == heads,
            (1 ... 32).contains(relativeClip.scale.dim(1)),
            [queryClip.scale, relativeClip.scale].allSatisfy({
                [.float16, .bfloat16, .float32].contains($0.dtype)
            }),
            let keyState = states.0, let valueState = states.1,
            keyState.dtype == .float32, valueState.dtype == .float32,
            keyState.shape == [query.dim(0), 3, kvHeads * headDim],
            valueState.shape == keyState.shape,
            !keyConvolution.training, !valueConvolution.training
        else { return nil }

        let keyWeight: MLXArray
        let valueWeight: MLXArray
        let dynamic: (Inkling2DynamicConvolution, Inkling2DynamicConvolution)?
        if let k = keyConvolution as? Inkling2ShortConvolution,
            let v = valueConvolution as? Inkling2ShortConvolution,
            k.kernelSize == 4, v.kernelSize == 4
        {
            keyWeight = k.conv1d.weight
            valueWeight = v.conv1d.weight
            dynamic = nil
            guard keyWeight.shape == [kvHeads * headDim, 4, 1],
                valueWeight.shape == keyWeight.shape
            else { return nil }
        } else if let k = keyConvolution as? Inkling2DynamicConvolution,
            let v = valueConvolution as? Inkling2DynamicConvolution,
            k.kernelSize == 4, v.kernelSize == 4, k.heads == kvHeads, v.heads == kvHeads
        {
            keyWeight = k.staticWeight
            valueWeight = v.staticWeight
            dynamic = (k, v)
            guard keyWeight.shape == [4, kvHeads * headDim], valueWeight.shape == keyWeight.shape
            else { return nil }
        } else {
            return nil
        }
        guard
            [keyWeight, valueWeight].allSatisfy({
                [.float16, .bfloat16, .float32].contains($0.dtype)
            })
        else { return nil }

        let rawKeys = kProjection(key)
        let keyTaps = dynamic?.0.weightProjection(key).asType(.float32)
        let rawValues = vProjection(value)
        let valueTaps = dynamic?.1.weightProjection(value).asType(.float32)
        let rawQueries = qProjection(query)
        let rawRelative = rProjection(query)
        let batch = query.dim(0)
        let relativeDim = relativeClip.scale.dim(1)
        guard [rawQueries, rawRelative, rawKeys, rawValues].allSatisfy({ $0.dtype == .bfloat16 }),
            rawQueries.shape == [batch, 1, heads * headDim],
            rawRelative.shape == [batch, 1, heads * relativeDim],
            rawKeys.shape == [batch, 1, kvHeads * headDim], rawValues.shape == rawKeys.shape,
            keyTaps.map({ $0.shape == [batch, 1, kvHeads * 4] }) ?? true,
            valueTaps.map({ $0.shape == [batch, 1, kvHeads * 4] }) ?? true
        else { return nil }
        let output = Inkling2AttentionPrologueKernel.shared.kernel(
            [
                rawQueries, rawRelative, rawKeys, rawValues,
                keyTaps ?? rawKeys, valueTaps ?? rawValues, keyWeight, valueWeight,
                keyState, valueState, queryClip.scale, relativeClip.scale,
                MLXArray([
                    queryClip.threshold, qkClipThreshold, relativeClip.threshold, eps,
                    Float(dynamic == nil ? 5 : 9).squareRoot(),
                ]),
            ],
            template: [
                ("T", query.dtype), ("KW", keyWeight.dtype), ("VW", valueWeight.dtype),
                ("DH", headDim), ("HQ", heads), ("HKV", kvHeads), ("DR", relativeDim),
                ("DYNAMIC", dynamic != nil),
            ],
            grid: ((2 * heads + 2 * kvHeads) * 32, batch, 1), threadGroup: (32, 1, 1),
            outputShapes: [
                [batch, 1, heads, headDim], [batch, 1, heads, relativeDim],
                [batch, kvHeads, 1, headDim], [batch, kvHeads, 1, headDim],
                keyState.shape, valueState.shape,
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16, .bfloat16, .float32, .float32])
        return (
            output[0], output[1], output[2], output[3], gateProjection(query),
            sinkProjection(query),
            output[4], output[5]
        )
    }
}
