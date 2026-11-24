// Copyright © 2026 Apple Inc.

// SIMD selection adapted from aleroot/mlx-swift-lm PR #821, ba8b1af3 (MIT).
// Keep the existing router's ordered-float keys and normalization arithmetic.

import MLX

private let inkling2SIMDRouterSource = """
        constexpr int R = E / 32;
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        float scores[R];
        uint keys[R];
        bool available[R];
        for (int i = 0; i < R; i++) {
            const uint expert = lane + 32 * i;
            const float x = logits[size_t(row) * E + expert];
            const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            const float score = (x < 0) ? y : 1 - y;
            const float value = score + bias[expert];
            const uint bits = (value == 0.0f) ? 0u : as_type<uint>(value);
            keys[i] = isnan(value) ? 0xFFFFFFFFu
                : (bits ^ (uint(int(bits) >> 31) | 0x80000000u));
            scores[i] = score;
            available[i] = true;
        }
        float chosen[K];
        for (int pick = 0; pick < K; pick++) {
            uint best_key = 0;
            uint best_expert = E;
            for (int i = 0; i < R; i++) {
                const uint expert = lane + 32 * i;
                if (available[i] && (keys[i] > best_key
                    || (keys[i] == best_key && expert < best_expert))) {
                    best_key = keys[i];
                    best_expert = expert;
                }
            }
            const uint top = simd_max(best_key);
            const uint winner = simd_min(best_key == top ? best_expert : uint(E));
            uint score_bits = 0;
            for (int i = 0; i < R; i++) {
                if (lane + 32 * i == winner) {
                    score_bits = as_type<uint>(scores[i]);
                    available[i] = false;
                }
            }
            // Select bits without summing scores, retaining NaN payloads and signed zero.
            chosen[pick] = as_type<float>(simd_or(score_bits));
            if (lane == 0) {
                indices[size_t(row) * K + pick] = winner;
            }
        }
        if (lane == 0) {
            float total = 0.0f;
            for (int pick = 0; pick < K; pick++) {
                total = chosen[pick] + total;
            }
            for (int pick = 0; pick < K; pick++) {
                weights[size_t(row) * K + pick] = (chosen[pick] / total) * route_scale[0];
            }
        }
    """

private final class Inkling2SIMDRouterKernels: Sendable {
    static let shared = Inkling2SIMDRouterKernels()

    let router = MLXFast.metalKernel(
        name: "inkling2_simd_router",
        inputNames: ["logits", "bias", "route_scale"], outputNames: ["indices", "weights"],
        source: inkling2SIMDRouterSource)
}

/// The original router's top eight of 128 experts, using one SIMD group per row.
/// Unsupported shapes keep the existing path; callers must also exclude training.
func inkling2SIMDRouterKernel(
    logits: MLXArray, bias: MLXArray, k: Int, routeScale: Float
) -> (indices: MLXArray, weights: MLXArray)? {
    guard inkling2UsesKernels(), logits.ndim > 0, logits.dim(-1) == 128,
        logits.dtype == .float32, bias.shape == [128], bias.dtype == .float32,
        k == 8, (1 ... 8).contains(logits.size / 128)
    else { return nil }
    let rows = logits.size / 128
    let shape = Array(logits.shape.dropLast()) + [k]
    let output = Inkling2SIMDRouterKernels.shared.router(
        [logits, bias, MLXArray([routeScale])], template: [("E", 128), ("K", k)],
        grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
        outputShapes: [shape, shape], outputDTypes: [.uint32, .float32])
    return (output[0], output[1])
}
