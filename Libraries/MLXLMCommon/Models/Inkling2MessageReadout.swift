// Copyright © 2026 Apple Inc.

// PR #821 groups every core layer's four message queries in one launch. Each group keeps
// the same reductions as Inkling2Kernels' four-query depth attention.

import Foundation
import MLX

private let messageReadoutSource = """
        constexpr int N = K + 1;
        uint row = threadgroup_position_in_grid.x;
        uint rows = threadgroups_per_grid.x;
        uint group = threadgroup_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        const device float* group_queries = queries + size_t(group) * Q * D;

        float acc[N * Q];
        for (int i = 0; i < N * Q; i++) {
            acc[i] = 0.0f;
        }
        for (int d = tid; d < D; d += TG) {
            float x[N];
            for (int i = 0; i < K; i++) {
                x[i] = blocks[(size_t(i) * rows + row) * D + d];
            }
            x[K] = partial[size_t(row) * D + d];
            for (int j = 0; j < Q; j++) {
                float qv = group_queries[j * D + d];
                for (int i = 0; i < N; i++) {
                    acc[i * Q + j] += x[i] * qv;
                }
            }
        }

        threadgroup float buf[N * Q][TG / 32];
        for (int i = 0; i < N * Q; i++) {
            float v = simd_sum(acc[i]);
            if (lane == 0) {
                buf[i][sg] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float w[N * Q];
        for (int i = 0; i < N * Q; i++) {
            float t = 0.0f;
            for (int s = 0; s < TG / 32; s++) {
                t += buf[i][s];
            }
            w[i] = t * scale;
        }
        for (int j = 0; j < Q; j++) {
            float m = -INFINITY;
            for (int i = 0; i < N; i++) {
                m = max(m, w[i * Q + j]);
            }
            float z = 0.0f;
            for (int i = 0; i < N; i++) {
                w[i * Q + j] = metal::exp(w[i * Q + j] - m);
                z += w[i * Q + j];
            }
            for (int i = 0; i < N; i++) {
                w[i * Q + j] /= z;
            }
        }

        const uint total_queries = threadgroups_per_grid.y * Q;
        for (int d = tid; d < D; d += TG) {
            float x[N];
            for (int i = 0; i < K; i++) {
                x[i] = blocks[(size_t(i) * rows + row) * D + d];
            }
            x[K] = partial[size_t(row) * D + d];
            for (int j = 0; j < Q; j++) {
                float o = 0.0f;
                for (int i = 0; i < N; i++) {
                    o += w[i * Q + j] * x[i];
                }
                out[(size_t(row) * total_queries + group * Q + j) * D + d] = o;
            }
        }
    """

private final class Inkling2MessageReadoutKernel: Sendable {
    static let shared = Inkling2MessageReadoutKernel()
    let kernel = MLXFast.metalKernel(
        name: "inkling2_grouped_message_readout",
        inputNames: ["blocks", "partial", "queries", "scale"], outputNames: ["out"],
        source: messageReadoutSource)
}

extension Inkling2MessageInput {
    /// Every loop layer's four messages, float32 `[B, L, layers, 4, D]`.
    /// Training keeps differentiable reference operations; no derived weights are retained.
    func allMessages(_ bank: Inkling2MessageBank) -> MLXArray {
        let dimensions = bank.partial.dim(-1)
        let layers = query.dim(0)
        let shape = Array(bank.partial.shape.dropLast()) + [layers, 4, dimensions]
        guard !training, inkling2UsesKernels(), let blocks = bank.blocks, blocks.dim(0) > 0
        else {
            return inkling2DepthAttention(
                blocks: bank.blocks, partial: bank.partial,
                query: query.reshaped(-1, dimensions), useKernel: false
            ).reshaped(shape)
        }
        let threads = min(1024, max(32, dimensions / 8 / 32 * 32))
        let rows = bank.partial.size / dimensions
        return Inkling2MessageReadoutKernel.shared.kernel(
            [blocks, bank.partial, query.asType(.float32), MLXArray(pow(Float(dimensions), -0.5))],
            template: [("K", blocks.dim(0)), ("D", dimensions), ("Q", 4), ("TG", threads)],
            grid: (rows * threads, layers, 1), threadGroup: (threads, 1, 1),
            outputShapes: [shape], outputDTypes: [.float32])[0]
    }
}
