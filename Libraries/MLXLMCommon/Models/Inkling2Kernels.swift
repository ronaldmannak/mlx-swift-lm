//
//  Inkling2Kernels.swift
//  mlx-swift-lm
//

// Port of `mlx_lm/models/inkling2/kernels.py` from jundot/mlx-lm@0c4c9919 (MIT): Inkling 2's
// depth attention and its prefill attention with a relative-position bias and per-token
// sinks, which MLX's fused attention does not take.
//
// The decode kernels below replace chains of small ops. The short convolution, router and
// expert combine follow exo's Inkling kernels (exo-explore/exo#2530, Apache-2.0).

import Foundation
import MLX

private let depthAttentionSource = """
        constexpr int N = K + 1;
        uint row = threadgroup_position_in_grid.x;
        uint rows = threadgroups_per_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;

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
                float qv = queries[j * D + d];
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
                out[(size_t(row) * Q + j) * D + d] = o;
            }
        }
    """

// Queries per threadgroup: 8 for each of its 4 simdgroups.
private let relativeAttentionQueryBlock = 32
private let relativeAttentionThreads = 128

// The online softmax of each row starts from its sink logit, so the zero-value sink needs
// no key.
private let relativeAttentionSource = """
        constexpr float LOG2E = 1.4426950408889634f;
        constexpr int BK = 16;
        constexpr int LDK = BK + 8;
        constexpr int LDV = D + 8;
        const int L = sizes[0];
        const int S = sizes[1];
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int batch = threadgroup_position_in_grid.z;
        const int head = threadgroup_position_in_grid.y;
        const int kv_head = head / (HQ / HKV);
        const int q0 = threadgroup_position_in_grid.x * BQ;
        const int start = key_start[batch];
        const float qscale = scale * LOG2E;

        // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
        const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
        const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
        const int row = q0 + int(sg) * 8 + fm;
        const bool row_ok = row < L;
        const int diag = S - L + row;

        // q and k are clipped and v has unit RMS, so half holds them.
        threadgroup half Ks[D * LDK];
        threadgroup half Vs[BK * LDV];

        simdgroup_half8x8 q_frag[D / 8];
        for (int db = 0; db < D / 8; ++db) {
            half2 qv = half2(0.0h);
            if (row_ok) {
                const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                    + row * q_strides[2] + db * 8 + fn;
                qv = half2(float(qp[0]), float(qp[1]));
            }
            q_frag[db].thread_elements()[0] = qv.x;
            q_frag[db].thread_elements()[1] = qv.y;
        }
        simdgroup_float8x8 acc[D / 8];
        for (int db = 0; db < D / 8; ++db) {
            acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
        float row_sum = 1.0f;
        const int bias_row = batch * L + (row_ok ? row : 0);
        const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

        const int first_diag = S - L + q0;
        const int last_diag = S - L + min(q0 + BQ, L) - 1;
        int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
        first_key = max(first_key, start);
        const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
        const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

        for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int c = tid; c < BK * D / 8; c += TG) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                if (key < S) {
                    auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                    auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                    kv[0] = kp[0];
                    kv[1] = kp[1];
                    vv[0] = vp[0];
                    vv[1] = vp[1];
                }
                for (int j = 0; j < 8; ++j) {
                    Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                    Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q0 + int(sg) * 8 >= L) {
                continue;
            }

            simdgroup_float8x8 scores[BK / 8];
            for (int kb = 0; kb < BK / 8; ++kb) {
                scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (int db = 0; db < D / 8; ++db) {
                    simdgroup_half8x8 k_frag;
                    simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                    simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
                }
            }

            float tile_max = -INFINITY;
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = diag - key;
                    const bool visible = row_ok && d >= 0 && key >= start
                        && (WINDOW == 0 || d < WINDOW);
                    float s = -INFINITY;
                    if (visible) {
                        s = scores[kb].thread_elements()[e] * qscale;
                        if (d < R) {
                            s += float(bias[d]) * LOG2E;
                        }
                    }
                    scores[kb].thread_elements()[e] = s;
                    tile_max = max(tile_max, s);
                }
            }
            // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
            const float new_max = max(row_max, tile_max);
            const float factor = fast::exp2(row_max - new_max);
            float tile_sum = 0.0f;
            simdgroup_half8x8 probs[BK / 8];
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                    probs[kb].thread_elements()[e] = half(p);
                    tile_sum += p;
                }
            }
            tile_sum += simd_shuffle_xor(tile_sum, 1);
            tile_sum += simd_shuffle_xor(tile_sum, 8);
            row_sum = row_sum * factor + tile_sum;
            row_max = new_max;
            for (int db = 0; db < D / 8; ++db) {
                acc[db].thread_elements()[0] *= factor;
                acc[db].thread_elements()[1] *= factor;
                for (int kb = 0; kb < BK / 8; ++kb) {
                    simdgroup_half8x8 v_frag;
                    simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                    simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
                }
            }
        }

        if (row_ok) {
            device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
            for (int db = 0; db < D / 8; ++db) {
                op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
                op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
            }
        }
    """

// The kernels below replace chains of small ops, so they are written in the ops' order and
// give the same bits.

// One thread per output element, and one per channel for each state row. The convolution
// runs on the window in the weight's dtype and adds in order from zero, as MLX's depthwise
// `conv1d` does.
private let shortConvolutionSource = """
        constexpr int K1 = K - 1;
        const int L = x_shape[1];
        const int C = x_shape[2];
        const int c = int(thread_position_in_grid.x);
        const int t = int(thread_position_in_grid.y);
        const int b = int(thread_position_in_grid.z);
        if (c >= C) {
            return;
        }
        const size_t x0 = size_t(b) * L * C + c;
        const size_t s0 = size_t(b) * K1 * C + c;
        if (t < L) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                const int i = t + j;
                float v = 0.0f;
                if (i < K1) {
                    if (HAS_STATE) {
                        v = state[s0 + size_t(i) * C];
                    }
                } else {
                    v = static_cast<float>(x[x0 + size_t(i - K1) * C]);
                }
                acc += static_cast<float>(static_cast<W>(v)) * static_cast<float>(weight[c * K + j]);
            }
            const float conv = static_cast<float>(static_cast<W>(acc));
            const size_t at = x0 + size_t(t) * C;
            out[at] = static_cast<T>((conv + static_cast<float>(x[at])) / scale);
        } else {
            // The next state is the window's last K - 1 rows.
            float v = 0.0f;
            if (t < K1) {
                if (HAS_STATE) {
                    v = state[s0 + size_t(t) * C];
                }
            } else {
                v = static_cast<float>(x[x0 + size_t(t - K1) * C]);
            }
            new_state[s0 + size_t(t - L) * C] = v;
        }
    """

// `L` queries per row: one thread per head, key and query (`z` is the row's query `b * L +
// l`). Masked keys get -inf; visible keys the bias at their distance, or zero past the profile.
// With SINK_SLOT, the last key, a zero key, gets the query's sink.
private let decodeMaskSource = """
        const int S = stored_shape[1];
        const int L = positions_shape[1];
        const int H = bias_shape[2];
        const int s = int(thread_position_in_grid.x);
        const int h = int(thread_position_in_grid.y);
        const int z = int(thread_position_in_grid.z);
        if (s >= S) {
            return;
        }
        const int b = z / L;
        const int l = z - b * L;
        T value = static_cast<T>(-INFINITY);
        if (SINK_SLOT && s == S - 1) {
            value = static_cast<T>(sinks[size_t(z) * H + h]);
        } else {
            const float stored_position = stored[size_t(b) * S + s];
            const float d = positions[z] - (stored_position - 1.0f);
            const bool visible = d >= 0.0f && stored_position > 0.0f
                && (WINDOW == 0 || d < float(WINDOW));
            if (visible) {
                value = d < float(R)
                    ? bias[(size_t(z) * H + h) * R + int(metal::clamp(d, 0.0f, float(R - 1)))]
                    : static_cast<T>(0.0f);
            }
        }
        out[((size_t(b) * H + h) * L + l) * S + s] = value;
    """

// One threadgroup per row, one thread per expert: MLX's sigmoid, the correction bias, then
// `fusedRouterTopK`'s descending choice and normalization, and the route scale.
private let routerSource = """
        const uint row = threadgroup_position_in_grid.y;
        const uint t = thread_position_in_threadgroup.x;
        threadgroup ulong keys[E];
        threadgroup float top[TOP_K];

        const float x = logits[row * E + t];
        const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        const float score = (x < 0) ? y : 1 - y;
        const float v = score + static_cast<float>(bias[t]);
        const uint bits = (v == 0.0f) ? 0u : as_type<uint>(v);
        const uint mono = isnan(v) ? 0xFFFFFFFFu : (bits ^ ((uint)(((int)bits) >> 31) | 0x80000000u));
        const ulong key = (((ulong)mono) << 32) | (ulong)(0xFFFFFFFFu - t);
        keys[t] = key;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int above = 0;
        for (uint j = 0; j < E; ++j) {
            above += (keys[j] > key) ? 1 : 0;
        }
        if (above < TOP_K) {
            top[above] = score;
            indices[row * TOP_K + above] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t == 0) {
            float total = 0.0f;
            for (int q = 0; q < TOP_K; ++q) {
                total = top[q] + total;
            }
            for (int q = 0; q < TOP_K; ++q) {
                weights[row * TOP_K + q] = (top[q] / total) * route_scale;
            }
        }
    """

// One thread per output element: the experts' outputs times their weights in float32,
// added in order, as MLX's column sum adds a few rows.
private let expertCombineSource = """
        const uint d = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        const int D = outputs_shape[outputs_ndim - 1];
        if (int(d) >= D) {
            return;
        }
        float total = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            const float product = static_cast<float>(outputs[(size_t(row) * TOP_K + k) * D + d])
                * weights[size_t(row) * TOP_K + k];
            total = k == 0 ? product : total + product;
        }
        out[size_t(row) * D + d] = static_cast<OutT>(total);
    """

// One simdgroup per row: the scaled float32 states, their squared norm summed as MLX sums a
// row (one thread in order up to 64 values; otherwise runs of four per lane and a simd sum,
// MLX's looped reduce for fewer than 32 rows), then the clip.
private let l2ClipSource = """
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_position_in_threadgroup.x;
        const device auto* xr = x + size_t(row) * D;
        const int h = int(row) % H;
        auto state = [&](int d) {
            float s = static_cast<float>(xr[d]);
            if (HAS_SCALE) {
                s = s * static_cast<float>(scale[h * D + d]);
            }
            return s;
        };

        float total = 0.0f;
        if (D <= 64) {
            for (int d = 0; d < D; ++d) {
                const float s = state(d);
                const float square = s * s;
                total = square + total;
            }
        } else {
            constexpr int blocks = D / 128;
            constexpr int extra = D - blocks * 128;
            float t = 0.0f;
            int at = int(lane) * 4;
            for (int b = 0; b < blocks; ++b) {
                for (int i = 0; i < 4; ++i) {
                    const float s = state(at + i);
                    const float square = s * s;
                    t = square + t;
                }
                at += 128;
            }
            for (int i = 0; i < 4 && int(lane) * 4 + i < extra; ++i) {
                const float s = state(at + i);
                const float square = s * s;
                t = square + t;
            }
            total = simd_sum(0.0f + t);
        }
        const float bound = threshold / (metal::precise::sqrt(total) + 1e-6f);
        const float factor = metal::isnan(bound) || bound < 1.0f ? bound : 1.0f;
        device T* o = out + size_t(row) * D;
        for (int d = int(lane); d < D; d += 32) {
            o[d] = static_cast<T>(state(d) * factor);
        }
    """

private final class Inkling2Kernels: Sendable {
    static let shared = Inkling2Kernels()
    let depthAttention: MLXFast.MLXFastKernel
    let relativeAttention: MLXFast.MLXFastKernel
    let shortConvolution: MLXFast.MLXFastKernel
    let decodeMask: MLXFast.MLXFastKernel
    let router: MLXFast.MLXFastKernel
    let expertCombine: MLXFast.MLXFastKernel
    let l2Clip: MLXFast.MLXFastKernel

    private init() {
        depthAttention = MLXFast.metalKernel(
            name: "inkling2_depth_attention",
            inputNames: ["blocks", "partial", "queries", "scale"],
            outputNames: ["out"],
            source: depthAttentionSource)
        relativeAttention = MLXFast.metalKernel(
            name: "inkling2_relative_attention",
            inputNames: ["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
            outputNames: ["out"],
            source: relativeAttentionSource,
            ensureRowContiguous: false)
        shortConvolution = MLXFast.metalKernel(
            name: "inkling2_short_convolution",
            inputNames: ["x", "state", "weight", "scale"],
            outputNames: ["out", "new_state"],
            source: shortConvolutionSource)
        decodeMask = MLXFast.metalKernel(
            name: "inkling2_decode_mask",
            inputNames: ["bias", "positions", "stored", "sinks"],
            outputNames: ["out"],
            source: decodeMaskSource)
        router = MLXFast.metalKernel(
            name: "inkling2_router",
            inputNames: ["logits", "bias", "route_scale"],
            outputNames: ["indices", "weights"],
            source: routerSource)
        expertCombine = MLXFast.metalKernel(
            name: "inkling2_expert_combine",
            inputNames: ["outputs", "weights"],
            outputNames: ["out"],
            source: expertCombineSource)
        l2Clip = MLXFast.metalKernel(
            name: "inkling2_l2_clip",
            inputNames: ["x", "scale", "threshold"],
            outputNames: ["out"],
            source: l2ClipSource)
    }
}

/// Threads for a threadgroup that handles one row of `dimensions` values.
private func rowThreads(_ dimensions: Int) -> Int {
    min(1024, max(32, dimensions / 8 / 32 * 32))
}

/// Whether the kernels can run here: on a GPU with Metal.
func inkling2UsesKernels() -> Bool {
    Device.defaultDevice().deviceType == .gpu
}

/// Depth attention in one kernel. `blocks` `[K, ..., D]` and `partial` `[..., D]` are
/// float32 and also the values; `query` is `[Q, D]`. Returns `[..., Q, D]`.
func inkling2DepthAttentionKernel(
    blocks: MLXArray, partial: MLXArray, query: MLXArray, scale: Float
) -> MLXArray {
    let D = partial.dim(-1)
    let Q = query.dim(0)
    let threads = rowThreads(D)
    let rows = partial.size / D
    return Inkling2Kernels.shared.depthAttention(
        [blocks, partial, query, MLXArray(scale)],
        template: [("K", blocks.dim(0)), ("D", D), ("Q", Q), ("TG", threads)],
        grid: (rows * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [Array(partial.shape.dropLast()) + [Q, D]],
        outputDTypes: [.float32])[0]
}

/// Causal attention over keys in temporal order, with the queries at the last `L` keys.
/// `queries` `[B, Hq, L, D]` and `keys`, `values` `[B, Hkv, S, D]` are bfloat16;
/// `positionBias` `[B, L, Hq, R]`; `sinks` `[B, L, Hq]`; `keyStart` `[B]`, the first key of
/// each row. Returns `[B, L, Hq, D]`.
func inkling2RelativeAttentionKernel(
    queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
    sinks: MLXArray, keyStart: MLXArray, scale: Float, window: Int
) -> MLXArray {
    let (B, Hq, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let (Hkv, S) = (keys.dim(1), keys.dim(2))
    let block = relativeAttentionQueryBlock
    return Inkling2Kernels.shared.relativeAttention(
        [
            queries, keys, values, contiguous(positionBias),
            contiguous(sinks.asType(.float32)), contiguous(keyStart.asType(.int32)),
            MLXArray([Int32(L), Int32(S)]), MLXArray(scale),
        ],
        template: [
            ("T", queries.dtype), ("D", D), ("HQ", Hq), ("HKV", Hkv),
            ("R", positionBias.dim(-1)), ("WINDOW", window), ("BQ", block),
            ("TG", relativeAttentionThreads),
        ],
        grid: ((L + block - 1) / block * relativeAttentionThreads, Hq, B),
        threadGroup: (relativeAttentionThreads, 1, 1),
        outputShapes: [[B, L, Hq, D]],
        outputDTypes: [queries.dtype])[0]
}

/// A short convolution, its residual and scale, and the next state in one kernel. `x`
/// `[B, L, C]`; `state` `[B, K - 1, C]` float32, or nil for zeros; `weight` `[C, K, 1]`.
/// Returns the output in `x`'s dtype and the next state.
func inkling2ShortConvolutionKernel(
    _ x: MLXArray, state: MLXArray?, weight: MLXArray
) -> (MLXArray, MLXArray) {
    let (B, L, C) = (x.dim(0), x.dim(1), x.dim(2))
    let K = weight.dim(1)
    let out = Inkling2Kernels.shared.shortConvolution(
        [x, state ?? x, weight, MLXArray(sqrt(Float(K + 1)))],
        template: [
            ("T", x.dtype), ("W", weight.dtype), ("K", K), ("HAS_STATE", state != nil),
        ],
        grid: (C, L + K - 1, B),
        threadGroup: (min(C, 256), 1, 1),
        outputShapes: [x.shape, [B, K - 1, C]],
        outputDTypes: [x.dtype, .float32])
    return (out[0], out[1])
}

/// The additive mask of `L` queries per row: the relative-position bias of each visible key
/// and -inf elsewhere. `positionBias` `[B, L, H, R]`; `queryPositions` `[B, L]` and
/// `storedKeyPositions` `[B, S]` float32. With `sinks` `[B, L, H]`, the last key, which must be
/// a zero key, gets each query's sink instead. Returns `[B, H, L, S]` in `positionBias`'s dtype.
func inkling2DecodeMaskKernel(
    positionBias: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray,
    window: Int, sinks: MLXArray? = nil
) -> MLXArray {
    let (B, L, H, R) = (
        positionBias.dim(0), positionBias.dim(1), positionBias.dim(2), positionBias.dim(3)
    )
    let S = storedKeyPositions.dim(1)
    return Inkling2Kernels.shared.decodeMask(
        [positionBias, queryPositions, storedKeyPositions, sinks ?? queryPositions],
        template: [
            ("T", positionBias.dtype), ("R", R), ("WINDOW", window), ("SINK_SLOT", sinks != nil),
        ],
        grid: (S, H, B * L),
        threadGroup: (min(S, 256), 1, 1),
        outputShapes: [[B, H, L, S]],
        outputDTypes: [positionBias.dtype])[0]
}

/// Inkling 2's router after its matmul: sigmoid scores, top-`k` by score plus `bias`, and
/// the chosen scores normalized and scaled. `logits` `[..., E]` float32, at most 1024
/// experts. Returns `uint32` indices and float32 weights, `[..., k]`, by descending choice.
func inkling2RouterKernel(
    logits: MLXArray, bias: MLXArray, k: Int, routeScale: Float
) -> (indices: MLXArray, weights: MLXArray) {
    let E = logits.dim(-1)
    let shape = Array(logits.shape.dropLast()) + [k]
    let out = Inkling2Kernels.shared.router(
        [logits, bias, MLXArray(routeScale)],
        template: [("E", E), ("TOP_K", k)],
        grid: (E, logits.size / E, 1),
        threadGroup: (E, 1, 1),
        outputShapes: [shape, shape],
        outputDTypes: [.uint32, .float32])
    return (out[0], out[1])
}

/// The routed experts' outputs `[..., k, D]` weighted by `weights` `[..., k]` float32 and
/// summed in float32. Returns `[..., D]` in `dtype`.
func inkling2ExpertCombineKernel(
    _ outputs: MLXArray, weights: MLXArray, dtype: DType
) -> MLXArray {
    let (k, D) = (outputs.dim(-2), outputs.dim(-1))
    return Inkling2Kernels.shared.expertCombine(
        [outputs, weights],
        template: [("OutT", dtype), ("TOP_K", k)],
        grid: (D, outputs.size / (k * D), 1),
        threadGroup: (min(D, 256), 1, 1),
        outputShapes: [Array(outputs.shape.dropLast(2)) + [D]],
        outputDTypes: [dtype])[0]
}

/// Whether ``inkling2L2ClipKernel(_:scale:threshold:)`` sums `x`'s rows as MLX's `sum` does:
/// up to 64 values, or up to 512 in fewer than 32 rows.
func inkling2L2ClipKernelTakes(_ x: MLXArray) -> Bool {
    let D = x.dim(-1)
    return inkling2UsesKernels() && (D <= 64 || (D <= 512 && x.size / D < 32))
}

/// An L2 clip of each row of `x` `[..., H, D]`, after the per-head `scale` `[H, D]`.
func inkling2L2ClipKernel(_ x: MLXArray, scale: MLXArray?, threshold: Float) -> MLXArray {
    let (H, D) = (x.dim(-2), x.dim(-1))
    return Inkling2Kernels.shared.l2Clip(
        [x, scale ?? x, MLXArray(threshold)],
        template: [("T", x.dtype), ("H", H), ("D", D), ("HAS_SCALE", scale != nil)],
        grid: (32, x.size / D, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [x.shape],
        outputDTypes: [x.dtype])[0]
}
