// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Inkling 2's Metal kernels against the reference ops they replace.
@Suite(.serialized)
struct Inkling2KernelTests {
    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    @Test("Depth attention matches the reference", arguments: [(1, 1), (3, 1), (4, 3), (2, 4)])
    func depthAttention(blocks: Int, queries: Int) throws {
        MLXRandom.seed(7)
        let D = 2304
        let blockStates = inkling2UnitRMS(MLXRandom.normal([blocks, 2, 5, D]))
        let partial = inkling2UnitRMS(MLXRandom.normal([2, 5, D]))
        let query = MLXRandom.normal([queries, D]) * 0.05
        let kernel = inkling2DepthAttentionKernel(
            blocks: blockStates, partial: partial, query: query, scale: pow(Float(D), -0.5))
        let reference = Device.withDefaultDevice(.cpu) {
            inkling2DepthAttention(blocks: blockStates, partial: partial, query: query)
        }
        #expect(kernel.shape == [2, 5, queries, D])
        #expect(maxDifference(kernel, reference) < 1e-4)
    }

    /// Prefill attention of `L` queries at the end of `S` keys, against the reference path.
    @Test(
        "Relative attention matches the reference",
        arguments: [(37, 37, 0, 0), (40, 100, 16, 0), (9, 60, 0, 0), (33, 45, 8, 5)])
    func relativeAttention(L: Int, S: Int, window: Int, padding: Int) throws {
        MLXRandom.seed(11)
        let (B, Hq, Hkv, D, R) = (2, 8, 2, 128, 24)
        let args = try JSONDecoder().decode(
            Inkling2TextConfiguration.self,
            from: Data(
                """
                {"hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 8,
                 "num_attention_heads": \(Hq), "num_key_value_heads": \(Hkv), "head_dim": \(D),
                 "swa_num_attention_heads": \(Hq), "swa_num_key_value_heads": \(Hkv),
                 "swa_head_dim": \(D), "sliding_window_size": \(max(window, 1)),
                 "local_layer_ids": \(window > 0 ? "[0]" : "[]"), "d_rel": 4, "rel_extent": \(R),
                 "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
                 "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8,
                 "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 6,
                 "n_routed_experts": 2, "num_experts_per_tok": 1, "intermediate_size": 64,
                 "mlp_gate_softcap": 4, "mlp_up_softcap": 25, "route_scale": 1,
                 "final_logit_softcapping": 30}
                """.utf8))
        let attention = Inkling2Attention(args, sliding: window > 0)
        func clipped(_ shape: [Int]) -> MLXArray {
            let x = MLXRandom.normal(shape)
            return (x / sqrt((x * x).sum(axis: -1, keepDims: true)) * 3).asType(.bfloat16)
        }
        let queries = clipped([B, Hq, L, D])
        let keys = clipped([B, Hkv, S, D])
        let values = inkling2UnitRMS(MLXRandom.normal([B, Hkv, S, D])).asType(.bfloat16)
        let bias = (MLXRandom.normal([B, L, Hq, R]) * 0.5).asType(.bfloat16)
        let sinks = (MLXRandom.normal([B, L, Hq]) * 0.5).asType(.bfloat16)
        // Row 1 starts after `padding` keys that hold no position.
        let lastPosition = MLXArray([Float(S - 1), Float(S - 1 - padding)]).reshaped(B, 1)
        let queryPositions = lastPosition - Float(L - 1) + MLXArray(0 ..< L).asType(.float32)
        var stored = lastPosition - Float(S - 1) + MLXArray(0 ..< S).asType(.float32) + 1
        stored = MLX.maximum(stored, MLXArray(Float(0)))

        let kernel = inkling2RelativeAttentionKernel(
            queries: queries, keys: keys, values: values, positionBias: bias, sinks: sinks,
            keyStart: MLXArray([Int32(0), Int32(padding)]), scale: attention.scale,
            window: window)
        let reference = Device.withDefaultDevice(.cpu) {
            attention.attention(
                queries: queries, keys: keys, values: values, positionBias: bias, sinks: sinks,
                queryPositions: queryPositions, storedKeyPositions: stored, forceReference: true)
        }
        #expect(kernel.shape == [B, L, Hq, D])
        // The kernel holds q, k, v and the probabilities in half precision.
        #expect(maxDifference(kernel, reference) < 2e-2)
    }

    /// A small bfloat16 Inkling 2 Mini whose looped layers complete depth blocks, with the
    /// router, depth queries and experts in float32 as the checkpoints keep them.
    private func bfloat16Model() throws -> Inkling2Model {
        let model = Inkling2Model(
            try JSONDecoder().decode(
                Inkling2Configuration.self,
                from: Data(
                    """
                    {"model_type": "inkling2_loop", "text_config": {
                        "hidden_size": 256, "num_hidden_layers": 8, "vocab_size": 128,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
                        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
                        "swa_head_dim": 64, "sliding_window_size": 16,
                        "local_layer_ids": [0, 1, 2, 3, 4, 6], "d_rel": 8, "rel_extent": 32,
                        "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1,
                        "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 3.0,
                        "rel_l2_clip_threshold": 2.0, "sconv_kernel_size": 4,
                        "n_layers_per_attn_res_block": 3, "n_routed_experts": 2,
                        "num_experts_per_tok": 2, "intermediate_size": 128,
                        "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.0,
                        "final_logit_softcapping": 30.0,
                        "loop_config": {"core_start": 3, "core_end": 6, "num_passes": 3,
                            "gate_rank": 8}}}
                    """.utf8)))
        let weights = model.parameters().flattened().map { name, value -> (String, MLXArray) in
            if name.hasSuffix("norm.weight") || name.hasSuffix("l2_clip.scale") {
                return (name, (1 + 0.1 * MLXRandom.normal(value.shape)).asType(.bfloat16))
            }
            let fanIn = value.ndim > 1 ? value.dim(-1) : 16
            let random = MLXRandom.normal(value.shape) * pow(Float(fanIn), -0.5)
            // The router and the depth queries stay float32, as in the checkpoints, and so do
            // the experts: the CPU's gathered matmul takes float32 only.
            let keepsFloat32 =
                name.contains("mlp.gate.") || name.contains(".mlp.experts.")
                || name.hasSuffix(".query")
            return (name, keepsFloat32 ? random : random.asType(.bfloat16))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
        return model
    }

    @Test("A bfloat16 model prefills through the kernels as through the reference path")
    func modelPrefill() throws {
        MLXRandom.seed(0)
        let model = try bfloat16Model()
        let tokens = MLXRandom.randInt(Int32(0) ..< 120, [2, 48])

        func prefill() throws -> [MLXArray] {
            let cache = try model.newCache(parameters: nil)
            let chunked = [
                model(tokens[0..., ..<20], cache: cache), model(tokens[0..., 20...], cache: cache),
            ]
            return [model(tokens, cache: nil), concatenated(chunked, axis: 1)]
        }
        let kernels = try prefill()
        let reference = try Device.withDefaultDevice(.cpu) { try prefill() }
        for (kernel, expected) in zip(kernels, reference) {
            // bfloat16 activations through eight layers and a loop, on two devices. With the
            // kernels off the GPU and CPU already differ by about 0.18 here. Every token takes
            // both experts, so no near tie in routing can flip and amplify the difference.
            #expect(maxDifference(kernel, expected) < 0.3)
        }
    }

    @Test("Training takes the reference paths, so gradients flow through attention")
    func trainingGradients() throws {
        MLXRandom.seed(0)
        let model = try bfloat16Model()
        // On the GPU, a bfloat16 prefill would take both kernels, which have no gradient.
        let tokens = MLXRandom.randInt(Int32(0) ..< 120, [2, 24])
        model.train()
        let lossAndGradient = valueAndGrad(model: model) { model, tokens, _ in
            model(tokens, cache: nil).asType(.float32).square().mean()
        }
        let (loss, gradients) = try withError {
            let (loss, gradients) = lossAndGradient(model, tokens, MLXArray(0))
            let flat = gradients.flattened()
            eval([loss] + flat.map(\.1))
            return (loss, flat)
        }
        #expect(loss.item(Float.self).isFinite)
        #expect(gradients.allSatisfy { all(isFinite($0.1)).item(Bool.self) })
        // The depth queries only reach the loss through depth attention.
        let depthQueries = gradients.filter { $0.0.hasSuffix(".query") }
        #expect(!depthQueries.isEmpty)
        #expect(depthQueries.contains { abs($0.1).max().item(Float.self) > 0 })
    }
}
