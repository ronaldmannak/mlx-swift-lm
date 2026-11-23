// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private func fusionConfiguration() throws -> Inkling2DSparkConfiguration {
    try JSONDecoder().decode(
        Inkling2DSparkConfiguration.self,
        from: Data(
            """
            {"model_type": "inkling2_loop", "text_config": {
                "hidden_size": 256, "num_hidden_layers": 8, "vocab_size": 128,
                "unpadded_vocab_size": 16, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 128, "swa_num_attention_heads": 4,
                "swa_num_key_value_heads": 2, "swa_head_dim": 128,
                "sliding_window_size": 16, "local_layer_ids": [0, 1, 2, 3, 4, 6],
                "d_rel": 8, "rel_extent": 32, "log_scaling_n_floor": 16,
                "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
                "qk_l2_clip_threshold": 3.0, "rel_l2_clip_threshold": 2.0,
                "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 3,
                "n_routed_experts": 2, "num_experts_per_tok": 2, "intermediate_size": 128,
                "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.0,
                "final_logit_softcapping": 30.0,
                "loop_config": {"core_start": 3, "core_end": 6, "num_passes": 3,
                    "gate_rank": 8}},
             "mtp_config": {"hidden_size": 256, "num_hidden_layers": 3,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
                "intermediate_size": 128, "max_position_embeddings": 4096,
                "q_bias": false, "o_bias": false, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "local_layer_ids": [0, 2],
                "sliding_window_size": 8, "block_size": 4, "markov_rank": 8,
                "markov_head_type": "rnn_hidden", "target_layer_ids": [1, 4, 9, 14]}}
            """.utf8))
}

private func fusionWeights(_ module: Module, seed: UInt64, uniformHead: Bool = false) {
    MLXRandom.seed(seed)
    let weights = module.parameters().flattened().map { name, value -> (String, MLXArray) in
        if uniformHead && name == "lm_head.weight" {
            return (name, MLXArray.zeros(value.shape, dtype: .bfloat16))
        }
        if name.hasSuffix("norm.weight") || name.hasSuffix("l2_clip.scale") {
            return (name, (1 + 0.1 * MLXRandom.normal(value.shape)).asType(.bfloat16))
        }
        let fanIn = value.ndim > 1 ? value.dim(-1) : 16
        let random = MLXRandom.normal(value.shape) * pow(Float(fanIn), -0.5)
        let float32 =
            name.contains("mlp.gate.") || name.hasSuffix(".query")
            || name.contains("forget_gate_proj.") || name.contains("message_gate_proj.")
        return (name, float32 ? random : random.asType(.bfloat16))
    }
    module.update(parameters: ModuleParameters.unflattened(weights))
    module.train(false)
    eval(module)
}

private func fusionTarget(uniformHead: Bool = false, quantized: Bool = true) throws -> Inkling2Model
{
    let target = Inkling2Model(try fusionConfiguration().target)
    fusionWeights(target, seed: 37, uniformHead: uniformHead)
    if quantized {
        quantize(model: target) { _, module in
            module is SwitchLinear ? (16, 4, .nvfp4) : nil
        }
        target.train(false)
        eval(target)
    }
    return target
}

private func fusionPrompt(_ length: Int) -> LMInput {
    LMInput(tokens: MLXArray((0 ..< length).map { Int32(($0 * 7 + 3) % 16) }))
}

private func fusionTokens(
    _ target: Inkling2Model, input: LMInput, parameters: GenerateParameters
) throws -> [Int] {
    var iterator = try TokenIterator(input: input, model: target, parameters: parameters)
    var output = [Int]()
    while let token = iterator.next() { output.append(token) }
    Stream.defaultStream(.gpu).synchronize()
    return output
}

@Suite(.serialized)
struct Inkling2FusionCompatibilityTests {
    @Test(
        "Compiled decode traces read parameter updates without invalidation",
        arguments: [false, true])
    func compiledParameterUpdates(rewind: Bool) throws {
        try Device.withDefaultDevice(.gpu) {
            let target = try fusionTarget()
            var parameters = GenerateParameters(temperature: 0)
            if rewind {
                parameters.kvCache = KVCacheConfiguration(rewind: try .init(maxTokens: 4))
            }
            let prompt = fusionPrompt(21).text.tokens.reshaped(1, -1)
            let warm = try target.newCache(parameters: parameters)
            eval(target(prompt, cache: warm))
            for token in [Int32(1), 6] {
                eval(target(MLXArray([token]).reshaped(1, 1), cache: warm))
            }

            // Updating existing array objects must reach both compiled halves, without a reset.
            let updated = target.parameters().flattened().map { name, value in
                let floating = [DType.float32, .bfloat16, .float16].contains(value.dtype)
                return (name, floating ? (value + Float(0.005)).asType(value.dtype) : value)
            }
            eval(updated.map(\.1))
            target.update(parameters: ModuleParameters.unflattened(updated))
            let fresh = try fusionTarget()
            fresh.update(parameters: target.parameters())
            let cache = try target.newCache(parameters: parameters)
            let freshCache = try fresh.newCache(parameters: parameters)

            func compare(_ actual: MLXArray, _ expected: MLXArray, step: Int) {
                #expect(actual.shape == expected.shape)
                #expect(actual.dtype == .float32 && expected.dtype == .float32)
                #expect(all(isFinite(actual)).item(Bool.self))
                #expect(all(isFinite(expected)).item(Bool.self))
                #expect(
                    arrayEqual(actual.view(dtype: .uint32), expected.view(dtype: .uint32))
                        .item(Bool.self),
                    "Parameter update at step \(step), rewind=\(rewind)")
            }

            compare(target(prompt, cache: cache), fresh(prompt, cache: freshCache), step: -1)
            for step in 0 ..< 8 {
                let token = MLXArray([Int32((step * 5 + 1) % 16)]).reshaped(1, 1)
                compare(target(token, cache: cache), fresh(token, cache: freshCache), step: step)
            }
        }
    }

    @Test("Fused decode logits track the unfused model on identical incremental histories")
    func incrementalDecodeParity() throws {
        try Device.withDefaultDevice(.gpu) {
            let fused = try fusionTarget()
            let reference = try fusionTarget()
            reference.update(parameters: fused.parameters())
            reference.train(false)
            #expect(!fused.training && !reference.training)
            #expect(fused.configuration.text.headDim == 128)
            #expect(
                fused.model.languageModel.layers[0].mlp.experts.gateUpProjection
                    is Inkling2ScaledSwitchLinear)
            let fusedCache = try fused.newCache(parameters: nil)
            let referenceCache = try reference.newCache(parameters: nil)
            let prompt = fusionPrompt(21)
            let prefill = PrefillParameters(stepSize: 512)
            guard
                case .logits(let initial) = try fused.prepare(
                    prompt, cache: fusedCache, state: nil, prefill: prefill),
                case .logits(let expectedInitial) = try reference.prepare(
                    prompt, cache: referenceCache, state: nil, prefill: prefill)
            else {
                Issue.record("Expected fully prepared prompt logits")
                return
            }

            func compare(_ actual: MLXArray, _ expected: MLXArray, position: Int) {
                let actual = actual.asType(.float32)
                let expected = expected.asType(.float32)
                #expect(actual.shape == [1, 1, 16] && expected.shape == actual.shape)
                #expect(all(isFinite(actual)).item(Bool.self))
                #expect(all(isFinite(expected)).item(Bool.self))
                #expect((actual.max() - actual.min()).item(Float.self) > 1e-3)
                let error = abs(actual - expected)
                let maximum = error.max().item(Float.self)
                let mean = error.mean().item(Float.self)
                print("Inkling2 decode parity position=\(position) maximum=\(maximum) mean=\(mean)")
                // These bounds cover accumulated BF16 reductions, rather than token choices.
                #expect(maximum < 0.1, "Position \(position): max \(maximum), mean \(mean)")
                #expect(mean < 0.02, "Position \(position): max \(maximum), mean \(mean)")
            }

            compare(initial.logits, expectedInitial.logits, position: 20)
            var state = initial.state
            var referenceState = expectedInitial.state
            // Both targets see the same tokens, through the sliding-window wrap and bias extent.
            for step in 0 ..< 24 {
                let token = Int32((step * 5 + 1) % 16)
                let input = LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1))
                let actual = fused(input, cache: fusedCache, state: state)
                // An all-true padding mask selects the general layer path, keeping inference attention.
                for case let convolution as ArraysCache in referenceCache {
                    convolution.prepare(lengths: [1])
                }
                let expected = reference(input, cache: referenceCache, state: referenceState)
                state = actual.state
                referenceState = expected.state
                eval(actual.logits, expected.logits)
                if [0, 1, 3, 7, 15, 23].contains(step) {
                    compare(actual.logits, expected.logits, position: 21 + step)
                }
            }
        }
    }

    @Test("Bfloat16 decode preserves seeded sampling", arguments: [Float(0.6), 1.0])
    func seededSampling(temperature: Float) throws {
        let target = try fusionTarget()
        var parameters = GenerateParameters(maxTokens: 16, temperature: temperature, topP: 0.9)
        parameters.seed = 41
        let input = fusionPrompt(21)
        let first = try fusionTokens(target, input: input, parameters: parameters)
        let second = try fusionTokens(target, input: input, parameters: parameters)
        #expect(first == second)
        #expect(first.count == 16)
    }

    @Test("Bfloat16 DSpark keeps greedy and sampled target draws", arguments: [Float(0), 0.8])
    func dsparkSampling(temperature: Float) throws {
        // Uniform logits isolate RNG consumption from harmless prefill/decode rounding.
        // The target still runs the quantized experts, bfloat16 decoder and rewind operations.
        let target = try fusionTarget(uniformHead: true)
        let draft = Inkling2DSparkDraftModel(try fusionConfiguration())
        fusionWeights(draft, seed: 43)
        #expect(!draft.requiresGreedySampling)
        let input = fusionPrompt(21)
        var parameters = GenerateParameters(maxTokens: 24, temperature: temperature)
        parameters.seed = 47
        let expected = try fusionTokens(target, input: input, parameters: parameters)
        parameters.kvCache = KVCacheConfiguration(rewind: try .init(maxTokens: 4))
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: target, drafter: draft, parameters: parameters, blockSize: 4)
        var output = [Int]()
        while let token = iterator.next() { output.append(token) }
        Stream.defaultStream(.gpu).synchronize()
        #expect(output == expected)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
        if temperature == 0 {
            #expect(iterator.acceptedCount > 0)
        } else {
            #expect(Set(output).count > 1)
        }
    }

    @Test("Ragged bfloat16 batch rows keep independent seeded samplers")
    func batchedSampling() throws {
        let target = try fusionTarget(uniformHead: true)
        let prompts = [[1, 2, 3], (0 ..< 21).map { ($0 * 7 + 3) % 16 }]
        var parameters = GenerateParameters(maxTokens: 12, temperature: 0.8)
        parameters.seed = 53
        let alone = try prompts.map {
            try fusionTokens(target, input: LMInput(tokens: MLXArray($0)), parameters: parameters)
        }
        parameters.prefill = PrefillParameters(stepSize: 7)
        var iterator = try BatchTokenIterator(model: target, parameters: parameters)
        let ids = iterator.insert(
            prompts: prompts, samplers: prompts.map { _ in parameters.sampler() })
        defer { iterator.remove(ids) }
        var output = [Int: [Int]]()
        while let events = try iterator.next() {
            for event in events {
                switch event {
                case .token(let id, let token):
                    output[id, default: []].append(token)
                case .failed(_, let error):
                    throw error
                case .prefill, .finished:
                    break
                }
            }
        }
        let together = ids.map { output[$0] ?? [] }
        #expect(together == alone)
    }

    @Test("Training after fused decode reaches controllers, depth queries and router")
    func trainingAfterDecode() throws {
        let target = try fusionTarget(quantized: false)
        let cache = try target.newCache(parameters: nil)
        _ = target(fusionPrompt(8).text.tokens.reshaped(1, -1), cache: cache)
        eval(target(MLXArray([Int32(1)]).reshaped(1, 1), cache: cache))
        target.train()
        let lossAndGradient = valueAndGrad(model: target) { model, tokens, _ in
            model(tokens, cache: nil).asType(.float32).square().mean()
        }
        let tokens = fusionPrompt(12).text.tokens.reshaped(1, -1)
        let (loss, gradients) = try withError {
            let (loss, gradients) = lossAndGradient(target, tokens, MLXArray(0))
            let flat = gradients.flattened()
            eval([loss] + flat.map(\.1))
            return (loss, flat)
        }
        #expect(loss.item(Float.self).isFinite)
        #expect(gradients.allSatisfy { all(isFinite($0.1)).item(Bool.self) })
        for suffix in ["message_gate_proj.weight", ".query", "mlp.gate.weight"] {
            let selected = gradients.filter { $0.0.hasSuffix(suffix) }
            #expect(!selected.isEmpty)
            #expect(selected.contains { abs($0.1).max().item(Float.self) > 0 })
        }
    }
}
