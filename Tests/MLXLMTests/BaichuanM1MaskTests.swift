// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// BaichuanM1 mixes sliding-window and full-attention layers. Each kind needs the mask of
/// its own attention cache, as mlx-lm builds it.
@Suite(.serialized)
struct BaichuanM1MaskTests {

    @Test("A prompt longer than the sliding window prefills like token-by-token decoding")
    func prefillPastTheWindow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyBaichuanM1()
            let prompt = (1 ... 10).map(Int32.init)

            let prefilled = try model.newCache(parameters: nil)
            let whole = model(MLXArray(prompt).reshaped([1, prompt.count]), cache: prefilled)

            let stepped = try model.newCache(parameters: nil)
            var last = whole
            for token in prompt {
                last = model(MLXArray([token]).reshaped([1, 1]), cache: stepped)
            }
            let expected = last[0..., -1, 0...]
            let actual = whole[0..., -1, 0...]
            #expect(abs(actual - expected).max().item(Float.self) < 1e-4)
        }
    }

    @Test("Ragged rows decode past the sliding window like separate sequences")
    func batchedRowsPastTheWindow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyBaichuanM1()
            let prompts: [[Int]] = [[1, 2, 3, 4, 5, 6], [7, 8], [9, 10, 11]]
            let expected = try prompts.map { prompt in
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), model: model,
                    cache: nil, processor: nil, sampler: ArgMaxSampler(), maxTokens: 6)
                return Array(IteratorSequence(iterator))
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 6)
            var uids = iterator.insert(prompts: Array(prompts[0 ..< 2]))
            var tokens = [Int: [Int]]()
            for _ in 0 ..< 2 {
                for case .token(let id, let token) in try iterator.next() ?? [] {
                    tokens[id, default: []].append(token)
                }
            }
            uids += iterator.insert(prompts: [prompts[2]])
            while let events = try iterator.next() {
                for case .token(let id, let token) in events {
                    tokens[id, default: []].append(token)
                }
            }
            #expect(uids.map { tokens[$0] ?? [] } == expected)
        }
    }
}

/// Layer 0 attends through a 4-token sliding window, layer 1 over the whole sequence.
private func tinyBaichuanM1() throws -> BaichuanM1Model {
    let configuration = try JSONDecoder().decode(
        BaichuanM1Configuration.self,
        from: Data(
            """
            {"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
             "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
             "rope_theta": 10000.0, "sliding_window": 4, "sliding_window_layers": [0],
             "conv_window": 2, "rms_norm_eps": 1e-6, "tie_word_embeddings": false}
            """.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 17)) {
        let model = BaichuanM1Model(configuration)
        // The convolution weights start at zero; random ones exercise the conv state.
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { key, value in
                    (
                        key,
                        key.hasSuffix("conv_k") || key.hasSuffix("conv_v")
                            ? MLXRandom.normal(value.shape) : value
                    )
                }))
        eval(model)
        return model
    }
}
