// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXImageGeneration
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// A tokenizer whose token ids are UTF-8 bytes.
private struct ByteTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite struct ImageIteratorTests {

    /// A model with random weights and 32-dimensional states, small enough for unit tests.
    private func makeModel(sampleSigmas: [Float]? = nil) throws -> QwenImageModel {
        func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
            try JSONDecoder().decode(type, from: Data(json.utf8))
        }
        return try withRandomState(MLXRandom.RandomState(seed: 11)) {
            let textEncoder = Qwen3VL(
                try decode(
                    Qwen3VLConfiguration.self,
                    """
                    {
                      "model_type": "qwen3_vl", "image_token_id": 500, "video_token_id": 501,
                      "vision_start_token_id": 502, "vision_end_token_id": 503, "vocab_size": 512,
                      "text_config": {
                        "model_type": "qwen3_vl", "hidden_size": 32, "num_hidden_layers": 2,
                        "intermediate_size": 64, "num_attention_heads": 2,
                        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 512,
                        "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
                        "rope_theta": 100000.0,
                        "rope_scaling": {"type": "default", "mrope_section": [4, 2, 2]}
                      },
                      "vision_config": {
                        "model_type": "qwen3_vl", "depth": 1, "hidden_size": 32,
                        "intermediate_size": 64, "out_hidden_size": 32, "num_heads": 2,
                        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                        "num_position_embeddings": 64, "deepstack_visual_indexes": [0]
                      }
                    }
                    """))
            let transformer = try QwenImageTransformer(
                try decode(
                    QwenImageTransformerConfiguration.self,
                    """
                    {
                      "in_channels": 8, "out_channels": 8, "num_layers": 2,
                      "num_attention_heads": 2, "attention_head_dim": 16, "context_in_dim": 32,
                      "mlp_ratio": 3, "axes_dims_rope": [4, 6, 6]
                    }
                    """))
            let decoder = try QwenImageVAEDecoder(
                try decode(
                    QwenImageVAEConfiguration.self,
                    """
                    {
                      "base_dim": 2, "z_dim": 8, "dim_mult": [1, 2, 4, 8, 8], "num_res_blocks": 2,
                      "temperal_downsample": [false, true, true, true], "out_channels": 4,
                      "latents_mean": [0, 0, 0, 0, 0, 0, 0, 0],
                      "latents_std": [1, 1, 1, 1, 1, 1, 1, 1]
                    }
                    """))
            // The decoder's weights start at zero, which would paint every image one color.
            let randomized = decoder.parameters().flattened().map { key, value in
                (key, MLXRandom.normal(value.shape) * 0.2 + (key.hasSuffix(".gamma") ? 1 : 0))
            }
            decoder.update(parameters: ModuleParameters.unflattened(randomized))
            eval(textEncoder, transformer, decoder)

            return QwenImageModel(
                textEncoder: textEncoder, tokenizer: ByteTokenizer(), transformer: transformer,
                decoder: decoder,
                schedulerConfiguration: .init(
                    maxImageSequenceLength: 8192, maxShift: 0.9, shiftTerminal: 0.02,
                    useDynamicShifting: true),
                sampleSigmas: sampleSigmas)
        }
    }

    /// Runs the iterator to its end and returns each event's name and the image.
    private func run(_ iterator: ImageIterator) -> (events: [String], image: MLXArray?) {
        var iterator = iterator
        var events = [String]()
        var image: MLXArray?
        while let event = iterator.next() {
            switch event {
            case .promptEncoded: events.append("prompt")
            case .denoised(let step, let total): events.append("step \(step)/\(total)")
            case .image(let pixels):
                events.append("image")
                image = pixels
            }
        }
        #expect(iterator.next() == nil, "the iterator ends after the image")
        return (events, image)
    }

    @Test("Each call completes one stage: the prompt, every denoising step, then the image")
    func stagesInOrder() throws {
        let model = try makeModel()
        // 64 wide and 32 high: not square, so swapped sides cannot pass.
        let iterator = try ImageIterator(
            model: model, prompt: "a fox",
            parameters: .init(width: 64, height: 32, stepCount: 3, seed: 1))
        #expect(iterator.stepCount == 3)

        let (events, image) = run(iterator)
        #expect(events == ["prompt", "step 1/3", "step 2/3", "step 3/3", "image"])
        let pixels = try #require(image)
        #expect(pixels.shape == [32, 64, 4])
        #expect(pixels.dtype == .uint8)
    }

    @Test("A seed fixes the image")
    func seedFixesImage() throws {
        let model = try makeModel()
        func image(seed: UInt64) throws -> [UInt8] {
            let iterator = try ImageIterator(
                model: model, prompt: "a fox",
                parameters: .init(width: 32, height: 32, stepCount: 2, seed: seed))
            return try #require(run(iterator).image).asArray(UInt8.self)
        }
        let first = try image(seed: 5)
        #expect(try image(seed: 5) == first)
        #expect(try image(seed: 6) != first)
        // The random decoder still has to paint something other than a flat color.
        #expect(Set(first).count > 8)
    }

    @Test("A checkpoint's own sampling grid sets the number of steps")
    func fixedGridSetsStepCount() throws {
        let model = try makeModel(sampleSigmas: [1, 0.6, 0.3])
        let iterator = try ImageIterator(
            model: model, prompt: "a fox",
            parameters: .init(width: 32, height: 32, stepCount: 40))
        #expect(iterator.stepCount == 3)
        #expect(run(iterator).events == ["prompt", "step 1/3", "step 2/3", "step 3/3", "image"])
    }

    @Test("Guidance above 1 steers the image away from the negative prompt")
    func guidanceChangesImage() throws {
        let model = try makeModel()
        func image(guidanceScale: Float, negativePrompt: String = " ") throws -> [UInt8] {
            let iterator = try ImageIterator(
                model: model, prompt: "a fox",
                parameters: .init(
                    width: 32, height: 32, stepCount: 2, guidanceScale: guidanceScale,
                    negativePrompt: negativePrompt, seed: 5))
            return try #require(run(iterator).image).asArray(UInt8.self)
        }
        let plain = try image(guidanceScale: 1)
        // At 1 the negative prompt is not used at all.
        #expect(try image(guidanceScale: 1, negativePrompt: "a dog") == plain)
        let guided = try image(guidanceScale: 4)
        #expect(guided != plain)
        #expect(try image(guidanceScale: 4, negativePrompt: "a dog") != guided)
    }

    @Test("A size or step count the model cannot use is rejected before any work")
    func invalidParameters() throws {
        let model = try makeModel()
        #expect(throws: ImageGenerationError.invalidSize(width: 40, height: 32)) {
            try ImageIterator(model: model, prompt: "", parameters: .init(width: 40, height: 32))
        }
        #expect(throws: ImageGenerationError.invalidSize(width: 32, height: 0)) {
            try ImageIterator(model: model, prompt: "", parameters: .init(width: 32, height: 0))
        }
        #expect(throws: ImageGenerationError.invalidStepCount(0)) {
            try ImageIterator(
                model: model, prompt: "", parameters: .init(width: 32, height: 32, stepCount: 0))
        }
    }
}
