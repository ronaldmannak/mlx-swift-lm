// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// A loaded Qwen-Image-2.1 checkpoint: text encoder, denoiser, image decoder and schedule.
///
/// Create an ``ImageIterator`` from it to generate an image. The model's weights carry their
/// own license; check it before you ship a checkpoint.
public final class QwenImageModel {
    let textEncoder: Qwen3VL
    let tokenizer: any Tokenizer
    let transformer: QwenImageTransformer
    let decoder: QwenImageVAEDecoder
    let schedulerConfiguration: FlowMatchEulerScheduler.Configuration
    /// The noise levels a distilled checkpoint was trained to sample on.
    let sampleSigmas: [Float]?

    /// The turn that opens every prompt. Its tokens are dropped from the encoder's states.
    private static let systemTurn =
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    private static let maximumPromptTokens = 1024

    package init(
        textEncoder: Qwen3VL, tokenizer: any Tokenizer, transformer: QwenImageTransformer,
        decoder: QwenImageVAEDecoder,
        schedulerConfiguration: FlowMatchEulerScheduler.Configuration, sampleSigmas: [Float]?
    ) {
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.decoder = decoder
        self.schedulerConfiguration = schedulerConfiguration
        self.sampleSigmas = sampleSigmas
    }

    /// Loads a checkpoint from its directory.
    ///
    /// The directory holds `text_encoder`, `transformer`, `vae`, `scheduler` and `processor`
    /// folders, as `Qwen/Qwen-Image-2.1` and its MLX conversions do. Quantized and
    /// full-precision weights both load.
    ///
    /// - Parameters:
    ///   - directory: the checkpoint's directory
    ///   - tokenizerLoader: loads the tokenizer from the `processor` folder
    public static func load(from directory: URL, tokenizerLoader: any TokenizerLoader)
        async throws -> QwenImageModel
    {
        let textEncoderDirectory = directory.appending(component: "text_encoder")
        let textEncoderConfiguration = try Data(
            contentsOf: textEncoderDirectory.appending(component: "config.json"))
        let textEncoder = Qwen3VL(
            try JSONDecoder.json5().decode(
                Qwen3VLConfiguration.self, from: textEncoderConfiguration))
        try await loadWeights(
            modelDirectory: textEncoderDirectory, model: textEncoder,
            perLayerQuantization: JSONDecoder.json5().decode(
                BaseConfiguration.self, from: textEncoderConfiguration
            ).perLayerQuantization)

        struct ModelIndex: Decodable {
            let sampleSigmas: [Float]?

            enum CodingKeys: String, CodingKey {
                case sampleSigmas = "sample_sigmas"
            }
        }
        let index = try JSONDecoder().decode(
            ModelIndex.self,
            from: Data(contentsOf: directory.appending(component: "model_index.json")))
        let schedulerConfiguration = try JSONDecoder().decode(
            FlowMatchEulerScheduler.Configuration.self,
            from: Data(
                contentsOf: directory.appending(components: "scheduler", "scheduler_config.json")
            ))

        return QwenImageModel(
            textEncoder: textEncoder,
            tokenizer: try await tokenizerLoader.load(
                from: directory.appending(component: "processor")),
            transformer: try QwenImageTransformer.load(
                from: directory.appending(component: "transformer")),
            decoder: try QwenImageVAEDecoder.load(from: directory.appending(component: "vae")),
            schedulerConfiguration: schedulerConfiguration,
            sampleSigmas: index.sampleSigmas)
    }

    /// The text encoder's states for `prompt`, `[1, tokens, dimensions]`, without the system
    /// turn.
    func encode(_ prompt: String) -> MLXArray {
        // The checkpoint was trained on this template as plain text, not on a chat template.
        let text =
            Self.systemTurn + "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        let systemTokenCount = tokenizer.encode(text: Self.systemTurn, addSpecialTokens: false)
            .count
        let tokens = tokenizer.encode(text: text, addSpecialTokens: true)
            .prefix(systemTokenCount + Self.maximumPromptTokens)
        let states = textEncoder.lastHiddenState(
            MLXArray(tokens.map { Int32($0) }).reshaped(1, -1))
        return states[0..., systemTokenCount...].asType(.bfloat16)
    }
}
