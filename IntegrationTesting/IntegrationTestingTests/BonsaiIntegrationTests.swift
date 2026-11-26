// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

struct BonsaiCheckpoint: Sendable {
    let name: String
    let revision: String

    var configuration: ModelConfiguration {
        if let root = ProcessInfo.processInfo.environment["MLX_BONSAI_MODEL_ROOT"] {
            return ModelConfiguration(
                directory: URL(fileURLWithPath: root).appendingPathComponent(name))
        }
        return ModelConfiguration(id: "prism-ml/\(name)", revision: revision)
    }
}

private let bonsaiCheckpoints = [
    BonsaiCheckpoint(
        name: "Ternary-Bonsai-2-2B-mlx-2bit",
        revision: "1c6cbae21f8c38830ce337972b46fdae7b18eaaa"),
    BonsaiCheckpoint(
        name: "Ternary-Bonsai-2-4B-mlx-2bit",
        revision: "dff2caaa23fa8706164c44c10f1b58feb405f76a"),
    BonsaiCheckpoint(
        name: "Ternary-Bonsai-2-9B-mlx-2bit",
        revision: "f4e28a59531731c41b6dfac88cf271bd58f0a287"),
    BonsaiCheckpoint(
        name: "Ternary-Bonsai-2-27B-mlx-2bit",
        revision: "24b5937a4a104a44f790b6ccbd0c79b6fabd4299"),
    BonsaiCheckpoint(
        name: "Ternary-Bonsai-2.5-27B-mlx-2bit",
        revision: "aa432fd0e17612f6684e728d4d3c42b6656bcabb"),
].filter { checkpoint in
    guard let filter = ProcessInfo.processInfo.environment["MLX_BONSAI_MODEL_FILTER"] else {
        return true
    }
    return checkpoint.name.contains(filter)
}

extension ModelLoadingSuite {
    /// Opt in with `TEST_RUNNER_MLX_RUN_BONSAI_INTEGRATION=1` for xcodebuild.
    /// `MLX_BONSAI_MODEL_ROOT` can point to predownloaded checkpoint directories.
    @Suite(
        .serialized,
        .timeLimit(.minutes(20)),
        .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_BONSAI_INTEGRATION"] == "1"))
    struct BonsaiIntegrationTests {
        private let downloader = #hubDownloader()
        private let tokenizerLoader = #huggingFaceTokenizerLoader()
    }
}

extension ModelLoadingSuite.BonsaiIntegrationTests {
    @Test(arguments: bonsaiCheckpoints)
    func textAndVisionInference(checkpoint: BonsaiCheckpoint) async throws {
        let textTokens = try await validateTextLoader(checkpoint)
        let configuration = checkpoint.configuration
        let context = try await TestModelManager.shared.load("bonsai-vlm", configuration) {
            try await VLMModelFactory.shared.load(
                from: downloader, using: tokenizerLoader, configuration: configuration)
        }

        let arithmetic = try await response(
            context, prompt: "What is 17 plus 25? Reply with only the number.")
        #expect(arithmetic.text.trimmingCharacters(in: .whitespacesAndNewlines) == "42")
        #expect(arithmetic.tokens == textTokens)
        print("BONSAI \(checkpoint.name) VLM arithmetic: \(arithmetic.text)")

        for (color, image) in [
            ("red", VisionTestImages.solidColor(.red)),
            ("blue", VisionTestImages.solidColor(.blue)),
        ] {
            let answer = try await response(
                context,
                prompt: "What color is this image? Reply with only the color name.",
                image: image)
            let words = answer.text.lowercased().split(whereSeparator: { !$0.isLetter })
            #expect(words.contains(Substring(color)), "\(checkpoint.name): \(answer.text)")
            print("BONSAI \(checkpoint.name) \(color) image: \(answer.text)")
        }
    }

    private func validateTextLoader(_ checkpoint: BonsaiCheckpoint) async throws -> [Int] {
        let configuration = checkpoint.configuration
        let context = try await TestModelManager.shared.load("bonsai-llm", configuration) {
            try await LLMModelFactory.shared.load(
                from: downloader, using: tokenizerLoader, configuration: configuration)
        }
        let prompt = "What is 17 plus 25? Reply with only the number."
        let arithmetic = try await response(context, prompt: prompt)
        #expect(arithmetic.text.trimmingCharacters(in: .whitespacesAndNewlines) == "42")
        print("BONSAI \(checkpoint.name) LLM arithmetic: \(arithmetic.text)")

        let chunked = try await response(context, prompt: prompt, prefillStepSize: 8)
        #expect(chunked.tokens == arithmetic.tokens, "chunked prefill changed greedy decoding")

        let capital = try await response(
            context, prompt: "What is the capital of France? Reply with only the city name.")
        #expect(capital.text.lowercased().contains("paris"), "\(checkpoint.name): \(capital.text)")
        print("BONSAI \(checkpoint.name) capital: \(capital.text)")
        return arithmetic.tokens
    }

    private func response(
        _ context: ModelContext,
        prompt: String,
        image: CIImage? = nil,
        prefillStepSize: Int = 512
    ) async throws -> (text: String, tokens: [Int]) {
        let input = try await context.processor.prepare(
            input: UserInput(
                chat: [.user(prompt, images: image.map { [.ciImage($0)] } ?? [])],
                additionalContext: ["enable_thinking": false]))
        let stream = try generateTokens(
            input: input,
            parameters: GenerateParameters(
                maxTokens: 64, temperature: 0, prefill: .init(stepSize: prefillStepSize)),
            context: context)
        var tokens = [Int]()
        for await event in stream {
            if case .token(let token) = event { tokens.append(token) }
        }
        return (context.tokenizer.decode(tokenIds: tokens), tokens)
    }
}
