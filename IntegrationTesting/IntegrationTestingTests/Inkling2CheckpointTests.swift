import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers
import XCTest

/// Runs a published Inkling 2 checkpoint: chat, batched rows against solo runs, prefill and
/// decode speed, and an image prompt. Set `MLX_INKLING2_CHECKPOINT` to a local checkpoint.
/// A Debug build decodes about ten times slower; for speed numbers, build with
/// `GCC_OPTIMIZATION_LEVEL=3 SWIFT_OPTIMIZATION_LEVEL=-O`.
final class Inkling2CheckpointTests: XCTestCase {
    private static let endSampling = 200_006

    private func directory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"] else {
            throw XCTSkip("Set MLX_INKLING2_CHECKPOINT to a local Inkling 2 checkpoint")
        }
        return URL(filePath: path)
    }

    private func loadText() async throws -> ModelContainer {
        let start = Date.timeIntervalSinceReferenceDate
        let container = try await LLMModelFactory.shared.loadContainer(
            from: try directory(), using: #huggingFaceTokenizerLoader())
        print("Inkling2 text load_seconds=\(Date.timeIntervalSinceReferenceDate - start)")
        return container
    }

    private static func prompt(_ text: String, context: ModelContext) throws -> [Int] {
        try context.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": text]], tools: nil,
            additionalContext: ["reasoning_effort": "none"])
    }

    /// Greedy tokens of one prompt through the single-sequence iterator.
    private static func solo(_ tokens: [Int], maxTokens: Int, context: ModelContext) throws
        -> (tokens: [Int], info: GenerateCompletionInfo)
    {
        var output: [Int] = []
        let info = try generate(
            input: LMInput(tokens: MLXArray(tokens)),
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            context: context
        ) { token in
            if token == endSampling { return .stop }
            output.append(token)
            return .more
        }
        return (output, info)
    }

    func testChatAnswersInText() async throws {
        let container = try await loadText()
        let (text, reasoning, info) = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(
                    prompt: "What is the capital of France? Answer in one short sentence.",
                    additionalContext: ["reasoning_effort": "none"]))
            var text = ""
            var reasoning = ""
            var info: GenerateCompletionInfo?
            for await event in try generate(
                input: input, parameters: GenerateParameters(maxTokens: 512, temperature: 0),
                context: context)
            {
                switch event {
                case .chunk(let chunk): text += chunk
                case .reasoning(let chunk): reasoning += chunk
                case .info(let value): info = value
                default: break
                }
            }
            return (text, reasoning, info)
        }
        print("Inkling2 chat text=\(text.debugDescription) reasoning=\(reasoning.debugDescription)")
        if let info {
            print("Inkling2 chat stop=\(info.stopReason) tokens=\(info.generationTokenCount)")
        }
        XCTAssertTrue(text.contains("Paris"))
        XCTAssertFalse(text.contains("<|"))
    }

    func testThinkingThenAnswer() async throws {
        let container = try await loadText()
        let (text, reasoning) = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(
                    prompt: "What is 17 times 23? Give just the number.",
                    additionalContext: ["reasoning_effort": "low"]))
            var text = ""
            var reasoning = ""
            for await event in try generate(
                input: input, parameters: GenerateParameters(maxTokens: 512, temperature: 0),
                context: context)
            {
                switch event {
                case .chunk(let chunk): text += chunk
                case .reasoning(let chunk): reasoning += chunk
                default: break
                }
            }
            return (text, reasoning)
        }
        print("Inkling2 think reasoning=\(reasoning.debugDescription)")
        print("Inkling2 think text=\(text.debugDescription)")
        XCTAssertTrue(text.contains("391"))
    }

    func testBatchedRowsMatchSoloRuns() async throws {
        let container = try await loadText()
        let questions = [
            "Name three primary colors.",
            "Write one sentence about the ocean.",
            "What is the boiling point of water at sea level in Celsius? Explain briefly why it changes with altitude.",
            "List the first five prime numbers.",
        ]
        let maxTokens = 48
        let (solo, batched) = try await container.perform { context in
            let prompts = try questions.map { try Self.prompt($0, context: context) }
            let solo = try prompts.map {
                try Self.solo($0, maxTokens: maxTokens, context: context).tokens
            }
            var batched = Array(repeating: [Int](), count: prompts.count)
            for try await event in try batchGenerateTokens(
                model: context.model, prompts: prompts, tokenizer: context.tokenizer,
                modelConfiguration: context.configuration, maxTokens: maxTokens,
                stopTokens: [Self.endSampling])
            {
                if case .output(let uid, .token(let token)) = event { batched[uid].append(token) }
            }
            return (solo, batched)
        }
        try await container.perform { context in
            for (row, (a, b)) in zip(solo, batched).enumerated() {
                let shared = zip(a, b).prefix { $0 == $1 }.count
                if shared < min(a.count, b.count) {
                    // How close the two choices were in the solo run's logits at the split.
                    let prefix =
                        try Self.prompt(questions[row], context: context) + a.prefix(shared)
                    let logits = context.model(
                        LMInput.Text(tokens: MLXArray(prefix).reshaped(1, -1)),
                        cache: try context.model.newCache(parameters: nil), state: nil
                    ).logits[0, -1].asType(.float32)
                    let gap = (logits[a[shared]] - logits[b[shared]]).item(Float.self)
                    print(
                        "Inkling2 row \(row) split: solo token \(a[shared]) leads batched token \(b[shared]) by \(gap) logits"
                    )
                }
                print(
                    "Inkling2 row \(row): solo=\(a.count) batched=\(b.count) shared_prefix=\(shared)\n  solo: \(context.tokenizer.decode(tokenIds: a, skipSpecialTokens: false).debugDescription)\n  batch: \(context.tokenizer.decode(tokenIds: b, skipSpecialTokens: false).debugDescription)"
                )
            }
        }
        for (a, b) in zip(solo, batched) {
            let shared = zip(a, b).prefix { $0 == $1 }.count
            XCTAssertGreaterThanOrEqual(shared, min(16, a.count))
        }
    }

    func testPrefillAndDecodeSpeed() async throws {
        let container = try await loadText()
        try await container.perform { context in
            let filler = String(
                repeating:
                    "The quick brown fox jumps over the lazy dog while the river flows to the sea. ",
                count: 400)
            let whole = try Self.prompt(
                filler + "Summarize the text above in one sentence.", context: context)
            for length in [1024, 4096] {
                let tokens = Array(whole.suffix(length))
                Memory.peakMemory = 0
                let (_, info) = try Self.solo(tokens, maxTokens: 128, context: context)
                print(
                    "Inkling2 speed prompt=\(info.promptTokenCount) prefill_tok_s=\(info.promptTokensPerSecond) decode_tokens=\(info.generationTokenCount) decode_tok_s=\(info.tokensPerSecond) peak_gb=\(Double(Memory.peakMemory) / 1e9)"
                )
            }
        }
    }

    func testImagePrompt() async throws {
        let start = Date.timeIntervalSinceReferenceDate
        let container = try await VLMModelFactory.shared.loadContainer(
            from: try directory(), using: #huggingFaceTokenizerLoader())
        print("Inkling2 vision load_seconds=\(Date.timeIntervalSinceReferenceDate - start)")
        // A red square on a blue background.
        let background = CIImage(color: CIColor(red: 0, green: 0, blue: 1))
            .cropped(to: CGRect(x: 0, y: 0, width: 400, height: 400))
        let square = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 100, y: 100, width: 200, height: 200))
        let image = square.composited(over: background)
        let (text, reasoning) = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(
                    prompt:
                        "What shape is in the middle of this image, and what colors are the shape and the background? Answer in one sentence.",
                    images: [.ciImage(image)], additionalContext: ["reasoning_effort": "none"]))
            var text = ""
            var reasoning = ""
            for await event in try generate(
                input: input, parameters: GenerateParameters(maxTokens: 512, temperature: 0),
                context: context)
            {
                switch event {
                case .chunk(let chunk): text += chunk
                case .reasoning(let chunk): reasoning += chunk
                default: break
                }
            }
            return (text, reasoning)
        }
        print("Inkling2 image reasoning=\(reasoning.debugDescription)")
        print("Inkling2 image text=\(text.debugDescription)")
        XCTAssertTrue(text.lowercased().contains("red"))
        XCTAssertTrue(text.lowercased().contains("blue"))
    }
}
