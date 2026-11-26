// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

private final class CacheOverrideRecordingModel: Module, LanguageModel, KVCacheDimensionProvider {
    let configurationData: Data
    let kvHeads: [Int] = []
    @ModuleInfo(key: "layer") var layer: Linear

    init(configurationData: Data) {
        self.configurationData = configurationData
        _layer.wrappedValue = Linear(2, 2, bias: false)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray { input }
}

private struct CacheOverrideTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct CacheOverrideTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { CacheOverrideTokenizer() }
}

private struct CacheOverrideProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray([Int32(1)]))
    }
}

@Suite(.serialized)
struct Inkling2ConfigurationOverrideTests {
    @Test("The optional override survives resolution and defaults to preserving the checkpoint")
    func configurationResolution() {
        let directory = URL(filePath: "/unused/inkling2")
        #expect(ModelConfiguration(id: "org/inkling2").inkling2RecurrentKVCacheMode == nil)
        #expect(
            ResolvedModelConfiguration(directory: directory).inkling2RecurrentKVCacheMode == nil)
        let remote = ModelConfiguration(id: "org/inkling2", inkling2RecurrentKVCacheMode: .lastLoop)
        #expect(
            remote.resolved(modelDirectory: directory, tokenizerDirectory: directory)
                .inkling2RecurrentKVCacheMode == .lastLoop)
        var local = ModelConfiguration(directory: directory, inkling2RecurrentKVCacheMode: .perLoop)
        #expect(
            local.resolved(modelDirectory: directory, tokenizerDirectory: directory)
                .inkling2RecurrentKVCacheMode == .perLoop)
        local.inkling2RecurrentKVCacheMode = nil
        #expect(
            local.resolved(modelDirectory: directory, tokenizerDirectory: directory)
                .inkling2RecurrentKVCacheMode == nil)
    }

    @Test(
        "Both factories override JSON5 model configuration without changing files",
        arguments: [false, true], [Inkling2RecurrentKVCacheMode.perLoop, .lastLoop])
    func factoriesApplyOverride(vision: Bool, mode: Inkling2RecurrentKVCacheMode) async throws {
        let diskMode: Inkling2RecurrentKVCacheMode = mode == .perLoop ? .lastLoop : .perLoop
        let directory = try checkpoint(mode: diskMode)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configURL = directory.appending(component: "config.json")
        let original = try Data(contentsOf: configURL)
        let context = try await load(directory: directory, vision: vision, override: mode)
        let model = try #require(context.model as? CacheOverrideRecordingModel)
        let object = try JSONDecoder.json5().decode(
            [String: JSONValue].self, from: model.configurationData)
        guard case .object(let text)? = object["text_config"] else {
            Issue.record("Missing text_config")
            return
        }
        #expect(text["recurrent_kv_cache_mode"] == .string(mode.rawValue))
        #expect(object["custom_metadata"] == .object(["version": .int(1), "enabled": .bool(true)]))
        #expect(context.configuration.inkling2RecurrentKVCacheMode == mode)
        #expect(model.layer.weight.asArray(Float.self) == [1, 2, 3, 4])
        #expect(try Data(contentsOf: configURL) == original)
    }

    @Test(
        "Nil preserves the checkpoint mode and original JSON5 bytes in both factories",
        arguments: [false, true])
    func nilHonorsCheckpoint(vision: Bool) async throws {
        let directory = try checkpoint(mode: .lastLoop)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try Data(contentsOf: directory.appending(component: "config.json"))
        let context = try await load(directory: directory, vision: vision, override: nil)
        let model = try #require(context.model as? CacheOverrideRecordingModel)
        #expect(model.configurationData == original)
        #expect(context.configuration.inkling2RecurrentKVCacheMode == nil)
    }

    @Test("Explicit lastLoop refuses unsupported models before loading", arguments: [false, true])
    func unsupportedLastLoop(vision: Bool) async throws {
        for (modelType, passes) in [("llama", 3), ("inkling2", 0), ("inkling2_loop", 1)] {
            let directory = try checkpoint(mode: .perLoop, modelType: modelType, passes: passes)
            defer { try? FileManager.default.removeItem(at: directory) }
            do {
                _ = try await load(
                    directory: directory, vision: vision, override: .lastLoop,
                    modelType: modelType)
                Issue.record("Expected lastLoop to be refused for \(modelType), \(passes) passes")
            } catch ModelFactoryError.invalidConfiguration(let message) {
                #expect(message.contains("repeated layers"))
            }
        }
    }

    @Test("Explicit perLoop on another architecture is a byte-preserving no-op")
    func perLoopOnOtherArchitecture() throws {
        let configuration = ResolvedModelConfiguration(
            directory: URL(filePath: "/unused/model"), inkling2RecurrentKVCacheMode: .perLoop)
        let original = Data("{model_type:'llama',custom:1,}".utf8)
        #expect(
            try configuration.applyingInkling2CacheMode(to: original, modelType: "llama")
                == original)
    }

    private func checkpoint(
        mode: Inkling2RecurrentKVCacheMode, modelType: String = "inkling2_loop", passes: Int = 3
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "Inkling2Override-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let loop =
            passes == 0 ? "" : "loop_config: {core_start: 1, core_end: 2, num_passes: \(passes)},"
        let data = Data(
            """
            {// JSON5 must remain supported with an explicit override.
              model_type: '\(modelType)',
              text_config: {\(loop)recurrent_kv_cache_mode: '\(mode.rawValue)',},
              custom_metadata: {version: 1, enabled: true},
            }
            """.utf8)
        try data.write(to: directory.appending(component: "config.json"))
        try Data(#"{"processor_class":"CacheOverrideProcessor"}"#.utf8)
            .write(to: directory.appending(component: "processor_config.json"))
        try save(
            arrays: ["layer.weight": MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)],
            url: directory.appending(component: "model.safetensors"))
        return directory
    }

    private func load(
        directory: URL, vision: Bool, override: Inkling2RecurrentKVCacheMode?,
        modelType: String = "inkling2_loop"
    ) async throws -> sending ModelContext {
        let registry = ModelTypeRegistry<LanguageModel>(creators: [
            modelType: { CacheOverrideRecordingModel(configurationData: $0) }
        ])
        let configuration = ResolvedModelConfiguration(
            directory: directory, inkling2RecurrentKVCacheMode: override)
        if vision {
            let factory = VLMModelFactory(
                typeRegistry: registry,
                processorRegistry: ProcessorTypeRegistry(creators: [
                    "CacheOverrideProcessor": { _, _ in CacheOverrideProcessor() }
                ]), modelRegistry: AbstractModelRegistry(),
                processorLoadingRegistry: VLMProcessorLoadingRegistry())
            return try await factory._load(
                configuration: configuration,
                tokenizerLoader: CacheOverrideTokenizerLoader())
        }
        let factory = LLMModelFactory(
            typeRegistry: registry, modelRegistry: AbstractModelRegistry())
        return try await factory._load(
            configuration: configuration,
            tokenizerLoader: CacheOverrideTokenizerLoader())
    }
}
