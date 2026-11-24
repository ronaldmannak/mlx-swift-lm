//
//  Inkling2.swift
//  mlx-swift-lm
//

// Port of `mlx_lm/models/inkling2/model.py` from jundot/mlx-lm@0c4c9919 (MIT): Thinking
// Machines' Inkling 2 (`inkling2`) and Inkling 2 Mini (`inkling2_loop`), text only.
//
// The configuration, text model and checkpoint mapping are in MLXLMCommon
// (`Models/Inkling2*.swift`), shared with the MLXVLM wrapper.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The module that holds the text model, at `model.language_model`.
public final class Inkling2Backbone: Module {
    @ModuleInfo(key: "language_model") public var languageModel: Inkling2TextModel

    init(_ args: Inkling2TextConfiguration) {
        _languageModel.wrappedValue = Inkling2TextModel(args)
    }
}

public final class Inkling2Model: Module, LLMModel {
    public let configuration: Inkling2Configuration
    public let vocabularySize: Int

    @ModuleInfo(key: "model") public var model: Inkling2Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: Inkling2Configuration) {
        configuration = args
        vocabularySize = args.text.unpaddedVocabularySize
        _model.wrappedValue = Inkling2Backbone(args.text)
        _lmHead.wrappedValue = Linear(args.text.hiddenSize, args.text.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inkling2Logits(
            model.languageModel(inputs, cache: cache), lmHead: lmHead,
            configuration: configuration.text)
    }

    /// Also emits the hidden states a DSpark drafter asks for.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        inkling2Output(
            embeddings: model.languageModel.embed(input.tokens), cache: cache, state: state)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let batched = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        return try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            try inkling2Prepare(
                embeddings: model.languageModel.embed(batched), cache: cache, state: state,
                prefill: prefill)
        }
    }

    /// The vision tower is never read.
    public var excludedCheckpointNamespaces: [String] {
        ["model.visual", "vision_tower", "model.vision_tower"]
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try Inkling2Checkpoint.prepare(
            checkpoint, configuration: configuration, includeVision: false)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.languageModel.newCache(parameters: parameters)
    }
}

// MARK: - Chat conventions

extension Inkling2Model {
    /// Messages framed by special tokens; the decoder also separates reasoning.
    public var toolCallFormat: ToolCallFormat? { .inkling }
}

extension Inkling2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.languageModel.layers
    }
}

extension Inkling2Model: TensorParallelModel {
    /// Keeps this process' share of the text model; see ``Inkling2TextModel/shard(group:)``.
    public func shard(group: MLXDistributed.Group) throws {
        try validateShardable(self)
        try model.languageModel.shard(group: group)
    }
}

// Batched admission keeps the generic resumable chunk hooks. Normal prefill also collects
// all hidden-state rows requested by a DSpark drafter.
extension Inkling2Model: ResumablePrefillModel, BatchDecodableModel {}

extension Inkling2Model: Inkling2LanguageModel {
    package var inkling2TextModel: Inkling2TextModel { model.languageModel }
    package var inkling2Head: Linear { lmHead }
}
