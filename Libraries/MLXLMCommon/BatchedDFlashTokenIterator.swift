// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Errors from batched DFlash v1 generation.
public enum BatchedDFlashError: Error, Equatable {
    case emptyBatch, emptyPrompt, unsupportedTarget, unsupportedCache, geometryMismatch
    case unsupportedBlockSize, unsupportedDrafter, incompatiblePrefix, invalidParameters
    case unsupportedLogProbabilities
}

public struct BatchedDFlashTokenChunk {
    /// Stable request identifier, independent of the current tensor row.
    public let id: Int
    public let tokens: [Int]
    public let finished: Bool
}

/// Projected context rows for a cohort. Invalid slots never enter attention.
public final class BatchedDFlashContextCache {
    public let window: Int
    public let batchSize: Int
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var rowPositions: [[Int32]]
    public private(set) var rowValid: [[Bool]]

    public init(window: Int, batchSize: Int) {
        precondition(window > 0 && batchSize > 0)
        self.window = window
        self.batchSize = batchSize
        rowPositions = Array(repeating: [], count: batchSize)
        rowValid = Array(repeating: [], count: batchSize)
    }

    public var count: Int { rowPositions[0].count }

    public func append(
        keys newKeys: MLXArray, values newValues: MLXArray,
        positions: [Int], validCounts: [Int]
    ) {
        precondition(positions.count == batchSize && validCounts.count == batchSize)
        precondition(newKeys.dim(0) == batchSize && newValues.dim(0) == batchSize)
        let n = newKeys.dim(2)
        precondition(validCounts.allSatisfy { (0 ... n).contains($0) })
        guard n > 0 else { return }
        keys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        values = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        for row in 0 ..< batchSize {
            rowPositions[row] += (0 ..< n).map { Int32(positions[row] + $0) }
            rowValid[row] += (0 ..< n).map { $0 < validCounts[row] }
        }
        if window < Int.max - 256 && count > window + 256 {
            compact()
        }
    }

    private func compact() {
        let kept = rowValid.map { valid in
            Array(valid.indices.filter { valid[$0] }.suffix(window))
        }
        let width = kept.map(\.count).max() ?? 0
        guard width > 0 else { return }
        let gather = MLXArray(
            kept.flatMap { $0.map(Int32.init) + Array(repeating: 0, count: width - $0.count) }
        )
        .reshaped(batchSize, 1, width, 1)
        keys = contiguous(takeAlong(keys!, gather, axis: 2))
        values = contiguous(takeAlong(values!, gather, axis: 2))
        rowPositions = kept.enumerated().map { row, indices in
            indices.map { rowPositions[row][$0] }
                + Array(repeating: 0, count: width - indices.count)
        }
        rowValid = kept.map {
            Array(repeating: true, count: $0.count)
                + Array(repeating: false, count: width - $0.count)
        }
    }

    public func copy() -> BatchedDFlashContextCache {
        let result = BatchedDFlashContextCache(window: window, batchSize: batchSize)
        result.keys = keys.map { $0 + 0 }
        result.values = values.map { $0 + 0 }
        result.rowPositions = rowPositions
        result.rowValid = rowValid
        return result
    }

    public func extract(row: Int) -> BatchedDFlashContextCache {
        precondition((0 ..< batchSize).contains(row))
        let result = BatchedDFlashContextCache(window: window, batchSize: 1)
        result.keys = keys.map { $0[row ..< row + 1] + 0 }
        result.values = values.map { $0[row ..< row + 1] + 0 }
        result.rowPositions = [rowPositions[row]]
        result.rowValid = [rowValid[row]]
        return result
    }

    public static func merge(_ rows: [BatchedDFlashContextCache]) -> BatchedDFlashContextCache {
        precondition(
            !rows.isEmpty && rows.allSatisfy { $0.batchSize == 1 && $0.window == rows[0].window })
        let result = BatchedDFlashContextCache(window: rows[0].window, batchSize: rows.count)
        guard let sample = rows.first(where: { $0.keys != nil }) else { return result }
        let width = rows.map(\.count).max()!
        func arrays(_ value: (BatchedDFlashContextCache) -> MLXArray?) -> MLXArray {
            let template = value(sample)!
            return concatenated(
                rows.map { row in
                    let shape = [1, template.dim(1), width - row.count, template.dim(3)]
                    let padding = MLXArray.zeros(shape, dtype: template.dtype)
                    return value(row).map { concatenated([$0, padding], axis: 2) }
                        ?? MLXArray.zeros(
                            [1, template.dim(1), width, template.dim(3)], dtype: template.dtype)
                }, axis: 0)
        }
        result.keys = arrays { $0.keys }
        result.values = arrays { $0.values }
        result.rowPositions = rows.map {
            $0.rowPositions[0] + Array(repeating: 0, count: width - $0.count)
        }
        result.rowValid = rows.map {
            $0.rowValid[0] + Array(repeating: false, count: width - $0.count)
        }
        return result
    }
}

public struct BatchedDFlashDrafterState {
    public var contextCaches: [BatchedDFlashContextCache]
    public init(contextCaches: [BatchedDFlashContextCache]) { self.contextCaches = contextCaches }
    public func copy() -> Self { .init(contextCaches: contextCaches.map { $0.copy() }) }
    public func extract(row: Int) -> Self {
        .init(contextCaches: contextCaches.map { $0.extract(row: row) })
    }
}

public struct BatchedDFlashProposalRequest {
    public var block: MLXArray
    public var targetHidden: MLXArray
    public var contextPositions: [Int]
    public var validCounts: [Int]
    public var suppressedTokens: SuppressTokensProcessor?

    public init(
        block: MLXArray, targetHidden: MLXArray, contextPositions: [Int], validCounts: [Int],
        suppressedTokens: SuppressTokensProcessor? = nil
    ) {
        self.block = block
        self.targetHidden = targetHidden
        self.contextPositions = contextPositions
        self.validCounts = validCounts
        self.suppressedTokens = suppressedTokens
    }
}

public protocol BatchedDFlashDrafterModel: DFlash2DrafterModel {
    var supportsBatchedDFlashV1: Bool { get }
    func makeBatchState(batchSize: Int) -> BatchedDFlashDrafterState
    func appendBatchContext(
        hidden: MLXArray, positions: [Int], validCounts: [Int],
        state: inout BatchedDFlashDrafterState)
    func proposeBatch(
        _ request: BatchedDFlashProposalRequest, target: any DFlash2TargetModel,
        state: inout BatchedDFlashDrafterState
    ) -> MLXArray
}

/// Both halves of a warm prefix. A target-only cache cannot initialize this iterator.
public struct BatchedDFlashPrefixState {
    public let tokens: [Int]
    public let targetCache: [KVCache]
    public let drafterState: BatchedDFlashDrafterState
    public let lastLogits: MLXArray
    fileprivate let targetOwner: AnyObject
    fileprivate let drafterOwner: AnyObject
    fileprivate var targetIdentity: ObjectIdentifier { ObjectIdentifier(targetOwner) }
    fileprivate var drafterIdentity: ObjectIdentifier { ObjectIdentifier(drafterOwner) }
}

/// Batched v1 generation with admission and removal between committed rounds.
/// Keep this non-Sendable iterator and its models on one serial executor.
/// Admission uses a full synchronous text prefill, without chunked prefill scheduling.
/// Only uncompressed, unbounded target caches and v1 drafters are supported.
/// Custom processors must isolate speculative state and hook side effects in `copy()`.
public final class BatchedDFlashTokenIterator {
    public let target: any DFlash2TargetModel
    public let drafter: any BatchedDFlashDrafterModel
    public let blockSize: Int
    public let maxTokens: Int?
    public private(set) var cache: [KVCache]
    /// Per-row arrays follow the order in `ids`, which can change after removal.
    public private(set) var positions: [Int]
    public private(set) var generatedCountByRow: [Int]
    public private(set) var acceptedCountByRow: [Int]
    public private(set) var proposedCountByRow: [Int]
    /// Prompt and tokens returned by `next`, including an emitted but unfed anchor.
    public private(set) var tokenLedgerByRow: [[Int]]
    public private(set) var roundCount = 0
    public private(set) var promptPrefillTime: TimeInterval = 0

    private var draftState: BatchedDFlashDrafterState
    private var processors: [LogitProcessor?]
    private var randomStates: [MLXRandom.RandomState]
    private let parameters: GenerateParameters
    private let components: GenerationComponents
    private var parametersByRow: [GenerateParameters]
    private var rowIDs: [Int]
    private var nextID: Int
    private var processorFinalized: [Bool]
    private var emissionSensitiveByRow: [Bool]

    /// Retained requests, including finished rows available for snapshots.
    public var ids: [Int] { rowIDs }
    public var activeIDs: [Int] { rowIDs.indices.filter { !finished[$0] }.map { rowIDs[$0] } }
    public var maxTokensByRow: [Int?] { parametersByRow.map(\.maxTokens) }
    private let suppressor: SuppressTokensProcessor?
    private let eosTokenIds: Set<Int>
    private var finished: [Bool]
    private var anchors: [Int]
    private var hidden: MLXArray
    private var hiddenPositions: [Int]
    private var hiddenCounts: [Int]
    private var initialChunks: [BatchedDFlashTokenChunk]?
    private var lastPromptLogitsByRow: [MLXArray] = []

    public init(
        inputs: [LMInput], mainModel: any LanguageModel,
        drafter: any BatchedDFlashDrafterModel, parameters: GenerateParameters,
        blockSize: Int? = nil, eosTokenIds: Set<Int> = [], maxTokens: [Int]? = nil,
        prefixStates: [BatchedDFlashPrefixState?]? = nil,
        components: GenerationComponents = .init()
    ) throws {
        guard !inputs.isEmpty else { throw BatchedDFlashError.emptyBatch }
        guard parameters.maxTokens.map({ $0 >= 0 }) ?? true,
            maxTokens.map({ $0.count == inputs.count && $0.allSatisfy { $0 >= 0 } }) ?? true
        else { throw BatchedDFlashError.invalidParameters }
        let perRowParameters = inputs.indices.map { row in
            var value = parameters
            if let maxTokens { value.maxTokens = maxTokens[row] }
            return value
        }
        guard let target = mainModel as? any DFlash2TargetModel else {
            throw BatchedDFlashError.unsupportedTarget
        }
        guard drafter.supportsBatchedDFlashV1 else { throw BatchedDFlashError.unsupportedDrafter }
        guard
            drafter.targetGeometry.accepts(target.dflash2Geometry, layerIds: drafter.targetLayerIds)
        else {
            throw BatchedDFlashError.geometryMismatch
        }
        let width = blockSize ?? drafter.blockSize
        guard width >= 2 && width <= drafter.blockSize else {
            throw BatchedDFlashError.unsupportedBlockSize
        }
        guard parameters.logProbabilities == nil else {
            throw BatchedDFlashError.unsupportedLogProbabilities
        }
        if let config = try parameters.kvCachePlan().configuration,
            config.capacity != nil || config.rewind != nil || config.strategy != .fullPrecision
        {
            throw BatchedDFlashError.unsupportedCache
        }
        let prefixes = prefixStates ?? Array(repeating: nil, count: inputs.count)
        guard prefixes.count == inputs.count else { throw BatchedDFlashError.incompatiblePrefix }
        guard
            prefixes.allSatisfy({ prefix in
                prefix.map {
                    $0.targetIdentity == ObjectIdentifier(mainModel as AnyObject)
                        && $0.drafterIdentity == ObjectIdentifier(drafter as AnyObject)
                } ?? true
            })
        else { throw BatchedDFlashError.incompatiblePrefix }
        for value in perRowParameters { try components.validate(parameters: value) }
        self.target = target
        self.drafter = drafter
        self.parameters = parameters
        self.components = components
        self.parametersByRow = perRowParameters
        self.rowIDs = Array(inputs.indices)
        self.nextID = inputs.count
        self.processorFinalized = Array(repeating: false, count: inputs.count)
        self.emissionSensitiveByRow = Array(
            repeating: components.logitProcessorFactory != nil, count: inputs.count)
        self.blockSize = width
        self.maxTokens = parameters.maxTokens
        self.eosTokenIds = eosTokenIds
        self.suppressor = makeSuppressTokensProcessor(model: mainModel)
        self.randomStates = inputs.indices.map { row in
            parameters.seed.map { MLXRandom.RandomState(seed: $0 &+ UInt64(row)) }
                ?? MLXRandom.RandomState()
        }
        processors = inputs.indices.map { row in
            components.logitProcessor(parameters: perRowParameters[row], model: mainModel)?.copy()
        }
        acceptedCountByRow = Array(repeating: 0, count: inputs.count)
        proposedCountByRow = acceptedCountByRow
        generatedCountByRow = acceptedCountByRow
        tokenLedgerByRow = Array(repeating: [], count: inputs.count)
        finished = perRowParameters.map { $0.maxTokens == 0 }
        anchors = acceptedCountByRow
        positions = acceptedCountByRow
        hiddenPositions = acceptedCountByRow
        hiddenCounts = acceptedCountByRow
        cache = []
        draftState = drafter.makeBatchState(batchSize: inputs.count)
        hidden = MLXArray.zeros([
            inputs.count, 0, target.dflash2Geometry.hiddenSize * drafter.targetLayerIds.count,
        ])

        let start = Date.timeIntervalSinceReferenceDate
        var rowCaches: [[KVCache]] = []
        var rowHidden: [MLXArray] = []
        var rowDraftStates: [BatchedDFlashDrafterState] = []
        var chunks: [BatchedDFlashTokenChunk] = []
        for row in inputs.indices {
            let input = inputs[row]
            let parameters = perRowParameters[row]
            guard input.image == nil && input.video == nil && input.audio == nil else {
                throw DFlash2SpeculationError.textOnly
            }
            guard input.text.tokens.ndim == 1 else { throw BatchedDFlashError.emptyPrompt }
            let tokens = input.text.tokens.asType(.int32).asArray(Int32.self).map(Int.init)
            guard !tokens.isEmpty || prefixes[row] != nil else {
                throw BatchedDFlashError.emptyPrompt
            }
            let prefix = prefixes[row]
            let localCache =
                try prefix?.targetCache.map { $0.copy() } ?? target.newCache(parameters: parameters)
            guard target.dflash2SupportsCache(localCache) else {
                throw BatchedDFlashError.unsupportedCache
            }
            let prior = prefix?.tokens.count ?? 0
            tokenLedgerByRow[row] = (prefix?.tokens ?? []) + tokens
            let result: (logits: MLXArray, hidden: [MLXArray])
            if tokens.isEmpty {
                result = (prefix!.lastLogits, [])
            } else {
                result = target.dflash2Prefill(
                    input.text.tokens.reshaped(1, -1), cache: localCache,
                    captureLayers: drafter.targetLayerIds)
            }
            let captured =
                result.hidden.isEmpty
                ? MLXArray.zeros([
                    1, 0, target.dflash2Geometry.hiddenSize * drafter.targetLayerIds.count,
                ])
                : concatenated(result.hidden, axis: -1)
            let retained = min(captured.dim(1), drafter.contextWindow)
            rowHidden.append(captured[0..., (captured.dim(1) - retained)..., 0...])
            positions[row] = prior + tokens.count
            hiddenPositions[row] = positions[row] - retained
            hiddenCounts[row] = retained
            rowCaches.append(localCache)
            rowDraftStates.append(
                prefix?.drafterState.copy() ?? drafter.makeBatchState(batchSize: 1))
            processors[row]?.prompt(MLXArray(((prefix?.tokens ?? []) + tokens).map(Int32.init)))
            var logits = result.logits[0..., -1, 0...]
            lastPromptLogitsByRow.append(logits.expandedDimensions(axis: 1))
            logits = processors[row]?.process(logits: logits) ?? logits
            if !finished[row] {
                let token: MLXArray
                if parameters.temperature <= 0 {
                    token = argMax(logits, axis: -1)
                } else {
                    let probabilities = speculativeSamplingProbabilities(
                        logits, temperature: parameters.temperature, topP: parameters.topP,
                        minP: parameters.minP, topK: parameters.topK)
                    token = withRandomState(randomStates[row]) { categorical(log(probabilities)) }
                }
                anchors[row] = token.item(Int.self)
                processors[row]?.didSample(token: MLXArray(Int32(anchors[row])))
                generatedCountByRow[row] = 1
                finished[row] = eosTokenIds.contains(anchors[row]) || parameters.maxTokens == 1
                chunks.append(.init(id: row, tokens: [anchors[row]], finished: finished[row]))
            }
        }
        cache = try Self.mergeTargetCaches(rowCaches)
        let layerCount = rowDraftStates[0].contextCaches.count
        guard rowDraftStates.allSatisfy({ $0.contextCaches.count == layerCount }) else {
            throw BatchedDFlashError.incompatiblePrefix
        }
        draftState = .init(
            contextCaches: (0 ..< layerCount).map { layer in
                BatchedDFlashContextCache.merge(rowDraftStates.map { $0.contextCaches[layer] })
            })
        hidden = Self.padHidden(rowHidden)
        eval(cache.flatMap { $0.innerState() } + [hidden])
        initialChunks = chunks
        promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
        for row in inputs.indices where finished[row] && generatedCountByRow[row] == 0 {
            finalizeProcessor(row: row)
        }
    }

    /// New rows use `parameters.seed + id`; existing random states are never reseeded.
    /// Prefill completes synchronously before this method changes the active membership.
    @discardableResult
    public func insert(
        inputs: [LMInput], parameters: GenerateParameters? = nil, maxTokens: [Int]? = nil,
        prefixStates: [BatchedDFlashPrefixState?]? = nil,
        components: GenerationComponents? = nil
    ) throws -> [Int] {
        guard !inputs.isEmpty else { return [] }
        guard inputs.count <= Int.max - nextID else { throw BatchedDFlashError.invalidParameters }
        var incomingParameters = parameters ?? self.parameters
        if let seed = incomingParameters.seed { incomingParameters.seed = seed &+ UInt64(nextID) }
        let incoming = try BatchedDFlashTokenIterator(
            inputs: inputs, mainModel: target, drafter: drafter,
            parameters: incomingParameters, blockSize: blockSize, eosTokenIds: eosTokenIds,
            maxTokens: maxTokens, prefixStates: prefixStates,
            components: components ?? self.components)
        let newIDs = Array(nextID ..< (nextID + inputs.count))
        let targetRows =
            positions.indices.map { extractCache(row: $0) }
            + incoming.positions.indices.map { incoming.extractCache(row: $0) }
        let merged = try Self.mergeTargetCaches(targetRows)
        let states =
            positions.indices.map { draftState.extract(row: $0) }
            + incoming.positions.indices.map { incoming.draftState.extract(row: $0) }
        let contexts = states[0].contextCaches.indices.map { layer in
            BatchedDFlashContextCache.merge(states.map { $0.contextCaches[layer] })
        }
        let hiddenRows =
            positions.indices.map {
                hidden[$0 ..< $0 + 1, ..<hiddenCounts[$0], 0...]
            }
            + incoming.positions.indices.map {
                incoming.hidden[$0 ..< $0 + 1, ..<incoming.hiddenCounts[$0], 0...]
            }
        cache = merged
        draftState = .init(contextCaches: contexts)
        hidden = Self.padHidden(hiddenRows)
        rowIDs += newIDs
        nextID += inputs.count
        positions += incoming.positions
        generatedCountByRow += incoming.generatedCountByRow
        acceptedCountByRow += incoming.acceptedCountByRow
        proposedCountByRow += incoming.proposedCountByRow
        tokenLedgerByRow += incoming.tokenLedgerByRow
        processors += incoming.processors
        processorFinalized += incoming.processorFinalized
        emissionSensitiveByRow += incoming.emissionSensitiveByRow
        incoming.processorFinalized = Array(repeating: true, count: incoming.processors.count)
        randomStates += incoming.randomStates
        parametersByRow += incoming.parametersByRow
        finished += incoming.finished
        anchors += incoming.anchors
        hiddenPositions += incoming.hiddenPositions
        hiddenCounts += incoming.hiddenCounts
        lastPromptLogitsByRow += incoming.lastPromptLogitsByRow
        let newChunks = (incoming.initialChunks ?? []).map { chunk in
            BatchedDFlashTokenChunk(
                id: newIDs[chunk.id], tokens: chunk.tokens, finished: chunk.finished)
        }
        initialChunks = (initialChunks ?? []) + newChunks
        promptPrefillTime += incoming.promptPrefillTime
        return newIDs
    }

    /// Cancel and release requests at the last returned block boundary.
    /// GPU work already submitted by a previous call is not interrupted.
    public func remove(_ ids: [Int]) {
        let removed = Set(ids)
        let keep = rowIDs.indices.filter { !removed.contains(rowIDs[$0]) }
        guard keep.count != rowIDs.count else { return }
        for row in rowIDs.indices where removed.contains(rowIDs[row]) {
            finalizeProcessor(row: row)
        }
        let selected = MLXArray(keep.map(Int32.init))
        let remainingPositions = keep.map { positions[$0] }
        if let upper = remainingPositions.max() {
            for entry in cache {
                if let attention = entry as? KVCacheSimple {
                    attention.keys = attention.keys.map { take($0, selected, axis: 0) }
                    attention.values = attention.values.map { take($0, selected, axis: 0) }
                    attention.offset = upper
                } else if let recurrent = entry as? MambaCache {
                    for index in 0 ..< 2 {
                        recurrent[index] = recurrent[index].map { take($0, selected, axis: 0) }
                    }
                    recurrent.offset = upper
                }
            }
            draftState = .init(
                contextCaches: draftState.contextCaches.map { context in
                    BatchedDFlashContextCache.merge(keep.map { context.extract(row: $0) })
                })
            hidden = take(hidden, selected, axis: 0)
        } else {
            cache = []
            draftState = .init(contextCaches: [])
            hidden = MLXArray.zeros([0, 0, hidden.dim(2)], dtype: hidden.dtype)
        }
        rowIDs = keep.map { rowIDs[$0] }
        positions = remainingPositions
        generatedCountByRow = keep.map { generatedCountByRow[$0] }
        acceptedCountByRow = keep.map { acceptedCountByRow[$0] }
        proposedCountByRow = keep.map { proposedCountByRow[$0] }
        tokenLedgerByRow = keep.map { tokenLedgerByRow[$0] }
        processors = keep.map { processors[$0] }
        processorFinalized = keep.map { processorFinalized[$0] }
        emissionSensitiveByRow = keep.map { emissionSensitiveByRow[$0] }
        randomStates = keep.map { randomStates[$0] }
        parametersByRow = keep.map { parametersByRow[$0] }
        finished = keep.map { finished[$0] }
        anchors = keep.map { anchors[$0] }
        hiddenPositions = keep.map { hiddenPositions[$0] }
        hiddenCounts = keep.map { hiddenCounts[$0] }
        lastPromptLogitsByRow = keep.map { lastPromptLogitsByRow[$0] }
        initialChunks = initialChunks.map { $0.filter { !removed.contains($0.id) } }
    }

    deinit {
        for row in processors.indices { finalizeProcessor(row: row) }
    }

    private func finalizeProcessor(row: Int) {
        guard !processorFinalized[row] else { return }
        processors[row]?.finalizeGeneration()
        processorFinalized[row] = true
    }

    public static func makePrefixState(
        input: LMInput, mainModel: any LanguageModel, drafter: any BatchedDFlashDrafterModel
    ) throws -> BatchedDFlashPrefixState {
        guard let target = mainModel as? any DFlash2TargetModel else {
            throw BatchedDFlashError.unsupportedTarget
        }
        guard drafter.supportsBatchedDFlashV1 else { throw BatchedDFlashError.unsupportedDrafter }
        guard
            drafter.targetGeometry.accepts(target.dflash2Geometry, layerIds: drafter.targetLayerIds)
        else { throw BatchedDFlashError.geometryMismatch }
        guard input.image == nil && input.video == nil && input.audio == nil else {
            throw DFlash2SpeculationError.textOnly
        }
        guard input.text.tokens.ndim == 1 else { throw BatchedDFlashError.emptyPrompt }
        let tokens = input.text.tokens.asType(.int32).asArray(Int32.self).map(Int.init)
        guard !tokens.isEmpty else { throw BatchedDFlashError.emptyPrompt }
        let cache = try target.newCache(parameters: nil)
        guard target.dflash2SupportsCache(cache) else { throw BatchedDFlashError.unsupportedCache }
        let result = target.dflash2Prefill(
            input.text.tokens.reshaped(1, -1), cache: cache, captureLayers: drafter.targetLayerIds)
        let hidden = concatenated(result.hidden, axis: -1)
        let retained = min(hidden.dim(1), drafter.contextWindow)
        var state = drafter.makeBatchState(batchSize: 1)
        drafter.appendBatchContext(
            hidden: hidden[0..., (hidden.dim(1) - retained)..., 0...],
            positions: [tokens.count - retained], validCounts: [retained], state: &state)
        eval(
            cache.flatMap { $0.innerState() }
                + state.contextCaches.flatMap { [$0.keys!, $0.values!] } + [
                    result.logits
                ])
        return .init(
            tokens: tokens, targetCache: cache, drafterState: state,
            lastLogits: result.logits[0..., -1, 0...].expandedDimensions(axis: 1),
            targetOwner: mainModel as AnyObject,
            drafterOwner: drafter as AnyObject)
    }

    public func next() -> [BatchedDFlashTokenChunk]? {
        if let chunks = initialChunks {
            initialChunks = nil
            for chunk in chunks {
                guard let row = rowIDs.firstIndex(of: chunk.id) else { continue }
                tokenLedgerByRow[row] += chunk.tokens
                for token in chunk.tokens { processors[row]?.didEmit(token: token) }
                if chunk.finished { finalizeProcessor(row: row) }
            }
            if !chunks.isEmpty { return chunks }
        }
        guard finished.contains(false) else { return nil }
        let batch = anchors.count
        let width = blockSize
        let block = MLXArray(
            anchors.flatMap {
                [Int32($0)] + Array(repeating: Int32(drafter.maskTokenId), count: width - 1)
            }
        ).reshaped(batch, width)
        let drafts = drafter.proposeBatch(
            .init(
                block: block, targetHidden: hidden, contextPositions: hiddenPositions,
                validCounts: hiddenCounts, suppressedTokens: suppressor), target: target,
            state: &draftState
        ).asType(.int32)
        let verifyTokens = concatenated(
            [MLXArray(anchors.map(Int32.init)).reshaped(batch, 1), drafts], axis: 1)
        let verified = target.dflash2Verify(
            .init(
                tokens: verifyTokens, position: MLXArray(positions.map(Int32.init)),
                positionUpperBound: positions.max()!, captureLayers: drafter.targetLayerIds),
            cache: cache)
        let gamma = width - 1
        var packedRows: [MLXArray] = []
        for row in 0 ..< batch {
            let parameters = parametersByRow[row]
            let rowDrafts = drafts[row]
            if finished[row] {
                packedRows.append(MLXArray.zeros([gamma + 2], dtype: .int32))
                continue
            }
            var logits = verified.logits[row]
            if let processor = processors[row] {
                var scratch = processor.copy()
                let emittedDrafts =
                    emissionSensitiveByRow[row] ? rowDrafts.asArray(Int32.self) : nil
                var processed: [MLXArray] = []
                for column in 0 ..< width {
                    processed.append(scratch.process(logits: logits[column ..< (column + 1)]))
                    if column < gamma {
                        scratch.didSample(token: rowDrafts[column ..< (column + 1)])
                        if let emittedDrafts { scratch.didEmit(token: Int(emittedDrafts[column])) }
                    }
                }
                logits = concatenated(processed, axis: 0)
            }
            if parameters.temperature <= 0 {
                let targets = argMax(logits, axis: -1).asType(.int32)
                let accepted = cumprod((rowDrafts .== targets[..<gamma]).asType(.int32)).sum()
                let bonus = takeAlong(targets, accepted.reshaped(1), axis: 0)
                packedRows.append(concatenated([rowDrafts, accepted.reshaped(1), bonus]))
            } else {
                let probabilities = speculativeSamplingProbabilities(
                    logits, temperature: parameters.temperature, topP: parameters.topP,
                    minP: parameters.minP,
                    topK: parameters.topK)
                let accepted = withRandomState(randomStates[row]) {
                    DFlash2SpeculativeTokenIterator.rejectionSample(
                        drafts: rowDrafts, targetProbs: probabilities,
                        draftProbs: MLXArray.ones([gamma, 1]),
                        candidates: rowDrafts.expandedDimensions(axis: -1))
                }
                packedRows.append(accepted.packed)
            }
        }
        // A single cohort synchronization resolves all acceptance counts.
        let packed = stacked(packedRows).asArray(Int32.self)
        let before = positions
        var retained = Array(repeating: 0, count: batch)
        var output: [BatchedDFlashTokenChunk] = []
        for row in 0 ..< batch where !finished[row] {
            let offset = row * (gamma + 2)
            let accepted = Int(packed[offset + gamma])
            var tokens =
                Array(packed[offset ..< (offset + accepted)]).map(Int.init) + [
                    Int(packed[offset + gamma + 1])
                ]
            if let maxTokens = parametersByRow[row].maxTokens {
                tokens = Array(tokens.prefix(max(0, maxTokens - generatedCountByRow[row])))
            }
            if let stop = tokens.firstIndex(where: eosTokenIds.contains) {
                tokens = Array(tokens[...stop])
                finished[row] = true
            }
            retained[row] = min(accepted + 1, tokens.count)
            positions[row] += retained[row]
            generatedCountByRow[row] += tokens.count
            tokenLedgerByRow[row] += tokens
            acceptedCountByRow[row] += min(accepted, tokens.count)
            proposedCountByRow[row] += gamma
            if let token = tokens.last { anchors[row] = token }
            if let maxTokens = parametersByRow[row].maxTokens,
                generatedCountByRow[row] >= maxTokens
            {
                finished[row] = true
            }
            for token in tokens {
                processors[row]?.didSample(token: MLXArray(Int32(token)))
                processors[row]?.didEmit(token: token)
            }
            if finished[row] { finalizeProcessor(row: row) }
            if retained[row] > 0 {
                lastPromptLogitsByRow[row] =
                    verified.logits[
                        row ..< row + 1, (retained[row] - 1) ..< retained[row], 0...]
            }
            output.append(.init(id: rowIDs[row], tokens: tokens, finished: finished[row]))
        }
        let valid = MLXArray(retained.map(Int32.init))
        let recurrent = cache.compactMap { $0 as? MambaCache }
        precondition(recurrent.count == verified.recurrentCaptures.count)
        for (entry, capture) in zip(recurrent, verified.recurrentCaptures) {
            let state = capture.replayBatch(validCounts: valid)
            entry[0] = state.conv
            entry[1] = state.recurrent
            entry.offset = positions.max()!
        }
        for entry in cache where type(of: entry) == KVCacheSimple.self {
            (entry as! KVCacheSimple).commitRows(count: positions.max()!)
        }
        hidden = concatenated(verified.hidden, axis: -1)
        hiddenPositions = before
        hiddenCounts = retained
        roundCount += 1
        return output
    }

    public func synchronize() {
        eval(
            cache.flatMap { $0.innerState() }
                + draftState.contextCaches.flatMap { [$0.keys, $0.values].compactMap { $0 } } + [
                    hidden
                ])
    }

    /// Capture only a whole returned boundary. A partial block is a cache miss.
    public func snapshot(id: Int, retainedTokens: [Int]) throws -> BatchedDFlashPrefixState? {
        guard let row = rowIDs.firstIndex(of: id), tokenLedgerByRow[row] == retainedTokens,
            !(initialChunks ?? []).contains(where: { $0.id == id })
        else {
            return nil
        }
        return try snapshot(id: id)
    }

    /// Capture the complete returned ledger, feeding its unfed anchor only into copied state.
    public func snapshot(id: Int) throws -> BatchedDFlashPrefixState {
        guard let row = rowIDs.firstIndex(of: id),
            !(initialChunks ?? []).contains(where: { $0.id == id })
        else { throw BatchedDFlashError.incompatiblePrefix }
        let ledger = tokenLedgerByRow[row]
        let unfed = ledger.count - positions[row]
        guard (0 ... 1).contains(unfed) else { throw BatchedDFlashError.incompatiblePrefix }
        let rowCache = extractCache(row: row)
        var rowState = draftState.extract(row: row)
        if hiddenCounts[row] > 0 {
            drafter.appendBatchContext(
                hidden: hidden[row ..< row + 1, ..<hiddenCounts[row], 0...],
                positions: [hiddenPositions[row]], validCounts: [hiddenCounts[row]],
                state: &rowState)
        }
        var logits = lastPromptLogitsByRow[row]
        if unfed == 1 {
            let anchor = MLXArray([Int32(ledger.last!)]).reshaped(1, 1)
            let result = target.dflash2Prefill(
                anchor, cache: rowCache, captureLayers: drafter.targetLayerIds)
            drafter.appendBatchContext(
                hidden: concatenated(result.hidden, axis: -1), positions: [positions[row]],
                validCounts: [1], state: &rowState)
            logits = result.logits
        }
        for entry in rowCache {
            if let recurrent = entry as? MambaCache { recurrent.offset = ledger.count }
        }
        eval(
            rowCache.flatMap { $0.innerState() }
                + rowState.contextCaches.flatMap { [$0.keys, $0.values].compactMap { $0 } } + [
                    logits
                ])
        return .init(
            tokens: ledger, targetCache: rowCache, drafterState: rowState, lastLogits: logits,
            targetOwner: target as AnyObject,
            drafterOwner: drafter as AnyObject)
    }

    private func extractCache(row: Int) -> [KVCache] {
        precondition(positions.indices.contains(row))
        return cache.map { entry in
            if let attention = entry as? KVCacheSimple {
                let result = KVCacheSimple()
                if let keys = attention.keys, let values = attention.values {
                    result.keys = contiguous(
                        keys[row ..< (row + 1), 0..., ..<positions[row], 0...] + 0)
                    result.values = contiguous(
                        values[row ..< (row + 1), 0..., ..<positions[row], 0...] + 0)
                }
                result.offset = positions[row]
                return result
            }
            let result = MambaCache()
            for index in 0 ..< 2 {
                if let value = (entry as! MambaCache)[index] {
                    result[index] = contiguous(value[row ..< (row + 1)] + 0)
                }
            }
            result.offset = positions[row]
            return result
        }
    }

    private static func padHidden(_ rows: [MLXArray]) -> MLXArray {
        let width = rows.map { $0.dim(1) }.max()!
        return concatenated(
            rows.map { row in
                concatenated(
                    [row, MLXArray.zeros([1, width - row.dim(1), row.dim(2)], dtype: row.dtype)],
                    axis: 1)
            }, axis: 0)
    }

    static func mergeTargetCaches(_ rows: [[KVCache]]) throws -> [KVCache] {
        guard let first = rows.first, rows.allSatisfy({ $0.count == first.count }) else {
            throw BatchedDFlashError.unsupportedCache
        }
        return try first.indices.map { index in
            let entries = rows.map { $0[index] }
            if entries.allSatisfy({ type(of: $0) == KVCacheSimple.self }) {
                let caches = entries.map { $0 as! KVCacheSimple }
                guard let firstKeys = caches[0].keys, let firstValues = caches[0].values,
                    firstKeys.ndim == 4, firstValues.ndim == 4,
                    caches.allSatisfy({ entry in
                        guard let keys = entry.keys, let values = entry.values else { return false }
                        return entry.offset >= 0 && keys.ndim == 4 && values.ndim == 4
                            && keys.dim(0) == 1 && values.dim(0) == 1
                            && keys.dim(1) == firstKeys.dim(1)
                            && values.dim(1) == firstValues.dim(1)
                            && keys.dim(3) == firstKeys.dim(3)
                            && values.dim(3) == firstValues.dim(3)
                            && keys.dim(2) >= entry.offset && values.dim(2) >= entry.offset
                            && keys.dtype == firstKeys.dtype && values.dtype == firstValues.dtype
                    })
                else { throw BatchedDFlashError.unsupportedCache }
                let width = caches.map(\.offset).max()!
                let merged = KVCacheSimple()
                func arrays(_ key: Bool) -> MLXArray {
                    concatenated(
                        caches.map { entry in
                            let array = (key ? entry.keys : entry.values)![
                                .ellipsis, ..<entry.offset, 0...]
                            let padding = MLXArray.zeros(
                                [1, array.dim(1), width - entry.offset, array.dim(3)],
                                dtype: array.dtype)
                            return concatenated([array, padding], axis: 2)
                        }, axis: 0)
                }
                merged.keys = arrays(true)
                merged.values = arrays(false)
                merged.offset = width
                return merged
            }
            guard entries.allSatisfy({ type(of: $0) == MambaCache.self }) else {
                throw BatchedDFlashError.unsupportedCache
            }
            let merged = MambaCache()
            for slot in 0 ..< 2 {
                let arrays = entries.compactMap { ($0 as! MambaCache)[slot] }
                guard let sample = arrays.first, arrays.count == entries.count,
                    sample.ndim > 0,
                    arrays.allSatisfy({ value in
                        value.ndim == sample.ndim && value.dim(0) == 1
                            && Array(value.shape.dropFirst()) == Array(sample.shape.dropFirst())
                            && value.dtype == sample.dtype
                    })
                else { throw BatchedDFlashError.unsupportedCache }
                merged[slot] = concatenated(arrays, axis: 0)
            }
            merged.offset = entries.map(\.offset).max()!
            return merged
        }
    }
}

extension GatedDeltaCapture {
    /// Replay each row's accepted prefix without sharing an acceptance count across rows.
    func replayBatch(validCounts: MLXArray) -> (recurrent: MLXArray, conv: MLXArray) {
        let outputs = batchedDFlashReplay([
            q, k, v, a, b, aLog, dtBias, initialState, convInput, validCounts,
        ])
        return (outputs[0], outputs[1])
    }
}

private let batchedDFlashReplay: @Sendable ([MLXArray]) -> [MLXArray] = compile { inputs in
    let (q, k, v) = (inputs[0], inputs[1], inputs[2])
    let valid = inputs[9].asType(.int32)
    let mask = MLXArray(Int32(0) ..< Int32(q.dim(1))).reshaped(1, -1) .< valid.reshaped(-1, 1)
    let (_, state) = gatedDeltaUpdate(
        q: q, k: k, v: v, a: inputs[3], b: inputs[4], aLog: inputs[5], dtBias: inputs[6],
        state: inputs[7], mask: mask)
    let convInput = inputs[8]
    let rows =
        (valid.reshaped(-1, 1)
        + MLXArray(Int32(0) ..< Int32(convInput.dim(1) - q.dim(1))).reshaped(1, -1))
        .reshaped(q.dim(0), -1, 1)
    return [state, contiguous(takeAlong(convInput, rows, axis: 1))]
}
