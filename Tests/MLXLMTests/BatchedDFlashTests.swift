// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct BatchedDFlashTests {
    private func target() throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {"model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":4,
                 "layer_types":["linear_attention","full_attention","linear_attention","full_attention"],
                 "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
                 "linear_num_value_heads":4,"linear_num_key_heads":2,"linear_key_head_dim":16,
                 "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"vocab_size":128,
                 "tie_word_embeddings":false,"partial_rotary_factor":1.0}
                """.utf8))
        MLXRandom.seed(72)
        let model = Qwen35TextModel(configuration)
        try model.prepare()
        // An exactly tied head makes accepted token identity deterministic while the
        // attention and recurrent layers still exercise real nonconstant state.
        model.update(
            parameters: ModuleParameters.unflattened([
                ("lm_head.weight", MLXArray.zeros([128, 64]))
            ]))
        model.train(false)
        return model
    }

    private func drafter() throws -> DFlash2DraftModel {
        let configuration = try JSONDecoder().decode(
            DFlash2Configuration.self,
            from: Data(
                """
                {"architectures":["DFlashDraftModel"],"model_type":"qwen3",
                 "dflash_config":{"block_size":4,"mask_token_id":126,"target_layer_ids":[1,3]},
                 "hidden_size":64,"num_hidden_layers":2,"intermediate_size":96,
                 "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
                 "num_target_layers":4,"vocab_size":128,"sliding_window":8,
                 "layer_types":["sliding_attention","full_attention"],"tie_word_embeddings":true,
                 "rms_norm_eps":1e-6,"rope_parameters":{"rope_theta":10000,"rope_type":"default"}}
                """.utf8))
        MLXRandom.seed(73)
        let model = DFlash2DraftModel(configuration)
        model.train(false)
        return model
    }

    private func inputs() -> [LMInput] {
        [[Int32(3), 4, 5], [Int32(6), 7, 8, 9, 10, 11, 12]].map { LMInput(tokens: MLXArray($0)) }
    }

    private func drain(_ iterator: BatchedDFlashTokenIterator) -> [[Int]] {
        var output = Array(repeating: [Int](), count: iterator.positions.count)
        while let chunks = iterator.next() {
            for chunk in chunks { output[chunk.id] += chunk.tokens }
        }
        iterator.synchronize()
        return output
    }

    private func record(_ chunks: [BatchedDFlashTokenChunk], into output: inout [Int: [Int]]) {
        for chunk in chunks {
            output[chunk.id, default: []] += chunk.tokens
        }
    }

    private func drain(_ iterator: BatchedDFlashTokenIterator, into output: inout [Int: [Int]]) {
        while let chunks = iterator.next() { record(chunks, into: &output) }
        iterator.synchronize()
    }

    private func expectSameTargetState(
        _ actual: BatchedDFlashPrefixState, _ expected: BatchedDFlashPrefixState
    ) {
        #expect(actual.tokens == expected.tokens)
        #expect(actual.targetCache.count == expected.targetCache.count)
        for (lhs, rhs) in zip(actual.targetCache, expected.targetCache) {
            #expect(lhs.offset == rhs.offset && lhs.state.count == rhs.state.count)
            for (a, b) in zip(lhs.state, rhs.state) {
                guard a.shape == b.shape else {
                    Issue.record("Target state shapes differ: \(a.shape) versus \(b.shape)")
                    continue
                }
                #expect(abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) < 0.002)
            }
        }
    }

    @Test("A real batched v1 proposal preserves each solo row's context mask and logits")
    func proposalParity() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            target.update(
                parameters: ModuleParameters.unflattened([
                    ("lm_head.weight", MLXRandom.normal([128, 64]) * 0.3)
                ]))
            let draft = try drafter()
            draft.recordsIntermediates = true
            draft.compiledTracesEnabled = false
            let hidden = MLXRandom.normal([2, 7, 128])
            let block = MLXArray([Int32(3), 126, 126, 126, 4, 126, 126, 126]).reshaped(2, 4)
            var state = draft.makeBatchState(batchSize: 2)
            let actual = draft.proposeBatch(
                .init(
                    block: block, targetHidden: hidden,
                    contextPositions: [0, 11], validCounts: [3, 7]), target: target, state: &state)
            let actualLogits = try #require(draft.lastLogits)
            for row in 0 ..< 2 {
                var soloState = draft.makeState()
                let count = row == 0 ? 3 : 7
                let expected = draft.propose(
                    .init(
                        block: block[row ..< (row + 1)],
                        targetHidden: hidden[row ..< (row + 1), ..<count, 0...],
                        contextPosition: row == 0 ? 0 : 11,
                        validRows: MLXArray(Int32(count)), temperature: 0), target: target,
                    state: &soloState)
                let expectedLogits = try #require(draft.lastLogits)
                #expect(arrayEqual(actual[row ..< (row + 1)], expected.tokens).item(Bool.self))
                #expect(
                    abs(actualLogits[row ..< (row + 1)] - expectedLogits).max().item(Float.self)
                        < 0.0001)
            }
        }
    }

    @Test("Ragged rows accept different prefixes and retain exact recurrent timelines")
    func mixedAcceptance() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let draft = ScriptedBatchDraft()
            let iterator = try BatchedDFlashTokenIterator(
                inputs: inputs(), mainModel: target,
                drafter: draft, parameters: .init(maxTokens: 19, temperature: 0), blockSize: 4)
            let output = drain(iterator)
            #expect(output == Array(repeating: Array(repeating: 0, count: 19), count: 2))
            #expect(iterator.acceptedCountByRow[0] > iterator.acceptedCountByRow[1])
            #expect(draft.batchWidths.allSatisfy { $0 == 2 })
            for row in output.indices {
                let reference = try target.newCache(parameters: nil)
                let prompt = inputs()[row].text.tokens
                let tokens = concatenated([prompt, MLXArray(output[row].map(Int32.init))])
                _ = target.dflash2Prefill(
                    tokens.reshaped(1, -1), cache: reference, captureLayers: [])
                let snapshot = try iterator.snapshot(id: row)
                let actual = snapshot.targetCache
                for (a, b) in zip(actual, reference) {
                    if a is KVCacheSimple {
                        #expect(a.offset == b.offset)
                    } else {
                        // Ordinary GDN forward leaves this metadata at zero; DFlash tracks its logical position.
                        #expect(a.offset == snapshot.tokens.count)
                    }
                    for (x, y) in zip(a.state, b.state) {
                        #expect(x.shape == y.shape)
                        #expect(
                            abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self)
                                < 0.002)
                    }
                }
            }
        }
    }

    @Test("Real v1 batch proposal and warm target/draft prefix match cold greedy streams")
    func warmV1() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let draft = try drafter()
            let parameters = GenerateParameters(maxTokens: 17, temperature: 0, seed: 7)
            let cold = try BatchedDFlashTokenIterator(
                inputs: inputs(), mainModel: target, drafter: draft, parameters: parameters)
            let expected = drain(cold)
            let prefixes = try inputs().map { input in
                try BatchedDFlashTokenIterator.makePrefixState(
                    input: .init(tokens: input.text.tokens[..<2]), mainModel: target, drafter: draft
                )
            }
            let suffixes = inputs().map { LMInput(tokens: $0.text.tokens[2...]) }
            let warm = try BatchedDFlashTokenIterator(
                inputs: suffixes, mainModel: target, drafter: draft,
                parameters: parameters, prefixStates: prefixes.map(Optional.some))
            #expect(drain(warm) == expected)
            #expect(warm.positions == cold.positions)
            let plainTokens = try inputs().map { input in
                let plain = try TokenIterator(input: input, model: target, parameters: parameters)
                defer { plain.synchronizeGeneration() }
                return Array(plain)
            }
            #expect(plainTokens == expected)
        }
    }

    @Test("Sampled rejection with a point-mass draft preserves the target distribution")
    func rejectionDistribution() {
        Device.withDefaultDevice(.cpu) {
            let state = MLXRandom.RandomState(seed: 123)
            var counts = [Int](repeating: 0, count: 3)
            for _ in 0 ..< 2000 {
                let result = withRandomState(state) {
                    DFlash2SpeculativeTokenIterator.rejectionSample(
                        drafts: MLXArray([Int32(0)]),
                        targetProbs: MLXArray([Float(0.2), 0.3, 0.5, 0.2, 0.3, 0.5]).reshaped(2, 3),
                        draftProbs: MLXArray.ones([1, 1]),
                        candidates: MLXArray([Int32(0)]).reshaped(1, 1))
                }
                let packed = result.packed.asArray(Int32.self)
                counts[packed[1] == 1 ? 0 : Int(packed[2])] += 1
            }
            for (count, probability) in zip(counts, [0.2, 0.3, 0.5]) {
                #expect(abs(Double(count) / 2000 - probability) < 0.045)
            }
        }

    }

    @Test("Sampled cohorts reproduce each row's seeded stream through real verification")
    func sampledReproducibility() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let parameters = GenerateParameters(
                maxTokens: 13, temperature: 0.8, topP: 0.9, seed: 7)
            let first = try BatchedDFlashTokenIterator(
                inputs: inputs(), mainModel: target,
                drafter: ScriptedBatchDraft(), parameters: parameters)
            let a = drain(first)
            let second = try BatchedDFlashTokenIterator(
                inputs: inputs(), mainModel: target,
                drafter: ScriptedBatchDraft(), parameters: parameters)
            let b = drain(second)
            #expect(a == b)
            #expect(a.allSatisfy { $0.count == 13 })
            #expect(a[0] != a[1])
            #expect(first.proposedCountByRow.allSatisfy { $0 > 0 })
        }
    }

    @Test("Live insertion and cancellation preserve stable IDs and each row's random stream")
    func liveMembershipPreservesSeeds() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let parameters = GenerateParameters(
                maxTokens: 19, temperature: 0.8, topP: 0.9, seed: 37)
            let draft = ScriptedBatchDraft(mixedAcceptance: false)
            let iterator = try BatchedDFlashTokenIterator(
                inputs: [inputs()[0]], mainModel: target, drafter: draft, parameters: parameters)
            let control = try BatchedDFlashTokenIterator(
                inputs: [inputs()[0]], mainModel: target,
                drafter: ScriptedBatchDraft(mixedAcceptance: false), parameters: parameters)
            let expected = drain(control)[0]
            var output = [Int: [Int]]()
            for _ in 0 ..< 2 {
                let chunks = iterator.next()
                record(try #require(chunks), into: &output)
            }
            let inserted = try iterator.insert(inputs: [inputs()[1]], parameters: parameters)
            #expect(inserted == [1])
            #expect(iterator.ids == [0, 1] && iterator.activeIDs == [0, 1])
            var reachedBatch = false
            for _ in 0 ..< 4 {
                let chunks = iterator.next()
                record(try #require(chunks), into: &output)
                if draft.batchWidths.contains(2) {
                    reachedBatch = true
                    break
                }
            }
            #expect(reachedBatch)
            iterator.remove(inserted)
            #expect(iterator.ids == [0] && iterator.activeIDs == [0])
            let removedPrefix = try #require(output[1])
            while let chunks = iterator.next() {
                #expect(chunks.allSatisfy { $0.id == 0 })
                record(chunks, into: &output)
            }
            iterator.synchronize()
            #expect(output[0] == expected)
            #expect(output[1] == removedPrefix)
            let surviving = try iterator.snapshot(id: 0)
            let unchangedCohort = try control.snapshot(id: 0)
            expectSameTargetState(surviving, unchangedCohort)

            let newRowControl = try BatchedDFlashTokenIterator(
                inputs: [inputs()[1]], mainModel: target,
                drafter: ScriptedBatchDraft(mixedAcceptance: false),
                parameters: .init(maxTokens: 19, temperature: 0.8, topP: 0.9, seed: 38))
            #expect(removedPrefix == Array(drain(newRowControl)[0].prefix(removedPrefix.count)))

            iterator.remove(iterator.ids)
            #expect(iterator.ids.isEmpty && iterator.activeIDs.isEmpty)
            let empty = iterator.next()
            #expect(empty == nil)
            let restarted = try iterator.insert(
                inputs: [inputs()[1]], parameters: parameters, maxTokens: [3])
            #expect(restarted == [2])
            var restartedOutput = [Int: [Int]]()
            drain(iterator, into: &restartedOutput)
            let restartedControl = try BatchedDFlashTokenIterator(
                inputs: [inputs()[1]], mainModel: target,
                drafter: ScriptedBatchDraft(mixedAcceptance: false),
                parameters: .init(maxTokens: 3, temperature: 0.8, topP: 0.9, seed: 39))
            #expect(restartedOutput[2] == drain(restartedControl)[0])
            #expect(iterator.ids == [2] && iterator.activeIDs.isEmpty)
        }
    }

    @Test("Removing a pending arrival drops its tokens and finalizes its processor once")
    func cancelPendingArrival() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let original = DFlashProcessorCounts()
            let cancelled = DFlashProcessorCounts()
            let parameters = GenerateParameters(maxTokens: 7, temperature: 0)
            let iterator = try BatchedDFlashTokenIterator(
                inputs: [inputs()[0]], mainModel: target,
                drafter: ScriptedBatchDraft(mixedAcceptance: false), parameters: parameters,
                components: .init(logitProcessorFactory: { DFlashCountingProcessor(original) }))
            let first = iterator.next()
            #expect(first?.map(\.id) == [0])
            let ids = try iterator.insert(
                inputs: [inputs()[1]], parameters: parameters,
                components: .init(logitProcessorFactory: { DFlashCountingProcessor(cancelled) }))
            #expect(ids == [1])
            iterator.remove(ids)
            let afterCancellation = cancelled.counts
            #expect(afterCancellation.finalized == 1)
            var output = [Int: [Int]]()
            drain(iterator, into: &output)
            #expect(output.keys.sorted() == [0])
            #expect(cancelled.counts.sampled == afterCancellation.sampled)
            #expect(cancelled.counts.emitted == 0)
            #expect(original.counts.finalized == 1 && cancelled.counts.finalized == 1)
            iterator.remove([0, 1])
            iterator.synchronize()
            #expect(original.counts.finalized == 1 && cancelled.counts.finalized == 1)
        }
    }

    @Test("Checked snapshots reject discarded chunk suffixes and require the emitted anchor")
    func checkedSnapshotBoundary() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let draft = ScriptedBatchDraft(mixedAcceptance: false)
            let iterator = try BatchedDFlashTokenIterator(
                inputs: [inputs()[0]], mainModel: target, drafter: draft,
                parameters: .init(maxTokens: 13, temperature: 0), blockSize: 4)
            let first = iterator.next()
            let initial = try #require(first?.first)
            let next = iterator.next()
            let chunk = try #require(next?.first)
            #expect(chunk.tokens.count >= 3)
            let prompt = inputs()[0].text.tokens.asArray(Int32.self).map(Int.init)
            let retained = prompt + initial.tokens + chunk.tokens
            #expect(iterator.tokenLedgerByRow[0] == retained)
            let positions = iterator.positions
            let partial = try iterator.snapshot(
                id: 0, retainedTokens: Array(retained.dropLast(2)))
            let missingAnchor = try iterator.snapshot(
                id: 0, retainedTokens: Array(retained.dropLast()))
            var differentHistory = retained
            differentHistory[differentHistory.count - 1] = 1
            let changed = try iterator.snapshot(id: 0, retainedTokens: differentHistory)
            #expect(partial == nil && missingAnchor == nil && changed == nil)
            let captured = try iterator.snapshot(id: 0, retainedTokens: retained)
            let snapshot = try #require(captured)
            #expect(snapshot.tokens == retained)
            #expect(snapshot.targetCache.allSatisfy { $0.offset == retained.count })
            #expect(iterator.positions == positions)
            iterator.remove([0])
            let suffix = LMInput(tokens: MLXArray([Int32(22), 23]))
            let warm = try BatchedDFlashTokenIterator(
                inputs: [suffix], mainModel: target, drafter: draft,
                parameters: .init(maxTokens: 7, temperature: 0), prefixStates: [snapshot])
            let fresh = try BatchedDFlashTokenIterator.makePrefixState(
                input: .init(tokens: MLXArray(retained.map(Int32.init))),
                mainModel: target, drafter: draft)
            let control = try BatchedDFlashTokenIterator(
                inputs: [suffix], mainModel: target, drafter: draft,
                parameters: .init(maxTokens: 7, temperature: 0), prefixStates: [fresh])
            #expect(drain(warm) == drain(control))
            #expect(warm.positions == control.positions)
        }
    }

    @Test("EOS in the first token finishes rows without a verify pass")
    func eos() throws {
        try Device.withDefaultDevice(.cpu) {
            let iterator = try BatchedDFlashTokenIterator(
                inputs: inputs(), mainModel: target(), drafter: ScriptedBatchDraft(),
                parameters: .init(maxTokens: 9, temperature: 0), eosTokenIds: [0])
            #expect(drain(iterator) == [[0], [0]])
            #expect(iterator.roundCount == 0)
        }
    }

    @Test(
        "Generated snapshots resume a nonempty suffix from active and EOS boundaries",
        arguments: [false, true])
    func generatedSnapshot(eos: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let draft = try drafter()
            let iterator = try BatchedDFlashTokenIterator(
                inputs: inputs(), mainModel: target,
                drafter: draft, parameters: .init(maxTokens: 9, temperature: 0),
                eosTokenIds: eos ? [0] : [])
            let prompt = inputs()[0].text.tokens.asArray(Int32.self).map(Int.init)
            let beforeEmission = try iterator.snapshot(id: 0, retainedTokens: prompt)
            #expect(beforeEmission == nil)
            #expect(throws: BatchedDFlashError.incompatiblePrefix) {
                _ = try iterator.snapshot(id: 0)
            }
            _ = iterator.next()
            if !eos { _ = iterator.next() }
            let positions = iterator.positions
            let ledgers = iterator.tokenLedgerByRow
            let snapshots = try inputs().indices.map { try iterator.snapshot(id: $0) }
            #expect(iterator.positions == positions)
            for row in inputs().indices {
                #expect(snapshots[row].tokens == ledgers[row])
                #expect(ledgers[row].count == positions[row] + 1)
                #expect(snapshots[row].targetCache.allSatisfy { $0.offset == ledgers[row].count })
            }
            // Snapshotting an active row must leave its original generation usable.
            _ = drain(iterator)
            #expect(iterator.generatedCountByRow == [eos ? 1 : 9, eos ? 1 : 9])
            let controls = try ledgers.map { tokens in
                try BatchedDFlashTokenIterator.makePrefixState(
                    input: .init(tokens: MLXArray(tokens.map(Int32.init))), mainModel: target,
                    drafter: draft)
            }
            let suffixes = [
                LMInput(tokens: MLXArray([Int32(22), 23])),
                LMInput(tokens: MLXArray([Int32(24), 25])),
            ]
            let parameters = GenerateParameters(maxTokens: 7, temperature: 0, seed: 11)
            let warm = try BatchedDFlashTokenIterator(
                inputs: suffixes, mainModel: target,
                drafter: draft, parameters: parameters, prefixStates: snapshots.map(Optional.some))
            let split = try BatchedDFlashTokenIterator(
                inputs: suffixes, mainModel: target,
                drafter: draft, parameters: parameters, prefixStates: controls.map(Optional.some))
            #expect(drain(warm) == drain(split))
            #expect(warm.positions == split.positions)
            for row in inputs().indices {
                let actual = try warm.snapshot(id: row)
                let expected = try split.snapshot(id: row)
                for (a, b) in zip(actual.targetCache, expected.targetCache) {
                    for (x, y) in zip(a.state, b.state) {
                        #expect(x.shape == y.shape)
                        #expect(
                            abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self)
                                < 0.002)
                    }
                }
            }
        }
    }

    @Test("Warm entries reject another target or drafter before generation")
    func prefixIdentity() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try target()
            let draft = try drafter()
            let prefix = try BatchedDFlashTokenIterator.makePrefixState(
                input: inputs()[0], mainModel: target, drafter: draft)
            let suffix = LMInput(tokens: MLXArray([Int32(4), 5]))
            let otherTarget = try self.target()
            let otherDraft = try drafter()
            #expect(throws: BatchedDFlashError.incompatiblePrefix) {
                _ = try BatchedDFlashTokenIterator(
                    inputs: [suffix], mainModel: otherTarget,
                    drafter: draft, parameters: .init(maxTokens: 3), prefixStates: [prefix])
            }
            #expect(throws: BatchedDFlashError.incompatiblePrefix) {
                _ = try BatchedDFlashTokenIterator(
                    inputs: [suffix], mainModel: target,
                    drafter: otherDraft, parameters: .init(maxTokens: 3), prefixStates: [prefix])
            }
        }
    }
}

private final class ScriptedBatchDraft: Module, BatchedDFlashDrafterModel {
    let blockSize = 4
    let maskTokenId = 126
    let targetLayerIds = [1, 3]
    let targetGeometry = DFlash2Geometry(layerCount: 4, hiddenSize: 64, vocabularySize: 128)
    let contextWindow = 8
    let supportsBatchedDFlashV1 = true
    var batchWidths: [Int] = []
    let mixedAcceptance: Bool
    init(mixedAcceptance: Bool = true) { self.mixedAcceptance = mixedAcceptance }
    func makeState() -> DFlash2DrafterState { .init(contextCaches: []) }
    func makeBatchState(batchSize: Int) -> BatchedDFlashDrafterState { .init(contextCaches: []) }
    func appendBatchContext(
        hidden: MLXArray, positions: [Int], validCounts: [Int],
        state: inout BatchedDFlashDrafterState
    ) {}
    func propose(
        _ request: DFlash2ProposalRequest, target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        fatalError("The batched test must not silently loop over solo draft calls")
    }
    func proposeBatch(
        _ request: BatchedDFlashProposalRequest, target: any DFlash2TargetModel,
        state: inout BatchedDFlashDrafterState
    ) -> MLXArray {
        batchWidths.append(request.block.dim(0))
        return MLXArray(
            (0 ..< request.block.dim(0)).flatMap { row in
                (0 ..< (request.block.dim(1) - 1)).map {
                    Int32(mixedAcceptance && row == 1 && $0 == 1 ? 1 : 0)
                }
            }
        ).reshaped(request.block.dim(0), request.block.dim(1) - 1)
    }
}

private final class DFlashProcessorCounts: @unchecked Sendable {
    private let lock = NSLock()
    private var values = (sampled: 0, emitted: 0, finalized: 0)
    var counts: (sampled: Int, emitted: Int, finalized: Int) { lock.withLock { values } }
    func sampled() { lock.withLock { values.sampled += 1 } }
    func emitted() { lock.withLock { values.emitted += 1 } }
    func finalized() { lock.withLock { values.finalized += 1 } }
}

private struct DFlashCountingProcessor: LogitProcessor {
    let counts: DFlashProcessorCounts
    init(_ counts: DFlashProcessorCounts) { self.counts = counts }
    mutating func prompt(_: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
    mutating func didSample(token _: MLXArray) { counts.sampled() }
    mutating func didEmit(token _: MLXArray) { counts.emitted() }
    mutating func finalizeGeneration() { counts.finalized() }
}
