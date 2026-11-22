// Copyright © 2026 Apple Inc.

// Speculative decoding of a tiny random Inkling 2 Mini with its bundled DSpark draft.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Inkling 2 Mini at the size of `Inkling2Tests`' models, 14 layer visits, with a draft whose
/// local layers see 4 context positions and whose block proposes 3 tokens.
private func dsparkConfigJSON(targetLayers: [Int] = [1, 4, 9, 14]) -> String {
    """
    {
        "model_type": "inkling2_loop", "image_token_id": 127,
        "text_config": {
            "hidden_size": 64, "num_hidden_layers": 8, "vocab_size": 128,
            "unpadded_vocab_size": 120, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2, "swa_head_dim": 16, "sliding_window_size": 8,
            "local_layer_ids": [0, 1, 2, 3, 4, 6], "d_rel": 4, "rel_extent": 8,
            "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 3.0, "rel_l2_clip_threshold": 2.0,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 3,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 64,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.0,
            "final_logit_softcapping": 30.0,
            "loop_config": {"core_start": 3, "core_end": 6, "num_passes": 3,
                "gate_rank": 8, "is_varloop": false}
        },
        "mtp_config": {
            "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96,
            "max_position_embeddings": 4096, "q_bias": false, "o_bias": false,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "local_layer_ids": [0, 2],
            "sliding_window_size": 4, "block_size": 4, "markov_rank": 8,
            "markov_head_type": "rnn_hidden", "target_layer_ids": \(targetLayers)
        }
    }
    """
}

private let vocabulary = 120

private func dsparkConfiguration(targetLayers: [Int] = [1, 4, 9, 14]) throws
    -> Inkling2DSparkConfiguration
{
    try JSONDecoder().decode(
        Inkling2DSparkConfiguration.self,
        from: Data(dsparkConfigJSON(targetLayers: targetLayers).utf8))
}

/// Random weights at the reference's scale: norms and clip scales near one, other weights
/// normal over the square root of their fan-in.
private func randomize(_ model: Module, seed: UInt64) {
    MLXRandom.seed(seed)
    let weights = model.parameters().flattened().map { name, value -> (String, MLXArray) in
        if name.hasSuffix("norm.weight") || name.hasSuffix("l2_clip.scale") {
            return (name, 1 + 0.1 * MLXRandom.normal(value.shape))
        }
        let fanIn = value.ndim > 1 ? value.dim(-1) : 16
        return (name, MLXRandom.normal(value.shape) * pow(Float(fanIn), -0.5))
    }
    model.update(parameters: ModuleParameters.unflattened(weights))
    eval(model)
}

private func makeTarget() throws -> Inkling2Model {
    let model = Inkling2Model(try dsparkConfiguration().target)
    randomize(model, seed: 0)
    return model
}

private func makeDraft(seed: UInt64 = 5) throws -> Inkling2DSparkDraftModel {
    let draft = Inkling2DSparkDraftModel(try dsparkConfiguration())
    randomize(draft, seed: seed)
    return draft
}

private func prompt(_ length: Int) -> LMInput {
    LMInput(tokens: MLXArray((0 ..< length).map { Int32(($0 * 7 + 3) % vocabulary) }))
}

/// Parameters whose cache can step back over a whole block.
private func speculative(maxTokens: Int, rewind: Int = 4) throws -> GenerateParameters {
    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    parameters.kvCache = KVCacheConfiguration(rewind: try .init(maxTokens: rewind))
    return parameters
}

/// What plain decoding samples at `temperature`, seeded, and the same parameters with a
/// rewind window for a speculative run.
private func sampled(
    _ model: any LanguageModel, _ input: LMInput, maxTokens: Int, temperature: Float,
    topP: Float = 1
) throws -> (tokens: [Int], speculative: GenerateParameters) {
    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: topP)
    parameters.seed = 7
    var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
    var tokens = [Int]()
    while let token = iterator.next() { tokens.append(token) }
    parameters.kvCache = KVCacheConfiguration(rewind: try .init(maxTokens: 4))
    return (tokens, parameters)
}

private func greedy(_ model: any LanguageModel, _ input: LMInput, maxTokens: Int) throws -> [Int] {
    var iterator = try TokenIterator(
        input: input, model: model,
        parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
    var tokens = [Int]()
    while let token = iterator.next() { tokens.append(token) }
    return tokens
}

private func onCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(.cpu, body)
}

/// Proposes the target's own continuation, right for a scripted number of proposals per round
/// and wrong after, so rounds keep every count from none to all. It draws nothing from the
/// request's random stream, so it also runs with a temperature.
private final class ScriptedDrafter: Module, MTPDrafterModel {
    let expected: [Int]
    let promptLength: Int
    let rightProposals: [Int]
    private var round = 0

    init(expected: [Int], promptLength: Int, rightProposals: [Int]) {
        self.expected = expected
        self.promptLength = promptLength
        self.rightProposals = rightProposals
    }

    var requiresSharedTargetKV: Bool { false }
    var requiresGreedySampling: Bool { false }

    func validateCompatibility(with target: any LanguageModel) throws {}

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] { weights }

    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        // The round's first token sits at `queryOffset`.
        let index = queryOffset - promptLength
        let right = rightProposals[round % rightProposals.count]
        round += 1
        let proposals = (1 ..< blockSize).map { step -> Int32 in
            let token = index + step < expected.count ? expected[index + step] : 0
            return Int32(step <= right ? token : (token + 1) % vocabulary)
        }
        return MLXArray(proposals).reshaped(1, -1)
    }
}

@Suite(.serialized)
struct Inkling2DSparkTests {

    @Test(
        "Rounds that keep none to all of their proposals give the greedy tokens",
        arguments: [5, 21])
    func roundsRewindExactly(promptLength: Int) throws {
        try onCPU {
            let target = try makeTarget()
            let input = prompt(promptLength)
            let expected = try greedy(target, input, maxTokens: 24)
            let drafter = ScriptedDrafter(
                expected: expected, promptLength: promptLength, rightProposals: [0, 3, 1, 2])
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target, drafter: drafter,
                parameters: try speculative(maxTokens: 24), blockSize: 4)
            var output = [Int]()
            while let token = iterator.next() { output.append(token) }

            #expect(output == expected)
            #expect(iterator.passthroughReason == nil)
            #expect(iterator.acceptedCount > 0)
            #expect(iterator.acceptedCount < iterator.proposedCount)
        }
    }

    @Test(
        "With a temperature, rounds that keep none to all of their proposals sample plain decoding's tokens",
        arguments: [5, 21])
    func sampledRoundsRewindExactly(promptLength: Int) throws {
        try onCPU {
            let target = try makeTarget()
            let input = prompt(promptLength)
            let (expected, parameters) = try sampled(
                target, input, maxTokens: 24, temperature: 0.8)
            let drafter = ScriptedDrafter(
                expected: expected, promptLength: promptLength, rightProposals: [0, 3, 1, 2])
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target, drafter: drafter, parameters: parameters,
                blockSize: 4)
            var output = [Int]()
            while let token = iterator.next() { output.append(token) }

            #expect(output == expected)
            #expect(iterator.passthroughReason == nil)
            #expect(iterator.acceptedCount > 0)
            #expect(iterator.acceptedCount < iterator.proposedCount)
        }
    }

    @Test(
        "With a temperature, the bundled draft samples plain decoding's tokens",
        arguments: [(Float(1.0), Float(1.0), 2), (0.7, 0.9, 4)])
    func draftKeepsSampledTokens(temperature: Float, topP: Float, blockSize: Int) throws {
        try onCPU {
            let target = try makeTarget()
            let input = prompt(13)
            let (expected, parameters) = try sampled(
                target, input, maxTokens: 20, temperature: temperature, topP: topP)
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target, drafter: try makeDraft(),
                parameters: parameters, blockSize: blockSize)
            var output = [Int]()
            while let token = iterator.next() { output.append(token) }

            #expect(output == expected)
            #expect(iterator.passthroughReason == nil)
            #expect(iterator.proposedCount > 0)
        }
    }

    @Test("Without a rewind window the target cannot stage a round")
    func needsRewindWindow() throws {
        let target = try makeTarget()
        let drafter = ScriptedDrafter(expected: [], promptLength: 5, rightProposals: [0])
        #expect(throws: (any Error).self) {
            _ = try MTPSpeculativeTokenIterator(
                input: prompt(5), mainModel: target, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 8, temperature: 0), blockSize: 4)
        }
    }

    @Test(
        "Drafting with the bundled draft gives the greedy tokens", arguments: [2, 4])
    func draftKeepsGreedyTokens(blockSize: Int) throws {
        try onCPU {
            let target = try makeTarget()
            let input = prompt(13)
            let expected = try greedy(target, input, maxTokens: 20)
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target, drafter: try makeDraft(),
                parameters: try speculative(maxTokens: 20), blockSize: blockSize)
            var output = [Int]()
            while let token = iterator.next() { output.append(token) }

            #expect(output == expected)
            #expect(iterator.passthroughReason == nil)
            #expect(iterator.proposedCount > 0)
        }
    }

    @Test("The target emits the taps the draft names, from the embeddings to the final state")
    func targetEmitsTaps() throws {
        try onCPU {
            let target = try makeTarget()
            let text = target.model.languageModel
            let tokens = prompt(6).text.tokens[.newAxis]
            var state = LMOutput.State()
            state[mtpEmitFlagKey] = true
            state[mtpTargetLayerIdsKey] = [0, 9, 14]
            let output = target(
                LMInput.Text(tokens: tokens), cache: try target.newCache(parameters: nil),
                state: state)
            let taps = try #require(output.state?[mtpLastHiddenStatesKey])
            #expect(taps.shape == [1, 6, 3 * 64])

            let embeddings = text.embed(tokens)
            let hidden = text.hiddenStates(
                embeddings: embeddings, cache: try target.newCache(parameters: nil),
                capturing: []
            ).hidden
            let logits = target(
                LMInput.Text(tokens: tokens), cache: try target.newCache(parameters: nil),
                state: nil
            ).logits
            #expect(maxDifference(taps[0..., 0..., ..<64], embeddings) == 0)
            #expect(maxDifference(taps[0..., 0..., 128...], hidden) == 0)
            #expect(maxDifference(output.logits, logits) == 0)
        }
    }

    @Test("Rewindable convolution states give the logits plain ones give")
    func rewindableCacheMatchesPlain() throws {
        try onCPU {
            let target = try makeTarget()
            let tokens = prompt(20).text.tokens[.newAxis]
            let parameters = try speculative(maxTokens: 1)
            let rewindable = try target.newCache(parameters: parameters)
            #expect(rewindable.count == 6 * 14)
            #expect(rewindable[2] is Inkling2ConvolutionCache)
            // The same attention caches, which a rewind window also widens, with the
            // convolution states in one plain cache per visit.
            let reference = try target.newCache(parameters: parameters)
            let plain = stride(from: 0, to: reference.count, by: 6).flatMap {
                [reference[$0], reference[$0 + 1], ArraysCache(size: 4)]
            }
            for range in [0 ..< 12, 12 ..< 13, 13 ..< 20] {
                let chunk = tokens[0..., range]
                #expect(
                    maxDifference(target(chunk, cache: plain), target(chunk, cache: rewindable))
                        == 0)
            }
        }
    }

    @Test("A rewindable convolution state steps back within its last call")
    func rewindableCacheTrimsLastCall() throws {
        let rows = MLXArray(0 ..< 30).asType(.float32).reshaped(1, 15, 2)
        let cache = Inkling2ConvolutionCache(stateLength: 3, rollbackCapacity: 4)
        // A prompt of 6 positions, longer than the rewind window, after a zero state.
        cache.record(window: rows[0..., ..<9], length: 6)
        #expect(cache.offset == 6)
        #expect(cache.maxTrimCount == 4)
        #expect(maxDifference(cache[0]!, rows[0..., 6 ..< 9]) == 0)
        // A verify of 3 positions after that state.
        cache.record(window: rows[0..., 6 ..< 12], length: 3)
        #expect(cache.offset == 9)
        #expect(cache.maxTrimCount == 3)
        #expect(cache.isTrimmable(after: 4))
        #expect(cache.trim(2) == 2)
        #expect(cache.offset == 7)
        #expect(maxDifference(cache[0]!, rows[0..., 7 ..< 10]) == 0)
        #expect(cache.trim(2) == 0)
        #expect(cache.trim(1) == 1)
        #expect(maxDifference(cache[0]!, rows[0..., 6 ..< 9]) == 0)
        #expect(cache.maxTrimCount == 0)
    }

    @Test("A ring of keys writes in place, returns long writes in order and steps back")
    func ringCache() throws {
        // A window of 4 and a reserve of 2: six slots, then the zero slot. Each row holds its
        // position plus one, plus `tag`, so a rewrite can be told apart.
        let cache = Inkling2RingCache(window: 4, reserve: 2)
        func rows(_ positions: Range<Int>, tag: Float = 0) -> MLXArray {
            MLXArray(positions.map { Float($0 + 1) + tag }).reshaped(1, 1, -1, 1)
        }
        func flat(_ x: MLXArray) -> [Float] { x.reshaped(-1).asArray(Float.self) }

        // Until the ring wraps, in-place writes return the slots up to the offset, in order.
        let fresh = Inkling2RingCache(window: 4, reserve: 2)
        #expect(flat(fresh.update(keys: rows(0 ..< 2), values: rows(0 ..< 2)).0) == [1, 2])
        #expect(!fresh.returnedRing(writing: 2))
        #expect(flat(fresh.update(keys: rows(2 ..< 4), values: rows(2 ..< 4)).0) == [1, 2, 3, 4])
        #expect(fresh.trim(1) == 1)
        let rewritten = fresh.update(keys: rows(3 ..< 5, tag: 100), values: rows(3 ..< 5)).0
        #expect(flat(rewritten) == [1, 2, 3, 104, 105])

        // A 5-position prompt is longer than an in-place write, so it comes back in order.
        var (keys, values) = cache.update(keys: rows(0 ..< 5), values: rows(0 ..< 5) * 10)
        #expect(flat(keys) == [1, 2, 3, 4, 5])
        #expect(flat(values) == [10, 20, 30, 40, 50])
        // Three positions go in place, wrapping, and the call reads the ring.
        (keys, values) = cache.update(keys: rows(5 ..< 8), values: rows(5 ..< 8) * 10)
        #expect(cache.returnedRing(writing: 3))
        #expect(flat(keys) == [7, 8, 3, 4, 5, 6, 0])
        #expect(flat(values) == [70, 80, 30, 40, 50, 60, 0])
        #expect(cache.offset == 8)
        #expect(cache.maxTrimCount == 2)
        #expect(cache.isTrimmable(after: 2))
        #expect(!cache.isTrimmable(after: 3))

        // Stepping back moves only the offset, and no further than the reserve.
        #expect(cache.trim(2) == 2)
        #expect(cache.offset == 6)
        #expect(cache.maxTrimCount == 0)
        #expect(cache.trim(1) == 0)
        let copy = cache.copy() as! Inkling2RingCache
        (keys, _) = cache.update(keys: rows(6 ..< 7, tag: 100), values: rows(6 ..< 7))
        #expect(flat(keys) == [107, 8, 3, 4, 5, 6, 0])
        #expect(cache.maxTrimCount == 1)
        #expect(flat(copy.state[0]) == [7, 8, 3, 4, 5, 6, 0])

        // A long write returns the last window - 1 positions, then its own.
        (keys, _) = cache.update(keys: rows(7 ..< 11, tag: 200), values: rows(7 ..< 11))
        #expect(flat(keys) == [5, 6, 107, 208, 209, 210, 211])
        (keys, _) = cache.update(keys: rows(11 ..< 12, tag: 300), values: rows(11 ..< 12))
        #expect(flat(keys) == [107, 208, 209, 210, 211, 312, 0])
        #expect(cache.offset == 12)

        let restored = try Inkling2RingCache.restore(
            state: cache.state, metaState: cache.metaState)
        #expect(restored.offset == 12)
        #expect(restored.maxTrimCount == 2)
        #expect(flat(restored.state[0]) == [107, 208, 209, 210, 211, 312, 0])
    }

    @Test("Rewindable convolution states save, restore and copy as themselves")
    func rewindableCacheRoundTrip() throws {
        try onCPU {
            let target = try makeTarget()
            let tokens = prompt(14).text.tokens[.newAxis]
            let cache = try target.newCache(parameters: try speculative(maxTokens: 1))
            _ = target(tokens[0..., ..<12], cache: cache)
            eval(cache)
            let url = FileManager.default.temporaryDirectory.appending(
                path: "inkling2-dspark-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: cache)
            let (restored, _) = try loadPromptCache(url: url)
            let copies = cache.map { $0.copy() }
            for entries in [restored, copies] {
                #expect(entries[2] is Inkling2ConvolutionCache)
                #expect(entries[2].maxTrimCount == cache[2].maxTrimCount)
                #expect(
                    maxDifference(
                        target(tokens[0..., 12...], cache: entries),
                        target(tokens[0..., 12...], cache: cache.map { $0.copy() })) == 0)
            }
        }
    }

    @Test("The draft proposes the same block whether its context came at once or in parts")
    func draftContextInParts() throws {
        try onCPU {
            let target = try makeTarget()
            let draft = try makeDraft()
            MLXRandom.seed(9)
            // More context than the local layers' window of 4.
            let hidden = MLXRandom.normal([1, 10, 4 * 64])
            let bonus = MLXArray([Int32(17)])
            let sampler = ArgMaxSampler()

            var whole = draft.makeState(parameters: nil)
            draft.prepareDrafterState(
                target: target, promptTokens: MLXArray([Int32]()), targetHidden: hidden,
                firstBonus: bonus, positionDeltas: nil, state: &whole, sampler: sampler)
            var parts = draft.makeState(parameters: nil)
            draft.prepareDrafterState(
                target: target, promptTokens: MLXArray([Int32]()),
                targetHidden: hidden[0..., ..<6], firstBonus: bonus, positionDeltas: nil,
                state: &parts, sampler: sampler)
            draft.commitDrafterState(
                target: target, targetHidden: hidden[0..., 6...],
                draftTokens: MLXArray([Int32(1), 2, 3]), acceptedCount: 3,
                finalToken: bonus, positionDeltas: nil, state: &parts, sampler: sampler)
            #expect(whole.nextPosition == 10)
            #expect(parts.nextPosition == 10)

            func block(_ state: inout MTPDrafterState) -> [Int32] {
                draft.draftBlock(
                    target: target, lastToken: bonus, lastHidden: hidden[0..., (-1)...],
                    sharedKV: [:], positionDeltas: nil, queryOffset: 10, blockSize: 4,
                    state: &state, sampler: sampler
                ).asArray(Int32.self)
            }
            let proposals = block(&whole)
            #expect(proposals.count == 3)
            #expect(proposals == block(&parts))
        }
    }

    @Test("The draft loads from the target's directory, reading only mtp.safetensors")
    func loadsFromTargetDirectory() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "inkling2-dspark-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let data = Data(dsparkConfigJSON().utf8)
        try data.write(to: directory.appending(path: "config.json"))
        let saved = try makeDraft(seed: 11)
        let weights = Dictionary(uniqueKeysWithValues: saved.parameters().flattened())
        try save(arrays: weights, url: directory.appending(path: "mtp.safetensors"))
        // The target's tensors, which the draft has no place for.
        try save(
            arrays: [
                "model.layers.0.input_layernorm.weight": MLXArray.ones([64]),
                "lm_head.weight": MLXArray.ones([128, 64]),
            ], url: directory.appending(path: "model.safetensors"))

        await Inkling2DSparkRegistration.register()
        let draft = try #require(
            try await MTPDrafterTypeRegistry.shared.createModel(
                configuration: data, modelType: "inkling2_loop") as? Inkling2DSparkDraftModel)
        try await loadWeights(modelDirectory: directory, model: draft)
        let loaded = Dictionary(uniqueKeysWithValues: draft.parameters().flattened())
        #expect(Set(loaded.keys) == Set(weights.keys))
        #expect(weights.allSatisfy { maxDifference($0.value, loaded[$0.key]!) == 0 })
        #expect(draft.targetLayerIds == [1, 4, 9, 14])
        #expect(draft.maximumBlockSize == 4)
        #expect(draft.isCompatible(with: try makeTarget()))
    }

    @Test("Taps that do not end at the target's final state are refused")
    func refusesTaps() throws {
        #expect(throws: ModelFactoryError.self) {
            try dsparkConfiguration(targetLayers: [1, 4, 9, 13]).validateModelConfiguration()
        }
        #expect(throws: ModelFactoryError.self) {
            try dsparkConfiguration(targetLayers: [4, 1, 14]).validateModelConfiguration()
        }
    }
}

private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}
