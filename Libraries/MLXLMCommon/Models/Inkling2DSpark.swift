//
//  Inkling2DSpark.swift
//  mlx-swift-lm
//

// Port of `mlx_vlm/speculative/drafters/inkling2_dspark/{config,dspark}.py` from
// Blaizzy/mlx-vlm@111ac66807 (Apache-2.0), itself adapted from vLLM's Inkling 2 DSpark
// (WoosukKwon/vllm@d664cd1e96, `vllm/models/inkling2/{dspark,ops/dspark}.py`, Apache-2.0).
//
// Inkling 2 Mini checkpoints bundle this draft model as `mtp.safetensors`. It proposes a block
// in one pass: three small decoder layers over the anchor token and mask embeddings attend to
// keys and values of the target's hidden states, then a recurrent Markov head conditions each
// proposal on the one before it.

import Foundation
import MLX
import MLXNN

/// The draft model as the target's `config.json` describes it: the target configuration, and
/// the draft in `mtp_config`.
public struct Inkling2DSparkConfiguration: Decodable, Sendable {
    public struct Draft: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let intermediateSize: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        /// The layers that attend to only the last `slidingWindow` context positions.
        public let localLayers: [Int]
        public let slidingWindow: Int
        /// The trained block: the anchor token and `blockSize - 1` proposals.
        public let blockSize: Int
        public let markovRank: Int
        public let markovHeadType: String
        /// The target's layer visits whose entering hidden states the draft reads; the last
        /// is the number of visits, the target's final normed state.
        public let targetLayers: [Int]
        public let queryBias: Bool
        public let outputBias: Bool

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case localLayers = "local_layer_ids"
            case slidingWindow = "sliding_window_size"
            case blockSize = "block_size"
            case markovRank = "markov_rank"
            case markovHeadType = "markov_head_type"
            case targetLayers = "target_layer_ids"
            case queryBias = "q_bias"
            case outputBias = "o_bias"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
            attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            headDim = try c.decode(Int.self, forKey: .headDim)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
            ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
            localLayers = try c.decode([Int].self, forKey: .localLayers)
            slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
            blockSize = try c.decode(Int.self, forKey: .blockSize)
            markovRank = try c.decode(Int.self, forKey: .markovRank)
            markovHeadType = try c.decode(String.self, forKey: .markovHeadType)
            targetLayers = try c.decode([Int].self, forKey: .targetLayers)
            queryBias = try c.decodeIfPresent(Bool.self, forKey: .queryBias) ?? false
            outputBias = try c.decodeIfPresent(Bool.self, forKey: .outputBias) ?? false
        }
    }

    public let target: Inkling2Configuration
    public let draft: Draft

    enum CodingKeys: String, CodingKey {
        case draft = "mtp_config"
    }

    public init(from decoder: Decoder) throws {
        target = try Inkling2Configuration(from: decoder)
        draft = try decoder.container(keyedBy: CodingKeys.self).decode(Draft.self, forKey: .draft)
    }
}

extension Inkling2DSparkConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        let text = target.text
        guard draft.markovHeadType == "rnn_hidden", !draft.queryBias, !draft.outputBias,
            draft.hiddenSize == text.hiddenSize, draft.blockSize >= 2, draft.hiddenLayers >= 1,
            draft.markovRank >= 1, draft.attentionHeads > 0, draft.kvHeads > 0,
            draft.attentionHeads.isMultiple(of: draft.kvHeads), draft.headDim.isMultiple(of: 2),
            draft.slidingWindow >= 1,
            draft.localLayers.allSatisfy({ (0 ..< draft.hiddenLayers).contains($0) })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark supports the RNN Markov head without projection biases, at "
                    + "the target's hidden size, with consistent head and layer counts.")
        }
        let taps = draft.targetLayers
        guard taps.count >= 2, taps == Array(Set(taps)).sorted(), taps[0] >= 0,
            taps.last == text.layerVisits.count
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark target_layer_ids must be ordered, unique layer visits ending "
                    + "at the target's final state, \(text.layerVisits.count).")
        }
    }
}

// MARK: - Layers

private final class Inkling2DSparkAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    /// Context positions a local layer sees; `nil` for a global layer.
    let window: Int?

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "k_proj_ctx") var contextKeyProjection: Linear
    @ModuleInfo(key: "v_proj_ctx") var contextValueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "k_norm_ctx") var contextKeyNorm: RMSNorm
    let rope: RoPE

    init(_ args: Inkling2DSparkConfiguration.Draft, local: Bool) {
        heads = args.attentionHeads
        kvHeads = args.kvHeads
        headDim = args.headDim
        scale = pow(Float(args.headDim), -0.5)
        window = local ? args.slidingWindow : nil
        let (dim, q, kv) = (args.hiddenSize, heads * headDim, kvHeads * headDim)
        _queryProjection.wrappedValue = Linear(dim, q, bias: false)
        _keyProjection.wrappedValue = Linear(dim, kv, bias: false)
        _valueProjection.wrappedValue = Linear(dim, kv, bias: false)
        _contextKeyProjection.wrappedValue = Linear(dim, kv, bias: false)
        _contextValueProjection.wrappedValue = Linear(dim, kv, bias: false)
        _outputProjection.wrappedValue = Linear(q, dim, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _contextKeyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta)
    }

    private func split(_ x: MLXArray, heads: Int) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), heads, headDim).transposed(0, 2, 1, 3)
    }

    /// Stores the keys and values of target context `[B, n, hidden]` whose first row is at
    /// `position`. A local layer projects only the rows its window can still see.
    func appendContext(_ context: MLXArray, position: Int, cache: KVCache) {
        var (context, position) = (context, position)
        if let window, context.dim(1) > window {
            let skip = context.dim(1) - window
            context = context[0..., skip...]
            position += skip
        }
        let keys = contextKeyNorm(split(contextKeyProjection(context), heads: kvHeads))
        let values = split(contextValueProjection(context), heads: kvHeads)
        _ = cache.update(keys: rope(keys, offset: position), values: values)
    }

    /// The block `[B, T, hidden]` whose first row is at `position`. Every query sees the
    /// stored context and the whole block, so there is no mask; proposals never enter the
    /// cache.
    func callAsFunction(_ x: MLXArray, position: Int, cache: KVCache) -> MLXArray {
        var keys = keyNorm(split(keyProjection(x), heads: kvHeads))
        var values = split(valueProjection(x), heads: kvHeads)
        keys = rope(keys, offset: position)
        if var (contextKeys, contextValues) = cache.currentKeyValues() {
            if let window, contextKeys.dim(2) > window {
                contextKeys = contextKeys[.ellipsis, (-window)..., 0...]
                contextValues = contextValues[.ellipsis, (-window)..., 0...]
            }
            keys = concatenated([contextKeys, keys], axis: 2)
            values = concatenated([contextValues, values], axis: 2)
        }
        let queries = rope(queryNorm(split(queryProjection(x), heads: heads)), offset: position)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none)
        return outputProjection(output.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), -1))
    }
}

private final class Inkling2DSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ args: Inkling2DSparkConfiguration.Draft) {
        _gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        _up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        _down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let product = silu(gate(x).asType(.float32)) * up(x).asType(.float32)
        return down(product.asType(x.dtype))
    }
}

private final class Inkling2DSparkLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    /// Normalizes this layer's mix of the target taps.
    @ModuleInfo(key: "dflare_norm") var contextNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: Inkling2DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2DSparkMLP

    init(_ args: Inkling2DSparkConfiguration.Draft, index: Int) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _contextNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _attention.wrappedValue = Inkling2DSparkAttention(
            args, local: args.localLayers.contains(index))
        _mlp.wrappedValue = Inkling2DSparkMLP(args)
    }

    /// Returns the output and the residual. The sum is normed unrounded but kept rounded, as
    /// the reference's fused add and RMS norm do.
    func callAsFunction(
        _ x: MLXArray, residual: MLXArray?, position: Int, cache: KVCache
    ) -> (MLXArray, MLXArray) {
        var summed = residual.map { x.asType(.float32) + $0.asType(.float32) } ?? x
        let residual = summed.asType(x.dtype)
        let h = attention(
            inputNorm(summed).asType(x.dtype), position: position, cache: cache)
        summed = h.asType(.float32) + residual.asType(.float32)
        return (mlp(postAttentionNorm(summed).asType(x.dtype)), summed.asType(x.dtype))
    }
}

/// Proposes each token from the block's hidden state, the previous token and a recurrent
/// state. Its output projection is folded into the target's vocabulary head.
private final class Inkling2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var previousTokenEmbedding: Embedding
    @ModuleInfo(key: "rnn_proj") var recurrentProjection: Linear
    @ModuleInfo(key: "hidden_out") var hiddenOutput: Linear
    @ModuleInfo(key: "dflare_norm") var contextNorm: RMSNorm

    let rank: Int

    init(_ args: Inkling2DSparkConfiguration.Draft, vocabularySize: Int) {
        rank = args.markovRank
        _previousTokenEmbedding.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: rank)
        _recurrentProjection.wrappedValue = Linear(
            2 * rank + 2 * args.hiddenSize, 3 * rank, bias: true)
        _hiddenOutput.wrappedValue = Linear(rank, args.hiddenSize, bias: true)
        _contextNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    /// The part of each step's recurrent projection that does not depend on the recurrence,
    /// from the block's hidden states `[B, T, hidden]` and the mixed last context row
    /// `[B, 1, hidden]`. Returns `[B, T, 3 * rank]` float32.
    func context(_ hidden: MLXArray, last: MLXArray) -> MLXArray {
        let inputs = concatenated([hidden, broadcast(last, to: hidden.shape)], axis: -1)
        let weight = recurrentProjection.weight[0..., (2 * rank)...].asType(.float32)
        return matmul(inputs.asType(.float32), weight.T)
            + recurrentProjection.bias!.asType(.float32)
    }

    /// One step: the next state and the vocabulary bias for the next proposal.
    func step(
        state: MLXArray, previous: MLXArray, context: MLXArray, fold: (MLXArray, MLXArray)
    ) -> (MLXArray, MLXArray) {
        let dtype = state.dtype
        let inputs = concatenated([state, previousTokenEmbedding(previous)], axis: -1)
        let weight = recurrentProjection.weight[0..., ..<(2 * rank)].asType(.float32)
        let raw = (matmul(inputs.asType(.float32), weight.T) + context).asType(dtype)
        let parts = raw.split(parts: 3, axis: -1)
        let gate = sigmoid(parts[0].asType(.float32)).asType(dtype)
        let candidate = tanh(parts[1].asType(.float32)).asType(dtype)
        let next = gate * state + (1 - gate) * candidate
        let latent = tanh(parts[2].asType(.float32)).asType(dtype)
        return (next, addMM(fold.1, latent, fold.0.T))
    }
}

// MARK: - Model

/// The DSpark draft model that Inkling 2 Mini checkpoints bundle as `mtp.safetensors`.
///
/// It loads from the target's own directory: the target's `config.json` describes it in
/// `mtp_config`, and only `mtp.safetensors` is read. Token embeddings and the vocabulary head
/// are the target's. Generation needs the target's cache to keep a rewind window of at least
/// the block size (``KVCacheConfiguration/Rewind``), so the target's convolutions can step
/// back over rejected proposals.
public final class Inkling2DSparkDraftModel: Module, StatefulMTPDrafterModel,
    AdditionalWeightFilesProviding
{
    public let configuration: Inkling2DSparkConfiguration

    @ModuleInfo(key: "layers") fileprivate var layers: [Inkling2DSparkLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") fileprivate var markovHead: Inkling2DSparkMarkovHead
    /// Per draft layer and then the Markov head, the weights of the target taps.
    @ParameterInfo(key: "fuse_weight") var fuseWeight: MLXArray
    @ParameterInfo(key: "mask_embed") var maskEmbedding: MLXArray

    private let fold = Inkling2DSparkFold()

    public init(_ configuration: Inkling2DSparkConfiguration) {
        self.configuration = configuration
        let args = configuration.draft
        _layers.wrappedValue = (0 ..< args.hiddenLayers).map {
            Inkling2DSparkLayer(args, index: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _markovHead.wrappedValue = Inkling2DSparkMarkovHead(
            args, vocabularySize: configuration.target.text.vocabularySize)
        _fuseWeight.wrappedValue = MLXArray.zeros([args.hiddenLayers + 1, args.targetLayers.count])
        _maskEmbedding.wrappedValue = MLXArray.zeros([args.hiddenSize])
        super.init()
    }

    public var maximumBlockSize: Int? { configuration.draft.blockSize }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetLayerIds: [Int]? { configuration.draft.targetLayers }

    public var additionalWeightFiles: [String] { ["mtp.safetensors"] }

    /// The target's own weights, which share the directory, are never read.
    public var excludedCheckpointNamespaces: [String] {
        ["model", "language_model", "vision_tower", "lm_head"]
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(
            cache: layers.map { layer -> KVCache in
                layer.attention.window.map { RotatingKVCache(maxSize: $0, keep: 0) }
                    ?? KVCacheSimple()
            })
    }

    public func prepareDrafterState(
        target: any LanguageModel, promptTokens _: MLXArray, targetHidden: MLXArray,
        firstBonus _: MLXArray, positionDeltas _: MLXArray?, state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        _ = textModel(target)
        appendContext(targetHidden, state: &state)
    }

    public func commitDrafterState(
        target: any LanguageModel, targetHidden: MLXArray, draftTokens _: MLXArray,
        acceptedCount: Int, finalToken _: MLXArray, positionDeltas _: MLXArray?,
        state: inout MTPDrafterState, sampler _: any LogitSampler
    ) {
        _ = textModel(target)
        // The verified tokens before the final one join the context: the anchor and the
        // accepted proposals.
        let committed = acceptedCount + 1
        precondition(
            targetHidden.dim(1) >= committed,
            "Inkling 2 did not emit a hidden state for every committed position")
        appendContext(targetHidden[0..., ..<committed, 0...], state: &state)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        appendContext(lastHidden, state: &state)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        let text = textModel(target)
        let args = configuration.draft
        let count = blockSize - 1
        precondition(
            (1 ..< args.blockSize).contains(count),
            "Inkling 2 DSpark proposes 1 to \(args.blockSize - 1) tokens")
        guard let last = state.seedHidden else {
            preconditionFailure("Inkling 2 DSpark drafts after its context is prepared")
        }

        let anchor = normalizedMTPColumn(lastToken).reshaped(-1)
        let embedding = text.embed(expandedDimensions(anchor, axis: 1))
        let B = anchor.dim(0)
        // Bidirectional attention needs the whole trained block, also when fewer proposals
        // are asked for.
        let masks = broadcast(
            maskEmbedding.asType(embedding.dtype), to: [B, args.blockSize - 1, args.hiddenSize])
        var x = concatenated([embedding, masks], axis: 1)
        var residual: MLXArray?
        for (layer, cache) in zip(layers, state.cache) {
            (x, residual) = layer(x, residual: residual, position: state.nextPosition, cache: cache)
        }
        x =
            norm(x.asType(.float32) + residual!.asType(.float32)).asType(embedding.dtype)[
                0..., 1 ..< (count + 1)]

        // The draft's logits stay in its dtype, unlike the target's float32 projection.
        let head = (target as! any Inkling2LanguageModel).inkling2Head
        let base = head(x).asType(x.dtype)
        let folded = fold.weights(head: head, projection: markovHead.hiddenOutput)
        let context = markovHead.context(
            x, last: mix(last, row: args.hiddenLayers, norm: markovHead.contextNorm))
        var recurrent = MLXArray.zeros([B, args.markovRank], dtype: x.dtype)
        var previous = anchor
        var proposals = [MLXArray]()
        let cap = configuration.target.text.finalLogitSoftcap
        let vocabulary = configuration.target.text.unpaddedVocabularySize
        for i in 0 ..< count {
            let (next, bias) = markovHead.step(
                state: recurrent, previous: previous, context: context[0..., i], fold: folded)
            recurrent = next
            var logits = base[0..., i] + bias
            logits = (cap * tanh(logits.asType(.float32) / cap)).asType(x.dtype)
            previous = sampler.sample(logits: logits[0..., ..<vocabulary]).reshaped(-1)
            proposals.append(expandedDimensions(previous, axis: 1))
        }
        return concatenated(proposals, axis: 1).asType(.int32)
    }

    /// Adds target taps `[B, n, taps * hidden]` to the context: each layer stores its mix's
    /// keys and values, and the last row is kept for the Markov head.
    private func appendContext(_ hidden: MLXArray, state: inout MTPDrafterState) {
        precondition(
            hidden.dim(-1) == configuration.draft.targetLayers.count
                * configuration.draft.hiddenSize,
            "Inkling 2 DSpark reads \(configuration.draft.targetLayers.count) target taps")
        for (row, (layer, cache)) in zip(layers, state.cache).enumerated() {
            layer.attention.appendContext(
                mix(hidden, row: row, norm: layer.contextNorm), position: state.nextPosition,
                cache: cache)
        }
        state.nextPosition += hidden.dim(1)
        state.seedHidden = hidden[0..., (-1)..., 0...]
    }

    /// The taps weighted by a softmax over `row` of the fuse weights, then normed.
    private func mix(_ hidden: MLXArray, row: Int, norm: RMSNorm) -> MLXArray {
        let taps = configuration.draft.targetLayers.count
        let features = hidden.reshaped(hidden.dim(0), hidden.dim(1), taps, -1).asType(.float32)
        let alpha = softmax(fuseWeight.asType(.float32), axis: -1).asType(hidden.dtype)[row]
        var mixed = features[0..., 0..., 0] * alpha[0].asType(.float32)
        for i in 1 ..< taps {
            mixed = mixed + features[0..., 0..., i] * alpha[i].asType(.float32)
        }
        return norm(mixed).asType(hidden.dtype)
    }

    private func textModel(_ target: any LanguageModel) -> Inkling2TextModel {
        guard let target = target as? any Inkling2LanguageModel, isCompatible(with: target)
        else {
            preconditionFailure("Inkling 2 DSpark drafts for its own Inkling 2 target only")
        }
        return target.inkling2TextModel
    }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        guard let target = target as? any Inkling2LanguageModel else { return false }
        let ours = configuration.target.text
        let theirs = target.inkling2TextModel.configuration
        return theirs.hiddenSize == ours.hiddenSize
            && theirs.vocabularySize == ours.vocabularySize
            && theirs.hiddenLayers == ours.hiddenLayers
            && theirs.layerVisits == ours.layerVisits
    }

    public func validateCompatibility(with target: any LanguageModel) throws {
        guard isCompatible(with: target) else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Inkling2DSparkDraftModel",
                expected: "the Inkling 2 model whose checkpoint bundles it",
                actual: String(describing: type(of: target)))
        }
    }
}

/// The Markov head's output projection folded into a target's vocabulary head, made once per
/// target head.
private final class Inkling2DSparkFold: @unchecked Sendable {
    private let lock = NSLock()
    private weak var head: Linear?
    private var weights: (MLXArray, MLXArray)?

    /// `head · projection` `[vocabulary, rank]` and `head · projection.bias` `[vocabulary]`,
    /// in the projection's dtype. The reference folds in float64; this folds in float32 on
    /// the CPU, where matrix products do not round their inputs as the GPU's float32 ones do.
    func weights(head: Linear, projection: Linear) -> (MLXArray, MLXArray) {
        lock.withLock {
            if let weights, self.head === head { return weights }
            let dtype = projection.weight.dtype
            let folded = Device.withDefaultDevice(.cpu) { () -> (MLXArray, MLXArray) in
                let weight = projection.weight.asType(.float32)
                let bias = projection.bias!.asType(.float32)
                var weights = [MLXArray]()
                var biases = [MLXArray]()
                // Tiles bound the dequantized head's memory.
                let rows = head.weight.dim(0)
                for start in stride(from: 0, to: rows, by: 4096) {
                    let tile = Self.rows(of: head, start ..< Swift.min(rows, start + 4096))
                    let w = matmul(tile, weight).asType(dtype)
                    let b = matmul(tile, bias).asType(dtype)
                    eval(w, b)
                    weights.append(w)
                    biases.append(b)
                }
                return (concatenated(weights, axis: 0), concatenated(biases, axis: 0))
            }
            eval(folded.0, folded.1)
            self.head = head
            weights = folded
            return folded
        }
    }

    /// Rows of a head's weight in float32, dequantized if need be.
    private static func rows(of head: Linear, _ range: Range<Int>) -> MLXArray {
        guard let head = head as? QuantizedLinear else {
            return head.weight[range].asType(.float32)
        }
        return dequantized(
            head.weight[range], scales: head.scales[range], biases: head.biases?[range],
            groupSize: head.groupSize, bits: head.bits, mode: head.mode,
            globalScale: head.globalScale
        ).asType(.float32)
    }
}

/// Registers the draft model Inkling 2 checkpoints bundle, for loading with
/// ``MTPDrafterModelFactory`` from the target's directory.
public enum Inkling2DSparkRegistration {
    public static func register() async {
        for modelType in ["inkling2_loop", "inkling2"] {
            await MTPDrafterTypeRegistry.shared.registerModelType(
                modelType,
                matches: { data in
                    let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                    return object?["mtp_config"] is [String: Any]
                },
                creator: { data in
                    let configuration = try JSONDecoder.json5().decode(
                        Inkling2DSparkConfiguration.self, from: data)
                    try configuration.validateModelConfiguration()
                    return Inkling2DSparkDraftModel(configuration)
                })
        }
    }
}
