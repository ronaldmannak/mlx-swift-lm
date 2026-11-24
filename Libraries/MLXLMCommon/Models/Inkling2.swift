//
//  Inkling2.swift
//  mlx-swift-lm
//

// Port of `mlx_lm/models/inkling2/{layers,model}.py` from jundot/mlx-lm@0c4c9919 (MIT), the
// draft mlx-lm support for Thinking Machines' Inkling 2 (ml-explore/mlx-lm#2061, #2086).
//
// Inkling 2 has no rotary embedding. Attention adds a learned relative-position bias and a
// per-token sink, keys and values pass through short causal convolutions, and each layer's
// input is an attention over the outputs of earlier blocks. Inkling 2 Mini runs its middle
// layers several times per token; each run of a layer has its own cache.
//
// The text model lives here so that MLXLLM's and MLXVLM's wrappers share it.

import Foundation
import MLX
import MLXNN

private let routerScoreEpsilon: Float = 1e-20

/// `rms_norm` without a weight on `x` in float32, as `mx.fast.rms_norm(x, None, eps)`,
/// returned as `dtype`.
func inkling2UnitRMS(_ x: MLXArray, eps: Float = 1e-6, dtype: DType = .float32) -> MLXArray {
    MLXFast.rmsNorm(x.asType(.float32), weight: nil, eps: eps).asType(dtype)
}

/// `norm(x).asType(dtype)`.
func inkling2Normed(_ norm: RMSNorm, _ x: MLXArray, dtype: DType) -> MLXArray {
    if inkling2UsesKernels(), x.dtype == .float32 {
        // MLX's float32 norm multiplies by the weight last, so applying it after the
        // unweighted norm gives the same bits, and the casts fuse into a compiled multiply.
        return (MLXFast.rmsNorm(x, weight: nil, eps: norm.eps) * norm.weight.asType(.float32))
            .asType(dtype)
    }
    return norm(x).asType(dtype)
}

func inkling2L2Clip(
    _ x: MLXArray, scale: MLXArray?, threshold: Float, useKernel: Bool = true
) -> MLXArray {
    if useKernel, inkling2L2ClipKernelTakes(x) {
        return inkling2L2ClipKernel(x, scale: scale, threshold: threshold)
    }
    var states = x.asType(.float32)
    if let scale {
        states = states * scale
    }
    let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
    return (states * MLX.minimum(threshold / (norm + 1e-6), MLXArray(Float(1))))
        .asType(x.dtype)
}

/// The gate is capped before the product but not inside its sigmoid, as in the reference.
private func softcappedSwiGLU(
    gate: MLXArray, up: MLXArray, gateCap: Float, upCap: Float
) -> MLXArray {
    let gate32 = gate.asType(.float32)
    let up32 = up.asType(.float32)
    let gateCapped = tanh(gate32 / gateCap) * gateCap
    let upCapped = tanh(up32 / upCap) * upCap
    return (gateCapped * sigmoid(gate32) * upCapped).asType(gate.dtype)
}

private func usesGPU() -> Bool {
    Device.defaultDevice().deviceType == .gpu
}

// MARK: - Small modules

/// A learned per-head scale followed by an L2 clip.
final class Inkling2ScaledClip: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray
    let threshold: Float

    init(heads: Int, dimensions: Int, threshold: Float) {
        self.threshold = threshold
        _scale.wrappedValue = MLXArray.ones([heads, dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        inkling2L2Clip(x, scale: scale, threshold: threshold, useKernel: !training)
    }
}

/// Projects each query's relative state onto a bias profile over key distances.
final class Inkling2RelativeLogits: Module {
    @ParameterInfo(key: "proj") var projection: MLXArray

    init(relativeDim: Int, extent: Int) {
        _projection.wrappedValue = MLXArray.zeros([relativeDim, extent])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, projection.asType(x.dtype))
    }
}

/// The convolution input: the previous `kernelSize - 1` positions, then `x` with masked
/// (padding) positions zeroed, in float32.
private func convolutionWindow(
    _ x: MLXArray, state: MLXArray?, mask: MLXArray?, kernelSize: Int
) -> MLXArray {
    var x = x.asType(.float32)
    if let mask {
        x = which(mask[.ellipsis, .newAxis], x, MLXArray(Float(0)))
    }
    let state =
        state ?? MLXArray.zeros([x.dim(0), kernelSize - 1, x.dim(-1)], dtype: .float32)
    return concatenated([state, x], axis: 1)
}

/// The last `count` valid positions of each row of `window`, which become the next state.
private func convolutionState(_ window: MLXArray, lengths: MLXArray?, count: Int) -> MLXArray {
    let positions: MLXArray
    if let lengths {
        let ends = clip(lengths, min: 0, max: window.dim(1) - count)
        positions = expandedDimensions(
            expandedDimensions(ends, axis: 1) + MLXArray(Int32(0) ..< Int32(count)), axis: -1)
    } else {
        positions = MLXArray(Int32(window.dim(1) - count) ..< Int32(window.dim(1)))
            .reshaped(1, count, 1)
    }
    // A gather copies, so the state does not keep the whole input alive.
    return takeAlong(window, positions, axis: 1)
}

/// A depthwise causal convolution with a residual, scaled by `1/sqrt(K + 1)`.
final class Inkling2ShortConvolution: Module {
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    let kernelSize: Int

    init(dimensions: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        _conv1d.wrappedValue = Conv1d(
            inputChannels: dimensions, outputChannels: dimensions, kernelSize: kernelSize,
            groups: dimensions, bias: false)
    }

    /// Returns the output and the next state: each row's last `kernelSize - 1` valid inputs.
    /// `lengths` are the rows' valid lengths of a right-padded batch prompt. With
    /// `keepWindow`, the second value is the whole window instead: the previous state, then
    /// the inputs, in float32.
    func callAsFunction(
        _ x: MLXArray, state: MLXArray?, mask: MLXArray?, lengths: MLXArray? = nil,
        forceReference: Bool = false, keepWindow: Bool = false
    ) -> (MLXArray, MLXArray) {
        if !forceReference, !training, mask == nil, lengths == nil, inkling2UsesKernels() {
            let (out, next) = inkling2ShortConvolutionKernel(x, state: state, weight: conv1d.weight)
            guard keepWindow else { return (out, next) }
            return (out, convolutionWindow(x, state: state, mask: nil, kernelSize: kernelSize))
        }
        let window = convolutionWindow(x, state: state, mask: mask, kernelSize: kernelSize)
        // As the reference, the convolution runs in the weight's dtype.
        var out = conv1d(window.asType(conv1d.weight.dtype)).asType(.float32)
        out = (out + x.asType(.float32)) / sqrt(Float(kernelSize + 1))
        return (
            out.asType(x.dtype),
            keepWindow ? window : convolutionState(window, lengths: lengths, count: kernelSize - 1)
        )
    }

    /// Keeps this process' channels; `groups` is fixed when a convolution is made.
    func shard(rank: Int, size: Int) {
        let channels = conv1d.weight.dim(0) / size
        let conv = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            groups: channels, bias: false)
        conv.update(
            parameters: ModuleParameters.unflattened([
                "weight": contiguous(conv1d.weight.split(parts: size, axis: 0)[rank])
            ]))
        update(modules: ModuleChildren(values: ["conv1d": .value(conv)]))
    }
}

/// A causal convolution whose per-head taps add a projection of `dynamicInput` to static
/// taps, scaled by `1/sqrt(2K + 1)`.
final class Inkling2DynamicConvolution: Module {
    @ParameterInfo(key: "static_weight") var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") var weightProjection: Linear
    let kernelSize: Int
    var heads: Int

    init(dimensions: Int, inputDimensions: Int, heads: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self.heads = heads
        _staticWeight.wrappedValue = MLXArray.zeros([kernelSize, dimensions])
        _weightProjection.wrappedValue = Linear(inputDimensions, heads * kernelSize, bias: false)
    }

    /// Returns the output and the next state or the window, as ``Inkling2ShortConvolution``
    /// does.
    func callAsFunction(
        _ x: MLXArray, dynamicInput: MLXArray, state: MLXArray?, mask: MLXArray?,
        lengths: MLXArray?, keepWindow: Bool = false
    ) -> (MLXArray, MLXArray) {
        let (B, L, C) = (x.dim(0), x.dim(1), x.dim(2))
        let K = kernelSize
        let dynamicWeights = weightProjection(dynamicInput).asType(.float32)
            .reshaped(B, L, heads, 1, K)
        let staticWeights = staticWeight.asType(.float32).T.reshaped(heads, -1, K)
        let weights = staticWeights + dynamicWeights

        let window = convolutionWindow(x, state: state, mask: mask, kernelSize: K)
        let windows = stacked(
            (0 ..< K).map { lag in window[0..., (K - 1 - lag) ..< (K - 1 - lag + L)] },
            axis: -1
        ).reshaped(B, L, heads, -1, K)
        var out = (windows * weights).sum(axis: -1).reshaped(B, L, C)
        out = (out + x.asType(.float32)) / sqrt(Float(2 * K + 1))
        return (
            out.asType(x.dtype),
            keepWindow ? window : convolutionState(window, lengths: lengths, count: K - 1)
        )
    }
}

/// Attention of learned queries over the depth sources of each token: the completed
/// blocks and the current partial block, all unit-RMS float32 `[B, L, D]`.
/// Returns `[B, L, Q, D]`. The kernel has no gradient, so training passes
/// `useKernel: false`.
func inkling2DepthAttention(
    blocks: MLXArray?, partial: MLXArray, query: MLXArray, useKernel: Bool = true
) -> MLXArray {
    let query = query.asType(.float32).reshaped(-1, partial.dim(-1))
    let scale = pow(Float(partial.dim(-1)), -0.5)
    if useKernel, inkling2UsesKernels(), let blocks, blocks.dim(0) > 0 {
        return inkling2DepthAttentionKernel(
            blocks: blocks, partial: partial, query: query, scale: scale)
    }
    var sources = partial[.newAxis]
    if let blocks, blocks.dim(0) > 0 {
        sources = concatenated([blocks, sources], axis: 0)
    }
    sources = sources.movedAxis(source: 0, destination: -2)
    let weights = softmax(matmul(sources, query.T) * scale, axis: -2, precise: true)
    return matmul(weights.swappedAxes(-1, -2), sources)
}

/// Learned depth-attention queries (an attention residual).
final class Inkling2DepthQuery: Module {
    @ParameterInfo(key: "query") var query: MLXArray

    init(dimensions: Int, queries: Int = 1) {
        _query.wrappedValue =
            queries == 1
            ? MLXArray.zeros([dimensions], dtype: .float32)
            : MLXArray.zeros([queries, dimensions], dtype: .float32)
    }

    func callAsFunction(blocks: MLXArray?, partial: MLXArray) -> MLXArray {
        inkling2DepthAttention(
            blocks: blocks, partial: partial, query: query, useKernel: !training)
    }
}

/// Mixes a message from the previous loop pass into a layer input.
final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var downProjection: Linear
    @ModuleInfo(key: "forget_gate_proj") var forgetGate: Linear
    @ModuleInfo(key: "message_gate_proj") var messageGate: Linear

    init(dimensions: Int, rank: Int) {
        _downProjection.wrappedValue = Linear(2 * dimensions, rank, bias: false)
        _forgetGate.wrappedValue = Linear(rank, dimensions, bias: false)
        _messageGate.wrappedValue = Linear(rank, dimensions, bias: false)
    }

    /// Mixes row `gate` of `messages` `[B, L, 4, D]` into `x`; both are float32.
    fileprivate func callAsFunction(
        _ x: MLXArray, messages: Inkling2Messages, gate: Int
    ) -> MLXArray {
        let message = messages.raw[.ellipsis, gate, 0...]
        var states = concatenated(
            [inkling2UnitRMS(x), messages.normed[.ellipsis, gate, 0...]],
            axis: -1)
        let dtype =
            (downProjection as? QuantizedLinear)?.scales.dtype ?? downProjection.weight.dtype
        states = downProjection(states.asType(dtype)).asType(.float32)
        let forget = tanh(forgetGate(states))
        let gain = tanh(messageGate(states))
        return (1 - forget) * x + gain * message
    }
}

/// The message queries of every looped layer: query, key, value and MLP.
final class Inkling2MessageInput: Module {
    @ParameterInfo(key: "query") var query: MLXArray

    init(dimensions: Int, loopedLayers: Int) {
        _query.wrappedValue = MLXArray.zeros([loopedLayers, 4, dimensions], dtype: .float32)
    }

    func callAsFunction(_ bank: Inkling2MessageBank, loopedLayer: Int) -> MLXArray {
        // A slice, not an index: an index gathers.
        inkling2DepthAttention(
            blocks: bank.blocks, partial: bank.partial,
            query: query[loopedLayer ..< loopedLayer + 1].squeezed(axis: 0), useKernel: !training)
    }
}

/// What a loop pass hands the next: the blocks it completed and its unit-RMS output.
struct Inkling2MessageBank {
    let blocks: MLXArray?
    let partial: MLXArray
}

// MARK: - Experts

/// An expert projection whose NVFP4 form carries a per-expert global scale.
final class Inkling2SwitchLinear: SwitchLinear {
    override public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode)
        -> Module
    {
        guard mode == .nvfp4 else {
            return super.toQuantized(groupSize: groupSize, bits: bits, mode: mode)
        }
        return Inkling2ScaledSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// NVFP4 experts with a global scale per expert, in MLX's convention: the output is
/// multiplied by `global_scale / (448 * 6)`. Applied after the matmul, so it also runs on
/// the CPU.
final class Inkling2ScaledSwitchLinear: QuantizedSwitchLinear {
    static let unitScale: Float = 448 * 6

    @ParameterInfo(key: "global_scale") var globalScale: MLXArray

    override init(
        _ other: SwitchLinear, groupSize: Int = 16, bits: Int = 4, mode: QuantizationMode = .nvfp4
    ) {
        _globalScale.wrappedValue =
            MLXArray.ones([other.numExperts], dtype: .float32) * Self.unitScale
        super.init(other, groupSize: groupSize, bits: bits, mode: mode)
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let y = super.callAsFunction(x, indices, sortedIndices: sortedIndices)
        let scale = (globalScale[indices] / Self.unitScale).asType(y.dtype)
        return y * scale.reshaped(scale.shape + [1, 1])
    }
}

/// Routed experts. `gate_up_proj` keeps the checkpoint's rows, which alternate between
/// gate and up.
final class Inkling2Experts: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProjection: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProjection: SwitchLinear
    let gateCap: Float
    let upCap: Float

    init(_ args: Inkling2TextConfiguration) {
        _gateUpProjection.wrappedValue = Inkling2SwitchLinear(
            inputDims: args.hiddenSize, outputDims: 2 * args.intermediateSize,
            numExperts: args.routedExperts, bias: false)
        _downProjection.wrappedValue = Inkling2SwitchLinear(
            inputDims: args.intermediateSize, outputDims: args.hiddenSize,
            numExperts: args.routedExperts, bias: false)
        gateCap = args.gateSoftcap
        upCap = args.upSoftcap
    }

    /// `[..., k, D]` expert outputs for `x` `[..., D]` and `indices` `[..., k]`.
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var x = expandedDimensions(x, axes: [-2, -3])
        let sort = shouldSortExpertRoutes(
            routeCount: indices.size, quantized: gateUpProjection is QuantizedSwitchLinear)
        var routes = indices
        var inverseOrder = MLXArray()
        if sort {
            (x, routes, inverseOrder) = gatherSort(x: x, indices: indices)
        }
        let gateUp = gateUpProjection(x, routes, sortedIndices: sort)
        let activated = softcappedSwiGLU(
            gate: gateUp[.ellipsis, .stride(from: 0, by: 2)],
            up: gateUp[.ellipsis, .stride(from: 1, by: 2)],
            gateCap: gateCap, upCap: upCap)
        var y = downProjection(activated, routes, sortedIndices: sort)
        if sort {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }
        return y.squeezed(axis: -2)
    }
}

/// Sigmoid router: top-k by score plus correction bias, weights from the unbiased scores,
/// renormalized and scaled.
final class Inkling2Router: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray
    let topK: Int
    let routeScale: Float

    init(_ args: Inkling2TextConfiguration) {
        topK = args.expertsPerToken
        routeScale = args.routeScale
        _weight.wrappedValue = MLXArray.zeros(
            [args.routedExperts, args.hiddenSize], dtype: .float32)
        _correctionBias.wrappedValue = MLXArray.zeros([args.routedExperts], dtype: .float32)
    }

    func callAsFunction(
        _ x: MLXArray, forceReference: Bool = false
    ) -> (indices: MLXArray, weights: MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        if !forceReference, !training, usesGPU(), supportsFusedRouterTopK(logits, k: topK) {
            if let routed = inkling2SIMDRouterKernel(
                logits: logits, bias: correctionBias.asType(.float32), k: topK,
                routeScale: routeScale)
            {
                return routed
            }
            return inkling2RouterKernel(
                logits: logits, bias: correctionBias.asType(.float32), k: topK,
                routeScale: routeScale)
        }
        let scores = sigmoid(logits)
        let biased = scores + correctionBias.asType(.float32)
        if !training, usesGPU(), supportsFusedRouterTopK(biased, k: topK) {
            let (indices, weights) = fusedRouterTopK(
                selection: biased, values: scores, k: topK, normalize: true,
                order: .descending)
            return (indices, weights * routeScale)
        }
        // The choice of experts has no gradient; the weights of the chosen ones do.
        let indices = stopGradient(
            argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + routerScoreEpsilon)
        return (indices, weights * routeScale)
    }
}

final class Inkling2SparseMoE: Module {
    @ModuleInfo(key: "gate") var router: Inkling2Router
    @ModuleInfo(key: "experts") var experts: Inkling2Experts

    /// The group the experts are split across, once the block holds a shard.
    var shardingGroup: MLXDistributed.Group?

    init(_ args: Inkling2TextConfiguration) {
        _router.wrappedValue = Inkling2Router(args)
        _experts.wrappedValue = Inkling2Experts(args)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = router(x)
        let outputs = experts(x, indices: indices)
        guard let shardingGroup else {
            return combine(outputs, weights: weights, dtype: x.dtype)
        }
        let y = combine(outputs, weights: weights, dtype: .float32)
        return MLXDistributed.allSum(y, group: shardingGroup).asType(x.dtype)
    }

    /// The experts' outputs `[..., k, D]` weighted and summed in float32, as `dtype`.
    func combine(
        _ outputs: MLXArray, weights: MLXArray, dtype: DType, forceReference: Bool = false
    ) -> MLXArray {
        if !forceReference, !training, usesGPU() {
            return inkling2ExpertCombineKernel(outputs, weights: weights, dtype: dtype)
        }
        return weightedExpertSum(outputs.asType(.float32), weights).asType(dtype)
    }
}

// MARK: - Attention

/// The caches of one layer visit. `positions` stores each key's position plus one, written
/// through a cache of the same kind as `keyValues`, so it follows every trim, wrap, merge
/// and roll of the keys; a zero marks a slot that holds no key.
///
/// The four convolution states, of the key, value, post-attention and post-MLP
/// convolutions, are the slots of one `ArraysCache`. When generation asks for a rewind window
/// (``KVCacheConfiguration/Rewind``), each convolution has an ``Inkling2ConvolutionCache``
/// instead, which also keeps its last call's inputs, so a speculative round can step back
/// over the positions it rejects.
struct Inkling2VisitCache {
    let keyValues: KVCache
    let positions: KVCache
    let convolutions: [ArraysCache]

    /// Whether the convolutions keep their last call's inputs. Their calls then return
    /// windows.
    var rewinds: Bool { convolutions.count > 1 }

    /// Each row's valid length during a ragged prefill.
    var lengths: MLXArray? { convolutions[0].currentLengths }

    func makeMask(N: Int) -> MLXArray? { convolutions[0].makeMask(N: N) }

    /// The state of convolution `i`.
    func state(_ i: Int) -> MLXArray? { rewinds ? convolutions[i][0] : convolutions[0][i] }

    /// Stores what the four convolutions return for `length` new positions: their next
    /// states, or, with ``rewinds``, their windows.
    func commit(_ results: [MLXArray], length: Int) {
        guard rewinds else {
            for (i, state) in results.enumerated() {
                convolutions[0][i] = state
            }
            convolutions[0].advance(length)
            return
        }
        precondition(lengths == nil, "A rewindable Inkling 2 cache holds one unpadded row")
        for (cache, window) in zip(convolutions, results) {
            (cache as! Inkling2ConvolutionCache).record(window: window, length: length)
        }
    }
}

/// One convolution's state and its last call's inputs, for a cache with a rewind window.
///
/// A staged speculative round writes through an auxiliary entry and trims it back to the
/// accepted positions, so this cache joins rounds whose sliding layers stage their keys.
/// (LFM2 rewinds its `RewindableConvolutionCache`s in place instead, which a wrapped ring
/// cannot.)
///
/// A round's trim steps back within its one verify call, so, as in mlx-vlm, the cache keeps
/// only that call's window, and the state and a trim are views of it. The base class's rolling
/// history copies on every call and trim instead: Inkling 2 Mini's 192 convolutions record
/// every step, and those copies cost about 3 ms of each.
final class Inkling2ConvolutionCache: RewindableConvolutionCache, AuxiliaryKVCache {
    override class var serializedClassName: String? { "Inkling2ConvolutionCache" }

    override class func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        let restored = try RewindableConvolutionCache.restore(state: state, metaState: metaState)
        return (restored as! RewindableConvolutionCache).converted()
    }

    override func copy() -> any KVCache { converted() }

    /// Records a call's window: the previous state, then `length` inputs, in float32. Its end
    /// is the next state, and the history a trim steps back over.
    func record(window: MLXArray, length: Int) {
        // A prompt keeps only what a trim can step back over, as a copy that frees the rest.
        let history =
            length > rollbackCapacity
            ? contiguous(window[0..., (length - rollbackCapacity)...]) : window
        self[0] = history[0..., (history.dim(1) - stateLength)...]
        self[1] = history
        offset += length
        advance(length)
    }

    @discardableResult
    override func trim(_ n: Int) -> Int {
        guard n > 0, n <= maxTrimCount, let history = self[1] else { return 0 }
        let retained = history[0..., ..<(history.dim(1) - n)]
        self[0] = retained[0..., (retained.dim(1) - stateLength)...]
        self[1] = retained
        offset -= n
        return n
    }
}

extension RewindableConvolutionCache {
    /// A copy as an ``Inkling2ConvolutionCache``.
    fileprivate func converted() -> Inkling2ConvolutionCache {
        let cache = Inkling2ConvolutionCache(
            stateLength: stateLength, rollbackCapacity: rollbackCapacity)
        copyContents(to: cache)
        return cache
    }
}

/// A sliding layer's keys and values, or their positions, for a cache with a rewind window: a
/// ring of `window + reserve` slots, position `p` in slot `p % capacity`, then one slot of
/// zeros.
///
/// A write of up to `reserve + 1` positions, such as a speculative round's verify, goes in
/// place. Once the ring has wrapped, the call reads the whole ring in ring order: the stored
/// positions say where each key belongs, and the zero slot takes each query's sink (see
/// ``Inkling2Attention/ringAttention``). Before that, the slots up to the offset are in order
/// and the call reads them. A wrapped rotating cache returns its keys in temporal order
/// instead, which copies the window on every call, and a staged round copies it again to
/// append the staged keys. Longer writes, prompts, return the last `window - 1` positions in
/// temporal order and then the new ones, for the prefill kernel.
///
/// A trim of up to `reserve` positions only moves the offset. What the trimmed write left is
/// ahead of every later query, so masked, until a write replaces it, and what it overwrote is
/// older than any later query's window. A speculative round writes through this auxiliary
/// entry and trims it back, as mlx-vlm's rounds write their rotating caches and roll back.
final class Inkling2RingCache: BaseKVCache, AuxiliaryKVCache {
    override class var serializedClassName: String? { "Inkling2RingCache" }

    let window: Int
    let reserve: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    /// The furthest offset written. A trim reaches at most `reserve` back from it.
    private var furthest = 0

    init(window: Int, reserve: Int) {
        precondition(window > 0 && reserve > 0)
        self.window = window
        self.reserve = reserve
        super.init()
    }

    var capacity: Int { window + reserve }

    override var maxSize: Int? { window }

    override var evictedTokenCount: Int { Swift.max(0, offset - capacity) }

    /// Whether a write of `count` positions goes in place, so that its call reads the ring.
    func writesInPlace(_ count: Int) -> Bool { count <= reserve + 1 }

    override func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    override func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let count = newKeys.dim(2)
        if keys == nil {
            func zeros(_ x: MLXArray) -> MLXArray {
                MLXArray.zeros([x.dim(0), x.dim(1), capacity + 1, x.dim(3)], dtype: x.dtype)
            }
            (keys, values) = (zeros(newKeys), zeros(newValues))
        }
        var presented: (MLXArray, MLXArray)?
        if !writesInPlace(count) {
            let previous = Swift.min(offset, window - 1)
            presented = (
                concatenated([recent(keys!, previous), newKeys], axis: 2),
                concatenated([recent(values!, previous), newValues], axis: 2)
            )
        }
        // The last `capacity` positions of a long write fill the ring.
        let written = Swift.min(count, capacity)
        for (slots, rows) in runs(from: offset + count - written, count: written) {
            let rows = (count - written + rows.lowerBound) ..< (count - written + rows.upperBound)
            keys![.ellipsis, slots, 0...] = newKeys[.ellipsis, rows, 0...]
            values![.ellipsis, slots, 0...] = newValues[.ellipsis, rows, 0...]
        }
        offset += count
        furthest = Swift.max(furthest, offset)
        if let presented {
            return presented
        }
        // Until the ring wraps, its slots hold the positions in order.
        if furthest <= capacity {
            return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
        }
        return (keys!, values!)
    }

    /// Whether the write of `count` positions just made returned the ring in ring order: an
    /// in-place write once the ring has wrapped.
    func returnedRing(writing count: Int) -> Bool {
        writesInPlace(count) && furthest > capacity
    }

    /// The slots of `count` positions from `start`, and their rows among those positions, as
    /// at most two runs.
    private func runs(from start: Int, count: Int) -> [(slots: Range<Int>, rows: Range<Int>)] {
        guard count > 0 else { return [] }
        let first = start % capacity
        let head = Swift.min(count, capacity - first)
        var runs = [(slots: first ..< first + head, rows: 0 ..< head)]
        if head < count {
            runs.append((slots: 0 ..< count - head, rows: head ..< count))
        }
        return runs
    }

    /// The last `count` positions before the offset, in temporal order.
    private func recent(_ ring: MLXArray, _ count: Int) -> MLXArray {
        let parts = runs(from: offset - count, count: count).map {
            ring[.ellipsis, $0.slots, 0...]
        }
        return switch parts.count {
        case 0: ring[.ellipsis, ..<0, 0...]
        case 1: parts[0]
        default: concatenated(parts, axis: 2)
        }
    }

    override var isTrimmable: Bool { true }

    override func isTrimmable(after positions: Int) -> Bool {
        positions >= 0 && positions <= reserve
    }

    override var maxTrimCount: Int {
        Swift.max(0, Swift.min(offset, reserve - (furthest - offset)))
    }

    @discardableResult
    override func trim(_ n: Int) -> Int {
        guard n > 0, n <= maxTrimCount else { return 0 }
        offset -= n
        return n
    }

    override func copy() -> any KVCache {
        let copy = Inkling2RingCache(window: window, reserve: reserve)
        // New array objects: a write assigns through a subscript, which would reach a shared one.
        copy.keys = keys.map { $0[.ellipsis] }
        copy.values = values.map { $0[.ellipsis] }
        copy.offset = offset
        copy.furthest = furthest
        return copy
    }

    override var state: [MLXArray] {
        get { innerState() }
        set {
            precondition(newValue.isEmpty || newValue.count == 2, "a ring holds keys and values")
            keys = newValue.first
            values = newValue.count == 2 ? newValue[1] : nil
        }
    }

    override var metaState: [String] {
        get { ["\(offset)", "\(furthest)", "\(window)", "\(reserve)"] }
        set {
            assertionFailure(
                "Inkling2RingCache.metaState cannot be set directly; restore through prompt-cache loading."
            )
        }
    }

    override class func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
        let numbers = metaState.compactMap { Int($0) }
        guard metaState.count == 4, numbers.count == 4 else {
            throw KVCacheError(message: "Corrupt prompt cache: invalid Inkling2RingCache metadata.")
        }
        let (offset, furthest, window, reserve) = (numbers[0], numbers[1], numbers[2], numbers[3])
        guard window > 0, reserve > 0, offset >= 0, furthest >= offset,
            furthest - offset <= reserve,
            state.isEmpty
                ? offset == 0
                : state.count == 2
                    && state.allSatisfy({ $0.ndim == 4 && $0.dim(2) == window + reserve + 1 })
        else {
            throw KVCacheError(message: "Corrupt prompt cache: invalid Inkling2RingCache state.")
        }
        let cache = Inkling2RingCache(window: window, reserve: reserve)
        cache.state = state
        cache.offset = offset
        cache.furthest = furthest
        return cache
    }
}

final class Inkling2Attention: Module {
    let isSliding: Bool
    var heads: Int
    var kvHeads: Int
    let headDim: Int
    let window: Int
    let scale: Float
    let qkClipThreshold: Float
    let logScalingFloor: Float
    let logScalingAlpha: Float
    let eps: Float

    @ModuleInfo(key: "q_proj") var qProjection: Linear
    @ModuleInfo(key: "k_proj") var kProjection: Linear
    @ModuleInfo(key: "v_proj") var vProjection: Linear
    @ModuleInfo(key: "r_proj") var rProjection: Linear
    @ModuleInfo(key: "o_proj") var oProjection: Linear
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "sink_proj") var sinkProjection: Linear
    @ModuleInfo(key: "rel_logits_proj") var relativeLogits: Inkling2RelativeLogits
    @ModuleInfo(key: "q_l2_clip") var queryClip: Inkling2ScaledClip
    @ModuleInfo(key: "rel_l2_clip") var relativeClip: Inkling2ScaledClip
    /// An `Inkling2ShortConvolution` on sliding layers, an `Inkling2DynamicConvolution`
    /// on global ones.
    @ModuleInfo(key: "k_sconv") var keyConvolution: Module
    @ModuleInfo(key: "v_sconv") var valueConvolution: Module

    init(_ args: Inkling2TextConfiguration, sliding: Bool) {
        isSliding = sliding
        heads = sliding ? args.slidingAttentionHeads : args.attentionHeads
        kvHeads = sliding ? args.slidingKVHeads : args.kvHeads
        headDim = sliding ? args.slidingHeadDim : args.headDim
        window = sliding ? args.slidingWindow : 0
        scale = pow(Float(headDim), -0.5)
        qkClipThreshold = args.qkClipThreshold
        logScalingFloor = Float(args.logScalingFloor)
        logScalingAlpha = args.logScalingAlpha
        eps = args.rmsNormEps

        let dim = args.hiddenSize
        let qDim = heads * headDim
        let kvDim = kvHeads * headDim
        _qProjection.wrappedValue = Linear(dim, qDim, bias: false)
        _kProjection.wrappedValue = Linear(dim, kvDim, bias: false)
        _vProjection.wrappedValue = Linear(dim, kvDim, bias: false)
        _rProjection.wrappedValue = Linear(dim, heads * args.relativeDim, bias: false)
        _oProjection.wrappedValue = Linear(qDim, dim, bias: false)
        _gateProjection.wrappedValue = Linear(dim, qDim, bias: false)
        _sinkProjection.wrappedValue = Linear(dim, heads, bias: false)
        _relativeLogits.wrappedValue = Inkling2RelativeLogits(
            relativeDim: args.relativeDim,
            extent: sliding ? args.slidingWindow : args.relativeExtent)
        _queryClip.wrappedValue = Inkling2ScaledClip(
            heads: heads, dimensions: headDim, threshold: args.qkClipThreshold)
        _relativeClip.wrappedValue = Inkling2ScaledClip(
            heads: heads, dimensions: args.relativeDim, threshold: args.relativeClipThreshold)
        if sliding {
            _keyConvolution.wrappedValue = Inkling2ShortConvolution(
                dimensions: kvDim, kernelSize: args.convKernelSize)
            _valueConvolution.wrappedValue = Inkling2ShortConvolution(
                dimensions: kvDim, kernelSize: args.convKernelSize)
        } else {
            _keyConvolution.wrappedValue = Inkling2DynamicConvolution(
                dimensions: kvDim, inputDimensions: dim, heads: kvHeads,
                kernelSize: args.convKernelSize)
            _valueConvolution.wrappedValue = Inkling2DynamicConvolution(
                dimensions: kvDim, inputDimensions: dim, heads: kvHeads,
                kernelSize: args.convKernelSize)
        }
    }

    private func convolve(
        _ convolution: Module, _ x: MLXArray, input: MLXArray, state: MLXArray?, mask: MLXArray?,
        lengths: MLXArray?, keepWindow: Bool
    ) -> (MLXArray, MLXArray) {
        if let convolution = convolution as? Inkling2ShortConvolution {
            return convolution(
                x, state: state, mask: mask, lengths: lengths, keepWindow: keepWindow)
        }
        return (convolution as! Inkling2DynamicConvolution)(
            x, dynamicInput: input, state: state, mask: mask, lengths: lengths,
            keepWindow: keepWindow)
    }

    /// Everything before the cache write, from the normed inputs of the query, key and value
    /// projections: clipped queries `[B, L, H, D]` and relative states, keys and values
    /// `[B, Hkv, L, D]`, the output gate, the sinks, and the next key and value convolution
    /// states, or with `keepWindows` their windows.
    func project(
        query: MLXArray, key: MLXArray, value: MLXArray, states: (MLXArray?, MLXArray?),
        mask: MLXArray?, lengths: MLXArray?, keepWindows: Bool = false
    ) -> (
        queries: MLXArray, relative: MLXArray, keys: MLXArray, values: MLXArray,
        gate: MLXArray, sinks: MLXArray, keyState: MLXArray, valueState: MLXArray
    ) {
        if let fused = fusedDecodePrologue(
            query: query, key: key, value: value, states: states, mask: mask, lengths: lengths,
            keepWindows: keepWindows)
        {
            return fused
        }
        let (B, L) = (query.dim(0), query.dim(1))
        let (keys, keyState) = convolve(
            keyConvolution, kProjection(key), input: key, state: states.0, mask: mask,
            lengths: lengths, keepWindow: keepWindows)
        let (values, valueState) = convolve(
            valueConvolution, vProjection(value), input: value, state: states.1, mask: mask,
            lengths: lengths, keepWindow: keepWindows)

        let queries = queryClip(qProjection(query).reshaped(B, L, heads, -1))
        let relative = relativeClip(rProjection(query).reshaped(B, L, heads, -1))
        let clippedKeys = inkling2L2Clip(
            keys.reshaped(B, L, kvHeads, -1), scale: nil, threshold: qkClipThreshold,
            useKernel: !training)
        let headValues = values.reshaped(B, L, kvHeads, -1)
        let normedValues = inkling2UnitRMS(
            headValues, eps: eps, dtype: headValues.dtype)
        return (
            queries, relative, clippedKeys.transposed(0, 2, 1, 3),
            normedValues.transposed(0, 2, 1, 3), gateProjection(query), sinkProjection(query),
            keyState, valueState
        )
    }

    /// Longer contexts scale a global layer's queries and relative states up.
    func lengthScaled(
        queries: MLXArray, relative: MLXArray, positions: MLXArray
    ) -> (MLXArray, MLXArray) {
        guard !isSliding else { return (queries, relative) }
        let tau =
            1 + logScalingAlpha
            * log(MLX.maximum((positions + 1) / logScalingFloor, MLXArray(Float(1))))
        let tau4 = tau.reshaped(positions.dim(0), positions.dim(1), 1, 1)
        return (
            (queries.asType(.float32) * tau4).asType(queries.dtype),
            (relative.asType(.float32) * tau4).asType(relative.dtype)
        )
    }

    /// The gated output projection of the attention output `[B, L, H, D]`.
    func output(_ attention: MLXArray, gate: MLXArray) -> MLXArray {
        let (B, L) = (attention.dim(0), attention.dim(1))
        let gated = attention.reshaped(B, L, -1).asType(.float32) * sigmoid(gate.asType(.float32))
        return oProjection(gated.asType(gate.dtype))
    }

    /// `query`, `key` and `value` are the normed inputs of the three projections. Returns
    /// the output and the next key and value convolution states, or their windows when the
    /// cache ``Inkling2VisitCache/rewinds``.
    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray, cache: Inkling2VisitCache?,
        states: (MLXArray?, MLXArray?), mask: MLXArray?, lengths: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (B, L) = (query.dim(0), query.dim(1))
        let projected = project(
            query: query, key: key, value: value, states: states, mask: mask, lengths: lengths,
            keepWindows: cache?.rewinds ?? false)
        var keys = projected.keys
        var values = projected.values

        // Positions follow the cache after its write, which may roll ragged rows.
        let queryPositions: MLXArray
        var storedKeyPositions: MLXArray
        if let cache {
            (keys, values) = cache.keyValues.update(keys: keys, values: values)
            queryPositions = Self.queryPositions(cache.keyValues, batch: B, length: L)
            // Padding stores zero, which marks no key, also once a cache rolls it ahead of
            // the row's tokens.
            var stored = queryPositions + 1
            if let mask {
                stored = which(mask, stored, MLXArray(Float(0)))
            }
            stored = stored.reshaped(B, 1, L, 1)
            storedKeyPositions = cache.positions.update(keys: stored, values: stored).0
            storedKeyPositions = storedKeyPositions.reshaped(B, -1)
        } else {
            queryPositions = broadcast(
                MLXArray(Int32(0) ..< Int32(L)).asType(.float32)[.newAxis], to: [B, L])
            storedKeyPositions = queryPositions + 1
        }

        let (queries, relative) = lengthScaled(
            queries: projected.queries, relative: projected.relative, positions: queryPositions)
        let output = attention(
            queries: queries.transposed(0, 2, 1, 3), keys: keys, values: values,
            positionBias: relativeLogits(relative), sinks: projected.sinks,
            queryPositions: queryPositions, storedKeyPositions: storedKeyPositions,
            keyOrder: cache.map { Self.keyOrder($0.keyValues, writing: L) } ?? .temporal,
            // The prefill kernel has no gradient.
            forceReference: training)
        return (
            self.output(output, gate: projected.gate), projected.keyState, projected.valueState
        )
    }

    /// Each row's positions of the `length` tokens just written, `[B, L]` float32, or with
    /// `written: false` of the next `length` tokens.
    static func queryPositions(
        _ cache: KVCache, batch: Int, length: Int, written: Bool = true
    ) -> MLXArray {
        let start = written ? -length : 0
        switch cache.ropeOffset {
        case .scalar(let offset):
            let positions = (0 ..< length).map { Float(offset + start + $0) }
            return broadcast(MLXArray(positions)[.newAxis], to: [batch, length])
        case .batch(let offsets):
            let steps = MLXArray(Int32(0) ..< Int32(length))[.newAxis]
            return (expandedDimensions(offsets.asType(.int32), axis: 1) + Int32(start) + steps)
                .asType(.float32)
        }
    }

    /// The order of the keys a cache write returns.
    enum KeyOrder {
        /// Temporal, ending at the new keys, as the prefill kernel requires.
        case temporal
        /// An ``Inkling2RingCache``'s ring order, ending at a slot of zeros.
        case ring
        /// Anything else, such as a ring with a pinned prefix.
        case other
    }

    /// The order of the keys the write of `count` positions to `cache` just made returned.
    fileprivate static func keyOrder(_ cache: KVCache, writing count: Int) -> KeyOrder {
        switch cache {
        case let cache as Inkling2RingCache: cache.returnedRing(writing: count) ? .ring : .temporal
        case let cache as RotatingKVCache: cache.keepCount == 0 ? .temporal : .other
        // A speculative round's view of a ring: its window, then the staged keys.
        case let cache as RotatingStagedKVCache: cache.live.keepCount == 0 ? .temporal : .other
        case is KVCacheSimple, is BatchKVCache, is BatchRotatingKVCache: .temporal
        default: .other
        }
    }

    /// Attention with the relative-position bias and a sink per query and head.
    /// `queries` `[B, H, L, D]`; returns `[B, L, H, D]`. The prefill kernel requires
    /// `keyOrder` ``KeyOrder/temporal``.
    func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray,
        keyOrder: KeyOrder = .temporal, forceReference: Bool = false
    ) -> MLXArray {
        let (B, L) = (queries.dim(0), queries.dim(2))
        if !forceReference, L > 1, keyOrder == .temporal, inkling2UsesKernels(), headDim % 8 == 0,
            [queries.dtype, keys.dtype, values.dtype].allSatisfy({ $0 == .bfloat16 })
        {
            // Each row's keys before its first position are padding.
            let S = keys.dim(2)
            let keyStart = MLX.maximum(
                Float(S) - (queryPositions[0..., L - 1] + 1), MLXArray(Float(0)))
            return inkling2RelativeAttentionKernel(
                queries: queries, keys: keys, values: values, positionBias: positionBias,
                sinks: sinks, keyStart: keyStart, scale: scale, window: window)
        }
        if !forceReference, L == 1 {
            return decodeAttention(
                queries: queries, keys: keys, values: values, positionBias: positionBias,
                sinks: sinks, queryPositions: queryPositions,
                storedKeyPositions: storedKeyPositions)
        }
        if !forceReference, L > 1, keyOrder == .ring, inkling2UsesKernels() {
            return ringAttention(
                queries: queries, keys: keys, values: values, positionBias: positionBias,
                sinks: sinks, queryPositions: queryPositions,
                storedKeyPositions: storedKeyPositions)
        }

        // The dense bias of a call stays under 2^26 elements.
        let S = keys.dim(2)
        let block = Swift.max(1, (1 << 26) / Swift.max(1, B * heads * S))
        var outputs = [MLXArray]()
        var start = 0
        while start < L {
            let end = Swift.min(L, start + block)
            outputs.append(
                referenceAttention(
                    queries: queries[0..., 0..., start ..< end], keys: keys, values: values,
                    positionBias: positionBias[0..., start ..< end],
                    sinks: sinks[0..., start ..< end],
                    queryPositions: queryPositions[0..., start ..< end],
                    storedKeyPositions: storedKeyPositions))
            start = end
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1)
    }

    /// Distances from each query to each key, `[B, L, S]`, and whether each key is visible.
    private func distances(
        queryPositions: MLXArray, storedKeyPositions: MLXArray
    ) -> (distance: MLXArray, visible: MLXArray) {
        let keyPositions = storedKeyPositions - 1
        let distance =
            expandedDimensions(queryPositions, axis: 2) - expandedDimensions(keyPositions, axis: 1)
        var visible = (distance .>= 0) & (expandedDimensions(storedKeyPositions, axis: 1) .> 0)
        if window > 0 {
            visible = visible & (distance .< Float(window))
        }
        return (distance, visible)
    }

    private func referenceAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray
    ) -> MLXArray {
        let (B, L, D) = (queries.dim(0), queries.dim(2), queries.dim(3))
        let S = keys.dim(2)
        let R = positionBias.dim(-1)
        let groups = heads / kvHeads

        let (distance, visible) = distances(
            queryPositions: queryPositions, storedKeyPositions: storedKeyPositions)
        let headDistance = expandedDimensions(distance, axis: 1)
        let index = broadcast(
            clip(headDistance, min: 0, max: R - 1).asType(.int32), to: [B, heads, L, S])
        var bias = takeAlong(positionBias.transposed(0, 2, 1, 3), index, axis: -1)
            .asType(.float32)
        bias = which(headDistance .< Float(R), bias, MLXArray(Float(0)))
        let mask = which(expandedDimensions(visible, axis: 1), bias, MLXArray(-Float.infinity))
            .reshaped(B, kvHeads, groups, L, S)

        let q = queries.asType(.float32).reshaped(B, kvHeads, groups, L, D)
        let k = expandedDimensions(keys.asType(.float32), axis: 2)
        let scores = matmul(q, k.swappedAxes(-1, -2)) * scale + mask
        let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1)
            .reshaped(B, kvHeads, groups, L, 1)
        let weights = softmax(concatenated([scores, sinkLogits], axis: -1), axis: -1)[
            .ellipsis, ..<S]
        let output = matmul(weights.asType(values.dtype), expandedDimensions(values, axis: 2))
        return output.reshaped(B, heads, L, -1).transposed(0, 2, 1, 3)
    }

    /// One query per row: the fused attention takes the bias as an additive mask. It takes
    /// one sink per head, so each row's heads become heads of a single row.
    private func decodeAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray
    ) -> MLXArray {
        let B = queries.dim(0)
        let mask = decodeMask(
            positionBias: positionBias, queryPositions: queryPositions,
            storedKeyPositions: storedKeyPositions)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.reshaped(1, B * heads, 1, -1),
            keys: keys.reshaped(1, B * kvHeads, keys.dim(2), -1),
            values: values.reshaped(1, B * kvHeads, values.dim(2), -1), scale: scale,
            mask: mask.asType(queries.dtype).reshaped(1, B * heads, 1, -1),
            sinks: sinks.reshaped(-1).asType(queries.dtype))
        return output.reshaped(B, heads, 1, -1).transposed(0, 2, 1, 3)
    }

    /// A few queries per row over an ``Inkling2RingCache``'s keys, in ring order.
    ///
    /// The fused attention takes one sink per head, and Inkling 2's sinks vary by query. So the
    /// mask kernel gives each visible key its bias and the ring's last slot, whose key and
    /// value are zeros, each query's sink. That slot's score is then exactly the sink, and it
    /// adds nothing to the output, as a sink column does.
    func ringAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray
    ) -> MLXArray {
        let mask = inkling2DecodeMaskKernel(
            positionBias: positionBias, queryPositions: queryPositions,
            storedKeyPositions: storedKeyPositions, window: window, sinks: sinks)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask.asType(queries.dtype))
        return output.transposed(0, 2, 1, 3)
    }

    /// The additive mask of one query per row, `[B, H, 1, S]` in `positionBias`'s dtype.
    func decodeMask(
        positionBias: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray,
        forceReference: Bool = false
    ) -> MLXArray {
        if !forceReference, !training, inkling2UsesKernels() {
            return inkling2DecodeMaskKernel(
                positionBias: positionBias, queryPositions: queryPositions,
                storedKeyPositions: storedKeyPositions, window: window)
        }
        let (B, R) = (positionBias.dim(0), positionBias.dim(-1))
        let (distance, visible) = distances(
            queryPositions: queryPositions, storedKeyPositions: storedKeyPositions)
        let index = clip(distance, min: 0, max: R - 1).asType(.int32)
        var bias = takeAlong(
            positionBias[0..., 0], broadcast(index, to: [B, heads, distance.dim(2)]), axis: -1)
        bias = which(distance .< Float(R), bias, MLXArray(Float(0)).asType(bias.dtype))
        return which(visible, bias, MLXArray(-Float.infinity).asType(bias.dtype))
            .reshaped(B, heads, 1, -1)
    }
}

// MARK: - Decoder layer

/// A loop pass's messages for a layer `[B, L, 4, D]` (query, key, value and MLP) and their
/// unit RMS, normed once for the four gates.
private typealias Inkling2Messages = (raw: MLXArray, normed: MLXArray)

package final class Inkling2DecoderLayer: Module {
    let index: Int
    let isSliding: Bool
    let isBlockBoundary: Bool

    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2DepthQuery?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2DepthQuery
    @ModuleInfo(key: "mlp") var mlp: Inkling2SparseMoE
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttentionConvolution:
        Inkling2ShortConvolution
    @ModuleInfo(key: "post_mlp_convolution") var postMLPConvolution: Inkling2ShortConvolution
    @ModuleInfo(key: "self_attn") var attention: Inkling2Attention

    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var queryInputNorm: RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var keyInputNorm: RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var valueInputNorm: RMSNorm?

    @ModuleInfo(key: "q_message_gate") var queryMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") var keyMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") var valueMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") var mlpMessageGate: Inkling2MessageGate?

    init(_ args: Inkling2TextConfiguration, index: Int) {
        self.index = index
        isSliding = args.isSliding(index)
        isBlockBoundary = index % args.layersPerResidualBlock == 0
        let dim = args.hiddenSize
        let eps = args.rmsNormEps

        if index > 0 {
            _attentionInput.wrappedValue = Inkling2DepthQuery(
                dimensions: dim, queries: isSliding ? 1 : 3)
        }
        _mlpInput.wrappedValue = Inkling2DepthQuery(dimensions: dim)
        _mlp.wrappedValue = Inkling2SparseMoE(args)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postAttentionConvolution.wrappedValue = Inkling2ShortConvolution(
            dimensions: dim, kernelSize: args.convKernelSize)
        _postMLPConvolution.wrappedValue = Inkling2ShortConvolution(
            dimensions: dim, kernelSize: args.convKernelSize)
        _attention.wrappedValue = Inkling2Attention(args, sliding: isSliding)

        if isSliding {
            _inputNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        } else {
            _queryInputNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
            _keyInputNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
            _valueInputNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        }

        if let loop = args.loop, loop.core.contains(index) {
            _queryMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dim, rank: loop.gateRank)
            _keyMessageGate.wrappedValue = Inkling2MessageGate(dimensions: dim, rank: loop.gateRank)
            _valueMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dim, rank: loop.gateRank)
            _mlpMessageGate.wrappedValue = Inkling2MessageGate(dimensions: dim, rank: loop.gateRank)
        }
    }

    /// The normed inputs of the query, key and value projections.
    private func attentionInputs(
        _ x: MLXArray, depth: MLXArray?, messages: Inkling2Messages?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let dtype = x.dtype
        var inputs: (MLXArray, MLXArray, MLXArray)
        if isSliding {
            let h = depth?[.ellipsis, 0, 0...] ?? x
            if messages == nil, let inputNorm {
                let normed = inkling2Normed(inputNorm, h, dtype: dtype)
                return (normed, normed, normed)
            }
            inputs = (h, h, h)
        } else if let depth {
            inputs = (
                depth[.ellipsis, 0, 0...], depth[.ellipsis, 1, 0...], depth[.ellipsis, 2, 0...]
            )
        } else {
            inputs = (x, x, x)
        }
        if let messages {
            inputs = (
                queryMessageGate!(inputs.0, messages: messages, gate: 0),
                keyMessageGate!(inputs.1, messages: messages, gate: 1),
                valueMessageGate!(inputs.2, messages: messages, gate: 2)
            )
        }
        if let inputNorm {
            return (
                inkling2Normed(inputNorm, inputs.0, dtype: dtype),
                inkling2Normed(inputNorm, inputs.1, dtype: dtype),
                inkling2Normed(inputNorm, inputs.2, dtype: dtype)
            )
        }
        return (
            inkling2Normed(queryInputNorm!, inputs.0, dtype: dtype),
            inkling2Normed(keyInputNorm!, inputs.1, dtype: dtype),
            inkling2Normed(valueInputNorm!, inputs.2, dtype: dtype)
        )
    }

    /// Returns the layer output and the blocks completed so far.
    func callAsFunction(
        _ x: MLXArray, blocks: MLXArray?, cache: Inkling2VisitCache?, mask: MLXArray?,
        messages: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let messages = messages.map { ($0, inkling2UnitRMS($0)) }
        if let step = decode(x, blocks: blocks, cache: cache, mask: mask, messages: messages) {
            return step
        }
        let lengths = cache?.lengths

        let normedInput = inkling2UnitRMS(x)
        let depth = attentionInput?(blocks: blocks, partial: normedInput)
        let (q, k, v) = attentionInputs(x, depth: depth, messages: messages)
        let (h, keyState, valueState) = attention(
            query: q, key: k, value: v, cache: cache,
            states: (cache?.state(0), cache?.state(1)), mask: mask, lengths: lengths)
        let (out, newBlocks, attentionState, mlpState) = feedForward(
            x, normedInput: normedInput, attention: h, blocks: blocks, messages: messages,
            states: (cache?.state(2), cache?.state(3)), mask: mask, lengths: lengths,
            keepWindows: cache?.rewinds ?? false)

        cache?.commit([keyState, valueState, attentionState, mlpState], length: x.dim(1))
        return (out, newBlocks)
    }

    /// The rest of the layer after the attention's output projection `h`. Returns the
    /// output, the blocks, and the next post-attention and post-MLP convolution states, or
    /// with `keepWindows` their windows.
    private func feedForward(
        _ x: MLXArray, normedInput: MLXArray, attention h: MLXArray, blocks: MLXArray?,
        messages: Inkling2Messages?, states: (MLXArray?, MLXArray?), mask: MLXArray?,
        lengths: MLXArray?, keepWindows: Bool = false
    ) -> (MLXArray, MLXArray?, MLXArray, MLXArray) {
        let dtype = x.dtype
        var blocks = blocks
        if isBlockBoundary {
            let block = normedInput[.newAxis]
            blocks = blocks.map { concatenated([$0, block], axis: 0) } ?? block
        }

        let residual: MLXArray
        let attentionState: MLXArray
        var m: MLXArray
        let normalized: MLXArray?
        if !training, !mlpInput.training, !postAttentionConvolution.training,
            mlp.shardingGroup == nil, mask == nil, lengths == nil,
            !keepWindows, dtype == .bfloat16, let blocks, let state = states.0,
            postAttentionConvolution.kernelSize == 4,
            let mix = inkling2FusedDepthMix(
                blocks: blocks, base: isBlockBoundary ? nil : x,
                convolution: (h, state, postAttentionConvolution.conv1d.weight),
                query: mlpInput.query,
                normWeight: messages == nil ? postAttentionNorm.weight : nil,
                eps: postAttentionNorm.eps, outputType: dtype)
        {
            residual = mix.residual
            attentionState = mix.state!
            m = mix.readout[.ellipsis, 0, 0...]
            normalized = mix.normalizedReadout.map { $0[.ellipsis, 0, 0...] }
        } else {
            normalized = nil
            let convolved = postAttentionConvolution(
                h, state: states.0, mask: mask, lengths: lengths, keepWindow: keepWindows)
            attentionState = convolved.1
            residual =
                isBlockBoundary
                ? convolved.0.asType(.float32) : x.asType(.float32) + convolved.0.asType(.float32)
            m =
                mlpInput(blocks: blocks, partial: inkling2UnitRMS(residual))[
                    .ellipsis, 0, 0...]
        }
        if let messages {
            m = mlpMessageGate!(m, messages: messages, gate: 3)
        }
        let expertInput = normalized ?? inkling2Normed(postAttentionNorm, m, dtype: dtype)
        if !training, mask == nil, lengths == nil,
            let fused = mlp.fusedExpertDecode(
                expertInput, residual: residual, state: states.1,
                convolutionWeight: postMLPConvolution.conv1d.weight, keepWindow: keepWindows)
        {
            return (fused.0, blocks, attentionState, fused.1)
        }
        let (out, mlpState) = postMLPConvolution(
            mlp(expertInput),
            state: states.1, mask: mask,
            lengths: lengths, keepWindow: keepWindows)
        return ((residual + out.asType(.float32)).asType(dtype), blocks, attentionState, mlpState)
    }

    // MARK: Decode

    // A decode step runs as two compiled traces, before and after the cache writes, whose
    // offsets move every token. Arguments: `[x, key state, value state, positions]` and
    // `[x, normed input, attention, gate, post-attention state, post-MLP state]`, each
    // followed by the blocks (every layer but the first) and the loop messages and their
    // unit RMS, if any. A cache with a rewind window takes the second pair, whose
    // convolutions return their windows.
    private func decodeInputState() -> [Module] {
        let optional: [Module?] = [
            attentionInput, inputNorm, queryInputNorm, keyInputNorm, valueInputNorm,
            queryMessageGate, keyMessageGate, valueMessageGate,
        ]
        return [attention] + optional.compactMap { $0 }
    }

    private func decodeOutputState() -> [Module] {
        let state: [Module] = [
            attention.oProjection, postAttentionConvolution, mlpInput, postAttentionNorm, mlp,
            postMLPConvolution,
        ]
        return state + [mlpMessageGate].compactMap { $0 }
    }

    private let compiledDecodeInput = CompiledTrace<Inkling2DecoderLayer>(
        state: { $0.decodeInputState() }
    ) { layer, arguments in
        layer.decodeInput(arguments, keepWindows: false)
    }
    private let compiledDecodeOutput = CompiledTrace<Inkling2DecoderLayer>(
        state: { $0.decodeOutputState() }
    ) { layer, arguments in
        layer.decodeOutput(arguments, keepWindows: false)
    }
    private let compiledWindowInput = CompiledTrace<Inkling2DecoderLayer>(
        state: { $0.decodeInputState() }
    ) { layer, arguments in
        layer.decodeInput(arguments, keepWindows: true)
    }
    private let compiledWindowOutput = CompiledTrace<Inkling2DecoderLayer>(
        state: { $0.decodeOutputState() }
    ) { layer, arguments in
        layer.decodeOutput(arguments, keepWindows: true)
    }

    /// The blocks and messages after `count` leading arguments.
    private func extras(
        _ arguments: [MLXArray], after count: Int
    ) -> (MLXArray?, Inkling2Messages?) {
        let blocks = index > 0 ? arguments[count] : nil
        let start = count + (index > 0 ? 1 : 0)
        let messages =
            arguments.count > start
            ? (arguments[start], arguments[start + 1]) : nil
        return (blocks, messages)
    }

    /// Returns `[normed input, queries, keys, values, position bias, sinks, gate, stored
    /// positions, key state, value state]`, the states being windows with `keepWindows`.
    private func decodeInput(_ arguments: [MLXArray], keepWindows: Bool) -> [MLXArray] {
        let (x, positions) = (arguments[0], arguments[3])
        let (blocks, messages) = extras(arguments, after: 4)
        let normedInput: MLXArray
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let norms =
            inputNorm.map { $0.weight.reshaped(1, -1) }
            ?? stacked([queryInputNorm!.weight, keyInputNorm!.weight, valueInputNorm!.weight])
        let normEpsilon = inputNorm?.eps ?? queryInputNorm!.eps
        let sharedEpsilon =
            inputNorm != nil
            || (queryInputNorm!.eps == keyInputNorm!.eps
                && queryInputNorm!.eps == valueInputNorm!.eps)
        if !keepWindows, x.dtype == .bfloat16, let attentionInput, !attentionInput.training,
            let blocks, messages != nil || sharedEpsilon,
            let mix = inkling2FusedDepthMix(
                blocks: blocks, base: x, convolution: nil, query: attentionInput.query,
                normWeight: messages == nil ? norms : nil, eps: normEpsilon,
                outputType: x.dtype)
        {
            normedInput = mix.normed
            if let readout = mix.normalizedReadout {
                q = readout[.ellipsis, 0, 0...]
                k = isSliding ? q : readout[.ellipsis, 1, 0...]
                v = isSliding ? q : readout[.ellipsis, 2, 0...]
            } else {
                (q, k, v) = attentionInputs(x, depth: mix.readout, messages: messages)
            }
        } else {
            normedInput = inkling2UnitRMS(x)
            let depth = attentionInput?(blocks: blocks, partial: normedInput)
            (q, k, v) = attentionInputs(x, depth: depth, messages: messages)
        }
        let projected = attention.project(
            query: q, key: k, value: v, states: (arguments[1], arguments[2]), mask: nil,
            lengths: nil, keepWindows: keepWindows)
        let (queries, relative) = attention.lengthScaled(
            queries: projected.queries, relative: projected.relative, positions: positions)
        return [
            normedInput, queries.transposed(0, 2, 1, 3), projected.keys, projected.values,
            attention.relativeLogits(relative), projected.sinks, projected.gate,
            (positions + 1).reshaped(x.dim(0), 1, x.dim(1), 1), projected.keyState,
            projected.valueState,
        ]
    }

    /// Returns `[output, blocks, post-attention state, post-MLP state]`, the states being
    /// windows with `keepWindows`.
    private func decodeOutput(_ arguments: [MLXArray], keepWindows: Bool) -> [MLXArray] {
        let (blocks, messages) = extras(arguments, after: 6)
        let (out, newBlocks, attentionState, mlpState) = feedForward(
            arguments[0], normedInput: arguments[1],
            attention: attention.output(arguments[2], gate: arguments[3]), blocks: blocks,
            messages: messages, states: (arguments[4], arguments[5]), mask: nil, lengths: nil,
            keepWindows: keepWindows)
        return [out, newBlocks!, attentionState, mlpState]
    }

    /// Runs on the GPU through the compiled traces: one token per row, or with a rewind
    /// window up to 16, a speculative round's verify. Nil where the general path must run:
    /// prompts, padding, training, a sharded layer, or no convolution states yet.
    private func decode(
        _ x: MLXArray, blocks: MLXArray?, cache: Inkling2VisitCache?, mask: MLXArray?,
        messages: Inkling2Messages?
    ) -> (MLXArray, MLXArray?)? {
        let (B, L) = (x.dim(0), x.dim(1))
        guard let cache, L == 1 || (cache.rewinds && L <= 16), mask == nil,
            cache.lengths == nil, inkling2UsesKernels(), !training, mlp.shardingGroup == nil,
            (blocks != nil) == (index > 0)
        else { return nil }
        let states = (0 ..< 4).compactMap { cache.state($0) }
        guard states.count == 4 else { return nil }
        let extras = [blocks, messages?.raw, messages?.normed].compactMap { $0 }
        let (inputTrace, outputTrace) =
            cache.rewinds
            ? (compiledWindowInput, compiledWindowOutput)
            : (compiledDecodeInput, compiledDecodeOutput)

        // No row is padded, so the next positions are the query positions.
        let positions = Inkling2Attention.queryPositions(
            cache.keyValues, batch: B, length: L, written: false)
        let input = inputTrace(self, [x, states[0], states[1], positions] + extras)
        let (keys, values) = cache.keyValues.update(keys: input[2], values: input[3])
        let stored = cache.positions.update(keys: input[7], values: input[7]).0.reshaped(B, -1)
        let attended = attention.attention(
            queries: input[1], keys: keys, values: values, positionBias: input[4],
            sinks: input[5], queryPositions: positions, storedKeyPositions: stored,
            keyOrder: Inkling2Attention.keyOrder(cache.keyValues, writing: L))
        let output = outputTrace(
            self, [x, input[0], attended, input[6], states[2], states[3]] + extras)

        cache.commit([input[8], input[9], output[2], output[3]], length: L)
        return (output[0], output[1])
    }
}

// MARK: - Text model

/// Why an Inkling 2 cache or model cannot be built.
public enum Inkling2Error: LocalizedError {
    case unsupportedKVCacheStrategy
    case unsupportedLastLoopRewind

    public var errorDescription: String? {
        switch self {
        case .unsupportedKVCacheStrategy:
            "Inkling 2 attention reads full-precision keys and values only; "
                + "KV cache quantization and compression are not supported."
        case .unsupportedLastLoopRewind:
            "Inkling 2's experimental last_loop cache does not support rewind or speculative decoding."
        }
    }
}

/// The Inkling 2 text model: token embedding, decoder layers (looped for Inkling 2 Mini),
/// and the final depth attention and norm.
///
/// The caller owns the caches from ``newCache(parameters:)``. Convolution states are
/// per visit; attention history is per visit or per layer in experimental `last_loop` mode.
public final class Inkling2TextModel: Module {
    package let configuration: Inkling2TextConfiguration
    package let layerVisits: [Int]
    private let attentionOwners: [Int]
    private let attentionCacheCount: Int

    @ModuleInfo(key: "embed_tokens") package var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    package let layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2DepthQuery
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    package init(_ args: Inkling2TextConfiguration) {
        configuration = args
        layerVisits = args.layerVisits
        if args.recurrentKVCacheMode == .lastLoop {
            let lastVisits = Dictionary(
                args.layerVisits.enumerated().map { ($1, $0) },
                uniquingKeysWith: { _, last in last })
            attentionOwners = args.layerVisits.map { lastVisits[$0]! }
            attentionCacheCount = 2 * lastVisits.count
        } else {
            attentionOwners = Array(args.layerVisits.indices)
            attentionCacheCount = 2 * args.layerVisits.count
        }
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _embedNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        layers = (0 ..< args.hiddenLayers).map { Inkling2DecoderLayer(args, index: $0) }
        _finalInput.wrappedValue = Inkling2DepthQuery(dimensions: args.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if let loop = args.loop {
            _messageInput.wrappedValue = Inkling2MessageInput(
                dimensions: args.hiddenSize, loopedLayers: loop.core.count)
        }
    }

    /// Normed token embeddings `[B, L, hidden]`.
    package func embed(_ inputs: MLXArray) -> MLXArray {
        embedNorm(embedTokens(inputs))
    }

    /// Final normed hidden states `[B, L, hidden]` for token ids `[B, L]`.
    package func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        callAsFunction(embeddings: embed(inputs), cache: cache)
    }

    /// Final normed hidden states `[B, L, hidden]` for input embeddings `[B, L, hidden]`.
    package func callAsFunction(embeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        hiddenStates(embeddings: embeddings, cache: cache, capturing: []).hidden
    }

    /// Final normed hidden states `[B, L, hidden]`, and a tap for each of `visits`: the
    /// hidden state entering that layer visit, the partial block, or for `layerVisits.count`
    /// the final normed state. A DSpark drafter reads these taps.
    package func hiddenStates(
        embeddings: MLXArray, cache: [KVCache]?, outputRows: Range<Int>? = nil,
        capturing visits: [Int]
    ) -> (hidden: MLXArray, captured: [MLXArray]) {
        precondition(
            cache.map {
                [1, 4].map { attentionCacheCount + $0 * layerVisits.count }.contains($0.count)
            } ?? true,
            "Inkling 2 requires the caches from newCache(parameters:)")
        precondition(
            visits.allSatisfy { (0 ... layerVisits.count).contains($0) },
            "Inkling 2 taps name layer visits 0 through \(layerVisits.count)")
        let dtype = embeddings.dtype
        let caches = cache.map { visitCaches($0) }
        let mask = caches?.first?.makeMask(N: embeddings.dim(1))
        var taps = [Int: MLXArray]()
        // A pass over a few positions, such as a speculative verify, costs mostly per-op
        // overhead. Starting what is built every four visits lets the GPU work while the rest
        // of the graph is built, as mlx-vlm's verify does; it saves 3-4 ms of a 2- to
        // 8-token verify on Inkling 2 Mini. Training traces the whole graph instead.
        let pipelines = !training && (2 ... 16).contains(embeddings.dim(1))

        var visit = 0
        func run(
            _ range: Range<Int>, _ h: MLXArray, _ blocks: MLXArray?,
            bank: Inkling2MessageBank? = nil
        ) -> (MLXArray, MLXArray?) {
            var (h, blocks) = (h, blocks)
            // Decode reuses the same bank throughout a pass. Group its readouts once.
            let passMessages = h.dim(1) == 1 ? bank.map { messageInput!.allMessages($0) } : nil
            for (i, index) in range.enumerated() {
                if visits.contains(visit) {
                    taps[visit] = h
                }
                let messages =
                    passMessages.map {
                        $0[0..., 0..., i ..< i + 1, 0..., 0...].squeezed(axis: 2)
                    } ?? bank.map { messageInput!($0, loopedLayer: i) }
                let visitCache = caches.map { caches -> Inkling2VisitCache in
                    let stored = caches[visit]
                    guard attentionOwners[visit] != visit else { return stored }
                    // A temporary update reads final-pass history without changing it. Copy
                    // only this visit's attention caches; convolution states stay pass-local.
                    return Inkling2VisitCache(
                        keyValues: stored.keyValues.copy(), positions: stored.positions.copy(),
                        convolutions: stored.convolutions)
                }
                (h, blocks) = layers[index](
                    h, blocks: blocks, cache: visitCache, mask: mask, messages: messages)
                visit += 1
                if pipelines, visit.isMultiple(of: 4) {
                    asyncEval(h)
                }
            }
            return (h, blocks)
        }

        var h: MLXArray
        var blocks: MLXArray?
        if let loop = configuration.loop {
            let prelude = run(0 ..< loop.coreStart, embeddings, nil)
            let preludeBlocks = prelude.1?.dim(0) ?? 0
            (h, blocks) = prelude
            for pass in 0 ..< loop.passes {
                var bank: Inkling2MessageBank?
                if pass > 0 {
                    let completed = blocks.map { $0[(preludeBlocks + 1)...] }
                    bank = Inkling2MessageBank(
                        blocks: completed, partial: inkling2UnitRMS(h))
                    (h, blocks) = prelude
                }
                (h, blocks) = run(loop.core, h, blocks, bank: bank)
            }
            (h, blocks) = run(loop.coreEnd ..< configuration.hiddenLayers, h, blocks)
        } else {
            (h, blocks) = run(0 ..< configuration.hiddenLayers, embeddings, nil)
        }

        if let outputRows {
            h = h[0..., outputRows, 0...]
            blocks = blocks.map { $0[0..., 0..., outputRows, 0...] }
        }
        h =
            finalInput(blocks: blocks, partial: inkling2UnitRMS(h))[
                .ellipsis, 0, 0...]
        let hidden = inkling2Normed(norm, h, dtype: dtype)
        taps[layerVisits.count] = hidden
        return (hidden, visits.map { taps[$0]! })
    }

    private func visitCaches(_ cache: [KVCache]) -> [Inkling2VisitCache] {
        let convolutionCount = (cache.count - attentionCacheCount) / layerVisits.count
        let sharesHistory = attentionCacheCount < 2 * layerVisits.count
        return layerVisits.indices.map { visit in
            let i = sharesHistory ? 2 * layerVisits[visit] : (2 + convolutionCount) * visit
            let convolutionStart =
                sharesHistory ? attentionCacheCount + convolutionCount * visit : i + 2
            let convolutions = cache[convolutionStart ..< (convolutionStart + convolutionCount)].map
            { entry in
                guard let convolution = entry as? ArraysCache else {
                    preconditionFailure(
                        "Inkling 2 cache \(convolutionStart) must hold convolution states")
                }
                return convolution
            }
            precondition(
                usesPlainAttentionCacheRoute(cache[i]),
                "Inkling 2 attention reads full-precision keys and values only")
            return Inkling2VisitCache(
                keyValues: cache[i], positions: cache[i + 1], convolutions: convolutions)
        }
    }

    /// Each visit stores its four convolution states. In `last_loop`, all physical layers'
    /// attention pairs precede the visits' convolution entries. Otherwise they interleave.
    /// With a rewind
    /// window, sliding layers' keys and positions take an ``Inkling2RingCache`` each and the
    /// convolution states an ``Inkling2ConvolutionCache`` each.
    package func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        if let strategy = try parameters?.resolvedKVCacheConfiguration()?.strategy,
            strategy != .fullPrecision
        {
            throw Inkling2Error.unsupportedKVCacheStrategy
        }
        let rewind = parameters?.kvCache?.rewind?.maxTokens
        if rewind != nil, attentionCacheCount < 2 * layerVisits.count {
            throw Inkling2Error.unsupportedLastLoopRewind
        }
        let sharesHistory = attentionCacheCount < 2 * layerVisits.count
        let attentionLayers = sharesHistory ? Array(0 ..< configuration.hiddenLayers) : layerVisits
        let attentions = try attentionLayers.map { index -> [KVCache] in
            func attentionCache() throws -> KVCache {
                let cache = try makeHybridAttentionKVCache(
                    parameters: parameters, slidingWindow: configuration.slidingWindow,
                    usesSlidingWindow: configuration.isSliding(index))
                // A sliding layer with a rewind window writes in place, unless a smaller
                // `maxKVSize` shortens its window, which a rotating cache keeps. A global
                // layer keeps the rotating cache that a requested capacity gives it, with its
                // pinned prefix and no reserve in view.
                if configuration.isSliding(index), let rotating = cache as? RotatingKVCache,
                    rotating.rewindCapacity > 0, rotating.maxSize == configuration.slidingWindow
                {
                    return Inkling2RingCache(
                        window: configuration.slidingWindow, reserve: rotating.rewindCapacity)
                }
                return cache
            }
            return [try attentionCache(), try attentionCache()]
        }
        let convolutions: [[KVCache]] = layerVisits.map { _ in
            rewind.map { rewind in
                (0 ..< 4).map { _ in
                    Inkling2ConvolutionCache(
                        stateLength: configuration.convKernelSize - 1,
                        rollbackCapacity: rewind)
                }
            } ?? [ArraysCache(size: 4)]
        }
        if sharesHistory {
            return attentions.flatMap { $0 } + convolutions.flatMap { $0 }
        }
        return zip(attentions, convolutions).flatMap { $0 + $1 }
    }
}

/// Vocabulary logits for final hidden states: a final softcap, then the unpadded vocabulary.
package func inkling2Logits(
    _ hidden: MLXArray, lmHead: Linear, configuration: Inkling2TextConfiguration
) -> MLXArray {
    let cap = configuration.finalLogitSoftcap
    let logits = tanh(lmHead(hidden).asType(.float32) / cap) * cap
    return logits[.ellipsis, ..<configuration.unpaddedVocabularySize]
}

// MARK: - Speculative decoding

/// An Inkling 2 model, whose text model and vocabulary head a DSpark drafter shares.
///
/// The drafter lives beside the text model; MLXLLM's and MLXVLM's wrappers, which conform,
/// live in other modules.
package protocol Inkling2LanguageModel: LanguageModel {
    var inkling2TextModel: Inkling2TextModel { get }
    var inkling2Head: Linear { get }
}

extension Inkling2LanguageModel {
    /// Logits for input embeddings `[B, L, hidden]`. When `state` asks for drafter state
    /// (``mtpEmitFlagKey``), the output state also holds the taps ``mtpTargetLayerIdsKey``
    /// names, concatenated in that order, under ``mtpLastHiddenStatesKey``.
    package func inkling2Output(
        embeddings: MLXArray, cache: [KVCache]?, state: LMOutput.State?,
        outputRows: Range<Int>? = nil
    ) -> LMOutput {
        let emits = state?[mtpEmitFlagKey] ?? false
        let visits = emits ? (state?[mtpTargetLayerIdsKey] ?? []) : []
        let (hidden, taps) = inkling2TextModel.hiddenStates(
            embeddings: embeddings, cache: cache, outputRows: emits ? nil : outputRows,
            capturing: visits)
        let readout = emits ? outputRows.map { hidden[0..., $0, 0...] } ?? hidden : hidden
        let logits = inkling2Logits(
            readout, lmHead: inkling2Head, configuration: inkling2TextModel.configuration)
        guard emits else { return LMOutput(logits: logits) }

        var output = state ?? LMOutput.State()
        output[mtpLastHiddenStatesKey] = taps.isEmpty ? hidden : concatenated(taps, axis: -1)
        // DSpark reads hidden states, not the target's keys and values.
        output[mtpSharedKVStatesKey] = [:]
        output[mtpSharedKVOffsetsKey] = [:]
        output[mtpSharedKVSourceIndicesKey] = [:]
        return LMOutput(logits: logits, state: output)
    }
}

// MARK: - Tensor parallelism

/// Splits an expert's hidden width: gate and up by output row, down by input column. A
/// per-expert global scale applies to whole outputs, so it stays whole.
private func inkling2ExpertSharding(
    down: Bool
) -> (String, MLXArray) -> (axis: Int, segments: Segments)? {
    { path, weight in
        guard path == "weight" || path == "scales" || path == "biases" else { return nil }
        return (down ? -1 : weight.ndim - 2, .count(1))
    }
}

extension Inkling2TextModel {
    /// Keeps this process' share of every attention head and every expert's width.
    /// Embeddings, norms, routers, message gates and the post-attention and post-MLP
    /// convolutions stay whole.
    package func shard(group: MLXDistributed.Group) throws {
        let size = try group.size
        let rank = try group.rank
        let args = configuration
        try requireDivisible(
            [
                "attention heads": args.attentionHeads, "key-value heads": args.kvHeads,
                "sliding attention heads": args.slidingAttentionHeads,
                "sliding key-value heads": args.slidingKVHeads,
            ], by: size)
        // A shard of a quantized expert must hold whole quantization groups.
        for layer in layers {
            let down = layer.mlp.experts.downProjection
            let groupSize = (down as? QuantizedSwitchLinear)?.groupSize ?? 1
            try requireDivisible(
                ["expert width in groups of \(groupSize)": args.intermediateSize / groupSize],
                by: size)
        }

        func part(_ array: MLXArray, axis: Int) -> MLXArray {
            contiguous(array.split(parts: size, axis: axis)[rank])
        }

        for layer in layers {
            let attention = layer.attention
            try attention.shardLinears(
                allToSharded: ["q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj"],
                shardedToAll: ["o_proj"], group: group)
            attention.queryClip.update(
                parameters: ModuleParameters.unflattened([
                    "scale": part(attention.queryClip.scale, axis: 0)
                ]))
            attention.relativeClip.update(
                parameters: ModuleParameters.unflattened([
                    "scale": part(attention.relativeClip.scale, axis: 0)
                ]))
            for convolution in [attention.keyConvolution, attention.valueConvolution] {
                if let convolution = convolution as? Inkling2ShortConvolution {
                    convolution.shard(rank: rank, size: size)
                } else if let convolution = convolution as? Inkling2DynamicConvolution {
                    convolution.update(
                        parameters: ModuleParameters.unflattened([
                            "static_weight": part(convolution.staticWeight, axis: 1)
                        ]))
                    try convolution.shardLinears(allToSharded: ["weight_proj"], group: group)
                    convolution.heads /= size
                }
            }
            attention.heads /= size
            attention.kvHeads /= size

            let experts = layer.mlp.experts
            try shardInPlace(
                experts.gateUpProjection, predicate: inkling2ExpertSharding(down: false),
                group: group)
            try shardInPlace(
                experts.downProjection, predicate: inkling2ExpertSharding(down: true),
                group: group)
            layer.mlp.shardingGroup = group
        }
    }
}
