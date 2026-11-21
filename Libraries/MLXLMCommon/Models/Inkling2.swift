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

/// `rms_norm` without a weight, as `mx.fast.rms_norm(x, None, eps)`.
func inkling2UnitRMS(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: eps)
}

private func l2Clip(_ x: MLXArray, scale: MLXArray?, threshold: Float) -> MLXArray {
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
        l2Clip(x, scale: scale, threshold: threshold)
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

    /// Returns the output and the convolution window, from which the caller takes the
    /// next state.
    func callAsFunction(
        _ x: MLXArray, state: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let window = convolutionWindow(x, state: state, mask: mask, kernelSize: kernelSize)
        // As the reference, the convolution runs in the weight's dtype.
        var out = conv1d(window.asType(conv1d.weight.dtype)).asType(.float32)
        out = (out + x.asType(.float32)) / sqrt(Float(kernelSize + 1))
        return (out.asType(x.dtype), window)
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

    func callAsFunction(
        _ x: MLXArray, dynamicInput: MLXArray, state: MLXArray?, mask: MLXArray?
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
        return (out.asType(x.dtype), window)
    }
}

/// Attention of learned queries over the depth sources of each token: the completed
/// blocks and the current partial block, all unit-RMS float32 `[B, L, D]`.
/// Returns `[B, L, Q, D]`.
func inkling2DepthAttention(
    blocks: MLXArray?, partial: MLXArray, query: MLXArray
) -> MLXArray {
    let query = query.asType(.float32).reshaped(-1, partial.dim(-1))
    let scale = pow(Float(partial.dim(-1)), -0.5)
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
        inkling2DepthAttention(blocks: blocks, partial: partial, query: query)
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

    /// `x` and `message` are float32.
    func callAsFunction(_ x: MLXArray, message: MLXArray) -> MLXArray {
        var states = concatenated([inkling2UnitRMS(x), inkling2UnitRMS(message)], axis: -1)
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
        inkling2DepthAttention(
            blocks: bank.blocks, partial: bank.partial, query: query[loopedLayer])
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

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        let scores = sigmoid(logits)
        let biased = scores + correctionBias.asType(.float32)
        if usesGPU(), supportsFusedRouterTopK(biased, k: topK) {
            let (indices, weights) = fusedRouterTopK(
                selection: biased, values: scores, k: topK, normalize: true,
                order: .descending)
            return (indices, weights * routeScale)
        }
        let indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
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
        let outputs = experts(x, indices: indices).asType(.float32)
        var y = weightedExpertSum(outputs, weights)
        if let shardingGroup {
            y = MLXDistributed.allSum(y, group: shardingGroup)
        }
        return y.asType(x.dtype)
    }
}

// MARK: - Attention

/// The caches of one layer visit. `positions` stores each key's position plus one, written
/// through a cache of the same kind as `keyValues`, so it follows every trim, wrap, merge
/// and roll of the keys; a zero marks a slot that holds no key.
struct Inkling2VisitCache {
    let keyValues: KVCache
    let positions: KVCache
    let convolution: ArraysCache
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
        _ convolution: Module, _ x: MLXArray, input: MLXArray, state: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        if let convolution = convolution as? Inkling2ShortConvolution {
            return convolution(x, state: state, mask: mask)
        }
        return (convolution as! Inkling2DynamicConvolution)(
            x, dynamicInput: input, state: state, mask: mask)
    }

    /// `query`, `key` and `value` are the normed inputs of the three projections.
    /// `convolutionStates` holds the key and value states in slots 0 and 1.
    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray, cache: Inkling2VisitCache?,
        convolutionStates: inout [MLXArray?], windows: inout [MLXArray?], mask: MLXArray?
    ) -> MLXArray {
        let (B, L) = (query.dim(0), query.dim(1))

        var keys: MLXArray
        var values: MLXArray
        (keys, windows[0]) = convolve(
            keyConvolution, kProjection(key), input: key, state: convolutionStates[0],
            mask: mask)
        (values, windows[1]) = convolve(
            valueConvolution, vProjection(value), input: value, state: convolutionStates[1],
            mask: mask)

        let gate = gateProjection(query)
        let sinks = sinkProjection(query)
        var queries = queryClip(qProjection(query).reshaped(B, L, heads, -1))
        var relative = relativeClip(rProjection(query).reshaped(B, L, heads, -1))
        keys = l2Clip(keys.reshaped(B, L, kvHeads, -1), scale: nil, threshold: qkClipThreshold)
        values = values.reshaped(B, L, kvHeads, -1)
        values = inkling2UnitRMS(values.asType(.float32), eps: eps).asType(values.dtype)

        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

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

        if !isSliding {
            // Longer contexts scale the queries and relative states up.
            let tau =
                1 + logScalingAlpha
                * log(MLX.maximum((queryPositions + 1) / logScalingFloor, MLXArray(Float(1))))
            let tau4 = tau.reshaped(B, L, 1, 1)
            queries = (queries.asType(.float32) * tau4).asType(queries.dtype)
            relative = (relative.asType(.float32) * tau4).asType(relative.dtype)
        }
        let positionBias = relativeLogits(relative)

        queries = queries.transposed(0, 2, 1, 3)
        let output = attention(
            queries: queries, keys: keys, values: values, positionBias: positionBias,
            sinks: sinks, queryPositions: queryPositions, storedKeyPositions: storedKeyPositions)

        let gated = output.reshaped(B, L, -1).asType(.float32) * sigmoid(gate.asType(.float32))
        return oProjection(gated.asType(gate.dtype))
    }

    /// Each row's positions of the `length` tokens just written, `[B, L]` float32.
    private static func queryPositions(_ cache: KVCache, batch: Int, length: Int) -> MLXArray {
        let steps = MLXArray(Int32(0) ..< Int32(length))[.newAxis]
        switch cache.ropeOffset {
        case .scalar(let offset):
            return broadcast(
                (steps + Int32(offset - length)).asType(.float32), to: [batch, length])
        case .batch(let offsets):
            return (expandedDimensions(offsets.asType(.int32), axis: 1) - Int32(length) + steps)
                .asType(.float32)
        }
    }

    /// Attention with the relative-position bias and a sink per query and head.
    /// `queries` `[B, H, L, D]`; returns `[B, L, H, D]`.
    func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray
    ) -> MLXArray {
        let (B, L) = (queries.dim(0), queries.dim(2))
        if B == 1 && L == 1 {
            return singleTokenAttention(
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
                    positionBias: positionBias[0..., start ..< end], sinks: sinks[0..., start ..< end],
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

    /// One query of one row: the fused attention takes the bias as an additive mask and the
    /// row's sinks.
    private func singleTokenAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, queryPositions: MLXArray, storedKeyPositions: MLXArray
    ) -> MLXArray {
        let R = positionBias.dim(-1)
        let (distance, visible) = distances(
            queryPositions: queryPositions, storedKeyPositions: storedKeyPositions)
        let flatDistance = distance[0, 0]
        var bias = take(
            positionBias[0, 0], clip(flatDistance, min: 0, max: R - 1).asType(.int32), axis: -1)
        bias = which(flatDistance .< Float(R), bias, MLXArray(Float(0)).asType(bias.dtype))
        let mask = which(visible[0, 0], bias, MLXArray(-Float.infinity).asType(bias.dtype))
            .asType(queries.dtype)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask.reshaped(1, heads, 1, -1),
            sinks: sinks[0, 0].asType(queries.dtype))
        return output.transposed(0, 2, 1, 3)
    }
}

// MARK: - Decoder layer

package final class Inkling2DecoderLayer: Module {
    let index: Int
    let isSliding: Bool
    let isBlockBoundary: Bool
    let convKernelSize: Int

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
        convKernelSize = args.convKernelSize
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
        _ x: MLXArray, depth: MLXArray?, messages: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let dtype = x.dtype
        var inputs: (MLXArray, MLXArray, MLXArray)
        if isSliding {
            let h = depth?[.ellipsis, 0, 0...] ?? x
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
                queryMessageGate!(inputs.0, message: messages[.ellipsis, 0, 0...]),
                keyMessageGate!(inputs.1, message: messages[.ellipsis, 1, 0...]),
                valueMessageGate!(inputs.2, message: messages[.ellipsis, 2, 0...])
            )
        }
        if let inputNorm {
            return (
                inputNorm(inputs.0).asType(dtype), inputNorm(inputs.1).asType(dtype),
                inputNorm(inputs.2).asType(dtype)
            )
        }
        return (
            queryInputNorm!(inputs.0).asType(dtype), keyInputNorm!(inputs.1).asType(dtype),
            valueInputNorm!(inputs.2).asType(dtype)
        )
    }

    /// Returns the layer output and the blocks completed so far.
    func callAsFunction(
        _ x: MLXArray, blocks: MLXArray?, cache: Inkling2VisitCache?, mask: MLXArray?,
        messages: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let dtype = x.dtype
        let convolution = cache?.convolution
        var states: [MLXArray?] = (0 ..< 4).map { convolution?[$0] }
        var windows: [MLXArray?] = Array(repeating: nil, count: 4)

        let blockInput = x
        let normedInput = inkling2UnitRMS(blockInput.asType(.float32))
        let depth = attentionInput?(blocks: blocks, partial: normedInput)
        let (q, k, v) = attentionInputs(x, depth: depth, messages: messages)
        var h = attention(
            query: q, key: k, value: v, cache: cache, convolutionStates: &states,
            windows: &windows, mask: mask)
        (h, windows[2]) = postAttentionConvolution(h, state: states[2], mask: mask)

        var blocks = blocks
        var residual = h.asType(.float32)
        if isBlockBoundary {
            let block = normedInput[.newAxis]
            blocks = blocks.map { concatenated([$0, block], axis: 0) } ?? block
        } else {
            residual = blockInput.asType(.float32) + residual
        }

        h = mlpInput(blocks: blocks, partial: inkling2UnitRMS(residual))[.ellipsis, 0, 0...]
        if let messages {
            h = mlpMessageGate!(h, message: messages[.ellipsis, 3, 0...])
        }
        h = mlp(postAttentionNorm(h).asType(dtype))
        (h, windows[3]) = postMLPConvolution(h, state: states[3], mask: mask)

        if let convolution {
            let lengths = convolution.currentLengths
            for (i, window) in windows.enumerated() {
                convolution[i] = convolutionState(
                    window!, lengths: lengths, count: convKernelSize - 1)
            }
            convolution.advance(x.dim(1))
        }
        return ((residual + h.asType(.float32)).asType(dtype), blocks)
    }
}

// MARK: - Text model

/// Why an Inkling 2 cache or model cannot be built.
public enum Inkling2Error: LocalizedError {
    case unsupportedKVCacheStrategy

    public var errorDescription: String? {
        switch self {
        case .unsupportedKVCacheStrategy:
            "Inkling 2 attention reads full-precision keys and values only; "
                + "KV cache quantization and compression are not supported."
        }
    }
}

/// The Inkling 2 text model: token embedding, decoder layers (looped for Inkling 2 Mini),
/// and the final depth attention and norm.
///
/// The caller owns the caches from ``newCache(parameters:)``: three per layer visit (keys
/// and values, key positions, convolution states). Positions come from the caches, so a
/// sequence can be fed in chunks.
public final class Inkling2TextModel: Module {
    package let configuration: Inkling2TextConfiguration
    package let layerVisits: [Int]

    @ModuleInfo(key: "embed_tokens") package var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    package let layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2DepthQuery
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    package static let cachesPerVisit = 3

    package init(_ args: Inkling2TextConfiguration) {
        configuration = args
        layerVisits = args.layerVisits
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
        precondition(
            cache == nil || cache?.count == Self.cachesPerVisit * layerVisits.count,
            "Inkling 2 requires the caches from newCache(parameters:)")
        let dtype = embeddings.dtype
        let caches = cache.map(Self.visitCaches)
        let mask = caches?.first?.convolution.makeMask(N: embeddings.dim(1))

        var visit = 0
        func run(
            _ range: Range<Int>, _ h: MLXArray, _ blocks: MLXArray?,
            bank: Inkling2MessageBank? = nil
        ) -> (MLXArray, MLXArray?) {
            var (h, blocks) = (h, blocks)
            for (i, index) in range.enumerated() {
                let messages = bank.map { messageInput!($0, loopedLayer: i) }
                (h, blocks) = layers[index](
                    h, blocks: blocks, cache: caches?[visit], mask: mask, messages: messages)
                visit += 1
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
                        blocks: completed, partial: inkling2UnitRMS(h.asType(.float32)))
                    (h, blocks) = prelude
                }
                (h, blocks) = run(loop.core, h, blocks, bank: bank)
            }
            (h, blocks) = run(loop.coreEnd ..< configuration.hiddenLayers, h, blocks)
        } else {
            (h, blocks) = run(0 ..< configuration.hiddenLayers, embeddings, nil)
        }

        h = finalInput(blocks: blocks, partial: inkling2UnitRMS(h.asType(.float32)))[
            .ellipsis, 0, 0...]
        return norm(h).asType(dtype)
    }

    private static func visitCaches(_ cache: [KVCache]) -> [Inkling2VisitCache] {
        stride(from: 0, to: cache.count, by: cachesPerVisit).map { i in
            guard let convolution = cache[i + 2] as? ArraysCache else {
                preconditionFailure("Inkling 2 cache \(i + 2) must hold convolution states")
            }
            precondition(
                usesPlainAttentionCacheRoute(cache[i]),
                "Inkling 2 attention reads full-precision keys and values only")
            return Inkling2VisitCache(
                keyValues: cache[i], positions: cache[i + 1], convolution: convolution)
        }
    }

    /// Three caches per layer visit: keys and values (rotating for sliding layers), their
    /// positions in a cache of the same kind, and the four convolution states.
    package func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        if let strategy = try parameters?.resolvedKVCacheConfiguration()?.strategy,
            strategy != .fullPrecision
        {
            throw Inkling2Error.unsupportedKVCacheStrategy
        }
        return try layerVisits.flatMap { index -> [KVCache] in
            func attentionCache() throws -> KVCache {
                try makeHybridAttentionKVCache(
                    parameters: parameters, slidingWindow: configuration.slidingWindow,
                    usesSlidingWindow: configuration.isSliding(index))
            }
            return [try attentionCache(), try attentionCache(), ArraysCache(size: 4)]
        }
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
