//
//  Inkling2Configuration.swift
//  mlx-swift-lm
//

// Port of `mlx_lm/models/inkling2/config.py` from jundot/mlx-lm@0c4c9919 (MIT), the draft
// mlx-lm support for Thinking Machines' Inkling 2 (ml-explore/mlx-lm#2061, #2086).

import Foundation

/// The configuration of Inkling 2 (`inkling2`) and its looped variant Inkling 2 Mini
/// (`inkling2_loop`), decoded from `config.json`.
public struct Inkling2Configuration: Codable, Sendable {
    public let modelType: String
    public let text: Inkling2TextConfiguration
    public let vision: Inkling2VisionConfiguration?
    public let imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case imageTokenId = "image_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "inkling2"
        text = try container.decode(Inkling2TextConfiguration.self, forKey: .text)
        vision = try container.decodeIfPresent(Inkling2VisionConfiguration.self, forKey: .vision)
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200_054
    }
}

/// The loop of Inkling 2 Mini: layers `coreStart ..< coreEnd` run `passes` times per token.
public struct Inkling2LoopConfiguration: Codable, Sendable {
    public let coreStart: Int
    public let coreEnd: Int
    public let passes: Int
    public let gateRank: Int
    public let isVariableLoop: Bool

    enum CodingKeys: String, CodingKey {
        case coreStart = "core_start"
        case coreEnd = "core_end"
        case passes = "num_passes"
        case gateRank = "gate_rank"
        case isVariableLoop = "is_varloop"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        coreStart = try container.decode(Int.self, forKey: .coreStart)
        coreEnd = try container.decode(Int.self, forKey: .coreEnd)
        passes = try container.decode(Int.self, forKey: .passes)
        gateRank = try container.decode(Int.self, forKey: .gateRank)
        isVariableLoop = try container.decodeIfPresent(Bool.self, forKey: .isVariableLoop) ?? false
        guard !isVariableLoop else {
            throw DecodingError.dataCorruptedError(
                forKey: .isVariableLoop, in: container,
                debugDescription: "Inkling 2 variable loops are not supported")
        }
        guard coreStart >= 0, coreEnd > coreStart, passes >= 1, gateRank > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .coreEnd, in: container,
                debugDescription: "Inkling 2 loop_config needs 0 <= core_start < core_end, "
                    + "num_passes >= 1 and gate_rank > 0")
        }
    }

    public var core: Range<Int> { coreStart ..< coreEnd }
}

/// Experimental attention-history storage for Inkling 2's repeated layers.
public enum Inkling2RecurrentKVCacheMode: String, Codable, Sendable {
    case perLoop = "per_loop"
    /// Earlier passes read the final pass's history. Outputs can depend on prefill chunking.
    case lastLoop = "last_loop"
}

public struct Inkling2TextConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let vocabularySize: Int
    public let unpaddedVocabularySize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let slidingAttentionHeads: Int
    public let slidingKVHeads: Int
    public let slidingHeadDim: Int
    public let slidingWindow: Int
    public let localLayerIds: Set<Int>
    public let relativeDim: Int
    public let relativeExtent: Int
    public let logScalingFloor: Int
    public let logScalingAlpha: Float
    public let rmsNormEps: Float
    public let qkClipThreshold: Float
    public let relativeClipThreshold: Float
    public let convKernelSize: Int
    public let layersPerResidualBlock: Int
    public let routedExperts: Int
    public let expertsPerToken: Int
    public let intermediateSize: Int
    public let gateSoftcap: Float
    public let upSoftcap: Float
    public let routeScale: Float
    public let finalLogitSoftcap: Float
    public let loop: Inkling2LoopConfiguration?
    public let recurrentKVCacheMode: Inkling2RecurrentKVCacheMode

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingAttentionHeads = "swa_num_attention_heads"
        case slidingKVHeads = "swa_num_key_value_heads"
        case slidingHeadDim = "swa_head_dim"
        case slidingWindow = "sliding_window_size"
        case localLayerIds = "local_layer_ids"
        case relativeDim = "d_rel"
        case relativeExtent = "rel_extent"
        case logScalingFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps"
        case qkClipThreshold = "qk_l2_clip_threshold"
        case relativeClipThreshold = "rel_l2_clip_threshold"
        case convKernelSize = "sconv_kernel_size"
        case layersPerResidualBlock = "n_layers_per_attn_res_block"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case intermediateSize = "intermediate_size"
        case gateSoftcap = "mlp_gate_softcap"
        case upSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case finalLogitSoftcap = "final_logit_softcapping"
        case loop = "loop_config"
        case recurrentKVCacheMode = "recurrent_kv_cache_mode"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        unpaddedVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabularySize) ?? vocabularySize
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        slidingAttentionHeads = try c.decode(Int.self, forKey: .slidingAttentionHeads)
        slidingKVHeads = try c.decode(Int.self, forKey: .slidingKVHeads)
        slidingHeadDim = try c.decode(Int.self, forKey: .slidingHeadDim)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        localLayerIds = Set(try c.decode([Int].self, forKey: .localLayerIds))
        relativeDim = try c.decode(Int.self, forKey: .relativeDim)
        relativeExtent = try c.decode(Int.self, forKey: .relativeExtent)
        logScalingFloor = try c.decode(Int.self, forKey: .logScalingFloor)
        logScalingAlpha = try c.decode(Float.self, forKey: .logScalingAlpha)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        qkClipThreshold = try c.decode(Float.self, forKey: .qkClipThreshold)
        relativeClipThreshold = try c.decode(Float.self, forKey: .relativeClipThreshold)
        convKernelSize = try c.decode(Int.self, forKey: .convKernelSize)
        layersPerResidualBlock = try c.decode(Int.self, forKey: .layersPerResidualBlock)
        routedExperts = try c.decode(Int.self, forKey: .routedExperts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        gateSoftcap = try c.decode(Float.self, forKey: .gateSoftcap)
        upSoftcap = try c.decode(Float.self, forKey: .upSoftcap)
        routeScale = try c.decode(Float.self, forKey: .routeScale)
        finalLogitSoftcap = try c.decode(Float.self, forKey: .finalLogitSoftcap)
        loop = try c.decodeIfPresent(Inkling2LoopConfiguration.self, forKey: .loop)
        recurrentKVCacheMode =
            try c.decodeIfPresent(Inkling2RecurrentKVCacheMode.self, forKey: .recurrentKVCacheMode)
            ?? .perLoop

        for (key, value) in [
            (CodingKeys.hiddenSize, hiddenSize), (.hiddenLayers, hiddenLayers),
            (.attentionHeads, attentionHeads), (.kvHeads, kvHeads), (.headDim, headDim),
            (.slidingAttentionHeads, slidingAttentionHeads), (.slidingKVHeads, slidingKVHeads),
            (.slidingHeadDim, slidingHeadDim), (.slidingWindow, slidingWindow),
            (.relativeDim, relativeDim), (.relativeExtent, relativeExtent),
            (.logScalingFloor, logScalingFloor), (.convKernelSize, convKernelSize),
            (.layersPerResidualBlock, layersPerResidualBlock), (.routedExperts, routedExperts),
            (.expertsPerToken, expertsPerToken), (.intermediateSize, intermediateSize),
        ] where value <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: c, debugDescription: "\(key.rawValue) must be positive")
        }
        guard attentionHeads % kvHeads == 0, slidingAttentionHeads % slidingKVHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads, in: c,
                debugDescription: "attention heads must be a multiple of key-value heads")
        }
        guard expertsPerToken <= routedExperts else {
            throw DecodingError.dataCorruptedError(
                forKey: .expertsPerToken, in: c,
                debugDescription: "num_experts_per_tok exceeds n_routed_experts")
        }
        guard unpaddedVocabularySize <= vocabularySize else {
            throw DecodingError.dataCorruptedError(
                forKey: .unpaddedVocabularySize, in: c,
                debugDescription: "unpadded_vocab_size exceeds vocab_size")
        }
        if let loop, loop.coreEnd > hiddenLayers {
            throw DecodingError.dataCorruptedError(
                forKey: .loop, in: c, debugDescription: "loop_config core_end exceeds the layers")
        }
    }

    public func isSliding(_ layer: Int) -> Bool { localLayerIds.contains(layer) }

    /// The layers in the order they run for one token: the loop's core repeats.
    public var layerVisits: [Int] {
        guard let loop else { return Array(0 ..< hiddenLayers) }
        return Array(0 ..< loop.coreStart)
            + Array(repeating: Array(loop.core), count: loop.passes).flatMap { $0 }
            + Array(loop.coreEnd ..< hiddenLayers)
    }
}

public struct Inkling2VisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let channels: Int
    public let patchSize: Int
    public let mergeSize: Int
    public let positionGridSize: Int
    public let projectionHiddenDim: Int
    public let projectorBias: Bool
    public let mlpRatio: Float
    public let layerNormEps: Float
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case channels = "num_channels"
        case patchSize = "native_patch_size"
        case mergeSize = "adapter_merge_size"
        case positionGridSize = "posemb_grid_size"
        case projectionHiddenDim = "projection_hidden_dim"
        case projectorBias = "projector_bias"
        case mlpRatio = "mlp_ratio"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 23
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 20
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        positionGridSize = try c.decodeIfPresent(Int.self, forKey: .positionGridSize) ?? 32
        projectionHiddenDim =
            try c.decodeIfPresent(Int.self, forKey: .projectionHiddenDim) ?? 4096
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        guard hiddenSize > 0, attentionHeads > 0, hiddenSize % attentionHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .attentionHeads, in: c,
                debugDescription: "vision hidden_size must be a multiple of num_attention_heads")
        }
    }

    public var headDim: Int { hiddenSize / attentionHeads }
    public var intermediateSize: Int { Int(Float(hiddenSize) * mlpRatio) }
}
