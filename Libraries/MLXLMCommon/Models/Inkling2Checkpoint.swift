//
//  Inkling2Checkpoint.swift
//  mlx-swift-lm
//

// Maps Inkling 2 checkpoints onto the module tree of `Inkling2.swift`:
// - Thinking Machines' native checkpoint (`model.llm.*`): port of
//   `mlx_lm/models/inkling2/checkpoint.py` from jundot/mlx-lm@0c4c9919 (MIT).
// - oMLX's oQ checkpoints (`model.layers.N.attn.*`): the inverse of oMLX's
//   `omlx/patches/inkling2/loader.py` (jundot/omlx, Apache-2.0), which fuses some
//   projections and stores the loop controllers and convolutions in its own shapes.
// - mlx-lm conversions (`model.language_model.*`), which already use this layout.

import Foundation
import MLX

package enum Inkling2Checkpoint {
    package enum Layout: Equatable {
        case native
        case oMLX
        case mlxLM
    }

    package static func layout(of names: some Collection<String>) -> Layout {
        if names.contains(where: { $0.hasPrefix("model.llm.") }) { return .native }
        if names.contains(where: { $0.hasPrefix("model.layers.") }) { return .oMLX }
        return .mlxLM
    }

    /// Renames, splits and casts the checkpoint's tensors and quantization settings for the
    /// model, and keeps only its text model, its `lm_head` and, if asked, its vision tower.
    package static func prepare(
        _ checkpoint: ModelCheckpoint, configuration: Inkling2Configuration, includeVision: Bool
    ) throws -> ModelCheckpoint {
        let text = configuration.text
        var checkpoint = checkpoint
        switch layout(of: checkpoint.weights.keys) {
        case .native:
            checkpoint.weights = try nativeWeights(checkpoint.weights, text: text)
        case .oMLX:
            checkpoint.weights = try oMLXWeights(checkpoint.weights, text: text)
            checkpoint.perLayerQuantization = checkpoint.perLayerQuantization.map {
                oMLXQuantization($0, text: text)
            }
        case .mlxLM:
            break
        }

        func keeps(_ name: String) -> Bool {
            name.hasPrefix("model.language_model.") || name.hasPrefix("lm_head.")
                || name == "lm_head"
                || (includeVision && name.hasPrefix("model.vision_tower."))
        }
        checkpoint = try checkpoint.mapNames { keeps($0) ? $0 : nil }

        var weights = checkpoint.weights
        for (name, value) in weights {
            if name.hasSuffix(".query") || name.hasSuffix(".mlp.gate.weight")
                || name.hasSuffix(".mlp.gate.e_score_correction_bias")
                || name.hasSuffix(".forget_gate_proj.weight")
                || name.hasSuffix(".message_gate_proj.weight")
            {
                weights[name] = value.asType(.float32)
            }
        }
        checkpoint.weights = weights
        try quantizeExperts(&checkpoint, text: text)
        return checkpoint
    }

    // MARK: - Experts in NVIDIA formats

    /// Expert projections with `uint8` scales are NVFP4, MXFP4 or MXFP8; the group size
    /// and bits follow from the shapes. NVFP4 experts get a unit global scale when the
    /// checkpoint has none.
    private static func quantizeExperts(
        _ checkpoint: inout ModelCheckpoint, text: Inkling2TextConfiguration
    ) throws {
        var settings = checkpoint.perLayerQuantization?.perLayerQuantization ?? [:]
        var weights = checkpoint.weights
        for (name, scales) in checkpoint.weights
        where name.hasSuffix(".scales") && scales.dtype == .uint8 {
            let module = String(name.dropLast(".scales".count))
            let inputs: Int
            if module.hasSuffix(".mlp.experts.gate_up_proj") {
                inputs = text.hiddenSize
            } else if module.hasSuffix(".mlp.experts.down_proj") {
                inputs = text.intermediateSize
            } else {
                throw ModelFactoryError.invalidConfiguration(
                    "Inkling 2 tensor \(name) has uint8 scales outside the routed experts")
            }
            guard let weight = weights[module + ".weight"], scales.dim(-1) > 0,
                inputs % scales.dim(-1) == 0
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "Inkling 2 experts \(module) have scales that do not match their inputs")
            }
            let groupSize = inputs / scales.dim(-1)
            let bits = weight.dim(-1) * 32 / inputs
            let mode: QuantizationMode
            switch (bits, groupSize) {
            case (4, 16): mode = .nvfp4
            case (4, 32): mode = .mxfp4
            case (8, 32): mode = .mxfp8
            default:
                throw ModelFactoryError.invalidConfiguration(
                    "Inkling 2 experts \(module) use an unknown format: "
                        + "\(bits) bits in groups of \(groupSize)")
            }
            settings[module] = .quantize(
                .init(groupSize: groupSize, bits: bits, mode: mode))
            if mode == .nvfp4, weights[module + ".global_scale"] == nil {
                weights[module + ".global_scale"] =
                    MLXArray.ones([weight.dim(0)], dtype: .float32)
                    * Inkling2ScaledSwitchLinear.unitScale
            }
        }
        checkpoint.weights = weights
        if !settings.isEmpty {
            checkpoint.perLayerQuantization = .init(
                quantization: checkpoint.perLayerQuantization?.quantization,
                perLayerQuantization: settings)
        }
    }

    // MARK: - Native

    private static let nativeRenames: [(NSRegularExpression, String)] = [
        (#"^model\.llm\.unembed\."#, "lm_head."),
        (#"^model\.llm\.embed\."#, "model.language_model.embed_tokens."),
        (#"^model\.llm\."#, "model.language_model."),
        (#"\.attn_res\.mlp_queries\.(\d+)$"#, ".layers.$1.mlp_input.query"),
        (#"\.attn_res\.output_query$"#, ".final_input.query"),
        (#"\.attn\.wq_du\."#, ".self_attn.q_proj."),
        (#"\.attn\.wk_dv\."#, ".self_attn.k_proj."),
        (#"\.attn\.wv_dv\."#, ".self_attn.v_proj."),
        (#"\.attn\.wr_du\."#, ".self_attn.r_proj."),
        (#"\.attn\.wo_ud\."#, ".self_attn.o_proj."),
        (#"\.attn\.gate_attn\."#, ".self_attn.gate_proj."),
        (#"\.attn\.attn_sink_proj\."#, ".self_attn.sink_proj."),
        (#"\.controllers\.attn\."#, ".q_message_gate."),
        (#"\.controllers\.(k|v|mlp)\."#, ".$1_message_gate."),
        (#"_message_gate\.down_weight$"#, "_message_gate.down_proj.weight"),
        (#"_message_gate\.forget_weight$"#, "_message_gate.forget_gate_proj.weight"),
        (#"_message_gate\.gain_weight$"#, "_message_gate.message_gate_proj.weight"),
        (#"\.attn\."#, ".self_attn."),
        (#"\.attn_sconv\."#, ".post_attention_convolution."),
        (#"\.mlp_sconv\."#, ".post_mlp_convolution."),
        (#"sconv\.weight$"#, "sconv.conv1d.weight"),
        (#"convolution\.weight$"#, "convolution.conv1d.weight"),
        (#"sconv\.static_w$"#, "sconv.static_weight"),
        (#"sconv\.weight_proj$"#, "sconv.weight_proj.weight"),
        (#"\.mlp_norm\."#, ".post_attention_layernorm."),
        (#"\.attn_norm\."#, ".input_layernorm."),
        (#"\.(q|k|v)_input_norm\."#, ".$1_input_layernorm."),
        (#"\.mlp\.gate\.bias$"#, ".mlp.gate.e_score_correction_bias"),
        (#"^model\.visual\.backbone\.ln_pre\."#, "model.vision_tower.pre_layernorm."),
        (#"^model\.visual\.backbone\."#, "model.vision_tower."),
        (#"^model\.visual\.proj_in\."#, "model.vision_tower.merger.linear_1."),
        (#"^model\.visual\.proj_out\."#, "model.vision_tower.merger.linear_2."),
        (#"embeddings\.position_embedding$"#, "embeddings.position_embedding.weight"),
        (#"(vision_tower\.layers\.\d+)\.ln_1\."#, "$1.layernorm_before."),
        (#"(vision_tower\.layers\.\d+)\.ln_2\."#, "$1.layernorm_after."),
        (#"(vision_tower\.layers\.\d+)\.mlp\.c_fc\."#, "$1.mlp.fc1."),
        (#"(vision_tower\.layers\.\d+)\.mlp\.c_proj\."#, "$1.mlp.fc2."),
        (#"(vision_tower\.layers\.\d+)\.ls_(\d)\.gamma$"#, "$1.lambda_$2"),
    ].map { (try! NSRegularExpression(pattern: $0.0), $0.1) }

    private static let attentionQuery = try! NSRegularExpression(
        pattern: #"^model\.llm\.attn_res\.attn_queries\.(\d+)$"#)
    private static let nativeExpert = try! NSRegularExpression(
        pattern: #"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$"#)

    private static func groups(_ regex: NSRegularExpression, _ name: String) -> [String]? {
        let range = NSRange(name.startIndex..., in: name)
        guard let match = regex.firstMatch(in: name, range: range) else { return nil }
        return (1 ..< match.numberOfRanges).map {
            Range(match.range(at: $0), in: name).map { String(name[$0]) } ?? ""
        }
    }

    package static func nativeName(_ name: String) -> String {
        if let index = groups(attentionQuery, name)?.first.flatMap({ Int($0) }) {
            // Layer 0 has no attention input, so query i belongs to layer i + 1.
            return "model.language_model.layers.\(index + 1).attention_input.query"
        }
        var name = name
        for (regex, template) in nativeRenames {
            name = regex.stringByReplacingMatches(
                in: name, range: NSRange(name.startIndex..., in: name), withTemplate: template)
        }
        return name
    }

    private static func nativeWeights(
        _ weights: [String: MLXArray], text: Inkling2TextConfiguration
    ) throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        var amax = [String: MLXArray]()
        var denominator = [String: MLXArray]()
        var messageQueries = [String: MLXArray]()
        for (original, value) in weights {
            guard original.hasPrefix("model.llm.") || original.hasPrefix("model.visual.") else {
                continue
            }
            let name = nativeName(original)
            var value = value
            if name.hasSuffix("conv1d.weight") {
                // PyTorch stores the depthwise convolution as [C, 1, K]; MLX as [C, K, 1].
                value = contiguous(value.swappedAxes(1, 2))
            }
            if name.hasSuffix("cross_pass_queries") {
                messageQueries["q"] = value
                continue
            }
            if name.hasSuffix("cross_pass_kv_queries") {
                messageQueries["kv"] = value
                continue
            }
            if let parts = groups(nativeExpert, name) {
                let module = "\(parts[0]).\(parts[1] == "w13" ? "gate_up_proj" : "down_proj")"
                switch parts[2] {
                case "_amax": amax[module] = value
                case "_outer_scale_denominator": denominator[module] = value
                case "_scale": result[module + ".scales"] = value
                case "":
                    // Packed NVFP4 and MXFP8 codes are stored as bytes.
                    result[module + ".weight"] =
                        value.dtype == .uint8 ? value.view(dtype: .uint32) : value
                default:
                    throw ModelFactoryError.invalidConfiguration(
                        "Unknown Inkling 2 expert tensor \(original)")
                }
                continue
            }
            result[name] = value
        }

        if let q = messageQueries["q"], let kv = messageQueries["kv"] {
            // Per looped layer the checkpoint has [query, mlp] and [key, value] rows.
            let dims = q.dim(-1)
            let q = q.reshaped(-1, 2, dims)
            let kv = kv.reshaped(-1, 2, dims)
            result["model.language_model.message_input.query"] = concatenated(
                [q[0..., ..<1], kv, q[0..., 1...]], axis: 1)
        }
        for (module, amax) in amax {
            guard let denominator = denominator[module] else { continue }
            let scale = Inkling2ScaledSwitchLinear.unitScale / denominator.asType(.float32)
            result[module + ".global_scale"] = amax.asType(.float32) * scale
        }
        return result
    }

    // MARK: - oMLX

    private static let oMLXLayer = try! NSRegularExpression(
        pattern: #"^model\.layers\.(\d+)\.(.+)$"#)

    private struct Splits {
        let names: [String]
        let sizes: [Int]
    }

    /// The projections oMLX fuses, and their output rows in order.
    private static func fusedSplits(
        layer: Int, module: String, text: Inkling2TextConfiguration
    ) -> Splits? {
        let K = text.convKernelSize
        if text.isSliding(layer) {
            guard module == "attn.in_proj" else { return nil }
            let (h, kv, d) = (text.slidingAttentionHeads, text.slidingKVHeads, text.slidingHeadDim)
            return Splits(
                names: ["q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj"].map {
                    "self_attn.\($0)"
                },
                sizes: [h * d, kv * d, kv * d, h * text.relativeDim, h * d, h])
        }
        let (h, kv, d) = (text.attentionHeads, text.kvHeads, text.headDim)
        switch module {
        case "attn.q_proj":
            return Splits(
                names: ["q_proj", "r_proj", "gate_proj", "sink_proj"].map { "self_attn.\($0)" },
                sizes: [h * d, h * text.relativeDim, h * d, h])
        case "attn.k_proj", "attn.v_proj":
            let kind = module == "attn.k_proj" ? "k" : "v"
            return Splits(
                names: ["self_attn.\(kind)_proj", "self_attn.\(kind)_sconv.weight_proj"],
                sizes: [kv * d, kv * K])
        default:
            return nil
        }
    }

    private static let oMLXRenames: [String: String] = [
        "attn.o_proj": "self_attn.o_proj",
        "attn.k_static": "self_attn.k_sconv.static_weight",
        "attn.v_static": "self_attn.v_sconv.static_weight",
        "attn.k_conv": "self_attn.k_sconv.conv1d.weight",
        "attn.v_conv": "self_attn.v_sconv.conv1d.weight",
        "attn.q_scale": "self_attn.q_l2_clip.scale",
        "attn.r_scale": "self_attn.rel_l2_clip.scale",
        "attn.rel_proj": "self_attn.rel_logits_proj.proj",
        "attn_conv": "post_attention_convolution.conv1d.weight",
        "mlp_conv": "post_mlp_convolution.conv1d.weight",
        "attn_input.query": "attention_input.query",
        "mlp_input.query": "mlp_input.query",
        "attn_norm": "input_layernorm",
        "mlp_norm": "post_attention_layernorm",
        "q_input_norm": "q_input_layernorm",
        "k_input_norm": "k_input_layernorm",
        "v_input_norm": "v_input_layernorm",
        "mlp.router_weight": "mlp.gate.weight",
        "mlp.router_bias": "mlp.gate.e_score_correction_bias",
        "mlp.experts.w13": "mlp.experts.gate_up_proj",
        "mlp.experts.w2": "mlp.experts.down_proj",
    ]

    /// The model's name for an oMLX module or tensor path outside the layers, or within a
    /// layer when it is not fused.
    private static func oMLXName(_ name: String) -> String? {
        for (source, target) in [
            ("model.embed.", "model.language_model.embed_tokens."),
            ("model.embed_norm.", "model.language_model.embed_norm."),
            ("model.norm.", "model.language_model.norm."),
            ("model.final_input.", "model.language_model.final_input."),
        ] where name.hasPrefix(source) {
            return target + name.dropFirst(source.count)
        }
        if name == "model.embed" { return "model.language_model.embed_tokens" }
        if name == "lm_head" || name.hasPrefix("lm_head.") { return name }
        if name.hasPrefix("vision_tower.") {
            return nativeName("model.visual." + name.dropFirst("vision_tower.".count))
        }
        guard let parts = groups(oMLXLayer, name) else { return nil }
        let rest = parts[1]
        let prefix = "model.language_model.layers.\(parts[0])."
        for (source, target) in oMLXRenames {
            if rest == source { return prefix + target }
            if rest.hasPrefix(source + ".") {
                return prefix + target + rest.dropFirst(source.count)
            }
        }
        return nil
    }

    private static func oMLXWeights(
        _ weights: [String: MLXArray], text: Inkling2TextConfiguration
    ) throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        var messageQueries = [Int: MLXArray]()
        let hidden = text.hiddenSize
        for (name, value) in weights {
            if let parts = groups(oMLXLayer, name), let layer = Int(parts[0]) {
                let rest = parts[1]
                let prefix = "model.language_model.layers.\(layer)."
                let suffixStart = rest.lastIndex(of: ".") ?? rest.endIndex
                let module = String(rest[..<suffixStart])
                if let splits = fusedSplits(layer: layer, module: module, text: text) {
                    let leaf = rest[suffixStart...]
                    guard value.dim(0) == splits.sizes.reduce(0, +) else {
                        throw ModelFactoryError.invalidConfiguration(
                            "Inkling 2 tensor \(name) has \(value.dim(0)) rows, expected "
                                + "\(splits.sizes.reduce(0, +))")
                    }
                    let bounds = Array(splits.sizes.dropLast()).reduce(into: [Int]()) {
                        $0.append(($0.last ?? 0) + $1)
                    }
                    for (target, part) in zip(splits.names, value.split(indices: bounds, axis: 0)) {
                        result[prefix + target + leaf] = contiguous(part)
                    }
                    continue
                }
                switch rest {
                case "loop_down":
                    // [4, 2, h, r]: the x and message halves per gate. The reference rounds
                    // the down projection to bf16, and these are bf16 values widened.
                    for (i, gate) in ["q", "k", "v", "mlp"].enumerated() {
                        let down = value[i].transposed(2, 0, 1).reshaped(-1, 2 * hidden)
                        result[prefix + "\(gate)_message_gate.down_proj.weight"] =
                            contiguous(down.asType(.bfloat16))
                    }
                    continue
                case "loop_gain_forget":
                    // [4, r, 2h]: gain then forget.
                    for (i, gate) in ["q", "k", "v", "mlp"].enumerated() {
                        let gains = value[i]
                        result[prefix + "\(gate)_message_gate.message_gate_proj.weight"] =
                            contiguous(gains[0..., ..<hidden].T)
                        result[prefix + "\(gate)_message_gate.forget_gate_proj.weight"] =
                            contiguous(gains[0..., hidden...].T)
                    }
                    continue
                case "loop_message.query":
                    messageQueries[layer] = value
                    continue
                default:
                    break
                }
                guard let target = oMLXName(name) else {
                    throw ModelFactoryError.invalidConfiguration(
                        "Unknown Inkling 2 tensor \(name)")
                }
                result[target] = oMLXShape(target, value)
                continue
            }
            if let target = oMLXName(name) {
                result[target] = oMLXShape(target, value)
            }
        }
        if !messageQueries.isEmpty, let loop = text.loop {
            let rows = try loop.core.map { layer -> MLXArray in
                guard let query = messageQueries[layer] else {
                    throw ModelFactoryError.invalidConfiguration(
                        "Inkling 2 checkpoint has no loop messages for layer \(layer)")
                }
                return query
            }
            result["model.language_model.message_input.query"] = stacked(rows, axis: 0)
        }
        return result
    }

    /// oMLX keeps convolutions as [C, K] and single queries as [1, D].
    private static func oMLXShape(_ name: String, _ value: MLXArray) -> MLXArray {
        if name.hasSuffix("conv1d.weight"), value.ndim == 2 {
            return expandedDimensions(value, axis: -1)
        }
        if name.hasSuffix(".query"), value.ndim == 2, value.dim(0) == 1 {
            return value.reshaped(-1)
        }
        return value
    }

    /// The quantization settings under the model's names. A fused projection's setting
    /// applies to each of its parts.
    private static func oMLXQuantization(
        _ quantization: BaseConfiguration.PerLayerQuantization, text: Inkling2TextConfiguration
    ) -> BaseConfiguration.PerLayerQuantization {
        var settings = [String: BaseConfiguration.QuantizationOption]()
        for (path, option) in quantization.perLayerQuantization {
            if let parts = groups(oMLXLayer, path), let layer = Int(parts[0]),
                let splits = fusedSplits(layer: layer, module: parts[1], text: text)
            {
                for target in splits.names {
                    settings["model.language_model.layers.\(layer).\(target)"] = option
                }
            } else if let target = oMLXName(path) {
                settings[target] = option
            }
        }
        return .init(quantization: quantization.quantization, perLayerQuantization: settings)
    }
}
