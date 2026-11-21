// Copyright © 2026 Apple Inc.

// Tiny random Inkling 2 models, after `tests/models/test_inkling2.py` of the draft mlx-lm
// port (jundot/mlx-lm@0c4c9919).

import Foundation
import MLX
import MLXNN
import Testing
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

private func inkling2ConfigJSON(loop: Bool, gateRank: Int = 8, vision: Bool = false) -> String {
    let visionConfig =
        vision
        ? """
        "vision_config": {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
            "native_patch_size": 2, "adapter_merge_size": 2, "posemb_grid_size": 4,
            "projection_hidden_dim": 48},
        """ : ""
    let loopConfig =
        loop
        ? """
        , "loop_config": {"core_start": 3, "core_end": 6, "num_passes": 3,
            "gate_rank": \(gateRank), "is_varloop": false}
        """ : ""
    return """
        {
            "model_type": "\(loop ? "inkling2_loop" : "inkling2")",
            "image_token_id": 127, \(visionConfig)
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
                "final_logit_softcapping": 30.0\(loopConfig)
            }
        }
        """
}

private let unpaddedVocabulary = 120

private func inkling2Configuration(loop: Bool, gateRank: Int = 8, vision: Bool = false) throws
    -> Inkling2Configuration
{
    try JSONDecoder().decode(
        Inkling2Configuration.self,
        from: Data(inkling2ConfigJSON(loop: loop, gateRank: gateRank, vision: vision).utf8))
}

/// Random weights at the reference's scale: norms, clip and layer scales near one, other
/// weights normal over the square root of their fan-in.
private func randomize(_ model: Module, seed: UInt64) {
    MLXRandom.seed(seed)
    let weights = model.parameters().flattened().map { name, value -> (String, MLXArray) in
        if name.hasSuffix("norm.weight") || name.hasSuffix("l2_clip.scale")
            || name.hasSuffix("lambda_1") || name.hasSuffix("lambda_2")
        {
            return (name, 1 + 0.1 * MLXRandom.normal(value.shape))
        }
        let fanIn = value.ndim > 1 ? value.dim(-1) : 16
        return (name, MLXRandom.normal(value.shape) * pow(Float(fanIn), -0.5))
    }
    model.update(parameters: ModuleParameters.unflattened(weights))
    eval(model)
}

/// A seeded text model with random weights.
private func makeInkling2(loop: Bool = false, gateRank: Int = 8, seed: UInt64 = 0) throws
    -> Inkling2Model
{
    let model = Inkling2Model(try inkling2Configuration(loop: loop, gateRank: gateRank))
    randomize(model, seed: seed)
    return model
}

/// A seeded model with a vision tower and random weights.
private func makeInkling2VLM(loop: Bool = true, seed: UInt64 = 0) throws -> Inkling2VLM {
    let model = Inkling2VLM(try inkling2Configuration(loop: loop, vision: true))
    randomize(model, seed: seed)
    return model
}

/// NVFP4 experts with random global scales, and MXFP8 experts in the last layer, as in
/// Thinking Machines' checkpoint.
private func quantizeExpertsNatively(_ model: Module) {
    quantize(model: model) { path, _ in
        guard path.contains(".mlp.experts.") else { return nil }
        return path.contains(".layers.7.") ? (32, 8, .mxfp8) : (16, 4, .nvfp4)
    }
    let scales = model.parameters().flattened().compactMap { name, value in
        name.hasSuffix(".global_scale")
            ? (name, MLXRandom.uniform(Float(0.5) ..< 1.5, value.shape) * 2688) : nil
    }
    model.update(parameters: ModuleParameters.unflattened(scales))
    eval(model)
}

/// Affine experts and 8-bit attention, embeddings and head, as in oMLX's oQ builds.
private func quantizeLikeOQ(_ model: Module, expertBits: Int) {
    quantize(model: model) { path, module in
        if path.contains(".mlp.experts.") { return (64, expertBits, .affine) }
        if path.contains("_message_gate.") { return nil }
        // oQ keeps the vision tower in bf16.
        if path.hasPrefix("model.vision_tower.") { return nil }
        if path.contains(".self_attn.") || path.hasSuffix("embed_tokens") || path == "lm_head" {
            return module is Quantizable ? (64, 8, .affine) : nil
        }
        return nil
    }
    // oMLX keeps the loop controllers as float32 copies of bf16 weights.
    let downs = model.parameters().flattened().compactMap { name, value in
        name.hasSuffix("_message_gate.down_proj.weight") ? (name, value.asType(.bfloat16)) : nil
    }
    model.update(parameters: ModuleParameters.unflattened(downs))
    eval(model)
}

private func tokens(_ shape: [Int], seed: UInt64 = 1) -> MLXArray {
    MLXRandom.seed(seed)
    return MLXRandom.randInt(Int32(0) ..< Int32(unpaddedVocabulary), shape)
}

private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

/// Logits for `tokens` fed in chunks of six, then one at a time.
private func cachedLogits(_ model: Inkling2Model, _ tokens: MLXArray) throws -> MLXArray {
    let cache = try model.newCache(parameters: nil)
    var steps = [
        model(tokens[0..., ..<6], cache: cache), model(tokens[0..., 6 ..< 12], cache: cache),
    ]
    for i in 12 ..< tokens.dim(1) {
        steps.append(model(tokens[0..., i ..< (i + 1)], cache: cache))
    }
    return concatenated(steps, axis: 1)
}

// MARK: - Checkpoint layouts

private func rename(_ name: String, _ rules: [(String, String)]) -> String {
    var name = name
    for (pattern, template) in rules {
        let regex = try! NSRegularExpression(pattern: pattern)
        name = regex.stringByReplacingMatches(
            in: name, range: NSRange(name.startIndex..., in: name), withTemplate: template)
    }
    return name
}

/// The reverse of the native renames, from `test_inkling2.py`.
private let nativeNames: [(String, String)] = [
    (#"^lm_head\."#, "model.llm.unembed."),
    (#"^model\.language_model\.embed_tokens\."#, "model.llm.embed."),
    (#"^model\.language_model\."#, "model.llm."),
    (#"\.final_input\.query$"#, ".attn_res.output_query"),
    (#"\.layers\.(\d+)\.mlp_input\.query$"#, ".attn_res.mlp_queries.$1"),
    (#"\.self_attn\.q_proj\."#, ".attn.wq_du."),
    (#"\.self_attn\.k_proj\."#, ".attn.wk_dv."),
    (#"\.self_attn\.v_proj\."#, ".attn.wv_dv."),
    (#"\.self_attn\.r_proj\."#, ".attn.wr_du."),
    (#"\.self_attn\.o_proj\."#, ".attn.wo_ud."),
    (#"\.self_attn\.gate_proj\."#, ".attn.gate_attn."),
    (#"\.self_attn\.sink_proj\."#, ".attn.attn_sink_proj."),
    (#"\.self_attn\."#, ".attn."),
    (#"\.q_message_gate\."#, ".controllers.attn."),
    (#"\.(k|v|mlp)_message_gate\."#, ".controllers.$1."),
    (#"(\.controllers\.\w+)\.down_proj\.weight$"#, "$1.down_weight"),
    (#"(\.controllers\.\w+)\.forget_gate_proj\.weight$"#, "$1.forget_weight"),
    (#"(\.controllers\.\w+)\.message_gate_proj\.weight$"#, "$1.gain_weight"),
    (#"\.post_attention_convolution\."#, ".attn_sconv."),
    (#"\.post_mlp_convolution\."#, ".mlp_sconv."),
    (#"\.conv1d\.weight$"#, ".weight"),
    (#"sconv\.static_weight$"#, "sconv.static_w"),
    (#"sconv\.weight_proj\.weight$"#, "sconv.weight_proj"),
    (#"\.post_attention_layernorm\."#, ".mlp_norm."),
    (#"\.input_layernorm\."#, ".attn_norm."),
    (#"\.(q|k|v)_input_layernorm\."#, ".$1_input_norm."),
    (#"\.mlp\.gate\.e_score_correction_bias$"#, ".mlp.gate.bias"),
]

/// The reverse of the native vision renames. Vision names keep `self_attn.q_proj`.
private let nativeVisionNames: [(String, String)] = [
    (#"^model\.vision_tower\.pre_layernorm\."#, "model.visual.backbone.ln_pre."),
    (#"^model\.vision_tower\.merger\.linear_1\."#, "model.visual.proj_in."),
    (#"^model\.vision_tower\.merger\.linear_2\."#, "model.visual.proj_out."),
    (#"^model\.vision_tower\."#, "model.visual.backbone."),
    (#"position_embedding\.weight$"#, "position_embedding"),
    (#"\.layernorm_before\."#, ".ln_1."),
    (#"\.layernorm_after\."#, ".ln_2."),
    (#"\.mlp\.fc1\."#, ".mlp.c_fc."),
    (#"\.mlp\.fc2\."#, ".mlp.c_proj."),
    (#"\.lambda_(\d)$"#, ".ls_$1.gamma"),
]

/// Thinking Machines' layout of `model`, after `checkpoint()` in `test_inkling2.py`.
private func nativeCheckpoint(_ model: Module) -> [String: MLXArray] {
    let expert = try! NSRegularExpression(
        pattern: #"^(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$"#)
    let attentionQuery = try! NSRegularExpression(
        pattern: #"\.layers\.(\d+)\.attention_input\.query$"#)
    // A power-of-two share of 448 * 6, so the global scale survives the round trip exactly.
    let denominator = MLXArray([Float(672)])
    var checkpoint = [String: MLXArray]()
    for (name, value) in model.parameters().flattened() {
        let range = NSRange(name.startIndex..., in: name)
        if name.hasPrefix("model.vision_tower.") {
            checkpoint[rename(name, nativeVisionNames)] = value
            continue
        }
        if name.hasSuffix(".message_input.query") {
            let q = concatenated([value[0..., ..<1], value[0..., 3...]], axis: 1)
            checkpoint["model.llm.middle_cycle_cross_pass_queries"] = q.reshaped(-1, value.dim(-1))
            checkpoint["model.llm.middle_cycle_cross_pass_kv_queries"] = value[0..., 1 ..< 3]
                .reshaped(-1, value.dim(-1))
            continue
        }
        if let match = expert.firstMatch(in: name, range: range) {
            let part = { (i: Int) in String(name[Range(match.range(at: i), in: name)!]) }
            let prefix = part(1).replacingOccurrences(
                of: "model.language_model.", with: "model.llm.")
            let tensor = "\(prefix).\(part(2) == "gate_up" ? "w13" : "w2")_weight"
            switch part(3) {
            case "weight": checkpoint[tensor] = value.view(dtype: .uint8)
            case "scales": checkpoint[tensor + "_scale"] = value
            default:
                checkpoint[tensor + "_amax"] = value * (denominator / 2688)
                checkpoint[tensor + "_outer_scale_denominator"] = denominator
            }
            continue
        }
        if let match = attentionQuery.firstMatch(in: name, range: range) {
            let layer = Int(name[Range(match.range(at: 1), in: name)!])!
            checkpoint["model.llm.attn_res.attn_queries.\(layer - 1)"] = value
            continue
        }
        let target = rename(name, nativeNames)
        checkpoint[target] = target.hasSuffix("sconv.weight") ? value.swappedAxes(1, 2) : value
    }
    return checkpoint
}

/// oMLX's layout of `model`: what `omlx/patches/inkling2/loader.py` writes, with fused
/// projections, [C, K] convolutions, [Q, D] queries and its loop controller shapes.
private func oMLXCheckpoint(_ model: Module, _ args: Inkling2TextConfiguration) -> [String:
    MLXArray]
{
    let hidden = args.hiddenSize
    let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    var checkpoint = [String: MLXArray]()
    // oMLX keeps the native vision names under `vision_tower.`.
    for (name, value) in parameters where name.hasPrefix("model.vision_tower.") {
        let native = rename(name, nativeVisionNames)
        checkpoint["vision_tower." + native.dropFirst("model.visual.".count)] = value
    }
    for (name, value) in parameters
    where !name.contains(".layers.") && !name.hasPrefix("model.vision_tower.") {
        let target =
            name
            .replacingOccurrences(of: "model.language_model.embed_tokens.", with: "model.embed.")
            .replacingOccurrences(of: "model.language_model.", with: "model.")
        if name.hasSuffix(".message_input.query") { continue }
        checkpoint[target] = name.hasSuffix(".query") ? value.reshaped(-1, hidden) : value
    }

    for layer in 0 ..< args.hiddenLayers {
        let source = "model.language_model.layers.\(layer)."
        let target = "model.layers.\(layer)."
        func tensors(_ module: String) -> [String: MLXArray] {
            Dictionary(
                uniqueKeysWithValues: parameters.compactMap { name, value in
                    name.hasPrefix(source + module + ".")
                        ? (String(name.dropFirst((source + module + ".").count)), value) : nil
                })
        }
        func fuse(_ modules: [String], into fused: String) {
            for (leaf, _) in tensors(modules[0]) {
                checkpoint[target + fused + "." + leaf] = concatenated(
                    modules.map { tensors($0)[leaf]! }, axis: 0)
            }
        }
        func copy(_ module: String, to destination: String) {
            for (leaf, value) in tensors(module) {
                checkpoint[target + destination + "." + leaf] = value
            }
        }
        if args.isSliding(layer) {
            fuse(
                ["q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj"].map {
                    "self_attn.\($0)"
                }, into: "attn.in_proj")
            checkpoint[target + "attn.k_conv"] =
                parameters[source + "self_attn.k_sconv.conv1d.weight"]![0..., 0..., 0]
            checkpoint[target + "attn.v_conv"] =
                parameters[source + "self_attn.v_sconv.conv1d.weight"]![0..., 0..., 0]
            copy("input_layernorm", to: "attn_norm")
        } else {
            fuse(
                ["q_proj", "r_proj", "gate_proj", "sink_proj"].map { "self_attn.\($0)" },
                into: "attn.q_proj")
            for kind in ["k", "v"] {
                fuse(
                    ["self_attn.\(kind)_proj", "self_attn.\(kind)_sconv.weight_proj"],
                    into: "attn.\(kind)_proj")
                checkpoint[target + "attn.\(kind)_static"] =
                    parameters[source + "self_attn.\(kind)_sconv.static_weight"]!
                copy("\(kind)_input_layernorm", to: "\(kind)_input_norm")
            }
            copy("q_input_layernorm", to: "q_input_norm")
        }
        copy("self_attn.o_proj", to: "attn.o_proj")
        checkpoint[target + "attn.q_scale"] = parameters[source + "self_attn.q_l2_clip.scale"]!
        checkpoint[target + "attn.r_scale"] = parameters[source + "self_attn.rel_l2_clip.scale"]!
        checkpoint[target + "attn.rel_proj"] =
            parameters[source + "self_attn.rel_logits_proj.proj"]!
        checkpoint[target + "attn_conv"] =
            parameters[source + "post_attention_convolution.conv1d.weight"]![0..., 0..., 0]
        checkpoint[target + "mlp_conv"] =
            parameters[source + "post_mlp_convolution.conv1d.weight"]![0..., 0..., 0]
        copy("post_attention_layernorm", to: "mlp_norm")
        checkpoint[target + "mlp.router_weight"] = parameters[source + "mlp.gate.weight"]!
        checkpoint[target + "mlp.router_bias"] =
            parameters[source + "mlp.gate.e_score_correction_bias"]!
        copy("mlp.experts.gate_up_proj", to: "mlp.experts.w13")
        copy("mlp.experts.down_proj", to: "mlp.experts.w2")
        if let query = parameters[source + "attention_input.query"] {
            checkpoint[target + "attn_input.query"] = query.reshaped(-1, hidden)
        }
        checkpoint[target + "mlp_input.query"] =
            parameters[source + "mlp_input.query"]!.reshaped(-1, hidden)

        if let loop = args.loop, loop.core.contains(layer) {
            var downs = [MLXArray]()
            var gates = [MLXArray]()
            for gate in ["q", "k", "v", "mlp"] {
                let prefix = source + "\(gate)_message_gate."
                let down = parameters[prefix + "down_proj.weight"]!.asType(.float32)
                downs.append(down.reshaped(down.dim(0), 2, -1).transposed(1, 2, 0))
                gates.append(
                    concatenated(
                        [
                            parameters[prefix + "message_gate_proj.weight"]!.T,
                            parameters[prefix + "forget_gate_proj.weight"]!.T,
                        ], axis: 1))
            }
            checkpoint[target + "loop_down"] = stacked(downs)
            checkpoint[target + "loop_gain_forget"] = stacked(gates)
            checkpoint[target + "loop_message.query"] =
                parameters["model.language_model.message_input.query"]![layer - loop.coreStart]
        }
    }
    return checkpoint
}

/// oQ's settings for `quantizeLikeOQ`, under oMLX's module names.
private func oQQuantization(_ args: Inkling2TextConfiguration, expertBits: Int)
    -> BaseConfiguration.PerLayerQuantization
{
    let eight = BaseConfiguration.QuantizationOption.quantize(.init(groupSize: 64, bits: 8))
    var settings: [String: BaseConfiguration.QuantizationOption] = [
        "model.embed": eight, "lm_head": eight,
    ]
    for layer in 0 ..< args.hiddenLayers {
        let modules =
            args.isSliding(layer)
            ? ["attn.in_proj", "attn.o_proj"]
            : ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"]
        for module in modules {
            settings["model.layers.\(layer).\(module)"] = eight
        }
    }
    return .init(
        quantization: .init(groupSize: 64, bits: expertBits), perLayerQuantization: settings)
}

/// The reverse of the mlx-vlm renames, for tensor and module paths alike. The experts' gate
/// and up halves are split by `mlxVLMCheckpoint`.
private let mlxVLMNames: [(String, String)] = [
    (#"^lm_head(?=\.|$)"#, "language_model.lm_head"),
    (#"^model\.language_model\.message_input\.query$"#, "language_model.model.message_queries"),
    (#"^model\.language_model(?=\.)"#, "language_model.model"),
    (#"\.q_message_gate\."#, ".controllers.attn."),
    (#"\.(k|v|mlp)_message_gate\."#, ".controllers.$1."),
    (#"(\.controllers\.\w+)\.down_proj\.weight$"#, "$1.down_weight"),
    (#"(\.controllers\.\w+)\.forget_gate_proj\.weight$"#, "$1.forget_weight"),
    (#"(\.controllers\.\w+)\.message_gate_proj\.weight$"#, "$1.gain_weight"),
    (#"\.post_attention_convolution\.conv1d(?=\.|$)"#, ".attn_sconv.conv"),
    (#"\.post_mlp_convolution\.conv1d(?=\.|$)"#, ".mlp_sconv.conv"),
    (#"sconv\.conv1d(?=\.|$)"#, "sconv.conv"),
    (#"\.self_attn\.q_l2_clip\.scale$"#, ".self_attn.q_scale"),
    (#"\.self_attn\.rel_l2_clip\.scale$"#, ".self_attn.rel_scale"),
    (#"\.self_attn\.rel_logits_proj\.proj$"#, ".self_attn.rel_proj"),
    (#"\.mlp\.gate\.weight$"#, ".mlp.gate_weight"),
    (#"\.mlp\.gate\.e_score_correction_bias$"#, ".mlp.e_score_correction_bias"),
    (#"\.mlp\.experts\.down_proj(?=\.|$)"#, ".mlp.switch_mlp.down_proj"),
    (#"\.global_scale$"#, ".weight_scale_2"),
    (#"^model\.vision_tower\.merger\.linear_1(?=\.|$)"#, "vision_tower.proj_in"),
    (#"^model\.vision_tower\.merger\.linear_2(?=\.|$)"#, "vision_tower.proj_out"),
    (#"^model\.vision_tower(?=\.)"#, "vision_tower"),
    (#"embeddings\.position_embedding\.weight$"#, "embeddings.position_embedding"),
]

/// mlx-vlm's layout of `model`, as `Model.sanitize` in mlx-vlm's `inkling2.py` writes it:
/// gate and up rows split apart, and each NVFP4 expert's outer factor `amax / denominator`
/// in place of its global scale.
private func mlxVLMCheckpoint(_ model: Module) -> [String: MLXArray] {
    let gateUp = try! NSRegularExpression(pattern: #"^(.+)\.mlp\.experts\.gate_up_proj\.(\w+)$"#)
    var checkpoint = [String: MLXArray]()
    for (name, value) in model.parameters().flattened() {
        let value = name.hasSuffix(".global_scale") ? value / 2688 : value
        guard let match = gateUp.firstMatch(in: name, range: NSRange(name.startIndex..., in: name))
        else {
            checkpoint[rename(name, mlxVLMNames)] = value
            continue
        }
        let part = { (i: Int) in String(name[Range(match.range(at: i), in: name)!]) }
        let prefix = rename(part(1), mlxVLMNames) + ".mlp.switch_mlp."
        if part(2) == "global_scale" {
            checkpoint[prefix + "gate_proj.weight_scale_2"] = value
            checkpoint[prefix + "up_proj.weight_scale_2"] = value
        } else {
            // Alternating rows hold gate and up.
            var shape = value.shape
            shape[shape.count - 2] /= 2
            shape.insert(2, at: shape.count - 1)
            let rows = value.reshaped(shape)
            checkpoint[prefix + "gate_proj." + part(2)] = rows[.ellipsis, 0, 0...]
            checkpoint[prefix + "up_proj." + part(2)] = rows[.ellipsis, 1, 0...]
        }
    }
    return checkpoint
}

/// The settings of `model`'s quantized layers under mlx-vlm's module names, with the
/// experts' gate and up halves listed apart.
private func mlxVLMQuantization(
    _ model: Module, default quantization: BaseConfiguration.Quantization
) -> BaseConfiguration.PerLayerQuantization {
    var settings = [String: BaseConfiguration.QuantizationOption]()
    for (path, module) in model.namedModules() {
        guard let layer = module as? Quantized else { continue }
        let option = BaseConfiguration.QuantizationOption.quantize(
            .init(groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode))
        if path.hasSuffix(".mlp.experts.gate_up_proj") {
            let prefix =
                rename(String(path.dropLast(".experts.gate_up_proj".count)), mlxVLMNames)
                + ".switch_mlp."
            settings[prefix + "gate_proj"] = option
            settings[prefix + "up_proj"] = option
        } else {
            settings[rename(path, mlxVLMNames)] = option
        }
    }
    return .init(quantization: quantization, perLayerQuantization: settings)
}

/// Quantizes like NetProjects' affine mlx-vlm builds: experts at `expertBits`, attention at 6
/// bits, embeddings, head and vision tower at 4. Sinks, convolution projections, loop gates,
/// the router and the patch embedding stay as they are. The tiny vision tower is 32 wide, so
/// it uses groups of 32, and a layer whose inputs don't divide into groups stays too.
private func quantizeLikeMLXVLM(_ model: Module, expertBits: Int)
    -> BaseConfiguration.PerLayerQuantization
{
    quantize(model: model) { path, module in
        let setting: (Int, Int, QuantizationMode)
        if path.contains(".mlp.experts.") {
            setting = (64, expertBits, .affine)
        } else if path.hasPrefix("model.vision_tower.") {
            guard !path.contains(".embeddings.") else { return nil }
            setting = (32, 4, .affine)
        } else if path.contains(".self_attn."), !path.hasSuffix("sink_proj"),
            !path.contains("sconv")
        {
            setting = (64, 6, .affine)
        } else if path.hasSuffix("embed_tokens") || path == "lm_head" {
            setting = (64, 4, .affine)
        } else {
            return nil
        }
        if let linear = module as? Linear, linear.weight.dim(-1) % setting.0 != 0 { return nil }
        return module is Quantizable ? setting : nil
    }
    eval(model)
    return mlxVLMQuantization(model, default: .init(groupSize: 64, bits: expertBits))
}

/// Every tensor name in `netprojects/Inkling-2-Mini-MLX-3bit` and `-4-8bit`, with layer
/// numbers replaced by `N`.
private let publishedMLXVLMNames = """
    language_model.lm_head.biases language_model.lm_head.scales language_model.lm_head.weight
    language_model.model.embed_norm.weight language_model.model.embed_tokens.biases
    language_model.model.embed_tokens.scales language_model.model.embed_tokens.weight
    language_model.model.final_input.query language_model.model.layers.N.attention_input.query
    language_model.model.layers.N.attn_sconv.conv.weight
    language_model.model.layers.N.controllers.attn.down_weight
    language_model.model.layers.N.controllers.attn.forget_weight
    language_model.model.layers.N.controllers.attn.gain_weight
    language_model.model.layers.N.controllers.k.down_weight
    language_model.model.layers.N.controllers.k.forget_weight
    language_model.model.layers.N.controllers.k.gain_weight
    language_model.model.layers.N.controllers.mlp.down_weight
    language_model.model.layers.N.controllers.mlp.forget_weight
    language_model.model.layers.N.controllers.mlp.gain_weight
    language_model.model.layers.N.controllers.v.down_weight
    language_model.model.layers.N.controllers.v.forget_weight
    language_model.model.layers.N.controllers.v.gain_weight
    language_model.model.layers.N.input_layernorm.weight
    language_model.model.layers.N.k_input_layernorm.weight
    language_model.model.layers.N.mlp.e_score_correction_bias
    language_model.model.layers.N.mlp.gate_weight
    language_model.model.layers.N.mlp.switch_mlp.down_proj.biases
    language_model.model.layers.N.mlp.switch_mlp.down_proj.scales
    language_model.model.layers.N.mlp.switch_mlp.down_proj.weight
    language_model.model.layers.N.mlp.switch_mlp.down_proj.weight_scale_2
    language_model.model.layers.N.mlp.switch_mlp.gate_proj.biases
    language_model.model.layers.N.mlp.switch_mlp.gate_proj.scales
    language_model.model.layers.N.mlp.switch_mlp.gate_proj.weight
    language_model.model.layers.N.mlp.switch_mlp.gate_proj.weight_scale_2
    language_model.model.layers.N.mlp.switch_mlp.up_proj.biases
    language_model.model.layers.N.mlp.switch_mlp.up_proj.scales
    language_model.model.layers.N.mlp.switch_mlp.up_proj.weight
    language_model.model.layers.N.mlp.switch_mlp.up_proj.weight_scale_2
    language_model.model.layers.N.mlp_input.query
    language_model.model.layers.N.mlp_sconv.conv.weight
    language_model.model.layers.N.post_attention_layernorm.weight
    language_model.model.layers.N.q_input_layernorm.weight
    language_model.model.layers.N.self_attn.gate_proj.biases
    language_model.model.layers.N.self_attn.gate_proj.scales
    language_model.model.layers.N.self_attn.gate_proj.weight
    language_model.model.layers.N.self_attn.k_proj.biases
    language_model.model.layers.N.self_attn.k_proj.scales
    language_model.model.layers.N.self_attn.k_proj.weight
    language_model.model.layers.N.self_attn.k_sconv.conv.weight
    language_model.model.layers.N.self_attn.k_sconv.static_weight
    language_model.model.layers.N.self_attn.k_sconv.weight_proj.weight
    language_model.model.layers.N.self_attn.o_proj.biases
    language_model.model.layers.N.self_attn.o_proj.scales
    language_model.model.layers.N.self_attn.o_proj.weight
    language_model.model.layers.N.self_attn.q_proj.biases
    language_model.model.layers.N.self_attn.q_proj.scales
    language_model.model.layers.N.self_attn.q_proj.weight
    language_model.model.layers.N.self_attn.q_scale
    language_model.model.layers.N.self_attn.r_proj.biases
    language_model.model.layers.N.self_attn.r_proj.scales
    language_model.model.layers.N.self_attn.r_proj.weight
    language_model.model.layers.N.self_attn.rel_proj
    language_model.model.layers.N.self_attn.rel_scale
    language_model.model.layers.N.self_attn.sink_proj.weight
    language_model.model.layers.N.self_attn.v_proj.biases
    language_model.model.layers.N.self_attn.v_proj.scales
    language_model.model.layers.N.self_attn.v_proj.weight
    language_model.model.layers.N.self_attn.v_sconv.conv.weight
    language_model.model.layers.N.self_attn.v_sconv.static_weight
    language_model.model.layers.N.self_attn.v_sconv.weight_proj.weight
    language_model.model.layers.N.v_input_layernorm.weight language_model.model.message_queries
    language_model.model.norm.weight vision_tower.embeddings.class_embedding
    vision_tower.embeddings.cls_position_embedding
    vision_tower.embeddings.patch_embedding.weight vision_tower.embeddings.position_embedding
    vision_tower.layers.N.lambda_1 vision_tower.layers.N.lambda_2
    vision_tower.layers.N.layernorm_after.bias vision_tower.layers.N.layernorm_after.weight
    vision_tower.layers.N.layernorm_before.bias vision_tower.layers.N.layernorm_before.weight
    vision_tower.layers.N.mlp.fc1.bias vision_tower.layers.N.mlp.fc1.biases
    vision_tower.layers.N.mlp.fc1.scales vision_tower.layers.N.mlp.fc1.weight
    vision_tower.layers.N.mlp.fc2.bias vision_tower.layers.N.mlp.fc2.biases
    vision_tower.layers.N.mlp.fc2.scales vision_tower.layers.N.mlp.fc2.weight
    vision_tower.layers.N.self_attn.k_proj.bias vision_tower.layers.N.self_attn.k_proj.biases
    vision_tower.layers.N.self_attn.k_proj.scales vision_tower.layers.N.self_attn.k_proj.weight
    vision_tower.layers.N.self_attn.out_proj.bias
    vision_tower.layers.N.self_attn.out_proj.biases
    vision_tower.layers.N.self_attn.out_proj.scales
    vision_tower.layers.N.self_attn.out_proj.weight vision_tower.layers.N.self_attn.q_proj.bias
    vision_tower.layers.N.self_attn.q_proj.biases vision_tower.layers.N.self_attn.q_proj.scales
    vision_tower.layers.N.self_attn.q_proj.weight vision_tower.layers.N.self_attn.v_proj.bias
    vision_tower.layers.N.self_attn.v_proj.biases vision_tower.layers.N.self_attn.v_proj.scales
    vision_tower.layers.N.self_attn.v_proj.weight vision_tower.pre_layernorm.bias
    vision_tower.pre_layernorm.weight vision_tower.proj_in.biases vision_tower.proj_in.scales
    vision_tower.proj_in.weight vision_tower.proj_out.biases vision_tower.proj_out.scales
    vision_tower.proj_out.weight
    """.split(whereSeparator: \.isWhitespace).map(String.init)

/// `model`, with other random weights, then loaded from `weights` written to a directory.
private func loaded<Model: BaseLanguageModel>(
    _ weights: [String: MLXArray], into model: Model,
    quantization: BaseConfiguration.PerLayerQuantization? = nil
) throws -> Model {
    let directory = FileManager.default.temporaryDirectory.appending(
        path: "inkling2-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try save(arrays: weights, url: directory.appending(path: "model.safetensors"))
    try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: quantization)
    return model
}

/// A fresh text model of the same configuration loaded from `weights`.
private func loaded(
    _ weights: [String: MLXArray], loop: Bool, gateRank: Int = 8,
    quantization: BaseConfiguration.PerLayerQuantization? = nil
) throws -> Inkling2Model {
    try loaded(
        weights, into: makeInkling2(loop: loop, gateRank: gateRank, seed: 99),
        quantization: quantization)
}

// MARK: - Tests

/// On the GPU, MLX multiplies float32 matrices in TF32 unless `MLX_ENABLE_TF32=0`, except in
/// its one-row kernels, so a decode step and a full forward pass round differently. Exact
/// comparisons run on the CPU.
private func onCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(.cpu, body)
}

@Suite(.serialized)
struct Inkling2Tests {

    @Test("A forward pass returns float32 logits over the unpadded vocabulary")
    func forwardShapes() throws {
        for loop in [false, true] {
            let model = try makeInkling2(loop: loop)
            let logits = model(MLXArray([Int32(0), 1]).reshaped(1, 2), cache: nil)
            #expect(logits.shape == [1, 2, unpaddedVocabulary])
            #expect(logits.dtype == .float32)
        }
    }

    @Test(
        "Chunked prefill and decoding match one forward pass",
        arguments: [(false, false), (true, false), (false, true), (true, true)])
    func cachedMatchesForward(loop: Bool, quantized: Bool) throws {
        try onCPU {
            let model = try makeInkling2(loop: loop)
            if quantized {
                quantizeExpertsNatively(model)
            }
            // A window of 8 is shorter than the 20 tokens, so the rotating caches wrap.
            for batch in [1, 2] {
                let input = tokens([batch, 20])
                let full = model(input, cache: nil)
                #expect(maxDifference(try cachedLogits(model, input), full) < 1e-4)
            }
        }
    }

    @Test("The loop gives each visit its own caches")
    func loopCacheCount() throws {
        let model = try makeInkling2(loop: true)
        // Layers 0-2, three passes of 3-5, then 6 and 7: 14 visits.
        #expect(model.model.languageModel.layerVisits.count == 14)
        #expect(try model.newCache(parameters: nil).count == 3 * 14)
    }

    @Test("Ragged prompts decoded together match each decoded alone")
    func batchMatchesSingle() throws {
        try onCPU {
            let model = try makeInkling2(loop: true)
            let prompts = [[1, 2, 3], Array(5 ..< 30), Array(40 ..< 52)]
            let prefill = PrefillParameters(stepSize: 7)
            let alone = try prompts.map { prompt in
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model,
                    processor: nil, sampler: ArgMaxSampler(), prefill: prefill, maxTokens: 10)
                return Array(IteratorSequence(iterator))
            }
            let together = try batchGenerate(
                model: model, prompts: prompts, maxTokens: 10, prefill: prefill)
            #expect(together == alone)
        }
    }

    @Test("KV cache quantization is refused")
    func refusesKVQuantization() throws {
        let model = try makeInkling2()
        #expect(throws: Inkling2Error.self) {
            _ = try model.newCache(parameters: GenerateParameters(kvBits: 4))
        }
    }

    @Test("Training reaches the router through the chosen experts' weights")
    func trainingGradients() throws {
        try onCPU {
            let model = try makeInkling2(loop: true)
            model.train()
            let lossAndGradient = valueAndGrad(model: model) { model, tokens, _ in
                model(tokens, cache: nil).square().mean()
            }
            let gradients = try withError {
                let (loss, gradients) = lossAndGradient(model, tokens([2, 12]), MLXArray(0))
                let flat = gradients.flattened()
                eval([loss] + flat.map(\.1))
                return flat
            }
            #expect(gradients.allSatisfy { all(isFinite($0.1)).item(Bool.self) })
            let router = gradients.filter { $0.0.hasSuffix("mlp.gate.weight") }
            #expect(!router.isEmpty)
            #expect(router.allSatisfy { abs($0.1).max().item(Float.self) > 0 })
        }
    }

    @Test("A prompt cache saves and restores")
    func promptCacheRoundTrip() throws {
        try onCPU {
            let model = try makeInkling2(loop: true)
            let input = tokens([1, 14])
            let cache = try model.newCache(parameters: nil)
            _ = model(input[0..., ..<12], cache: cache)
            eval(cache)
            let url = FileManager.default.temporaryDirectory.appending(
                path: "inkling2-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: cache)
            let (restored, _) = try loadPromptCache(url: url)
            #expect(
                maxDifference(
                    model(input[0..., 12...], cache: restored),
                    model(input[0..., 12...], cache: cache))
                    == 0)
        }
    }

    @Test("NVFP4 experts apply their per-expert global scale")
    func scaledExpertsMatchDequantized() throws {
        MLXRandom.seed(3)
        let linear = Inkling2SwitchLinear(
            inputDims: 64, outputDims: 32, numExperts: 4, bias: false)
        let quantized = try #require(
            linear.toQuantized(groupSize: 16, bits: 4, mode: .nvfp4)
                as? Inkling2ScaledSwitchLinear)
        let scale = MLXRandom.uniform(Float(0.5) ..< 1.5, [4]) * 2688
        quantized.update(parameters: ModuleParameters.unflattened(["global_scale": scale]))
        let x = MLXRandom.normal([3, 1, 1, 64])
        let indices = MLXArray([Int32(0), 2, 3, 1, 2, 2]).reshaped(3, 2)

        let weights = dequantized(
            quantized.weight, scales: quantized.scales, biases: nil, groupSize: 16, bits: 4,
            mode: .nvfp4)
        let expected =
            gatherMM(x, weights.swappedAxes(-1, -2), rhsIndices: indices)
            * (scale[indices] / 2688).reshaped(3, 2, 1, 1)
        #expect(maxDifference(quantized(x, indices), expected) < 1e-3)
    }

    @Test("Thinking Machines' layout loads exactly", arguments: [false, true])
    func nativeCheckpointLoads(loop: Bool) throws {
        try onCPU {
            let model = try makeInkling2(loop: loop)
            quantizeExpertsNatively(model)
            let reloaded = try loaded(nativeCheckpoint(model), loop: loop)
            let input = tokens([1, 10])
            #expect(maxDifference(reloaded(input, cache: nil), model(input, cache: nil)) < 1e-5)
            #expect(
                reloaded.model.languageModel.layers[0].mlp.experts.gateUpProjection
                    is Inkling2ScaledSwitchLinear)
        }
    }

    @Test(
        "oMLX's oQ layout loads exactly",
        arguments: [(false, 3), (true, 2), (true, 3)])
    func oMLXCheckpointLoads(loop: Bool, expertBits: Int) throws {
        try onCPU {
            let model = try makeInkling2(loop: loop)
            quantizeLikeOQ(model, expertBits: expertBits)
            let reloaded = try loaded(
                oMLXCheckpoint(model, model.configuration.text), loop: loop,
                quantization: oQQuantization(model.configuration.text, expertBits: expertBits))
            let input = tokens([1, 10])
            #expect(maxDifference(reloaded(input, cache: nil), model(input, cache: nil)) < 1e-5)
        }
    }

    @Test("An mlx-lm conversion loads as saved")
    func mlxLMCheckpointLoads() throws {
        try onCPU {
            let model = try makeInkling2(loop: true)
            quantizeExpertsNatively(model)
            let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            let reloaded = try loaded(weights, loop: true)
            let input = tokens([1, 10])
            #expect(maxDifference(reloaded(input, cache: nil), model(input, cache: nil)) < 1e-5)
        }
    }

    @Test(
        "An mlx-vlm conversion with NVFP4 and MXFP8 experts loads exactly",
        arguments: [false, true])
    func mlxVLMNVFP4CheckpointLoads(loop: Bool) throws {
        try onCPU {
            let model = try makeInkling2(loop: loop)
            quantizeExpertsNatively(model)
            let reloaded = try loaded(
                mlxVLMCheckpoint(model), loop: loop,
                quantization: mlxVLMQuantization(
                    model, default: .init(groupSize: 16, bits: 4, mode: .nvfp4)))
            let input = tokens([1, 10])
            #expect(maxDifference(reloaded(input, cache: nil), model(input, cache: nil)) < 1e-5)
            #expect(
                reloaded.model.languageModel.layers[0].mlp.experts.gateUpProjection
                    is Inkling2ScaledSwitchLinear)
        }
    }

    @Test(
        "An affine mlx-vlm conversion loads exactly",
        arguments: [(false, 3), (true, 2), (true, 3)])
    func mlxVLMAffineCheckpointLoads(loop: Bool, expertBits: Int) throws {
        try onCPU {
            let model = try makeInkling2(loop: loop)
            let quantization = quantizeLikeMLXVLM(model, expertBits: expertBits)
            let reloaded = try loaded(
                mlxVLMCheckpoint(model), loop: loop, quantization: quantization)
            let input = tokens([1, 10])
            #expect(maxDifference(reloaded(input, cache: nil), model(input, cache: nil)) < 1e-5)
        }
    }

    @Test("Gate and up halves that an expert scales differently are refused")
    func mlxVLMRefusesSplitScales() throws {
        let model = try makeInkling2(loop: true)
        quantizeExpertsNatively(model)
        var weights = mlxVLMCheckpoint(model)
        let name = "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight_scale_2"
        weights[name] = try #require(weights[name]) * 2
        #expect(throws: ModelFactoryError.self) {
            _ = try Inkling2Checkpoint.prepare(
                .init(weights: weights), configuration: model.configuration, includeVision: false)
        }
    }

    @Test("Every tensor a published mlx-vlm build names has a place in the model")
    func mlxVLMNamesMatchPublishedBuilds() throws {
        let model = try makeInkling2VLM()
        let parameters = Set(model.parameters().flattened().map(\.0))
        let modules = Set(model.namedModules().map(\.0))
        let quantizationLeaves: Set<Substring> = ["scales", "biases", "global_scale"]
        // Some tensors exist only in some layers, so a name needs a place in one of them.
        let unplaced = publishedMLXVLMNames.filter { pattern in
            !(0 ..< 8).contains { layer in
                let name = Inkling2Checkpoint.mlxVLMName(
                    pattern.replacingOccurrences(of: ".N.", with: ".\(layer)."))
                guard let dot = name.lastIndex(of: "."),
                    quantizationLeaves.contains(name[name.index(after: dot)...])
                else { return parameters.contains(name) }
                // The tiny model isn't quantized; the module that would hold them is enough.
                return modules.contains(String(name[..<dot]))
            }
        }
        #expect(unplaced.isEmpty)
    }

    @Test("Loading never reads the bundled draft or other roots")
    func dropsForeignTensors() throws {
        let model = try makeInkling2()
        var weights = oMLXCheckpoint(model, model.configuration.text)
        weights["layers.0.self_attn.q_proj.weight"] = MLXArray.zeros([4, 4])
        let prepared = try Inkling2Checkpoint.prepare(
            .init(weights: weights), configuration: model.configuration, includeVision: false)
        #expect(prepared.weights.keys.allSatisfy { !$0.hasPrefix("layers.") })
    }
}

/// Tensor-parallel sharding of Inkling 2, in a group of one.
final class Inkling2ShardingTests: XCTestCase {
    private func makeModel(loop: Bool) -> (Bool) throws -> Inkling2Model {
        { quantized in
            // A gate rank of 64 lets 4-bit quantization reach the message gates.
            let model = try makeInkling2(loop: loop, gateRank: 64)
            if quantized {
                quantize(model: model, groupSize: 64, bits: 4)
                eval(model)
            }
            return model
        }
    }

    func testShardingKeepsTheModel() throws {
        try assertShardingKeepsTheModel(makeModel(loop: false))
        try assertShardingKeepsTheModel(makeModel(loop: true))
    }

    func testShardingReplacesTheProjections() throws {
        let model = try makeInkling2()
        try model.shard(group: MLXDistributed.initialize())
        for layer in model.model.languageModel.layers {
            XCTAssertTrue(layer.attention.qProjection is AllToShardedLinear)
            XCTAssertTrue(layer.attention.oProjection is ShardedToAllLinear)
            XCTAssertNotNil(layer.mlp.shardingGroup)
        }
    }
}

// MARK: - Chat protocol

private final class InklingTestTokenizer: Tokenizer, @unchecked Sendable {
    let messageModel = 200_001
    let contentText = 200_004
    let endSampling = 200_006
    let contentThinking = 200_008
    let endMessage = 200_010
    let invokeToolJSON = 200_049
    private var nextID = 1
    private var textToID: [String: Int] = [:]
    private var idToText: [Int: String] = [:]

    init() {
        for (token, id) in [
            ("<|message_model|>", messageModel), ("<|content_text|>", contentText),
            ("<|content_model_end_sampling|>", endSampling),
            ("<|content_thinking|>", contentThinking), ("<|end_message|>", endMessage),
            ("<|content_invoke_tool_json|>", invokeToolJSON),
        ] {
            textToID[token] = id
            idToText[id] = token
        }
    }

    /// A plain-text token, even when it reads like a marker.
    func id(_ text: String) -> Int {
        while idToText[nextID] != nil { nextID += 1 }
        let id = nextID
        nextID += 1
        idToText[id] = text
        return id
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [id(text)] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { idToText[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { textToID[token] }
    func convertIdToToken(_ id: Int) -> String? { idToText[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite
struct InklingProtocolTests {
    private func decode(
        _ tokens: [Int], tokenizer: InklingTestTokenizer, tools: [[String: any Sendable]]? = nil
    ) throws -> [TokenStreamEvent] {
        var decoder = try #require(
            ToolCallFormat.inkling.makeProtocolTokenStreamDecoder(
                tokenizer: tokenizer, tools: tools, stopStrings: []))
        #expect(decoder.additionalStopTokenIDs == [tokenizer.endSampling])
        var events: [TokenStreamEvent] = []
        for token in tokens {
            _ = decoder.push(token) {
                events.append($0)
                return true
            }
        }
        _ = decoder.finish {
            events.append($0)
            return true
        }
        return events
    }

    private func text(_ events: [TokenStreamEvent], reasoning: Bool = false) -> String {
        events.compactMap {
            switch $0 {
            case .reasoning(let text) where reasoning: text
            case .response(let text) where !reasoning: text
            default: nil
            }
        }.joined()
    }

    private func calls(_ events: [TokenStreamEvent]) -> [ToolCall.Function] {
        events.compactMap {
            if case .toolCall(let call) = $0 { return call.function }
            return nil
        }
    }

    @Test("Reasoning and the answer are separate messages")
    func reasoningThenAnswer() throws {
        let t = InklingTestTokenizer()
        let events = try decode(
            [
                t.messageModel, t.contentThinking, t.id("Plan "), t.id("it."), t.endMessage,
                t.messageModel, t.contentText, t.id("Hello"), t.id("."), t.endMessage,
                t.endSampling,
            ], tokenizer: t)
        #expect(text(events, reasoning: true) == "Plan it.")
        #expect(text(events) == "Hello.")
        #expect(!events.contains { if case .protocolError = $0 { true } else { false } })
    }

    @Test("Several calls in one turn come back in order, with typed arguments")
    func parallelToolCalls() throws {
        let t = InklingTestTokenizer()
        let events = try decode(
            [
                t.messageModel, t.id("get_weather"), t.invokeToolJSON,
                t.id(#"{"name":"get_weather","args":{"city":"Paris","days":3}}"#), t.endMessage,
                t.messageModel, t.id("ping"), t.invokeToolJSON, t.id(#"{"name":"ping"}"#),
                t.endMessage, t.endSampling,
            ], tokenizer: t)
        let functions = calls(events)
        #expect(functions.map(\.name) == ["get_weather", "ping"])
        #expect(functions[0].arguments["city"] == .string("Paris"))
        #expect(functions[0].arguments["days"] == .int(3))
        #expect(functions[1].arguments.isEmpty)
    }

    @Test("A prompt that began the answer continues it as text")
    func prefillContinuation() throws {
        let t = InklingTestTokenizer()
        let events = try decode(
            [t.id("world, "), t.id("again"), t.endMessage, t.endSampling], tokenizer: t)
        #expect(text(events) == "world, again")
        #expect(calls(events).isEmpty)
    }

    @Test("Marker-shaped text is only text")
    func literalMarkers() throws {
        let t = InklingTestTokenizer()
        let events = try decode(
            [t.messageModel, t.contentText, t.id("<|end_message|> is a marker"), t.endMessage],
            tokenizer: t)
        #expect(text(events) == "<|end_message|> is a marker")
    }

    @Test("Malformed and cut-off calls are rejected, not run")
    func rejectedCalls() throws {
        let t = InklingTestTokenizer()
        func reasons(_ events: [TokenStreamEvent]) -> [RejectedToolCall.Reason] {
            events.compactMap {
                if case .rejectedToolCall(let rejection) = $0 { return rejection.reason }
                return nil
            }
        }
        let malformed = try decode(
            [t.messageModel, t.id("f"), t.invokeToolJSON, t.id("{not json"), t.endMessage],
            tokenizer: t)
        #expect(reasons(malformed) == [.malformedSyntax])
        let cutOff = try decode(
            [t.messageModel, t.id("f"), t.invokeToolJSON, t.id(#"{"name":"f","ar"#)],
            tokenizer: t)
        #expect(reasons(cutOff) == [.incompleteOutput])
        #expect(calls(malformed + cutOff).isEmpty)
    }

    @Test("A call to a tool the caller didn't declare is rejected, not run")
    func undeclaredTools() throws {
        let t = InklingTestTokenizer()
        let turn = [
            t.messageModel, t.id("get_weather"), t.invokeToolJSON,
            t.id(#"{"name":"get_weather","args":{"city":"Paris"}}"#), t.endMessage,
            t.messageModel, t.id("ping"), t.invokeToolJSON, t.id(#"{"name":"ping"}"#),
            t.endMessage, t.endSampling,
        ]
        func rejected(_ events: [TokenStreamEvent]) -> [String?] {
            events.compactMap {
                if case .rejectedToolCall(let rejection) = $0, rejection.reason == .undeclaredTool {
                    return rejection.toolName
                }
                return nil
            }
        }
        let weather: [String: any Sendable] = [
            "type": "function", "function": ["name": "get_weather"] as [String: any Sendable],
        ]
        let declared = try decode(turn, tokenizer: t, tools: [weather])
        #expect(calls(declared).map(\.name) == ["get_weather"])
        #expect(rejected(declared) == ["ping"])

        // A tool given without a `function` wrapper is declared by its own name.
        let bare = try decode(turn, tokenizer: t, tools: [["name": "ping"]])
        #expect(calls(bare).map(\.name) == ["ping"])
        #expect(rejected(bare) == ["get_weather"])

        let none = try decode(turn, tokenizer: t, tools: [])
        #expect(calls(none).isEmpty)
        #expect(rejected(none) == ["get_weather", "ping"])
    }
}

// MARK: - Vision

private final class InklingImageTokenizer: Tokenizer, @unchecked Sendable {
    let imageToken = 127

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? {
        token == "<|unused_200054|>" ? imageToken : nil
    }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, imageToken, 2] }
}

@Suite(.serialized)
struct Inkling2VisionTests {
    /// Two tokens, one image token per 4x4-pixel merge block, two tokens.
    private func imageInput(height: Int, width: Int, seed: UInt64, imageTokens: Int? = nil)
        -> LMInput
    {
        MLXRandom.seed(seed)
        let count = imageTokens ?? (height / 4) * (width / 4)
        let tokens = [1, 2] + Array(repeating: 127, count: count) + [3, 4]
        let pixels = MLXRandom.uniform(Float(-1) ..< 1, [3, height, width])
        return LMInput(
            text: .init(tokens: MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)),
            image: .init(pixels: pixels.reshaped(-1), frames: [THW(1, height, width)]))
    }

    private func logits(_ model: Inkling2VLM, _ input: LMInput) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init())
        else {
            Issue.record("prepare returned tokens")
            return MLXArray(0)
        }
        return output.logits
    }

    @Test("The vision tower gives one text-width row per merge block")
    func visionRows() throws {
        try onCPU {
            let model = try makeInkling2VLM()
            let tower = try #require(model.model.visionTower)
            let pixels = MLXRandom.normal([3, 8, 12])
            let rows: MLXArray = try tower(pixels)
            #expect(rows.shape == [6, 64])
        }
    }

    @Test("A text prompt gives what the text model gives")
    func textMatchesTextModel() throws {
        try onCPU {
            let vlm = try makeInkling2VLM()
            let weights = Dictionary(uniqueKeysWithValues: vlm.parameters().flattened())
            let text = try loaded(weights, loop: true)
            let input = tokens([1, 12])
            #expect(maxDifference(vlm(input, cache: nil), text(input, cache: nil)) == 0)
        }
    }

    @Test("Each image's rows reach its tokens, and the counts must agree")
    func imageRowsReachTheirTokens() throws {
        try onCPU {
            let model = try makeInkling2VLM()
            let first = try logits(model, imageInput(height: 8, width: 12, seed: 2))
            let again = try logits(model, imageInput(height: 8, width: 12, seed: 2))
            let second = try logits(model, imageInput(height: 8, width: 12, seed: 3))
            #expect(first.dim(-1) == unpaddedVocabulary)
            // The prompt's last token sees the image.
            #expect(maxDifference(first, again) == 0)
            #expect(maxDifference(first, second) > 0)
            #expect(throws: (any Error).self) {
                _ = try logits(model, imageInput(height: 8, width: 12, seed: 2, imageTokens: 5))
            }
        }
    }

    @Test("Thinking Machines' and oMLX's layouts load the vision tower exactly")
    func visionCheckpointsLoad() throws {
        try onCPU {
            let input = imageInput(height: 8, width: 12, seed: 2)
            let native = try makeInkling2VLM()
            quantizeExpertsNatively(native)
            let fromNative = try loaded(nativeCheckpoint(native), into: makeInkling2VLM(seed: 99))
            #expect(maxDifference(try logits(fromNative, input), try logits(native, input)) == 0)

            let oQ = try makeInkling2VLM()
            quantizeLikeOQ(oQ, expertBits: 3)
            let fromOMLX = try loaded(
                oMLXCheckpoint(oQ, oQ.configuration.text), into: makeInkling2VLM(seed: 99),
                quantization: oQQuantization(oQ.configuration.text, expertBits: 3))
            #expect(maxDifference(try logits(fromOMLX, input), try logits(oQ, input)) == 0)
        }
    }

    @Test("An mlx-vlm conversion loads its quantized vision tower exactly")
    func mlxVLMVisionLoads() throws {
        try onCPU {
            let input = imageInput(height: 8, width: 12, seed: 2)
            let model = try makeInkling2VLM()
            let quantization = quantizeLikeMLXVLM(model, expertBits: 3)
            let reloaded = try loaded(
                mlxVLMCheckpoint(model), into: makeInkling2VLM(seed: 99),
                quantization: quantization)
            #expect(maxDifference(try logits(reloaded, input), try logits(model, input)) == 0)
            let names = Set(reloaded.parameters().flattened().map(\.0))
            #expect(names.contains("model.vision_tower.layers.0.mlp.fc1.scales"))
            #expect(names.contains("model.vision_tower.merger.linear_1.scales"))
        }
    }

    @Test("The processor pads each image to whole 40-pixel blocks")
    func processorPads() async throws {
        let config = try JSONDecoder().decode(
            Inkling2ProcessorConfiguration.self,
            from: Data(
                #"{"image_token": "<|unused_200054|>", "processor_class": "Inkling2Processor"}"#
                    .utf8))
        let tokenizer = InklingImageTokenizer()
        let processor = Inkling2Processor(config, tokenizer: tokenizer)
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 90, height: 50))
        let input = try await processor.prepare(
            input: UserInput(prompt: "Describe it.", images: [.ciImage(image)]))

        let frames = try #require(input.image?.frames)
        #expect(frames.map { [$0.t, $0.h, $0.w] } == [[1, 80, 120]])
        // Two rows and three columns of blocks.
        let tokens = input.text.tokens.asArray(Int.self)
        #expect(tokens.filter { $0 == tokenizer.imageToken }.count == 6)
        #expect(tokens.first == 1 && tokens.last == 2)

        let pixels = try #require(input.image?.pixels).reshaped(3, 80, 120)
        // Red in the image; -1 on the 0-255 scale in the padding.
        #expect(
            abs(pixels[0..., 10, 10] - MLXArray([Float(1), -1, -1])).max().item(Float.self) < 1e-2)
        let fill = (-1 / 255 - 0.5) / Float(0.5)
        #expect(abs(pixels[0..., 79, 119] - fill).max().item(Float.self) < 1e-5)
    }
}
