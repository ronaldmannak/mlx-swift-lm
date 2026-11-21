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

private func inkling2ConfigJSON(loop: Bool, gateRank: Int = 8) -> String {
    let loopConfig =
        loop
        ? """
        , "loop_config": {"core_start": 3, "core_end": 6, "num_passes": 3,
            "gate_rank": \(gateRank), "is_varloop": false}
        """ : ""
    return """
        {
            "model_type": "\(loop ? "inkling2_loop" : "inkling2")",
            "image_token_id": 127,
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

private func inkling2Configuration(loop: Bool, gateRank: Int = 8) throws
    -> Inkling2Configuration
{
    try JSONDecoder().decode(
        Inkling2Configuration.self,
        from: Data(inkling2ConfigJSON(loop: loop, gateRank: gateRank).utf8))
}

/// A seeded model whose weights are random at the reference's scale: norms and clip scales
/// near one, other weights normal over the square root of their fan-in.
private func makeInkling2(loop: Bool = false, gateRank: Int = 8, seed: UInt64 = 0) throws
    -> Inkling2Model
{
    MLXRandom.seed(seed)
    let model = Inkling2Model(try inkling2Configuration(loop: loop, gateRank: gateRank))
    let weights = model.parameters().flattened().map { name, value -> (String, MLXArray) in
        if name.hasSuffix("norm.weight") || name.hasSuffix("l2_clip.scale") {
            return (name, 1 + 0.1 * MLXRandom.normal(value.shape))
        }
        let fanIn = value.ndim > 1 ? value.dim(-1) : 16
        return (name, MLXRandom.normal(value.shape) * pow(Float(fanIn), -0.5))
    }
    model.update(parameters: ModuleParameters.unflattened(weights))
    eval(model)
    return model
}

/// NVFP4 experts with random global scales, and MXFP8 experts in the last layer, as in
/// Thinking Machines' checkpoint.
private func quantizeExpertsNatively(_ model: Inkling2Model) {
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
private func quantizeLikeOQ(_ model: Inkling2Model, expertBits: Int) {
    quantize(model: model) { path, module in
        if path.contains(".mlp.experts.") { return (64, expertBits, .affine) }
        if path.contains("_message_gate.") { return nil }
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

/// Thinking Machines' layout of `model`, after `checkpoint()` in `test_inkling2.py`.
private func nativeCheckpoint(_ model: Inkling2Model) -> [String: MLXArray] {
    let expert = try! NSRegularExpression(
        pattern: #"^(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$"#)
    let attentionQuery = try! NSRegularExpression(
        pattern: #"\.layers\.(\d+)\.attention_input\.query$"#)
    // A power-of-two share of 448 * 6, so the global scale survives the round trip exactly.
    let denominator = MLXArray([Float(672)])
    var checkpoint = [String: MLXArray]()
    for (name, value) in model.parameters().flattened() {
        let range = NSRange(name.startIndex..., in: name)
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
private func oMLXCheckpoint(_ model: Inkling2Model) -> [String: MLXArray] {
    let args = model.configuration.text
    let hidden = args.hiddenSize
    let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    var checkpoint = [String: MLXArray]()
    for (name, value) in parameters where !name.contains(".layers.") {
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
private func oQQuantization(_ model: Inkling2Model, expertBits: Int)
    -> BaseConfiguration.PerLayerQuantization
{
    let eight = BaseConfiguration.QuantizationOption.quantize(.init(groupSize: 64, bits: 8))
    var settings: [String: BaseConfiguration.QuantizationOption] = [
        "model.embed": eight, "lm_head": eight,
    ]
    for layer in 0 ..< model.configuration.text.hiddenLayers {
        let modules =
            model.configuration.text.isSliding(layer)
            ? ["attn.in_proj", "attn.o_proj"]
            : ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"]
        for module in modules {
            settings["model.layers.\(layer).\(module)"] = eight
        }
    }
    return .init(
        quantization: .init(groupSize: 64, bits: expertBits), perLayerQuantization: settings)
}

/// A fresh model of the same configuration loaded from `weights` written to a directory.
private func loaded(
    _ weights: [String: MLXArray], loop: Bool, gateRank: Int = 8,
    quantization: BaseConfiguration.PerLayerQuantization? = nil
) throws -> Inkling2Model {
    let directory = FileManager.default.temporaryDirectory.appending(
        path: "inkling2-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try save(arrays: weights, url: directory.appending(path: "model.safetensors"))
    let model = try makeInkling2(loop: loop, gateRank: gateRank, seed: 99)
    try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: quantization)
    return model
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
                oMLXCheckpoint(model), loop: loop,
                quantization: oQQuantization(model, expertBits: expertBits))
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

    @Test("Loading never reads the bundled draft or other roots")
    func dropsForeignTensors() throws {
        let model = try makeInkling2()
        var weights = oMLXCheckpoint(model)
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
