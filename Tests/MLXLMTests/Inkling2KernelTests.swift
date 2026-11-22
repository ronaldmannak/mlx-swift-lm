// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A small text configuration with `Hq` query and `Hkv` key-value heads of `D` dimensions.
private func textConfiguration(
    Hq: Int = 8, Hkv: Int = 2, D: Int = 128, R: Int = 24, window: Int = 0,
    experts: Int = 2, topK: Int = 1, routeScale: Float = 1
) throws -> Inkling2TextConfiguration {
    try JSONDecoder().decode(
        Inkling2TextConfiguration.self,
        from: Data(
            """
            {"hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 8,
             "num_attention_heads": \(Hq), "num_key_value_heads": \(Hkv), "head_dim": \(D),
             "swa_num_attention_heads": \(Hq), "swa_num_key_value_heads": \(Hkv),
             "swa_head_dim": \(D), "sliding_window_size": \(max(window, 1)),
             "local_layer_ids": \(window > 0 ? "[0]" : "[]"), "d_rel": 4, "rel_extent": \(R),
             "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
             "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8,
             "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 6,
             "n_routed_experts": \(experts), "num_experts_per_tok": \(topK),
             "intermediate_size": 64, "mlp_gate_softcap": 4, "mlp_up_softcap": 25,
             "route_scale": \(routeScale), "final_logit_softcapping": 30}
            """.utf8))
}

/// Whether `a` and `b` hold the same bits.
private func sameBits(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.dtype == b.dtype, a.shape == b.shape else { return false }
    let width: DType =
        switch a.dtype.size {
        case 1: .uint8
        case 2: .uint16
        case 4: .uint32
        default: .uint64
        }
    return arrayEqual(a.view(dtype: width), b.view(dtype: width)).item(Bool.self)
}

/// Inkling 2's Metal kernels against the reference ops they replace.
@Suite(.serialized)
struct Inkling2KernelTests {
    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    @Test("Depth attention matches the reference", arguments: [(1, 1), (3, 1), (4, 3), (2, 4)])
    func depthAttention(blocks: Int, queries: Int) throws {
        MLXRandom.seed(7)
        let D = 2304
        let blockStates = inkling2UnitRMS(MLXRandom.normal([blocks, 2, 5, D]))
        let partial = inkling2UnitRMS(MLXRandom.normal([2, 5, D]))
        let query = MLXRandom.normal([queries, D]) * 0.05
        let kernel = inkling2DepthAttentionKernel(
            blocks: blockStates, partial: partial, query: query, scale: pow(Float(D), -0.5))
        let reference = Device.withDefaultDevice(.cpu) {
            inkling2DepthAttention(blocks: blockStates, partial: partial, query: query)
        }
        #expect(kernel.shape == [2, 5, queries, D])
        #expect(maxDifference(kernel, reference) < 1e-4)
    }

    /// Prefill attention of `L` queries at the end of `S` keys, against the reference path.
    @Test(
        "Relative attention matches the reference",
        arguments: [(37, 37, 0, 0), (40, 100, 16, 0), (9, 60, 0, 0), (33, 45, 8, 5)])
    func relativeAttention(L: Int, S: Int, window: Int, padding: Int) throws {
        MLXRandom.seed(11)
        let (B, Hq, Hkv, D, R) = (2, 8, 2, 128, 24)
        let args = try textConfiguration(Hq: Hq, Hkv: Hkv, D: D, R: R, window: window)
        let attention = Inkling2Attention(args, sliding: window > 0)
        func clipped(_ shape: [Int]) -> MLXArray {
            let x = MLXRandom.normal(shape)
            return (x / sqrt((x * x).sum(axis: -1, keepDims: true)) * 3).asType(.bfloat16)
        }
        let queries = clipped([B, Hq, L, D])
        let keys = clipped([B, Hkv, S, D])
        let values = inkling2UnitRMS(MLXRandom.normal([B, Hkv, S, D])).asType(.bfloat16)
        let bias = (MLXRandom.normal([B, L, Hq, R]) * 0.5).asType(.bfloat16)
        let sinks = (MLXRandom.normal([B, L, Hq]) * 0.5).asType(.bfloat16)
        // Row 1 starts after `padding` keys that hold no position.
        let lastPosition = MLXArray([Float(S - 1), Float(S - 1 - padding)]).reshaped(B, 1)
        let queryPositions = lastPosition - Float(L - 1) + MLXArray(0 ..< L).asType(.float32)
        var stored = lastPosition - Float(S - 1) + MLXArray(0 ..< S).asType(.float32) + 1
        stored = MLX.maximum(stored, MLXArray(Float(0)))

        let kernel = inkling2RelativeAttentionKernel(
            queries: queries, keys: keys, values: values, positionBias: bias, sinks: sinks,
            keyStart: MLXArray([Int32(0), Int32(padding)]), scale: attention.scale,
            window: window)
        let reference = Device.withDefaultDevice(.cpu) {
            attention.attention(
                queries: queries, keys: keys, values: values, positionBias: bias, sinks: sinks,
                queryPositions: queryPositions, storedKeyPositions: stored, forceReference: true)
        }
        #expect(kernel.shape == [B, L, Hq, D])
        // The kernel holds q, k, v and the probabilities in half precision.
        #expect(maxDifference(kernel, reference) < 2e-2)
    }

    /// A small bfloat16 Inkling 2 Mini whose looped layers complete depth blocks, with the
    /// router, depth queries and experts in float32 as the checkpoints keep them.
    private func bfloat16Model(experts: Int = 2, topK: Int = 2) throws -> Inkling2Model {
        let model = Inkling2Model(
            try JSONDecoder().decode(
                Inkling2Configuration.self,
                from: Data(
                    """
                    {"model_type": "inkling2_loop", "text_config": {
                        "hidden_size": 256, "num_hidden_layers": 8, "vocab_size": 128,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
                        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
                        "swa_head_dim": 64, "sliding_window_size": 16,
                        "local_layer_ids": [0, 1, 2, 3, 4, 6], "d_rel": 8, "rel_extent": 32,
                        "log_scaling_n_floor": 16, "log_scaling_alpha": 0.1,
                        "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 3.0,
                        "rel_l2_clip_threshold": 2.0, "sconv_kernel_size": 4,
                        "n_layers_per_attn_res_block": 3, "n_routed_experts": \(experts),
                        "num_experts_per_tok": \(topK), "intermediate_size": 128,
                        "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.0,
                        "final_logit_softcapping": 30.0,
                        "loop_config": {"core_start": 3, "core_end": 6, "num_passes": 3,
                            "gate_rank": 8}}}
                    """.utf8)))
        let weights = model.parameters().flattened().map { name, value -> (String, MLXArray) in
            if name.hasSuffix("norm.weight") || name.hasSuffix("l2_clip.scale") {
                return (name, (1 + 0.1 * MLXRandom.normal(value.shape)).asType(.bfloat16))
            }
            let fanIn = value.ndim > 1 ? value.dim(-1) : 16
            let random = MLXRandom.normal(value.shape) * pow(Float(fanIn), -0.5)
            // The router and the depth queries stay float32, as in the checkpoints, and so do
            // the experts: the CPU's gathered matmul takes float32 only.
            let keepsFloat32 =
                name.contains("mlp.gate.") || name.contains(".mlp.experts.")
                || name.hasSuffix(".query")
            return (name, keepsFloat32 ? random : random.asType(.bfloat16))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
        return model
    }

    @Test("A bfloat16 model prefills through the kernels as through the reference path")
    func modelPrefill() throws {
        MLXRandom.seed(0)
        let model = try bfloat16Model()
        let tokens = MLXRandom.randInt(Int32(0) ..< 120, [2, 48])

        func prefill() throws -> [MLXArray] {
            let cache = try model.newCache(parameters: nil)
            let chunked = [
                model(tokens[0..., ..<20], cache: cache), model(tokens[0..., 20...], cache: cache),
            ]
            return [model(tokens, cache: nil), concatenated(chunked, axis: 1)]
        }
        let kernels = try prefill()
        let reference = try Device.withDefaultDevice(.cpu) { try prefill() }
        for (kernel, expected) in zip(kernels, reference) {
            // bfloat16 activations through eight layers and a loop, on two devices. With the
            // kernels off the GPU and CPU already differ by about 0.18 here. Every token takes
            // both experts, so no near tie in routing can flip and amplify the difference.
            #expect(maxDifference(kernel, expected) < 0.3)
        }
    }

    @Test(
        "The short convolution kernel matches its ops",
        arguments: [(1, true), (1, false), (2, true), (5, true), (33, false), (33, true)])
    func shortConvolution(length: Int, hasState: Bool) throws {
        MLXRandom.seed(3)
        for (channels, dtype) in [(1024, DType.bfloat16), (2304, .bfloat16), (64, .float32)] {
            let convolution = Inkling2ShortConvolution(dimensions: channels, kernelSize: 4)
            // A new module starts in training, which skips the kernels; loading ends it.
            convolution.train(false)
            convolution.conv1d.update(
                parameters: ModuleParameters.unflattened([
                    "weight": (MLXRandom.normal([channels, 4, 1]) * 0.5).asType(dtype)
                ]))
            let x = MLXRandom.normal([2, length, channels]).asType(dtype)
            let state = hasState ? MLXRandom.normal([2, 3, channels]) : nil
            let (out, next) = convolution(x, state: state, mask: nil)
            let (expected, expectedNext) = convolution(
                x, state: state, mask: nil, forceReference: true)
            #expect(sameBits(out, expected))
            #expect(sameBits(next, expectedNext))
        }
    }

    /// Keys at random positions, some empty (0) and some ahead of the query, as a wrapped
    /// rotating cache holds them.
    @Test(
        "The decode mask kernel matches its ops",
        arguments: [(0, 40, 24), (16, 40, 24), (16, 10, 32), (0, 300, 64)])
    func decodeMask(window: Int, keys: Int, extent: Int) throws {
        MLXRandom.seed(5)
        let args = try textConfiguration(R: extent, window: window)
        let attention = Inkling2Attention(args, sliding: window > 0)
        attention.train(false)
        let query = Float(keys + 7)
        let bias = MLXRandom.normal([1, 1, 8, extent]).asType(.bfloat16)
        let stored = MLXRandom.randInt(0 ..< Int32(query + 4), [1, keys]).asType(.float32)
        let positions = MLXArray([query]).reshaped(1, 1)
        let mask = attention.decodeMask(
            positionBias: bias, queryPositions: positions, storedKeyPositions: stored)
        let expected = attention.decodeMask(
            positionBias: bias, queryPositions: positions, storedKeyPositions: stored,
            forceReference: true)
        #expect(sameBits(mask, expected))
    }

    @Test(
        "The router kernel matches the fused router it replaces",
        arguments: [(128, 8), (16, 4), (2, 2)])
    func router(experts: Int, topK: Int) throws {
        MLXRandom.seed(9)
        let router = Inkling2Router(
            try textConfiguration(experts: experts, topK: topK, routeScale: 2.5))
        router.train(false)
        // Each expert's twin ties it, so the order of equal scores is checked too.
        func twins(_ x: MLXArray) -> MLXArray {
            concatenated([x[..<(experts / 2)], x[..<(experts / 2)]], axis: 0)
        }
        router.update(
            parameters: ModuleParameters.unflattened([
                "weight": twins(MLXRandom.normal([experts, 64]) * 0.2),
                "e_score_correction_bias": twins(MLXRandom.normal([experts]) * 0.01),
            ]))
        for _ in 0 ..< 16 {
            let x = MLXRandom.normal([1, 1, 64]).asType(.bfloat16)
            let (indices, weights) = router(x)
            let (expectedIndices, expectedWeights) = router(x, forceReference: true)
            #expect(sameBits(indices, expectedIndices))
            #expect(sameBits(weights, expectedWeights))
        }
    }

    @Test(
        "The expert combine kernel matches its ops",
        arguments: [[1, 1, 8, 2304], [2, 3, 8, 64], [1, 5, 2, 128]])
    func expertCombine(shape: [Int]) throws {
        MLXRandom.seed(13)
        let moe = Inkling2SparseMoE(try textConfiguration())
        moe.train(false)
        let outputs = MLXRandom.normal(shape).asType(.bfloat16)
        let weights = MLXRandom.uniform(0 ..< 2, Array(shape.dropLast()))
        for dtype in [DType.bfloat16, .float32] {
            #expect(
                sameBits(
                    moe.combine(outputs, weights: weights, dtype: dtype),
                    moe.combine(outputs, weights: weights, dtype: dtype, forceReference: true)))
        }
    }

    /// Rows of 16 and 64 values are summed in order; longer rows in runs of four per lane.
    @Test(
        "The L2 clip kernel matches its ops",
        arguments: [(24, 16), (24, 64), (8, 128), (24, 128), (3, 200)])
    func l2Clip(heads: Int, dimensions: Int) throws {
        MLXRandom.seed(19)
        let x = (MLXRandom.normal([1, 1, heads, dimensions]) * 3).asType(.bfloat16)
        let scale = (1 + 0.1 * MLXRandom.normal([heads, dimensions])).asType(.bfloat16)
        // A low threshold clips some rows and leaves others.
        let threshold = Float(dimensions).squareRoot() * 3
        #expect(inkling2L2ClipKernelTakes(x))
        for scale in [scale, nil] {
            #expect(
                sameBits(
                    inkling2L2Clip(x, scale: scale, threshold: threshold),
                    inkling2L2Clip(x, scale: scale, threshold: threshold, useKernel: false)))
        }
    }

    /// Hundreds of rows, since a rounding difference in one row's norm can be rare.
    @Test("Norms without a ones weight, and with the weight after, give MLX's bits")
    func rmsNorm() throws {
        MLXRandom.seed(17)
        for dimensions in [16, 128, 2304] {
            let x = MLXRandom.normal([1, 512, dimensions]) * 3
            #expect(
                sameBits(
                    inkling2UnitRMS(x),
                    MLXFast.rmsNorm(x, weight: MLXArray.ones([dimensions]), eps: 1e-6)))
            for (weightType, dtype) in [
                (DType.bfloat16, DType.bfloat16), (.bfloat16, .float32), (.float32, .bfloat16),
            ] {
                let norm = RMSNorm(dimensions: dimensions, eps: 1e-6)
                norm.update(
                    parameters: ModuleParameters.unflattened([
                        "weight": (1 + 0.1 * MLXRandom.normal([dimensions])).asType(weightType)
                    ]))
                #expect(sameBits(inkling2Normed(norm, x, dtype: dtype), norm(x).asType(dtype)))
            }
        }
    }

    /// Prepared lengths give every decode step a mask, which sends it down the general
    /// path. One token per row is valid, so the mask hides nothing.
    @Test("Compiled decode steps give the general path's bits", arguments: [1, 2])
    func compiledDecode(batch: Int) throws {
        MLXRandom.seed(0)
        let model = try bfloat16Model(experts: 8, topK: 2)
        model.train(false)
        let tokens = MLXRandom.randInt(Int32(0) ..< 120, [batch, 60])
        func decode(general: Bool) throws -> [MLXArray] {
            let cache = try model.newCache(parameters: nil)
            var logits = [model(tokens[0..., ..<20], cache: cache)]
            for step in 20 ..< 60 {
                if general {
                    for case let convolution as ArraysCache in cache {
                        convolution.prepare(lengths: Array(repeating: 1, count: batch))
                    }
                }
                logits.append(model(tokens[0..., step ..< step + 1], cache: cache))
                eval(logits.last!)
            }
            return logits
        }
        let compiled = try decode(general: false)
        let general = try decode(general: true)
        for (a, b) in zip(compiled, general) {
            #expect(sameBits(a, b))
        }
    }

    /// A rewind window sends a few tokens of one row through the compiled traces, as a
    /// speculative round's verify does. The same attention caches with plain convolution
    /// states send them down the general path.
    @Test("Compiled verify passes give the general path's bits")
    func compiledVerify() throws {
        MLXRandom.seed(0)
        let model = try bfloat16Model(experts: 8, topK: 2)
        model.train(false)
        let tokens = MLXRandom.randInt(Int32(0) ..< 120, [1, 96])
        var parameters = GenerateParameters(temperature: 0)
        parameters.kvCache = KVCacheConfiguration(rewind: try .init(maxTokens: 16))
        let rewindable = try model.newCache(parameters: parameters)
        #expect(rewindable[2] is Inkling2ConvolutionCache)
        let reference = try model.newCache(parameters: parameters)
        let plain = stride(from: 0, to: reference.count, by: 6).flatMap {
            [reference[$0], reference[$0 + 1], ArraysCache(size: 4)]
        }
        // Past the sliding window and the bias extent, through the loop.
        var start = 0
        for length in [20, 2, 9, 1, 3, 16, 5, 8, 1, 7, 4, 6, 9, 5] {
            let chunk = tokens[0..., start ..< start + length]
            let a = model(chunk, cache: rewindable)
            let b = model(chunk, cache: plain)
            eval(a, b)
            #expect(sameBits(a, b), "\(length) tokens at \(start)")
            start += length
        }
        #expect(start == tokens.dim(1))
    }

    @Test("Training takes the reference paths, so gradients flow through attention")
    func trainingGradients() throws {
        MLXRandom.seed(0)
        let model = try bfloat16Model()
        // On the GPU, a bfloat16 prefill would take both kernels, which have no gradient.
        let tokens = MLXRandom.randInt(Int32(0) ..< 120, [2, 24])
        model.train()
        let lossAndGradient = valueAndGrad(model: model) { model, tokens, _ in
            model(tokens, cache: nil).asType(.float32).square().mean()
        }
        let (loss, gradients) = try withError {
            let (loss, gradients) = lossAndGradient(model, tokens, MLXArray(0))
            let flat = gradients.flattened()
            eval([loss] + flat.map(\.1))
            return (loss, flat)
        }
        #expect(loss.item(Float.self).isFinite)
        #expect(gradients.allSatisfy { all(isFinite($0.1)).item(Bool.self) })
        // The depth queries only reach the loss through depth attention.
        let depthQueries = gradients.filter { $0.0.hasSuffix(".query") }
        #expect(!depthQueries.isEmpty)
        #expect(depthQueries.contains { abs($0.1).max().item(Float.self) > 0 })
    }
}
