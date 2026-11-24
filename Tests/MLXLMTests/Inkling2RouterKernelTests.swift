// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct Inkling2RouterKernelTests {
    private func configuration(experts: Int, topK: Int) throws -> Inkling2TextConfiguration {
        try JSONDecoder().decode(
            Inkling2TextConfiguration.self,
            from: Data(
                """
                {"hidden_size": 16, "num_hidden_layers": 1, "vocab_size": 32,
                 "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                 "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1,
                 "swa_head_dim": 8, "sliding_window_size": 16, "local_layer_ids": [],
                 "d_rel": 4, "rel_extent": 16, "log_scaling_n_floor": 8,
                 "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
                 "qk_l2_clip_threshold": 3.0, "rel_l2_clip_threshold": 2.0,
                 "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 1,
                 "n_routed_experts": \(experts), "num_experts_per_tok": \(topK),
                 "intermediate_size": 16, "mlp_gate_softcap": 4.0,
                 "mlp_up_softcap": 25.0, "route_scale": 2.0,
                 "final_logit_softcapping": 30.0}
                """.utf8))
    }

    private func expectOriginal(logits: MLXArray, bias: MLXArray, scale: Float = 1) throws {
        let expected = inkling2RouterKernel(logits: logits, bias: bias, k: 8, routeScale: scale)
        let actual = try #require(
            inkling2SIMDRouterKernel(logits: logits, bias: bias, k: 8, routeScale: scale))
        #expect(actual.indices.shape == expected.indices.shape)
        #expect(actual.weights.shape == expected.weights.shape)
        #expect(arrayEqual(actual.indices, expected.indices).item(Bool.self))
        #expect(
            arrayEqual(actual.weights.view(dtype: .uint32), expected.weights.view(dtype: .uint32))
                .item(Bool.self))
    }

    @Test(
        "SIMD routing keeps random expert choices and normalized weight bits",
        arguments: [(1, Float(1)), (3, Float(2.5)), (8, Float(0)), (6, Float(-0.75))])
    func random(rows: Int, scale: Float) throws {
        MLXRandom.seed(111)
        let logits = MLXRandom.normal([rows, 128]) * 3
        let bias = MLXRandom.normal([128]) * 0.3
        try expectOriginal(logits: logits, bias: bias, scale: scale)
    }

    @Test("Equal router scores prefer lower expert indices")
    func ties() throws {
        let logits = MLXArray.zeros([2, 1, 128], dtype: .float32)
        let bias = MLXArray.zeros([128], dtype: .float32)
        try expectOriginal(logits: logits, bias: bias)
        let result = try #require(
            inkling2SIMDRouterKernel(logits: logits, bias: bias, k: 8, routeScale: 1))
        #expect(result.indices.reshaped(-1).asArray(UInt32.self) == Array(0 ..< 8) + Array(0 ..< 8))
        #expect(all(result.weights .== 0.125).item(Bool.self))
    }

    @Test("NaNs, infinities, signed zero and extreme logits keep the ordered-key behavior")
    func exceptionalValues() throws {
        var logits = [Float](repeating: 0, count: 128)
        var bias = [Float](repeating: 0, count: 128)
        bias[2] = Float(bitPattern: 0x7fc0_0123)
        bias[71] = Float(bitPattern: 0xffc0_0321)
        bias[4] = .infinity
        bias[5] = -.infinity
        logits[8] = .infinity
        logits[9] = -.infinity
        logits[10] = -0.0
        logits[11] = 0.0
        logits[12] = Float.greatestFiniteMagnitude
        logits[13] = -Float.greatestFiniteMagnitude
        try expectOriginal(logits: MLXArray(logits), bias: MLXArray(bias), scale: 2)
        logits[3] = Float(bitPattern: 0x7fc0_0042)
        logits[97] = Float(bitPattern: 0xffc0_0061)
        try expectOriginal(logits: MLXArray(logits), bias: MLXArray(bias), scale: 0.5)
        // Underflowed sigmoid scores normalize as 0/0, matching the original router.
        logits = [Float](repeating: -.infinity, count: 128)
        bias = (0 ..< 128).map { $0.isMultiple(of: 2) ? Float(0) : -Float(0) }
        try expectOriginal(logits: MLXArray(logits), bias: MLXArray(bias))
    }

    @Test("Routing preserves leading dimensions and noncontiguous input views")
    func leadingDimensionsAndStrides() throws {
        MLXRandom.seed(112)
        try expectOriginal(
            logits: MLXRandom.normal([2, 3, 128]), bias: MLXRandom.normal([128]))
        try expectOriginal(
            logits: MLXRandom.normal([2, 256])[0..., .stride(from: 0, by: 2)],
            bias: MLXRandom.normal([256])[.stride(from: 0, by: 2)])
    }

    @Test(
        "Single-row GPU training differentiates normalized router weights",
        arguments: [(128, 8), (16, 4)])
    func singleRowTrainingGradients(experts: Int, topK: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let router = Inkling2Router(try configuration(experts: experts, topK: topK))
            var weights = [Float](repeating: 0, count: experts * 16)
            var bias = [Float](repeating: 0, count: experts)
            let scores = (0 ..< topK).map { expert -> Float in
                let logit = Float(expert + 1) * 0.1
                weights[expert * 16] = logit
                bias[expert] = 10
                return 1 / (1 + exp(-logit))
            }
            router.update(
                parameters: ModuleParameters.unflattened([
                    ("weight", MLXArray(weights).reshaped(experts, 16)),
                    ("e_score_correction_bias", MLXArray(bias)),
                ]))
            router.train()
            let input = MLXArray([Float(1)] + [Float](repeating: 0, count: 15))
                .reshaped(1, 1, 16)
            let rewards = MLXArray((0 ..< experts).map { Float($0 + 1) })
            #expect(supportsFusedRouterTopK(MLXArray.zeros([1, 1, experts]), k: topK))

            let lossAndGradient = valueAndGrad(model: router) { router, input, rewards in
                let routed = router(input)
                return (routed.weights * take(rewards, routed.indices, axis: 0)).sum()
            }
            let (loss, gradients) = try withError {
                let (loss, gradients) = lossAndGradient(router, input, rewards)
                let flat = gradients.flattened()
                eval([loss] + flat.map(\.1))
                return (loss, flat)
            }
            let gradient = try #require(gradients.first { $0.0 == "weight" }?.1)
            let scoreTotal = scores.reduce(0, +)
            let weightedScore = scores.enumerated().reduce(Float(0)) {
                $0 + $1.element * Float($1.offset + 1)
            }
            let weightedMean = weightedScore / scoreTotal
            var expectedLogitGradient = [Float](repeating: 0, count: experts)
            for expert in 0 ..< topK {
                // Expert choices are fixed by the bias; selected sigmoid weights retain gradients.
                let score = scores[expert]
                expectedLogitGradient[expert] =
                    router.routeScale * score * (1 - score) / scoreTotal
                    * (Float(expert + 1) - weightedMean)
            }
            // Apply the analytical derivative through the same GPU outer product, including TF32.
            let expectedGradient = matmul(
                MLXArray(expectedLogitGradient).reshaped(experts, 1), input.reshaped(1, 16))
            let maximumError = abs(gradient - expectedGradient).max().item(Float.self)
            #expect(loss.item(Float.self).isFinite)
            #expect(abs(loss.item(Float.self) - router.routeScale * weightedMean) < 1e-5)
            #expect(all(isFinite(gradient)).item(Bool.self))
            #expect(abs(gradient).max().item(Float.self) > 1e-3)
            #expect(maximumError < 1e-5, "Router weight gradient maximum error: \(maximumError)")
            let selected = router(input).indices.asType(.int32).asArray(Int32.self).map(Int.init)
            #expect(Set(selected) == Set(0 ..< topK))
        }
    }

    @Test("Unsupported expert counts, row counts, ranks, dtypes and CPU use the original path")
    func fallbacks() {
        let logits = MLXArray.zeros([1, 128], dtype: .float32)
        let bias = MLXArray.zeros([128], dtype: .float32)
        #expect(inkling2SIMDRouterKernel(logits: logits, bias: bias, k: 4, routeScale: 1) == nil)
        #expect(
            inkling2SIMDRouterKernel(
                logits: MLXArray.zeros([1, 64]), bias: MLXArray.zeros([64]), k: 8, routeScale: 1)
                == nil)
        #expect(
            inkling2SIMDRouterKernel(
                logits: MLXArray.zeros([9, 128]), bias: bias, k: 8, routeScale: 1) == nil)
        #expect(
            inkling2SIMDRouterKernel(
                logits: logits.asType(.bfloat16), bias: bias, k: 8, routeScale: 1) == nil)
        #expect(
            inkling2SIMDRouterKernel(
                logits: logits, bias: bias.asType(.bfloat16), k: 8, routeScale: 1) == nil)
        #expect(
            inkling2SIMDRouterKernel(
                logits: logits, bias: MLXArray.zeros([1, 128]), k: 8, routeScale: 1) == nil)
        #expect(
            inkling2SIMDRouterKernel(logits: MLXArray(Float(1)), bias: bias, k: 8, routeScale: 1)
                == nil)
        #expect(
            Device.withDefaultDevice(.cpu) {
                inkling2SIMDRouterKernel(logits: logits, bias: bias, k: 8, routeScale: 1)
            } == nil)
    }
}
