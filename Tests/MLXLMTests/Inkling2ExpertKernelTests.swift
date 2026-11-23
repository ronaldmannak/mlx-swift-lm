// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private final class Inkling2CustomExpertProjection: QuantizedSwitchLinear {}

@Suite(.serialized)
struct Inkling2ExpertKernelTests {
    private func configuration() throws -> Inkling2TextConfiguration {
        try JSONDecoder().decode(
            Inkling2TextConfiguration.self,
            from: Data(
                """
                {"hidden_size":128,"num_hidden_layers":1,"vocab_size":8,
                 "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
                 "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":32,
                 "sliding_window_size":1024,"local_layer_ids":[0],
                 "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"rms_norm_eps":1e-6,
                 "qk_l2_clip_threshold":22.6,"rel_l2_clip_threshold":8,
                 "d_rel":4,"rel_extent":16,"sconv_kernel_size":4,
                 "n_layers_per_attn_res_block":6,"n_routed_experts":6,
                 "num_experts_per_tok":3,"intermediate_size":64,
                 "mlp_gate_softcap":4,"mlp_up_softcap":25,"route_scale":1,
                 "final_logit_softcapping":30}
                """.utf8))
    }

    private func model(
        mode: QuantizationMode = .nvfp4, bits: Int = 4, groupSize: Int = 16
    ) throws -> Inkling2SparseMoE {
        let model = Inkling2SparseMoE(try configuration())
        model.router.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([6, 128]) * 0.1,
                "e_score_correction_bias": MLXRandom.normal([6]) * 0.1,
            ]))
        model.experts.gateUpProjection.update(
            parameters: ModuleParameters.unflattened([
                "weight": (MLXRandom.normal([6, 128, 128]) * 0.1).asType(.bfloat16)
            ]))
        model.experts.downProjection.update(
            parameters: ModuleParameters.unflattened([
                "weight": (MLXRandom.normal([6, 128, 64]) * 0.1).asType(.bfloat16)
            ]))
        quantize(model: model) { _, module in
            module is SwitchLinear ? (groupSize, bits, mode) : nil
        }
        if let gate = model.experts.gateUpProjection as? Inkling2ScaledSwitchLinear,
            let down = model.experts.downProjection as? Inkling2ScaledSwitchLinear
        {
            gate.update(
                parameters: ModuleParameters.unflattened([
                    "global_scale": MLXArray([Float(0.503), 0.997, 1.003, 1.117, 1.249, 1.497])
                        * 2688
                ]))
            down.update(
                parameters: ModuleParameters.unflattened([
                    "global_scale": MLXArray([Float(1.497), 1.249, 1.117, 1.003, 0.997, 0.503])
                        * 2688
                ]))
        }
        model.train(false)
        return model
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func parity(
        mode: QuantizationMode, bits: Int, groupSize: Int, dtype: DType,
        convolutionType: DType, batch: Int, keepWindow: Bool
    ) throws {
        MLXRandom.seed(821)
        let model = try model(mode: mode, bits: bits, groupSize: groupSize)
        let x = MLXRandom.normal([batch, 1, 128]).asType(dtype)
        let residual = MLXRandom.normal([batch, 1, 128])
        let state = MLXRandom.normal([batch, 3, 128])
        let convolutionWeight = (MLXRandom.normal([128, 4, 1]) * 0.25).asType(convolutionType)
        let fused = try #require(
            model.fusedExpertDecode(
                x, residual: residual, state: state, convolutionWeight: convolutionWeight,
                keepWindow: keepWindow))
        let (indices, weights) = model.router(x)
        let outputs = model.experts(x, indices: indices)
        let combined = model.combine(outputs, weights: weights, dtype: x.dtype)
        let convolution = Inkling2ShortConvolution(dimensions: 128, kernelSize: 4)
        convolution.conv1d.update(
            parameters: ModuleParameters.unflattened(["weight": convolutionWeight]))
        convolution.train(false)
        let expected = convolution(
            combined, state: state, mask: nil, keepWindow: keepWindow)
        let expectedHidden = (residual + expected.0.asType(.float32)).asType(dtype)

        #expect(fused.0.shape == x.shape)
        #expect(fused.0.dtype == dtype)
        #expect(fused.1.dtype == .float32)
        #expect(fused.1.shape == [batch, keepWindow ? 4 : 3, 128])
        // The dot-product reduction differs from gathered matmul; allow a few BF16 steps.
        #expect(maxDifference(fused.0, expectedHidden) < 2e-2)
        #expect(maxDifference(fused.1, expected.1) < 2e-2)
        if keepWindow {
            #expect(arrayEqual(fused.1[0..., ..<3, 0...], state).item(Bool.self))
        } else {
            #expect(arrayEqual(fused.1[0..., ..<2, 0...], state[0..., 1..., 0...]).item(Bool.self))
        }
    }

    @Test(
        "Fused experts match routed projections, convolution and residual",
        arguments: [
            (QuantizationMode.nvfp4, 4, 16), (.mxfp4, 4, 32), (.mxfp8, 8, 32),
            (.affine, 4, 64), (.affine, 8, 32),
        ])
    func quantizedExperts(mode: QuantizationMode, bits: Int, groupSize: Int) throws {
        for keepWindow in [false, true] {
            try parity(
                mode: mode, bits: bits, groupSize: groupSize, dtype: .bfloat16,
                convolutionType: .bfloat16, batch: 2, keepWindow: keepWindow)
        }
    }

    @Test("Float16 activations retain float32 convolution precision")
    func mixedPrecisionConvolution() throws {
        try parity(
            mode: .nvfp4, bits: 4, groupSize: 16, dtype: .float16,
            convolutionType: .float32, batch: 1, keepWindow: true)
    }

    @Test("Gate and down projections may use different quantization formats")
    func mixedExpertFormats() throws {
        MLXRandom.seed(822)
        let model = try model()
        let down = SwitchLinear(inputDims: 64, outputDims: 128, numExperts: 6, bias: false)
        down.update(
            parameters: ModuleParameters.unflattened([
                "weight": (MLXRandom.normal([6, 128, 64]) * 0.1).asType(.bfloat16)
            ]))
        let quantizedDown = QuantizedSwitchLinear(
            down, groupSize: 32, bits: 8, mode: .mxfp8)
        model.experts.update(
            modules: ModuleChildren(values: ["down_proj": .value(quantizedDown)]))
        model.train(false)
        let x = MLXRandom.normal([1, 1, 128]).asType(.bfloat16)
        let residual = MLXArray.zeros([1, 1, 128])
        let state = MLXArray.zeros([1, 3, 128])
        let convolution = MLXArray.zeros([128, 4, 1]).asType(.bfloat16)
        let fused = try #require(
            model.fusedExpertDecode(
                x, residual: residual, state: state, convolutionWeight: convolution,
                keepWindow: false))
        let expected = (model(x).asType(.float32) / sqrt(Float(5))).asType(.bfloat16)
        #expect(maxDifference(fused.0, expected) < 2e-2)
        #expect(maxDifference(fused.1[0..., 2 ..< 3, 0...], model(x)) < 2e-2)
    }

    @Test("Unsupported execution keeps the general expert path")
    func fallbackGuards() throws {
        MLXRandom.seed(823)
        let model = try model()
        let x = MLXArray.ones([1, 1, 128]).asType(.bfloat16)
        let state = MLXArray.zeros([1, 3, 128])
        let convolution = MLXArray.zeros([128, 4, 1]).asType(.bfloat16)
        func attempt(
            _ input: MLXArray, stateOverride: MLXArray? = nil, omitState: Bool = false
        ) -> (MLXArray, MLXArray)? {
            model.fusedExpertDecode(
                input, residual: MLXArray.zeros(input.shape),
                state: omitState ? nil : stateOverride ?? state,
                convolutionWeight: convolution, keepWindow: false)
        }
        #expect(attempt(x, omitState: true) == nil)
        #expect(attempt(x.asType(.float32)) == nil)
        #expect(attempt(MLXArray.ones([1, 2, 128]).asType(.bfloat16)) == nil)
        #expect(attempt(MLXArray.ones([32, 1, 128]).asType(.bfloat16)) == nil)
        #expect(attempt(x, stateOverride: state.asType(.bfloat16)) == nil)
        model.train()
        #expect(attempt(x) == nil)
        model.train(false)
        #expect(Device.withDefaultDevice(.cpu) { attempt(x) == nil })

        let dense = SwitchLinear(inputDims: 128, outputDims: 128, numExperts: 6, bias: false)
        model.experts.update(modules: ModuleChildren(values: ["gate_up_proj": .value(dense)]))
        model.train(false)
        #expect(attempt(x) == nil)
        let custom = Inkling2CustomExpertProjection(
            dense, groupSize: 16, bits: 4, mode: .nvfp4)
        model.experts.update(modules: ModuleChildren(values: ["gate_up_proj": .value(custom)]))
        model.train(false)
        #expect(attempt(x) == nil)
        let unsupported = QuantizedSwitchLinear(
            dense, groupSize: 64, bits: 2, mode: .affine)
        model.experts.update(
            modules: ModuleChildren(values: ["gate_up_proj": .value(unsupported)]))
        model.train(false)
        #expect(attempt(x) == nil)
    }
}
