// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

struct Inkling2QuantizedVisionTests {
    @Test(
        "A loaded quantized patch embedding keeps fractional and negative pixels",
        arguments: [DType.float32, .float16, .bfloat16], [4, 8])
    func checkpointPreservesFloatingPixels(dtype: DType, bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try configuration()
            let source = Inkling2VLM(configuration)
            initialize(source, dtype: dtype)
            let path = "model.vision_tower.embeddings.patch_embedding"
            quantize(model: source) { name, _ in
                name == path ? (64, bits, .affine) : nil
            }
            let sourceTower = try #require(source.model.visionTower)
            let patch = try #require(sourceTower.embeddings.patchEmbedding as? QuantizedLinear)
            #expect(patch.weight.dtype == .uint32)
            #expect(patch.scales.dtype == dtype)

            let directory = FileManager.default.temporaryDirectory.appending(
                path: "inkling2-quantized-vision-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
            try save(arrays: weights, url: directory.appending(path: "model.safetensors"))
            let loaded = Inkling2VLM(configuration)
            try loadWeights(
                modelDirectory: directory, model: loaded,
                perLayerQuantization: .init(
                    perLayerQuantization: [
                        path: .quantize(.init(groupSize: 64, bits: bits))
                    ]))
            let tower = try #require(loaded.model.visionTower)
            let loadedPatch = try #require(tower.embeddings.patchEmbedding as? QuantizedLinear)
            #expect(loadedPatch.weight.dtype == .uint32)
            #expect(loadedPatch.scales.dtype == dtype)

            // The reference uses the same checkpoint with only the packed patch weight
            // dequantized, so it takes the ordinary floating-point activation path.
            var denseWeights = weights
            denseWeights[path + ".weight"] = dequantized(
                patch.weight, scales: patch.scales, biases: patch.biases,
                groupSize: patch.groupSize, bits: patch.bits, mode: patch.mode, dtype: dtype)
            denseWeights.removeValue(forKey: path + ".scales")
            denseWeights.removeValue(forKey: path + ".biases")
            let reference = Inkling2VLM(configuration)
            try reference.update(
                parameters: ModuleParameters.unflattened(denseWeights), verify: [.all])
            let referenceTower = try #require(reference.model.visionTower)
            let pixelCount = 3 * 16 * 16
            let pixelValues: [Float] = (0 ..< pixelCount).map { (index: Int) -> Float in
                let value = (index * 13) % 63 - 31
                return Float(value) / 16
            }
            let pixels = MLXArray(pixelValues, [3, 16, 16])
            let expected = try referenceTower(pixels)
            let actual = try withError {
                let output = try tower(pixels)
                eval(output)
                return output
            }
            #expect(actual.shape == [1, configuration.text.hiddenSize])
            #expect(actual.dtype == dtype)
            let tolerance: Double = dtype == .float32 ? 0.0001 : dtype == .float16 ? 0.004 : 0.02
            #expect(
                allClose(actual, expected, rtol: tolerance, atol: tolerance)
                    .item(Bool.self))
            let rounded = try referenceTower(round(pixels))
            #expect(
                abs(expected.asType(.float32) - rounded.asType(.float32)).max().item(Float.self)
                    > 0.001)
        }
    }

    private func initialize(_ model: Module, dtype: DType) {
        let parameters = model.parameters().flattened().map { name, value -> (String, MLXArray) in
            let array: MLXArray
            if name.contains("norm") && name.hasSuffix("weight") || name.hasSuffix("lambda_1")
                || name.hasSuffix("lambda_2")
            {
                array = .ones(value.shape)
            } else if name.hasSuffix(".bias") {
                array = .zeros(value.shape)
            } else if name.contains(".merger.linear_"), name.hasSuffix(".weight") {
                let width = value.dim(-1)
                array = MLXArray(
                    (0 ..< value.size).map { index in
                        Float(index % width == index / width ? 1 : 0)
                    }, value.shape)
            } else if name.hasSuffix("patch_embedding.weight") {
                array = MLXArray(
                    (0 ..< value.size).map { index in
                        Float((index % 192 + (index / 192) * 3) % 16 - 8) * 0.5
                    }, value.shape)
            } else {
                array =
                    sin(MLXArray(0 ..< value.size).asType(.float32) * 0.37)
                    .reshaped(value.shape) * 0.03
            }
            return (name, array.asType(dtype))
        }
        model.update(parameters: ModuleParameters.unflattened(parameters))
        model.train(false)
    }

    private func configuration() throws -> Inkling2Configuration {
        try JSONDecoder().decode(
            Inkling2Configuration.self,
            from: Data(
                """
                {"model_type": "inkling2", "image_token_id": 63,
                 "vision_config": {"hidden_size": 32, "num_hidden_layers": 1,
                    "num_attention_heads": 2, "native_patch_size": 8, "adapter_merge_size": 2,
                    "posemb_grid_size": 2, "projection_hidden_dim": 32},
                 "text_config": {"hidden_size": 32, "num_hidden_layers": 1, "vocab_size": 64,
                    "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
                    "swa_num_attention_heads": 2, "swa_num_key_value_heads": 2, "swa_head_dim": 16,
                    "sliding_window_size": 8, "local_layer_ids": [0], "d_rel": 4, "rel_extent": 8,
                    "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
                    "qk_l2_clip_threshold": 3.0, "rel_l2_clip_threshold": 2.0,
                    "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 1,
                    "n_routed_experts": 2, "num_experts_per_tok": 2, "intermediate_size": 64,
                    "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 2.0,
                    "final_logit_softcapping": 30.0}}
                """.utf8))
    }
}
