// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite struct Qwen3VLHiddenStateTests {

    private func makeTinyModel() throws -> Qwen3VL {
        let json = """
            {
                "model_type": "qwen3_vl",
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
                "vocab_size": 512,
                "text_config": {
                    "model_type": "qwen3_vl",
                    "hidden_size": 64,
                    "num_hidden_layers": 4,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "vocab_size": 512,
                    "max_position_embeddings": 4096,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 100000.0,
                    "rope_scaling": {
                        "type": "default",
                        "mrope_section": [4, 2, 2]
                    }
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 2,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 64,
                    "deepstack_visual_indexes": [0, 1]
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 1)) { Qwen3VL(config) }
    }

    @Test("The last hidden state is the decoder output before the final norm")
    func lastHiddenStatePrecedesFinalNorm() throws {
        let model = try makeTinyModel()
        // A norm weight of all ones would make normalizing twice look like normalizing once.
        let normKey = "language_model.model.norm.weight"
        let normWeight = MLXArray((0 ..< 64).map { 0.5 + Float($0) / 64 })
        model.update(parameters: ModuleParameters.unflattened([(normKey, normWeight)]))

        let tokens = MLXArray([3, 17, 42, 7, 99, 250].map { Int32($0) }).reshaped([1, 6])
        let hidden = model.lastHiddenState(tokens)
        #expect(hidden.shape == [1, 6, 64])

        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let head = try #require(
            parameters["language_model.lm_head.weight"]
                ?? parameters["language_model.model.embed_tokens.weight"])
        let normalized =
            hidden * rsqrt(mean(hidden * hidden, axis: -1, keepDims: true) + 1e-6) * normWeight
        let expected = matmul(normalized, head.T)

        let logits = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        #expect(abs(logits - expected).max().item(Float.self) < 1e-4)
    }
}
