import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct RoPEApplicationTests {

    /// Gemma3nAttention applies rope, updates the cache and applies rope again.
    /// Ensure that it is correctly implemented.  We can observe prefill vs single token
    /// generaton and they should produce the same answers if implemented correctly.
    @Test func gemma3nAttentionTest() {
        // Task-local rather than MLXRandom.seed: parallel tests must not share
        // (or perturb) the global random stream. The scope covers the module
        // initializer too, so the attention weights are pinned as well.
        withRandomState(MLXRandom.RandomState(seed: 42)) {
            let config = Gemma3nTextConfiguration()
            let attention = Gemma3nAttention(config, layerIdx: 3)

            #expect(!attention.isKvSharedLayer)

            let B = 1
            let L = 4
            let D = config.hiddenSize

            let x = MLXRandom.normal([B, L, D])
            eval(x)

            // Batch: process all L tokens at once with a causal mask
            let cacheBatch = KVCacheSimple()
            let causalMask = createAttentionMask(h: x, cache: cacheBatch)
            let outputBatch = attention(x, mask: causalMask, cache: cacheBatch)
            eval(outputBatch)

            // Sequential: process one token at a time (mask=.none since L=1 with cache)
            let cacheSeq = KVCacheSimple()
            var seqOutputs: [MLXArray] = []
            for i in 0 ..< L {
                let token = x[0..., i ..< (i + 1), 0...]
                let mask = createAttentionMask(h: token, cache: cacheSeq)
                let out = attention(token, mask: mask, cache: cacheSeq)
                seqOutputs.append(out)
            }
            let outputSeq = concatenated(seqOutputs, axis: 1)
            eval(outputSeq)

            // With correct RoPE these would match.  The buggy code would use
            // different offsets for keys/queries.  The tolerance is wider on
            // machines whose float32 matmuls run as TF32; see MatmulPrecision.
            let match = allClose(
                outputBatch, outputSeq,
                atol: MatmulPrecision.tolerance(float32: 1e-4, reduced: 5e-3))
            eval(match)
            #expect(
                match.item(Bool.self),
                """
                batch and sequential attention differ by up to \
                \(abs(outputSeq - outputBatch).max().item(Float.self))
                """)
        }
    }
}

/// Rows at different positions, left-padded to one width, as a batched cache holds them.
private final class RowOffsetCache: KVCacheSimple, BatchPositionedKVCache {
    private let leftPadding: [Int32]

    /// Joins prefilled single-sequence caches into one batch.
    init(rows: [any KVCache]) {
        let longest = rows.map(\.offset).max() ?? 0
        leftPadding = rows.map { Int32(longest - $0.offset) }
        super.init()
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for row in rows {
            let widths: [IntOrPair] = [0, 0, .init((longest - row.offset, 0)), 0]
            keys.append(MLX.padded(row.state[0], widths: widths))
            values.append(MLX.padded(row.state[1], widths: widths))
        }
        state = [concatenated(keys), concatenated(values)]
    }

    var batchOffset: MLXArray { MLXArray(leftPadding.map { Int32(offset) - $0 }) }

    override var ropeOffset: RoPEOffset { .batch(batchOffset) }

    override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .array(
            createCausalMask(
                n: n, offset: offset, windowSize: windowSize, leftPadding: MLXArray(leftPadding)))
    }
}

/// An image row and a text row prefilled separately, then decoded as one batch, must each get
/// the logits they get on their own.
@Suite(.serialized)
struct VLMRowPositionTests {

    @Test("FastVLM positions each row of a batch at its own offset")
    func fastVLM() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                FastVLMConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "fastvlm", "hidden_size": 16, "num_hidden_layers": 1,
                      "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                      "vocab_size": 16, "eos_token_id": 15, "mm_projector_type": "mlp2x_gelu",
                      "mm_hidden_size": 16, "tokenizer_model_max_length": 1024,
                      "tokenizer_padding_side": "right",
                      "vision_config": {
                        "cls_ratio": 1, "down_patch_size": 3, "down_stride": 2,
                        "downsamples": [false, false], "embed_dims": [16, 16],
                        "hidden_size": 16, "image_size": 128, "intermediate_size": 32,
                        "layers": [1, 1], "layer_scale_init_value": 0.00001,
                        "mlp_ratios": [2, 2], "num_classes": 16, "patch_size": 99,
                        "pos_embs_shapes": [null, null], "projection_dim": 16,
                        "repmixer_kernel_size": 3, "token_mixers": ["repmixer", "repmixer"]
                      }
                    }
                    """.utf8))
            let model = withRandomState(MLXRandom.RandomState(seed: 501)) {
                let model = FastVLM(configuration)
                eval(model)
                return model
            }
            let inputs = [
                LMInput(
                    text: .init(tokens: MLXArray([1, -200, 2]).reshaped([1, 3])),
                    image: .init(pixels: MLXArray.ones([1, 3, 16, 16]), frames: [])),
                LMInput(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])),
            ]
            try expectRowsMatchSingles(model, inputs: inputs, positions: [18, 3])
        }
    }

    @Test("Gemma 3 positions each row of a batch at its own offset")
    func gemma3() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                Gemma3Configuration.self,
                from: Data(
                    """
                    {
                      "model_type": "gemma3", "mm_tokens_per_image": 4,
                      "text_config": {
                        "model_type": "gemma3_text", "hidden_size": 16, "num_hidden_layers": 6,
                        "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                        "head_dim": 8, "query_pre_attn_scalar": 8, "sliding_window": 64
                      },
                      "vision_config": {
                        "model_type": "siglip_vision_model", "num_hidden_layers": 1,
                        "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
                        "patch_size": 2, "image_size": 8
                      }
                    }
                    """.utf8))
            let model = withRandomState(MLXRandom.RandomState(seed: 502)) {
                let model = Gemma3(configuration)
                eval(model)
                return model
            }
            let inputs = [
                LMInput(
                    text: .init(
                        tokens: MLXArray([1, 262144, 262144, 262144, 262144, 2])
                            .reshaped([1, 6])),
                    image: .init(pixels: MLXArray.ones([1, 3, 8, 8]), frames: [])),
                LMInput(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])),
            ]
            try expectRowsMatchSingles(model, inputs: inputs, positions: [6, 3])
        }
    }

    @Test("Gemma 4 positions each row, also on its KV-shared layers")
    func gemma4() throws {
        try expectTextRows(seed: 503) {
            Gemma4(
                try JSONDecoder().decode(
                    Gemma4Configuration.self,
                    from: Data(
                        """
                        {
                          "text_config": {
                            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                            "num_attention_heads": 2, "num_key_value_heads": 1,
                            "head_dim": 16, "global_head_dim": 32,
                            "vocab_size": 64, "vocab_size_per_layer_input": 64,
                            "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 8,
                            "sliding_window": 64, "sliding_window_pattern": 2,
                            "max_position_embeddings": 512
                          },
                          "vision_config": {
                            "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
                            "num_attention_heads": 2, "head_dim": 8, "position_embedding_size": 64
                          }
                        }
                        """.utf8)))
        }
    }

    @Test("Idefics3 positions each row of a batch at its own offset")
    func idefics3() throws {
        try expectTextRows(seed: 504) {
            Idefics3(
                try JSONDecoder().decode(
                    Idefics3Configuration.self,
                    from: Data(
                        """
                        {
                          "model_type": "idefics3", "vocab_size": 64, "image_token_id": 63,
                          "scale_factor": 2,
                          "text_config": {
                            "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2,
                            "intermediate_size": 64, "num_attention_heads": 4,
                            "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 64,
                            "rope_theta": 10000.0
                          },
                          "vision_config": {
                            "model_type": "idefics3_vision", "hidden_size": 16,
                            "num_hidden_layers": 1, "intermediate_size": 32,
                            "num_attention_heads": 2, "patch_size": 4, "image_size": 16
                          }
                        }
                        """.utf8)))
        }
    }

    @Test("LFM2-VL positions each row of a batch at its own offset")
    func lfm2VL() throws {
        try expectTextRows(seed: 505) {
            LFM2VL(
                try JSONDecoder().decode(
                    LFM2VLConfiguration.self,
                    from: Data(
                        """
                        {
                          "model_type": "lfm2_vl", "image_token_id": 63,
                          "downsample_factor": 2, "projector_hidden_size": 16,
                          "projector_use_layernorm": false,
                          "text_config": {
                            "model_type": "lfm2", "vocab_size": 64, "hidden_size": 32,
                            "num_hidden_layers": 2, "num_attention_heads": 4,
                            "num_key_value_heads": 2, "intermediate_size": 64, "block_dim": 32,
                            "block_multiple_of": 8, "block_auto_adjust_ff_dim": false,
                            "layer_types": ["full_attention", "full_attention"]
                          },
                          "vision_config": {
                            "model_type": "siglip2_vision_model", "hidden_size": 16,
                            "intermediate_size": 32, "num_hidden_layers": 1,
                            "num_attention_heads": 2, "num_channels": 3,
                            "num_patches": 16, "patch_size": 2, "layer_norm_eps": 1e-6
                          }
                        }
                        """.utf8)))
        }
    }

    @Test("Mistral 3 positions and Llama 4-scales each row of a batch at its own offset")
    func mistral3() throws {
        try expectTextRows(seed: 506) {
            Mistral3VLM(
                try JSONDecoder().decode(
                    Mistral3VLMConfiguration.self,
                    from: Data(
                        """
                        {
                          "model_type": "mistral3", "image_token_index": 63,
                          "spatial_merge_size": 1, "vision_feature_layer": -1,
                          "text_config": {
                            "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                            "intermediate_size": 64, "num_attention_heads": 2,
                            "num_key_value_heads": 1, "head_dim": 16, "rms_norm_eps": 1e-5,
                            "vocab_size": 64, "max_position_embeddings": 512,
                            "rope_parameters": {
                              "rope_theta": 10000.0, "llama_4_scaling_beta": 0.5,
                              "original_max_position_embeddings": 4
                            }
                          },
                          "vision_config": {
                            "model_type": "pixtral", "hidden_size": 32, "num_hidden_layers": 1,
                            "num_attention_heads": 2, "intermediate_size": 64,
                            "patch_size": 8, "image_size": 32
                          }
                        }
                        """.utf8)))
        }
    }

    @Test("MuseGlimmer positions each row of a batch at its own offset")
    func museGlimmer() throws {
        // The tiny model's window is eight: these rows stay inside it while they decode.
        try expectTextRows(seed: 507, prompts: [[1, 2], [1, 2, 3, 4, 5]]) {
            try MuseGlimmerForwardTests.model()
        }
    }

    @Test("Pixtral positions each row of a batch at its own offset")
    func pixtral() throws {
        try expectTextRows(seed: 508) {
            PixtralVLM(
                try JSONDecoder().decode(
                    PixtralConfiguration.self,
                    from: Data(
                        """
                        {
                          "model_type": "pixtral", "vocab_size": 64, "image_token_index": 63,
                          "text_config": {
                            "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                            "intermediate_size": 64, "num_attention_heads": 4,
                            "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-5,
                            "vocab_size": 64, "rope_theta": 10000.0, "use_qk_norm": true
                          },
                          "vision_config": {
                            "model_type": "pixtral", "hidden_size": 16, "num_hidden_layers": 1,
                            "num_attention_heads": 2, "intermediate_size": 32,
                            "patch_size": 4, "image_size": 8
                          }
                        }
                        """.utf8)))
        }
    }

    @Test(
        "Qwen 2 and 2.5 VL position each text row of a batch at its own offset",
        arguments: QwenVLTextFamily.allCases)
    func qwenVL(family: QwenVLTextFamily) throws {
        try Device.withDefaultDevice(.cpu) {
            try expectTextRowsMatchSingles(try tinyQwenVLText(family))
        }
    }
}

/// Text models whose Llama 4 query scaling follows each row's position.
@Suite(.serialized)
struct LLMRowPositionTests {

    /// Positions 3 and 9 fall in different Llama 4 scaling buckets of four positions.
    private static let llama4Scaling = """
        "rope_parameters": {
          "rope_theta": 10000.0, "llama_4_scaling_beta": 0.5,
          "original_max_position_embeddings": 4
        }
        """

    @Test("Mistral 3 text Llama 4-scales each row of a batch at its own position")
    func mistral3Text() throws {
        try expectTextRows(seed: 509) {
            Mistral3TextModel(
                try JSONDecoder().decode(
                    Mistral3TextConfiguration.self,
                    from: Data(
                        """
                        {
                          "model_type": "ministral3", "hidden_size": 16, "num_hidden_layers": 2,
                          "intermediate_size": 32, "num_attention_heads": 2, "rms_norm_eps": 1e-6,
                          "vocab_size": 32, "num_key_value_heads": 1, "head_dim": 8,
                          "tie_word_embeddings": true,
                          "layer_types": ["full_attention", "full_attention"],
                          \(Self.llama4Scaling)
                        }
                        """.utf8)))
        }
    }

    @Test("Nemotron Labs Diffusion Llama 4-scales each row of a batch at its own position")
    func nemotronLabsDiffusion() throws {
        try expectTextRows(seed: 510) {
            NemotronLabsDiffusionModel(
                try JSONDecoder().decode(
                    NemotronLabsDiffusionConfiguration.self,
                    from: Data(
                        """
                        {
                          "hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
                          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                          "rms_norm_eps": 1e-6, "vocab_size": 32,
                          \(Self.llama4Scaling)
                        }
                        """.utf8)))
        }
    }
}

/// Builds a seeded tiny model on the CPU and checks its text rows decoded as one batch.
private func expectTextRows(
    seed: UInt64, prompts: [[Int32]] = [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    _ makeModel: () throws -> any LanguageModel
) throws {
    try Device.withDefaultDevice(.cpu) {
        let model = try withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = try makeModel()
            eval(model)
            return model
        }
        try expectTextRowsMatchSingles(model, prompts: prompts)
    }
}

enum QwenVLTextFamily: String, CaseIterable, Sendable {
    case qwen2 = "qwen2_vl"
    case qwen25 = "qwen2_5_vl"
}

func tinyQwenVLText(_ family: QwenVLTextFamily) throws -> any LanguageModel {
    var configuration: [String: Any] = [
        "model_type": family.rawValue, "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6, "vocab_size": 64, "max_position_embeddings": 4096,
        "rope_theta": 1000000.0, "rope_traditional": false, "tie_word_embeddings": false,
        "image_token_id": 60, "video_token_id": 61, "vision_start_token_id": 62,
        "vision_end_token_id": 63, "vision_token_id": 59,
        "rope_scaling": ["type": "mrope", "mrope_section": [2, 3, 3]],
    ]
    if family == .qwen2 {
        configuration["vision_config"] = [
            "depth": 1, "embed_dim": 32, "hidden_size": 64, "num_heads": 2,
            "patch_size": 16, "mlp_ratio": 2.0, "spatial_patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3,
        ]
    } else {
        configuration["sliding_window"] = 32768
        configuration["use_sliding_window"] = false
        configuration["max_window_layers"] = 2
        configuration["vision_config"] = [
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 64,
            "num_heads": 2, "patch_size": 16, "spatial_patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "window_size": 64,
            "fullatt_block_indexes": [0], "tokens_per_second": 2, "in_chans": 3,
        ]
    }
    let data = try JSONSerialization.data(withJSONObject: configuration)
    let model: any LanguageModel
    switch family {
    case .qwen2:
        let config = try JSONDecoder().decode(Qwen2VLConfiguration.self, from: data)
        model = withRandomState(MLXRandom.RandomState(seed: 917)) { Qwen2VL(config) }
    case .qwen25:
        let config = try JSONDecoder().decode(Qwen25VLConfiguration.self, from: data)
        model = withRandomState(MLXRandom.RandomState(seed: 918)) { Qwen25VL(config) }
    }
    model.train(false)
    eval(model)
    return model
}

/// Prefills each input alone, joins the caches as one batch and decodes three tokens,
/// comparing every row's logits with the same sequence continued alone.
private func expectRowsMatchSingles(
    _ model: any LanguageModel, inputs: [LMInput], positions: [Int]
) throws {
    func prefill(_ input: LMInput) throws -> [any KVCache] {
        let cache = try model.newCache(parameters: nil)
        switch try model.prepare(input, cache: cache, state: nil, prefill: .init()) {
        case .logits(let output): eval(output.logits)
        case .tokens: Issue.record("The model must complete its own prefill")
        }
        eval(cache)
        return cache
    }

    try expectBatchDecodeMatchesSingles(
        model, singles: try inputs.map(prefill), positions: positions)
}

/// Prefills each prompt alone through the model's forward pass, then decodes the rows as one
/// batch as `expectRowsMatchSingles` does.
private func expectTextRowsMatchSingles(
    _ model: any LanguageModel,
    prompts: [[Int32]] = [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
) throws {
    let singles = try prompts.map { prompt in
        let cache = try model.newCache(parameters: nil)
        eval(forward(model, MLXArray(prompt).reshaped([1, prompt.count]), cache: cache))
        eval(cache)
        return cache
    }
    try expectBatchDecodeMatchesSingles(model, singles: singles, positions: prompts.map(\.count))
}

/// Joins prefilled single-row caches as one batch and decodes three tokens, comparing every
/// row's logits with the same sequence continued alone.
private func expectBatchDecodeMatchesSingles(
    _ model: any LanguageModel, singles: [[any KVCache]], positions: [Int]
) throws {
    #expect(singles.map { $0[0].offset } == positions)
    let batch = singles[0].indices.map { layer in
        RowOffsetCache(rows: singles.map { $0[layer] }) as KVCache
    }
    #expect((batch[0] as? RowOffsetCache)?.batchOffset.asArray(Int.self) == positions)

    for step in 0 ..< 3 {
        let token = Int32(4 + step)
        let output = forward(model, MLXArray([token, token]).reshaped([2, 1]), cache: batch)
        for row in 0 ..< 2 {
            let expected = forward(model, MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], expected) < 1e-4,
                "row \(row), step \(step)")
        }
    }
}

/// The logits of a stateless text step, through the entry point every model implements.
private func forward(_ model: any LanguageModel, _ tokens: MLXArray, cache: [any KVCache])
    -> MLXArray
{
    model.nextTokenLogits(.init(tokens: tokens), cache: cache, state: nil).logits
}

func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
}

@Suite(.serialized)
struct QwenVLBatchedPositionTests {
    @Test(
        "Stateless Qwen VL text rows use their own positions", arguments: QwenVLTextFamily.allCases)
    func raggedTextLogits(family: QwenVLTextFamily) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyQwenVLText(family)
            let prompts = [[Int32(1), 2, 3], [4, 5, 6, 7, 8, 9, 10]]
            var singles = try prompts.map { prompt in
                let cache = try model.newCache(parameters: nil)
                let output = model.nextTokenLogits(
                    .init(tokens: MLXArray(prompt)[.newAxis]), cache: cache, state: nil)
                #expect(output.state?.isEmpty != false)
                eval(output.logits, cache)
                return cache
            }
            let batch = try singles[0].indices.map { layer in
                try mergeCaches(singles.map { $0[layer].copy() })
            }
            let first = try #require(batch.first as? BatchKVCache)
            #expect(first.batchOffset.asArray(Int32.self) == [3, 7])

            for length in [1, 2, 1] {
                let tokens = MLXArray((0 ..< 2 * length).map { Int32(11 + $0) })
                    .reshaped([2, length])
                let actual = model.nextTokenLogits(.init(tokens: tokens), cache: batch, state: nil)
                #expect(actual.state?.isEmpty != false)
                for row in singles.indices {
                    let expected = model.nextTokenLogits(
                        .init(tokens: tokens[row ..< row + 1]), cache: singles[row], state: nil)
                    #expect(
                        maxAbsDifference(actual.logits[row ..< row + 1], expected.logits) < 1e-4,
                        "\(family), row \(row), chunk length \(length)")
                }
                let expectedOffsets = singles.map { $0[0].offset }
                #expect(first.batchOffset.asArray(Int.self) == expectedOffsets)
            }
            batch.forEach { $0.filter(batchIndices: batchRows(0)) }
            singles = [singles[0]]
            if case .scalar(let offset) = first.ropeOffset {
                #expect(offset == singles[0][0].offset)
            } else {
                Issue.record("A remaining single row must use the scalar RoPE path")
            }
            let tokens = MLXArray([Int32(21), 22])[.newAxis]
            let actual = model.nextTokenLogits(.init(tokens: tokens), cache: batch, state: nil)
            let expected = model.nextTokenLogits(
                .init(tokens: tokens), cache: singles[0], state: nil)
            #expect(maxAbsDifference(actual.logits, expected.logits) < 1e-4)
        }
    }

}

/// The Qwen VL families whose image rows batch, as tiny random-weight models.
enum QwenVLImageFamily: String, CaseIterable, Sendable {
    case qwen2, qwen25, qwen3, qwen3MoE, qwen35, qwen35MoE

    func model() throws -> any LanguageModel {
        switch self {
        case .qwen2: return try tinyQwenVLText(.qwen2)
        case .qwen25: return try tinyQwenVLText(.qwen25)
        case .qwen35, .qwen35MoE:
            // Hybrid: linear-attention layers with a full-attention layer every second one.
            let moe = self == .qwen35MoE
            var text: [String: Any] = [
                "model_type": moe ? "qwen3_5_moe" : "qwen3_5", "hidden_size": 64,
                "num_hidden_layers": 4, "intermediate_size": 128, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 512,
                "full_attention_interval": 2, "linear_num_value_heads": 4,
                "linear_num_key_heads": 2, "linear_key_head_dim": 32,
                "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
                "max_position_embeddings": 4096,
                "rope_parameters": [
                    "type": "default", "mrope_section": [8, 4, 4], "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ]
            if moe {
                // Both experts for every token: a top-1 choice between near-tied experts can
                // flip under the GPU's batch rounding and change a row's output wholesale.
                text["num_experts"] = 2
                text["num_experts_per_tok"] = 2
                text["moe_intermediate_size"] = 128
                text["shared_expert_intermediate_size"] = 128
            }
            let configuration: [String: Any] = [
                "model_type": moe ? "qwen3_5_moe" : "qwen3_5_vl", "image_token_id": 500,
                "video_token_id": 501, "vision_start_token_id": 502,
                "vision_end_token_id": 503, "vocab_size": 512, "text_config": text,
                "vision_config": [
                    "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                    "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "num_position_embeddings": 64,
                ],
            ]
            let config = try JSONDecoder().decode(
                MLXVLM.Qwen35Configuration.self,
                from: try JSONSerialization.data(withJSONObject: configuration))
            let model: any LanguageModel = withRandomState(
                MLXRandom.RandomState(seed: moe ? 922 : 921)
            ) { moe ? MLXVLM.Qwen35MoE(config) : MLXVLM.Qwen35(config) }
            model.train(false)
            eval(model)
            return model
        case .qwen3, .qwen3MoE:
            let moe = self == .qwen3MoE
            let modelType = moe ? "qwen3_vl_moe" : "qwen3_vl"
            var text: [String: Any] = [
                "model_type": modelType, "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 16, "vocab_size": 512, "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-6, "rope_theta": 100000.0,
                "rope_scaling": ["type": "default", "mrope_section": [4, 2, 2]],
            ]
            if moe {
                text["num_experts"] = 2
                text["num_experts_per_tok"] = 1
                text["decoder_sparse_step"] = 1
                text["mlp_only_layers"] = [Int]()
                text["moe_intermediate_size"] = 128
                text["tie_word_embeddings"] = false
            }
            let configuration: [String: Any] = [
                "model_type": modelType, "image_token_id": 500, "video_token_id": 501,
                "vision_start_token_id": 502, "vision_end_token_id": 503, "vocab_size": 512,
                "text_config": text,
                "vision_config": [
                    "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                    "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "num_position_embeddings": 64, "deepstack_visual_indexes": [0, 1],
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: configuration)
            let model: any LanguageModel =
                moe
                ? try withRandomState(MLXRandom.RandomState(seed: 920)) {
                    Qwen3VLMoE(try JSONDecoder().decode(Qwen3VLMoEConfiguration.self, from: data))
                }
                : try withRandomState(MLXRandom.RandomState(seed: 919)) {
                    Qwen3VL(try JSONDecoder().decode(Qwen3VLConfiguration.self, from: data))
                }
            model.train(false)
            eval(model)
            return model
        }
    }

    /// Qwen 3.5's linear-attention layers use a Metal kernel, so its families run on the GPU,
    /// where a batch's matmul shapes round differently from a single row's: every family here
    /// differs from its single runs by 2e-3 to 7e-3 on the GPU (logits up to about 5), while
    /// those that run on the CPU agree within 1e-4.
    var device: Device { [.qwen35, .qwen35MoE].contains(self) ? .gpu : .cpu }
    var logitTolerance: Float { device == .gpu ? 1e-2 : 1e-4 }

    /// The image, vision-start and vision-end token ids of the family's configuration.
    var imageTokens: (image: Int32, start: Int32, end: Int32) {
        switch self {
        case .qwen2, .qwen25: (60, 62, 63)
        case .qwen3, .qwen3MoE, .qwen35, .qwen35MoE: (500, 502, 503)
        }
    }
}

@Suite(.serialized)
struct QwenVLBatchedImageTests {
    /// Image and text inputs of unequal lengths: image grids of 4 and 2 merged tokens. Tokens
    /// are `[1, length]`, as the Qwen VL processors prepare them.
    static func inputs(_ family: QwenVLImageFamily) -> [LMInput] {
        let (pad, start, end) = family.imageTokens
        func tokens(_ values: [Int32]) -> MLXArray { MLXArray(values).reshaped([1, -1]) }
        func image(_ before: [Int32], pads: Int, after: [Int32], grid: THW, seed: UInt64)
            -> LMInput
        {
            LMInput(
                text: .init(
                    tokens: tokens(
                        before + [start] + [Int32](repeating: pad, count: pads) + [end] + after)),
                image: .init(
                    pixels: MLXRandom.normal(
                        [grid.t * grid.h * grid.w, 3 * 2 * 16 * 16], key: MLXRandom.key(seed)),
                    frames: [grid]))
        }
        return [
            image([1], pads: 4, after: [2, 3], grid: THW(1, 4, 4), seed: 1),
            LMInput(tokens: tokens([4, 5, 6])),
            image([7, 8], pads: 2, after: [9, 10, 11, 12, 13], grid: THW(1, 2, 4), seed: 2),
            LMInput(tokens: tokens((14 ..< 21).map { Int32($0) })),
        ]
    }

    @Test(
        "Qwen VL image and text rows decode as a batch like each row alone",
        arguments: QwenVLImageFamily.allCases)
    func rowsMatchSingles(family: QwenVLImageFamily) throws {
        try Device.withDefaultDevice(family.device) {
            let model = try family.model()
            let stateModel = try #require(model as? any BatchedGenerationStateProviding)
            let inputs = Self.inputs(family)
            func prefill(_ input: LMInput) throws -> ([any KVCache], LMOutput.State) {
                let cache = try model.newCache(parameters: nil)
                guard
                    case .logits(let output) = try model.prepare(
                        input, cache: cache, state: nil, prefill: .init())
                else {
                    Issue.record("The model must complete its own prefill")
                    return (cache, LMOutput.State())
                }
                eval(output.logits, cache)
                return (cache, output.state ?? LMOutput.State())
            }
            // Each row continued alone, and separate prefills for the batch, whose caches it
            // changes.
            var singles = try inputs.map(prefill)
            let joining = try inputs.map(prefill)
            #expect(singles[0].1.isEmpty == false, "an image prefill leaves rope deltas")

            var rows = [0, 1]
            let batch = try singles[0].0.indices.map { layer in
                try mergeCaches(rows.map { joining[$0].0[layer] })
            }
            var state = stateModel.mergeGenerationStates(
                rows.map { joining[$0].1 }, rowCounts: [1, 1])
            func check(_ step: Int) {
                let token = Int32(30 + step)
                let output = model.nextTokenLogits(
                    .init(
                        tokens: MLXArray([Int32](repeating: token, count: rows.count))
                            .reshaped([rows.count, 1])),
                    cache: batch, state: state)
                state = output.state ?? LMOutput.State()
                for (index, row) in rows.enumerated() {
                    let expected = model.nextTokenLogits(
                        .init(tokens: MLXArray([token]).reshaped([1, 1])), cache: singles[row].0,
                        state: singles[row].1)
                    singles[row].1 = expected.state ?? LMOutput.State()
                    #expect(
                        maxAbsDifference(output.logits[index ..< index + 1], expected.logits)
                            < family.logitTolerance, "\(family), row \(row), step \(step)")
                }
            }
            check(0)
            check(1)
            // Rows join while the others decode.
            for row in [2, 3] {
                for (layer, cache) in batch.enumerated() {
                    cache.extend(other: try mergeCaches([joining[row].0[layer]]))
                }
                state = stateModel.mergeGenerationStates(
                    [state, joining[row].1], rowCounts: [rows.count, 1])
                rows.append(row)
            }
            check(2)
            check(3)
            // A row leaves; the others keep their own positions.
            let keep = MLXArray([Int32(3), 0, 2])
            batch.forEach { $0.filter(batchIndices: keep) }
            state = stateModel.selectGenerationState(state, batchIndices: keep)
            rows = [rows[3], rows[0], rows[2]]
            check(4)
            check(5)
        }
    }

    @Test(
        "Qwen VL image rows join a batch and decode the tokens they decode alone",
        arguments: QwenVLImageFamily.allCases)
    func iteratorRowsMatchSingles(family: QwenVLImageFamily) throws {
        try Device.withDefaultDevice(family.device) {
            let model = try family.model()
            let inputs = Self.inputs(family)
            let expected = try inputs.map { input in
                Array(
                    IteratorSequence(
                        try TokenIterator(
                            input: input, model: model, processor: nil, sampler: ArgMaxSampler(),
                            prefill: .init(), maxTokens: 6)))
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 6)
            var ids = iterator.insert(inputs: Array(inputs[0 ..< 2]))
            var output = [Int: [Int]]()
            var steps = 0
            while let events = try iterator.next() {
                for event in events {
                    switch event {
                    case .token(let id, let token): output[id, default: []].append(token)
                    case .failed(let id, let error): Issue.record("row \(id): \(error)")
                    case .prefill, .finished: break
                    }
                }
                steps += 1
                // Two more rows join while the first ones decode.
                if steps == 2 { ids += iterator.insert(inputs: Array(inputs[2...])) }
            }
            #expect(ids.map { output[$0] ?? [] } == expected, "\(family)")
        }
    }
}

@Suite(.serialized)
struct BatchedVLMPositionTests {

    @Test(
        "Stateless VLM text rows keep their positions through ragged decode and admission",
        arguments: BatchedVLMPositionFamily.allCases)
    func raggedTextRows(family: BatchedVLMPositionFamily) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try withRandomState(MLXRandom.RandomState(seed: 741)) {
                try family.makeModel()
            }
            eval(model)
            try expectVLMPositionParity(model)
        }
    }
}

enum BatchedVLMPositionFamily: Sendable, CaseIterable {
    case pixtral
    case lfm2VL
    case museGlimmer

    func makeModel() throws -> any LanguageModel {
        switch self {
        case .pixtral:
            let configuration = try JSONDecoder().decode(
                PixtralConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "pixtral", "vocab_size": 64, "image_token_index": 63,
                      "text_config": {
                        "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                        "intermediate_size": 64, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-5,
                        "vocab_size": 64, "rope_theta": 10000.0, "use_qk_norm": true
                      },
                      "vision_config": {
                        "model_type": "pixtral", "hidden_size": 16, "num_hidden_layers": 1,
                        "num_attention_heads": 2, "intermediate_size": 32,
                        "patch_size": 4, "image_size": 8
                      }
                    }
                    """.utf8))
            return PixtralVLM(configuration)
        case .lfm2VL:
            let configuration = try JSONDecoder().decode(
                LFM2VLConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "lfm2_vl", "image_token_id": 63,
                      "downsample_factor": 2, "projector_hidden_size": 16,
                      "projector_use_layernorm": false,
                      "text_config": {
                        "model_type": "lfm2", "vocab_size": 64, "hidden_size": 32,
                        "num_hidden_layers": 3, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "intermediate_size": 64, "block_dim": 32,
                        "block_multiple_of": 8, "block_auto_adjust_ff_dim": false,
                        "layer_types": ["full_attention", "conv", "full_attention"]
                      },
                      "vision_config": {
                        "model_type": "siglip2_vision_model", "hidden_size": 16,
                        "intermediate_size": 32, "num_hidden_layers": 1,
                        "num_attention_heads": 2, "num_channels": 3,
                        "num_patches": 16, "patch_size": 2, "layer_norm_eps": 1e-6
                      }
                    }
                    """.utf8))
            return LFM2VL(configuration)
        case .museGlimmer:
            return try MuseGlimmerForwardTests.model()
        }
    }
}

/// Use the model's single-row prefill, then compare each batched forward with its own cache.
private func expectVLMPositionParity(_ model: any LanguageModel) throws {
    func prefill(_ prompt: [Int32]) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        let output: LMOutput
        switch try model.prepare(
            LMInput(tokens: MLXArray(prompt)), cache: cache, state: nil,
            prefill: .init(stepSize: 2))
        {
        case .logits(let prepared):
            output = prepared
        case .tokens(let remaining):
            let text = remaining.tokens.ndim == 1 ? remaining[text: .newAxis] : remaining
            output = model.nextTokenLogits(text, cache: cache, state: nil)
        }
        try #require(output.state?.isEmpty != false, "these text rows must be stateless")
        eval(cache)
        return cache
    }

    var singles = try [[Int32(1), 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]].map(prefill)
    let batch = try singles[0].indices.map { layer in
        try mergeCaches(singles.map { $0[layer].copy() })
    }
    func decode(_ tokens: [[Int32]]) {
        let input = MLXArray(tokens.flatMap { $0 }).reshaped([tokens.count, tokens[0].count])
        let actual = model.nextTokenLogits(.init(tokens: input), cache: batch, state: nil)
        for row in tokens.indices {
            let expected = model.nextTokenLogits(
                .init(tokens: MLXArray(tokens[row])[.newAxis]), cache: singles[row], state: nil)
            #expect(
                maxAbsDifference(actual.logits[row ..< row + 1], expected.logits) < 1e-4,
                "row \(row), chunk length \(tokens[row].count)")
        }
    }
    decode([[12], [13]])
    decode([[14, 15], [16, 17]])

    let added = try [[Int32(18)], [19, 20, 21, 22, 23]].map(prefill)
    for layer in batch.indices {
        let incoming = try mergeCaches(added.map { $0[layer].copy() })
        try batch[layer].validateExtension(other: incoming)
        batch[layer].extend(other: incoming)
    }
    singles += added
    decode([[24, 25], [26, 27], [28, 29], [30, 31]])

    batch.forEach { $0.filter(batchIndices: batchRows(3, 0)) }
    singles = [singles[3], singles[0]]
    decode([[32, 33, 34], [35, 36, 37]])
    batch.forEach { $0.filter(batchIndices: batchRows(1)) }
    singles = [singles[1]]
    decode([[38]])
}
