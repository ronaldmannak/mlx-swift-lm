// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Inkling2PrefillTests {
    private func model() throws -> Inkling2Model {
        let config = try JSONDecoder().decode(
            Inkling2Configuration.self,
            from: Data(
                """
                {"model_type":"inkling2","text_config":{
                    "hidden_size":64,"num_hidden_layers":2,"vocab_size":128,
                    "unpadded_vocab_size":120,"num_attention_heads":4,
                    "num_key_value_heads":2,"head_dim":16,"swa_num_attention_heads":4,
                    "swa_num_key_value_heads":2,"swa_head_dim":16,"sliding_window_size":32,
                    "local_layer_ids":[],"d_rel":4,"rel_extent":16,
                    "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"rms_norm_eps":1e-6,
                    "qk_l2_clip_threshold":3,"rel_l2_clip_threshold":2,"sconv_kernel_size":4,
                    "n_layers_per_attn_res_block":3,"n_routed_experts":4,
                    "num_experts_per_tok":2,"intermediate_size":64,"mlp_gate_softcap":4,
                    "mlp_up_softcap":25,"route_scale":2,"final_logit_softcapping":30}}
                """.utf8))
        MLXRandom.seed(47)
        let model = Inkling2Model(config)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map {
                    name, value in
                    let nearOne = name.hasSuffix("norm.weight") || name.hasSuffix("l2_clip.scale")
                    let noise = MLXRandom.normal(value.shape) * 0.05
                    return (name, nearOne ? noise + 1 : noise)
                }))
        model.train(false)
        eval(model)
        return model
    }

    @Test(
        "Prefill reads out only the last row and keeps all requested draft taps",
        arguments: [PrefillParameters.Chunking.balanced, .remainder, .unchunked])
    func prefillPreservesRows(chunking: PrefillParameters.Chunking) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let input = LMInput(tokens: MLXArray((0 ..< 13).map { Int32($0 + 1) }))
            let prefill = PrefillParameters(stepSize: 5, chunking: chunking)
            var state = LMOutput.State()
            state[mtpEmitFlagKey] = true
            state[mtpTargetLayerIdsKey] = [0, 2]
            let cache = try model.newCache(parameters: nil)
            guard
                case .logits(let output) = try model.prepare(
                    input, cache: cache, state: state, prefill: prefill)
            else {
                Issue.record("Expected prepared logits")
                return
            }
            let taps = try #require(output.state?[mtpLastHiddenStatesKey])
            #expect(output.logits.shape == [1, 1, 120])
            #expect(taps.shape == [1, 13, 128])
            #expect(cache.first?.offset == 13)
            let plain = try model.prepare(
                input, cache: model.newCache(parameters: nil), state: nil, prefill: prefill)
            guard case .logits(let plainOutput) = plain else { return }
            #expect(abs(plainOutput.logits - output.logits).max().item(Float.self) < 1e-4)
        }
    }

    @Test("A short prompt's last-row readout matches its full forward")
    func lastRowReadout() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let tokens = MLXArray([Int32(1), 2, 3, 4, 5])
            let expected = model(tokens[.newAxis], cache: try model.newCache(parameters: nil))
            guard
                case .logits(let output) = try model.prepare(
                    LMInput(tokens: tokens), cache: model.newCache(parameters: nil), state: nil,
                    prefill: .init(stepSize: 512))
            else {
                Issue.record("Expected prepared logits")
                return
            }
            #expect(output.logits.shape == [1, 1, 120])
            #expect(
                abs(output.logits - expected[0..., 4 ..< 5, 0...]).max().item(Float.self) < 1e-4)
        }
    }
}
