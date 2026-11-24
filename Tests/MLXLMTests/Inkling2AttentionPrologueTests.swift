// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private final class PrologueAdapterLinear: Linear {
    let adjustment: Float

    init(weight: MLXArray, adjustment: Float) {
        self.adjustment = adjustment
        super.init(weight: weight, bias: nil)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        (super.callAsFunction(x).asType(.float32) + adjustment).asType(x.dtype)
    }
}

private typealias PrologueOutput = (
    queries: MLXArray, relative: MLXArray, keys: MLXArray, values: MLXArray,
    gate: MLXArray, sinks: MLXArray, keyState: MLXArray, valueState: MLXArray
)

@Suite(.serialized)
struct Inkling2AttentionPrologueTests {
    private func attention(sliding: Bool, headDim: Int = 128) throws -> Inkling2Attention {
        let config = try JSONDecoder().decode(
            Inkling2TextConfiguration.self,
            from: Data(
                """
                {"hidden_size":32,"num_hidden_layers":1,"vocab_size":64,
                 "num_attention_heads":6,"num_key_value_heads":2,"head_dim":\(headDim),
                 "swa_num_attention_heads":6,"swa_num_key_value_heads":2,"swa_head_dim":\(headDim),
                 "sliding_window_size":1024,"local_layer_ids":[0],"d_rel":16,"rel_extent":1024,
                 "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"rms_norm_eps":1e-5,
                 "qk_l2_clip_threshold":3.0,"rel_l2_clip_threshold":2.0,"sconv_kernel_size":4,
                 "n_layers_per_attn_res_block":6,"n_routed_experts":2,"num_experts_per_tok":1,
                 "intermediate_size":64,"mlp_gate_softcap":4,"mlp_up_softcap":25,
                 "route_scale":1,"final_logit_softcapping":30}
                """.utf8))
        let attention = Inkling2Attention(config, sliding: sliding)
        let parameters = attention.parameters().flattened().map { name, value in
            let random = MLXRandom.normal(value.shape)
            let data = name.hasSuffix(".scale") ? 1 + random * 0.1 : random * 0.2
            return (name, name.hasSuffix(".scale") ? data : data.asType(.bfloat16))
        }
        attention.update(parameters: ModuleParameters.unflattened(parameters))
        attention.train(false)
        return attention
    }

    private func reference(
        _ attention: Inkling2Attention, query: MLXArray, key: MLXArray, value: MLXArray,
        states: (MLXArray, MLXArray)
    ) -> PrologueOutput {
        func convolve(
            _ module: Module, x: MLXArray, input: MLXArray, state: MLXArray
        ) -> (MLXArray, MLXArray) {
            if let convolution = module as? Inkling2ShortConvolution {
                return convolution(x, state: state, mask: nil)
            }
            return (module as! Inkling2DynamicConvolution)(
                x, dynamicInput: input, state: state, mask: nil, lengths: nil)
        }
        let (keys, keyState) = convolve(
            attention.keyConvolution, x: attention.kProjection(key), input: key, state: states.0)
        let (values, valueState) = convolve(
            attention.valueConvolution, x: attention.vProjection(value), input: value,
            state: states.1)
        let batch = query.dim(0)
        let queries = attention.queryClip(
            attention.qProjection(query).reshaped(batch, 1, attention.heads, attention.headDim))
        let relative = attention.relativeClip(
            attention.rProjection(query).reshaped(batch, 1, attention.heads, -1))
        let clipped = inkling2L2Clip(
            keys.reshaped(batch, 1, attention.kvHeads, attention.headDim), scale: nil,
            threshold: attention.qkClipThreshold)
        let normalized = inkling2UnitRMS(
            values.reshaped(batch, 1, attention.kvHeads, attention.headDim),
            eps: attention.eps, dtype: values.dtype)
        return (
            queries, relative, clipped.transposed(0, 2, 1, 3),
            normalized.transposed(0, 2, 1, 3), attention.gateProjection(query),
            attention.sinkProjection(query), keyState, valueState
        )
    }

    private func expectClose(_ fused: PrologueOutput, _ reference: PrologueOutput) {
        for (a, b) in zip(
            [fused.queries, fused.relative, fused.keys, fused.values, fused.gate, fused.sinks],
            [
                reference.queries, reference.relative, reference.keys, reference.values,
                reference.gate, reference.sinks,
            ])
        {
            #expect(a.shape == b.shape)
            #expect(a.dtype == b.dtype)
            #expect(all(isFinite(a)).item(Bool.self))
            #expect(abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) < 2e-2)
        }
        #expect(fused.keyState.dtype == .float32 && fused.valueState.dtype == .float32)
        #expect(arrayEqual(fused.keyState, reference.keyState).item(Bool.self))
        #expect(arrayEqual(fused.valueState, reference.valueState).item(Bool.self))
    }

    @Test(
        "Fused prologue retains clipping, convolution rounding and float32 state",
        arguments: [(true, 1), (false, 1), (true, 2), (false, 2)])
    func parity(sliding: Bool, batch: Int) throws {
        MLXRandom.seed(101)
        let attention = try attention(sliding: sliding)
        // Key/value convolution weights may use different floating-point types.
        if let convolution = attention.valueConvolution as? Inkling2ShortConvolution {
            convolution.conv1d.update(
                parameters: ModuleParameters.unflattened([
                    "weight": convolution.conv1d.weight.asType(.float32)
                ]))
        } else if let convolution = attention.valueConvolution as? Inkling2DynamicConvolution {
            convolution.update(
                parameters: ModuleParameters.unflattened([
                    "static_weight": convolution.staticWeight.asType(.float32)
                ]))
        }
        let query = MLXRandom.normal([batch, 1, 32]).asType(.bfloat16)
        let key = MLXRandom.normal([batch, 1, 32]).asType(.bfloat16)
        let value = MLXRandom.normal([batch, 1, 32]).asType(.bfloat16)
        // These histories deliberately contain values not representable in bfloat16.
        let states = (
            MLXRandom.normal([batch, 3, 256]), MLXRandom.normal([batch, 3, 256])
        )
        let expected = reference(attention, query: query, key: key, value: value, states: states)
        let fused = try #require(
            attention.fusedDecodePrologue(
                query: query, key: key, value: value, states: states, mask: nil, lengths: nil))
        expectClose(fused, expected)
    }

    @Test("Fused prologue reads current projection modules and weights on every call")
    func liveWeightsAndAdapter() throws {
        MLXRandom.seed(102)
        let attention = try attention(sliding: false)
        let input = MLXRandom.normal([1, 1, 32]).asType(.bfloat16)
        let states = (MLXRandom.normal([1, 3, 256]), MLXRandom.normal([1, 3, 256]))
        let first = try #require(
            attention.fusedDecodePrologue(
                query: input, key: input, value: input, states: states, mask: nil, lengths: nil))
        eval(first.queries)
        attention.qProjection.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXArray.zeros(attention.qProjection.weight.shape, dtype: .bfloat16)
            ]))
        let zero = try #require(
            attention.fusedDecodePrologue(
                query: input, key: input, value: input, states: states, mask: nil, lengths: nil))
        #expect(all(zero.queries .== 0).item(Bool.self))
        #expect(abs(first.queries).max().item(Float.self) > 0)
        let adapter = PrologueAdapterLinear(weight: attention.qProjection.weight, adjustment: 0.375)
        adapter.train(false)
        try attention.update(
            modules: ModuleChildren(values: ["q_proj": .value(adapter)]), verify: [])
        let expected = reference(attention, query: input, key: input, value: input, states: states)
        let adapted = try #require(
            attention.fusedDecodePrologue(
                query: input, key: input, value: input, states: states, mask: nil, lengths: nil))
        expectClose(adapted, expected)
        #expect(abs(adapted.queries).max().item(Float.self) > 0)
    }

    @Test("Training, padding, prefill, rewinds and unsupported states retain the original path")
    func fallbacks() throws {
        MLXRandom.seed(103)
        let attention = try attention(sliding: true)
        let input = MLXArray.zeros([1, 1, 32], dtype: .bfloat16)
        let state = MLXArray.zeros([1, 3, 256], dtype: .float32)
        func fused(
            input overrideInput: MLXArray? = nil,
            states overrideStates: (MLXArray?, MLXArray?)? = nil,
            mask: MLXArray? = nil, lengths: MLXArray? = nil, keep: Bool = false
        ) -> PrologueOutput? {
            let selected = overrideInput ?? input
            return attention.fusedDecodePrologue(
                query: selected, key: selected, value: selected,
                states: overrideStates ?? (state, state),
                mask: mask, lengths: lengths, keepWindows: keep)
        }
        #expect(Device.withDefaultDevice(.cpu) { fused() } == nil)
        #expect(fused(keep: true) == nil)
        #expect(fused(mask: MLXArray.ones([1, 1], dtype: .bool)) == nil)
        #expect(fused(lengths: MLXArray([Int32(1)])) == nil)
        #expect(fused(states: (nil, nil)) == nil)
        #expect(fused(states: (state.asType(.bfloat16), state)) == nil)
        #expect(fused(input: MLXArray.zeros([1, 2, 32], dtype: .bfloat16)) == nil)
        #expect(fused(input: input.asType(.float32)) == nil)
        attention.train()
        #expect(fused() == nil)
        attention.train(false)
        attention.keyConvolution.train()
        #expect(fused() == nil)
    }
}
