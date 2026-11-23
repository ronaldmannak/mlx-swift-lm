// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct Inkling2DepthMixTests {
    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func sameBits(_ a: MLXArray, _ b: MLXArray) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        let bits: DType = a.dtype.size == 2 ? .uint16 : .uint32
        return arrayEqual(a.view(dtype: bits), b.view(dtype: bits)).item(Bool.self)
    }

    @Test(
        "Depth mix retains native RMS and depth-readout bits",
        arguments: [
            (16, 1), (16, 2), (16, 8), (64, 1), (64, 2), (64, 8),
            (128, 1), (128, 2), (128, 8), (2304, 1), (2304, 2), (2304, 8),
            (4096, 1), (4096, 2), (4096, 8),
        ])
    func nativePrecision(dimensions: Int, batch: Int) throws {
        for seed in 0 ..< 3 {
            MLXRandom.seed(UInt64(920 + seed))
            let shape = [batch, 1, dimensions]
            let base = MLXRandom.normal(shape) * MLXRandom.uniform(0.1 ..< 4, [batch, 1, 1])
            let blocks = inkling2UnitRMS(MLXRandom.normal([3, batch, 1, dimensions]))
            let query = MLXRandom.normal([1, dimensions]) * 0.1
            let normed = inkling2UnitRMS(base)
            let readout = inkling2DepthAttention(blocks: blocks, partial: normed, query: query)
            let result = try #require(
                inkling2FusedDepthMix(
                    blocks: blocks, base: base, convolution: nil, query: query,
                    normWeight: nil, eps: 1e-6))
            #expect(sameBits(result.normed, normed))
            #expect(sameBits(result.readout, readout))

            for weightType: DType in [.bfloat16, .float32] {
                let norm = RMSNorm(dimensions: dimensions, eps: 1e-5)
                norm.update(
                    parameters: ModuleParameters.unflattened([
                        "weight": MLXRandom.uniform(0.5 ..< 1.5, [dimensions]).asType(weightType)
                    ]))
                for outputType: DType in [.bfloat16, .float32] {
                    let weighted = try #require(
                        inkling2FusedDepthMix(
                            blocks: blocks, base: base, convolution: nil, query: query,
                            normWeight: norm.weight, eps: norm.eps, outputType: outputType))
                    let actual = try #require(weighted.normalizedReadout)
                    let expected = inkling2Normed(norm, readout, dtype: outputType)
                    #expect(sameBits(actual, expected))
                }
            }
        }
    }

    @Test(
        "Fused depth mix matches residual, unit RMS and depth attention",
        arguments: [
            (64, 0, 1, 1), (128, 3, 3, 2), (2304, 4, 1, 1), (2304, 4, 3, 2), (4096, 16, 3, 1),
        ])
    func depthReadout(dimensions: Int, blocksCount: Int, queries: Int, batch: Int) throws {
        MLXRandom.seed(913)
        let blocks =
            blocksCount == 0
            ? MLXArray.zeros([0, batch, 1, dimensions])
            : inkling2UnitRMS(MLXRandom.normal([blocksCount, batch, 1, dimensions]))
        let base = MLXRandom.normal([batch, 1, dimensions])
        let query = MLXRandom.normal([queries, dimensions]) * 0.1
        let result = try #require(
            inkling2FusedDepthMix(
                blocks: blocks, base: base, convolution: nil, query: query,
                normWeight: nil, eps: 1e-6))
        let normed = inkling2UnitRMS(base)
        let expected = inkling2DepthAttention(blocks: blocks, partial: normed, query: query)
        #expect(result.residual.shape == base.shape)
        #expect(result.readout.shape == [batch, 1, queries, dimensions])
        #expect(result.readout.dtype == .float32)
        #expect(arrayEqual(result.residual, base).item(Bool.self))
        #expect(maxDifference(result.normed, normed) < 2e-5)
        #expect(maxDifference(result.readout, expected) < 5e-5)
        #expect(result.normalizedReadout == nil)
        #expect(result.state == nil)
    }

    @Test(
        "Weighted depth readouts use the caller's output dtype",
        arguments: [(DType.float32, DType.bfloat16), (.bfloat16, .float32), (.float32, .float32)])
    func weightedReadout(weightType: DType, outputType: DType) throws {
        MLXRandom.seed(914)
        let dimensions = 2304
        let blocks = inkling2UnitRMS(MLXRandom.normal([4, 2, 1, dimensions]))
        let base = MLXRandom.normal([2, 1, dimensions])
        let query = MLXRandom.normal([3, dimensions]) * 0.1
        let weight = MLXRandom.uniform(0.5 ..< 1.5, [3, dimensions]).asType(weightType)
        let result = try #require(
            inkling2FusedDepthMix(
                blocks: blocks, base: base, convolution: nil, query: query,
                normWeight: weight, eps: 1e-5, outputType: outputType))
        let readout = inkling2DepthAttention(
            blocks: blocks, partial: inkling2UnitRMS(base), query: query)
        let expected = (MLXFast.rmsNorm(readout, weight: nil, eps: 1e-5) * weight.asType(.float32))
            .asType(outputType)
        let normalized = try #require(result.normalizedReadout)
        #expect(normalized.dtype == outputType)
        #expect(normalized.shape == [2, 1, 3, dimensions])
        #expect(maxDifference(normalized, expected) < (outputType == .float32 ? 2e-4 : 2e-2))
    }

    @Test(
        "Post-attention convolution keeps its weight precision and float32 state",
        arguments: [
            (DType.bfloat16, DType.bfloat16, true), (.bfloat16, .float32, false),
            (.float16, .bfloat16, true),
        ])
    func convolution(inputType: DType, weightType: DType, hasBase: Bool) throws {
        MLXRandom.seed(915)
        let (batch, dimensions) = (2, 128)
        let blocks = inkling2UnitRMS(MLXRandom.normal([3, batch, 1, dimensions]))
        let base = hasBase ? MLXRandom.normal([batch, 1, dimensions]) : nil
        let x = MLXRandom.normal([batch, 1, dimensions]).asType(inputType)
        let state = MLXRandom.normal([batch, 3, dimensions])
        let weight = (MLXRandom.normal([dimensions, 4, 1]) * 0.25).asType(weightType)
        let query = MLXRandom.normal([3, dimensions]) * 0.1
        let normWeight = MLXRandom.uniform(0.5 ..< 1.5, [1, dimensions])
        let result = try #require(
            inkling2FusedDepthMix(
                blocks: blocks, base: base, convolution: (x, state, weight), query: query,
                normWeight: normWeight, eps: 1e-6, outputType: inputType))
        let convolved = inkling2ShortConvolutionKernel(x, state: state, weight: weight)
        let expectedResidual = convolved.0.asType(.float32) + (base ?? MLXArray(Float(0)))
        let expectedNormed = inkling2UnitRMS(expectedResidual)
        let readout = inkling2DepthAttention(blocks: blocks, partial: expectedNormed, query: query)
        let normalized = (MLXFast.rmsNorm(readout, weight: nil, eps: 1e-6) * normWeight)
            .asType(inputType)
        let next = try #require(result.state)
        #expect(next.dtype == .float32)
        #expect(arrayEqual(next, convolved.1).item(Bool.self))
        #expect(maxDifference(result.residual, expectedResidual) < 2e-5)
        #expect(maxDifference(result.normed, expectedNormed) < 2e-5)
        #expect(maxDifference(result.readout, readout) < 5e-5)
        let actualNormalized = try #require(result.normalizedReadout)
        #expect(maxDifference(actualNormalized, normalized) < 2e-2)
    }

    @Test("Unsupported shapes and state precision retain the general path")
    func fallbacks() {
        let blocks = MLXArray.zeros([2, 1, 1, 64])
        let base = MLXArray.ones([1, 1, 64])
        let query = MLXArray.ones([1, 64])
        #expect(
            inkling2FusedDepthMix(
                blocks: blocks, base: nil, convolution: nil, query: query, normWeight: nil,
                eps: 1e-6) == nil)
        #expect(
            inkling2FusedDepthMix(
                blocks: blocks.asType(.bfloat16), base: base, convolution: nil, query: query,
                normWeight: nil, eps: 1e-6) == nil)
        #expect(
            inkling2FusedDepthMix(
                blocks: blocks, base: base, convolution: nil, query: MLXArray.ones([4, 64]),
                normWeight: nil, eps: 1e-6) == nil)
        #expect(
            inkling2FusedDepthMix(
                blocks: blocks, base: base, convolution: nil, query: query,
                normWeight: MLXArray.ones([2, 64]), eps: 1e-6) == nil)
        #expect(
            inkling2FusedDepthMix(
                blocks: blocks, base: base,
                convolution: (
                    base, MLXArray.zeros([1, 3, 64]).asType(.bfloat16), MLXArray.zeros([64, 4, 1])
                ),
                query: query, normWeight: nil, eps: 1e-6) == nil)
        #expect(
            Device.withDefaultDevice(.cpu) {
                inkling2FusedDepthMix(
                    blocks: blocks, base: base, convolution: nil, query: query,
                    normWeight: nil, eps: 1e-6) == nil
            })
    }
}
