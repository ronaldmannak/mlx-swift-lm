// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct Inkling2RouterKernelTests {
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
