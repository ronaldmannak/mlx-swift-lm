// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Attention over a ``BatchKVCache`` passes the padding counts to the kernel when its
/// mask encodes nothing else, and keeps the mask otherwise.
@Suite(.serialized)
struct BatchLeftPaddingAttentionTests {

    private let heads = 4
    private let kvHeads = 2
    private let headDim = 64
    private var scale: Float { 1 / Float(headDim).squareRoot() }

    /// A two-row batch with left padding `[2, 0]` holding four positions.
    private func paddedCache(seed: UInt64) -> BatchKVCache {
        let cache = BatchKVCache(leftPadding: [2, 0])
        let (keys, values) = withRandomState(MLXRandom.RandomState(seed: seed)) {
            (
                MLXRandom.normal([2, kvHeads, 4, headDim]),
                MLXRandom.normal([2, kvHeads, 4, headDim])
            )
        }
        _ = cache.update(keys: keys, values: values)
        return cache
    }

    private func step(n: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            (
                MLXRandom.normal([2, heads, n, headDim]),
                MLXRandom.normal([2, kvHeads, n, headDim]),
                MLXRandom.normal([2, kvHeads, n, headDim])
            )
        }
    }

    @Test(
        "Decode and verify steps over a padded batch use the counts and match the mask",
        arguments: [1, 4])
    func paddingCountsMatchTheMask(n: Int) throws {
        let cache = paddedCache(seed: 1)
        let reference = paddedCache(seed: 1)
        guard case .array(let mask) = cache.makeMask(n: n, windowSize: nil, returnArray: false)
        else {
            Issue.record("A padded batch must return a mask array")
            return
        }
        #expect(cache.leftPaddingForAttention(mask: mask, n: n) != nil)

        let (queries, keys, values) = step(n: n, seed: 2)
        let actual = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale,
            mask: .array(mask))
        let (allKeys, allValues) = reference.update(keys: keys, values: values)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues, scale: scale,
            mask: .array(mask))
        #expect(maxAbsDifference(actual, expected) < 1e-4)
        #expect(cache.offset == 4 + n)
    }

    @Test("Every layer's cache with the same padding and offset reuses the shared mask")
    func sharedMaskAcrossLayers() throws {
        let first = paddedCache(seed: 1)
        let second = paddedCache(seed: 3)
        guard case .array(let mask) = first.makeMask(n: 1, windowSize: nil, returnArray: false)
        else {
            Issue.record("A padded batch must return a mask array")
            return
        }
        #expect(second.leftPaddingForAttention(mask: mask, n: 1) != nil)

        let other = BatchKVCache(leftPadding: [1, 0])
        let zeros = MLXArray.zeros([2, kvHeads, 4, headDim])
        _ = other.update(keys: zeros, values: zeros)
        #expect(
            other.leftPaddingForAttention(mask: mask, n: 1) == nil,
            "a cache with other padding needs its own mask")
    }

    @Test("A window that cuts keys keeps the mask")
    func windowKeepsTheMask() throws {
        let cache = paddedCache(seed: 1)
        let reference = paddedCache(seed: 1)
        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: 2, returnArray: false)
        else {
            Issue.record("A windowed batch must return a mask array")
            return
        }
        #expect(cache.leftPaddingForAttention(mask: mask, n: 1) == nil)

        let (queries, keys, values) = step(n: 1, seed: 2)
        let actual = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale,
            mask: .array(mask))
        let (allKeys, allValues) = reference.update(keys: keys, values: values)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues, scale: scale,
            mask: .array(mask))
        #expect(maxAbsDifference(actual, expected) < 1e-4)
    }

    @Test("A window that covers every key still uses the counts")
    func coveringWindowUsesTheCounts() throws {
        let cache = paddedCache(seed: 1)
        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: 5, returnArray: false)
        else {
            Issue.record("A padded batch must return a mask array")
            return
        }
        #expect(cache.leftPaddingForAttention(mask: mask, n: 1) != nil)
    }

    @Test("A mask the cache did not build keeps the mask")
    func foreignMaskKeepsTheMask() {
        let cache = paddedCache(seed: 1)
        let mask = createCausalMask(n: 1, offset: 4, leftPadding: cache.leftPadding)
        #expect(cache.leftPaddingForAttention(mask: mask, n: 1) == nil)
    }

    @Test("Padding that reaches into the new queries keeps the mask")
    func paddingIntoQueriesKeepsTheMask() {
        let cache = BatchKVCache(leftPadding: [2, 0])
        guard case .array(let mask) = cache.makeMask(n: 4, windowSize: nil, returnArray: false)
        else {
            Issue.record("A padded batch must return a mask array")
            return
        }
        #expect(cache.leftPaddingForAttention(mask: mask, n: 4) == nil)
    }
}
