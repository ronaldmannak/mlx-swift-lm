// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Rows leaving and joining a ``BatchKVCache`` without its buffers being rebuilt must leave
/// it in the state the copying path gives: `filter(batchIndices:)` for a leave, and
/// `extend(other:)` on a batch with no spare row for a join.
@Suite(.serialized)
struct BatchKVCacheInPlaceResizeTests {
    private let heads = 2
    private let headDim = 4

    /// A batch whose rows and positions carry distinct stamps.
    private func batch(leftPadding: [Int], width: Int) -> BatchKVCache {
        let cache = BatchKVCache(leftPadding: leftPadding)
        let kv = makePositionKV(
            positions: 0 ..< width, heads: heads, headDim: headDim,
            batchSize: leftPadding.count, rowStride: 1000)
        _ = cache.update(keys: kv.0, values: kv.1)
        return cache
    }

    /// One row of `length` positions, stamped apart from the batch's rows.
    private func row(length: Int) -> BatchKVCache {
        let cache = BatchKVCache(leftPadding: [0])
        let kv = makePositionKV(positions: 9000 ..< 9000 + length, heads: heads, headDim: headDim)
        _ = cache.update(keys: kv.0, values: kv.1)
        return cache
    }

    /// Feeds every cache the same `count` new positions.
    private func feed(_ caches: [BatchKVCache], from position: Int, count: Int = 1) {
        let kv = makePositionKV(
            positions: position ..< position + count, heads: heads, headDim: headDim,
            batchSize: caches[0].batchSize, rowStride: 1000)
        for cache in caches { _ = cache.update(keys: kv.0, values: kv.1) }
    }

    /// Removes `removed` from `cache` in place and from `reference` by copy.
    private func remove(_ removed: [Int], from cache: BatchKVCache, reference: BatchKVCache)
        throws
    {
        let order = BatchKVCache.orderAfterRemoving(removed, from: cache.batchSize)
        try #require(cache.canMoveRows(to: order))
        let moved = cache.rowsMoving(to: order)
        eval(moved)
        cache.moveRows(to: order, moved: moved)
        reference.filter(batchIndices: MLXArray(order.map { Int32($0) }))
    }

    private func expectSameState(
        _ cache: BatchKVCache, _ reference: BatchKVCache,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let (state, expected) = (cache.state, reference.state)
        #expect(state.count == expected.count, sourceLocation: sourceLocation)
        for (array, other) in zip(state, expected) {
            #expect(array.shape == other.shape, sourceLocation: sourceLocation)
            guard array.shape == other.shape else { continue }
            #expect(maxAbsDifference(array, other) == 0, sourceLocation: sourceLocation)
        }
        #expect(cache.offset == reference.offset, sourceLocation: sourceLocation)
        #expect(cache.batchSize == reference.batchSize, sourceLocation: sourceLocation)
        for index in 0 ..< min(cache.batchSize, reference.batchSize) {
            let (row, other) = (cache.extract(index), reference.extract(index))
            #expect(row.offset == other.offset, sourceLocation: sourceLocation)
            for (array, expected) in zip(row.state, other.state) {
                #expect(array.shape == expected.shape, sourceLocation: sourceLocation)
                guard array.shape == expected.shape else { continue }
                #expect(maxAbsDifference(array, expected) == 0, sourceLocation: sourceLocation)
            }
        }
    }

    @Test("The last rows take the slots of the rows that leave")
    func orderAfterRemoving() {
        #expect(BatchKVCache.orderAfterRemoving([1], from: 4) == [0, 3, 2])
        #expect(BatchKVCache.orderAfterRemoving([3], from: 4) == [0, 1, 2])
        #expect(BatchKVCache.orderAfterRemoving([0, 3], from: 4) == [2, 1])
        #expect(BatchKVCache.orderAfterRemoving([0, 1, 2], from: 4) == [3])
        #expect(BatchKVCache.orderAfterRemoving([], from: 3) == [0, 1, 2])
    }

    @Test("A row that leaves in place leaves the state filter gives, and keeps the buffers")
    func leaveInPlace() throws {
        let cache = batch(leftPadding: [0, 2, 0, 1], width: 5)
        let reference = batch(leftPadding: [0, 2, 0, 1], width: 5)
        try remove([1], from: cache, reference: reference)
        expectSameState(cache, reference)
        // Four row slots remain, three in use.
        #expect(cache.innerState()[0].dim(0) == 4)
        #expect(reference.innerState()[0].dim(0) == 3)

        feed([cache, reference], from: 5)
        expectSameState(cache, reference)
        #expect(cache.copy().state.map(\.shape) == reference.state.map(\.shape))
    }

    @Test("Left padding every remaining row shares is skipped, as filter drops it")
    func leaveOfTheLongestRow() throws {
        let cache = batch(leftPadding: [0, 2, 1, 3], width: 6)
        let reference = batch(leftPadding: [0, 2, 1, 3], width: 6)
        try remove([0], from: cache, reference: reference)
        #expect(cache.leftPadding.asArray(Int32.self) == [2, 1, 0])
        expectSameState(cache, reference)

        feed([cache, reference], from: 6)
        expectSameState(cache, reference)

        // Growing past the buffers' width keeps the rows, and the spare slot.
        feed([cache, reference], from: 7, count: 300)
        expectSameState(cache, reference)
        feed([cache, reference], from: 307)
        expectSameState(cache, reference)
    }

    @Test("A row joins a freed slot with the state extend gives by copy")
    func joinIntoSpareRow() throws {
        let cache = batch(leftPadding: [0, 0, 0, 0], width: 5)
        let reference = batch(leftPadding: [0, 0, 0, 0], width: 5)
        try remove([0], from: cache, reference: reference)

        cache.extend(other: row(length: 3))
        reference.extend(other: row(length: 3))
        expectSameState(cache, reference)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 0, 0, 2])
        // Written into the slot the first row left: the buffers did not grow.
        #expect(cache.innerState()[0].dim(0) == 4)

        feed([cache, reference], from: 5)
        expectSameState(cache, reference)
    }

    @Test("A row joins a freed slot behind skipped columns")
    func joinAfterSkippedColumns() throws {
        let cache = batch(leftPadding: [0, 1, 2, 1], width: 5)
        let reference = batch(leftPadding: [0, 1, 2, 1], width: 5)
        try remove([0], from: cache, reference: reference)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 0, 1])

        cache.extend(other: row(length: 2))
        reference.extend(other: row(length: 2))
        expectSameState(cache, reference)
        #expect(cache.innerState()[0].dim(0) == 4)
        feed([cache, reference], from: 5)
        expectSameState(cache, reference)
    }

    @Test("A row longer than the batch joins by copy")
    func joinOfALongerRow() throws {
        let cache = batch(leftPadding: [0, 0, 0], width: 4)
        let reference = batch(leftPadding: [0, 0, 0], width: 4)
        try remove([1], from: cache, reference: reference)

        cache.extend(other: row(length: 7))
        reference.extend(other: row(length: 7))
        expectSameState(cache, reference)
        #expect(cache.leftPadding.asArray(Int32.self) == [3, 3, 0])
        feed([cache, reference], from: 4)
        expectSameState(cache, reference)
    }

    @Test("A batch that shrinks by half gives its buffers back")
    func shrinkingBatchIsRebuilt() {
        let cache = batch(leftPadding: [0, 0, 0, 0], width: 5)
        #expect(cache.canMoveRows(to: BatchKVCache.orderAfterRemoving([2], from: 4)))
        #expect(!cache.canMoveRows(to: BatchKVCache.orderAfterRemoving([1, 2], from: 4)))

        // Skipping five of six columns would leave most of the buffers unused.
        let padded = batch(leftPadding: [0, 5, 5], width: 6)
        #expect(!padded.canMoveRows(to: BatchKVCache.orderAfterRemoving([0], from: 3)))
    }
}
