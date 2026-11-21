// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A KV cache for a batch of left-padded sequences.
///
/// Port of mlx-lm's `BatchKVCache`, with the fixes mlx-vlm's port of it carries
/// (`mlx_vlm/models/cache.py` at mlx-vlm dcea787, noted where they apply). Rows are
/// left-padded so that sequences of
/// different lengths end at the same position. Prompts `[1, 3, 5]`, `[7]` and
/// `[2, 6, 8, 9]` are stored as
/// ```
/// [0, 1, 3, 5]
/// [0, 0, 0, 7]
/// [2, 6, 8, 9]
/// ```
/// with ``leftPadding`` `[1, 3, 0]`. ``KVCache/offset`` is the longest row's position, 4 here:
/// after finalization it counts tokens, not padding.
///
/// Row positions and padding live on the host. The arrays the model reads are
/// built from them on demand, so a decode step advances positions without a
/// kernel per layer, and a batch whose rows carry no padding decodes one token
/// with no mask at all.
public final class BatchKVCache: BaseKVCache, BatchPositionedKVCache, BatchableKVCache {
    /// Each row's left padding, shape `[B]`.
    public private(set) var leftPadding: MLXArray

    /// Each row's left padding on the host; `leftPadding` mirrors it.
    private var leftPaddingValues: [Int32]

    /// Each row's position on the host. It starts at `-leftPadding`.
    private var positions: [Int32]

    /// `positions` as an array, built on first use after every change.
    private var positionsArray: MLXArray?

    /// Each row's position, shape `[B]`. It starts at `-leftPadding`.
    public var batchOffset: MLXArray {
        if let positionsArray { return positionsArray }
        let array = MLXArray(positions)
        positionsArray = array
        return array
    }

    var keys: MLXArray?
    var values: MLXArray?

    /// The buffer position the next token is written to: the width in use, padding included.
    private var idx = 0

    /// The position of the longest row, as ``KVCache/offset`` counts it. After
    /// finalization that is tokens, not padding, so it is 0 until a row has consumed its
    /// left padding. During a ragged prefill it also counts right padding, until
    /// `finalize()` turns that into left padding. Rows keep their own positions in
    /// ``batchOffset``, so the batch as a whole cannot be moved.
    public override var offset: Int {
        get { Int(max(0, positions.max() ?? 0)) }
        set { preconditionFailure("BatchKVCache positions are per row; see batchOffset") }
    }

    /// The right padding of a ragged prefill, from `prepare` until `finalize`.
    private var rightPadding: MLXArray?
    private var rightPaddingValues: [Int32]?

    public var step = 256

    /// - Parameter leftPadding: the left padding of each row
    public init(leftPadding: [Int]) {
        let padding = leftPadding.map { Int32($0) }
        self.leftPaddingValues = padding
        self.leftPadding = MLXArray(padding)
        self.positions = padding.map { -$0 }
        super.init()
    }

    /// The number of rows.
    public var batchSize: Int { positions.count }

    /// The rows the buffers have room for. It exceeds ``batchSize`` after rows left in
    /// place: the spare slots follow the live rows, and the next row to join is written
    /// into one instead of every row being copied into a larger buffer.
    private var rowCapacity: Int { keys?.dim(0) ?? 0 }

    /// The buffer position of the batch's first column. It moves right when rows leave in
    /// place and every remaining row is left with padding: those columns are skipped
    /// rather than copied away, until the buffers next grow or are rebuilt.
    private var start = 0

    /// The buffers without their spare row slots and skipped columns.
    private var liveStorage: (keys: MLXArray, values: MLXArray)? {
        guard let keys, let values else { return nil }
        if rowCapacity == batchSize, start == 0 { return (keys, values) }
        return (
            keys[..<batchSize, 0..., start..., 0...], values[..<batchSize, 0..., start..., 0...]
        )
    }

    /// Whether any row carries left padding.
    private var hasLeftPadding: Bool { leftPaddingValues.contains { $0 != 0 } }

    private func setPositions(_ new: [Int32]) {
        positions = new
        positionsArray = nil
    }

    private func setLeftPadding(_ new: [Int32]) {
        leftPaddingValues = new
        leftPadding = MLXArray(new)
    }

    private func setRightPadding(_ new: [Int32]?) {
        rightPaddingValues = new
        rightPadding = new.map { MLXArray($0) }
    }

    /// Per-row positions, also when the cache is typed as `KVCache`.
    ///
    /// Rows at one shared position, such as a batch of one, get the scalar
    /// form, so no array is built per layer per step. Otherwise no copy:
    /// every change replaces the array rather than mutating it.
    public override var ropeOffset: RoPEOffset {
        if let first = positions.first, positions.allSatisfy({ $0 == first }) {
            return .scalar(Int(first))
        }
        return .batch(batchOffset)
    }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = idx
        if self.keys == nil || start + previous + keys.dim(2) > self.keys!.dim(2) {
            let length = (step + keys.dim(2) - 1) / step * step
            // Spare row slots grow with the live rows.
            let rows = self.keys?.dim(0) ?? keys.dim(0)
            let newKeys = MLXArray.zeros(
                [rows, keys.dim(1), length, keys.dim(3)], dtype: keys.dtype)
            let newValues = MLXArray.zeros(
                [rows, values.dim(1), length, values.dim(3)], dtype: values.dtype)
            if var currentKeys = self.keys, var currentValues = self.values {
                if start > 0 || previous % step != 0 {
                    // Growing copies the buffers anyway, so the skipped columns go too.
                    currentKeys = currentKeys[.ellipsis, start ..< start + previous, 0...]
                    currentValues = currentValues[.ellipsis, start ..< start + previous, 0...]
                    start = 0
                }
                self.keys = concatenated([currentKeys, newKeys], axis: 2)
                self.values = concatenated([currentValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        let n = Int32(keys.dim(2))
        idx += keys.dim(2)
        setPositions(positions.map { $0 + n })
        guard rowCapacity > batchSize || start > 0 else {
            self.keys![.ellipsis, previous ..< idx, 0...] = keys
            self.values![.ellipsis, previous ..< idx, 0...] = values
            return (self.keys![.ellipsis, ..<idx, 0...], self.values![.ellipsis, ..<idx, 0...])
        }
        let rows = batchSize
        let written = start + previous ..< start + idx
        self.keys![..<rows, 0..., written, 0...] = keys
        self.values![..<rows, 0..., written, 0...] = values
        let used = start ..< start + idx
        return (self.keys![..<rows, 0..., used, 0...], self.values![..<rows, 0..., used, 0...])
    }

    public override var state: [MLXArray] {
        get {
            guard let (keys, values) = liveStorage else { return [batchOffset, leftPadding] }
            return [
                keys[.ellipsis, ..<idx, 0...], values[.ellipsis, ..<idx, 0...],
                batchOffset, leftPadding,
            ]
        }
        set {
            switch newValue.count {
            case 2:
                keys = nil
                values = nil
                idx = 0
                start = 0
                setPositions(newValue[0].asArray(Int32.self))
                setLeftPadding(newValue[1].asArray(Int32.self))
            case 4:
                keys = newValue[0]
                values = newValue[1]
                idx = newValue[0].dim(2)
                start = 0
                setPositions(newValue[2].asArray(Int32.self))
                setLeftPadding(newValue[3].asArray(Int32.self))
            default:
                fatalError("BatchKVCache state must have 2 or 4 arrays, got \(newValue.count)")
            }
        }
    }

    public override var isTrimmable: Bool { true }

    public override func isTrimmable(after positions: Int) -> Bool {
        // A round can discard its whole write only after every row has consumed its padding.
        positions >= 0
            && (positions == 0
                || (!self.positions.isEmpty && self.positions.allSatisfy { $0 >= 0 }))
    }

    /// No row can lose more tokens than the shortest row holds.
    public override var maxTrimCount: Int {
        let shortest = Int(positions.min() ?? 0)
        return min(idx, max(0, shortest))
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(maxTrimCount, n)
        idx -= trimmed
        setPositions(positions.map { $0 - Int32(trimmed) })
        return trimmed
    }

    /// No mask for one token when no row is padded and the window, if any,
    /// covers every retained key; otherwise an explicit mask, so left padding
    /// and the window stay masked. This cache never evicts, so a window
    /// smaller than the history still needs the mask.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let windowCoversKeys = windowSize.map({ $0 >= idx + n }) ?? true
        if n == 1, !hasLeftPadding, windowCoversKeys {
            return .none
        }
        let mask = createCausalMask(
            n: n, offset: idx, windowSize: windowSize, leftPadding: leftPadding)
        if windowCoversKeys {
            PaddingMasks.shared.record(
                mask, offset: idx, n: n, leftPadding: leftPaddingValues)
        }
        return .array(mask)
    }

    /// The left padding to pass to the attention kernel in place of `mask`, or `nil`
    /// when `mask` must be applied as given.
    ///
    /// A model builds its mask once, from one layer's cache, and hands it to every
    /// layer. When that mask came from ``makeMask(n:windowSize:returnArray:)`` with no
    /// window cutting keys, and this cache has the same padding and offset, the
    /// mask says nothing beyond causality and this cache's ``leftPadding``. Rows
    /// whose padding reaches into the new queries are left to the mask: the kernel
    /// does not accept them.
    func leftPaddingForAttention(mask: MLXArray, n: Int) -> MLXArray? {
        guard let maxPadding = leftPaddingValues.max(), maxPadding <= idx,
            PaddingMasks.shared.matches(
                mask, offset: idx, n: n, leftPadding: leftPaddingValues)
        else { return nil }
        return leftPadding
    }

    /// Prepares a right-padded prefill of rows with the given lengths.
    public override func prepare(lengths: [Int]?) {
        guard let lengths, let longest = lengths.max(), lengths.contains(where: { $0 < longest })
        else { return }
        setRightPadding(lengths.map { Int32(longest - $0) })
    }

    public override func prepare(lengths: MLXArray?) {
        prepare(lengths: lengths?.asArray(Int.self))
    }

    /// Moves the right padding of a prefill to the left, where the mask excludes it.
    public override func finalize() {
        guard let rightPadding, let rightPaddingValues else { return }
        // A circular roll of the whole buffer lands each row's tokens at
        // `padding ..< idx`; what wraps around is masked or unused.
        if let (keys, values) = liveStorage {
            self.keys = dynamicRoll(keys, shifts: rightPadding[0..., .newAxis], axis: 2)
            self.values = dynamicRoll(values, shifts: rightPadding[0..., .newAxis], axis: 2)
            start = 0
        }
        setPositions(zip(positions, rightPaddingValues).map { $0 - $1 })
        setLeftPadding(zip(leftPaddingValues, rightPaddingValues).map { $0 + $1 })
        setRightPadding(nil)
    }

    /// Keeps only the rows at `batchIndices`, in that order.
    public func filter(batchIndices: MLXArray) {
        let rows = batchIndices.asArray(Int32.self).map { Int($0) }
        guard !rows.isEmpty else {
            keys = nil
            values = nil
            setRightPadding(nil)
            idx = 0
            start = 0
            setLeftPadding([])
            setPositions([])
            return
        }
        let indices = MLXArray(rows.map { Int32($0) })
        let storage = liveStorage
        keys = storage?.keys[indices]
        values = storage?.values[indices]
        start = 0
        setPositions(rows.map { positions[$0] })
        var padding = rows.map { leftPaddingValues[$0] }
        // A ragged prefill's right padding goes with its rows, as in mlx-vlm:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1673-L1674
        setRightPadding(rightPaddingValues.map { values in rows.map { values[$0] } })

        // Drop the left padding that every remaining row has, but no more than the buffer
        // holds: a chunked prefill can be filtered before all of its padding is written.
        // mlx-vlm bounds `min_left_pad` by `_idx` the same way:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1677
        let shared = min(padding.min() ?? 0, Int32(idx))
        if shared > 0, keys != nil {
            keys = keys?[.ellipsis, Int(shared)..., 0...]
            values = values?[.ellipsis, Int(shared)..., 0...]
            idx -= Int(shared)
            padding = padding.map { $0 - shared }
        }
        setLeftPadding(padding)
    }

    public func extend(other: any KVCache) {
        guard let other = other as? BatchKVCache else {
            preconditionFailure("BatchKVCache can only be extended with a BatchKVCache")
        }
        extend(other: other)
    }

    public func validateExtension(other: any KVCache) throws {
        guard let other = other as? BatchKVCache else {
            throw KVCacheError(message: "BatchKVCache can only append a BatchKVCache")
        }
        guard rightPadding == nil, other.rightPadding == nil else {
            throw KVCacheError(message: "BatchKVCache extension requires finalized rows")
        }
        for row in [self, other] {
            guard row.leftPaddingValues.count == row.batchSize,
                row.leftPadding.ndim == 1, row.leftPadding.size == row.batchSize,
                zip(row.positions, row.leftPaddingValues).allSatisfy({
                    Int($0) + Int($1) == row.idx
                })
            else {
                throw KVCacheError(message: "BatchKVCache metadata must agree on the row count")
            }
        }
        try validateBatchKVExtension(
            keys: liveStorage?.keys, values: liveStorage?.values, batchSize: batchSize, end: idx,
            otherKeys: other.liveStorage?.keys, otherValues: other.liveStorage?.values,
            otherBatchSize: other.batchSize, otherEnd: other.idx)
    }

    /// Appends the rows of `other`. Port of mlx-lm's `BatchKVCache.extend`: rows without keys
    /// yet join as all-padding rows, so empty and prefilled rows can share a batch, and a
    /// batch with no rows changes nothing. Both batches must be finalized: a ragged prefill in
    /// progress cannot take or give rows.
    public func extend(other: BatchKVCache) {
        guard other.batchSize > 0 else { return }
        // Right padding of a prefill in progress is not combined: finalize first.
        precondition(
            rightPadding == nil && other.rightPadding == nil,
            "BatchKVCache.extend takes finalized rows; finalize a ragged prefill first")
        guard batchSize > 0 else {
            // New array objects, so writes to either cache leave the other unchanged.
            keys = other.liveStorage?.keys[.ellipsis]
            values = other.liveStorage?.values[.ellipsis]
            start = 0
            idx = other.idx
            setLeftPadding(other.leftPaddingValues)
            setPositions(other.positions)
            setRightPadding(other.rightPaddingValues)
            return
        }
        guard let templateKeys = keys ?? other.keys, let templateValues = values ?? other.values
        else {
            setLeftPadding(leftPaddingValues + other.leftPaddingValues)
            setPositions(positions + other.positions)
            return
        }
        if extendIntoSpareRows(other) { return }
        // Rows without keys take zero-width arrays, which the padding below fills.
        func arrays(_ cache: BatchKVCache) -> (MLXArray, MLXArray) {
            if let storage = cache.liveStorage { return storage }
            return (
                MLXArray.zeros(
                    [cache.batchSize, templateKeys.dim(1), 0, templateKeys.dim(3)],
                    dtype: templateKeys.dtype),
                MLXArray.zeros(
                    [cache.batchSize, templateValues.dim(1), 0, templateValues.dim(3)],
                    dtype: templateValues.dtype)
            )
        }
        let (ownKeys, ownValues) = arrays(self)
        let (otherKeys, otherValues) = arrays(other)

        let end = max(idx, other.idx)
        let width = max(ownKeys.dim(2), otherKeys.dim(2))
        let own = rightJustify(ownKeys, ownValues, end: idx, to: end, width: width)
        let added = rightJustify(otherKeys, otherValues, end: other.idx, to: end, width: width)
        keys = concatenated([own.keys, added.keys])
        values = concatenated([own.values, added.values])
        start = 0
        let ownShift = Int32(end - idx)
        let addedShift = Int32(end - other.idx)
        setLeftPadding(
            leftPaddingValues.map { $0 + ownShift }
                + other.leftPaddingValues.map { $0 + addedShift })
        setPositions(positions + other.positions)
        idx = end
    }

    /// Writes the rows of `other` into this batch's spare row slots, when it has enough of
    /// them and the new rows are no longer than the batch. Only the new rows are written:
    /// the live rows stay where they are, in the buffer they are in.
    ///
    /// - Returns: `false`, with nothing changed, when the rows have to be appended by copy
    private func extendIntoSpareRows(_ other: BatchKVCache) -> Bool {
        guard !inPlaceBatchResizeDisabled, keys != nil, values != nil,
            let added = other.liveStorage,
            rowCapacity - batchSize >= other.batchSize, other.idx <= idx
        else { return false }
        // The same contents `extend` gives the rows by copy: tokens ending at `idx`, zeros
        // to their left.
        let rows = rightJustify(
            added.keys[.ellipsis, ..<other.idx, 0...], added.values[.ellipsis, ..<other.idx, 0...],
            end: other.idx, to: idx, width: idx)
        let slots = batchSize ..< batchSize + other.batchSize
        keys![slots, 0..., start ..< start + idx, 0...] = rows.keys
        values![slots, 0..., start ..< start + idx, 0...] = rows.values
        let shift = Int32(idx - other.idx)
        setLeftPadding(leftPaddingValues + other.leftPaddingValues.map { $0 + shift })
        setPositions(positions + other.positions)
        return true
    }

    /// The order the rows are in after the rows at `removed` leave a batch of `count` and
    /// its last rows move into their slots: entry `i` is the current index of the row that
    /// slot `i` holds afterwards. Only the rows that change slot have to be copied.
    static func orderAfterRemoving(_ removed: [Int], from count: Int) -> [Int] {
        var order = Array(0 ..< count)
        for index in removed.sorted(by: >) {
            let last = order.removeLast()
            if index < order.count { order[index] = last }
        }
        return order
    }

    /// Whether ``moveRows(to:moved:)`` can leave the rows in `order` without copying the
    /// rows that stay in their slots.
    ///
    /// Not when half the buffers' rows, or half their columns in use, would then hold
    /// nothing: a shrinking batch gives memory back through ``filter(batchIndices:)``.
    func canMoveRows(to order: [Int]) -> Bool {
        guard !inPlaceBatchResizeDisabled, keys != nil, values != nil, rightPadding == nil,
            !order.isEmpty, rowCapacity < 2 * order.count
        else { return false }
        let shared = Int(min(order.map { leftPaddingValues[$0] }.min() ?? 0, Int32(idx)))
        return start + shared <= idx - shared
    }

    /// The rows `order` moves to another slot, as arrays of their own: keys, then values.
    /// Empty when no row moves.
    ///
    /// Evaluate them before ``moveRows(to:moved:)``. A row copied out of the buffer while
    /// the buffer is still being read cannot be written back into it in place.
    func rowsMoving(to order: [Int]) -> [MLXArray] {
        let sources = order.enumerated().filter { $0.offset != $0.element }.map {
            Int32($0.element)
        }
        guard !sources.isEmpty, let keys, let values else { return [] }
        let indices = MLXArray(sources)
        return [keys[indices], values[indices]]
    }

    /// Leaves the rows in `order`, which ``canMoveRows(to:)`` accepted, writing `moved`,
    /// from ``rowsMoving(to:)``, into the slots that change rows. The slots after
    /// `order.count` become spare.
    func moveRows(to order: [Int], moved: [MLXArray]) {
        var next = 0
        for (slot, source) in order.enumerated() where slot != source {
            keys![slot ..< slot + 1] = moved[0][next ..< next + 1]
            values![slot ..< slot + 1] = moved[1][next ..< next + 1]
            next += 1
        }
        setPositions(order.map { positions[$0] })
        // The left padding every remaining row has is skipped, where `filter` slices it off.
        let padding = order.map { leftPaddingValues[$0] }
        let shared = min(padding.min() ?? 0, Int32(idx))
        start += Int(shared)
        idx -= Int(shared)
        setLeftPadding(padding.map { $0 - shared })
    }

    /// The row at `index` without its left padding, as a ``KVCacheSimple``. A row whose
    /// padding covers every written position comes back empty, as mlx-vlm's slice does.
    public func extract(_ index: Int) -> any KVCache {
        // mlx-vlm checks the row index first too:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1730-L1736
        precondition(
            leftPaddingValues.indices.contains(index), "BatchKVCache row index out of range")
        precondition(
            rightPadding == nil,
            "BatchKVCache.extract takes finalized rows; finalize a ragged prefill first")
        let cache = KVCacheSimple()
        let padding = Int(leftPaddingValues[index])
        if let keys, let values, padding < idx {
            let columns = start + padding ..< start + idx
            cache.keys = MLX.contiguous(keys[index ..< index + 1, 0..., columns, 0...])
            cache.values = MLX.contiguous(values[index ..< index + 1, 0..., columns, 0...])
            cache.offset = idx - padding
        }
        return cache
    }

    /// The rows of `caches`, exactly `KVCacheSimple` rows, as one batch. See
    /// ``KVCacheSimple/merge(_:)``.
    public static func merge(_ caches: [any KVCache]) throws -> any BatchableKVCache {
        let rows = try batchRows(caches, of: KVCacheSimple.self)
        try validateBatchKVRows(rows.map { ($0.keys, $0.values, $0.offset) })
        return BatchKVCache(merging: rows)
    }

    /// The rows of `caches` as one batch, left-padded to the longest row.
    fileprivate convenience init(merging caches: [KVCacheSimple]) {
        let longest = caches.map(\.offset).max() ?? 0
        self.init(leftPadding: caches.map { longest - $0.offset })
        // An all-empty merge is an empty batch; the first update sizes it. mlx-vlm returns
        // an empty batch for it too:
        // https://github.com/Blaizzy/mlx-vlm/blob/dcea787/mlx_vlm/models/cache.py#L1750-L1751
        guard let template = caches.first(where: { $0.keys != nil }),
            let templateKeys = template.keys, let templateValues = template.values
        else { return }
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for cache in caches {
            if let rowKeys = cache.keys, let rowValues = cache.values, cache.offset > 0 {
                // Slice off allocation slack, then pad on the left to `longest`.
                let row = rightJustify(
                    rowKeys[.ellipsis, ..<cache.offset, 0...],
                    rowValues[.ellipsis, ..<cache.offset, 0...],
                    end: cache.offset, to: longest, width: longest)
                keys.append(row.keys)
                values.append(row.values)
            } else {
                keys.append(
                    MLXArray.zeros(
                        [1, templateKeys.dim(1), longest, templateKeys.dim(3)],
                        dtype: templateKeys.dtype))
                values.append(
                    MLXArray.zeros(
                        [1, templateValues.dim(1), longest, templateValues.dim(3)],
                        dtype: templateValues.dtype))
            }
        }
        self.keys = concatenated(keys)
        self.values = concatenated(values)
        idx = longest
        setPositions(caches.map { Int32($0.offset) })
    }

    public override func copy() -> any KVCache {
        let copy = BatchKVCache(leftPadding: [])
        copy.setLeftPadding(leftPaddingValues)
        copy.setPositions(positions)
        copy.keys = keys?[.ellipsis]
        copy.values = values?[.ellipsis]
        copy.setRightPadding(rightPaddingValues)
        copy.idx = idx
        copy.start = start
        copy.step = step
        return copy
    }
}

extension KVCacheSimple {
    /// The rows of `caches` as one batch, left-padded so every row ends at the
    /// longest row's position. Port of mlx-lm's `KVCache.merge`. Every cache
    /// must be exactly a `KVCacheSimple` holding one row of one layout; empty
    /// caches become empty rows. ``BatchKVCache/merge(_:)`` throws where this stops.
    public static func merge(_ caches: [KVCacheSimple]) -> BatchKVCache {
        precondition(!caches.isEmpty, "merge needs a cache")
        // Exact type: a ChunkedKVCache can report an offset beyond the arrays it holds.
        precondition(
            caches.allSatisfy { type(of: $0) == KVCacheSimple.self },
            "KVCacheSimple.merge takes exactly KVCacheSimple rows")
        do {
            try validateBatchKVRows(caches.map { ($0.keys, $0.values, $0.offset) })
        } catch {
            preconditionFailure("KVCacheSimple.merge: \(error.localizedDescription)")
        }
        return BatchKVCache(merging: caches)
    }
}

/// `keys` and `values`, whose rows end at position `end`, padded to end at `newEnd`
/// in a buffer `width` wide.
func rightJustify(
    _ keys: MLXArray, _ values: MLXArray, end: Int, to newEnd: Int, width: Int
) -> (keys: MLXArray, values: MLXArray) {
    var keys = keys
    var values = values
    let left = newEnd - end
    var right = width - keys.dim(2) - left
    if right < 0 {
        keys = keys[.ellipsis, ..<(keys.dim(2) + right), 0...]
        values = values[.ellipsis, ..<(values.dim(2) + right), 0...]
        right = 0
    }
    if left != 0 || right != 0 {
        let widths: [IntOrPair] = [0, 0, .init((left, right)), 0]
        keys = MLX.padded(keys, widths: widths)
        values = MLX.padded(values, widths: widths)
    }
    return (keys, values)
}

/// Rolls each row of `x` along `axis` by its own shift. Port of mlx-lm's `dynamic_roll`.
func dynamicRoll(_ x: MLXArray, shifts: MLXArray, axis: Int) -> MLXArray {
    let axis = axis >= 0 ? axis : x.ndim + axis
    let n = x.dim(axis)
    var positionShape = [Int](repeating: 1, count: x.ndim)
    positionShape[axis] = n
    var shiftShape = [Int](repeating: 1, count: x.ndim)
    for dimension in 0 ..< min(shifts.ndim, x.ndim) {
        shiftShape[dimension] = shifts.dim(dimension)
    }
    let positions = MLXArray(Int32(0) ..< Int32(n)).reshaped(positionShape)
    let size = MLXArray(Int32(n))
    // Adding `n` keeps the index non-negative for negative shifts.
    let indices = ((positions - shifts.reshaped(shiftShape)) % size + size) % size
    return takeAlong(x, indices.asType(.int32), axis: axis)
}

/// The masks ``BatchKVCache`` built from causality and left padding alone, so that
/// attention in any layer can tell them from masks that also apply a window.
///
/// Entries hold their masks weakly and drop out once a forward pass releases them.
private final class PaddingMasks: @unchecked Sendable {
    static let shared = PaddingMasks()

    private struct Entry {
        weak var mask: MLXArray?
        let offset: Int
        let n: Int
        let leftPadding: [Int32]
    }

    private let lock = NSLock()
    private var entries: [ObjectIdentifier: Entry] = [:]

    func record(_ mask: MLXArray, offset: Int, n: Int, leftPadding: [Int32]) {
        lock.withLock {
            entries = entries.filter { $0.value.mask != nil }
            entries[ObjectIdentifier(mask)] = Entry(
                mask: mask, offset: offset, n: n, leftPadding: leftPadding)
        }
    }

    func matches(_ mask: MLXArray, offset: Int, n: Int, leftPadding: [Int32]) -> Bool {
        lock.withLock {
            guard let entry = entries[ObjectIdentifier(mask)], entry.mask === mask else {
                return false
            }
            return entry.offset == offset && entry.n == n && entry.leftPadding == leftPadding
        }
    }
}

// BENCH ONLY: resize by copy as before, to measure what the in-place path buys.
private let inPlaceBatchResizeDisabled = getenv("BENCH_NO_INPLACE_RESIZE") != nil
