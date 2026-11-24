// Copyright © 2026 Apple Inc.

import MLX

extension Inkling2LanguageModel {
    /// Prefills every cache row while projecting only the final row to vocabulary logits.
    /// A drafter receives all requested hidden-state taps, including earlier chunks.
    package func inkling2Prepare(
        embeddings: MLXArray, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters, chunkShortPrompts: Bool = false
    ) throws -> PrepareResult {
        let total = embeddings.dim(1)
        let emits = state?[mtpEmitFlagKey] ?? false
        var features = [MLXArray]()
        let processed: Int
        if chunkShortPrompts || total > prefill.resolvedStepSize() {
            let reservation =
                !chunkShortPrompts && prefill.chunking == .remainder
                ? prefill.resolvedStepSize() : 1
            processed = try prefill.forEachChunk(total: total, reserving: reservation) { range in
                let output = inkling2Output(
                    embeddings: embeddings[0..., range, 0...], cache: cache, state: state,
                    outputRows: 0 ..< 1)
                if emits, let taps = output.state?[mtpLastHiddenStatesKey] {
                    features.append(taps)
                }
                asyncEval(cache, features)
            }
            if processed > 0 { eval(cache) }
        } else {
            processed = 0
        }
        let remaining = embeddings[0..., processed..., 0...]
        let last = remaining.dim(1) - 1
        let output = inkling2Output(
            embeddings: remaining, cache: cache, state: state, outputRows: last ..< last + 1)
        prefill.progress?(total, total)
        guard emits, var next = output.state,
            let taps = next[mtpLastHiddenStatesKey]
        else { return .logits(output) }
        features.append(taps)
        next[mtpLastHiddenStatesKey] = concatenated(features, axis: 1)
        return .logits(LMOutput(logits: output.logits, state: next))
    }
}
