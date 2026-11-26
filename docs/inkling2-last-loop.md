# Inkling 2 last-loop cache prototype

Callers can opt in without editing checkpoint files:

```swift
var configuration = ModelConfiguration(directory: modelDirectory)
configuration.inkling2RecurrentKVCacheMode = .lastLoop
// Pass configuration to LLMModelFactory or VLMModelFactory when loading.
```

The typed override defaults to `nil`, which preserves the checkpoint setting.
Passing `.perLoop` explicitly restores ordinary per-pass history even if the
checkpoint requests `last_loop`. Both factories apply the override to model
construction data in memory and retain it in the loaded context configuration.
An explicit `.lastLoop` request requires an Inkling 2 model with repeated layers.

This prototype adds `text_config.recurrent_kv_cache_mode` to Inkling 2's model
configuration. It defaults to `"per_loop"`. Set it to `"last_loop"` before loading
the model to retain attention history only for the final visit to each layer.

```json
{
  "text_config": {
    "recurrent_kv_cache_mode": "last_loop"
  }
}
```

Add the field to the existing `text_config`; the snippet omits the other required
model fields. Use a separate model directory and retain its own `config.json`.
Weights and tokenizer files can be symlinked from the original directory. Keep
saved prompt caches separate for the two configurations. Reload the model when
changing its mode.

## Behavior

Each layer's final loop pass writes its persistent attention keys, values and
positions. Earlier passes update temporary copies of that history, so they attend
to previous tokens' final-pass history and their own current tokens. All four
convolution states remain separate for every visit. Non-looped models and
single-pass loops retain their existing cache layout and behavior.

The persistent cache is a flat list of ordinary cache entries without shared
object aliases. It supports copying, serialization, rotating attention caches and
batched decoding through the existing cache infrastructure. Temporary attention
copies are created one visit at a time. Persistent cache bytes and peak live MLX
allocations are different measurements; both should be checked.

This mode changes the model's cached attention history. Matching the first full
forward does not establish equivalent subsequent logits, generated tokens or
quality. Results can depend on prefill chunk boundaries and on where a saved
prefix ends. Compare batching and cache continuation using the same chunk
schedule. The default remains `per_loop`.

For repeated layers, the prototype rejects rewind caches and DSpark speculative
decoding. The temporary-copy route has not been implemented for round-scoped
staged caches. Existing KV quantization/compression restrictions still apply.

## Validation

`Inkling2Tests` exercises configuration decoding, a separate per-visit reference
implementation of shared history, initial forward parity, reduced stored bytes,
convolution independence, bounded/sliding attention, cache copies and saved-cache
continuation, ragged batches, and explicit rewind/DSpark rejection.

The reference keeps the old per-visit layout and replaces earlier passes'
historical attention entries with copies of the final pass's entries before each
call. It preserves pass-local convolution state. This validates the intended
shared-history semantics without requiring equality with ordinary per-loop
decoding.

Run the repository's Xcode tests; `swift test` does not build MLX's Metal shaders:

```sh
xcodebuild test -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' -skipPackagePluginValidation
```

See [the Mini benchmark](../Benchmarks/Inkling2LastLoop) for the comparison harness
and recorded measurements. This is an opt-in prototype in the shared MLX model
implementation; no product dependency pins or application defaults are changed.

## Mini measurements, 2026-11-24

The local `netprojects/Inkling-2-Mini-MLX-4-8bit` checkpoint used the same weights
and MLX 0.32.5 backend for both modes. Two measured runs per mode used ABBA order,
512-token prefill chunks, eight warmup decode steps and 128 measured greedy decode
steps. The first emitted token comes from prefill, so each run compares 129 tokens.
All 95 Swift Testing tests in 13 suites and two XCTest tests passed.

| Prompt tokens | Per-loop cache, MiB | Last-loop cache, MiB | Reduction |
| --- | --- | --- | --- |
| 2,048 | 236.1 | 119.9 | 49.2% |
| 8,192 | 428.5 | 216.1 | 49.6% |

Attention-cache tensor storage fell by exactly 50%; convolution state remains
per visit. Overall peak live MLX allocations, including weights and prefill
temporaries, fell by only 0.8% and 1.3%, respectively (152 and 248 MiB). These are
allocator readings, not process RSS.

Median synchronous decode rates were 45.6 versus 46.0 tokens/s at 2K, and 44.2
versus 43.4 tokens/s at 8K. The small differences are within the variation between
the two runs; this does not establish a decode speed improvement.

Both chunked synthetic prompts differed at the first emitted token. With a
single 2K prefill, the first token matched and the second differed. Three short
chat prompts first differed at emitted tokens 32, 5 and 5. These fixed-step runs
continue past EOS and do not measure response quality. Output equivalence is not
a property of this mode.

The [recorded JSON](../Benchmarks/Inkling2LastLoop/Results/2026-11-24-mini.json)
includes raw tokens, phase memory peaks, timings, source and executable hashes,
and the test/backend scope. The full production dependency graph, real-model
vision, Small, tensor parallelism and concurrent-user throughput were not tested.

The design follows [Transformers PR #51161](https://github.com/huggingface/transformers/pull/51161).
