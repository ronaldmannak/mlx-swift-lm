# DFlashBatch

A local Release benchmark of the integration branch's real `BatchTokenIterator` against `BatchedDFlashTokenIterator`. It loads one trained target and drafter using `BenchmarkHelpers.NoOpTokenizerLoader`. Both stay resident in both arms. No checkpoint is downloaded.

Use the Qwen 3.5 4B BF16 target and trained `z-lab/Qwen3.5-4B-DFlash` v1 drafter. The included four held-out prompts are byte-identical to the prior speculative batch study. They cover rewriting, JSON extraction, a Swift function and ocean tides, with Qwen token lengths 75, 80, 57 and 35.

Build with Xcode Release so the MLX Metal bundle is available beside the executable. From the repository root:

```sh
cd Benchmarks/DFlashBatch
xcodebuild -scheme DFlashBatchBench -configuration Release -destination 'platform=macOS' \
  -derivedDataPath /tmp/dflash-integration-build -skipPackagePluginValidation \
  'GCC_PREPROCESSOR_DEFINITIONS=$(inherited) NDEBUG=1' build
/tmp/dflash-integration-build/Build/Products/Release/dflashbatch \
  "$TARGET_DIR" "$DRAFT_DIR" qwen-holdout-prompts.json both \
  > /tmp/dflash-integration.jsonl
python3 summarize.py /tmp/dflash-integration.jsonl
```

`MLX_SWIFT_LM_PATH` can point the package at another compatible checkout. Run arms sequentially with no other GPU work. Model loading is outside the clock. Each scenario runs two complete warmups per arm, followed by three measured repetitions. Arm order alternates on every repetition. Both engines use temperature 0.8, top-p 0.9 and independent seeds 37 through 40. The parameters record a prefill step of 512. Ordinary batching honors that chunk budget, while DFlash prefills each admitted row synchronously in full. Every supplied prompt is shorter than 512 tokens, so each fits into a single ordinary prefill chunk as well. This benchmark does not assess long-prompt chunking or resumable-prefill fairness. DFlash uses block size four. EOS stopping is disabled in both arms so the fixed workload emits exactly 64 tokens per row, including the first token. This is a controlled workload rather than normal chat stop behavior.

## Scenarios

The optional last argument is `fixed` (default), `staggered` or `both`.

- `fixed` admits all four requests at time zero. Every row must return exactly 64 tokens and a completion.
- `staggered` starts rows zero and one. After a returned chunk takes row zero to at least eight tokens, row two arrives. After row zero reaches at least 16, row one is removed while active and row three arrives. Rows zero, two and three must each return 64 tokens. Removed row one must have a nonempty partial output and never emit another chunk. The trigger runs at chunk boundaries, so the recorded token count may exceed its threshold. Each request keeps seed `37 + stableID`, including after removal compacts physical rows.

Completed rows are retired after their final chunks. A cache-state fence runs before retirement. The run ends only after iterator synchronization and the GPU stream fence. This includes final-state work rather than allowing discarded lazy arrays to make the result appear faster. The benchmark records real membership changes and keeps counters for removed rows.

The staggered workload has the same event rule in both engines, but row one's cancellation count may differ. Its aggregate rate uses the actual delivered token count. Inspect per-row arrival latency and output counts as well as throughput. Do not combine its results with the fixed 256-token workload.

## Timing and checks

The clock starts before iterator construction and initial admission. It includes prefill, sampling, decoding, membership changes, row retirement and final synchronization. `aggregateTokensPerSecond` is all delivered tokens divided by that complete elapsed time. It is not decode-only throughput.

Each row records its admission, first token, completion or removal, exact output tokens and delivery timestamps. A speculative chunk's tokens share one visible timestamp. The summary reports medians of per-run p95 gaps, including zero gaps within a chunk and pauses between chunks. `admissionToFirstMSByRow` uses each request's own arrival clock. Peak MLX memory is decimal GB and includes both resident models. Ordinary `iteratorCalls` are not target forward counts. `targetVerifyCalls` is supplied only by DFlash's cumulative verifier counter.

The executable checks output lengths, admission/removal behavior and exact repetition of each arm's measured token streams. It does not require sampled ordinary and DFlash outputs to match: DFlash's rejection sampler consumes random draws differently. Raw tokens remain available for separate distribution and numerical audits.
