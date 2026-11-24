# Inkling 2 last-loop cache benchmark

This standalone harness compares `per_loop` and experimental `last_loop` modes
using the same local Mini checkpoint, source snapshot and MLX backend. It does
not modify the checkpoint or the package's production manifest.

The [prototype guide](../../docs/inkling2-last-loop.md) summarizes the
[2026-11-24 measurements](Results/2026-11-24-mini.json), including observed token
differences and total peak allocation savings.

Prerequisites: Apple silicon, Xcode with the Metal toolchain, a local
`mlx-swift` checkout compatible with this source, and an Inkling 2 Mini MLX
checkpoint. The backend's own dependencies must also be available locally for
an offline build.

Prepare a new temporary workspace, supplying your local paths:

```sh
python3 Benchmarks/Inkling2LastLoop/prepare.py \
  --backend /path/to/mlx-swift \
  --model /path/to/Inkling-2-Mini-MLX-4-8bit \
  --workspace /tmp/inkling-last-loop-run
```

The script copies the source and focused tests into an isolated package, records
file hashes in `source.json`, and creates two checkpoint views with symlinked
weights. Only their private `config.json` files differ:
`text_config.recurrent_kv_cache_mode` is `per_loop` or `last_loop`.

Build and run the focused Inkling tests with Xcode, which builds the benchmark
executable and its Metal resources too:

```sh
cd /tmp/inkling-last-loop-run/view
xcodebuild test -scheme InklingBench -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/inkling-last-loop-run/dd \
  -skipPackagePluginValidation -parallel-testing-enabled NO ENABLE_TESTABILITY=YES
```

From the source checkout, inspect the plan, then execute it:

```sh
python3 Benchmarks/Inkling2LastLoop/run.py --workspace /tmp/inkling-last-loop-run
python3 Benchmarks/Inkling2LastLoop/run.py --workspace /tmp/inkling-last-loop-run --execute
```

The default matrix uses ABBA order, 2,048- and 8,192-token synthetic prompts,
512-token prefill chunks, and 128 measured decode steps. Each measured length
receives an eight-step warmup at the same length. `--lengths`, `--decode` and
`--chunk` customize the matrix; setting the chunk larger than the prompt tests
a single prefill. An optional `--chat-fixtures` JSON list of objects containing
`name` and `tokens` adds natural prompts. Fixture lengths must be distinct.

Each decode run emits **129 tokens for 128 decode steps**: the first token comes
from prefill and is outside the decode timer. Outputs contain raw token IDs;
decoding and interpreting text requires the checkpoint's tokenizer.

Results go to a new timestamped directory under the workspace's `results`:
raw JSONL, stderr, the binary hash, the source snapshot and a comparison summary.
The benchmark uses a synchronous manual greedy loop for both modes, so its
throughput differs from the pipelined production iterator. The allocator pool
is cleared after synchronization before each measurement; kernel warmup is retained.

`cacheBytes` sums persistent cache `innerState()` tensor sizes, including reserve,
positions and convolution state. `attentionBytes` excludes the convolution caches.
These describe tensor storage rather than process RSS; shared views may differ
from physical allocations. MLX active/peak allocator counters are reported
separately, including model weights. `prefillActiveBytes` records the allocation
after prefill; `activeBytes` records it after decode while explicitly keeping the
cache, final output and next-token array alive. The peak counter resets before
prefill and again before decode: `prefillPeakBytes` and `decodePeakBytes` report
those phases, and `peakBytes` is their maximum. `peakIncrementBytes` subtracts
the active allocation immediately before prefill from that combined peak.

This is a memory and behavior prototype, not a quality evaluation. Last-loop
history changes attention during cached calls, and chunk boundaries can change
outputs. Matching greedy tokens on these prompts does not establish equivalence
for arbitrary prompts or sampling. Rewind/speculative decoding, Small, images,
tensor parallelism, and concurrent multi-user throughput are outside this harness.
