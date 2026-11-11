# DecodeBench

Decode throughput and time to first token on a local model. It compares the single-sequence `TokenIterator` with `BatchTokenIterator` at each batch size asked for, and reports medians over rounds. It loads a model directory you already have and downloads nothing.

```
swift run -c release --package-path Benchmarks/DecodeBench decodebench <model directory> \
    [decode tokens: 128] [rounds: 5] [prompt tokens: 145] [sampler: greedy|topp|categorical] \
    [batch sizes: 1,4,8] [prompt lengths: equal|mixed|arrival[:tokens]] \
    [prefill step: the model's default]
```

- **Decode rate:** counts only the batch steps in which every row decodes and no prompt is prefilled. With resumable prefill, the steps that also feed a prefill chunk are reported as overlap steps.
- **Time to first token:** covers the prefill.
- **Prompt lengths:**
  - `equal` gives every row a prompt of the prompt length.
  - `mixed` shrinks the rows' prompts evenly from the full length to a sixth of it, so every row but the first is left-padded.
- **Prefill step:** sets `PrefillParameters.stepSize` for both paths.
- **Checks:** every path must repeat its tokens in every round. The output also reports how far each batch row agrees with a single-sequence run of its prompt.

## Choosing a prefill step: `arrival`

When a long prompt joins rows that are decoding, `BatchTokenIterator` prefills it one chunk per step, after each decode step, for models that support resumable prefill. The prefill step sets the balance between the two:
- Smaller chunks give the decoding rows shorter gaps between tokens.
- But they admit the new prompt more slowly, mainly because small chunks use the GPU less efficiently.

How it works:
- Each batch size is a number of active rows, with prompts of the prompt length.
- After 16 steps in which every row decoded, a prompt of `tokens` arrives (default 4096).
- The rounds replay this, and the order of the widths alternates from round to round.

```
decodebench <model directory> 128 5 145 greedy 1,3 arrival:4096 512
decodebench <model directory> 128 5 145 greedy 1,3 arrival:4096 256
```

It reports, per width:
- the newcomer's time to first token, from its insertion;
- the active rows' gaps between tokens while the newcomer is admitted (p50, p95, max), from its insertion to its first token;
- how many tokens each active row emitted while it waited.

Compare step sizes in separate runs, and alternate their order (for example 512, 256, 256, 512) so a thermal slope lands on both.

The results depend on the GPU and the model. The library's default of 512 was chosen on an Apple M5 Max, so measure on the hardware that will run the model before changing it.

Built with `-Xswiftc -DSINGLE_ONLY`, DecodeBench times `TokenIterator` alone, so it also builds against a checkout without `BatchTokenIterator`. That build has no `arrival` mode.
