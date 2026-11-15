// Copyright © 2026 Apple Inc.

// Decode throughput and time to first token on a local model, as medians over rounds: the
// single-sequence `TokenIterator`, and `BatchTokenIterator` at every batch size asked for.
// Decode is timed after the prefill and the first token; the time to first token covers the
// prefill. A batch's decode rate counts only the steps in which every row decodes and no prompt
// is prefilled; with resumable prefill the steps that also feed a prefill chunk are reported
// apart. Every path must repeat its tokens in every round; how far each batch row agrees
// with a single-sequence run of the same prompt is reported.
//
//   swift run -c release --package-path Benchmarks/DecodeBench decodebench <model directory> \
//       [decode tokens: 128] [rounds: 5] [prompt tokens: 145] [sampler: greedy|topp|categorical] \
//       [batch sizes: 1,4,8] [prompt lengths: equal|mixed|arrival[:tokens]] \
//       [prefill step: the model's default]
//
// With `mixed` prompt lengths, the rows of a batch get prompts that shrink evenly from the full
// length to a sixth of it, so every row but the first is left-padded. With the greedy sampler,
// a row whose positions or padding mask are wrong parts ways with its single-sequence run at
// once, while correct rows agree until the different matmul shapes of a batch round a near tie
// the other way.
//
// `arrival` measures a long prompt joining rows that are decoding. Each batch size is a number
// of active rows with prompts of the prompt length; after 16 steps in which every row decoded,
// a prompt of `tokens` (default 4096) arrives. It reports the active rows' gaps between tokens
// while the newcomer is admitted (from its insertion to its first token), the newcomer's time
// to first token, and how many tokens each active row emitted while it waited. The prefill step
// sets the balance: smaller chunks shorten the gaps but admit the newcomer more slowly. The
// balance depends on the GPU and the model, so measure it on the hardware that will run it.
//
// The prefill step is `PrefillParameters.stepSize` for both the single-sequence and the batch
// paths.
//
// Built with `-Xswiftc -DSINGLE_ONLY` it times `TokenIterator` alone, so it also builds
// against a checkout without `BatchTokenIterator`.

import BenchmarkHelpers
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

let usage =
    "usage: decodebench <model directory> [decode tokens] [rounds] [prompt tokens] [sampler] "
    + "[batch sizes] [prompt lengths: equal|mixed|arrival[:tokens]] [prefill step]"
let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    print(usage)
    exit(1)
}
let modelDirectory = URL(fileURLWithPath: arguments[1])
let decodeTokens = arguments.count > 2 ? Int(arguments[2]) ?? 128 : 128
let rounds = max(1, arguments.count > 3 ? Int(arguments[3]) ?? 5 : 5)
let promptLength = arguments.count > 4 ? Int(arguments[4]) ?? 145 : 145
let samplerName = arguments.count > 5 ? arguments[5] : "greedy"
let batchSizes = Set(
    (arguments.count > 6 ? arguments[6] : "1,4,8").split(separator: ",").compactMap { Int($0) }
        .filter { $0 > 0 }
).sorted()
let lengthsName = arguments.count > 7 ? arguments[7] : "equal"
/// The arriving prompt's length in `arrival` mode, else nil.
let arrivalLength: Int? =
    lengthsName == "arrival"
    ? 4096
    : lengthsName.hasPrefix("arrival:") ? Int(lengthsName.dropFirst("arrival:".count)) : nil
let prefillStep = arguments.count > 8 ? Int(arguments[8]) : nil
let prefill = PrefillParameters(stepSize: prefillStep)
guard !batchSizes.isEmpty,
    ["equal", "mixed"].contains(lengthsName) || (arrivalLength ?? 0) > 0,
    arguments.count <= 8 || (prefillStep ?? 0) > 0
else {
    print(usage)
    print(
        "batch sizes are a comma-separated list such as 1,4,8; prompt lengths equal, mixed, "
            + "arrival or arrival:<tokens>; the prefill step a positive number of tokens")
    exit(1)
}

let context = try await LLMModelFactory.shared.load(
    from: modelDirectory, using: NoOpTokenizerLoader())
let model = context.model
print("model: \(modelDirectory.lastPathComponent)")
#if SINGLE_ONLY
print(
    "prompt tokens: \(promptLength), decode tokens: \(decodeTokens), rounds: \(rounds), "
        + "sampler: \(samplerName), prefill step: \(prefillStep.map(String.init) ?? "default")")
#else
print(
    "prompt tokens: \(promptLength), decode tokens: \(decodeTokens), rounds: \(rounds), "
        + "sampler: \(samplerName), batch sizes: \(batchSizes), prompt lengths: \(lengthsName), "
        + "prefill step: \(prefillStep.map(String.init) ?? "default")")
#endif

/// A fresh sampler per run, seeded, so every run of a path draws the same tokens.
@MainActor func makeSampler() -> LogitSampler {
    switch samplerName {
    case "topp": return TopPSampler(temperature: 0.8, topP: 0.9, seed: 7)
    case "categorical": return CategoricalSampler(temperature: 0.8, seed: 7)
    case "greedy": return ArgMaxSampler()
    default:
        print("unknown sampler \(samplerName): greedy, topp or categorical")
        exit(1)
    }
}

/// A fixed pseudo-random prompt, so every round and every checkout decodes the same tokens.
@MainActor func prompt(seed: UInt64, length: Int = promptLength) -> [Int32] {
    var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return (0 ..< length).map { _ in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int32(1000 + (state >> 33) % 20000)
    }
}

func seconds(
    from start: ContinuousClock.Instant, to end: ContinuousClock.Instant = ContinuousClock.now
) -> Double {
    let elapsed = end - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

func seconds(since start: ContinuousClock.Instant) -> Double {
    seconds(from: start)
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

struct Run {
    /// Decode tokens per second, counting one row.
    let tokensPerSecond: Double
    /// Seconds from the start of the run to the first token: the prefill.
    let firstToken: Double
    /// Every row's tokens, the first one included.
    let tokens: [[Int]]
    /// Batch steps that decoded while also feeding a prefill chunk, and their total seconds.
    var overlapSteps = 0
    var overlapSeconds = 0.0
}

/// Prefills the prompt, then times `decodeTokens` steps of `TokenIterator`.
@MainActor func runSingle(prompt: [Int32]) throws -> Run {
    defer { Stream.defaultStream.synchronize() }
    let start = ContinuousClock.now
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt)), model: model, processor: nil,
        sampler: makeSampler(), prefill: prefill, maxTokens: decodeTokens + 1)
    var tokens: [Int] = []
    // The first token comes from the prefill; the steps after it are the decode.
    if let first = iterator.next() { tokens.append(first) }
    let firstToken = seconds(since: start)
    let decodeStart = ContinuousClock.now
    while let token = iterator.next() { tokens.append(token) }
    return Run(
        tokensPerSecond: Double(decodeTokens) / seconds(since: decodeStart),
        firstToken: firstToken, tokens: [tokens])
}

#if !SINGLE_ONLY
/// Queues the prompts in a `BatchTokenIterator` and runs it to the end. The time to first
/// token ends at the first `.token` event. The decode rate counts the steady steps only: every
/// row decodes, none of them its first token, and no prefill chunk runs. With resumable
/// prefill the rows join one step apart, so the steps that also feed a prefill chunk are
/// reported apart from it.
@MainActor func runBatched(prompts: [[Int32]]) throws -> Run {
    defer { Stream.defaultStream.synchronize() }
    let start = ContinuousClock.now
    var iterator = BatchTokenIterator(
        model: model, sampler: makeSampler(), maxTokens: decodeTokens + 1, prefill: prefill)
    let ids = iterator.insert(prompts: prompts.map { $0.map { Int($0) } })
    var tokens = [[Int]](repeating: [], count: prompts.count)
    var firstToken: Double?
    var steadySteps = 0
    var steadySeconds = 0.0
    var run = Run(tokensPerSecond: 0, firstToken: 0, tokens: [])
    while true {
        let stepStart = ContinuousClock.now
        guard let events = try iterator.next() else { break }
        let stepSeconds = seconds(since: stepStart)
        var decoded = 0
        var prefilled = false
        var firstTokens = false
        for event in events {
            switch event {
            case .token(let id, let token):
                let row = ids.firstIndex(of: id)!
                firstTokens = firstTokens || tokens[row].isEmpty
                tokens[row].append(token)
                decoded += 1
            case .prefill:
                prefilled = true
            // A row that cannot join ends the run, as an admission error did before the
            // iterator reported it as an event: a partial batch would report a rate for rows
            // it never ran.
            case .failed(_, let error):
                throw error
            case .finished:
                break
            }
        }
        if decoded > 0, firstToken == nil { firstToken = seconds(since: start) }
        if prefilled, decoded > 0 {
            run.overlapSteps += 1
            run.overlapSeconds += stepSeconds
        } else if !prefilled, !firstTokens, decoded == prompts.count {
            steadySteps += 1
            steadySeconds += stepSeconds
        }
    }
    // With resumable prefill the rows join one step apart, so a short run can end its first
    // rows before the last one decodes.
    guard steadySteps > 0 else {
        print(
            "batch(\(prompts.count)): no step decoded every row without a prefill chunk, so there "
                + "is no steady decode rate. Ask for more decode tokens than the rows and their "
                + "prefill chunks take to join.")
        exit(1)
    }
    return Run(
        tokensPerSecond: Double(steadySteps) / steadySeconds,
        firstToken: firstToken ?? seconds(since: start), tokens: tokens,
        overlapSteps: run.overlapSteps, overlapSeconds: run.overlapSeconds)
}

/// The prompts of a batch: row 0 is the single-sequence prompt and every other row has its
/// own. With mixed lengths the rows shrink evenly to a sixth of the full length.
@MainActor func prompts(rows: Int) -> [[Int32]] {
    (0 ..< rows).map { row in
        guard lengthsName == "mixed", rows > 1 else { return prompt(seed: UInt64(row + 1)) }
        let shortest = max(1, promptLength / 6)
        let length = promptLength - (promptLength - shortest) * row / (rows - 1)
        return prompt(seed: UInt64(row + 1), length: length)
    }
}

/// One `arrival` replay: what the active rows and the newcomer saw while it was admitted.
struct Arrival {
    /// Seconds between the active rows' consecutive tokens, from the newcomer's insertion to
    /// its first token. Every active row decodes in every step, so one gap per step.
    var gaps: [Double] = []
    /// Seconds from the newcomer's insertion to its first token.
    var firstToken: Double?
    /// Tokens each active row emitted in that time.
    var waitingTokens = 0
    /// Every row's tokens, the newcomer's last.
    var tokens: [[Int]]
}

/// Steps in which every active row decodes before the newcomer arrives.
let arrivalAfter = 16

/// `rows` rows decode prompts of the prompt length; after `arrivalAfter` steps a prompt of
/// `length` tokens arrives, and every row runs to its limit.
@MainActor func runArrival(rows: Int, length: Int) throws -> Arrival {
    defer { Stream.defaultStream.synchronize() }
    var iterator = BatchTokenIterator(
        model: model, sampler: makeSampler(), maxTokens: decodeTokens + 1, prefill: prefill)
    let active = iterator.insert(
        prompts: (0 ..< rows).map { prompt(seed: UInt64($0 + 1)).map { Int($0) } })
    var newcomer: Int?
    var inserted: ContinuousClock.Instant?
    var previous: ContinuousClock.Instant?
    var steps = 0
    var arrival = Arrival(tokens: [[Int]](repeating: [], count: rows + 1))
    while let events = try iterator.next() {
        let now = ContinuousClock.now
        var activeTokens = 0
        var newcomerToken = false
        for event in events {
            switch event {
            case .token(let id, let token):
                if id == newcomer {
                    arrival.tokens[rows].append(token)
                    newcomerToken = true
                } else {
                    arrival.tokens[active.firstIndex(of: id)!].append(token)
                    activeTokens += 1
                }
            case .failed(_, let error):
                throw error
            case .prefill, .finished:
                break
            }
        }
        let admitting = inserted != nil && arrival.firstToken == nil
        if admitting {
            // A gap counts only while every active row still decodes.
            guard activeTokens == rows, let previous else {
                print(
                    "arrival(\(rows)): an active row ended before the newcomer's first token. "
                        + "Ask for more decode tokens.")
                exit(1)
            }
            arrival.gaps.append(seconds(from: previous, to: now))
            arrival.waitingTokens += 1
            if newcomerToken { arrival.firstToken = seconds(since: inserted!) }
        }
        if activeTokens > 0 { previous = now }
        if newcomer == nil, activeTokens == rows {
            steps += 1
            if steps == arrivalAfter {
                newcomer = iterator.insert(
                    prompt: prompt(seed: 100, length: length).map { Int($0) })
                inserted = ContinuousClock.now
            }
        }
    }
    guard arrival.firstToken != nil else {
        print("arrival(\(rows)): the newcomer never arrived. Ask for more decode tokens.")
        exit(1)
    }
    return arrival
}

/// The `fraction` percentile of `values`, interpolating linearly between ranks.
func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted()
    let rank = fraction * Double(sorted.count - 1)
    let lower = Int(rank.rounded(.down))
    let upper = min(lower + 1, sorted.count - 1)
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - Double(lower))
}

if let arrivalLength {
    // Warm up every width once so kernel compilation is not timed.
    for rows in batchSizes { _ = try runArrival(rows: rows, length: arrivalLength) }
    var replays: [Int: [Arrival]] = [:]
    var repeated = true
    for round in 0 ..< rounds {
        // Alternate the order so a thermal or background-load slope lands on every width.
        for rows in round % 2 == 0 ? batchSizes : Array(batchSizes.reversed()) {
            let replay = try runArrival(rows: rows, length: arrivalLength)
            repeated = repeated && replay.tokens == (replays[rows]?.first?.tokens ?? replay.tokens)
            replays[rows, default: []].append(replay)
            print(
                String(
                    format: "round %d  arrival(%d): newcomer ttft %.0f ms, active p95 gap %.1f ms, "
                        + "%d tokens per active row while it waited", round + 1, rows,
                    1000 * replay.firstToken!, 1000 * percentile(replay.gaps, 0.95),
                    replay.waitingTokens))
            Memory.clearCache()
        }
    }
    for rows in batchSizes {
        let runs = replays[rows]!
        let gaps = runs.flatMap(\.gaps)
        print(
            String(
                format: "median  arrival(%d): newcomer ttft %.0f ms; active rows' gaps while it "
                    + "is admitted p50 %.1f ms, p95 %.1f ms, max %.1f ms (%d gaps); %.0f tokens "
                    + "per active row while it waited", rows,
                1000 * median(runs.map { $0.firstToken! }), 1000 * percentile(gaps, 0.5),
                1000 * percentile(gaps, 0.95), 1000 * gaps.max()!, gaps.count,
                median(runs.map { Double($0.waitingTokens) })))
    }
    print("each width repeats its tokens in every round: \(repeated)")
    exit(0)
}
#endif

let singlePrompt = prompt(seed: 1)

// Warm up every path once so kernel compilation is not timed.
_ = try runSingle(prompt: singlePrompt)
#if !SINGLE_ONLY
let batches = batchSizes.map { (rows: $0, prompts: prompts(rows: $0)) }
for batch in batches {
    _ = try runBatched(prompts: batch.prompts)
    if lengthsName == "mixed", batch.rows > 1 {
        print("batch(\(batch.rows)) prompt lengths: \(batch.prompts.map(\.count))")
    }
}

/// Every row's prompt run alone, untimed, to compare the batch rows with.
var singleTokens: [[Int32]: [Int]] = [:]
for batch in batches {
    for prompt in batch.prompts where singleTokens[prompt] == nil {
        singleTokens[prompt] = try runSingle(prompt: prompt).tokens[0]
    }
}
#endif

/// The tokens of a path in its first round, to check every later round against.
var references: [String: [[Int]]] = [:]
var mismatchedRounds: [Int] = []
@MainActor func check(_ path: String, _ run: Run, round: Int) {
    if let reference = references[path] {
        if run.tokens != reference, !mismatchedRounds.contains(round) {
            mismatchedRounds.append(round)
        }
    } else {
        references[path] = run.tokens
    }
}
@MainActor func format(_ run: Run, rows: Int) -> String {
    let perRow = String(format: "%.2f tok/s", run.tokensPerSecond)
    let rate =
        rows == 1
        ? perRow
        : perRow + String(format: " per row, %.2f total", Double(rows) * run.tokensPerSecond)
    let overlap =
        run.overlapSteps == 0
        ? ""
        : String(
            format: ", %d overlap steps at %.1f ms", run.overlapSteps,
            1000 * run.overlapSeconds / Double(run.overlapSteps))
    return rate + String(format: " (ttft %.0f ms", 1000 * run.firstToken) + overlap + ")"
}

var single: [Double] = []
var singleFirst: [Double] = []
#if !SINGLE_ONLY
var batchRates: [Int: [Double]] = [:]
var batchFirst: [Int: [Double]] = [:]
/// Per batch size, the fewest leading tokens each row shares with its single-sequence run.
var rowAgreement: [Int: [Int]] = [:]
#endif
for round in 0 ..< rounds {
    #if SINGLE_ONLY
    let plain = try runSingle(prompt: singlePrompt)
    single.append(plain.tokensPerSecond)
    singleFirst.append(plain.firstToken)
    check("single", plain, round: round + 1)
    print("round \(round + 1)  single \(format(plain, rows: 1))")
    #else
    // Alternate the order so a thermal or background-load slope lands on every path.
    let batchedFirst = round % 2 == 1
    var runs: [Run] = []
    if batchedFirst { runs = try batches.map { try runBatched(prompts: $0.prompts) } }
    let plain = try runSingle(prompt: singlePrompt)
    if !batchedFirst { runs = try batches.map { try runBatched(prompts: $0.prompts) } }

    single.append(plain.tokensPerSecond)
    singleFirst.append(plain.firstToken)
    check("single", plain, round: round + 1)
    var line = "round \(round + 1)  single \(format(plain, rows: 1))"
    for (batch, run) in zip(batches, runs) {
        batchRates[batch.rows, default: []].append(run.tokensPerSecond)
        batchFirst[batch.rows, default: []].append(run.firstToken)
        // Each path must repeat itself across rounds exactly. Against a row's single run the
        // arithmetic can differ (a batch runs other matmul shapes, and a sampler draws a batch
        // from one random stream), so their streams can part ways; how far they agree is
        // reported, not asserted.
        check("batch(\(batch.rows))", run, round: round + 1)
        let agreement = zip(batch.prompts, run.tokens).map { prompt, tokens in
            zip(singleTokens[prompt]!, tokens).prefix { $0 == $1 }.count
        }
        rowAgreement[batch.rows] =
            rowAgreement[batch.rows].map { zip($0, agreement).map { min($0, $1) } } ?? agreement
        line += "   batch(\(batch.rows)) \(format(run, rows: batch.rows))"
    }
    print(line + "   (\(batchedFirst ? "batched" : "single") first)")
    #endif
    Memory.clearCache()
}
print(
    "median  single "
        + format(
            Run(tokensPerSecond: median(single), firstToken: median(singleFirst), tokens: []),
            rows: 1))
#if !SINGLE_ONLY
for batch in batches {
    let run = Run(
        tokensPerSecond: median(batchRates[batch.rows]!),
        firstToken: median(batchFirst[batch.rows]!), tokens: [])
    print("median  batch(\(batch.rows)) " + format(run, rows: batch.rows))
}
for batch in batches {
    let agreement = rowAgreement[batch.rows]!
    print(
        "batch(\(batch.rows)) rows agree with their single runs on the first "
            + "\(agreement.min()!) of \(decodeTokens + 1) tokens"
            + (batch.rows == 1 ? "" : "; per row \(agreement)"))
}
if samplerName != "greedy" {
    print("(a batch draws every row from one random stream, so rows can part ways at once)")
}
#endif
print(
    "each path repeats its tokens in every round: \(mismatchedRounds.isEmpty)"
        + (mismatchedRounds.isEmpty ? "" : ", not in rounds \(mismatchedRounds)"))
