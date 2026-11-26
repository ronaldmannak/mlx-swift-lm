import Foundation
import MLX
import MLXLLM
import MLXLMCommon

struct StubLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { StubTokenizer() }
}
struct StubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
func elapsed(_ start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}
func emit(_ data: [String: Any]) {
    let raw = try! JSONSerialization.data(withJSONObject: data, options: [.sortedKeys])
    print(String(decoding: raw, as: UTF8.self))
    fflush(stdout)
}
let arguments = CommandLine.arguments
let directory = URL(filePath: arguments[1])
let lengths = arguments.count > 2 ? arguments[2].split(separator: ",").map { Int($0)! } : [2048]
let count = arguments.count > 3 ? Int(arguments[3])! : 128
let chunk = arguments.count > 4 ? Int(arguments[4])! : 512
let fixturePath = ProcessInfo.processInfo.environment["INKLING_CHAT_FIXTURES"]
let fixtures: [[String: Any]] =
    try fixturePath.map {
        try JSONSerialization.jsonObject(with: Data(contentsOf: URL(filePath: $0)))
            as! [[String: Any]]
    } ?? []
let loadStart = ContinuousClock.now
let context = try await LLMModelFactory.shared.load(from: directory, using: StubLoader())
Stream.defaultStream(.gpu).synchronize()
emit([
    "kind": "load", "mode": directory.lastPathComponent, "seconds": elapsed(loadStart),
    "activeBytes": Memory.activeMemory,
])

@MainActor func run(length: Int, count: Int, warmup: Bool) throws {
    var random: UInt64 = 7
    var tokens: [Int32] = (0 ..< length).map { _ in
        random = random &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int32(1000 + (random >> 33) % 20000)
    }
    let fixture = fixtures.first { ($0["tokens"] as! [Int]).count == length }
    if let fixture { tokens = (fixture["tokens"] as! [Int]).map(Int32.init) }
    let cache = try context.model.newCache(parameters: nil)
    Stream.defaultStream(.gpu).synchronize()
    Memory.clearCache()
    Memory.peakMemory = 0
    let baseActiveBytes = Memory.activeMemory
    let start = ContinuousClock.now
    let input = LMInput(tokens: MLXArray(tokens))
    var output: LMOutput
    switch try context.model.prepare(
        input, cache: cache, state: nil, prefill: .init(stepSize: chunk))
    {
    case .logits(let result): output = result
    case .tokens(let remaining):
        output = context.model(remaining[text: .newAxis], cache: cache, state: nil)
    }
    var next = argMax(output.logits[0, -1], axis: -1)
    eval(next, cache)
    Stream.defaultStream(.gpu).synchronize()
    let prefillSeconds = elapsed(start)
    let prefillCacheBytes = cache.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    let prefillAttentionBytes = cache.filter { $0.currentKeyValues() != nil }
        .flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    let prefillMemory = Memory.snapshot()
    Memory.peakMemory = 0
    var generated = [next.item(Int.self)]
    let decodeStart = ContinuousClock.now
    for _ in 0 ..< count {
        let text = LMInput.Text(tokens: next.asType(.int32).reshaped(1, 1))
        output = context.model(text, cache: cache, state: output.state)
        next = argMax(output.logits[0, -1], axis: -1)
        eval(next, cache)
        generated.append(next.item(Int.self))
    }
    Stream.defaultStream(.gpu).synchronize()
    let decodeSeconds = elapsed(decodeStart)
    let cacheBytes = cache.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    let attentionBytes = cache.filter { $0.currentKeyValues() != nil }
        .flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    let decodeMemory = withExtendedLifetime((cache, output, next)) { Memory.snapshot() }
    let peakBytes = max(prefillMemory.peakMemory, decodeMemory.peakMemory)
    emit([
        "kind": warmup ? "warmup" : "run", "mode": directory.lastPathComponent,
        "promptName": fixture?["name"] as? String ?? "synthetic",
        "promptTokens": length, "decodeTokens": count, "prefillChunk": chunk,
        "prefillSeconds": prefillSeconds, "decodeSeconds": decodeSeconds,
        "decodeTokensPerSecond": Double(count) / decodeSeconds,
        "cacheEntries": cache.count, "prefillCacheBytes": prefillCacheBytes,
        "prefillAttentionBytes": prefillAttentionBytes, "cacheBytes": cacheBytes,
        "attentionBytes": attentionBytes, "activeBytes": decodeMemory.activeMemory,
        "prefillActiveBytes": prefillMemory.activeMemory,
        "prefillPeakBytes": prefillMemory.peakMemory, "decodePeakBytes": decodeMemory.peakMemory,
        "baseActiveBytes": baseActiveBytes, "peakBytes": peakBytes,
        "peakIncrementBytes": peakBytes - baseActiveBytes, "tokens": generated,
    ])
}
for length in lengths {
    try run(length: length, count: 8, warmup: true)
    try run(length: length, count: count, warmup: false)
}
