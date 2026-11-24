import BenchmarkHelpers
import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

struct BenchError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Fixture: Decodable {
    let name: String
    let tokens: [Int]
}

struct RowRecord: Codable {
    let id: Int
    let promptName: String
    let promptTokenCount: Int
    let seed: UInt64
    var iteratorID: Int?
    var admissionMS: Double?
    var firstTokenMS: Double?
    var finishedMS: Double?
    var removedMS: Double?
    var endReason: String?
    var tokens = [Int]()
    var deliveryMS = [Double]()
}

struct MembershipEvent: Codable {
    let action: String
    let id: Int
    let atMS: Double
    let triggerRowTokens: Int
}

struct RowCounters: Codable {
    let id: Int
    let generated: Int
    let accepted: Int
    let proposed: Int
}

struct RunRecord: Codable {
    let kind: String
    let round: Int
    let arm: String
    let scenario: String
    let blockSize: Int
    let outputLimit: Int
    let temperature: Float
    let topP: Float
    let prefillStep: Int
    var rows: [RowRecord]
    var membership = [MembershipEvent]()
    var iteratorCalls = 0
    var targetVerifyCalls: Int?
    var counters = [RowCounters]()
    var elapsedMS = 0.0
    var aggregateTokensPerSecond = 0.0
    var peakGB = 0.0
}

struct Chunk {
    let id: Int
    let tokens: [Int]
    let finished: Bool
    let reason: String?
}

let outputLimit = 64
let draftBlockSize = 4
let prefillStep = 512
let firstSeed: UInt64 = 37

func milliseconds(_ start: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - start
    return Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
}

func input(_ tokens: [Int]) -> LMInput {
    LMInput(tokens: MLXArray(tokens.map { Int32($0) }))
}

func parameters(row: Int = 0) -> GenerateParameters {
    .init(
        maxTokens: outputLimit, temperature: 0.8, topP: 0.9,
        prefill: .init(stepSize: prefillStep), seed: firstSeed + UInt64(row))
}

@MainActor protocol Driver: AnyObject {
    var initialIDs: [Int] { get }
    var targetVerifyCalls: Int? { get }
    func counters() -> [RowCounters]
    func next() throws -> [Chunk]?
    func insert(_ row: Int) throws -> Int
    func remove(_ ids: [Int])
    func synchronize()
}

@MainActor final class OrdinaryDriver: Driver {
    var iterator: BatchTokenIterator
    let model: any LanguageModel
    let fixtures: [Fixture]
    let initialIDs: [Int]
    var targetVerifyCalls: Int? { nil }

    init(model: any LanguageModel, fixtures: [Fixture], initialRows: [Int]) throws {
        self.model = model
        self.fixtures = fixtures
        var created = try BatchTokenIterator(model: model, parameters: parameters(), stopTokens: [])
        initialIDs = created.insert(
            prompts: initialRows.map { fixtures[$0].tokens },
            samplers: initialRows.map { parameters(row: $0).sampler() },
            processors: initialRows.map {
                GenerationComponents().logitProcessor(parameters: parameters(row: $0), model: model)
            })
        iterator = created
    }

    func next() throws -> [Chunk]? {
        guard let events = try iterator.next() else { return nil }
        return try events.compactMap { event in
            switch event {
            case .token(let id, let token):
                return Chunk(id: id, tokens: [token], finished: false, reason: nil)
            case .finished(let id, let reason, _):
                return Chunk(id: id, tokens: [], finished: true, reason: String(describing: reason))
            case .failed(_, let error): throw error
            case .prefill: return nil
            }
        }
    }

    func insert(_ row: Int) throws -> Int {
        iterator.insert(
            prompt: fixtures[row].tokens, sampler: parameters(row: row).sampler(),
            processor: GenerationComponents().logitProcessor(
                parameters: parameters(row: row), model: model))
    }
    func remove(_ ids: [Int]) { iterator.remove(ids) }
    func synchronize() { iterator.synchronizeGeneration() }
    func counters() -> [RowCounters] { [] }
}

@MainActor final class DFlashDriver: Driver {
    let iterator: BatchedDFlashTokenIterator
    let fixtures: [Fixture]
    let initialIDs: [Int]
    var targetVerifyCalls: Int? { iterator.roundCount }

    init(
        model: any LanguageModel, draft: any BatchedDFlashDrafterModel,
        fixtures: [Fixture], initialRows: [Int]
    ) throws {
        self.fixtures = fixtures
        let created = try BatchedDFlashTokenIterator(
            inputs: initialRows.map { input(fixtures[$0].tokens) }, mainModel: model,
            drafter: draft, parameters: parameters(), blockSize: draftBlockSize, eosTokenIds: [])
        iterator = created
        initialIDs = created.ids
    }

    func next() throws -> [Chunk]? {
        iterator.next()?.map {
            Chunk(
                id: $0.id, tokens: $0.tokens, finished: $0.finished,
                reason: $0.finished ? "length" : nil)
        }
    }
    func insert(_ row: Int) throws -> Int {
        let ids = try iterator.insert(
            inputs: [input(fixtures[row].tokens)], parameters: parameters())
        guard let id = ids.first, ids.count == 1 else { throw BenchError("Missing inserted row") }
        return id
    }
    func remove(_ ids: [Int]) { iterator.remove(ids) }
    func synchronize() { iterator.synchronize() }
    func counters() -> [RowCounters] {
        iterator.ids.enumerated().map { index, id in
            RowCounters(
                id: id, generated: iterator.generatedCountByRow[index],
                accepted: iterator.acceptedCountByRow[index],
                proposed: iterator.proposedCountByRow[index])
        }
    }
}

@MainActor func run(
    arm: String, scenario: String, round: Int, warmup: Bool,
    model: any LanguageModel, draft: any BatchedDFlashDrafterModel, fixtures: [Fixture]
) throws -> RunRecord {
    Stream.defaultStream(.gpu).synchronize()
    Memory.peakMemory = 0
    var record = RunRecord(
        kind: warmup ? "warmup" : "run", round: round, arm: arm, scenario: scenario,
        blockSize: arm == "dflash" ? draftBlockSize : 1, outputLimit: outputLimit,
        temperature: 0.8, topP: 0.9, prefillStep: prefillStep,
        rows: fixtures.enumerated().map { index, fixture in
            RowRecord(
                id: index, promptName: fixture.name, promptTokenCount: fixture.tokens.count,
                seed: firstSeed + UInt64(index))
        })
    let initialRows = scenario == "fixed" ? Array(fixtures.indices) : [0, 1]
    let start = ContinuousClock.now
    let driver: any Driver
    if arm == "ordinary" {
        driver = try OrdinaryDriver(model: model, fixtures: fixtures, initialRows: initialRows)
    } else {
        driver = try DFlashDriver(
            model: model, draft: draft, fixtures: fixtures, initialRows: initialRows)
    }
    var logicalByIterator = [Int: Int]()
    var latestCounters = [Int: RowCounters]()
    for (logical, iteratorID) in zip(initialRows, driver.initialIDs) {
        guard logical == iteratorID else { throw BenchError("Unexpected initial stable row ID") }
        logicalByIterator[iteratorID] = logical
        record.rows[logical].iteratorID = iteratorID
        record.rows[logical].admissionMS = 0
        record.membership.append(.init(action: "insert", id: logical, atMS: 0, triggerRowTokens: 0))
    }
    func captureCounters() {
        for value in driver.counters() { latestCounters[value.id] = value }
    }
    func insert(_ logical: Int) throws {
        let at = milliseconds(start)
        let id = try driver.insert(logical)
        guard id == logical else { throw BenchError("Stable row ID changed the row seed") }
        logicalByIterator[id] = logical
        record.rows[logical].iteratorID = id
        record.rows[logical].admissionMS = at
        record.membership.append(
            .init(
                action: "insert", id: logical, atMS: at,
                triggerRowTokens: record.rows[0].tokens.count))
    }
    while true {
        record.iteratorCalls += 1
        guard let chunks = try driver.next() else { break }
        let at = milliseconds(start)
        var retired = Set<Int>()
        for chunk in chunks {
            guard let logical = logicalByIterator[chunk.id], record.rows[logical].removedMS == nil
            else {
                throw BenchError("A removed or unknown row emitted a chunk")
            }
            for token in chunk.tokens {
                guard record.rows[logical].tokens.count < outputLimit else {
                    throw BenchError("The iterator exceeded a row's output limit")
                }
                if record.rows[logical].firstTokenMS == nil {
                    record.rows[logical].firstTokenMS = at
                }
                record.rows[logical].tokens.append(token)
                record.rows[logical].deliveryMS.append(at)
            }
            if chunk.finished {
                record.rows[logical].finishedMS = at
                record.rows[logical].endReason = chunk.reason
                retired.insert(chunk.id)
            }
        }
        captureCounters()
        if !retired.isEmpty {
            // Materialize the final state before removing its last owning row.
            driver.synchronize()
            driver.remove(retired.sorted())
        }
        if scenario == "staggered" {
            if record.rows[2].admissionMS == nil, record.rows[0].tokens.count >= 8 { try insert(2) }
            if record.rows[3].admissionMS == nil, record.rows[0].tokens.count >= 16 {
                guard record.rows[1].finishedMS == nil, let id = record.rows[1].iteratorID else {
                    throw BenchError("The cancellation schedule missed its active row")
                }
                captureCounters()
                let at = milliseconds(start)
                driver.remove([id])
                record.rows[1].removedMS = at
                record.rows[1].endReason = "cancelled"
                record.membership.append(
                    .init(
                        action: "remove", id: 1, atMS: at,
                        triggerRowTokens: record.rows[0].tokens.count))
                try insert(3)
            }
        }
    }
    captureCounters()
    driver.synchronize()
    Stream.defaultStream(.gpu).synchronize()
    record.elapsedMS = milliseconds(start)
    record.targetVerifyCalls = driver.targetVerifyCalls
    record.counters = latestCounters.values.sorted { $0.id < $1.id }
    record.peakGB = Double(Memory.peakMemory) / 1e9
    for row in record.rows {
        guard row.admissionMS != nil else { throw BenchError("Not every planned row was admitted") }
        if scenario == "staggered", row.id == 1 {
            guard row.removedMS != nil, !row.tokens.isEmpty, row.tokens.count < outputLimit else {
                throw BenchError("The removed row did not retain a partial output")
            }
        } else {
            guard row.tokens.count == outputLimit, row.finishedMS != nil else {
                throw BenchError("A surviving row did not finish exactly 64 tokens")
            }
        }
    }
    record.aggregateTokensPerSecond =
        Double(record.rows.reduce(0) { $0 + $1.tokens.count }) * 1000 / record.elapsedMS
    return record
}

func emit<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    print(String(decoding: try encoder.encode(value), as: UTF8.self))
    fflush(stdout)
}

let args = CommandLine.arguments
guard (4 ... 5).contains(args.count) else {
    throw BenchError("dflashbatch TARGET_DIR DRAFT_DIR PROMPTS_JSON [fixed|staggered|both]")
}
let scenario = args.count == 5 ? args[4] : "fixed"
guard ["fixed", "staggered", "both"].contains(scenario) else {
    throw BenchError("Unknown scenario")
}
let fixtureData = try Data(contentsOf: URL(filePath: args[3]))
let fixtures = try JSONDecoder().decode([Fixture].self, from: fixtureData)
guard fixtures.count == 4,
    fixtures.allSatisfy({
        !$0.tokens.isEmpty && $0.tokens.allSatisfy { $0 >= 0 && Int32(exactly: $0) != nil }
    })
else { throw BenchError("Expected four nonempty Int32 token fixtures") }
let model = try await LLMModelFactory.shared.load(
    from: URL(filePath: args[1]), using: NoOpTokenizerLoader()
).model
await DFlash2Registration.register()
let loadedDraft = try await DFlash2DrafterModelFactory.shared.load(
    from: URL(filePath: args[2]), using: NoOpTokenizerLoader()
).model
guard let draft = loadedDraft as? any BatchedDFlashDrafterModel else {
    throw BenchError("The loaded drafter does not implement batched DFlash")
}
Stream.defaultStream(.gpu).synchronize()
struct Header: Encodable {
    let kind = "environment"
    let target: String
    let drafter: String
    let fixtureSHA256: String
    let promptLengths: [Int]
    let rowSeeds: [UInt64]
    let warmupsPerArm = 2
    let measuredRounds = 3
    let modelLoadingTimed = false
    let residentDrafterInBothArms = true
}
try emit(
    Header(
        target: args[1], drafter: args[2],
        fixtureSHA256: SHA256.hash(data: fixtureData).map { String(format: "%02x", $0) }.joined(),
        promptLengths: fixtures.map { $0.tokens.count },
        rowSeeds: (0 ..< 4).map { firstSeed + UInt64($0) }))
var repeated = [String: [[Int]]]()
for workload in scenario == "both" ? ["fixed", "staggered"] : [scenario] {
    for iteration in 0 ..< 5 {
        let warmup = iteration < 2
        let order = iteration.isMultiple(of: 2) ? ["ordinary", "dflash"] : ["dflash", "ordinary"]
        for arm in order {
            let result = try run(
                arm: arm, scenario: workload, round: iteration - 2,
                warmup: warmup, model: model, draft: draft, fixtures: fixtures)
            if !warmup {
                let key = workload + ":" + arm
                let tokens = result.rows.map { $0.tokens }
                if let previous = repeated[key], tokens != previous {
                    throw BenchError("Measured repetitions changed the same arm's token stream")
                }
                repeated[key] = tokens
            }
            try emit(result)
        }
    }
}
