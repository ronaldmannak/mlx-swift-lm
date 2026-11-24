// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private func separateMessages(_ input: Inkling2MessageInput, bank: Inkling2MessageBank)
    -> MLXArray
{
    stacked(
        (0 ..< input.query.dim(0)).map { input(bank, loopedLayer: $0) }, axis: 2)
}

private func readoutDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

private func readoutInput(dimensions: Int, layers: Int) -> Inkling2MessageInput {
    let input = Inkling2MessageInput(dimensions: dimensions, loopedLayers: layers)
    input.update(
        parameters: ModuleParameters.unflattened([
            "query": MLXRandom.normal([layers, 4, dimensions]) * 0.1
        ]))
    return input
}

@Suite(.serialized)
struct Inkling2MessageReadoutTests {
    @Test(
        "Grouped message readouts retain each four-query kernel's bits",
        arguments: [
            (1, 1, 1, 12, 2304), (2, 1, 3, 3, 256), (1, 5, 2, 2, 320),
            (2, 3, 0, 2, 128),
        ])
    func groupedReadout(batch: Int, length: Int, blocks: Int, layers: Int, dimensions: Int) throws {
        MLXRandom.seed(61)
        let input = readoutInput(dimensions: dimensions, layers: layers)
        input.train(false)
        let blockStates =
            blocks > 0
            ? inkling2UnitRMS(MLXRandom.normal([blocks, batch, length, dimensions])) : nil
        let bank = Inkling2MessageBank(
            blocks: blockStates,
            partial: inkling2UnitRMS(MLXRandom.normal([batch, length, dimensions])))
        let grouped = input.allMessages(bank)
        let separate = separateMessages(input, bank: bank)
        #expect(grouped.shape == [batch, length, layers, 4, dimensions])
        #expect(grouped.dtype == .float32)
        #expect(
            arrayEqual(grouped.view(dtype: .uint32), separate.view(dtype: .uint32)).item(Bool.self))
        for layer in 0 ..< layers {
            let view = grouped[0..., 0..., layer ..< layer + 1, 0..., 0...].squeezed(axis: 2)
            #expect(readoutDifference(view, input(bank, loopedLayer: layer)) == 0)
        }
    }

    @Test("Empty blocks retain the partial-only readout")
    func emptyBlocks() throws {
        MLXRandom.seed(67)
        let (batch, length, layers, dimensions) = (2, 3, 3, 128)
        let input = readoutInput(dimensions: dimensions, layers: layers)
        input.train(false)
        let bank = Inkling2MessageBank(
            blocks: MLXArray.zeros([0, batch, length, dimensions]),
            partial: MLXRandom.normal([batch, length, dimensions]))
        #expect(
            readoutDifference(input.allMessages(bank), separateMessages(input, bank: bank)) == 0)
    }

    @Test("Training readouts and query gradients match the per-layer reference")
    func trainingGradients() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(71)
            let (batch, length, layers, dimensions) = (2, 3, 3, 128)
            let input = readoutInput(dimensions: dimensions, layers: layers)
            input.train()
            let partial = inkling2UnitRMS(MLXRandom.normal([batch, length, dimensions]))
            let blocks = inkling2UnitRMS(MLXRandom.normal([2, batch, length, dimensions]))
            let groupedLoss = valueAndGrad(model: input) { input, partial, blocks in
                input.allMessages(Inkling2MessageBank(blocks: blocks, partial: partial))
                    .square().mean()
            }
            let separateLoss = valueAndGrad(model: input) { input, partial, blocks in
                separateMessages(input, bank: Inkling2MessageBank(blocks: blocks, partial: partial))
                    .square().mean()
            }
            let (loss, gradients) = try withError {
                let (loss, gradients) = groupedLoss(input, partial, blocks)
                eval([loss] + gradients.flattened().map(\.1))
                return (loss, gradients)
            }
            let (expectedLoss, expectedGradients) = separateLoss(input, partial, blocks)
            let actual = try #require(gradients.flattened().first { $0.0 == "query" }?.1)
            let expected = try #require(expectedGradients.flattened().first { $0.0 == "query" }?.1)
            #expect(abs(loss - expectedLoss).item(Float.self) < 1e-6)
            #expect(readoutDifference(actual, expected) < 1e-5)
            #expect(all(isFinite(actual)).item(Bool.self))
            #expect(abs(actual).max().item(Float.self) > 0)
        }
    }

    @Test("Grouped readouts use updated bfloat16 query parameters")
    func queryUpdates() throws {
        MLXRandom.seed(73)
        let (layers, dimensions) = (3, 256)
        let input = readoutInput(dimensions: dimensions, layers: layers)
        input.train(false)
        let bank = Inkling2MessageBank(
            blocks: inkling2UnitRMS(MLXRandom.normal([2, 1, 1, dimensions])),
            partial: inkling2UnitRMS(MLXRandom.normal([1, 1, dimensions])))
        let first = input.allMessages(bank)
        eval(first)
        input.update(
            parameters: ModuleParameters.unflattened([
                "query": (MLXRandom.normal([layers, 4, dimensions]) * 0.3).asType(.bfloat16)
            ]))
        let updated = input.allMessages(bank)
        let expected = separateMessages(input, bank: bank)
        #expect(input.query.dtype == .bfloat16)
        #expect(readoutDifference(first, updated) > 1e-3)
        #expect(
            arrayEqual(updated.view(dtype: .uint32), expected.view(dtype: .uint32)).item(Bool.self))
    }
}
