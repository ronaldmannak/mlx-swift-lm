// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXImageGeneration
import MLXNN
import Testing

@Suite struct QwenImageTransformerTests {

    /// Two layers of two 16-dimensional heads, small enough to run in a unit test.
    private static let tinyConfiguration = """
        {
          "in_channels": 32,
          "out_channels": 32,
          "num_layers": 2,
          "num_attention_heads": 2,
          "attention_head_dim": 16,
          "context_in_dim": 32,
          "mlp_ratio": 3,
          "axes_dims_rope": [4, 6, 6],
          "eps": 1e-06,
          "causal_condition": true,
          "mlx_format": true
        }
        """

    private func makeTransformer(_ json: String = tinyConfiguration) throws -> QwenImageTransformer
    {
        try QwenImageTransformer(
            JSONDecoder().decode(QwenImageTransformerConfiguration.self, from: Data(json.utf8)))
    }

    // The reference values below come from mlx-vlm's `QwenImageTransformer` with these same
    // weights: each tensor is a sine wave over its flat index, with a phase taken from its
    // name. That way the test needs no weight file.
    private func wave(_ count: Int, phase: Double, scale: Double, offset: Double = 0) -> [Float] {
        (0 ..< count).map { Float(offset + scale * sin(0.37 * Double($0) + phase)) }
    }

    private func checkpoint(for transformer: QwenImageTransformer) -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for (key, value) in transformer.parameters().flattened() {
            let phase = Double(key.unicodeScalars.reduce(0) { $0 + Int($1.value) } % 1000) * 0.001
            let values: [Float]
            if value.ndim == 1 {
                // Norm gains. The text norm stores its gain minus one.
                let offset: Double = key.hasSuffix("text_norm.weight") ? 0 : 1
                values = wave(value.size, phase: phase, scale: 0.2, offset: offset)
            } else {
                values = wave(
                    value.size, phase: phase, scale: 1 / Double(value.dim(1)).squareRoot())
            }
            weights[key] = MLXArray(values, value.shape)
        }
        return weights
    }

    /// A 2 x 3 latent image and five text tokens.
    private func velocity(_ transformer: QwenImageTransformer) -> MLXArray {
        transformer(
            MLXArray(wave(6 * 32, phase: 0.5, scale: 1), [1, 6, 32]),
            textStates: MLXArray(wave(5 * 32, phase: 1.5, scale: 1), [1, 5, 32]),
            noiseLevel: MLXArray([0.7] as [Float]), height: 2, width: 3)
    }

    private func expectReferenceVelocity(_ velocity: MLXArray) {
        let expected: [Float] = [
            -0.142881, -0.294974, -0.071666, 0.242848, 0.248299, -0.031013,
            -0.190107, -0.107259, 0.112093, 0.188789, 0.513710, 0.074362,
            -0.459624, -0.408665, -0.602832, -0.101458, 0.529038, 0.486249,
            -0.175369, 0.059876, -0.341915, -0.308565, 0.117483, 0.110092,
        ]
        #expect(velocity.shape == [1, 6, 32])
        guard velocity.shape == [1, 6, 32] else { return }
        let values = velocity.asArray(Float.self)
        for (index, expected) in expected.enumerated() {
            let actual = values[(index * 7) % values.count]
            #expect(abs(actual - expected) < 1e-5, "sample \(index): \(actual) is not \(expected)")
        }
        let meanMagnitude = values.reduce(0) { $0 + abs($1) } / Float(values.count)
        #expect(abs(meanMagnitude - 0.2265647) < 1e-5)
    }

    // The forward-pass tests run on the CPU, as the reference values were computed.

    @Test("The predicted velocity matches the reference implementation")
    func velocityMatchesReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let transformer = try makeTransformer()
            try transformer.update(weights: checkpoint(for: transformer))
            expectReferenceVelocity(velocity(transformer))
        }
    }

    @Test("A checkpoint with the original weight names gives the same result")
    func originalNames() throws {
        try Device.withDefaultDevice(.cpu) {
            let transformer = try makeTransformer(
                Self.tinyConfiguration.replacingOccurrences(
                    of: "\"mlx_format\": true", with: "\"mlx_format\": false"))
            var original = [String: MLXArray]()
            for (key, value) in checkpoint(for: try makeTransformer()) {
                let name =
                    key
                    .replacingOccurrences(
                        of: "time_text_embed.", with: "time_text_embed.timestep_embedder."
                    )
                    .replacingOccurrences(of: "modulation.0.", with: "modulation.1.")
                original[name] = value
            }
            #expect(original["modulation.1.weight"] != nil)
            try transformer.update(weights: original)
            expectReferenceVelocity(velocity(transformer))
        }
    }

    @Test("A quantized checkpoint loads from its directory and runs as it was saved")
    func quantizedCheckpoint() throws {
        let source = try makeTransformer()
        try source.update(weights: checkpoint(for: source))
        quantize(model: source, groupSize: 32, bits: 4)
        eval(source)

        let directory = FileManager.default.temporaryDirectory.appending(
            component: "qwen-image-transformer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = Self.tinyConfiguration.replacingOccurrences(
            of: "\"mlx_format\": true",
            with: "\"mlx_format\": true, \"quantization\": {\"group_size\": 32, \"bits\": 4}")
        try Data(configuration.utf8).write(to: directory.appending(component: "config.json"))
        let saved = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        #expect(saved["img_in.scales"] != nil)
        try save(arrays: saved, url: directory.appending(component: "model.safetensors"))

        let loaded = try QwenImageTransformer.load(from: directory)
        let difference = abs(velocity(loaded) - velocity(source)).max().item(Float.self)
        #expect(difference < 1e-5)

        // Without its quantization entry the same files do not fit the model.
        try Data(Self.tinyConfiguration.utf8).write(
            to: directory.appending(component: "config.json"))
        #expect(throws: (any Error).self) { try QwenImageTransformer.load(from: directory) }
    }

    @Test("The prediction is repeatable and follows the noise level")
    func noiseLevelChangesPrediction() throws {
        let transformer = try makeTransformer()
        try transformer.update(weights: checkpoint(for: transformer))
        func run(_ level: Float) -> MLXArray {
            transformer(
                MLXArray(wave(6 * 32, phase: 0.5, scale: 1), [1, 6, 32]),
                textStates: MLXArray(wave(5 * 32, phase: 1.5, scale: 1), [1, 5, 32]),
                noiseLevel: MLXArray([level]), height: 2, width: 3)
        }
        #expect(abs(run(0.7) - run(0.7)).max().item(Float.self) == 0)
        #expect(abs(run(0.7) - run(0.2)).max().item(Float.self) > 1e-3)
    }

    @Test("A configuration the transformer does not implement is rejected")
    func unsupportedConfiguration() {
        let notCausal = Self.tinyConfiguration.replacingOccurrences(
            of: "\"causal_condition\": true", with: "\"causal_condition\": false")
        #expect(throws: QwenImageTransformerError.self) { try makeTransformer(notCausal) }

        let wrongAxes = Self.tinyConfiguration.replacingOccurrences(
            of: "[4, 6, 6]", with: "[4, 6, 8]")
        #expect(throws: QwenImageTransformerError.self) { try makeTransformer(wrongAxes) }
    }
}
