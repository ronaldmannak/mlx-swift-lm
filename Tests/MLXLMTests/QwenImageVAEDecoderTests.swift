// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXImageGeneration
import MLXNN
import Testing

@Suite struct QwenImageVAEDecoderTests {

    /// A decoder with 16 to 2 channels, small enough to run in a unit test.
    private static let tinyConfiguration = """
        {
          "base_dim": 2,
          "decoder_base_dim": 2,
          "z_dim": 6,
          "dim_mult": [1, 2, 4, 8, 8],
          "num_res_blocks": 2,
          "temperal_downsample": [false, true, true, true],
          "in_channels": 4,
          "out_channels": 4,
          "is_residual": true,
          "latents_mean": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
          "latents_std": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        }
        """

    private func makeDecoder(_ json: String = tinyConfiguration) throws -> QwenImageVAEDecoder {
        try QwenImageVAEDecoder(
            JSONDecoder().decode(QwenImageVAEConfiguration.self, from: Data(json.utf8)))
    }

    // The reference values below come from mlx-vlm's `QwenImageVAE` with these same weights:
    // each tensor is a sine wave over its flat index, with a phase taken from its checkpoint
    // name. That way the test needs no weight file.
    private func wave(_ count: Int, phase: Double, scale: Double, offset: Double = 0) -> [Float] {
        (0 ..< count).map { Float(offset + scale * sin(0.37 * Double($0) + phase)) }
    }

    /// The weights under their checkpoint names, in MLX layout.
    private func checkpoint(for decoder: QwenImageVAEDecoder) -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for (name, value) in decoder.parameters().flattened() {
            let key = name.replacingOccurrences(
                of: ".upsampler.conv.", with: ".upsampler.resample.1.")
            let phase = Double(key.unicodeScalars.reduce(0) { $0 + Int($1.value) } % 1000) * 0.001
            let values: [Float]
            if key.hasSuffix(".gamma") {
                values = wave(value.size, phase: phase, scale: 0.2, offset: 1)
            } else if key.hasSuffix(".bias") {
                values = wave(value.size, phase: phase, scale: 0.1)
            } else {
                let fanIn = Double(value.dim(1) * value.dim(2) * value.dim(3))
                values = wave(value.size, phase: phase, scale: 1 / fanIn.squareRoot())
            }
            weights[key] = MLXArray(values, value.shape)
        }
        // Parts of a VAE checkpoint that decoding one image does not use.
        weights["quant_conv.weight"] = MLXArray.zeros([12, 1, 1, 12])
        weights["encoder.conv_in.bias"] = MLXArray.zeros([2])
        weights["decoder.up_blocks.0.upsampler.time_conv.weight"] = MLXArray.zeros([32, 1, 1, 16])
        return weights
    }

    private var latents: MLXArray { MLXArray(wave(36, phase: 0.5, scale: 1), [1, 3, 2, 6]) }

    private func expectReferencePixels(_ pixels: MLXArray) {
        let expected: [Float] = [
            -0.478232, 0.188606, 0.227083, 0.201037, 0.135422, 0.289296,
            0.248346, 0.167348, 0.073165, 0.286602, 0.239504, 0.167351,
            0.066042, 0.286308, 0.240955, 0.164106, 0.064582, 0.268685,
            0.148990, 0.117718, 0.033262, 0.198247, 0.182075, 0.379418,
        ]
        // 3 x 2 latents become 48 x 32 pixels with 4 channels.
        #expect(pixels.shape == [1, 48, 32, 4])
        guard pixels.shape == [1, 48, 32, 4] else { return }
        let values = pixels.asArray(Float.self)
        for (index, expected) in expected.enumerated() {
            let actual = values[(index * 263) % values.count]
            #expect(abs(actual - expected) < 1e-5, "sample \(index): \(actual) is not \(expected)")
        }
        let meanMagnitude = values.reduce(0) { $0 + abs($1) } / Float(values.count)
        #expect(abs(meanMagnitude - 0.1805626) < 1e-5)
    }

    // Both decoding tests run on the CPU, as the reference values were computed. Metal
    // rounds matrix products more coarsely by default, which moves a sample by up to 2e-4.

    @Test("Decoding matches the reference implementation")
    func decodeMatchesReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let decoder = try makeDecoder()
            try decoder.update(weights: checkpoint(for: decoder), isMLXFormat: true)
            expectReferencePixels(decoder(latents))
        }
    }

    @Test("A checkpoint in the original layout decodes the same")
    func originalLayout() throws {
        try Device.withDefaultDevice(.cpu) {
            let decoder = try makeDecoder()
            // The original checkpoint stores a convolution as [out, in, height, width] and a
            // gain with trailing axes of size 1.
            var original = [String: MLXArray]()
            for (key, value) in checkpoint(for: decoder) {
                if key.hasSuffix(".gamma") {
                    original[key] = value.reshaped(-1, 1, 1, 1)
                } else if value.ndim == 4 {
                    original[key] = value.transposed(0, 3, 1, 2)
                } else {
                    original[key] = value
                }
            }
            try decoder.update(weights: original, isMLXFormat: false)
            expectReferencePixels(decoder(latents))
        }
    }

    @Test("A checkpoint that lacks a decoder weight is rejected")
    func missingWeight() throws {
        let decoder = try makeDecoder()
        var weights = checkpoint(for: decoder)
        weights["decoder.conv_out.bias"] = nil
        #expect(throws: (any Error).self) {
            try decoder.update(weights: weights, isMLXFormat: true)
        }
    }

    @Test("A large 3x3 convolution matches the CPU result")
    func largeConvolutionAccuracy() {
        // The first convolution of the real decoder at 1024 x 1024 pixels: 64 to 1152 channels
        // on a 64 x 64 latent. Plain conv2d on Metal is off by about 6e-3 here.
        let (inputChannels, outputChannels, side) = (64, 1152, 64)
        let (input, weight) = withRandomState(MLXRandom.RandomState(seed: 7)) {
            (
                MLXRandom.normal([1, side, side, inputChannels]),
                MLXRandom.normal([outputChannels, 3, 3, inputChannels])
                    / Float(9 * inputChannels).squareRoot()
            )
        }
        let inputValues = input.asArray(Float.self)
        let weightValues = weight.asArray(Float.self)
        let reference: [Float] = Device.withDefaultDevice(.cpu) {
            conv2d(
                MLXArray(inputValues, input.shape), MLXArray(weightValues, weight.shape),
                stride: 1, padding: 1
            ).asArray(Float.self)
        }
        let actual = accurateConv2d(input, weight, padding: 1).asArray(Float.self)

        var squaredError = 0.0
        var squaredNorm = 0.0
        for (expected, actual) in zip(reference, actual) {
            squaredError += pow(Double(expected) - Double(actual), 2)
            squaredNorm += pow(Double(expected), 2)
        }
        #expect(actual.count == reference.count)
        #expect((squaredError / squaredNorm).squareRoot() < 1e-5)
    }

    @Test("A configuration the decoder does not implement is rejected")
    func unsupportedConfiguration() {
        let plain = Self.tinyConfiguration.replacingOccurrences(
            of: "\"is_residual\": true", with: "\"is_residual\": false")
        #expect(throws: QwenImageVAEError.self) { try makeDecoder(plain) }

        let shortMean = Self.tinyConfiguration.replacingOccurrences(
            of: "[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]", with: "[0.0]")
        #expect(throws: QwenImageVAEError.self) { try makeDecoder(shortMean) }
    }
}
