// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// Port of the single-image decode path of diffusers' `AutoencoderKLQwenImage21`, following
// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen_image/vae.py

/// A `conv2d` with stride 1 that stays off Metal's Winograd path.
///
/// MLX takes that path for a 3x3 convolution whose channel counts are both multiples of 32,
/// and in float32 it loses about 6e-3 of relative accuracy per layer. One extra input
/// channel of zeros gives the same result by the ordinary path.
package func accurateConv2d(_ input: MLXArray, _ weight: MLXArray, padding: Int) -> MLXArray {
    guard weight.dim(1) == 3, weight.dim(2) == 3, weight.dim(0) % 32 == 0, weight.dim(3) % 32 == 0
    else {
        return conv2d(input, weight, stride: 1, padding: IntOrPair(padding))
    }
    let extraChannel: [IntOrPair] = [0, 0, 0, [0, 1]]
    return conv2d(
        padded(input, widths: extraChannel), padded(weight, widths: extraChannel),
        stride: 1, padding: IntOrPair(padding))
}

/// The `vae/config.json` of a Qwen-Image-2.1 checkpoint, as far as decoding needs it.
package struct QwenImageVAEConfiguration: Codable, Sendable {
    package var baseDimensions: Int
    package var decoderBaseDimensions: Int?
    package var latentChannels: Int
    package var dimensionMultipliers: [Int]
    package var residualBlockCount: Int
    package var temporalDownsample: [Bool]
    package var outputChannels: Int
    package var isResidual: Bool?
    package var latentsMean: [Float]
    package var latentsStd: [Float]
    /// True when the weights are already in MLX layout, as a converted checkpoint stores them.
    package var isMLXFormat: Bool?

    enum CodingKeys: String, CodingKey {
        case baseDimensions = "base_dim"
        case decoderBaseDimensions = "decoder_base_dim"
        case latentChannels = "z_dim"
        case dimensionMultipliers = "dim_mult"
        case residualBlockCount = "num_res_blocks"
        case temporalDownsample = "temperal_downsample"
        case outputChannels = "out_channels"
        case isResidual = "is_residual"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
        case isMLXFormat = "mlx_format"
    }
}

package enum QwenImageVAEError: Error, LocalizedError {
    case unsupportedConfiguration(String)

    package var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration(let reason):
            "Unsupported Qwen-Image VAE configuration: \(reason)"
        }
    }
}

/// An RMS norm over the channel axis with a learned gain.
private final class ChannelRMSNorm: Module {
    var gamma: MLXArray
    let scale: Float

    init(_ dimensions: Int) {
        gamma = MLXArray.ones([dimensions])
        scale = Float(dimensions).squareRoot()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let value = x.asType(.float32)
        let inverseNorm = rsqrt((value * value).sum(axis: -1, keepDims: true) + 1e-12)
        return (value * inverseNorm).asType(x.dtype) * scale * gamma
    }
}

/// A stride-1 convolution that keeps the image size.
private final class Convolution: Module {
    var weight: MLXArray
    var bias: MLXArray

    init(_ inputChannels: Int, _ outputChannels: Int, kernelSize: Int) {
        weight = MLXArray.zeros([outputChannels, kernelSize, kernelSize, inputChannels])
        bias = MLXArray.zeros([outputChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        accurateConv2d(x, weight, padding: weight.dim(1) / 2) + bias
    }
}

private final class ResidualBlock: Module {
    @ModuleInfo var norm1: ChannelRMSNorm
    @ModuleInfo var conv1: Convolution
    @ModuleInfo var norm2: ChannelRMSNorm
    @ModuleInfo var conv2: Convolution
    @ModuleInfo(key: "conv_shortcut") var shortcut: Convolution?

    init(_ inputChannels: Int, _ outputChannels: Int) {
        _norm1.wrappedValue = ChannelRMSNorm(inputChannels)
        _conv1.wrappedValue = Convolution(inputChannels, outputChannels, kernelSize: 3)
        _norm2.wrappedValue = ChannelRMSNorm(outputChannels)
        _conv2.wrappedValue = Convolution(outputChannels, outputChannels, kernelSize: 3)
        if inputChannels != outputChannels {
            _shortcut.wrappedValue = Convolution(inputChannels, outputChannels, kernelSize: 1)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = conv1(silu(norm1(x)))
        hidden = conv2(silu(norm2(hidden)))
        return hidden + (shortcut?(x) ?? x)
    }
}

/// Single-head self-attention over the image's positions.
private final class AttentionBlock: Module {
    @ModuleInfo var norm: ChannelRMSNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Convolution
    @ModuleInfo var proj: Convolution

    init(_ channels: Int) {
        _norm.wrappedValue = ChannelRMSNorm(channels)
        _toQKV.wrappedValue = Convolution(channels, channels * 3, kernelSize: 1)
        _proj.wrappedValue = Convolution(channels, channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let qkv = toQKV(norm(x)).reshaped(batch, 1, height * width, 3 * channels)
        let parts = split(qkv, parts: 3, axis: -1)
        let attended = MLXFast.scaledDotProductAttention(
            queries: parts[0], keys: parts[1], values: parts[2],
            scale: 1 / Float(channels).squareRoot(), mask: .none)
        return proj(attended.reshaped(batch, height, width, channels)) + x
    }
}

private final class MidBlock: Module {
    @ModuleInfo var resnets: [ResidualBlock]
    @ModuleInfo var attentions: [AttentionBlock]

    init(_ channels: Int) {
        _resnets.wrappedValue = [
            ResidualBlock(channels, channels), ResidualBlock(channels, channels),
        ]
        _attentions.wrappedValue = [AttentionBlock(channels)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

/// Doubles the image size, then convolves.
private final class Upsampler: Module {
    @ModuleInfo var conv: Convolution

    init(_ channels: Int) {
        _conv.wrappedValue = Convolution(channels, channels, kernelSize: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let doubled = broadcast(
            x.reshaped(batch, height, 1, width, 1, channels),
            to: [batch, height, 2, width, 2, channels]
        ).reshaped(batch, height * 2, width * 2, channels)
        return conv(doubled)
    }
}

private final class UpBlock: Module {
    @ModuleInfo var resnets: [ResidualBlock]
    @ModuleInfo var upsampler: Upsampler?

    let outputChannels: Int
    /// How many frames the video path makes of one here. A single image keeps the last.
    let temporalFactor: Int

    init(
        _ inputChannels: Int, _ outputChannels: Int, residualBlockCount: Int, upsamples: Bool,
        temporalFactor: Int
    ) {
        self.outputChannels = outputChannels
        self.temporalFactor = temporalFactor
        _resnets.wrappedValue = (0 ... residualBlockCount).map { index in
            ResidualBlock(index == 0 ? inputChannels : outputChannels, outputChannels)
        }
        if upsamples {
            _upsampler.wrappedValue = Upsampler(outputChannels)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for resnet in resnets {
            hidden = resnet(hidden)
        }
        guard let upsampler else { return hidden }
        return upsampler(hidden) + duplicatedUp(x)
    }

    /// The block's parameter-free shortcut: repeats the channels, then moves groups of them
    /// into a 2x2 block of pixels.
    private func duplicatedUp(_ x: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let repeats = outputChannels * temporalFactor * 4 / channels
        return repeated(x, count: repeats, axis: -1)
            .reshaped(batch, height, width, outputChannels, temporalFactor, 2, 2)[
                .ellipsis, temporalFactor - 1, 0..., 0...
            ]
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped(batch, height * 2, width * 2, outputChannels)
    }
}

private final class DecoderBody: Module {
    @ModuleInfo(key: "conv_in") var convIn: Convolution
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock]
    @ModuleInfo(key: "norm_out") var normOut: ChannelRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: Convolution

    init(_ configuration: QwenImageVAEConfiguration) {
        let base = configuration.decoderBaseDimensions ?? configuration.baseDimensions
        let multipliers = configuration.dimensionMultipliers
        let channels =
            [base * multipliers[multipliers.count - 1]] + multipliers.reversed().map { base * $0 }
        let temporalUpsample = Array(configuration.temporalDownsample.reversed())

        _convIn.wrappedValue = Convolution(
            configuration.latentChannels, channels[0], kernelSize: 3)
        _midBlock.wrappedValue = MidBlock(channels[0])
        _upBlocks.wrappedValue = multipliers.indices.map { index in
            let upsamples = index != multipliers.count - 1
            return UpBlock(
                channels[index], channels[index + 1],
                residualBlockCount: configuration.residualBlockCount, upsamples: upsamples,
                temporalFactor: upsamples && temporalUpsample[index] ? 2 : 1)
        }
        _normOut.wrappedValue = ChannelRMSNorm(channels[channels.count - 1])
        _convOut.wrappedValue = Convolution(
            channels[channels.count - 1], configuration.outputChannels, kernelSize: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = midBlock(convIn(x))
        for block in upBlocks {
            hidden = block(hidden)
        }
        return convOut(silu(normOut(hidden)))
    }
}

/// The image decoder of Qwen-Image-2.1's VAE.
///
/// It turns 64-channel latents into RGBA pixels at 16 times the size. The checkpoint's
/// encoder and the temporal convolutions of its video path are not loaded.
package final class QwenImageVAEDecoder: Module {
    @ModuleInfo(key: "post_quant_conv") private var postQuantConv: Convolution
    @ModuleInfo private var decoder: DecoderBody

    package let configuration: QwenImageVAEConfiguration

    package init(_ configuration: QwenImageVAEConfiguration) throws {
        guard configuration.isResidual ?? true else {
            throw QwenImageVAEError.unsupportedConfiguration("is_residual is false")
        }
        guard configuration.temporalDownsample.count == configuration.dimensionMultipliers.count - 1
        else {
            throw QwenImageVAEError.unsupportedConfiguration(
                "temperal_downsample needs one entry per upsampling block")
        }
        guard configuration.latentsMean.count == configuration.latentChannels,
            configuration.latentsStd.count == configuration.latentChannels
        else {
            throw QwenImageVAEError.unsupportedConfiguration(
                "latents_mean and latents_std need one entry per latent channel")
        }
        self.configuration = configuration
        _postQuantConv.wrappedValue = Convolution(
            configuration.latentChannels, configuration.latentChannels, kernelSize: 1)
        _decoder.wrappedValue = DecoderBody(configuration)
    }

    /// Decodes normalized latents, `[batch, height, width, channels]`, into pixels in
    /// `-1 ... 1`, `[batch, 16 * height, 16 * width, 4]`.
    package func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let scaled =
            latents.asType(.float32) * MLXArray(configuration.latentsStd)
            + MLXArray(configuration.latentsMean)
        return clip(decoder(postQuantConv(scaled)), min: -1, max: 1)
    }

    /// Keeps the decoder's weights from a VAE checkpoint and gives them this module's names
    /// and layout.
    package static func sanitize(_ weights: [String: MLXArray], isMLXFormat: Bool)
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            guard key.hasPrefix("decoder.") || key.hasPrefix("post_quant_conv.") else { continue }
            // Only the video path uses the temporal convolutions.
            guard !key.contains(".time_conv.") else { continue }
            var value = value
            if !isMLXFormat {
                if key.hasSuffix(".gamma") {
                    value = value.reshaped(-1)
                } else if value.ndim == 4 {
                    value = value.transposed(0, 2, 3, 1)
                }
            }
            let name = key.replacingOccurrences(
                of: ".upsampler.resample.1.", with: ".upsampler.conv.")
            sanitized[name] = value
        }
        return sanitized
    }

    /// Loads the decoder from the `vae` directory of a checkpoint.
    package static func load(from directory: URL) throws -> QwenImageVAEDecoder {
        let configuration = try JSONDecoder().decode(
            QwenImageVAEConfiguration.self,
            from: Data(contentsOf: directory.appending(component: "config.json")))
        let decoder = try QwenImageVAEDecoder(configuration)
        try decoder.update(
            weights: loadSafetensors(in: directory),
            isMLXFormat: configuration.isMLXFormat ?? false)
        return decoder
    }

    /// Replaces the decoder's weights with those of a VAE checkpoint.
    package func update(weights: [String: MLXArray], isMLXFormat: Bool) throws {
        let sanitized = Self.sanitize(weights, isMLXFormat: isMLXFormat)
        try update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(self)
    }
}

/// Reads every safetensors file in `directory`.
func loadSafetensors(in directory: URL) throws -> [String: MLXArray] {
    let files = try FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.hasPrefix("._") }
    guard !files.isEmpty else {
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: directory.path])
    }
    var weights = [String: MLXArray]()
    for file in files {
        weights.merge(try loadArrays(url: file)) { _, new in new }
    }
    return weights
}
