// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of the text-to-image path of diffusers' `QwenImage21Transformer2DModel`, following
// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen_image/transformer.py

/// The `transformer/config.json` of a Qwen-Image-2.1 checkpoint.
package struct QwenImageTransformerConfiguration: Codable, Sendable {
    package var inputChannels: Int
    package var outputChannels: Int
    package var layerCount: Int
    package var headCount: Int
    package var headDimensions: Int
    package var contextDimensions: Int
    package var mlpRatio: Int
    /// How the head dimensions are split between the frame, height and width positions.
    package var ropeAxesDimensions: [Int]
    package var eps: Float?
    /// Text tokens attend only to earlier tokens and are modulated as at timestep 0.
    package var causalCondition: Bool?
    package var quantization: BaseConfiguration.Quantization?
    /// True when the weights already carry this module's names, as a converted checkpoint
    /// stores them.
    package var isMLXFormat: Bool?

    package var hiddenDimensions: Int { headCount * headDimensions }

    enum CodingKeys: String, CodingKey {
        case inputChannels = "in_channels"
        case outputChannels = "out_channels"
        case layerCount = "num_layers"
        case headCount = "num_attention_heads"
        case headDimensions = "attention_head_dim"
        case contextDimensions = "context_in_dim"
        case mlpRatio = "mlp_ratio"
        case ropeAxesDimensions = "axes_dims_rope"
        case eps
        case causalCondition = "causal_condition"
        case quantization
        case isMLXFormat = "mlx_format"
    }
}

package enum QwenImageTransformerError: Error, LocalizedError {
    case unsupportedConfiguration(String)

    package var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration(let reason):
            "Unsupported Qwen-Image transformer configuration: \(reason)"
        }
    }
}

/// Rotates pairs of `x`, `[batch, heads, tokens, dimensions]`, by per-token angles given as
/// `[tokens, dimensions / 2]` cosines and sines.
private func applyRotaryPositions(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let shape = x.shape
    let pairs = x.reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
    let even = pairs[.ellipsis, 0]
    let odd = pairs[.ellipsis, 1]
    return stacked([even * cos - odd * sin, even * sin + odd * cos], axis: -1)
        .reshaped(shape).asType(x.dtype)
}

/// Spreads per-sample modulation over the tokens. Image tokens get their sample's row; text
/// tokens get the last row, which was computed for timestep 0.
private func perToken(_ parameters: MLXArray, isImage: MLXArray) -> MLXArray {
    let last = parameters.dim(0) - 1
    return which(
        isImage, parameters[..<last].expandedDimensions(axis: 1),
        parameters[last...].expandedDimensions(axis: 0))
}

private final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ dimensions: Int) {
        _linear1.wrappedValue = Linear(256, dimensions, bias: false)
        _linear2.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }

    /// `noiseLevels` are in `0 ... 1`, one per row.
    func callAsFunction(_ noiseLevels: MLXArray, dtype: DType) -> MLXArray {
        let half = 128
        let frequencies = exp(-log(Float(10000)) * MLXArray(0 ..< half).asType(.float32) / half)
        let angles =
            (1000 * noiseLevels.asType(.float32)).expandedDimensions(axis: 1) * frequencies
        let embedding = concatenated([cos(angles), sin(angles)], axis: -1).asType(dtype)
        return linear2(silu(linear1(embedding)))
    }
}

/// An RMS norm whose stored weight is the gain minus one.
private final class ZeroCenteredRMSNorm: Module {
    var weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let value = x.asType(.float32)
        let inverseRMS = rsqrt((value * value).mean(axis: -1, keepDims: true) + eps)
        return (value * inverseRMS * (weight.asType(.float32) + 1)).asType(x.dtype)
    }
}

private final class TextProjection: Module {
    @ModuleInfo(key: "text_norm") var textNorm: ZeroCenteredRMSNorm
    @ModuleInfo(key: "in_layer") var inLayer: Linear
    @ModuleInfo(key: "out_layer") var outLayer: Linear

    init(_ contextDimensions: Int, _ dimensions: Int, eps: Float) {
        _textNorm.wrappedValue = ZeroCenteredRMSNorm(contextDimensions, eps: eps)
        _inLayer.wrappedValue = Linear(contextDimensions, dimensions, bias: false)
        _outLayer.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outLayer(geluApproximate(inLayer(textNorm(x))))
    }
}

private final class GatedMLP: Module {
    @ModuleInfo var proj: Linear
    @ModuleInfo var out: Linear
    @ModuleInfo(key: "gate_layer") var gateLayer: Linear

    init(_ dimensions: Int, hiddenDimensions: Int) {
        _proj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _out.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _gateLayer.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        out(silu(gateLayer(x)) * proj(x))
    }
}

private final class Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let headCount: Int
    let headDimensions: Int

    init(_ configuration: QwenImageTransformerConfiguration, eps: Float) {
        headCount = configuration.headCount
        headDimensions = configuration.headDimensions
        let dimensions = configuration.hiddenDimensions
        _toQ.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toK.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toV.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toOut.wrappedValue = [Linear(dimensions, dimensions, bias: false)]
        _normQ.wrappedValue = RMSNorm(dimensions: headDimensions, eps: eps)
        _normK.wrappedValue = RMSNorm(dimensions: headDimensions, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, tokens) = (x.dim(0), x.dim(1))
        func heads(_ value: MLXArray) -> MLXArray {
            value.reshaped(batch, tokens, headCount, headDimensions)
        }
        let queries = applyRotaryPositions(
            normQ(heads(toQ(x))).transposed(0, 2, 1, 3), cos: cos, sin: sin)
        let keys = applyRotaryPositions(
            normK(heads(toK(x))).transposed(0, 2, 1, 3), cos: cos, sin: sin)
        let values = heads(toV(x)).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / Float(headDimensions).squareRoot(), mask: .array(mask))
        return toOut[0](attended.transposed(0, 2, 1, 3).reshaped(batch, tokens, -1))
    }
}

private final class TransformerBlock: Module {
    @ModuleInfo var attn: Attention
    @ModuleInfo(key: "img_mlp") var mlp: GatedMLP

    let eps: Float

    init(_ configuration: QwenImageTransformerConfiguration, eps: Float) {
        self.eps = eps
        _attn.wrappedValue = Attention(configuration, eps: eps)
        _mlp.wrappedValue = GatedMLP(
            configuration.hiddenDimensions,
            hiddenDimensions: configuration.hiddenDimensions * configuration.mlpRatio)
    }

    /// Scales the normalized tokens and returns the gate for what is computed from them.
    private func modulated(_ x: MLXArray, _ modulation: MLXArray, isImage: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let parts = split(modulation, parts: 2, axis: -1)
        let normalized = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
        return (
            normalized * (1 + perToken(parts[0], isImage: isImage)),
            tanh(perToken(parts[1], isImage: isImage))
        )
    }

    func callAsFunction(
        _ x: MLXArray, modulation: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray,
        isImage: MLXArray
    ) -> MLXArray {
        let halves = split(modulation, parts: 2, axis: -1)
        let (attentionInput, attentionGate) = modulated(x, halves[0], isImage: isImage)
        let attended = x + attentionGate * attn(attentionInput, cos: cos, sin: sin, mask: mask)
        let (mlpInput, mlpGate) = modulated(attended, halves[1], isImage: isImage)
        return attended + mlpGate * mlp(mlpInput)
    }
}

private final class OutputNorm: Module {
    @ModuleInfo var linear: Linear

    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.eps = eps
        _linear.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }

    /// `x` holds image tokens only, so every token takes its own sample's row.
    func callAsFunction(_ x: MLXArray, timeEmbedding: MLXArray) -> MLXArray {
        let scale = linear(silu(timeEmbedding).asType(x.dtype))[..<(timeEmbedding.dim(0) - 1)]
        return MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
            * (1 + scale.expandedDimensions(axis: 1))
    }
}

/// The denoiser of Qwen-Image-2.1: one stream of text tokens followed by image tokens.
///
/// It predicts the velocity that moves noisy image latents toward the image. Reference
/// images for editing, and the key-value cache that serves them, are not implemented.
package final class QwenImageTransformer: Module {
    @ModuleInfo(key: "time_text_embed") private var timeEmbedding: TimestepEmbedding
    @ModuleInfo(key: "txt_in") private var textInput: TextProjection
    @ModuleInfo(key: "img_in") private var imageInput: Linear
    @ModuleInfo private var modulation: [Linear]
    @ModuleInfo(key: "transformer_blocks") private var blocks: [TransformerBlock]
    @ModuleInfo(key: "norm_out") private var outputNorm: OutputNorm
    @ModuleInfo(key: "proj_out") private var outputProjection: Linear

    package let configuration: QwenImageTransformerConfiguration

    package init(_ configuration: QwenImageTransformerConfiguration) throws {
        guard configuration.causalCondition ?? true else {
            throw QwenImageTransformerError.unsupportedConfiguration("causal_condition is false")
        }
        guard configuration.ropeAxesDimensions.count == 3,
            configuration.ropeAxesDimensions.reduce(0, +) == configuration.headDimensions,
            configuration.ropeAxesDimensions.allSatisfy({ $0 % 2 == 0 })
        else {
            throw QwenImageTransformerError.unsupportedConfiguration(
                "axes_dims_rope needs three even entries that add up to attention_head_dim")
        }
        self.configuration = configuration
        let dimensions = configuration.hiddenDimensions
        let eps = configuration.eps ?? 1e-6
        _timeEmbedding.wrappedValue = TimestepEmbedding(dimensions)
        _textInput.wrappedValue = TextProjection(
            configuration.contextDimensions, dimensions, eps: eps)
        _imageInput.wrappedValue = Linear(configuration.inputChannels, dimensions, bias: false)
        _modulation.wrappedValue = [Linear(dimensions, 4 * dimensions, bias: false)]
        _blocks.wrappedValue = (0 ..< configuration.layerCount).map { _ in
            TransformerBlock(configuration, eps: eps)
        }
        _outputNorm.wrappedValue = OutputNorm(dimensions, eps: eps)
        _outputProjection.wrappedValue = Linear(
            dimensions, configuration.outputChannels, bias: false)
    }

    /// Cosines and sines of every token's rotary angles, `[tokens, headDimensions / 2]`.
    ///
    /// A text token uses its index on all three axes. Image tokens share the frame position
    /// after the text and count rows and columns from the image's center.
    private func rotaryPositions(textLength: Int, height: Int, width: Int) -> (MLXArray, MLXArray) {
        let text = (0 ..< textLength).map(Float.init)
        var frame = text
        var row = text
        var column = text
        for y in 0 ..< height {
            for x in 0 ..< width {
                frame.append(Float(textLength))
                row.append(Float(y - (height - height / 2)))
                column.append(Float(x - (width - width / 2)))
            }
        }
        let angles = zip([frame, row, column], configuration.ropeAxesDimensions).map {
            positions, dimensions in
            let exponents = MLXArray(Array(stride(from: 0, to: dimensions, by: 2)))
                .asType(.float32)
            let frequencies = exp(-log(Float(10000)) * exponents / Float(dimensions))
            return MLXArray(positions).expandedDimensions(axis: 1) * frequencies
        }
        let joined = concatenated(angles, axis: -1)
        return (cos(joined), sin(joined))
    }

    /// Predicts the velocity of the image latents.
    ///
    /// - Parameters:
    ///   - latents: noisy image latents, `[batch, height * width, channels]`, row by row
    ///   - textStates: the text encoder's states for the prompt, `[batch, tokens, context]`
    ///   - noiseLevel: the current noise level in `0 ... 1`, one per batch row
    ///   - height: the latent image's height
    ///   - width: the latent image's width
    package func callAsFunction(
        _ latents: MLXArray, textStates: MLXArray, noiseLevel: MLXArray, height: Int, width: Int
    ) -> MLXArray {
        precondition(latents.dim(1) == height * width, "latents do not match the image size")
        let textLength = textStates.dim(1)
        let imageLength = height * width
        var hidden = concatenated([textInput(textStates), imageInput(latents)], axis: 1)

        let (cos, sin) = rotaryPositions(textLength: textLength, height: height, width: width)
        let index = MLXArray(0 ..< (textLength + imageLength))
        let imageToken = index .>= textLength
        // Text attends to earlier text. Image tokens attend to the text and to each other.
        let causal = index.expandedDimensions(axis: 1) .>= index.expandedDimensions(axis: 0)
        let bothImage =
            imageToken.expandedDimensions(axis: 1) .&& imageToken.expandedDimensions(axis: 0)
        let mask = (causal .|| bothImage).expandedDimensions(axes: [0, 1])
        let isImage = imageToken.reshaped(1, -1, 1)

        // The extra row at noise level 0 conditions the text tokens.
        let levels = concatenated([
            noiseLevel.asType(hidden.dtype), MLXArray.zeros([1], dtype: hidden.dtype),
        ])
        let embedded = timeEmbedding(levels, dtype: hidden.dtype)
        let modulation = self.modulation[0](silu(embedded))
        for block in blocks {
            hidden = block(
                hidden, modulation: modulation, cos: cos, sin: sin, mask: mask, isImage: isImage)
        }
        let image = hidden[0..., textLength...]
        return outputProjection(outputNorm(image, timeEmbedding: embedded))
    }

    /// Gives a checkpoint's weights this module's names.
    package static func sanitize(_ weights: [String: MLXArray], isMLXFormat: Bool)
        -> [String: MLXArray]
    {
        guard !isMLXFormat else { return weights }
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            let name =
                key
                .replacingOccurrences(
                    of: "time_text_embed.timestep_embedder.", with: "time_text_embed."
                )
                .replacingOccurrences(of: "modulation.1.", with: "modulation.0.")
            sanitized[name] = value
        }
        return sanitized
    }

    /// Loads the transformer from the `transformer` directory of a checkpoint.
    package static func load(from directory: URL) throws -> QwenImageTransformer {
        let configuration = try JSONDecoder().decode(
            QwenImageTransformerConfiguration.self,
            from: Data(contentsOf: directory.appending(component: "config.json")))
        let transformer = try QwenImageTransformer(configuration)
        try transformer.update(weights: loadSafetensors(in: directory))
        return transformer
    }

    /// Replaces the weights with a checkpoint's, quantizing the layers it stores quantized.
    package func update(weights: [String: MLXArray]) throws {
        let sanitized = Self.sanitize(weights, isMLXFormat: configuration.isMLXFormat ?? false)
        if let quantization = configuration.quantization {
            quantize(model: self) { path, _ in
                sanitized["\(path).scales"] != nil ? quantization.asTuple : nil
            }
        }
        try update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(self)
    }
}
