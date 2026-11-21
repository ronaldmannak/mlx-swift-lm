//
//  Inkling2.swift
//  mlx-swift-lm
//

// Port of `mlx_lm/models/inkling2/{model,vision,processor}.py` from jundot/mlx-lm@0c4c9919
// (MIT): Thinking Machines' Inkling 2 and Inkling 2 Mini with image input. The vision tower
// is a CLIP-style encoder with 2-D rotary positions and a 2x2 patch merger.
//
// The configuration, text model and checkpoint mapping are in MLXLMCommon
// (`Models/Inkling2*.swift`), shared with the MLXLLM wrapper.

import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Vision

private func preciseGELU(_ x: MLXArray) -> MLXArray {
    gelu(x.asType(.float32)).asType(x.dtype)
}

/// The row and column of every patch, with the patches of each merge block together.
private func gridPositions(height: Int, width: Int, merge: Int) -> (MLXArray, MLXArray) {
    let i = MLXArray(Int32(0) ..< Int32(height * width))
    let blocksWide = Int32(width / merge)
    let m = Int32(merge)
    let row = i.floorDivide(m * m * blocksWide) * m + i.floorDivide(m) % m
    let col = i.floorDivide(m * m) % blocksWide * m + i % m
    return (row, col)
}

/// The two grid taps and weights that interpolate `index` of `size` onto `side` positions.
private func bilinearTaps(_ index: MLXArray, size: Int, side: Int) -> (MLXArray, MLXArray) {
    let source = (index.asType(.float32) + 0.5) * Float(side) / Float(size) - 0.5
    let taps = expandedDimensions(floor(source), axis: 1) + MLXArray([Float(0), 1])
    let weights = MLX.maximum(
        1 - abs(expandedDimensions(source, axis: 1) - taps), MLXArray(Float(0)))
    return (clip(taps, min: 0, max: side - 1).asType(.int32), weights)
}

final class Inkling2VisionEmbeddings: Module {
    let mergeSize: Int
    let gridSize: Int

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "cls_position_embedding") var classPositionEmbedding: MLXArray

    init(_ args: Inkling2VisionConfiguration) {
        mergeSize = args.mergeSize
        gridSize = args.positionGridSize
        _patchEmbedding.wrappedValue = Linear(
            args.channels * args.patchSize * args.patchSize, args.hiddenSize, bias: false)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: args.positionGridSize * args.positionGridSize,
            dimensions: args.hiddenSize)
        _classEmbedding.wrappedValue = MLXArray.zeros([args.hiddenSize])
        _classPositionEmbedding.wrappedValue = MLXArray.zeros([args.hiddenSize])
    }

    func callAsFunction(_ patches: MLXArray, gridHeight: Int, gridWidth: Int) -> MLXArray {
        var embeddings = patchEmbedding(patches)
        let side = gridSize
        let (row, col) = gridPositions(height: gridHeight, width: gridWidth, merge: mergeSize)
        let (rowTaps, rowWeights) = bilinearTaps(row, size: gridHeight, side: side)
        let (colTaps, colWeights) = bilinearTaps(col, size: gridWidth, side: side)
        let indices =
            expandedDimensions(rowTaps, axis: 2) * Int32(side)
            + expandedDimensions(colTaps, axis: 1)
        let weights =
            expandedDimensions(rowWeights, axis: 2) * expandedDimensions(colWeights, axis: 1)
        let positions =
            positionEmbedding(indices.reshaped(-1, 4)).asType(.float32) * weights.reshaped(-1, 4, 1)
        embeddings = embeddings + positions.sum(axis: 1).asType(embeddings.dtype)

        let classToken = (classEmbedding + classPositionEmbedding).asType(embeddings.dtype)
        return concatenated([classToken.reshaped(1, -1), embeddings], axis: 0)
    }
}

/// Cosines and sines of the 2-D rotary positions: the first half of each head rotates by
/// column, the second by row.
private func visionRotary(row: MLXArray, col: MLXArray, headDim: Int, theta: Float)
    -> (MLXArray, MLXArray)
{
    let dims = headDim / 2
    let exponents = MLXArray(stride(from: 0, to: dims, by: 2).map { Float($0) / Float(dims) })
    let inverseFrequency = 1 / pow(MLXArray(theta), exponents)
    let rows = expandedDimensions(row.asType(.float32), axis: 1) * inverseFrequency
    let cols = expandedDimensions(col.asType(.float32), axis: 1) * inverseFrequency
    let frequencies = repeated(concatenated([cols, rows], axis: -1), count: 2, axis: -1)
    return (cos(frequencies), sin(frequencies))
}

/// Rotates interleaved pairs, `(x0, x1) -> (x0 cos - x1 sin, x1 cos + x0 sin)`.
private func applyVisionRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let states = x.asType(.float32)
    let pairs = states.reshaped(Array(states.shape.dropLast()) + [-1, 2])
    let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1)
        .reshaped(states.shape)
    return (states * cos + rotated * sin).asType(x.dtype)
}

final class Inkling2VisionAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProjection: Linear
    @ModuleInfo(key: "k_proj") var kProjection: Linear
    @ModuleInfo(key: "v_proj") var vProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    init(_ args: Inkling2VisionConfiguration) {
        heads = args.attentionHeads
        scale = pow(Float(args.headDim), -0.5)
        let dims = args.hiddenSize
        _qProjection.wrappedValue = Linear(dims, dims)
        _kProjection.wrappedValue = Linear(dims, dims)
        _vProjection.wrappedValue = Linear(dims, dims)
        _outProjection.wrappedValue = Linear(dims, dims)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let N = x.dim(0)
        let (cos, sin) = (expandedDimensions(cos, axis: 1), expandedDimensions(sin, axis: 1))
        let q = applyVisionRotary(qProjection(x).reshaped(N, heads, -1), cos: cos, sin: sin)
        let k = applyVisionRotary(kProjection(x).reshaped(N, heads, -1), cos: cos, sin: sin)
        let v = vProjection(x).reshaped(N, heads, -1)
        let output = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: scale, mask: .none)
        return outProjection(output[0].transposed(1, 0, 2).reshaped(N, -1))
    }
}

final class Inkling2VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ args: Inkling2VisionConfiguration) {
        _fc1.wrappedValue = Linear(args.hiddenSize, args.intermediateSize)
        _fc2.wrappedValue = Linear(args.intermediateSize, args.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(preciseGELU(fc1(x)))
    }
}

final class Inkling2VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2VisionAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2VisionMLP
    @ModuleInfo(key: "layernorm_before") var normBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var normAfter: LayerNorm
    @ParameterInfo(key: "lambda_1") var attentionScale: MLXArray
    @ParameterInfo(key: "lambda_2") var mlpScale: MLXArray

    init(_ args: Inkling2VisionConfiguration) {
        _attention.wrappedValue = Inkling2VisionAttention(args)
        _mlp.wrappedValue = Inkling2VisionMLP(args)
        _normBefore.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        _normAfter.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        _attentionScale.wrappedValue = MLXArray.ones([args.hiddenSize])
        _mlpScale.wrappedValue = MLXArray.ones([args.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attentionScale * attention(normBefore(x), cos: cos, sin: sin)
        return h + mlpScale * mlp(normAfter(h))
    }
}

final class Inkling2VisionMerger: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ args: Inkling2VisionConfiguration, textHiddenSize: Int) {
        _linear1.wrappedValue = Linear(
            args.mergeSize * args.mergeSize * args.hiddenSize, args.projectionHiddenDim,
            bias: args.projectorBias)
        _linear2.wrappedValue = Linear(
            args.projectionHiddenDim, textHiddenSize, bias: args.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(preciseGELU(linear1(x)))
    }
}

/// Encodes one image, `[channels, H, W]` with sides a multiple of the merge block, into
/// one text-width row per merge block, in raster order.
public final class Inkling2VisionModel: Module {
    let args: Inkling2VisionConfiguration

    @ModuleInfo(key: "embeddings") var embeddings: Inkling2VisionEmbeddings
    @ModuleInfo(key: "pre_layernorm") var preNorm: LayerNorm
    let layers: [Inkling2VisionLayer]
    @ModuleInfo(key: "merger") var merger: Inkling2VisionMerger

    package init(_ args: Inkling2VisionConfiguration, textHiddenSize: Int) {
        self.args = args
        _embeddings.wrappedValue = Inkling2VisionEmbeddings(args)
        _preNorm.wrappedValue = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        layers = (0 ..< args.hiddenLayers).map { _ in Inkling2VisionLayer(args) }
        _merger.wrappedValue = Inkling2VisionMerger(args, textHiddenSize: textHiddenSize)
    }

    /// The merge-block side in pixels.
    package var blockSize: Int { args.patchSize * args.mergeSize }

    package func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
        let (C, H, W) = (pixels.dim(0), pixels.dim(1), pixels.dim(2))
        let (p, m) = (args.patchSize, args.mergeSize)
        guard H % (p * m) == 0, W % (p * m) == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 image size \(H)x\(W) is not a multiple of \(p * m)")
        }
        let (rows, cols) = (H / (p * m), W / (p * m))
        // The merge blocks in raster order, then the patches of each block.
        let patches = pixels.transposed(1, 2, 0).reshaped(rows, m, p, cols, m, p, C)
            .transposed(0, 3, 1, 4, 2, 5, 6).reshaped(-1, p * p * C)
        let dtype = embeddings.patchEmbedding.weight.dtype
        var h = embeddings(patches.asType(dtype), gridHeight: rows * m, gridWidth: cols * m)
        h = preNorm(h)

        let (row, col) = gridPositions(height: rows * m, width: cols * m, merge: m)
        let zero = MLXArray.zeros([1], dtype: row.dtype)
        let (cos, sin) = visionRotary(
            row: concatenated([zero, row + 1]), col: concatenated([zero, col + 1]),
            headDim: args.headDim, theta: args.ropeTheta)
        for layer in layers {
            h = layer(h, cos: cos, sin: sin)
        }
        return merger(h[1...].reshaped(rows * cols, -1))
    }
}

// MARK: - Model

/// The text model and vision tower, at `model.language_model` and `model.vision_tower`.
public final class Inkling2VLMBackbone: Module {
    @ModuleInfo(key: "language_model") public var languageModel: Inkling2TextModel
    @ModuleInfo(key: "vision_tower") public var visionTower: Inkling2VisionModel?

    init(_ config: Inkling2Configuration) {
        _languageModel.wrappedValue = Inkling2TextModel(config.text)
        _visionTower.wrappedValue = config.vision.map {
            Inkling2VisionModel($0, textHiddenSize: config.text.hiddenSize)
        }
    }
}

public final class Inkling2VLM: Module, VLMModel {
    public let configuration: Inkling2Configuration

    @ModuleInfo(key: "model") public var model: Inkling2VLMBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public var vocabularySize: Int { configuration.text.unpaddedVocabularySize }
    public var toolCallFormat: ToolCallFormat? { .inkling }

    /// Without a `vision_config` the model reads text only.
    public init(_ config: Inkling2Configuration) {
        configuration = config
        _model.wrappedValue = Inkling2VLMBackbone(config)
        _lmHead.wrappedValue = Linear(
            config.text.hiddenSize, config.text.vocabularySize, bias: false)
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        inkling2Logits(hidden, lmHead: lmHead, configuration: configuration.text)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model.languageModel(inputs, cache: cache))
    }

    /// Also emits the hidden states a DSpark drafter asks for.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        inkling2Output(
            embeddings: model.languageModel.embed(input.tokens), cache: cache, state: state)
    }

    /// Token embeddings `[1, L, hidden]` with each image's rows at its image tokens.
    func inputEmbeddings(tokens: MLXArray, image: LMInput.ProcessedImage?) throws -> MLXArray {
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        var embeddings = model.languageModel.embed(tokens)
        guard let image else { return embeddings }
        guard let visionTower = model.visionTower else {
            throw VLMError.processing("This Inkling 2 checkpoint has no vision tower")
        }

        let channels = configuration.vision?.channels ?? 3
        let frames = image.frames ?? []
        var offset = 0
        var features = [MLXArray]()
        for frame in frames {
            let count = channels * frame.h * frame.w
            let pixels = image.pixels[offset ..< (offset + count)].reshaped(
                channels, frame.h, frame.w)
            features.append(try visionTower(pixels))
            offset += count
        }
        let rows = concatenated(features, axis: 0).asType(embeddings.dtype)
        let positions = tokens[0].asArray(Int.self).enumerated().compactMap {
            $0.element == configuration.imageTokenId ? Int32($0.offset) : nil
        }
        guard positions.count == rows.dim(0) else {
            throw VLMError.processing(
                "Inkling 2 has \(positions.count) image tokens for \(rows.dim(0)) image rows")
        }
        embeddings[0..., MLXArray(positions), 0...] = rows.expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let embeddings = try inputEmbeddings(tokens: input.text.tokens, image: input.image)
        // An image can add thousands of positions, so the prompt goes through in chunks.
        let total = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = model.languageModel(embeddings: embeddings[0..., range, 0...], cache: cache)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        // A drafter that asks for hidden states gets those of this last chunk, which is the
        // whole prompt when prefill is unchunked.
        let output = inkling2Output(
            embeddings: embeddings[0..., processed..., 0...], cache: cache, state: state)
        prefill.progress?(total, total)
        return .logits(output)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try Inkling2Checkpoint.prepare(
            checkpoint, configuration: configuration, includeVision: configuration.vision != nil)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.languageModel.newCache(parameters: parameters)
    }
}

extension Inkling2VLM: LoRAModel {
    public var loraLayers: [Module] { model.languageModel.layers }
}

extension Inkling2VLM: Inkling2LanguageModel {
    package var inkling2TextModel: Inkling2TextModel { model.languageModel }
    package var inkling2Head: Linear { lmHead }
}

// MARK: - Processor

public struct Inkling2ProcessorConfiguration: Codable, Sendable {
    public struct ImageProcessor: Codable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            imageMean =
                try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
            imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        }
    }

    public let imageProcessor: ImageProcessor?
    public let imageToken: String

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageToken = "image_token"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageProcessor = try c.decodeIfPresent(ImageProcessor.self, forKey: .imageProcessor)
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<|unused_200054|>"
    }
}

/// Pads each image to whole 40-pixel blocks at its own resolution, and gives it one image
/// token per block. A requested `resize` or `maxPixels` downscales first.
public struct Inkling2Processor: UserInputProcessor {
    private let config: Inkling2ProcessorConfiguration
    private let tokenizer: any Tokenizer
    /// The merge-block side in pixels: patch 20 x merge 2.
    static let blockSize = 40

    public init(_ config: Inkling2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// `[3, H, W]` pixels with sides padded to whole blocks, and the block count.
    func preprocess(image: CIImage, processing: UserInput.Processing?) -> (MLXArray, Int) {
        let block = Self.blockSize
        var image = MediaProcessing.apply(image, processing: processing)
        if let maxPixels = processing?.maxPixels {
            let pixels = image.extent.width * image.extent.height
            if pixels > CGFloat(maxPixels) {
                let scale = (CGFloat(maxPixels) / pixels).squareRoot()
                image = MediaProcessing.resampleBicubic(
                    image,
                    to: CGSize(
                        width: max(1, (image.extent.width * scale).rounded(.down)),
                        height: max(1, (image.extent.height * scale).rounded(.down))))
            }
        }
        // As the reference, transparent pixels go on a gray checkerboard of 8-pixel squares.
        let extent = image.extent
        let checkerboard = CIFilter.checkerboardGenerator()
        checkerboard.color0 = CIColor(red: 0.8, green: 0.8, blue: 0.8)
        checkerboard.color1 = CIColor(red: 0.6, green: 0.6, blue: 0.6)
        checkerboard.width = 8
        checkerboard.center = CGPoint(x: extent.minX, y: extent.maxY)
        if let background = checkerboard.outputImage?.cropped(to: extent) {
            image = image.composited(over: background)
        }

        let mean = config.imageProcessor?.imageMean ?? [0.5, 0.5, 0.5]
        let std = config.imageProcessor?.imageStd ?? [0.5, 0.5, 0.5]
        let normalized = MediaProcessing.normalize(
            MediaProcessing.inSRGBToneCurveSpace(image), mean: (mean[0], mean[1], mean[2]),
            std: (std[0], std[1], std[2]))
        let pixels = MediaProcessing.asMLXArray(normalized)[0]
        let (H, W) = (pixels.dim(1), pixels.dim(2))
        let rows = (H + block - 1) / block
        let cols = (W + block - 1) / block
        // As the reference, the padding is -1 on the 0-255 scale.
        let fill = MLXArray(
            (0 ..< 3).map { Float((-1 / 255 - mean[$0]) / std[$0]) }
        ).reshaped(3, 1, 1)
        var padded = MLXArray.zeros([3, rows * block, cols * block]) + fill
        padded[0..., ..<H, ..<W] = pixels
        return (padded, rows * cols)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else { throw VLMError.singleMediaTypeAllowed }
        let messages = Qwen2VLMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        guard !input.images.isEmpty else {
            return LMInput(text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)))
        }

        let processed = try input.images.map {
            preprocess(image: try $0.asCIImage(), processing: input.processing)
        }
        guard let imageToken = tokenizer.convertTokenToId(config.imageToken) else {
            throw VLMError.processing("Inkling 2 tokenizer has no \(config.imageToken) token")
        }
        var expanded = [Int]()
        var index = 0
        for token in tokens {
            guard token == imageToken else {
                expanded.append(token)
                continue
            }
            guard index < processed.count else {
                throw VLMError.processing("More image placeholders than images")
            }
            expanded.append(contentsOf: Array(repeating: imageToken, count: processed[index].1))
            index += 1
        }
        guard index == processed.count else {
            throw VLMError.processing(
                "Prompt has \(index) image placeholders for \(processed.count) images")
        }
        tokens = expanded

        let pixels = concatenated(processed.map { $0.0.reshaped(-1) })
        let frames = processed.map { THW(1, $0.0.dim(1), $0.0.dim(2)) }
        let prompt = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)),
            image: .init(pixels: pixels, frames: frames))
    }
}
