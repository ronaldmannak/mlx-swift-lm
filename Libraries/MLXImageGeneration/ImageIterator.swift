// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// What an image is generated with.
public struct ImageGenerationParameters: Sendable {
    /// The image's width in pixels, a multiple of 32.
    public var width: Int
    /// The image's height in pixels, a multiple of 32.
    public var height: Int
    /// How many denoising steps to take. A checkpoint that ships its own sampling grid uses
    /// that grid's length instead.
    public var stepCount: Int
    /// Above 1, every step also runs the negative prompt and moves away from it by this
    /// factor, which doubles the step's work.
    public var guidanceScale: Float
    public var negativePrompt: String
    /// The same seed, size and prompt give the same image.
    public var seed: UInt64

    public init(
        width: Int = 1024, height: Int = 1024, stepCount: Int = 40, guidanceScale: Float = 1,
        negativePrompt: String = " ", seed: UInt64 = 0
    ) {
        self.width = width
        self.height = height
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.negativePrompt = negativePrompt
        self.seed = seed
    }
}

public enum ImageGenerationError: Error, LocalizedError, Equatable {
    case invalidSize(width: Int, height: Int)
    case invalidStepCount(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidSize(let width, let height):
            "Width and height must be positive multiples of 32, not \(width) x \(height)."
        case .invalidStepCount(let count):
            "The step count must be at least 1, not \(count)."
        }
    }
}

/// What one ``ImageIterator/next()`` call completed.
public enum ImageGenerationEvent {
    /// The text encoder has run on the prompt.
    case promptEncoded
    /// One denoising step has run.
    case denoised(step: Int, total: Int)
    /// The image is decoded: RGBA bytes, `[height, width, 4]`. This is the last event.
    case image(MLXArray)
}

/// Generates one image, a bounded piece of work at a time.
///
/// Each ``next()`` runs one stage to completion on the GPU and then returns: first the text
/// encoder, then one denoising step per call, then the image decoder. Nothing is left running
/// between calls, so the caller chooses what else uses the GPU in between.
///
/// ```swift
/// var iterator = try ImageIterator(model: model, prompt: "a red fox in fresh snow")
/// while let event = iterator.next() {
///     if case .image(let pixels) = event { ... }
/// }
/// ```
public struct ImageIterator: Sequence, IteratorProtocol {
    private let model: QwenImageModel
    private let prompt: String
    private let parameters: ImageGenerationParameters
    private let scheduler: FlowMatchEulerScheduler
    private let latentHeight: Int
    private let latentWidth: Int

    private var textStates: MLXArray?
    private var negativeTextStates: MLXArray?
    private var latents: MLXArray?
    private var completedSteps = 0
    private var isFinished = false

    /// Checks the parameters and builds the schedule. No model runs until ``next()``.
    public init(
        model: QwenImageModel, prompt: String, parameters: ImageGenerationParameters = .init()
    ) throws {
        guard parameters.width > 0, parameters.height > 0, parameters.width % 32 == 0,
            parameters.height % 32 == 0
        else {
            throw ImageGenerationError.invalidSize(
                width: parameters.width, height: parameters.height)
        }
        guard parameters.stepCount >= 1 else {
            throw ImageGenerationError.invalidStepCount(parameters.stepCount)
        }
        self.model = model
        self.prompt = prompt
        self.parameters = parameters
        latentHeight = parameters.height / 16
        latentWidth = parameters.width / 16
        scheduler = FlowMatchEulerScheduler(
            model.schedulerConfiguration, stepCount: parameters.stepCount,
            imageSequenceLength: latentHeight * latentWidth, sigmas: model.sampleSigmas)
    }

    /// The number of denoising steps this image takes.
    public var stepCount: Int { scheduler.stepCount }

    public mutating func next() -> ImageGenerationEvent? {
        guard !isFinished else { return nil }
        guard let textStates, let latents else {
            encodePrompt()
            return .promptEncoded
        }
        guard completedSteps == scheduler.stepCount else {
            denoise(latents, textStates: textStates)
            return .denoised(step: completedSteps, total: scheduler.stepCount)
        }
        isFinished = true
        return .image(decode(latents))
    }

    private mutating func encodePrompt() {
        let textStates = model.encode(prompt)
        let negativeTextStates =
            parameters.guidanceScale > 1 ? model.encode(parameters.negativePrompt) : nil
        let channels = model.decoder.configuration.latentChannels
        // Drawn channels first and then moved last, so a seed gives the noise it gives in the
        // reference implementations.
        let noise = withRandomState(MLXRandom.RandomState(seed: parameters.seed)) {
            MLXRandom.normal([1, 1, channels, latentHeight, latentWidth])
        }
        let latents = noise.asType(textStates.dtype)
            .reshaped(1, channels, latentHeight * latentWidth).transposed(0, 2, 1)
        eval([textStates, latents] + (negativeTextStates.map { [$0] } ?? []))
        self.textStates = textStates
        self.negativeTextStates = negativeTextStates
        self.latents = latents
    }

    private mutating func denoise(_ latents: MLXArray, textStates: MLXArray) {
        // The model takes the timestep rounded to its own precision, then scaled to 0 ... 1.
        let noiseLevel =
            MLXArray([scheduler.timesteps[completedSteps]]).asType(latents.dtype) / 1000
        func velocity(_ states: MLXArray) -> MLXArray {
            model.transformer(
                latents, textStates: states, noiseLevel: noiseLevel, height: latentHeight,
                width: latentWidth)
        }
        var prediction = velocity(textStates)
        if let negativeTextStates {
            let negative = velocity(negativeTextStates)
            prediction = negative + parameters.guidanceScale * (prediction - negative)
        }
        let next = scheduler.step(latents, velocity: prediction, index: completedSteps)
        eval(next)
        self.latents = next
        completedSteps += 1
    }

    private func decode(_ latents: MLXArray) -> MLXArray {
        let pixels = model.decoder(latents.reshaped(1, latentHeight, latentWidth, -1))
        let bytes = MLX.round((pixels[0] + 1) * 127.5).asType(.uint8)
        eval(bytes)
        return bytes
    }
}
