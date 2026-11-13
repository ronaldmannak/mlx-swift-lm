// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The noise schedule and Euler update of a flow-matching image model.
///
/// Port of diffusers' `FlowMatchEulerDiscreteScheduler`, limited to the deterministic
/// sampling that Qwen-Image-2.1 uses.
public struct FlowMatchEulerScheduler: Sendable {

    /// The scheduler's `scheduler_config.json`.
    public struct Configuration: Codable, Sendable, Equatable {
        public var baseImageSequenceLength: Int
        public var maxImageSequenceLength: Int
        public var baseShift: Float
        public var maxShift: Float
        public var shift: Float
        /// The noise level the last step starts from, or `nil` to leave the schedule as it is.
        public var shiftTerminal: Float?
        /// Shifts the schedule by the image's token count instead of by ``shift``.
        public var useDynamicShifting: Bool
        public var trainTimestepCount: Int

        public init(
            baseImageSequenceLength: Int = 256,
            maxImageSequenceLength: Int = 4096,
            baseShift: Float = 0.5,
            maxShift: Float = 1.15,
            shift: Float = 1.0,
            shiftTerminal: Float? = nil,
            useDynamicShifting: Bool = false,
            trainTimestepCount: Int = 1000
        ) {
            self.baseImageSequenceLength = baseImageSequenceLength
            self.maxImageSequenceLength = maxImageSequenceLength
            self.baseShift = baseShift
            self.maxShift = maxShift
            self.shift = shift
            self.shiftTerminal = shiftTerminal
            self.useDynamicShifting = useDynamicShifting
            self.trainTimestepCount = trainTimestepCount
        }

        enum CodingKeys: String, CodingKey {
            case baseImageSequenceLength = "base_image_seq_len"
            case maxImageSequenceLength = "max_image_seq_len"
            case baseShift = "base_shift"
            case maxShift = "max_shift"
            case shift
            case shiftTerminal = "shift_terminal"
            case useDynamicShifting = "use_dynamic_shifting"
            case trainTimestepCount = "num_train_timesteps"
        }

        /// Options this port does not implement. A checkpoint that sets one is rejected.
        private enum UnsupportedKeys: String, CodingKey, CaseIterable {
            case useKarrasSigmas = "use_karras_sigmas"
            case useExponentialSigmas = "use_exponential_sigmas"
            case useBetaSigmas = "use_beta_sigmas"
            case invertSigmas = "invert_sigmas"
            case stochasticSampling = "stochastic_sampling"
        }

        private enum ShiftTypeKey: String, CodingKey {
            case timeShiftType = "time_shift_type"
        }

        public init(from decoder: Decoder) throws {
            let defaults = Configuration()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            baseImageSequenceLength =
                try container.decodeIfPresent(Int.self, forKey: .baseImageSequenceLength)
                ?? defaults.baseImageSequenceLength
            maxImageSequenceLength =
                try container.decodeIfPresent(Int.self, forKey: .maxImageSequenceLength)
                ?? defaults.maxImageSequenceLength
            baseShift =
                try container.decodeIfPresent(Float.self, forKey: .baseShift) ?? defaults.baseShift
            maxShift =
                try container.decodeIfPresent(Float.self, forKey: .maxShift) ?? defaults.maxShift
            shift = try container.decodeIfPresent(Float.self, forKey: .shift) ?? defaults.shift
            shiftTerminal = try container.decodeIfPresent(Float.self, forKey: .shiftTerminal)
            useDynamicShifting =
                try container.decodeIfPresent(Bool.self, forKey: .useDynamicShifting)
                ?? defaults.useDynamicShifting
            trainTimestepCount =
                try container.decodeIfPresent(Int.self, forKey: .trainTimestepCount)
                ?? defaults.trainTimestepCount

            let unsupported = try decoder.container(keyedBy: UnsupportedKeys.self)
            for key in UnsupportedKeys.allCases
            where try unsupported.decodeIfPresent(Bool.self, forKey: key) == true {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: unsupported,
                    debugDescription: "\(key.rawValue) is not supported")
            }
            let shiftType = try decoder.container(keyedBy: ShiftTypeKey.self)
            if let type = try shiftType.decodeIfPresent(String.self, forKey: .timeShiftType),
                type != "exponential"
            {
                throw DecodingError.dataCorruptedError(
                    forKey: .timeShiftType, in: shiftType,
                    debugDescription: "time_shift_type \(type) is not supported")
            }
        }
    }

    /// The noise level each step starts from, followed by the final 0.
    public let sigmas: [Float]

    private let trainTimestepCount: Int

    /// The number of denoising steps.
    public var stepCount: Int { sigmas.count - 1 }

    /// The timestep the model is given at each step.
    public var timesteps: [Float] { sigmas.dropLast().map { $0 * Float(trainTimestepCount) } }

    /// - Parameters:
    ///   - configuration: the checkpoint's scheduler configuration
    ///   - stepCount: the number of denoising steps, ignored when `sigmas` is given
    ///   - imageSequenceLength: how many image tokens the model denoises, which sets the
    ///     dynamic shift
    ///   - sigmas: a fixed grid to sample on, such as a distilled checkpoint's `sample_sigmas`
    public init(
        _ configuration: Configuration, stepCount: Int, imageSequenceLength: Int,
        sigmas: [Float]? = nil
    ) {
        precondition(sigmas?.isEmpty == false || stepCount > 0, "A schedule needs a step")
        var grid: [Double]
        if let sigmas {
            grid = sigmas.map(Double.init)
        } else {
            let last = 1 / Double(stepCount)
            grid = (0 ..< stepCount).map { index in
                stepCount == 1 ? 1 : 1 + Double(index) * (last - 1) / Double(stepCount - 1)
            }
        }

        if configuration.useDynamicShifting {
            let slope =
                Double(configuration.maxShift - configuration.baseShift)
                / Double(
                    configuration.maxImageSequenceLength - configuration.baseImageSequenceLength)
            let mu =
                Double(configuration.baseShift)
                + slope * Double(imageSequenceLength - configuration.baseImageSequenceLength)
            grid = grid.map { exp(mu) / (exp(mu) + (1 / $0 - 1)) }
        } else {
            let shift = Double(configuration.shift)
            grid = grid.map { shift * $0 / (1 + (shift - 1) * $0) }
        }

        // A grid that ends at 1, such as a single step, has no span to stretch.
        if let terminal = configuration.shiftTerminal, terminal != 0, let last = grid.last,
            last < 1
        {
            let scale = (1 - last) / (1 - Double(terminal))
            grid = grid.map { 1 - (1 - $0) / scale }
        }

        self.sigmas = grid.map(Float.init) + [0]
        self.trainTimestepCount = configuration.trainTimestepCount
    }

    /// Moves `sample` from step `index`'s noise level to the next one along `velocity`.
    public func step(_ sample: MLXArray, velocity: MLXArray, index: Int) -> MLXArray {
        let delta = sigmas[index + 1] - sigmas[index]
        return (sample.asType(.float32) + delta * velocity.asType(.float32)).asType(velocity.dtype)
    }
}
