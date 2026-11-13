// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXImageGeneration
import Testing

@Suite struct FlowMatchEulerSchedulerTests {

    /// `scheduler/scheduler_config.json` of Qwen/Qwen-Image-2.1.
    private static let qwenImage21 = """
        {
          "_class_name": "FlowMatchEulerDiscreteScheduler",
          "base_image_seq_len": 256,
          "base_shift": 0.5,
          "invert_sigmas": false,
          "max_image_seq_len": 8192,
          "max_shift": 0.9,
          "num_train_timesteps": 1000,
          "shift": 1.0,
          "shift_terminal": 0.02,
          "stochastic_sampling": false,
          "time_shift_type": "exponential",
          "use_beta_sigmas": false,
          "use_dynamic_shifting": true,
          "use_exponential_sigmas": false,
          "use_karras_sigmas": false
        }
        """

    private func configuration(_ json: String) throws -> FlowMatchEulerScheduler.Configuration {
        try JSONDecoder().decode(
            FlowMatchEulerScheduler.Configuration.self, from: Data(json.utf8))
    }

    private func expectClose(_ actual: Float, _ expected: Float) {
        #expect(abs(actual - expected) < 1e-5, "\(actual) is not \(expected)")
    }

    // The expected values come from the reference formulas evaluated in double precision:
    // a linear grid from 1 to 1/steps, the exponential shift, then the terminal stretch.
    @Test("A dynamic schedule shifts with the image size and ends at the terminal level")
    func dynamicShift() throws {
        let configuration = try configuration(Self.qwenImage21)

        // 1024 x 1024 pixels: a 64 x 64 latent.
        let square = FlowMatchEulerScheduler(
            configuration, stepCount: 40, imageSequenceLength: 4096)
        #expect(square.stepCount == 40)
        #expect(square.sigmas.count == 41)
        expectClose(square.sigmas[0], 1)
        expectClose(square.sigmas[1], 0.9869637)
        expectClose(square.sigmas[20], 0.6566663)
        expectClose(square.sigmas[38], 0.0678806)
        expectClose(square.sigmas[39], 0.02)
        #expect(square.sigmas[40] == 0)
        #expect(abs(square.timesteps[1] - 986.9637) < 1e-2)
        #expect(square.timesteps.count == 40)

        // 2048 x 2048 pixels: more image tokens keep the noise level higher for longer.
        let large = FlowMatchEulerScheduler(
            configuration, stepCount: 40, imageSequenceLength: 16384)
        expectClose(large.sigmas[1], 0.9926460)
        expectClose(large.sigmas[20], 0.7724376)
        expectClose(large.sigmas[38], 0.1022296)
        expectClose(large.sigmas[39], 0.02)

        let short = FlowMatchEulerScheduler(
            configuration, stepCount: 4, imageSequenceLength: 1024)
        expectClose(short.sigmas[1], 0.7492683)
        expectClose(short.sigmas[2], 0.4325880)
        #expect(short.sigmas.count == 5)
    }

    @Test("A fixed grid without shifting is used as given")
    func fixedGrid() throws {
        // Qwen-Image-2.1-Turbo ships its eight sampling levels as `sample_sigmas`, and its
        // scheduler configuration turns the dynamic shift and the terminal stretch off.
        let turbo: [Float] = [
            1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
        ]
        let configuration = try configuration(
            Self.qwenImage21
                .replacingOccurrences(
                    of: "\"shift_terminal\": 0.02", with: "\"shift_terminal\": null"
                )
                .replacingOccurrences(
                    of: "\"use_dynamic_shifting\": true", with: "\"use_dynamic_shifting\": false"))
        #expect(configuration.shiftTerminal == nil)
        #expect(!configuration.useDynamicShifting)
        let scheduler = FlowMatchEulerScheduler(
            configuration, stepCount: 40, imageSequenceLength: 4096, sigmas: turbo)
        #expect(scheduler.stepCount == 8)
        #expect(scheduler.sigmas == turbo + [0])
        #expect(scheduler.timesteps == turbo.map { $0 * 1000 })
    }

    @Test("A static shift moves every level but the first")
    func staticShift() {
        let scheduler = FlowMatchEulerScheduler(
            .init(shift: 3), stepCount: 4, imageSequenceLength: 1024)
        // 3s / (1 + 2s) for s = 1, 0.75, 0.5, 0.25
        let expected: [Float] = [1, 0.9, 0.75, 0.5, 0]
        for (actual, expected) in zip(scheduler.sigmas, expected) {
            expectClose(actual, expected)
        }
    }

    @Test("A step moves the sample along the velocity by the change in noise level")
    func step() {
        let scheduler = FlowMatchEulerScheduler(
            .init(), stepCount: 3, imageSequenceLength: 1024, sigmas: [1, 0.75, 0.25])
        let sample = MLXArray([1, 2, 3, 4] as [Float]).asType(.float16)
        let velocity = MLXArray([4, -4, 8, 0] as [Float]).asType(.float16)

        // From 1 to 0.75.
        let first = scheduler.step(sample, velocity: velocity, index: 0)
        #expect(first.dtype == .float16)
        #expect(first.asType(.float32).asArray(Float.self) == [0, 3, 1, 4])

        // From 0.75 to 0.25.
        let second = scheduler.step(sample, velocity: velocity, index: 1)
        #expect(second.asType(.float32).asArray(Float.self) == [-1, 4, -1, 4])

        // From 0.25 to the final 0.
        let last = scheduler.step(sample, velocity: velocity, index: 2)
        #expect(last.asType(.float32).asArray(Float.self) == [0, 3, 1, 4])
    }

    @Test("A configuration that asks for an unsupported option is rejected")
    func unsupportedOptions() throws {
        for option in [
            "\"use_karras_sigmas\": true", "\"stochastic_sampling\": true",
            "\"invert_sigmas\": true", "\"time_shift_type\": \"linear\"",
        ] {
            #expect(throws: DecodingError.self) {
                try configuration("{ \(option) }")
            }
        }
        // Absent and false both mean "not requested".
        let plain = try configuration("{ \"use_karras_sigmas\": false }")
        #expect(plain == FlowMatchEulerScheduler.Configuration())
    }
}
