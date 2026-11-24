// Copyright © 2026 Apple Inc.

import Foundation

extension ResolvedModelConfiguration {
    /// Patches only the data used for model construction; checkpoint files stay untouched.
    package func applyingInkling2CacheMode(to data: Data, modelType: String) throws -> Data {
        guard let mode = inkling2RecurrentKVCacheMode else { return data }
        guard modelType == "inkling2" || modelType == "inkling2_loop" else {
            guard mode != .lastLoop else {
                throw ModelFactoryError.invalidConfiguration(
                    "lastLoop cache mode requires an Inkling 2 model with repeated layers")
            }
            return data
        }

        var configuration = try JSONDecoder.json5().decode([String: JSONValue].self, from: data)
        guard case .object(var text)? = configuration["text_config"] else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 cache-mode overrides require text_config")
        }
        if mode == .lastLoop {
            guard case .object(let loop)? = text["loop_config"],
                case .int(let passes)? = loop["num_passes"], passes > 1
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "lastLoop cache mode requires an Inkling 2 model with repeated layers")
            }
        }
        text["recurrent_kv_cache_mode"] = .string(mode.rawValue)
        configuration["text_config"] = .object(text)
        return try JSONEncoder().encode(configuration)
    }
}
