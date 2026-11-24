// swift-tools-version: 6.0
import PackageDescription

let library = Context.environment["MLX_SWIFT_LM_PATH"] ?? "../.."

let package = Package(
    name: "DFlashBatchBench",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "mlx-swift-lm", path: library)],
    targets: [
        .executableTarget(
            name: "dflashbatch",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
            ])
    ]
)
