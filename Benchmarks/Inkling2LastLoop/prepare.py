#!/usr/bin/env python3
"""Create an isolated benchmark package and two config views of the same weights."""
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(path):
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--backend', type=Path, required=True, help='Local mlx-swift package')
    parser.add_argument('--model', type=Path, required=True, help='Local Mini checkpoint directory')
    parser.add_argument('--workspace', type=Path, required=True, help='New temporary output directory')
    args = parser.parse_args()
    source, backend, model, workspace = (p.resolve() for p in
                                        (args.source, args.backend, args.model, args.workspace))
    for path in [source / 'Libraries/MLXLMCommon', backend / 'Package.swift', model / 'config.json']:
        if not path.exists():
            parser.error('Missing input: ' + str(path))
    if workspace.exists():
        parser.error('Workspace already exists; choose a new directory to preserve previous results')
    workspace.mkdir(parents=True)
    view = workspace / 'view'
    view.mkdir()
    for name in ['Libraries', 'Tests']:
        shutil.copytree(source / name, view / name, symlinks=True)
    harness = Path(__file__).with_name('benchmark.swift')
    (view / 'BenchHarness').mkdir()
    shutil.copy2(harness, view / 'BenchHarness/main.swift')
    tests = sorted(p.name for p in (view / 'Tests/MLXLMTests').glob('*Inkling*.swift'))
    tests += ['TensorParallelTestSupport.swift']
    common = '[.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXOptimizers", package: "mlx-swift")]'
    manifest = '''// swift-tools-version: 6.2
import PackageDescription
let package = Package(
    name: "InklingBench", platforms: [.macOS(.v14)],
    dependencies: [.package(name: "mlx-swift", path: BACKEND)],
    targets: [
        .target(name: "MLXLMCommon", dependencies: COMMON, path: "Libraries/MLXLMCommon", exclude: ["README.md"]),
        .target(name: "MLXLLM", dependencies: ["MLXLMCommon"] + COMMON, path: "Libraries/MLXLLM", exclude: ["README.md"]),
        .target(name: "MLXVLM", dependencies: ["MLXLMCommon"] + COMMON, path: "Libraries/MLXVLM", exclude: ["README.md"]),
        .executableTarget(name: "inklingbench", dependencies: ["MLXLMCommon", "MLXLLM", "MLXVLM", .product(name: "MLX", package: "mlx-swift")], path: "BenchHarness"),
        .testTarget(name: "MLXLMTests", dependencies: ["MLXLMCommon", "MLXLLM", "MLXVLM"] + COMMON, path: "Tests/MLXLMTests", sources: TESTS),
    ], cxxLanguageStandard: .cxx17)
'''.replace('BACKEND', json.dumps(str(backend))).replace('COMMON', common).replace('TESTS', json.dumps(tests))
    (view / 'Package.swift').write_text(manifest)
    for mode in ['per_loop', 'last_loop']:
        destination = workspace / 'models' / mode
        destination.mkdir(parents=True)
        for path in model.iterdir():
            if path.name == 'config.json':
                config = json.loads(path.read_text())
                config['text_config']['recurrent_kv_cache_mode'] = mode
                (destination / path.name).write_text(json.dumps(config, indent=2) + '\n')
            else:
                (destination / path.name).symlink_to(path)
    hashes = {str(p.relative_to(view)): sha(p)
              for name in ['Libraries', 'Tests'] for p in sorted((view / name).rglob('*')) if p.is_file()}
    hashes['BenchHarness/main.swift'] = sha(view / 'BenchHarness/main.swift')
    hashes['Package.swift'] = sha(view / 'Package.swift')
    metadata = {
        'preparedUTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': str(source), 'sourceHEAD': git_head(source), 'hashes': hashes,
        'backend': str(backend), 'backendHEAD': git_head(backend),
        'backendManifestSHA256': sha(backend / 'Package.swift'),
        'model': str(model), 'modelConfigSHA256': sha(model / 'config.json'),
        'harnessSHA256': sha(harness), 'tests': tests,
    }
    (workspace / 'source.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'workspace': str(workspace), 'package': str(view),
                      'sourceHEAD': metadata['sourceHEAD'], 'filesHashed': len(hashes),
                      'gpuWorkPerformed': False}, indent=2))


if __name__ == '__main__':
    main()
