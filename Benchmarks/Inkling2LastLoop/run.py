#!/usr/bin/env python3
"""Plan or run a serial ABBA comparison; execution is explicit."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--binary', type=Path, help='Defaults to the Xcode Release executable')
    parser.add_argument('--lengths', default='2048,8192')
    parser.add_argument('--decode', type=int, default=128)
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--chat-fixtures', type=Path,
                        help='Optional JSON list containing name and tokens fields')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.decode <= 0 or args.chunk <= 0:
        parser.error('--decode and --chunk must be positive')
    try:
        lengths = [int(value) for value in args.lengths.split(',')]
    except ValueError:
        parser.error('--lengths must be a comma-separated list of positive integers')
    if not lengths or any(value <= 0 for value in lengths):
        parser.error('--lengths must contain positive integers')
    binary = (args.binary or workspace / 'dd/Build/Products/Release/inklingbench').resolve()
    jobs = [{'mode': mode, 'phase': 'synthetic', 'lengths': args.lengths,
             'decode': args.decode, 'chunk': args.chunk}
            for mode in ['per_loop', 'last_loop', 'last_loop', 'per_loop']]
    if args.chat_fixtures:
        fixtures = json.loads(args.chat_fixtures.read_text())
        counts = [len(row['tokens']) for row in fixtures]
        if len(set(counts)) != len(counts):
            parser.error('Chat fixture token lengths must be distinct (the harness selects by length)')
        for mode in ['per_loop', 'last_loop']:
            jobs.append({'mode': mode, 'phase': 'chat', 'lengths': ','.join(map(str, counts)),
                         'decode': args.decode, 'chunk': args.chunk})
    plan = {'binary': str(binary), 'workspace': str(workspace), 'jobs': jobs}
    if not args.execute:
        print(json.dumps({'gpuWorkPerformed': False, **plan}, indent=2))
        return
    if not binary.is_file():
        parser.error('Build the benchmark executable first: ' + str(binary))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = workspace / 'results' / stamp
    output.mkdir(parents=True)
    plan['binarySHA256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    if (workspace / 'source.json').is_file():
        (output / 'source.json').write_bytes((workspace / 'source.json').read_bytes())
    records = []
    start = time.monotonic()
    for index, job in enumerate(jobs):
        name = f'{index:02d}-{job["phase"]}-{job["mode"]}'
        command = [str(binary), str(workspace / 'models' / job['mode']), job['lengths'],
                   str(job['decode']), str(job['chunk'])]
        environment = {key: value for key, value in os.environ.items() if key != 'INKLING_CHAT_FIXTURES'}
        if job['phase'] == 'chat':
            environment['INKLING_CHAT_FIXTURES'] = str(args.chat_fixtures.resolve())
        print('Starting ' + name, flush=True)
        with (output / (name + '.jsonl')).open('w') as stdout, (output / (name + '.stderr')).open('w') as stderr:
            subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=True)
        rows = [json.loads(line) for line in (output / (name + '.jsonl')).read_text().splitlines()
                if line.startswith('{')]
        for row in rows:
            row['phase'], row['job'] = job['phase'], index
        records.extend(rows)
        (output / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
        print('Completed ' + name, flush=True)
    comparisons = []
    for phase in ['synthetic', 'chat']:
        left = [r for r in records if r.get('phase') == phase and r.get('kind') == 'run' and r['mode'] == 'per_loop']
        right = [r for r in records if r.get('phase') == phase and r.get('kind') == 'run' and r['mode'] == 'last_loop']
        for a, b in zip(left, right):
            assert a['promptName'] == b['promptName'] and a['promptTokens'] == b['promptTokens']
            comparisons.append({
                'phase': phase, 'prompt': a['promptName'], 'promptTokens': a['promptTokens'],
                'sameTokens': a['tokens'] == b['tokens'],
                'firstDifferentIndex': next((i for i, pair in enumerate(zip(a['tokens'], b['tokens']))
                                             if pair[0] != pair[1]), None),
                'perLoopCacheBytes': a['cacheBytes'], 'lastLoopCacheBytes': b['cacheBytes'],
                'cacheReductionFraction': 1 - b['cacheBytes'] / a['cacheBytes'],
                'attentionReductionFraction': 1 - b['attentionBytes'] / a['attentionBytes'],
                'perLoopDecodeTokensPerSecond': a['decodeTokensPerSecond'],
                'lastLoopDecodeTokensPerSecond': b['decodeTokensPerSecond'],
                'perLoopPeakBytes': a['peakBytes'], 'lastLoopPeakBytes': b['peakBytes'],
            })
    summary = {'binarySHA256': plan['binarySHA256'], 'elapsedSeconds': time.monotonic() - start,
               'comparisons': comparisons, 'output': str(output)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
