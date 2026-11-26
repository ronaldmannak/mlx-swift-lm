"""Summarize measured runs, excluding environment records and full-length warmups."""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def median(values):
    values = [value for value in values if value is not None]
    return round(statistics.median(values), 4) if values else None


def gap_p95(run):
    gaps = [b - a for row in run["rows"]
            for a, b in zip(row["deliveryMS"], row["deliveryMS"][1:])]
    return sorted(gaps)[math.ceil(.95 * len(gaps)) - 1] if gaps else None


def summarize(path):
    groups = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        run = json.loads(line)
        if run["kind"] == "run":
            groups[(run["scenario"], run["arm"])].append(run)
    rates = {key: median(run["aggregateTokensPerSecond"] for run in runs)
             for key, runs in groups.items()}
    for (scenario, arm), runs in sorted(groups.items()):
        width = len(runs[0]["rows"])
        rate = rates[(scenario, arm)]
        ordinary = rates.get((scenario, "ordinary"))
        yield {
            "scenario": scenario, "arm": arm, "repetitions": len(runs),
            "aggregateTokensPerSecond": rate,
            "minTokensPerSecond": min(run["aggregateTokensPerSecond"] for run in runs),
            "maxTokensPerSecond": max(run["aggregateTokensPerSecond"] for run in runs),
            "ratioToOrdinary": round(rate / ordinary, 4) if ordinary else None,
            "elapsedMS": median(run["elapsedMS"] for run in runs),
            "firstTokenMSByRow": [median(run["rows"][row].get("firstTokenMS")
                                        for run in runs) for row in range(width)],
            "admissionToFirstMSByRow": [median(
                record["firstTokenMS"] - record["admissionMS"]
                for run in runs for record in [run["rows"][row]]
                if "firstTokenMS" in record and "admissionMS" in record)
                for row in range(width)],
            "finishedMSByRow": [median(run["rows"][row].get("finishedMS")
                                      for run in runs) for row in range(width)],
            "outputTokensByRow": [len(row["tokens"]) for row in runs[0]["rows"]],
            "tokenGapP95MS": median(gap_p95(run) for run in runs),
            "peakGB": median(run["peakGB"] for run in runs),
            "tokensRepeat": all([row["tokens"] for row in run["rows"]]
                                == [row["tokens"] for row in runs[0]["rows"]] for run in runs),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    for result in summarize(parser.parse_args().file):
        print(json.dumps(result, sort_keys=True))
