"""Descriptive seed distributions and task-level paired bootstrap intervals.

Refuses failed, unmatched or unequally budgeted comparisons. Confidence intervals
are descriptive for the supplied instance sample, not a universal superiority claim.
"""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def bootstrap_mean(values, rnd, repetitions=2000):
    if len(values) < 2:
        return None
    means = sorted(statistics.mean(rnd.choices(values, k=len(values))) for _ in range(repetitions))
    return [means[int(.025 * repetitions)], means[int(.975 * repetitions)]]


def analyze(directory, bootstrap_seed=42):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = directory / manifest["options"]["combined_output"]
    with summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["status"] != "success" for row in rows):
        raise ValueError("Summary is empty or contains failures; inspect failures before analysis")
    if manifest["budget_mode"] == "time_with_evaluation_cap" and any(row["stop_reason"] != "time" for row in rows):
        raise ValueError("Time comparison reached its evaluation cap; increase the cap")
    seen, groups = set(), defaultdict(list)
    for row in rows:
        key = row["instance"], row["algorithm"], int(row["seed"])
        if key in seen:
            raise ValueError("Duplicate run")
        seen.add(key)
        groups[key[:2]].append(row)
    options = manifest["options"]
    expected = {(instance, algorithm, seed) for instance in manifest["instances"]
                for algorithm in set(options["algorithms"])
                for seed in range(options["seed_start"], options["seed_start"] + options["seeds"])}
    if seen != expected:
        raise ValueError("Run coverage does not match manifest")
    rnd = random.Random(bootstrap_seed)
    distributions = []
    for (instance, algorithm), observations in sorted(groups.items()):
        observations.sort(key=lambda row: int(row["seed"]))
        costs = [int(row["best_makespan"]) for row in observations]
        distributions.append({"instance": instance, "algorithm": algorithm, "runs": len(costs),
                              "seeds": [int(row["seed"]) for row in observations],
                              "makespans": costs, "mean": statistics.mean(costs),
                              "median": statistics.median(costs), "min": min(costs), "max": max(costs),
                              "sample_std": statistics.stdev(costs) if len(costs) > 1 else None,
                              "seed_bootstrap_mean_ci95": bootstrap_mean(costs, rnd)})
    paired = []
    algorithms = {row["algorithm"] for row in rows}
    if algorithms == {"baseline", "meta"}:
        for instance in sorted({row["instance"] for row in rows}):
            base = {int(row["seed"]): row for row in groups[(instance, "baseline")]}
            meta = {int(row["seed"]): row for row in groups[(instance, "meta")]}
            if not base or base.keys() != meta.keys():
                raise ValueError("Algorithms have unmatched instance/seed coverage")
            differences = []
            for seed in sorted(base):
                if manifest["budget_mode"] == "evaluations" and base[seed]["evaluations"] != meta[seed]["evaluations"]:
                    raise ValueError("Unequal evaluation budgets")
                differences.append(int(meta[seed]["best_makespan"]) - int(base[seed]["best_makespan"]))
            base_mean = statistics.mean(int(row["best_makespan"]) for row in base.values())
            difference = statistics.mean(differences)
            paired.append({"instance": instance, "mean_meta_minus_baseline": difference,
                           "relative_difference_percent": difference / base_mean * 100 if base_mean else None,
                           "seed_differences": differences})
    # Instance is the resampling unit; repeated seeds are not treated as new tasks.
    task_differences = [row["relative_difference_percent"] for row in paired
                        if row["relative_difference_percent"] is not None]
    comparison = None if not paired else {
        "tasks": len(paired), "negative_favors": "meta",
        "mean_task_relative_difference_percent": statistics.mean(task_differences) if task_differences else None,
        "task_bootstrap_ci95": bootstrap_mean(task_differences, rnd) if task_differences else None,
        "wins_ties_losses_meta": [sum(row["mean_meta_minus_baseline"] < 0 for row in paired),
                                 sum(row["mean_meta_minus_baseline"] == 0 for row in paired),
                                 sum(row["mean_meta_minus_baseline"] > 0 for row in paired)]}
    return {"bootstrap_seed": bootstrap_seed, "bootstrap_repetitions": 2000,
            "distributions": distributions, "paired_tasks": paired, "comparison": comparison,
            "limitations": ["Intervals describe the supplied sample and assume exchangeable resampling units",
                             "Small or nonrepresentative task samples do not support general superiority claims",
                             "Identical seed labels do not imply identical algorithm trajectories",
                             "No multiplicity correction or hypothesis-testing claim is made"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(args.directory, args.bootstrap_seed)
    path = Path(args.directory) / "analysis.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(str(path))


if __name__ == "__main__":
    main()
