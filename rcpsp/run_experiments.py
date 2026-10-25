"""Isolated RCPSP comparisons with explicit budgets and schedule certificates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from .core import Schedule, load_psplib_instance, validate_schedule, validate_aco_params
    from .baseline_aco import ACOParams
    from .references import load_references, quality_metrics
except ImportError:
    from core import Schedule, load_psplib_instance, validate_schedule, validate_aco_params
    from baseline_aco import ACOParams
    from references import load_references, quality_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_REF_FILE = HERE / "RCPLIB (Parameters and BKS).xlsx"
FIELDS = ["instance", "algorithm", "seed", "status", "best_makespan", "evaluations",
          "aco_evaluations", "ls_evaluations", "search_runtime_sec", "process_runtime_sec",
          "stop_reason", "final_colonies", "elite_archive_size", "gap_to_optimum_percent",
          "deviation_to_bks_percent", "gap_to_lb_percent", "improves_bks", "result_file", "error"]


def parse_output(output):
    """Legacy logs only. New experiments use JSON, never human-readable stdout."""
    labels = {"Best makespan": None, "Final colonies count": None, "Elite archive size": None}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in labels:
            labels[key.strip()] = int(value.strip())
    if any(value is None for value in labels.values()):
        raise ValueError("Incomplete legacy solver output")
    return tuple(labels.values())


def collect_instances(instance, instance_dir, pattern):
    files = [Path(instance)] if instance else sorted(Path(instance_dir).glob(pattern))
    if not files or any(not path.is_file() for path in files):
        raise ValueError("No valid instance files found")
    files = [path.resolve(strict=True) for path in files]
    if len({path.name.lower() for path in files}) != len(files):
        raise ValueError("Duplicate instance names")
    return files


def load_psplib_reference(ref_file, instance_name):
    references = load_references(ref_file, [instance_name])
    if instance_name.lower() not in references:
        raise ValueError("Missing reference instance: " + instance_name)
    return references[instance_name.lower()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def validate_result(result, config, instance_path):
    if result.get("schema_version") != 1 or result.get("status") != "success":
        raise ValueError("Invalid worker result schema/status")
    if result.get("config") != config or result.get("instance_sha256") != sha256(instance_path):
        raise ValueError("Worker result does not match requested run")
    evaluations = result.get("evaluations")
    if type(evaluations) is not int or not 1 <= evaluations <= config["evaluations"]:
        raise ValueError("Invalid evaluation count")
    counts = result.get("evaluation_counts", {})
    if (set(counts) != {"aco", "ls"} or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != evaluations):
        raise ValueError("Invalid evaluation category counts")
    if config["time_limit"] is None and evaluations != config["evaluations"]:
        raise ValueError("Unused evaluation budget; raise --iterations")
    if result.get("stop_reason") not in {"evaluations", "time"}:
        raise ValueError("Run stopped before its declared resource limit; raise --iterations")
    for field in ("search_runtime_sec", "setup_runtime_sec"):
        value = result.get(field)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid runtime")
    colonies, elite = result.get("final_colonies"), result.get("elite_archive_size")
    if type(colonies) is not int or colonies < 1 or type(elite) is not int or elite < 0:
        raise ValueError("Invalid colony metadata")
    if len(result.get("colonies", [])) != colonies:
        raise ValueError("Inconsistent colony metadata")
    schedule = Schedule(**result["schedule"])
    validate_schedule(load_psplib_instance(str(instance_path)), schedule)
    history = result.get("history", [])
    if not history or history[-1]["makespan"] != schedule.makespan:
        raise ValueError("Inconsistent incumbent history")
    previous_eval, previous_cost = 0, math.inf
    for point in history:
        if not previous_eval < point["evaluation"] <= evaluations or not point["makespan"] < previous_cost:
            raise ValueError("Invalid incumbent history")
        previous_eval, previous_cost = point["evaluation"], point["makespan"]
    return schedule


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance")
    parser.add_argument("--instance-dir", default=str(ROOT / "psplib"))
    parser.add_argument("--pattern", default="*.sm")
    parser.add_argument("--ref-file", default=str(DEFAULT_REF_FILE))
    parser.add_argument("--no-reference", action="store_true", help="Explicitly omit quality gaps")
    parser.add_argument("--algorithms", nargs="+", choices=["baseline", "meta"], default=["baseline", "meta"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--evaluations", type=int, default=10000)
    parser.add_argument("--iterations", type=int, help="Optional iteration cap; unused evaluation budget is an error")
    parser.add_argument("--time-limit", type=float, help="Cooperative search deadline in seconds")
    parser.add_argument("--timeout", type=float, help="Hard process timeout; recorded as failure")
    parser.add_argument("--ants", type=int, default=20)
    parser.add_argument("--init-colonies", type=int, default=3)
    parser.add_argument("--max-colonies", type=int, default=10)
    parser.add_argument("--selection", choices=["ucb", "random", "round_robin"], default="ucb")
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--ucb-c", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["global_improvement", "quality"], default="global_improvement")
    parser.add_argument("--replication-mode", choices=["mutate", "restart"], default="mutate")
    parser.add_argument("--replication-threshold", type=float, default=0.02)
    parser.add_argument("--elite-strength", type=float, default=0.1)
    parser.add_argument("--disable-replication", action="store_true")
    parser.add_argument("--disable-elite", action="store_true")
    parser.add_argument("--disable-ls", action="store_true", help="Disable meta local search")
    parser.add_argument("--baseline-ls-iters", type=int, default=0)
    parser.add_argument("--baseline-ls-prob", type=float, default=0.0)
    parser.add_argument("--baseline-heuristic", choices=["none", "min_duration", "combo"], default="min_duration")
    parser.add_argument("--baseline-params-file", help="Full ACOParams JSON for separately tuned baseline")
    parser.add_argument("--output-dir", help="Must be new; previous experiments are preserved")
    parser.add_argument("--combined-output", default="summary.csv")
    return parser


def checked_args(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    for field in ("seeds", "evaluations", "ants", "init_colonies", "max_colonies"):
        if getattr(args, field) < 1:
            parser.error(field + " must be positive")
    if args.iterations is not None and args.iterations < 1:
        parser.error("iterations must be positive")
    if args.init_colonies > args.max_colonies:
        parser.error("init-colonies exceeds max-colonies")
    for field in ("time_limit", "timeout"):
        value = getattr(args, field)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(field + " must be finite and positive")
    for field in ("discount", "ucb_c", "replication_threshold", "elite_strength"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            parser.error(field + " must be finite and nonnegative")
    if not 0 < args.discount <= 1 or not 0 <= args.replication_threshold <= 1 or args.elite_strength > 1:
        parser.error("discount/threshold/elite strength outside permitted range")
    if Path(args.combined_output).name != args.combined_output:
        parser.error("combined-output must be a file name")
    baseline_parameters(args)
    return args


def baseline_parameters(args):
    params = (ACOParams(**json.loads(Path(args.baseline_params_file).read_text())) if args.baseline_params_file else
              ACOParams(heuristic_type=args.baseline_heuristic,
                        ls_iters=args.baseline_ls_iters, ls_prob=args.baseline_ls_prob))
    validate_aco_params(params)
    if not math.isfinite(params.tau0) or not 0 < params.tau0 <= 1:
        raise ValueError("Baseline tau0 must lie in (0, 1]")
    return params


def run_one(output_dir, instance, config, reference, timeout, source_hashes):
    name = f"{instance.stem}_{config['algorithm']}_seed{config['seed']}"
    config_path = output_dir / "runs" / (name + ".config.json")
    result_path = output_dir / "runs" / (name + ".json")
    write_json(config_path, config)
    row = {"instance": instance.name, "algorithm": config["algorithm"], "seed": config["seed"],
           "status": "error", "result_file": str(result_path.relative_to(output_dir))}
    stdout, stderr = "", ""
    started = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, str(output_dir / "sources" / "experiment_worker.py"),
                               str(instance), "--config", str(config_path), "--output", str(result_path)],
                              capture_output=True, text=True, timeout=timeout)
        stdout, stderr = proc.stdout, proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"Worker exited with code {proc.returncode}: {stderr[-2000:]}")
        result = json.loads(result_path.read_text())
        if result.get("sources_sha256") != source_hashes:
            raise ValueError("Worker source hashes do not match manifest")
        schedule = validate_result(result, config, instance)
        metrics = quality_metrics(schedule.makespan, reference)
        result.update(reference=reference, quality_metrics=metrics, driver_validation="passed")
        write_json(result_path, result)
        row.update(metrics)
        row.update(status="success", best_makespan=schedule.makespan, evaluations=result["evaluations"],
                   aco_evaluations=result["evaluation_counts"]["aco"], ls_evaluations=result["evaluation_counts"]["ls"],
                   search_runtime_sec=result["search_runtime_sec"], stop_reason=result["stop_reason"],
                   final_colonies=result["final_colonies"], elite_archive_size=result["elite_archive_size"])
    except subprocess.TimeoutExpired as error:
        row.update(status="timeout", error=str(error))
        stdout, stderr = error.stdout or "", error.stderr or ""
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        row["error"] = str(error)
    row["process_runtime_sec"] = time.perf_counter() - started
    write_json(output_dir / "runs" / (name + ".logs.json"),
               {"stdout": stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout,
                "stderr": stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr})
    if row["status"] != "success":
        write_json(output_dir / "runs" / (name + ".failure.json"), row)
        if result_path.exists():
            # Preserve an unaccepted worker result as evidence, with explicit status.
            try:
                rejected = json.loads(result_path.read_text())
                rejected.update(status=row["status"], driver_validation="failed", error=row["error"])
                write_json(result_path, rejected)
            except (ValueError, TypeError, AttributeError):
                pass
    print(name + ": " + row["status"], flush=True)
    return row


def main(argv=None):
    args = checked_args(argv)
    params = baseline_parameters(args)
    instances = collect_instances(args.instance, args.instance_dir, args.pattern)
    references = {} if args.no_reference else load_references(args.ref_file, [p.name for p in instances])
    for instance in instances:
        model = load_psplib_instance(str(instance))
        if not args.no_reference:
            reference = references.get(instance.name.lower())
            if reference is None:
                raise ValueError("Missing reference: " + instance.name)
            if reference["activities"] != model.n_jobs - 2:
                raise ValueError("Reference activity count mismatch: " + instance.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = Path(args.output_dir or ROOT / "experiment_results_v2" / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("runs", "sources", "inputs"):
        (output_dir / name).mkdir()
    source_hashes = {}
    for source in sorted(HERE.glob("*.py")):
        shutil.copy2(source, output_dir / "sources" / source.name)
        source_hashes[source.name] = sha256(source)
    for instance in instances:
        shutil.copy2(instance, output_dir / "inputs" / instance.name)
    # Counterbalance process order across seeds without touching algorithm RNGs.
    algorithms = list(dict.fromkeys(args.algorithms))
    planned = []
    for original in instances:
        for offset, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
            shift = offset % len(algorithms)
            for algorithm in algorithms[shift:] + algorithms[:shift]:
                planned.append((original.name, algorithm, seed))
    write_json(output_dir / "manifest.json",
               {"schema_version": 1, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv if argv is None else argv, "options": vars(args),
                "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "sources_sha256": source_hashes, "instances": {p.name: sha256(p) for p in instances},
                "execution_order": planned,
                "references": references, "reference_sha256": None if args.no_reference else sha256(args.ref_file),
                "budget_mode": "evaluations" if args.time_limit is None else "time_with_evaluation_cap",
                "limitations": ["UCB-like allocation has no stationary UCB1 guarantee",
                                 "The local 15-instance set is a validation subset",
                                 "A cooperative deadline permits in-flight operations to finish"]})
    rows = []
    summary = output_dir / args.combined_output
    with summary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for instance_name, algorithm, seed in planned:
            instance = output_dir / "inputs" / instance_name
            config = {"algorithm": algorithm, "seed": seed, "ants": args.ants,
                              "iterations": args.iterations or args.evaluations, "evaluations": args.evaluations,
                              "time_limit": args.time_limit, "init_colonies": args.init_colonies,
                              "max_colonies": args.max_colonies, "selection": args.selection,
                              "discount": args.discount, "ucb_c": args.ucb_c, "reward_mode": args.reward_mode,
                              "replication_mode": args.replication_mode, "replication": not args.disable_replication,
                              "replication_threshold": args.replication_threshold, "elite": not args.disable_elite,
                              "elite_strength": args.elite_strength, "local_search": not args.disable_ls,
                              "baseline_params": asdict(params)}
            row = run_one(output_dir, instance, config, references.get(instance.name.lower()),
                          args.timeout, source_hashes)
            writer.writerow(row)
            stream.flush()
            rows.append(row)
    print("Results: " + str(summary))
    return int(any(row["status"] != "success" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
