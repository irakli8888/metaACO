"""One isolated run. The driver validates this JSON and its schedule again."""
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

try:
    from .core import load_psplib_instance, validate_schedule
    from .baseline_aco import ACOParams, BaselineACO
    from .metaACO import ColonyManager
except ImportError:
    from core import load_psplib_instance, validate_schedule
    from baseline_aco import ACOParams, BaselineACO
    from metaACO import ColonyManager


def run_experiment(instance_path, config):
    setup_started = time.perf_counter()
    inst = load_psplib_instance(str(instance_path))
    algorithm = config["algorithm"]
    if algorithm == "baseline":
        params = ACOParams(**config["baseline_params"])
        solver = BaselineACO(inst, params, random.Random(config["seed"]),
                             ants_per_iteration=config["ants"],
                             max_evaluations=config["evaluations"], time_limit=config["time_limit"])
    elif algorithm == "meta":
        solver = ColonyManager(inst, random.Random(config["seed"]),
                               max_colonies=config["max_colonies"],
                               max_evaluations=config["evaluations"], time_limit=config["time_limit"],
                               replication_enabled=config["replication"], elite_enabled=config["elite"],
                               local_search_enabled=config["local_search"], selection=config["selection"],
                               discount=config["discount"], reward_mode=config["reward_mode"],
                               replication_mode=config["replication_mode"], ucb_c=config["ucb_c"],
                               elite_strength=config["elite_strength"])
        solver.init_random_colonies(config["init_colonies"])
    else:
        raise ValueError("Unknown algorithm")
    setup_runtime = time.perf_counter() - setup_started
    started = time.perf_counter()
    if algorithm == "baseline":
        schedule = solver.run(config["iterations"])
        colonies = [{"params": asdict(params), "best_cost": solver.best_cost}]
        events, elite_size = [], 0
    else:
        schedule, _ = solver.run(config["iterations"], config["ants"], config["replication_threshold"])
        colonies = [{"params": asdict(col.params),
                     "best_cost": col.best_cost if col.best_schedule is not None else None,
                     "pulls": solver.bandit.pulls[i], "discounted_reward_sum": solver.bandit.rewards[i]}
                    for i, col in enumerate(solver.bandit.colonies)]
        events, elite_size = solver.events, len(solver.elite.entries)
    search_runtime = time.perf_counter() - started
    if schedule is None:
        raise RuntimeError("Search ended without a schedule")
    validate_schedule(inst, schedule)
    if solver.budget.best_cost != schedule.makespan:
        raise RuntimeError("Returned schedule disagrees with evaluated incumbent")
    directory = Path(__file__).resolve().parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(directory.glob("*.py"))}
    return {"schema_version": 1, "status": "success", "config": config,
            "instance_sha256": hashlib.sha256(Path(instance_path).read_bytes()).hexdigest(),
            "sources_sha256": sources, "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "schedule": asdict(schedule),
            "evaluations": solver.budget.used, "evaluation_counts": solver.budget.counts,
            "history": solver.budget.history, "stop_reason": solver.budget.stop_reason or "iterations",
            "search_runtime_sec": search_runtime, "setup_runtime_sec": setup_runtime,
            "final_colonies": len(colonies), "elite_archive_size": elite_size,
            "colonies": colonies, "events": events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = run_experiment(Path(args.instance), config)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
