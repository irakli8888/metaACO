"""Regression checks and independent exhaustive scheduling oracle (stdlib)."""
import copy
import csv
import itertools
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rcpsp.core import (RcpspInstance, Schedule, SerialScheduleGenerator, EvaluationBudget,
                        BudgetExhausted, validate_schedule, load_psplib_instance, graph_features)
from rcpsp.baseline_aco import ACOParams, BaselineACO, Ant as BaselineAnt, PheromoneMatrix
from rcpsp.metaACO import ColonyParams, Ant, AntColony, ColonyManager, ColonyBanditController
from rcpsp.references import load_references, quality_metrics
from rcpsp.run_experiments import DEFAULT_REF_FILE, parse_output, validate_result, main
from rcpsp.analyze_results import analyze

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def instance(durations, demands, successors, capacity=1):
    predecessors = [[i for i, edges in enumerate(successors) if j in edges]
                    for j in range(len(durations))]
    return RcpspInstance(len(durations), 1, durations, [[d] for d in demands],
                         [capacity], successors, predecessors)


def independent_feasible(inst, starts):
    # No production order/schedule validators or SGS are used in this oracle.
    if any(starts[p] + inst.durations[p] > starts[j]
           for j in range(inst.n_jobs) for p in inst.predecessors[j]):
        return False
    horizon = max(starts[j] + inst.durations[j] for j in range(inst.n_jobs))
    return all(sum(inst.demands[j][r] for j in range(inst.n_jobs)
                   if starts[j] <= t < starts[j] + inst.durations[j]) <= inst.capacities[r]
               for r in range(inst.n_renewable) for t in range(horizon))


class CoreTests(unittest.TestCase):
    def test_exact_oracle_random_dags_and_numbering(self):
        rnd = random.Random(11)
        for _ in range(40):
            n = 4
            durations = [rnd.randint(1, 2) for _ in range(n)]
            demands = [rnd.randint(0, 1) for _ in range(n)]
            labels = list(range(n)); rnd.shuffle(labels)
            successors = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if rnd.random() < .3:
                        successors[labels[i]].append(labels[j])
            model = instance(durations, demands, successors)
            best_sgs = math.inf
            for order in itertools.permutations(range(n)):
                positions = {j: i for i, j in enumerate(order)}
                if any(positions[p] > positions[j] for j in range(n) for p in model.predecessors[j]):
                    continue
                schedule = SerialScheduleGenerator().generate(model, order)
                self.assertTrue(independent_feasible(model, schedule.start_times))
                best_sgs = min(best_sgs, schedule.makespan)
            exact = sum(durations)
            for starts in itertools.product(range(sum(durations)), repeat=n):
                cost = max(starts[j] + durations[j] for j in range(n))
                if cost < exact and independent_feasible(model, starts):
                    exact = cost
            self.assertEqual(exact, best_sgs)

    def test_cycle_and_impossible_resource_rejected(self):
        with self.assertRaises(ValueError): instance([1, 1], [1, 1], [[1], [0]])
        with self.assertRaises(ValueError): instance([1], [2], [[]])

    def test_zero_duration_and_simultaneous_events(self):
        model = instance([1, 0, 1], [1, 10, 1], [[1], [2], []])
        schedule = SerialScheduleGenerator().generate(model, [0, 1, 2])
        self.assertEqual(schedule.start_times, [0, 1, 1])
        validate_schedule(model, schedule)
        self.assertTrue(independent_feasible(model, schedule.start_times))

    def test_validator_detects_bad_certificates(self):
        model = instance([2, 2], [1, 1], [[], []])
        with self.assertRaises(ValueError): validate_schedule(model, Schedule([0, 0], 2))
        with self.assertRaises(ValueError): validate_schedule(model, Schedule([0, 2], 3))

    def test_budget_and_history(self):
        model = instance([2, 1], [1, 1], [[], []])
        budget = EvaluationBudget(1)
        sgs = SerialScheduleGenerator(budget)
        sgs.generate(model, [0, 1], category="ls")
        with self.assertRaises(BudgetExhausted): sgs.generate(model, [1, 0])
        self.assertEqual(budget.used, 1)
        self.assertEqual(budget.counts, {"aco": 0, "ls": 1})
        self.assertEqual(budget.history[0]["evaluation"], 1)

    def test_deadline_race_is_cooperative(self):
        budget = EvaluationBudget(2, time_limit=.1)
        with patch("rcpsp.core.time.perf_counter", return_value=0): budget.start()
        with patch("rcpsp.core.time.perf_counter", return_value=1):
            self.assertTrue(budget.exhausted)
            budget.consume()  # Operation that passed a prior loop check may finish.
            self.assertEqual(budget.used, 1)


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.model = load_psplib_instance(str(ROOT / "psplib/j301_1.sm"))

    def test_combo_shared_and_graph_features(self):
        model = instance([2, 1, 3], [1, 1, 1], [[], [0], [1]])
        counts, distance = graph_features(model)
        self.assertEqual(counts, [0, 1, 2])
        self.assertEqual(distance, [2, 3, 6])
        params = ACOParams(heuristic_type="combo", w_duration=.2, w_succ=.3, w_criticality=.5)
        baseline = BaselineAnt(model, params, PheromoneMatrix(3, .1), random.Random(2))
        meta = Ant(model, params, baseline.pheromones, random.Random(2))
        values = [baseline._heuristic(j) for j in range(3)]
        self.assertEqual(values, [meta._heuristic(j) for j in range(3)])
        self.assertGreater(len(set(values)), 1)

    def test_numeric_scores_and_random_ties(self):
        model = instance([1, 1], [1, 1], [[], []])
        params = ACOParams(alpha=1000, beta=1000, q0=1, heuristic_type="none")
        ant = BaselineAnt(model, params, PheromoneMatrix(2, .1), random.Random(3))
        self.assertEqual({ant._select_job(0, [0, 1]) for _ in range(100)}, {0, 1})
        schedule = BaselineACO(model, params, random.Random(1), max_evaluations=5).run(5)
        validate_schedule(model, schedule)
        params.alpha = 1001
        with self.assertRaises(ValueError): BaselineACO(model, params, random.Random(1))

    def test_replay_and_exact_partial_budget_all_instances(self):
        for path in sorted((ROOT / "psplib").glob("*.sm")):
            model = load_psplib_instance(str(path))
            observed = []
            for _ in range(2):
                manager = ColonyManager(model, random.Random(7), max_evaluations=23)
                manager.init_random_colonies(3)
                schedule, _ = manager.run(23, 20)
                self.assertEqual(manager.budget.used, 23)
                self.assertEqual(sum(manager.budget.counts.values()), 23)
                self.assertTrue(independent_feasible(model, schedule.start_times))
                observed.append((schedule.start_times, manager.events))
            self.assertEqual(observed[0], observed[1])
            baseline = BaselineACO(model, ACOParams(ls_iters=20, ls_prob=1), random.Random(7),
                                   max_evaluations=23)
            schedule = baseline.run(23)
            self.assertEqual(baseline.budget.used, 23)
            self.assertTrue(independent_feasible(model, schedule.start_times))

    def test_global_reward_does_not_replicate_worse_improvement(self):
        manager = ColonyManager(self.model, random.Random(5), max_colonies=3, elite_enabled=False,
                                selection="round_robin")
        manager.init_random_colonies(2)
        good = manager.bandit.colonies[0]
        schedule = SerialScheduleGenerator().generate(self.model, self.model.topological_order)
        good.best_schedule, good.best_cost = schedule, schedule.makespan
        worse = manager.bandit.colonies[1]
        worse.best_cost = 1000
        def run_good(_): return good.best_schedule, good.best_cost
        def run_worse(_):
            worse.best_cost = good.best_cost + 10
            return schedule, worse.best_cost
        with patch.object(good, "run_iteration", side_effect=run_good), patch.object(worse, "run_iteration", side_effect=run_worse):
            manager.run(2, 1)
        self.assertEqual(manager.events[-1]["reward"], 0)
        self.assertFalse(any(e["event"] == "replicate" for e in manager.events))

    def test_idle_pheromones_unchanged(self):
        manager = ColonyManager(self.model, random.Random(8), max_colonies=2,
                                replication_enabled=False, selection="round_robin")
        manager.init_random_colonies(2)
        idle = copy.deepcopy(manager.bandit.colonies[1].pheromones.tau)
        manager.run(1, 2)
        self.assertEqual(manager.bandit.colonies[1].pheromones.tau, idle)

    def test_parameter_rng_not_affected_by_ls_switch(self):
        managers = [ColonyManager(self.model, random.Random(10), local_search_enabled=value)
                    for value in (True, False)]
        for manager in managers: manager.init_random_colonies(3)
        for a, b in zip(managers[0].bandit.colonies, managers[1].bandit.colonies):
            self.assertEqual(a.params.alpha, b.params.alpha)
            self.assertEqual(a.rnd.getstate(), b.rnd.getstate())

    def test_policies_discount_and_invalid_rewards(self):
        controller = ColonyBanditController(selection="round_robin", discount=.5)
        controller.add_colony(None); controller.add_colony(None)
        self.assertEqual(controller.select_colony_index(), 0)
        controller.update(0, 1)
        self.assertEqual(controller.select_colony_index(), 1)
        controller.update(1, 0)
        self.assertEqual(controller.rewards, [.5, 0])
        self.assertEqual(controller.mass, [.5, 1])
        with self.assertRaises(ValueError): controller.update(0, float("nan"))

    def test_budget_one_zero_duration_and_ablation_modes(self):
        zero = instance([0, 0], [0, 0], [[1], []])
        manager = ColonyManager(zero, random.Random(1), max_evaluations=1)
        manager.init_random_colonies(3)
        schedule, _ = manager.run(10, 20)
        self.assertEqual(schedule.makespan, 0)
        self.assertEqual(manager.budget.used, 1)
        for selection in ("ucb", "random", "round_robin"):
            for reward_mode in ("global_improvement", "quality"):
                manager = ColonyManager(self.model, random.Random(2), max_evaluations=50,
                                        selection=selection, reward_mode=reward_mode,
                                        discount=.95, replication_mode="restart", elite_enabled=False)
                manager.init_random_colonies(3)
                schedule, _ = manager.run(50, 5, improvement_threshold=0)
                self.assertEqual(manager.budget.used, 50)
                self.assertTrue(independent_feasible(self.model, schedule.start_times))
                self.assertEqual(len(manager.elite.entries), 0)
                self.assertTrue(all(math.isfinite(e["reward"]) and 0 <= e["reward"] <= 1
                                    for e in manager.events if e["event"] == "pull"))

    def test_normalized_pheromones_scale_invariant_for_combo(self):
        model = instance([2, 1, 3], [1, 1, 1], [[], [], []])
        scaled = instance([20, 10, 30], [1, 1, 1], [[], [], []])
        params = ACOParams(heuristic_type="combo", w_duration=.2, w_succ=.3, w_criticality=.5)
        solvers = [BaselineACO(m, params, random.Random(6), max_evaluations=20) for m in (model, scaled)]
        schedules = [s.run(20) for s in solvers]
        self.assertEqual(schedules[0].order, schedules[1].order)
        self.assertEqual(schedules[0].makespan * 10, schedules[1].makespan)
        for a, b in zip(solvers[0].pheromones.tau, solvers[1].pheromones.tau):
            for x, y in zip(a, b): self.assertAlmostEqual(x, y)


class ProtocolTests(unittest.TestCase):
    def test_legacy_parser_is_strict(self):
        self.assertEqual(parse_output("Final colonies count: 7\nElite archive size: 5\nBest makespan: 43"), (43, 7, 5))
        with self.assertRaises(ValueError): parse_output("solver crashed")

    def test_actual_workbook_and_missing_reference(self):
        refs = load_references(DEFAULT_REF_FILE, ["j1201_1.sm", "j301_1.sm"])
        self.assertEqual(refs["j1201_1.sm"]["lower_bound"], 100)
        self.assertEqual(refs["j1201_1.sm"]["bks"], 105)
        self.assertIsNone(refs["j1201_1.sm"]["optimum"])
        self.assertEqual(refs["j301_1.sm"]["optimum"], 43)
        with self.assertRaises(FileNotFoundError): load_references("does_not_exist.csv")

    def test_reference_metrics_and_validation(self):
        ref = {"lower_bound": 100, "bks": 105, "optimum": None}
        values = quality_metrics(119, ref)
        self.assertAlmostEqual(values["deviation_to_bks_percent"], 13.333333333333334)
        self.assertEqual(values["gap_to_lb_percent"], 19)
        self.assertIsNone(values["gap_to_optimum_percent"])
        self.assertTrue(quality_metrics(102, ref)["improves_bks"])
        with self.assertRaises(ValueError): quality_metrics(99, ref)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            path.write_text("FileName,#Act,CPM,LBall,UB-lit,KeepOPT\nx.sm,4,2,5,4,Change\n")
            with self.assertRaises(ValueError): load_references(path)

    def test_insufficient_budget_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "short"
            args = ["--instance", str(ROOT / "psplib/j301_1.sm"), "--no-reference", "--algorithms", "baseline",
                    "--seeds", "1", "--evaluations", "50", "--iterations", "1", "--ants", "2",
                    "--output-dir", str(output)]
            self.assertEqual(main(args), 1)
            self.assertEqual(read_csv(output / "summary.csv")[0]["status"], "error")
            result = json.loads(next((output / "runs").glob("*_seed1.json")).read_text())
            self.assertEqual(result["driver_validation"], "failed")

    def test_coverage_and_unequal_budgets_rejected_by_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            main(["--instance", str(ROOT / "psplib/j301_1.sm"), "--no-reference", "--seeds", "1",
                  "--evaluations", "5", "--output-dir", str(output)])
            rows = read_csv(output / "summary.csv")
            rows[1]["evaluations"] = "4"
            with (output / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ValueError): analyze(output)
            with (output / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows[:1])
            with self.assertRaises(ValueError): analyze(output)

    def test_driver_success_replay_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = ["--instance", str(ROOT / "psplib/j301_1.sm"), "--no-reference",
                      "--evaluations", "23", "--seeds", "1"]
            output = root / "success"
            self.assertEqual(main(common + ["--output-dir", str(output)]), 0)
            rows = read_csv(output / "summary.csv")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["status"] == "success" and r["evaluations"] == "23" for r in rows))
            analysis = analyze(output)
            self.assertEqual(analysis["comparison"]["tasks"], 1)
            self.assertIsNone(analysis["comparison"]["task_bootstrap_ci95"])
            result = json.loads((output / rows[0]["result_file"]).read_text())
            validate_result(result, result["config"], output / "inputs/j301_1.sm")
            result["schedule"]["makespan"] += 1
            with self.assertRaises(ValueError): validate_result(result, result["config"], output / "inputs/j301_1.sm")
            with self.assertRaises(FileExistsError): main(common + ["--output-dir", str(output)])
            failed = root / "failed"
            with patch("rcpsp.run_experiments.subprocess.run", return_value=type("Proc", (),
                       {"returncode": 2, "stdout": "", "stderr": "intentional failure"})()):
                self.assertEqual(main(common + ["--output-dir", str(failed)]), 1)
            failed_rows = read_csv(failed / "summary.csv")
            self.assertTrue(all(r["status"] == "error" and not r["best_makespan"] for r in failed_rows))
            self.assertEqual(len(list((failed / "runs").glob("*.failure.json"))), 2)
            with self.assertRaises(ValueError): analyze(failed)

    def test_driver_timeout(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            args = ["--instance", str(ROOT / "psplib/j301_1.sm"), "--no-reference", "--algorithms", "baseline",
                    "--seeds", "1", "--evaluations", "5", "--output-dir", str(Path(directory) / "timeout")]
            with patch("rcpsp.run_experiments.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", .01)):
                self.assertEqual(main(args), 1)
            rows = read_csv(Path(directory) / "timeout/summary.csv")
            self.assertEqual(rows[0]["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
