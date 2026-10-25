import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple


try:
    from .core import (RcpspInstance, Schedule, SerialScheduleGenerator,
                       load_psplib_instance, validate_order, validate_schedule,
                       EvaluationBudget, validate_aco_params, heuristic_value,
                       log_score, select_from_logs, split_rng, graph_features)
except ImportError:
    from core import (RcpspInstance, Schedule, SerialScheduleGenerator,
                      load_psplib_instance, validate_order, validate_schedule,
                      EvaluationBudget, validate_aco_params, heuristic_value,
                       log_score, select_from_logs, split_rng, graph_features)


@dataclass
class ACOParams:
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    q0: float = 0.5
    tau0: float = 0.1
    heuristic_type: str = "min_duration"
    ls_iters: int = 0
    ls_prob: float = 0.0
    w_duration: float = 1.0
    w_succ: float = 0.0
    w_criticality: float = 0.0


class PheromoneMatrix:
    def __init__(self, n_jobs: int, tau0: float):
        self.tau = [[tau0 for _ in range(n_jobs)] for _ in range(n_jobs)]

    def get(self, pos: int, job: int) -> float:
        return self.tau[pos][job]

    def evaporate(self, rho: float):
        for i in range(len(self.tau)):
            for j in range(len(self.tau[i])):
                self.tau[i][j] *= (1.0 - rho)

    def deposit_on_order(self, order: Sequence[int], amount: float):
        for pos, job in enumerate(order):
            self.tau[pos][job] += amount


class Ant:
    def __init__(
        self,
        inst: RcpspInstance,
        params: ACOParams,
        pheromones: PheromoneMatrix,
        rnd: random.Random,
    ):
        self.inst = inst
        self.params = params
        self.pheromones = pheromones
        self.rnd = rnd

    def construct_order(self) -> List[int]:
        scheduled = set()
        order = []

        for pos in range(self.inst.n_jobs):
            eligible = self._eligible_jobs(scheduled)
            if not eligible:
                raise ValueError("No eligible job; invalid precedence graph")
            job = self._select_job(pos, eligible)
            order.append(job)
            scheduled.add(job)

        return order

    def _eligible_jobs(self, scheduled: set) -> List[int]:
        eligible = []

        for j in range(self.inst.n_jobs):
            if j in scheduled:
                continue

            if all(pred in scheduled for pred in self.inst.predecessors[j]):
                eligible.append(j)

        return eligible

    def _heuristic(self, job: int) -> float:
        return heuristic_value(self.inst, self.params, job)

    def _log_score(self, pos: int, job: int) -> float:
        return log_score(self.pheromones.get(pos, job), self._heuristic(job), self.params)

    def _select_job(self, pos: int, candidates: Sequence[int]) -> int:
        return select_from_logs(candidates, [self._log_score(pos, j) for j in candidates],
                                self.params.q0, self.rnd)


class BaselineACO:
    def __init__(
        self,
        inst: RcpspInstance,
        params: ACOParams,
        rnd: random.Random,
        ants_per_iteration: int = 20,
        max_evaluations: Optional[int] = None,
        time_limit: Optional[float] = None,
    ):
        validate_aco_params(params)
        if type(ants_per_iteration) is not int or ants_per_iteration < 1:
            raise ValueError("ants_per_iteration must be positive")
        if not math.isfinite(params.tau0) or not 0 < params.tau0 <= 1:
            raise ValueError("tau0 must lie in (0, 1]")
        self.inst = inst
        self.params = params
        self.rnd, self.ls_rnd, self.ls_gate_rnd = split_rng(rnd)
        self.ants_per_iteration = ants_per_iteration

        self.pheromones = PheromoneMatrix(inst.n_jobs, params.tau0)
        self.budget = EvaluationBudget(max_evaluations, time_limit)
        self.sgs = SerialScheduleGenerator(self.budget)

        self.best_schedule: Optional[Schedule] = None
        self.best_order: Optional[List[int]] = None
        self.best_cost = math.inf

    def run(self, iterations: int) -> Schedule:
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be positive")
        self.budget.start()
        for _ in range(iterations):
            if self.budget.exhausted:
                break
            iter_best_order = None
            iter_best_schedule = None
            iter_best_cost = math.inf

            for _ in range(self.ants_per_iteration):
                if self.budget.exhausted:
                    break
                ant = Ant(self.inst, self.params, self.pheromones, self.rnd)
                order = ant.construct_order()
                schedule = self.sgs.generate(self.inst, order)
                cost = schedule.makespan

                if cost < iter_best_cost:
                    iter_best_cost = cost
                    iter_best_order = order
                    iter_best_schedule = schedule

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_order = order
                    self.best_schedule = schedule

            if (iter_best_order is not None and self.params.ls_iters > 0
                    and not self.budget.exhausted
                    and self.ls_gate_rnd.random() < self.params.ls_prob):
                for _ in range(self.params.ls_iters):
                    if self.budget.exhausted:
                        break
                    i = self.ls_rnd.randrange(self.inst.n_jobs)
                    j = self.ls_rnd.randrange(self.inst.n_jobs)
                    if i == j:
                        continue
                    candidate = list(iter_best_order)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    try:
                        validate_order(self.inst, candidate)
                    except ValueError:
                        continue
                    schedule = self.sgs.generate(self.inst, candidate, category="ls")
                    if schedule.makespan < iter_best_cost:
                        iter_best_order = candidate
                        iter_best_schedule = schedule
                        iter_best_cost = schedule.makespan
                        if iter_best_cost < self.best_cost:
                            self.best_cost = iter_best_cost
                            self.best_order = candidate
                            self.best_schedule = schedule

            self.pheromones.evaporate(self.params.rho)

            if iter_best_order is not None:
                deposit = self.params.tau0 * max(1, sum(self.inst.durations)) / max(1, iter_best_cost)
                self.pheromones.deposit_on_order(iter_best_order, deposit)

        if self.best_schedule is None:
            raise RuntimeError("Не удалось построить расписание")

        validate_schedule(self.inst, self.best_schedule)
        return self.best_schedule


def solve_baseline_aco(
    path: str,
    seed: int = 42,
    iterations: int = 500,
    ants_per_iteration: int = 20,
    max_evaluations: Optional[int] = None,
    ls_iters: int = 0,
    ls_prob: float = 0.0,
) -> Tuple[Schedule, ACOParams]:
    rnd = random.Random(seed)
    inst = load_psplib_instance(path)

    params = ACOParams(
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q0=0.5,
        tau0=0.1,
        heuristic_type="min_duration",
        ls_iters=ls_iters,
        ls_prob=ls_prob,
    )

    solver = BaselineACO(
        inst=inst,
        params=params,
        rnd=rnd,
        ants_per_iteration=ants_per_iteration,
        max_evaluations=max_evaluations,
    )

    schedule = solver.run(iterations)
    print("Schedule evaluations:", solver.budget.used)
    return schedule, params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Базовый ACO для RCPSP")
    parser.add_argument("file", help="путь к файлу PSPLIB, например j301_1.sm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--ants", type=int, default=20)
    parser.add_argument("--init-colonies", type=int, choices=[1], default=1)
    parser.add_argument("--max-colonies", type=int, choices=[1], default=1)
    parser.add_argument("--evaluations", type=int, default=None)
    parser.add_argument("--ls-iters", type=int, default=0)
    parser.add_argument("--ls-prob", type=float, default=0.0)

    args = parser.parse_args()

    sched, params = solve_baseline_aco(
        args.file,
        seed=args.seed,
        iterations=args.iterations,
        ants_per_iteration=args.ants,
        max_evaluations=args.evaluations,
        ls_iters=args.ls_iters,
        ls_prob=args.ls_prob,
    )

    print("Final colonies count:", 1)
    print("Elite archive size:", 0)
    print(f"Best makespan: {sched.makespan}")
    print("Best colony params:", params)
    print("Start times:", sched.start_times)
