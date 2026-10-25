"""Shared, validated single-mode RCPSP model and serial schedule generation."""

import heapq
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class RcpspInstance:
    n_jobs: int
    n_renewable: int
    durations: List[int]
    demands: List[List[int]]
    capacities: List[int]
    successors: List[List[int]]
    predecessors: List[List[int]]

    def __post_init__(self):
        if type(self.n_jobs) is not int or self.n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        if type(self.n_renewable) is not int or self.n_renewable < 0:
            raise ValueError("n_renewable must be a nonnegative integer")
        if any(len(values) != self.n_jobs for values in
               (self.durations, self.demands, self.successors, self.predecessors)):
            raise ValueError("Job arrays must have n_jobs entries")
        if len(self.capacities) != self.n_renewable:
            raise ValueError("Resource array must have n_renewable entries")
        for value in self.durations + self.capacities:
            if type(value) is not int or value < 0:
                raise ValueError("Durations and capacities must be nonnegative integers")
        expected_preds = [set() for _ in range(self.n_jobs)]
        for j in range(self.n_jobs):
            if len(self.demands[j]) != self.n_renewable:
                raise ValueError("Demand row has incorrect resource count")
            for r, demand in enumerate(self.demands[j]):
                if type(demand) is not int or demand < 0:
                    raise ValueError("Demands must be nonnegative integers")
                if self.durations[j] > 0 and demand > self.capacities[r]:
                    raise ValueError("Job %d demand exceeds resource %d capacity" % (j, r))
            for edges in (self.successors[j], self.predecessors[j]):
                if any(type(k) is not int or not 0 <= k < self.n_jobs for k in edges):
                    raise ValueError("Invalid job index in precedence graph")
                if len(set(edges)) != len(edges) or j in edges:
                    raise ValueError("Duplicate or self precedence edge")
            for successor in self.successors[j]:
                expected_preds[successor].add(j)
        if any(set(self.predecessors[j]) != expected_preds[j] for j in range(self.n_jobs)):
            raise ValueError("Predecessors and successors disagree")
        self.topological_order = topological_order(self)


def topological_order(inst: RcpspInstance, priorities=None) -> List[int]:
    """Kahn's algorithm; priorities only break ties among eligible jobs."""
    priorities = list(range(inst.n_jobs)) if priorities is None else priorities
    indegree = [len(preds) for preds in inst.predecessors]
    ready = [(priorities[j], j) for j, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, job = heapq.heappop(ready)
        order.append(job)
        for successor in inst.successors[job]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (priorities[successor], successor))
    if len(order) != inst.n_jobs:
        raise ValueError("Precedence graph contains a cycle")
    return order


def validate_order(inst: RcpspInstance, order: Sequence[int]):
    if (len(order) != inst.n_jobs or
            any(type(j) is not int or not 0 <= j < inst.n_jobs for j in order) or
            len(set(order)) != inst.n_jobs):
        raise ValueError("Order must be a permutation of all jobs")
    positions = {job: pos for pos, job in enumerate(order)}
    if any(positions[p] >= positions[j] for j in range(inst.n_jobs)
           for p in inst.predecessors[j]):
        raise ValueError("Order violates precedence")


@dataclass
class Schedule:
    start_times: List[int]
    makespan: int
    order: Optional[List[int]] = None


def validate_schedule(inst: RcpspInstance, schedule: Schedule):
    """Independent event sweep checks feasibility, including resource usage."""
    starts = schedule.start_times
    if type(schedule.makespan) is not int or schedule.makespan < 0:
        raise ValueError("Invalid makespan")
    if len(starts) != inst.n_jobs or any(type(t) is not int or t < 0 for t in starts):
        raise ValueError("Invalid schedule start times")
    if schedule.makespan != max(starts[j] + inst.durations[j] for j in range(inst.n_jobs)):
        raise ValueError("Incorrect makespan")
    for j in range(inst.n_jobs):
        if any(starts[p] + inst.durations[p] > starts[j] for p in inst.predecessors[j]):
            raise ValueError("Schedule violates precedence")
    events = {}
    for j, duration in enumerate(inst.durations):
        if duration == 0:
            continue
        for time, sign in ((starts[j], 1), (starts[j] + duration, -1)):
            change = events.setdefault(time, [0] * inst.n_renewable)
            for r, demand in enumerate(inst.demands[j]):
                change[r] += sign * demand
    usage = [0] * inst.n_renewable
    for time in sorted(events):
        for r, change in enumerate(events[time]):
            usage[r] += change
            if not 0 <= usage[r] <= inst.capacities[r]:
                raise ValueError("Schedule violates resource capacity")
    if schedule.order is not None:
        validate_order(inst, schedule.order)


class BudgetExhausted(Exception):
    """Raised before an evaluation exceeding the declared search budget."""


class EvaluationBudget:
    def __init__(self, limit: Optional[int] = None, time_limit: Optional[float] = None):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Evaluation budget must be a positive integer")
        self.limit = limit
        self.used = 0
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("Time limit must be finite and positive")
        self.time_limit = time_limit
        self.started = None
        self.counts = {"aco": 0, "ls": 0}
        self.history = []
        self.best_cost = math.inf

    def start(self):
        if self.started is None:
            self.started = time.perf_counter()

    @property
    def stop_reason(self):
        if self.limit is not None and self.used >= self.limit:
            return "evaluations"
        if (self.time_limit is not None and self.started is not None
                and time.perf_counter() - self.started >= self.time_limit):
            return "time"
        return None

    @property
    def exhausted(self):
        return self.stop_reason is not None

    def consume(self):
        # Time is a cooperative limit checked by search loops; an in-flight
        # construction/evaluation is allowed to finish, avoiding deadline races.
        if self.limit is not None and self.used >= self.limit:
            raise BudgetExhausted()
        self.used += 1

    def record(self, schedule, category):
        self.counts[category] += 1
        if schedule.makespan < self.best_cost:
            self.best_cost = schedule.makespan
            self.history.append({"evaluation": self.used, "makespan": self.best_cost,
                                 "elapsed_sec": time.perf_counter() - self.started,
                                 "category": category})


class SerialScheduleGenerator:
    def __init__(self, budget: Optional[EvaluationBudget] = None):
        self.budget = budget if budget is not None else EvaluationBudget()

    def generate(self, inst: RcpspInstance, order: Sequence[int], category="aco") -> Schedule:
        validate_order(inst, order)
        if category not in self.budget.counts:
            raise ValueError("Unknown evaluation category")
        self.budget.start()
        self.budget.consume()
        starts = [0] * inst.n_jobs
        # Sum of durations is a feasible horizon: jobs can run sequentially.
        horizon = sum(inst.durations)
        usage = [[0] * horizon for _ in range(inst.n_renewable)]
        for job in order:
            duration = inst.durations[job]
            earliest = max((starts[p] + inst.durations[p]
                            for p in inst.predecessors[job]), default=0)
            if duration == 0:
                starts[job] = earliest
                continue
            for time in range(earliest, horizon - duration + 1):
                if all(usage[r][t] + demand <= inst.capacities[r]
                       for r, demand in enumerate(inst.demands[job]) if demand
                       for t in range(time, time + duration)):
                    starts[job] = time
                    for r, demand in enumerate(inst.demands[job]):
                        if demand:
                            for t in range(time, time + duration):
                                usage[r][t] += demand
                    break
            else:
                raise ValueError("No feasible start within serial horizon; invalid instance")
        schedule = Schedule(starts, max(starts[j] + inst.durations[j]
                                       for j in range(inst.n_jobs)), list(order))
        self.budget.record(schedule, category)
        return schedule


def load_psplib_instance(path: str) -> RcpspInstance:
    """Read the PSPLIB .sm subset supported by this solver, without dependencies.

    Multiple modes, nonrenewable/doubly constrained resources and release times
    are rejected rather than silently treated as classical single-mode RCPSP.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    def scalar(label):
        matches = [line for line in lines if line.strip().startswith(label)]
        if len(matches) != 1 or ":" not in matches[0]:
            raise ValueError("Missing/ambiguous PSPLIB field: " + label)
        return int(matches[0].split(":", 1)[1].split()[0])

    def rows_between(begin, end=None):
        start = next((i for i, line in enumerate(lines) if line.strip().startswith(begin)), None)
        if start is None:
            raise ValueError("Missing PSPLIB section: " + begin)
        stop = len(lines) if end is None else next(
            (i for i in range(start + 1, len(lines)) if lines[i].strip().startswith(end)), None)
        if stop is None:
            raise ValueError("Missing PSPLIB section: " + str(end))
        return [[int(value) for value in line.split()] for line in lines[start + 1:stop]
                if line.strip() and line.split()[0].isdigit()]

    if scalar("projects") != 1:
        raise ValueError("Only single-project RCPSP is supported")
    n = scalar("jobs (incl.")
    renewable = scalar("- renewable")
    if scalar("- nonrenewable") or scalar("- doubly constrained"):
        raise ValueError("Only renewable resources are supported")
    projects = rows_between("PROJECT INFORMATION", "PRECEDENCE RELATIONS")
    if len(projects) != 1 or len(projects[0]) < 3 or projects[0][2] != 0:
        raise ValueError("Only projects with zero release time are supported")
    precedence = rows_between("PRECEDENCE RELATIONS", "REQUESTS/DURATIONS")
    requests = rows_between("REQUESTS/DURATIONS", "RESOURCEAVAILABILITIES")
    capacities = rows_between("RESOURCEAVAILABILITIES")
    if (len(precedence) != n or len(requests) != n or len(capacities) != 1 or
            len(capacities[0]) != renewable):
        raise ValueError("Incorrect PSPLIB job/resource row counts")
    successors = [[] for _ in range(n)]
    durations = [0] * n
    demands = [[] for _ in range(n)]
    seen = set()
    for row in precedence:
        if len(row) < 3 or row[1] != 1 or len(row) != row[2] + 3:
            raise ValueError("Only valid single-mode precedence rows are supported")
        job = row[0] - 1
        if not 0 <= job < n or job in seen:
            raise ValueError("Invalid or duplicate PSPLIB job number")
        seen.add(job)
        successors[job] = [s - 1 for s in row[3:]]
    seen = set()
    for row in requests:
        if len(row) != renewable + 3 or row[1] != 1:
            raise ValueError("Only single-mode demand rows are supported")
        job = row[0] - 1
        if not 0 <= job < n or job in seen:
            raise ValueError("Invalid or duplicate PSPLIB demand job number")
        seen.add(job)
        durations[job], demands[job] = row[2], row[3:]
    predecessors = [[] for _ in range(n)]
    for job, edges in enumerate(successors):
        for successor in edges:
            if not 0 <= successor < n:
                raise ValueError("Invalid PSPLIB successor")
            predecessors[successor].append(job)
    return RcpspInstance(n, renewable, durations, demands, capacities[0], successors, predecessors)


def validate_aco_params(params):
    for name in ("alpha", "beta"):
        value = getattr(params, name)
        if not math.isfinite(value) or not 0 <= value <= 1000:
            raise ValueError(name + " must lie in [0, 1000]")
    for name in ("rho", "q0"):
        value = getattr(params, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + " must lie in [0, 1]")
    if params.heuristic_type not in {"none", "min_duration", "combo"}:
        raise ValueError("Unknown heuristic type")
    if hasattr(params, "ls_iters"):
        if type(params.ls_iters) is not int or params.ls_iters < 0:
            raise ValueError("ls_iters must be a nonnegative integer")
        if not math.isfinite(params.ls_prob) or not 0 <= params.ls_prob <= 1:
            raise ValueError("ls_prob must lie in [0, 1]")
    if hasattr(params, "w_duration"):
        weights = [params.w_duration, params.w_succ, params.w_criticality]
        if (any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isfinite(sum(weights)) or sum(weights) <= 0):
            raise ValueError("Heuristic weights must be finite, nonnegative and nonzero")


def graph_features(inst):
    """Transitive successor count and precedence-only remaining path length."""
    if not hasattr(inst, "_succ_count"):
        descendants = [set() for _ in range(inst.n_jobs)]
        distance = [0] * inst.n_jobs
        for j in reversed(inst.topological_order):
            for successor in inst.successors[j]:
                descendants[j].add(successor)
                descendants[j].update(descendants[successor])
            distance[j] = inst.durations[j] + max(
                (distance[k] for k in inst.successors[j]), default=0)
        inst._succ_count = [len(items) for items in descendants]
        inst._dist_to_end = distance
    return inst._succ_count, inst._dist_to_end


def heuristic_value(inst, params, job):
    if params.heuristic_type == "none":
        return 1.0
    if params.heuristic_type == "min_duration":
        return 1.0 / (1.0 + inst.durations[job])
    if params.heuristic_type != "combo":
        raise ValueError("Unknown heuristic type")
    successors, distance = graph_features(inst)
    weights = [params.w_duration, params.w_succ, params.w_criticality]
    total = sum(weights)
    score = (weights[0] / total * (1 - inst.durations[job] / (max(inst.durations) or 1))
             + weights[1] / total * successors[job] / (max(successors) or 1)
             + weights[2] / total * distance[job] / (max(distance) or 1))
    return max(score, 1e-6)


def log_score(tau, eta, params):
    if not math.isfinite(tau) or tau < 0 or not math.isfinite(eta) or eta <= 0:
        raise ValueError("Invalid pheromone/heuristic value")
    return params.alpha * math.log(max(tau, 1e-9)) + params.beta * math.log(max(eta, 1e-9))


def select_from_logs(candidates, logs, q0, rnd):
    """Stable roulette; exact greedy ties are chosen uniformly at random."""
    if not candidates or len(candidates) != len(logs) or any(not math.isfinite(x) for x in logs):
        raise ValueError("Invalid selection scores")
    maximum = max(logs)
    if rnd.random() < q0:
        return rnd.choice([j for j, value in zip(candidates, logs) if value == maximum])
    weights = [math.exp(value - maximum) for value in logs]
    target = rnd.random() * math.fsum(weights)
    cumulative = 0.0
    for job, weight in zip(candidates, weights):
        cumulative += weight
        if target < cumulative:
            return job
    return candidates[-1]


def split_rng(rnd):
    """Fixed draws irrespective of LS being enabled: ants, LS moves, LS gate."""
    return tuple(random.Random(rnd.getrandbits(64)) for _ in range(3))
