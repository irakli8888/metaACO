import math
import random
from dataclasses import dataclass, asdict
from typing import List, Sequence, Optional, Tuple


try:
    from .core import (RcpspInstance, Schedule, SerialScheduleGenerator,
                       load_psplib_instance, validate_order, validate_schedule,
                       topological_order, EvaluationBudget, validate_aco_params, heuristic_value,
                       log_score, select_from_logs, split_rng, graph_features)
except ImportError:
    from core import (RcpspInstance, Schedule, SerialScheduleGenerator,
                      load_psplib_instance, validate_order, validate_schedule,
                      topological_order, EvaluationBudget, validate_aco_params, heuristic_value,
                       log_score, select_from_logs, split_rng, graph_features)


# =========================
# 3. Феромоны и параметры
# =========================

@dataclass
class ColonyParams:
    alpha: float
    beta: float
    rho: float
    q0: float
    heuristic_type: str               # 'min_duration', 'combo', 'none'
    w_duration: float = 1.0           # веса для комбинированной эвристики
    w_succ: float = 0.0
    w_criticality: float = 0.0

    # параметры локального поиска
    ls_iters: int = 20                # сколько итераций локального поиска
    ls_prob: float = 0.5              # вероятность запуска LS для лучшего решения итерации

    @staticmethod
    def random_init(rnd: random.Random) -> "ColonyParams":
        heuristic_type = rnd.choice(["min_duration", "combo", "none"])
        # случайные веса для "combo"
        w_duration = rnd.random()
        w_succ = rnd.random()
        w_criticality = rnd.random()
        s = w_duration + w_succ + w_criticality
        if s == 0:
            w_duration, w_succ, w_criticality = 1.0, 0.0, 0.0
            s = 1.0
        w_duration /= s
        w_succ /= s
        w_criticality /= s

        return ColonyParams(
            alpha=0.5 + rnd.random(),
            beta=1.0 + 3.0 * rnd.random(),
            rho=0.02 + 0.2 * rnd.random(),
            q0=0.1 + 0.8 * rnd.random(),
            heuristic_type=heuristic_type,
            w_duration=w_duration,
            w_succ=w_succ,
            w_criticality=w_criticality,
            ls_iters=rnd.randint(5, 40),
            ls_prob=rnd.uniform(0.2, 0.8),
        )

    def mutate(self, rnd: random.Random) -> "ColonyParams":
        def clip(x, lo, hi):
            return max(lo, min(hi, x))

        child = ColonyParams(
            alpha=clip(self.alpha + rnd.gauss(0, 0.1), 0.1, 3.0),
            beta=clip(self.beta + rnd.gauss(0, 0.3), 0.1, 6.0),
            rho=clip(self.rho + rnd.gauss(0, 0.02), 0.01, 0.5),
            q0=clip(self.q0 + rnd.gauss(0, 0.05), 0.01, 0.99),
            heuristic_type=self.heuristic_type,
            w_duration=self.w_duration,
            w_succ=self.w_succ,
            w_criticality=self.w_criticality,
            ls_iters=self.ls_iters,
            ls_prob=self.ls_prob,
        )

        # иногда переключаем тип эвристики
        if rnd.random() < 0.1:
            child.heuristic_type = rnd.choice(["min_duration", "combo", "none"])

        # мягко мутируем веса
        if child.heuristic_type == "combo":
            child.w_duration = clip(child.w_duration + rnd.gauss(0, 0.1), 0.0, 1.0)
            child.w_succ = clip(child.w_succ + rnd.gauss(0, 0.1), 0.0, 1.0)
            child.w_criticality = clip(child.w_criticality + rnd.gauss(0, 0.1), 0.0, 1.0)
            s = child.w_duration + child.w_succ + child.w_criticality
            if s == 0:
                child.w_duration, child.w_succ, child.w_criticality = 1.0, 0.0, 0.0
            else:
                child.w_duration /= s
                child.w_succ /= s
                child.w_criticality /= s

        # мутируем параметры LS
        child.ls_iters = int(clip(child.ls_iters + rnd.gauss(0, 5), 0, 50))
        child.ls_prob = clip(child.ls_prob + rnd.gauss(0, 0.1), 0.0, 1.0)

        return child


class PheromoneMatrix:
    """
    Феромоны по схеме (позиция в порядке, работа).
    """

    def __init__(self, max_position: int, n_jobs: int, tau0: float):
        self.tau = [[tau0 for _ in range(n_jobs)] for _ in range(max_position)]

    def get(self, pos: int, job: int) -> float:
        return self.tau[pos][job]

    def evaporate(self, rho: float):
        for i in range(len(self.tau)):
            row = self.tau[i]
            for j in range(len(row)):
                row[j] *= (1.0 - rho)

    def deposit_on_order(self, order: Sequence[int], amount: float):
        for pos, job in enumerate(order):
            self.tau[pos][job] += amount


# =================
# 4. Муравей
# =================

class Ant:
    def __init__(
        self,
        inst: RcpspInstance,
        params: ColonyParams,
        pheromones: PheromoneMatrix,
        rnd: random.Random,
    ):
        self.inst = inst
        self.params = params
        self.pheromones = pheromones
        self.rnd = rnd
        graph_features(inst)

    def construct_order(self) -> List[int]:
        n_jobs = self.inst.n_jobs
        scheduled = set()
        order: List[int] = []

        for pos in range(n_jobs):
            eligible = self._eligible_jobs(scheduled)
            if not eligible:
                raise ValueError("No eligible job; invalid precedence graph")
            job = self._select_job(pos, eligible)
            order.append(job)
            scheduled.add(job)

        return order

    def _eligible_jobs(self, scheduled: set) -> List[int]:
        inst = self.inst
        eligible = []
        for j in range(inst.n_jobs):
            if j in scheduled:
                continue
            ok = True
            for p in inst.predecessors[j]:
                if p not in scheduled:
                    ok = False
                    break
            if ok:
                eligible.append(j)
        return eligible

    def _heuristic(self, job: int) -> float:
        return heuristic_value(self.inst, self.params, job)

    def _log_score(self, pos: int, job: int) -> float:
        return log_score(self.pheromones.get(pos, job), self._heuristic(job), self.params)

    def _select_job(self, pos: int, candidates: Sequence[int]) -> int:
        return select_from_logs(candidates, [self._log_score(pos, j) for j in candidates],
                                self.params.q0, self.rnd)


# ==========================
# 5. Одна колония муравьёв
# ==========================

class AntColony:
    def __init__(
        self,
        inst: RcpspInstance,
        params: ColonyParams,
        rnd: random.Random,
        tau0: float = 0.1,
        elite_orders: Optional[List[List[int]]] = None,
        elite_boost: float = 0.5,
        budget: Optional[EvaluationBudget] = None,
    ):
        validate_aco_params(params)
        if not math.isfinite(tau0) or not 0 < tau0 <= 1:
            raise ValueError("tau0 must lie in (0, 1]")
        if not math.isfinite(elite_boost) or not 0 <= elite_boost <= 1:
            raise ValueError("elite_boost must lie in [0, 1]")
        self.inst = inst
        self.params = params
        self.rnd, self.ls_rnd, self.ls_gate_rnd = split_rng(rnd)
        self.pheromones = PheromoneMatrix(inst.n_jobs, inst.n_jobs, tau0)
        self.tau0 = tau0
        self.duration_scale = max(1, sum(inst.durations))
        self.sgs = SerialScheduleGenerator(budget)

        self.best_schedule: Optional[Schedule] = None
        self.best_order: Optional[List[int]] = None
        self.best_cost: float = math.inf

        # начальная "подкрутка" феромонов по элитным решениям
        if elite_orders:
            for order in elite_orders:
                validate_order(inst, order)
                self.pheromones.deposit_on_order(order, amount=elite_boost * tau0)

    def _order_is_precedence_feasible(self, order: List[int]) -> bool:
        try:
            validate_order(self.inst, order)
        except ValueError:
            return False
        return True

    def _local_search_order(self, order: List[int], max_iters: int,
                            schedule: Optional[Schedule] = None) -> List[int]:
        if max_iters <= 0:
            return order

        best_order = list(order)
        best_sched = schedule if schedule is not None else self.sgs.generate(self.inst, best_order, category="ls")
        best_cost = best_sched.makespan

        for _ in range(max_iters):
            if self.sgs.budget.exhausted:
                break
            i = self.ls_rnd.randrange(0, len(best_order))
            j = self.ls_rnd.randrange(0, len(best_order))
            if i == j:
                continue
            if i > j:
                i, j = j, i

            cand = list(best_order)
            cand[i], cand[j] = cand[j], cand[i]

            if not self._order_is_precedence_feasible(cand):
                continue

            sched = self.sgs.generate(self.inst, cand, category="ls")
            cost = sched.makespan
            if cost < best_cost:
                best_cost = cost
                best_sched = sched
                best_order = cand

        # если улучшили, обновляем глобальный best колонии
        if best_cost < self.best_cost:
            self.best_cost = best_cost
            self.best_schedule = best_sched
            self.best_order = best_order

        self._last_ls_schedule = best_sched
        return best_order

    def run_iteration(self, n_ants: int) -> Tuple[Optional[Schedule], float]:
        if type(n_ants) is not int or n_ants < 1:
            raise ValueError("n_ants must be positive")
        iter_best_sched: Optional[Schedule] = None
        iter_best_cost = math.inf
        iter_best_order: Optional[List[int]] = None

        for _ in range(n_ants):
            if self.sgs.budget.exhausted:
                break
            ant = Ant(self.inst, self.params, self.pheromones, self.rnd)
            order = ant.construct_order()
            sched = self.sgs.generate(self.inst, order)
            cost = sched.makespan

            if cost < iter_best_cost:
                iter_best_cost = cost
                iter_best_sched = sched
                iter_best_order = order

            if cost < self.best_cost:
                self.best_cost = cost
                self.best_schedule = sched
                self.best_order = order

        # локальный поиск для лучшего порядка итерации с вероятностью ls_prob
        if (iter_best_order is not None and self.params.ls_iters > 0
                and not self.sgs.budget.exhausted
                and self.ls_gate_rnd.random() < self.params.ls_prob):
            improved_order = self._local_search_order(
                iter_best_order, self.params.ls_iters, iter_best_sched)
            improved_sched = self._last_ls_schedule
            improved_cost = improved_sched.makespan
            if improved_cost < iter_best_cost:
                iter_best_cost = improved_cost
                iter_best_sched = improved_sched
                iter_best_order = improved_order

        # обновление феромона
        if iter_best_order is not None:
            self.pheromones.evaporate(self.params.rho)
            self.pheromones.deposit_on_order(
                iter_best_order, amount=self.tau0 * self.duration_scale / max(1, iter_best_cost))

        return iter_best_sched, iter_best_cost


# =======================
# 6. Банк элитных решений
# =======================

@dataclass
class EliteEntry:
    schedule: Schedule


class EliteArchive:
    def __init__(self, capacity: int = 20):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Elite capacity must be positive")
        self.capacity = capacity
        self.entries: List[EliteEntry] = []

    def add(self, schedule: Schedule):
        # One entry per distinct schedule; repeated visits do not multiply its weight.
        if any(entry.schedule.start_times == schedule.start_times
               for entry in self.entries):
            return
        copied = Schedule(list(schedule.start_times), schedule.makespan,
                          list(schedule.order) if schedule.order is not None else None)
        self.entries.append(EliteEntry(copied))
        self.entries.sort(key=lambda e: e.schedule.makespan)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[: self.capacity]

    def best(self) -> Optional[Schedule]:
        return self.entries[0].schedule if self.entries else None


# ==================================
# 7. UCB-бандит для выбора колоний
# ==================================

class ColonyBanditController:
    """Empirical UCB-like allocation for learning arms; no UCB1 regret claim.

    Discounting is optional. It does not restore the stationary UCB1 assumptions.
    """
    def __init__(self, c: float = 1.0, selection="ucb", discount=1.0, rnd=None):
        if not math.isfinite(c) or c < 0:
            raise ValueError("UCB exploration coefficient must be finite and nonnegative")
        self.c = c
        if selection not in {"ucb", "random", "round_robin"}:
            raise ValueError("Unknown colony selection policy")
        if not math.isfinite(discount) or not 0 < discount <= 1:
            raise ValueError("Discount must lie in (0, 1]")
        self.selection = selection
        self.discount = discount
        self.rnd = rnd if rnd is not None else random.Random(0)
        self.mass = []
        self.colonies: List[AntColony] = []
        self.pulls: List[int] = []
        self.rewards: List[float] = []
        self.total_pulls: int = 0

    def add_colony(self, colony: AntColony):
        self.colonies.append(colony)
        self.pulls.append(0)
        self.rewards.append(0.0)
        self.mass.append(0.0)

    def select_colony_index(self) -> int:
        if not self.colonies:
            raise ValueError("No colonies initialized")
        if self.selection == "random":
            return self.rnd.randrange(len(self.colonies))
        if self.selection == "round_robin":
            return self.total_pulls % len(self.colonies)
        untried = [i for i, p in enumerate(self.pulls) if p == 0]
        if untried:
            return self.rnd.choice(untried)

        best_idx = 0
        best_ucb = -1e18
        for i, col in enumerate(self.colonies):
            mass = max(self.mass[i], 1e-12)
            mean = self.rewards[i] / mass
            bonus = self.c * math.sqrt(math.log(self.total_pulls) / mass)
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i
        return best_idx

    def update(self, idx: int, reward: float):
        if not 0 <= idx < len(self.colonies) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Invalid colony index/reward")
        self.mass = [value * self.discount for value in self.mass]
        self.rewards = [value * self.discount for value in self.rewards]
        self.total_pulls += 1
        self.pulls[idx] += 1
        self.mass[idx] += 1
        self.rewards[idx] += reward


# ===========================
# 8. Менеджер саморепликации
# ===========================

class ColonyManager:
    """
    Мета-уровень:
    - несколько колоний с разными параметрами,
    - выбор колонии через UCB-бандит,
    - репликация "успешных" колоний (мутация параметров),
    - общий банк элитных решений.
    """

    def __init__(
        self,
        inst: RcpspInstance,
        rnd: random.Random,
        max_colonies: int = 10,
        ucb_c: float = 1.0,
        elite_capacity: int = 30,
        tau0: float = 0.1,
        max_evaluations: Optional[int] = None,
        replication_enabled: bool = True,
        elite_enabled: bool = True,
        local_search_enabled: bool = True,
        time_limit: Optional[float] = None,
        selection: str = "ucb",
        discount: float = 1.0,
        reward_mode: str = "global_improvement",
        replication_mode: str = "mutate",
        elite_strength: float = 0.1,
    ):
        if type(max_colonies) is not int or max_colonies < 1:
            raise ValueError("max_colonies must be positive")
        self.budget = EvaluationBudget(max_evaluations, time_limit)
        if reward_mode not in {"global_improvement", "quality"}:
            raise ValueError("Unknown reward mode")
        if replication_mode not in {"mutate", "restart"}:
            raise ValueError("Unknown replication mode")
        if not math.isfinite(elite_strength) or not 0 <= elite_strength <= 1:
            raise ValueError("Elite strength must lie in [0, 1]")
        if not math.isfinite(tau0) or not 0 < tau0 <= 1:
            raise ValueError("Manager tau0 must lie in (0, 1]")
        self.reward_mode = reward_mode
        self.replication_mode = replication_mode
        self.elite_strength = elite_strength
        self.events = []
        self.master_seed = rnd.getrandbits(64)
        self.init_rnd = random.Random(str(self.master_seed) + ":init")
        self.mutation_rnd = random.Random(str(self.master_seed) + ":mutation")
        self.replication_enabled = replication_enabled
        self.elite_enabled = elite_enabled
        self.local_search_enabled = local_search_enabled
        self.inst = inst
        self.rnd = rnd
        self.max_colonies = max_colonies
        self.bandit = ColonyBanditController(c=ucb_c, selection=selection, discount=discount,
                                            rnd=random.Random(str(self.master_seed) + ":selection"))
        self.elite = EliteArchive(capacity=elite_capacity)
        self.tau0 = tau0

    def _schedule_to_order(self, sched: Schedule) -> List[int]:
        validate_schedule(self.inst, sched)
        if sched.order is not None:
            return list(sched.order)
        # For external schedules, ties are resolved only among eligible jobs.
        return topological_order(self.inst, [(sched.start_times[j], j)
                                            for j in range(self.inst.n_jobs)])

    def _elite_orders_subset(self, k: int = 3) -> List[List[int]]:
        orders: List[List[int]] = []
        if not self.elite_enabled:
            return orders
        for entry in self.elite.entries[:k]:
            orders.append(self._schedule_to_order(entry.schedule))
        return orders

    def init_random_colonies(self, k: int):
        if type(k) is not int or k < 1 or len(self.bandit.colonies) + k > self.max_colonies:
            raise ValueError("Initial colony count must be positive and within max_colonies")
        for _ in range(k):
            params = ColonyParams.random_init(self.init_rnd)
            if not self.local_search_enabled:
                params.ls_iters = 0
                params.ls_prob = 0
            elite_orders = self._elite_orders_subset(k=2)  # сначала пусто, потом появятся
            col = AntColony(
                self.inst,
                params,
                random.Random(str(self.master_seed) + ":colony:" + str(len(self.bandit.colonies))),
                tau0=self.tau0,
                elite_orders=elite_orders if elite_orders else None,
                elite_boost=0.3,
                budget=self.budget,
            )
            self.bandit.add_colony(col)
            self.events.append({"event": "init", "colony": len(self.bandit.colonies) - 1,
                                "evaluation": self.budget.used, "params": asdict(params)})

    def _replicate_from(self, parent: AntColony):
        if not self.replication_enabled or len(self.bandit.colonies) >= self.max_colonies:
            return
        child_params = (parent.params.mutate(self.mutation_rnd) if self.replication_mode == "mutate"
                        else ColonyParams.random_init(self.mutation_rnd))
        if not self.local_search_enabled:
            child_params.ls_iters = 0
            child_params.ls_prob = 0
        elite_orders = self._elite_orders_subset(k=3)
        child = AntColony(
            self.inst,
            child_params,
            random.Random(str(self.master_seed) + ":colony:" + str(len(self.bandit.colonies))),
            tau0=self.tau0,
            elite_orders=elite_orders if elite_orders else None,
            elite_boost=0.5,
            budget=self.budget,
        )
        self.bandit.add_colony(child)
        self.events.append({"event": "replicate", "parent": self.bandit.colonies.index(parent),
                            "child": len(self.bandit.colonies) - 1,
                            "mode": self.replication_mode, "evaluation": self.budget.used,
                            "params": asdict(child_params)})

    def run(
            self,
            iterations: int,
            ants_per_iteration: int,
            improvement_threshold: float = 0.02,
    ) -> Tuple[Optional[Schedule], Optional[AntColony]]:

        if (type(iterations) is not int or iterations < 1
                or type(ants_per_iteration) is not int or ants_per_iteration < 1):
            raise ValueError("Iterations and ants must be positive")
        if not math.isfinite(improvement_threshold) or not 0 <= improvement_threshold <= 1:
            raise ValueError("Improvement threshold must lie in [0, 1]")
        if not self.bandit.colonies:
            raise ValueError("No colonies initialized")
        best_colony = min(self.bandit.colonies, key=lambda col: col.best_cost)
        best_sched = best_colony.best_schedule
        best_cost = best_colony.best_cost
        self.budget.start()
        lower_bound = max(graph_features(self.inst)[1])

        for it in range(iterations):
            if self.budget.exhausted:
                break
            idx = self.bandit.select_colony_index()
            col = self.bandit.colonies[idx]

            previous_global = best_cost if math.isfinite(best_cost) else sum(self.inst.durations)
            before_evaluations = self.budget.used
            iter_best_sched, iter_best_cost = col.run_iteration(ants_per_iteration)
            if iter_best_sched is None:
                break

            if iter_best_sched is not None and self.elite_enabled:
                self.elite.add(iter_best_sched)

            new_best = col.best_cost

            # First observation is compared to the feasible sequential horizon.
            improvement = ((previous_global - new_best) / previous_global
                           if previous_global > 0 and new_best < previous_global else 0.0)
            reward = (improvement if self.reward_mode == "global_improvement" else
                      (min(1.0, lower_bound / iter_best_cost) if iter_best_cost > 0 else 1.0))
            self.bandit.update(idx, reward)
            self.events.append({"event": "pull", "colony": idx, "reward": reward,
                                "global_improvement": improvement,
                                "evaluation": self.budget.used,
                                "evaluations_used": self.budget.used - before_evaluations,
                                "iteration_best": iter_best_cost, "colony_best": new_best})

            if improvement > improvement_threshold and not self.budget.exhausted:
                self._replicate_from(col)

            if col.best_schedule is not None and col.best_cost < best_cost:
                best_cost = col.best_cost
                best_sched = col.best_schedule
                best_colony = col

            # Elite reinforcement follows a selected colony's own evaporation.
            # Idle arms are left untouched. Amount has the same units as local deposit.
            if self.elite_enabled and self.elite.best() is not None:
                best_elite_order = self._schedule_to_order(self.elite.best())
                amount = self.elite_strength * col.tau0 * col.duration_scale / max(1, self.elite.best().makespan)
                col.pheromones.deposit_on_order(best_elite_order, amount=amount)

        if best_sched is not None:
            validate_schedule(self.inst, best_sched)
        return best_sched, best_colony


# ==========================
# 9. Обёртка для запуска
# ==========================

def solve_rcpsp_psplib(
    path: str,
    seed: int = 42,
    init_colonies: int = 3,
    max_colonies: int = 10,
    iterations: int = 500,
    ants_per_iteration: int = 20,
    max_evaluations: Optional[int] = None,
    replication_enabled: bool = True,
    elite_enabled: bool = True,
    local_search_enabled: bool = True,
) -> Tuple[Schedule, ColonyParams]:
    rnd = random.Random(seed)
    inst = load_psplib_instance(path)

    manager = ColonyManager(
        inst,
        rnd,
        max_colonies=max_colonies,
        ucb_c=1.0,
        elite_capacity=30,
        tau0=0.1,
        max_evaluations=max_evaluations,
        replication_enabled=replication_enabled,
        elite_enabled=elite_enabled,
        local_search_enabled=local_search_enabled,
    )
    manager.init_random_colonies(init_colonies)

    best_sched, best_colony = manager.run(
        iterations=iterations,
        ants_per_iteration=ants_per_iteration,
        improvement_threshold=0.02,
    )

    if best_sched is None or best_colony is None:
        raise RuntimeError("Не удалось построить расписание")
    print("Final colonies count:", len(manager.bandit.colonies))

    print("Elite archive size:", len(manager.elite.entries))
    print("Schedule evaluations:", manager.budget.used)
    return best_sched, best_colony.params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Самореплицирующиеся ACO-колонии для RCPSP (PSPLIB j30)"
    )
    parser.add_argument("file", help="путь к файлу PSPLIB, например j301_1.sm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--ants", type=int, default=20)
    parser.add_argument("--init-colonies", type=int, default=3)
    parser.add_argument("--max-colonies", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=None)
    parser.add_argument("--disable-replication", action="store_true")
    parser.add_argument("--disable-elite", action="store_true")
    parser.add_argument("--disable-ls", action="store_true")

    args = parser.parse_args()

    sched, params = solve_rcpsp_psplib(
        args.file,
        seed=args.seed,
        init_colonies=args.init_colonies,
        max_colonies=args.max_colonies,
        iterations=args.iterations,
        ants_per_iteration=args.ants,
        max_evaluations=args.evaluations,
        replication_enabled=not args.disable_replication,
        elite_enabled=not args.disable_elite,
        local_search_enabled=not args.disable_ls,
    )

    print(f"Best makespan: {sched.makespan}")
    print("Best colony params:", params)
    print("Start times:", sched.start_times)
