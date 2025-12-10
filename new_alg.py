import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple


# =======================
# 1. Модель задачи RCPSP
# =======================

@dataclass
class RcpspInstance:
    n_jobs: int
    n_renewable: int
    durations: List[int]              # [job]
    demands: List[List[int]]          # [job][res]
    capacities: List[int]             # [res]
    successors: List[List[int]]       # [job] -> list of succ
    predecessors: List[List[int]]     # [job] -> list of preds


def load_psplib_instance(path: str) -> RcpspInstance:
    """
    Читает j-файлы PSPLIB (например, j301_1.sm) с помощью библиотеки psplib.
    Установить: pip install psplib
    """
    try:
        from psplib import parse as psplib_parse
    except ImportError as e:
        raise RuntimeError(
            "Не найден модуль psplib. Установите: pip install psplib"
        ) from e

    inst = psplib_parse(path, instance_format="psplib")  #

    n_jobs = inst.num_activities
    n_res = inst.num_resources

    durations: List[int] = []
    demands: List[List[int]] = []
    successors: List[List[int]] = []

    for act in inst.activities:
        mode = act.modes[0]  # j30 — single-mode
        durations.append(mode.duration)
        demands.append(list(mode.demands))
        successors.append(list(act.successors))

    capacities = [r.capacity for r in inst.resources]

    # строим списки предшественников
    predecessors: List[List[int]] = [[] for _ in range(n_jobs)]
    for j, succs in enumerate(successors):
        for s in succs:
            predecessors[s].append(j)

    return RcpspInstance(
        n_jobs=n_jobs,
        n_renewable=n_res,
        durations=durations,
        demands=demands,
        capacities=capacities,
        successors=successors,
        predecessors=predecessors,
    )


# ==========================
# 2. Расписание + SGS (серийный)
# ==========================

@dataclass
class Schedule:
    start_times: List[int]  # [job]
    makespan: int


class SerialScheduleGenerator:
    """
    Простой серийный SGS:
    - идём по порядку работ (topological/псевдо-топологический),
    - для каждой работы ищем минимальное возможное время начала,
      соблюдая предшественников и мощности ресурсов.
    """

    def generate(self, inst: RcpspInstance, order: Sequence[int]) -> Schedule:
        n_jobs = inst.n_jobs
        starts = [0] * n_jobs

        # профили использования ресурсов: usage[res][time]
        usage: List[List[int]] = [[] for _ in range(inst.n_renewable)]

        def ensure_horizon(h: int):
            for r in range(inst.n_renewable):
                while len(usage[r]) <= h:
                    usage[r].append(0)

        for job in order:
            dur = inst.durations[job]
            if dur < 0:
                raise ValueError("Negative duration")

            # 1) предшественники
            earliest = 0
            for p in inst.predecessors[job]:
                finish = starts[p] + inst.durations[p]
                if finish > earliest:
                    earliest = finish

            # 2) поиск earliest времени старта с учётом ресурсов
            t = earliest
            if dur == 0:
                starts[job] = t
                continue

            while True:
                end_t = t + dur - 1
                ensure_horizon(end_t)
                feasible = True
                for r in range(inst.n_renewable):
                    dem = inst.demands[job][r]
                    if dem == 0:
                        continue
                    cap = inst.capacities[r]
                    prof = usage[r]
                    for tt in range(t, t + dur):
                        if prof[tt] + dem > cap:
                            feasible = False
                            break
                    if not feasible:
                        break
                if feasible:
                    # ставим работу
                    for r in range(inst.n_renewable):
                        dem = inst.demands[job][r]
                        if dem == 0:
                            continue
                        for tt in range(t, t + dur):
                            usage[r][tt] += dem
                    starts[job] = t
                    break
                t += 1

        makespan = max(starts[j] + inst.durations[j] for j in range(n_jobs))
        return Schedule(start_times=starts, makespan=makespan)


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
        # лениво считаем характеристики графа и запоминаем в инстансе
        if not hasattr(inst, "_succ_count"):
            self._precompute_graph_features()

    def _precompute_graph_features(self):
        inst = self.inst
        n = inst.n_jobs

        # 1) количество (транзитивных) потомков
        succ_count = [0] * n

        def dfs_succ(v, visited):
            for s in inst.successors[v]:
                if s not in visited:
                    visited.add(s)
                    dfs_succ(s, visited)

        for j in range(n):
            visited = set()
            dfs_succ(j, visited)
            succ_count[j] = len(visited)

        # 2) "критичность": длина максимального пути от работы до конца (по длительностям)
        dist_to_end = [0] * n  # dist[j] = max суммарная длительность с j до конца (включая j)
        order = list(range(n))

        # топологический порядок (грубый, можно использовать Kahn, но j30 маленькие)
        # упрощённый вариант: несколько раз "релаксируем" в обратном направлении
        for _ in range(n):
            for j in range(n):
                for s in inst.successors[j]:
                    if dist_to_end[j] < inst.durations[j] + dist_to_end[s]:
                        dist_to_end[j] = inst.durations[j] + dist_to_end[s]

        self.inst._succ_count = succ_count
        self.inst._dist_to_end = dist_to_end

    def construct_order(self) -> List[int]:
        n_jobs = self.inst.n_jobs
        scheduled = set()
        order: List[int] = []

        for pos in range(n_jobs):
            eligible = self._eligible_jobs(scheduled)
            if not eligible:
                # должно быть топологически корректно, но на всякий
                break
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
        inst = self.inst
        t = self.params.heuristic_type

        if t == "none":
            return 1.0

        if t == "min_duration":
            return 1.0 / (1.0 + inst.durations[job])

        if t == "combo":
            dur = inst.durations[job]

            succ_count = getattr(inst, "_succ_count")
            dist_to_end = getattr(inst, "_dist_to_end")

            # нормализация для стабильности
            max_dur = max(inst.durations) or 1
            max_succ = max(succ_count) or 1
            max_dist = max(dist_to_end) or 1

            ndur = dur / max_dur
            nsucc = succ_count[job] / max_succ
            ncrit = dist_to_end[job] / max_dist

            # хотим, чтобы меньшая длительность → лучше, но больше критичность → лучше
            score = (
                    self.params.w_duration * (1.0 - ndur)
                    + self.params.w_succ * nsucc
                    + self.params.w_criticality * ncrit
            )
            # защита от нуля/отрицательных
            return max(score, 1e-6)

        # по умолчанию
        return 1.0

    def _score(self, pos: int, job: int) -> float:
        tau = self.pheromones.get(pos, job)
        eta = self._heuristic(job)
        alpha = self.params.alpha
        beta = self.params.beta
        # защита от нулей
        tau = max(tau, 1e-9)
        eta = max(eta, 1e-9)
        return (tau ** alpha) * (eta ** beta)

    def _select_job(self, pos: int, candidates: Sequence[int]) -> int:
        rnd = self.rnd
        q = rnd.random()
        scores = [self._score(pos, j) for j in candidates]

        if q < self.params.q0:
            # жадный выбор
            best_idx = max(range(len(candidates)), key=lambda i: scores[i])
            return candidates[best_idx]

        # рулетка
        total = sum(scores)
        if total <= 0:
            return rnd.choice(list(candidates))
        r = rnd.random() * total
        s = 0.0
        for idx, sc in enumerate(scores):
            s += sc
            if s >= r:
                return candidates[idx]
        return candidates[-1]


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
    ):
        self.inst = inst
        self.params = params
        self.rnd = rnd
        self.pheromones = PheromoneMatrix(inst.n_jobs, inst.n_jobs, tau0)
        self.sgs = SerialScheduleGenerator()

        self.best_schedule: Optional[Schedule] = None
        self.best_order: Optional[List[int]] = None
        self.best_cost: float = math.inf

        # начальная "подкрутка" феромонов по элитным решениям
        if elite_orders:
            for order in elite_orders:
                self.pheromones.deposit_on_order(order, amount=elite_boost)

    def _order_is_precedence_feasible(self, order: List[int]) -> bool:
        inst = self.inst
        pos = {job: i for i, job in enumerate(order)}
        for j in range(inst.n_jobs):
            for p in inst.predecessors[j]:
                if pos[p] > pos[j]:
                    return False
        return True

    def _local_search_order(self, order: List[int], max_iters: int) -> List[int]:
        if max_iters <= 0:
            return order

        best_order = list(order)
        best_sched = self.sgs.generate(self.inst, best_order)
        best_cost = best_sched.makespan

        for _ in range(max_iters):
            i = self.rnd.randrange(0, len(best_order))
            j = self.rnd.randrange(0, len(best_order))
            if i == j:
                continue
            if i > j:
                i, j = j, i

            cand = list(best_order)
            cand[i], cand[j] = cand[j], cand[i]

            if not self._order_is_precedence_feasible(cand):
                continue

            sched = self.sgs.generate(self.inst, cand)
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

        return best_order

    def run_iteration(self, n_ants: int) -> Tuple[Optional[Schedule], float]:
        iter_best_sched: Optional[Schedule] = None
        iter_best_cost = math.inf
        iter_best_order: Optional[List[int]] = None

        for _ in range(n_ants):
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
        if iter_best_order is not None and self.rnd.random() < self.params.ls_prob:
            improved_order = self._local_search_order(iter_best_order, self.params.ls_iters)
            improved_sched = self.sgs.generate(self.inst, improved_order)
            improved_cost = improved_sched.makespan
            if improved_cost < iter_best_cost:
                iter_best_cost = improved_cost
                iter_best_sched = improved_sched
                iter_best_order = improved_order

        # обновление феромона
        if iter_best_order is not None:
            self.pheromones.evaporate(self.params.rho)
            self.pheromones.deposit_on_order(iter_best_order, amount=1.0 / iter_best_cost)

        return iter_best_sched, iter_best_cost


# =======================
# 6. Банк элитных решений
# =======================

@dataclass
class EliteEntry:
    schedule: Schedule


class EliteArchive:
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.entries: List[EliteEntry] = []

    def add(self, schedule: Schedule):
        self.entries.append(EliteEntry(schedule))
        self.entries.sort(key=lambda e: e.schedule.makespan)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[: self.capacity]

    def best(self) -> Optional[Schedule]:
        return self.entries[0].schedule if self.entries else None


# ==================================
# 7. UCB-бандит для выбора колоний
# ==================================

class ColonyBanditController:
    def __init__(self, c: float = 1.0):
        self.c = c
        self.colonies: List[AntColony] = []
        self.pulls: List[int] = []
        self.rewards: List[float] = []
        self.total_pulls: int = 0

    def add_colony(self, colony: AntColony):
        self.colonies.append(colony)
        self.pulls.append(0)
        self.rewards.append(0.0)

    def select_colony_index(self) -> int:
        # сначала пробуем все колонии хотя бы раз
        for i, p in enumerate(self.pulls):
            if p == 0:
                return i

        best_idx = 0
        best_ucb = -1e18
        for i, col in enumerate(self.colonies):
            mean = self.rewards[i] / self.pulls[i]
            bonus = self.c * math.sqrt(math.log(self.total_pulls) / self.pulls[i])
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i
        return best_idx

    def update(self, idx: int, reward: float):
        self.total_pulls += 1
        self.pulls[idx] += 1
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
    ):
        self.inst = inst
        self.rnd = rnd
        self.max_colonies = max_colonies
        self.bandit = ColonyBanditController(c=ucb_c)
        self.elite = EliteArchive(capacity=elite_capacity)
        self.tau0 = tau0

    def _schedule_to_order(self, sched: Schedule) -> List[int]:
        # сортируем по времени старта, при равенстве — по номеру работы
        return sorted(
            range(self.inst.n_jobs),
            key=lambda j: (sched.start_times[j], j),
        )

    def _elite_orders_subset(self, k: int = 3) -> List[List[int]]:
        orders: List[List[int]] = []
        for entry in self.elite.entries[:k]:
            orders.append(self._schedule_to_order(entry.schedule))
        return orders

    def init_random_colonies(self, k: int):
        for _ in range(k):
            params = ColonyParams.random_init(self.rnd)
            elite_orders = self._elite_orders_subset(k=2)  # сначала пусто, потом появятся
            col = AntColony(
                self.inst,
                params,
                self.rnd,
                tau0=self.tau0,
                elite_orders=elite_orders if elite_orders else None,
                elite_boost=0.3,
            )
            self.bandit.add_colony(col)

    def _replicate_from(self, parent: AntColony):
        if len(self.bandit.colonies) >= self.max_colonies:
            return
        child_params = parent.params.mutate(self.rnd)
        elite_orders = self._elite_orders_subset(k=3)
        child = AntColony(
            self.inst,
            child_params,
            self.rnd,
            tau0=self.tau0,
            elite_orders=elite_orders if elite_orders else None,
            elite_boost=0.5,
        )
        self.bandit.add_colony(child)

    def run(
            self,
            iterations: int,
            ants_per_iteration: int,
            improvement_threshold: float = 0.02,
    ) -> Tuple[Optional[Schedule], Optional[AntColony]]:

        best_sched: Optional[Schedule] = None
        best_colony: Optional[AntColony] = None
        best_cost = math.inf

        for it in range(iterations):
            idx = self.bandit.select_colony_index()
            col = self.bandit.colonies[idx]

            prev_best = col.best_cost
            iter_best_sched, iter_best_cost = col.run_iteration(ants_per_iteration)

            if iter_best_sched is not None:
                self.elite.add(iter_best_sched)

            new_best = col.best_cost

            if prev_best == math.inf:
                reward = 1.0 if new_best < math.inf else 0.0
            else:
                reward = (prev_best - new_best) / prev_best if new_best < prev_best else 0.0
            reward -= 0.001
            self.bandit.update(idx, reward)

            if reward > improvement_threshold:
                self._replicate_from(col)

            if col.best_schedule is not None and col.best_cost < best_cost:
                best_cost = col.best_cost
                best_sched = col.best_schedule
                best_colony = col

            # глобальное элитное усиление
            if self.elite.best() is not None and (it + 1) % 10 == 0:
                best_elite_order = self._schedule_to_order(self.elite.best())
                for c in self.bandit.colonies:
                    c.pheromones.deposit_on_order(best_elite_order, amount=0.1)

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
    )
    manager.init_random_colonies(init_colonies)

    best_sched, best_colony = manager.run(
        iterations=iterations,
        ants_per_iteration=ants_per_iteration,
        improvement_threshold=0.02,
    )

    if best_sched is None or best_colony is None:
        raise RuntimeError("Не удалось построить расписание")

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

    args = parser.parse_args()

    sched, params = solve_rcpsp_psplib(
        args.file,
        seed=args.seed,
        init_colonies=args.init_colonies,
        max_colonies=args.max_colonies,
        iterations=args.iterations,
        ants_per_iteration=args.ants,
    )

    print(f"Best makespan: {sched.makespan}")
    print("Best colony params:", params)
    print("Start times:", sched.start_times)