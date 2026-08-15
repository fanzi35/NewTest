
from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "reference_formats")
FIGURE_DIR = os.path.join(BASE_DIR, "outputs", "figures")

DIST_PATH = os.path.join(DATA_DIR, "distances.csv")
DEMAND_PATH = os.path.join(DATA_DIR, "peopleQ1.csv")
OUT_ROUTES = os.path.join(OUTPUT_DIR, "q1-routes.csv")
OUT_ASSIGN = os.path.join(OUTPUT_DIR, "q1-assignments.csv")


# 常量

AIRPORTS = ("A01", "A02", "A03")
GAS_STATIONS = ("F006", "F011", "F018", "F024", "F031", "F038", "F044", "F050")
GAS_SET = set(GAS_STATIONS)

AC = {
    "T1": {"seats": 12, "speed": 250.0, "burn": 3.4, "tank": 1000.0, "reserve": 150.0},
    "T2": {"seats": 16, "speed": 220.0, "burn": 2.5, "tank": 1150.0, "reserve": 150.0},
    "T3": {"seats": 19, "speed": 190.0, "burn": 2.9, "tank": 1600.0, "reserve": 200.0},
}

MAX_LANDINGS = 5
MAX_SEATS = max(v["seats"] for v in AC.values())
EPS = 1e-9

D: Dict[str, Dict[str, float]] = {}


# 数据结构

@dataclass(frozen=True)
class RouteTemplate:
    airport: str
    aircraft_type: str
    stop_fids: Tuple[str, ...]
    refuels: Tuple[bool, ...]
    time_min: int
    fuel_kg: float
    pass_km: float
    avail_km: float
    intransit_min: int

    @property
    def seat_util(self) -> float:
        if self.avail_km <= EPS:
            return 0.0
        return self.pass_km / self.avail_km


@dataclass
class Sortie:
    airport: str
    aircraft_type: str
    stop_fids: List[str]
    refuels: List[bool]
    demands: List[dict]
    time_min: int
    fuel_kg: float
    pass_km: float
    avail_km: float
    intransit_min: int

    @property
    def seat_util(self) -> float:
        if self.avail_km <= EPS:
            return 0.0
        return self.pass_km / self.avail_km


@dataclass(frozen=True)
class AggregateMetrics:
    time_min: int = 0
    fuel_kg: float = 0.0
    pass_km: float = 0.0
    avail_km: float = 0.0
    intransit_min: int = 0

    @property
    def seat_util(self) -> float:
        if self.avail_km <= EPS:
            return 0.0
        return self.pass_km / self.avail_km


# 基础工具

def flight_min(dist_km: float, aircraft_type: str) -> int:
    return math.ceil(60.0 * dist_km / AC[aircraft_type]["speed"] - EPS)


def add_metrics(a: AggregateMetrics, b: AggregateMetrics) -> AggregateMetrics:
    return AggregateMetrics(
        time_min=a.time_min + b.time_min,
        fuel_kg=a.fuel_kg + b.fuel_kg,
        pass_km=a.pass_km + b.pass_km,
        avail_km=a.avail_km + b.avail_km,
        intransit_min=a.intransit_min + b.intransit_min,
    )


def metrics_of_template(t: RouteTemplate) -> AggregateMetrics:
    return AggregateMetrics(t.time_min, t.fuel_kg, t.pass_km, t.avail_km, t.intransit_min)


def metrics_of_sortie(s: Sortie) -> AggregateMetrics:
    return AggregateMetrics(s.time_min, s.fuel_kg, s.pass_km, s.avail_km, s.intransit_min)


def aggregate_sorties(sorties: Iterable[Sortie]) -> AggregateMetrics:
    out = AggregateMetrics()
    for s in sorties:
        out = add_metrics(out, metrics_of_sortie(s))
    return out


def lex_better(a: AggregateMetrics, b: Optional[AggregateMetrics]) -> bool:
    """按总时间、在途时间、燃油和全局座位利用率依次比较方案。"""
    if b is None:
        return True
    if a.time_min != b.time_min:
        return a.time_min < b.time_min
    if a.intransit_min != b.intransit_min:
        return a.intransit_min < b.intransit_min
    if abs(a.fuel_kg - b.fuel_kg) > EPS:
        return a.fuel_kg < b.fuel_kg
    if abs(a.seat_util - b.seat_util) > EPS:
        return a.seat_util > b.seat_util
    return False


def route_key(t: RouteTemplate) -> Tuple:
    return (
        t.time_min,
        t.intransit_min,
        round(t.fuel_kg, 9),
        -t.seat_util,
        t.aircraft_type,
        t.airport,
        t.stop_fids,
        t.refuels,
    )


def fixed_airport_of_demands(demands: Sequence[dict]) -> Optional[str]:
    fixed = {d["origin"] for d in demands if d["origin"] in AIRPORTS}
    if len(fixed) > 1:
        return "__CONFLICT__"
    return next(iter(fixed)) if fixed else None


def demand_signature(demands: Sequence[dict]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(Counter(d["dest"] for d in demands).items()))


def aggregate_coverage(demands: Sequence[dict]) -> Tuple[Tuple[str, str, int], ...]:
    counts = Counter((d["origin"], d["dest"]) for d in demands)
    return tuple(sorted((origin, dest, count) for (origin, dest), count in counts.items()))


@dataclass(frozen=True)
class RouteColumn:
    coverage: Tuple[Tuple[str, str, int], ...]
    airport: str
    aircraft_type: str
    stop_fids: Tuple[str, ...]
    refuels: Tuple[bool, ...]
    time_min: int
    fuel_tenths: int
    fuel_kg: float
    pass_km: float
    avail_km: float
    intransit_min: int

    @property
    def signature(self) -> Tuple:
        return (
            self.coverage, self.airport, self.aircraft_type,
            self.stop_fids, self.refuels,
        )

    @property
    def coverage_key(self) -> Tuple[Tuple[str, str, int], ...]:
        return self.coverage


def route_column_from_sortie(sortie: Sortie) -> RouteColumn:
    return RouteColumn(
        coverage=aggregate_coverage(sortie.demands),
        airport=sortie.airport,
        aircraft_type=sortie.aircraft_type,
        stop_fids=tuple(sortie.stop_fids),
        refuels=tuple(sortie.refuels),
        time_min=int(sortie.time_min),
        fuel_tenths=int(round(sortie.fuel_kg * 10.0)),
        fuel_kg=float(sortie.fuel_kg),
        pass_km=float(sortie.pass_km),
        avail_km=float(sortie.avail_km),
        intransit_min=int(sortie.intransit_min),
    )


def _column_dominates(a: RouteColumn, b: RouteColumn) -> bool:
    no_worse = (
        a.time_min <= b.time_min
        and a.intransit_min <= b.intransit_min
        and a.fuel_tenths <= b.fuel_tenths
        and a.pass_km + EPS >= b.pass_km
        and a.avail_km <= b.avail_km + EPS
    )
    strict = (
        a.time_min < b.time_min
        or a.intransit_min < b.intransit_min
        or a.fuel_tenths < b.fuel_tenths
        or a.pass_km > b.pass_km + EPS
        or a.avail_km + EPS < b.avail_km
    )
    return no_worse and strict


class RoutePool:
    """保存去重后的非支配路线列。"""

    def __init__(self, max_columns: int = 30000, max_per_coverage: int = 12) -> None:
        self.max_columns = max(1, max_columns)
        self.max_per_coverage = max(1, max_per_coverage)
        self._columns: Dict[Tuple, RouteColumn] = {}
        self._by_coverage: Dict[Tuple, List[Tuple]] = defaultdict(list)
        self.total_generated = 0
        self.duplicates = 0
        self.dominated_rejected = 0
        self.dominated_removed = 0
        self.capacity_removed = 0

    def __len__(self) -> int:
        return len(self._columns)

    def columns(self) -> List[RouteColumn]:
        return sorted(self._columns.values(), key=lambda c: (
            c.time_min, c.intransit_min, c.fuel_tenths,
            -c.pass_km / max(c.avail_km, EPS), c.signature,
        ))

    def add_sortie(self, sortie: Sortie) -> bool:
        return self.add_column(route_column_from_sortie(sortie))

    def add_many(self, sorties: Iterable[Sortie]) -> int:
        return sum(self.add_sortie(s) for s in sorties)

    def add_column(self, column: RouteColumn) -> bool:
        self.total_generated += 1
        sig = column.signature
        if sig in self._columns:
            self.duplicates += 1
            return False

        group_sigs = list(self._by_coverage.get(column.coverage_key, []))
        existing = [self._columns[s] for s in group_sigs]
        if any(_column_dominates(old, column) for old in existing):
            self.dominated_rejected += 1
            return False

        removed = [old.signature for old in existing if _column_dominates(column, old)]
        for old_sig in removed:
            del self._columns[old_sig]
            self.dominated_removed += 1
        kept = [s for s in group_sigs if s not in set(removed)]
        self._columns[sig] = column
        kept.append(sig)
        self._by_coverage[column.coverage_key] = kept
        self._trim_coverage(column.coverage_key)
        cleanup_trigger = self.max_columns + max(100, self.max_columns // 20)
        if len(self._columns) > cleanup_trigger:
            self._trim_global()
        return sig in self._columns

    def _column_rank(self, c: RouteColumn) -> Tuple:
        util = c.pass_km / max(c.avail_km, EPS)
        return (c.time_min, c.intransit_min, c.fuel_tenths, -util, c.signature)

    def _trim_coverage(self, coverage: Tuple) -> None:
        sigs = self._by_coverage[coverage]
        if len(sigs) <= self.max_per_coverage:
            return
        ranked = sorted(sigs, key=lambda s: self._column_rank(self._columns[s]))
        keep = set(ranked[:self.max_per_coverage])
        for sig in sigs:
            if sig not in keep:
                del self._columns[sig]
                self.capacity_removed += 1
        self._by_coverage[coverage] = [s for s in ranked if s in keep]

    def _trim_global(self) -> None:
        def global_rank(sig: Tuple) -> Tuple:
            c = self._columns[sig]
            pax = sum(count for _, _, count in c.coverage)
            return (
                c.time_min / max(1, pax), -pax,
                c.intransit_min / max(1, pax),
                c.fuel_tenths / max(1, pax),
                -c.pass_km / max(c.avail_km, EPS), c.signature,
            )

        ranked = sorted(self._columns, key=global_rank)
        keep = set(ranked[:self.max_columns])
        for sig in list(self._columns):
            if sig not in keep:
                del self._columns[sig]
                self.capacity_removed += 1
        for coverage, sigs in list(self._by_coverage.items()):
            self._by_coverage[coverage] = [s for s in sigs if s in keep]

    def clean(self) -> None:
        if len(self._columns) > self.max_columns:
            self._trim_global()


_ACTIVE_ROUTE_POOL: Optional[RoutePool] = None


@dataclass(frozen=True)
class MasterResult:
    columns: Tuple[RouteColumn, ...]
    multiplicities: Tuple[int, ...]
    metrics: AggregateMetrics
    status: str
    completed_stage: str
    selected_column_count: int
    pool_column_count: int
    full_pool_used: bool
    optimal_within_selected_subset: bool
    optimal_within_pool: bool
    last_bound: Optional[float]
    last_gap: Optional[float]
    solve_calls: int


class ProgressReporter:

    def __init__(self, pool: RoutePool, start_time: float, interval: float = 7.0) -> None:
        self.pool = pool
        self.start_time = start_time
        self.interval = max(1.0, interval)
        self.last_print = start_time
        self.last_generated = 0
        self.best_valid_solution: List[Sortie] = []
        self.neighborhoods_checked = 0
        self.last_master_status = "not-run"
        self.last_master_bound: Optional[float] = None
        self.last_master_gap: Optional[float] = None
        self.last_master_incumbent = "none"
        self.stagnation_rounds = 0
        self.level = 0
        self.solution_history: List[dict] = []
        self.status_history: List[dict] = []

    def set_best(self, sorties: Sequence[Sortie]) -> None:
        self.best_valid_solution = list(sorties)

    def record_solution(self, source: str, sorties: Sequence[Sortie], is_best: bool) -> None:
        if not is_best:
            return
        stats = compute_stats(sorties)
        self.solution_history.append({
            "wall_sec": time.time() - self.start_time,
            "source": source,
            "is_best": is_best,
            **stats,
        })
        if len(self.solution_history) > 20000:
            self.solution_history = self.solution_history[::2]

    def note_neighborhood(self, count: int = 1) -> None:
        self.neighborhoods_checked += count
        self.maybe_print()

    def set_master(self, result: MasterResult) -> None:
        self.last_master_status = result.status
        self.last_master_bound = result.last_bound
        self.last_master_gap = result.last_gap
        self.last_master_incumbent = (
            f"time={result.metrics.time_min}, intransit={result.metrics.intransit_min}, "
            f"fuel={result.metrics.fuel_kg:.1f}, util={result.metrics.seat_util:.6f}"
        )

    def maybe_print(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_print < self.interval:
            return
        best = compute_stats(self.best_valid_solution) if self.best_valid_solution else None
        new_generated = self.pool.total_generated - self.last_generated
        if best is None:
            best_text = "best=none"
        else:
            best_text = (
                f"best_time={best['total_time_min']}, "
                f"best_intransit={best['total_intransit_min']}, "
                f"best_fuel={best['total_fuel_kg']:.1f}, "
                f"best_util={best['seat_util']:.6f}, "
                f"best_sorties={best['total_sorties']}"
            )
        self.status_history.append({
            "wall_sec": now - self.start_time,
            "pool_generated": self.pool.total_generated,
            "pool_nondominated": len(self.pool),
            "neighborhoods_checked": self.neighborhoods_checked,
            "stagnation_rounds": self.stagnation_rounds,
            "level": self.level,
        })
        if len(self.status_history) > 20000:
            self.status_history = self.status_history[::2]
        print(
            f"  [STATUS] wall={now - self.start_time:.1f}s, {best_text}, "
            f"pool_generated={self.pool.total_generated}, "
            f"pool_nondominated={len(self.pool)}, new_columns={new_generated}, "
            f"neighborhoods_checked={self.neighborhoods_checked}, "
            f"master_status={self.last_master_status}, "
            f"master_bound={self.last_master_bound}, master_gap={self.last_master_gap}, "
            f"master_incumbent=({self.last_master_incumbent}), "
            f"stagnation_rounds={self.stagnation_rounds}, level={self.level}"
        )
        self.last_generated = self.pool.total_generated
        self.last_print = now


_PROGRESS: Optional[ProgressReporter] = None


def aggregate_demand_totals(demands: Sequence[dict]) -> Dict[Tuple[str, str], int]:
    return dict(Counter((d["origin"], d["dest"]) for d in demands))


def _select_master_columns(
    pool: RoutePool,
    incumbent_sorties: Sequence[Sortie],
    max_columns: int,
    portfolio_round: int = 0,
) -> Tuple[RouteColumn, ...]:
    incumbent_columns = [route_column_from_sortie(s) for s in incumbent_sorties]
    by_signature = {c.signature: c for c in pool.columns()}
    for column in incumbent_columns:
        by_signature[column.signature] = column
    all_columns = list(by_signature.values())
    limit = max(len(incumbent_columns), max(1, max_columns))
    if len(all_columns) <= limit:
        return tuple(all_columns)

    selected: Dict[Tuple, RouteColumn] = {c.signature: c for c in incumbent_columns}

    def pax(c: RouteColumn) -> int:
        return sum(count for _, _, count in c.coverage)

    rankings = (
        sorted(all_columns, key=lambda c: (
            c.time_min / max(1, pax(c)), -pax(c), c.time_min, c.signature,
        )),
        sorted(all_columns, key=lambda c: (
            c.intransit_min / max(1, pax(c)), c.time_min / max(1, pax(c)), c.signature,
        )),
        sorted(all_columns, key=lambda c: (
            c.fuel_tenths / max(1, pax(c)), c.time_min / max(1, pax(c)), c.signature,
        )),
        sorted(all_columns, key=lambda c: (
            -pax(c), c.time_min / max(1, pax(c)), c.signature,
        )),
        sorted(all_columns, key=lambda c: (
            -c.pass_km / max(c.avail_km, EPS), c.time_min / max(1, pax(c)), c.signature,
        )),
    )
    if portfolio_round:
        window = max(1, (limit - len(selected)) // max(1, len(rankings)))
        rotated = []
        for k, ranking in enumerate(rankings):
            offset = (portfolio_round * window * (2 * k + 1)) % len(ranking)
            rotated.append(ranking[offset:] + ranking[:offset])
        rankings = tuple(rotated)
    positions = [0] * len(rankings)
    while len(selected) < limit:
        added = False
        for k, ranking in enumerate(rankings):
            while positions[k] < len(ranking):
                column = ranking[positions[k]]
                positions[k] += 1
                if column.signature in selected:
                    continue
                selected[column.signature] = column
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break
    return tuple(selected.values())


def _master_metrics(columns: Sequence[RouteColumn], x: Sequence[int]) -> AggregateMetrics:
    return AggregateMetrics(
        time_min=sum(c.time_min * n for c, n in zip(columns, x)),
        fuel_kg=sum(c.fuel_kg * n for c, n in zip(columns, x)),
        pass_km=sum(c.pass_km * n for c, n in zip(columns, x)),
        avail_km=sum(c.avail_km * n for c, n in zip(columns, x)),
        intransit_min=sum(c.intransit_min * n for c, n in zip(columns, x)),
    )


def solve_restricted_master(
    pool: RoutePool,
    demands: Sequence[dict],
    incumbent_sorties: Sequence[Sortie] = (),
    max_master_columns: int = 1500,
    portfolio_round: int = 0,
    per_call_time: float = 5.0,
    retries: int = 3,
    dinkelbach_tol: float = 1e-8,
    dinkelbach_max_iter: int = 8,
) -> MasterResult:
    """在选定路线列上按四级字典序求解受限集合划分模型。"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    totals = aggregate_demand_totals(demands)
    groups = tuple(sorted(totals))
    group_index = {g: i for i, g in enumerate(groups)}
    pool_signatures = {column.signature for column in pool.columns()}
    selected_columns = _select_master_columns(
        pool, incumbent_sorties, max_master_columns, portfolio_round,
    )
    selected_signatures = {column.signature for column in selected_columns}
    full_pool_used = pool_signatures.issubset(selected_signatures)
    selected_column_count = len(selected_signatures)
    pool_column_count = len(pool_signatures)
    columns = tuple(
        c for c in selected_columns
        if all((origin, dest) in totals and count <= totals[(origin, dest)]
               for origin, dest, count in c.coverage)
    )
    if not columns:
        raise RuntimeError("Restricted Master 没有可用路线列")
    print("  Restricted Master:")
    print(f"    selected_columns = {selected_column_count}")
    print(f"    route_pool_columns = {pool_column_count}")
    print(f"    full_pool_used = {full_pool_used}")
    if len(columns) != selected_column_count:
        print(f"    usable_milp_columns = {len(columns)}")

    rows, cols, data = [], [], []
    upper = []
    for j, column in enumerate(columns):
        bounds = []
        for origin, dest, count in column.coverage:
            rows.append(group_index[(origin, dest)])
            cols.append(j)
            data.append(float(count))
            bounds.append(totals[(origin, dest)] // count)
        upper.append(float(min(bounds) if bounds else 0))
    matrix = csc_matrix((data, (rows, cols)), shape=(len(groups), len(columns)))
    demand_vector = np.array([totals[g] for g in groups], dtype=float)
    base_constraint = LinearConstraint(matrix, demand_vector, demand_vector)
    bounds = Bounds(np.zeros(len(columns)), np.array(upper, dtype=float))
    integrality = np.ones(len(columns), dtype=int)
    calls = 0
    last_result = None

    fallback_x = None
    if incumbent_sorties:
        needed = Counter(route_column_from_sortie(s).signature for s in incumbent_sorties)
        candidate = np.zeros(len(columns), dtype=int)
        column_index = {c.signature: j for j, c in enumerate(columns)}
        if all(sig in column_index for sig in needed):
            for sig, count in needed.items():
                candidate[column_index[sig]] = count
            if np.array_equal(
                np.asarray(matrix @ candidate).reshape(-1).astype(int),
                demand_vector.astype(int),
            ):
                fallback_x = candidate

    def integer_incumbent(result) -> Optional[np.ndarray]:
        if result is None or result.x is None:
            return None
        x = np.rint(result.x).astype(int)
        if np.any(x < 0) or np.any(x > np.array(upper, dtype=int)):
            return None
        covered = np.asarray(matrix @ x).reshape(-1)
        if not np.array_equal(covered.astype(int), demand_vector.astype(int)):
            return None
        return x

    def run_phase(
        name: str,
        objective: np.ndarray,
        fixed: List[LinearConstraint],
        fallback: Optional[np.ndarray] = None,
    ):
        nonlocal calls, last_result
        best_result = None
        best_x = None if fallback is None else fallback.copy()
        best_value = None if fallback is None else float(objective @ fallback)
        for attempt in range(max(1, retries)):
            budget = max(0.1, per_call_time) * (attempt + 1)
            started = time.time()
            result = milp(
                c=objective,
                integrality=integrality,
                bounds=bounds,
                constraints=[base_constraint] + fixed,
                options={"time_limit": budget, "mip_rel_gap": 0.0, "disp": False},
            )
            calls += 1
            last_result = result
            x = integer_incumbent(result)
            value = None if x is None else float(objective @ x)
            gap = getattr(result, "mip_gap", None)
            bound = getattr(result, "mip_dual_bound", None)
            print(
                f"  [Master] phase={name}, call={calls}, budget={budget:.1f}s, "
                f"elapsed={time.time() - started:.2f}s, status={result.message}, "
                f"incumbent={value}, bound={bound}, gap={gap}"
            )
            if _PROGRESS is not None:
                _PROGRESS.last_master_status = f"{name}: {result.message}"
                _PROGRESS.last_master_bound = bound
                _PROGRESS.last_master_gap = gap
                _PROGRESS.last_master_incumbent = str(value)
                _PROGRESS.maybe_print(force=True)
            if x is not None and (best_value is None or value < best_value - EPS):
                best_result, best_x, best_value = result, x, value
            if result.status == 0 and x is not None:
                return result, x, True
        return best_result, best_x, False

    time_obj = np.array([c.time_min for c in columns], dtype=float)
    transit_obj = np.array([c.intransit_min for c in columns], dtype=float)
    fuel_obj = np.array([c.fuel_tenths for c in columns], dtype=float)
    pass_obj = np.array([c.pass_km for c in columns], dtype=float)
    avail_obj = np.array([c.avail_km for c in columns], dtype=float)
    fixed: List[LinearConstraint] = []

    result, x, optimal = run_phase("1-aircraft-time", time_obj, fixed, fallback_x)
    if x is None:
        raise RuntimeError("Restricted Master 在第一阶段未找到可行整数解")
    completed_stage = "aircraft-time"
    status = "aircraft-time limit; lower objectives not attempted"
    if not optimal:
        return MasterResult(
            columns=columns,
            multiplicities=tuple(int(v) for v in x),
            metrics=_master_metrics(columns, x),
            status=status,
            completed_stage=completed_stage,
            selected_column_count=selected_column_count,
            pool_column_count=pool_column_count,
            full_pool_used=full_pool_used,
            optimal_within_selected_subset=False,
            optimal_within_pool=False,
            last_bound=getattr(result, "mip_dual_bound", None),
            last_gap=getattr(result, "mip_gap", None),
            solve_calls=calls,
        )
    time_opt = int(round(float(time_obj @ x)))
    fixed.append(LinearConstraint(time_obj[None, :], time_opt, time_opt))

    result2, x2, optimal2 = run_phase("2-passenger-intransit", transit_obj, fixed, x)
    if x2 is None:
        x2 = x
    else:
        x = x2
    completed_stage = "passenger-intransit"
    status = "passenger-intransit limit; lower objectives not attempted"
    if not optimal2:
        return MasterResult(
            columns=columns,
            multiplicities=tuple(int(v) for v in x),
            metrics=_master_metrics(columns, x),
            status=status,
            completed_stage=completed_stage,
            selected_column_count=selected_column_count,
            pool_column_count=pool_column_count,
            full_pool_used=full_pool_used,
            optimal_within_selected_subset=False,
            optimal_within_pool=False,
            last_bound=getattr(result2, "mip_dual_bound", None),
            last_gap=getattr(result2, "mip_gap", None),
            solve_calls=calls,
        )
    transit_opt = int(round(float(transit_obj @ x)))
    fixed.append(LinearConstraint(transit_obj[None, :], transit_opt, transit_opt))

    result3, x3, optimal3 = run_phase("3-fuel-tenths", fuel_obj, fixed, x)
    if x3 is None:
        x3 = x
    else:
        x = x3
    completed_stage = "fuel"
    status = "fuel limit; seat utilization not attempted"
    if not optimal3:
        return MasterResult(
            columns=columns,
            multiplicities=tuple(int(v) for v in x),
            metrics=_master_metrics(columns, x),
            status=status,
            completed_stage=completed_stage,
            selected_column_count=selected_column_count,
            pool_column_count=pool_column_count,
            full_pool_used=full_pool_used,
            optimal_within_selected_subset=False,
            optimal_within_pool=False,
            last_bound=getattr(result3, "mip_dual_bound", None),
            last_gap=getattr(result3, "mip_gap", None),
            solve_calls=calls,
        )
    fuel_opt = int(round(float(fuel_obj @ x)))
    fixed.append(LinearConstraint(fuel_obj[None, :], fuel_opt, fuel_opt))

    # Dinkelbach：将分式利用率目标转为迭代线性目标。
    best_x = x.copy()
    best_ratio = float(pass_obj @ x) / max(float(avail_obj @ x), EPS)
    fourth_optimal = False
    for iteration in range(max(1, dinkelbach_max_iter)):
        objective = -(pass_obj - best_ratio * avail_obj)
        result4, x4, transformed_optimal = run_phase(
            f"4-seat-util-{iteration + 1}", objective, fixed, best_x,
        )
        if x4 is None:
            status = "seat-utilization MILP has no incumbent"
            break
        numerator = float(pass_obj @ x4)
        denominator = float(avail_obj @ x4)
        ratio = numerator / max(denominator, EPS)
        residual = numerator - best_ratio * denominator
        if ratio > best_ratio + dinkelbach_tol:
            best_x, best_ratio = x4.copy(), ratio
        if transformed_optimal and abs(residual) <= dinkelbach_tol * max(1.0, denominator):
            fourth_optimal = True
            best_x = x4.copy()
            status = (
                "optimal within the current Route Pool"
                if full_pool_used
                else "optimal within the currently selected Restricted Master subset"
            )
            break
        if not transformed_optimal:
            status = "seat-utilization time limit"
            break
        best_ratio = ratio
    else:
        status = "seat-utilization iteration limit"

    last_bound = getattr(last_result, "mip_dual_bound", None)
    last_gap = getattr(last_result, "mip_gap", None)
    all_four_stages_optimal = bool(optimal and optimal2 and optimal3 and fourth_optimal)
    if all_four_stages_optimal:
        print("  MASTER STATUS:")
        print(
            "    optimal within current Route Pool"
            if full_pool_used
            else "    optimal within selected subset"
        )
    return MasterResult(
        columns=columns,
        multiplicities=tuple(int(v) for v in best_x),
        metrics=_master_metrics(columns, best_x),
        status=status,
        completed_stage="seat-utilization",
        selected_column_count=selected_column_count,
        pool_column_count=pool_column_count,
        full_pool_used=full_pool_used,
        optimal_within_selected_subset=all_four_stages_optimal,
        optimal_within_pool=bool(full_pool_used and all_four_stages_optimal),
        last_bound=last_bound,
        last_gap=last_gap,
        solve_calls=calls,
    )


def recover_master_solution(result: MasterResult, demands: Sequence[dict]) -> List[Sortie]:
    buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for demand in sorted(demands, key=lambda d: d["pid"]):
        buckets[(demand["origin"], demand["dest"])].append(demand)
    cursor = {g: 0 for g in buckets}
    sorties = []
    for column, multiplicity in zip(result.columns, result.multiplicities):
        for _ in range(multiplicity):
            assigned = []
            for origin, dest, count in column.coverage:
                group = (origin, dest)
                start = cursor[group]
                end = start + count
                if end > len(buckets[group]):
                    raise ValueError(f"Master 对聚合需求 {group} 超额覆盖")
                assigned.extend(buckets[group][start:end])
                cursor[group] = end
            provisional = Sortie(
                airport=column.airport,
                aircraft_type=column.aircraft_type,
                stop_fids=list(column.stop_fids),
                refuels=list(column.refuels),
                demands=assigned,
                time_min=0,
                fuel_kg=0.0,
                pass_km=0.0,
                avail_km=0.0,
                intransit_min=0,
            )
            simulated = simulate_sortie(provisional)
            sorties.append(template_to_sortie(simulated, assigned))
    unfilled = {g: len(v) - cursor[g] for g, v in buckets.items() if cursor[g] != len(v)}
    if unfilled:
        raise ValueError(f"Master 聚合需求覆盖不完整: {list(unfilled.items())[:5]}")
    sorties.sort(key=lambda s: (
        s.airport, s.aircraft_type, s.stop_fids[0] if s.stop_fids else "",
        len(s.demands), tuple(sorted(d["pid"] for d in s.demands)),
    ))
    validate_solution(sorties, demands)
    return sorties


def solve_master_portfolio(
    pool: RoutePool,
    demands: Sequence[dict],
    incumbent_sorties: Sequence[Sortie],
    rounds: int,
    start_round: int,
    max_master_columns: int,
    per_call_time: float,
    retries: int,
) -> Tuple[List[Sortie], List[MasterResult], int]:
    best = list(incumbent_sorties)
    best_metrics = aggregate_sorties(best)
    results = []
    improvements = 0
    for offset in range(max(1, rounds)):
        portfolio_round = start_round + offset
        print(f"\n  Master portfolio round {portfolio_round}...")
        result = solve_restricted_master(
            pool,
            demands,
            incumbent_sorties=best,
            max_master_columns=max_master_columns,
            portfolio_round=portfolio_round,
            per_call_time=per_call_time,
            retries=retries,
        )
        results.append(result)
        if _PROGRESS is not None:
            _PROGRESS.set_master(result)
        candidate = recover_master_solution(result, demands)
        candidate_metrics = aggregate_sorties(candidate)
        pool.add_many(candidate)
        if lex_better(candidate_metrics, best_metrics):
            print(
                f"  MASTER NEW BEST: {best_metrics.time_min} -> {candidate_metrics.time_min} min, "
                f"portfolio_round={portfolio_round}, status={result.status}"
            )
            best = candidate
            best_metrics = candidate_metrics
            improvements += 1
            if _PROGRESS is not None:
                _PROGRESS.set_best(best)
        elif candidate_metrics.time_min == best_metrics.time_min:
            print(
                f"  Master round {portfolio_round} reproduced primary optimum "
                f"without strict global improvement."
            )
        else:
            print(f"  Master round {portfolio_round} did not improve the validated incumbent.")
    return best, results, improvements


def template_to_sortie(t: RouteTemplate, demands: Sequence[dict]) -> Sortie:
    return Sortie(
        airport=t.airport,
        aircraft_type=t.aircraft_type,
        stop_fids=list(t.stop_fids),
        refuels=list(t.refuels),
        demands=list(demands),
        time_min=t.time_min,
        fuel_kg=t.fuel_kg,
        pass_km=t.pass_km,
        avail_km=t.avail_km,
        intransit_min=t.intransit_min,
    )


# 数据读取与校验

def load_distances(path: str) -> None:
    global D
    D = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        cols = [x.strip() for x in header[1:]]
        for row in r:
            if not row:
                continue
            u = row[0].strip()
            D[u] = {}
            for j, v in enumerate(cols):
                raw = row[j + 1].strip()
                D[u][v] = float(raw)

    nodes = set(D)
    if not set(AIRPORTS).issubset(nodes):
        raise ValueError("distances.csv 缺少 A01/A02/A03 中的机场行")
    for u, row in D.items():
        missing = nodes - set(row)
        if missing:
            raise ValueError(f"distances.csv 的 {u} 行缺少列: {sorted(missing)[:5]}...")


def load_demands(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"person_id", "origin_id", "destination_id"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"peopleQ1.csv 表头必须包含 {sorted(required)}")
        for row in r:
            d = {
                "pid": row["person_id"].strip(),
                "origin": row["origin_id"].strip(),
                "dest": row["destination_id"].strip(),
            }
            out.append(d)
    validate_q1_demands(out)
    return out


def validate_q1_demands(demands: Sequence[dict]) -> None:
    seen = set()
    for d in demands:
        if d["pid"] in seen:
            raise ValueError(f"重复 person_id: {d['pid']}")
        seen.add(d["pid"])

        if d["origin"] not in AIRPORTS and d["origin"] != "LAND":
            raise ValueError(
                f"Q1 应为出海需求，{d['pid']} 的 origin_id={d['origin']} 不是 LAND/A01/A02/A03"
            )
        if d["dest"] in AIRPORTS or d["dest"] == "LAND" or d["dest"] not in D:
            raise ValueError(
                f"Q1 应以海上设施为终点，{d['pid']} 的 destination_id={d['dest']} 非法"
            )


# 单架次路线优化

_ROUTE_CACHE: Dict[Tuple[str, str, Tuple[Tuple[str, int], ...]], Optional[RouteTemplate]] = {}
_TEMPLATE_CACHE: Dict[Tuple[Optional[str], Tuple[Tuple[str, int], ...]], Optional[RouteTemplate]] = {}


def _search_route_for_aircraft(
    airport: str,
    aircraft_type: str,
    dest_sig: Tuple[Tuple[str, int], ...],
) -> Optional[RouteTemplate]:
    """固定机场和机型后，在着陆限制内搜索单架次路线。"""
    cache_key = (airport, aircraft_type, dest_sig)
    if cache_key in _ROUTE_CACHE:
        return _ROUTE_CACHE[cache_key]

    counts = dict(dest_sig)
    total_pax = sum(counts.values())
    unique_dests = tuple(counts)
    if total_pax <= 0:
        _ROUTE_CACHE[cache_key] = None
        return None

    ac = AC[aircraft_type]
    seats = int(ac["seats"])
    if total_pax > seats or len(unique_dests) > MAX_LANDINGS:
        _ROUTE_CACHE[cache_key] = None
        return None

    full = float(ac["tank"])
    reserve = float(ac["reserve"])
    burn = float(ac["burn"])

    best: Optional[RouteTemplate] = None
    allow_pure_gas = False

    frontier: Dict[Tuple[str, Tuple[str, ...], int], List[Tuple[int, float]]] = defaultdict(list)

    def dominated(state_key: Tuple[str, Tuple[str, ...], int], elapsed: int, remain: float) -> bool:
        labels = frontier[state_key]
        for old_t, old_r in labels:
            if old_t < elapsed and old_r + EPS >= remain:
                return True
        frontier[state_key] = [
            (old_t, old_r)
            for old_t, old_r in labels
            if not (elapsed < old_t and remain + EPS >= old_r)
        ]
        frontier[state_key].append((elapsed, remain))
        return False

    def consider_complete(
        current: str,
        remain: float,
        elapsed: int,
        stops: Tuple[str, ...],
        refuels: Tuple[bool, ...],
        fuel_used: float,
        pass_km: float,
        avail_km: float,
        intransit: int,
    ) -> None:
        nonlocal best
        dist = D[current][airport]
        need = dist * burn
        if remain - need < reserve - EPS:
            return

        final_time = elapsed + flight_min(dist, aircraft_type)
        final_fuel = fuel_used + need
        final_avail = avail_km + seats * dist
        cand = RouteTemplate(
            airport=airport,
            aircraft_type=aircraft_type,
            stop_fids=stops,
            refuels=refuels,
            time_min=final_time,
            fuel_kg=final_fuel,
            pass_km=pass_km,
            avail_km=final_avail,
            intransit_min=intransit,
        )
        if best is None or route_key(cand) < route_key(best):
            best = cand

    def dfs(
        current: str,
        unserved: Tuple[str, ...],
        remain: float,
        landings_used: int,
        elapsed: int,
        stops: Tuple[str, ...],
        refuels: Tuple[bool, ...],
        fuel_used: float,
        pass_km: float,
        avail_km: float,
        intransit: int,
    ) -> None:
        nonlocal best

        if landings_used + len(unserved) > MAX_LANDINGS:
            return

        if best is not None and elapsed >= best.time_min:
            return

        state_key = (current, unserved, landings_used)
        if dominated(state_key, elapsed, remain):
            return

        if not unserved:
            consider_complete(
                current, remain, elapsed, stops, refuels,
                fuel_used, pass_km, avail_km, intransit,
            )

        if landings_used >= MAX_LANDINGS:
            return

        onboard = sum(counts[f] for f in unserved)
        unserved_set = set(unserved)

        for f in unserved:
            dist = D[current][f]
            need = dist * burn
            after = remain - need
            if after < reserve - EPS:
                continue

            fly = flight_min(dist, aircraft_type)
            arrival_time = elapsed + fly
            new_pass_km = pass_km + onboard * dist
            new_avail_km = avail_km + seats * dist
            new_fuel = fuel_used + need
            new_intransit = intransit + counts[f] * arrival_time
            new_unserved = tuple(x for x in unserved if x != f)
            new_stops = stops + (f,)

            if f not in GAS_SET:
                dfs(
                    current=f,
                    unserved=new_unserved,
                    remain=after,
                    landings_used=landings_used + 1,
                    elapsed=arrival_time + 10,
                    stops=new_stops,
                    refuels=refuels + (False,),
                    fuel_used=new_fuel,
                    pass_km=new_pass_km,
                    avail_km=new_avail_km,
                    intransit=new_intransit,
                )
            else:
                dfs(
                    current=f,
                    unserved=new_unserved,
                    remain=after,
                    landings_used=landings_used + 1,
                    elapsed=arrival_time + 10,
                    stops=new_stops,
                    refuels=refuels + (False,),
                    fuel_used=new_fuel,
                    pass_km=new_pass_km,
                    avail_km=new_avail_km,
                    intransit=new_intransit,
                )
                dfs(
                    current=f,
                    unserved=new_unserved,
                    remain=full,
                    landings_used=landings_used + 1,
                    elapsed=arrival_time + 20,
                    stops=new_stops,
                    refuels=refuels + (True,),
                    fuel_used=new_fuel,
                    pass_km=new_pass_km,
                    avail_km=new_avail_km,
                    intransit=new_intransit,
                )

        if allow_pure_gas and landings_used + 1 + len(unserved) <= MAX_LANDINGS:
            for g in GAS_STATIONS:
                if g == current:
                    continue
                if g in unserved_set:
                    continue

                dist = D[current][g]
                need = dist * burn
                after = remain - need
                if after < reserve - EPS:
                    continue

                targets = unserved if unserved else (airport,)
                if not any(full - D[g][v] * burn >= reserve - EPS for v in targets):
                    continue

                fly = flight_min(dist, aircraft_type)
                new_elapsed = elapsed + fly + 20
                new_pass_km = pass_km + onboard * dist
                new_avail_km = avail_km + seats * dist
                new_fuel = fuel_used + need

                dfs(
                    current=g,
                    unserved=unserved,
                    remain=full,
                    landings_used=landings_used + 1,
                    elapsed=new_elapsed,
                    stops=stops + (g,),
                    refuels=refuels + (True,),
                    fuel_used=new_fuel,
                    pass_km=new_pass_km,
                    avail_km=new_avail_km,
                    intransit=intransit,
                )

    initial_kwargs = dict(
        current=airport,
        unserved=tuple(sorted(unique_dests)),
        remain=full,
        landings_used=0,
        elapsed=0,
        stops=tuple(),
        refuels=tuple(),
        fuel_used=0.0,
        pass_km=0.0,
        avail_km=0.0,
        intransit=0,
    )

    # 先搜索交付节点；无可行解时再开放纯加油停靠。
    dfs(**initial_kwargs)

    if best is None:
        allow_pure_gas = True
        frontier.clear()
        dfs(**initial_kwargs)

    _ROUTE_CACHE[cache_key] = best
    return best


def optimize_template(
    dest_sig: Tuple[Tuple[str, int], ...],
    fixed_airport: Optional[str],
) -> Optional[RouteTemplate]:
    cache_key = (fixed_airport, dest_sig)
    if cache_key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[cache_key]

    total_pax = sum(n for _, n in dest_sig)
    if total_pax > MAX_SEATS or len(dest_sig) > MAX_LANDINGS:
        _TEMPLATE_CACHE[cache_key] = None
        return None

    airports = (fixed_airport,) if fixed_airport else AIRPORTS
    best = None
    for A in airports:
        for t in AC:
            if total_pax > AC[t]["seats"]:
                continue
            cand = _search_route_for_aircraft(A, t, dest_sig)
            if cand is None:
                continue
            if best is None or route_key(cand) < route_key(best):
                best = cand

    _TEMPLATE_CACHE[cache_key] = best
    return best


def optimize_demands(demands: Sequence[dict]) -> Optional[Sortie]:
    if not demands or len(demands) > MAX_SEATS:
        return None
    fixed = fixed_airport_of_demands(demands)
    if fixed == "__CONFLICT__":
        return None
    sig = demand_signature(demands)
    if len(sig) > MAX_LANDINGS:
        return None
    t = optimize_template(sig, fixed)
    if t is None:
        return None
    sortie = template_to_sortie(t, demands)
    if _ACTIVE_ROUTE_POOL is not None:
        _ACTIVE_ROUTE_POOL.add_sortie(sortie)
    if _PROGRESS is not None:
        _PROGRESS.maybe_print()
    return sortie


# Stage 1：初始打包与机场分配

_PACK_DP_CACHE: Dict[Tuple[str, str, int], Optional[List[Tuple[int, RouteTemplate]]]] = {}


def best_pack_one_airport(airport: str, dest: str, n_people: int) -> Optional[List[Tuple[int, RouteTemplate]]]:
    key = (airport, dest, n_people)
    if key in _PACK_DP_CACHE:
        return _PACK_DP_CACHE[key]
    if n_people == 0:
        _PACK_DP_CACHE[key] = []
        return []

    option: Dict[int, RouteTemplate] = {}
    for p in range(1, min(MAX_SEATS, n_people) + 1):
        sig = ((dest, p),)
        t = optimize_template(sig, airport)
        if t is not None:
            option[p] = t

    # dp[x] = (aggregate_metrics, [(batch_size, template), ...])
    dp: List[Optional[Tuple[AggregateMetrics, List[Tuple[int, RouteTemplate]]]]] = [None] * (n_people + 1)
    dp[0] = (AggregateMetrics(), [])

    for x in range(1, n_people + 1):
        best_state = None
        for p, t in option.items():
            if p > x or dp[x - p] is None:
                continue
            prev_m, prev_plan = dp[x - p]
            cand_m = add_metrics(prev_m, metrics_of_template(t))
            if best_state is None or lex_better(cand_m, best_state[0]):
                best_state = (cand_m, prev_plan + [(p, t)])
        dp[x] = best_state

    ans = None if dp[n_people] is None else dp[n_people][1]
    _PACK_DP_CACHE[key] = ans
    return ans


def metrics_of_template_plan(plan: Sequence[Tuple[int, RouteTemplate]]) -> AggregateMetrics:
    out = AggregateMetrics()
    for _, t in plan:
        out = add_metrics(out, metrics_of_template(t))
    return out


def stage1_pack(demands: Sequence[dict]) -> List[Sortie]:
    by_dest: Dict[str, List[dict]] = defaultdict(list)
    for d in demands:
        by_dest[d["dest"]].append(d)

    sorties: List[Sortie] = []

    for dest in sorted(by_dest):
        group = by_dest[dest]
        fixed_people = {A: [] for A in AIRPORTS}
        land_people = []
        for d in group:
            if d["origin"] in AIRPORTS:
                fixed_people[d["origin"]].append(d)
            else:
                land_people.append(d)

        L = len(land_people)
        fixed_n = {A: len(fixed_people[A]) for A in AIRPORTS}

        plans: Dict[str, Dict[int, Optional[List[Tuple[int, RouteTemplate]]]]] = {A: {} for A in AIRPORTS}
        for A in AIRPORTS:
            for extra in range(L + 1):
                n = fixed_n[A] + extra
                plans[A][extra] = best_pack_one_airport(A, dest, n)

        best_alloc = None
        best_metrics = None

        for x1 in range(L + 1):
            for x2 in range(L - x1 + 1):
                x3 = L - x1 - x2
                alloc = {AIRPORTS[0]: x1, AIRPORTS[1]: x2, AIRPORTS[2]: x3}
                combined = AggregateMetrics()
                feasible = True
                for A in AIRPORTS:
                    plan = plans[A][alloc[A]]
                    if plan is None:
                        feasible = False
                        break
                    combined = add_metrics(combined, metrics_of_template_plan(plan))
                if not feasible:
                    continue
                if best_metrics is None or lex_better(combined, best_metrics):
                    best_metrics = combined
                    best_alloc = alloc

        if best_alloc is None:
            raise RuntimeError(f"目的设施 {dest} 的人员无法构造任何可行初始方案")

        cursor = 0
        for A in AIRPORTS:
            extra = best_alloc[A]
            persons = list(fixed_people[A]) + land_people[cursor: cursor + extra]
            cursor += extra
            plan = plans[A][extra]
            assert plan is not None

            pos = 0
            for batch_size, t in plan:
                batch = persons[pos: pos + batch_size]
                pos += batch_size
                sorties.append(template_to_sortie(t, batch))

            if pos != len(persons):
                raise RuntimeError("内部错误：Stage 1 人数打包恢复失败")

        if cursor != L:
            raise RuntimeError("内部错误：LAND 人员分配计数不一致")

    return sorties


# Stage 2：节约合并

def quick_merge_possible(a: Sortie, b: Sortie) -> bool:
    if len(a.demands) + len(b.demands) > MAX_SEATS:
        return False
    fixed_a = fixed_airport_of_demands(a.demands)
    fixed_b = fixed_airport_of_demands(b.demands)
    if fixed_a == "__CONFLICT__" or fixed_b == "__CONFLICT__":
        return False
    if fixed_a is not None and fixed_b is not None and fixed_a != fixed_b:
        return False
    if len(set(d["dest"] for d in a.demands + b.demands)) > MAX_LANDINGS:
        return False
    return True


def pair_metrics(a: Sortie, b: Sortie) -> AggregateMetrics:
    return add_metrics(metrics_of_sortie(a), metrics_of_sortie(b))


def merge_improvement(a: Sortie, b: Sortie, merged: Sortie) -> Optional[Tuple[int, int, float, float]]:
    before = pair_metrics(a, b)
    after = metrics_of_sortie(merged)

    first_three_better = False
    if after.time_min != before.time_min:
        first_three_better = after.time_min < before.time_min
    elif after.intransit_min != before.intransit_min:
        first_three_better = after.intransit_min < before.intransit_min
    elif abs(after.fuel_kg - before.fuel_kg) > EPS:
        first_three_better = after.fuel_kg < before.fuel_kg
    else:
        # 前三项目标相同时，仍需检查全局座位利用率。
        first_three_better = True

    if not first_three_better:
        return None

    return (
        before.time_min - after.time_min,
        before.intransit_min - after.intransit_min,
        before.fuel_kg - after.fuel_kg,
        after.seat_util - before.seat_util,
    )


def try_merge(a: Sortie, b: Sortie) -> Optional[Sortie]:
    if not quick_merge_possible(a, b):
        return None
    return optimize_demands(a.demands + b.demands)


def stage2_merge(sorties: Sequence[Sortie]) -> List[Sortie]:
    active: Dict[int, Sortie] = {i: s for i, s in enumerate(sorties)}
    total = aggregate_sorties(sorties)
    next_id = len(active)
    heap = []
    counter = 0

    def push_candidate(i: int, j: int) -> None:
        nonlocal counter
        if i == j or i not in active or j not in active:
            return
        if i > j:
            i, j = j, i
        a, b = active[i], active[j]
        if not quick_merge_possible(a, b):
            return
        if _PROGRESS is not None:
            _PROGRESS.note_neighborhood()
        merged = try_merge(a, b)
        if merged is None:
            return
        imp = merge_improvement(a, b, merged)
        if imp is None:
            return
        save_t, save_it, save_f, gain_u = imp
        heapq.heappush(
            heap,
            (-save_t, -save_it, -save_f, -gain_u, counter, i, j, merged),
        )
        counter += 1

    ids = list(active)
    print(f"  初始架次数: {len(ids)}，构造合并候选...")
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            push_candidate(ids[x], ids[y])

    merges = 0
    while heap:
        neg_t, neg_it, neg_f, neg_u, _, i, j, merged = heapq.heappop(heap)
        if i not in active or j not in active:
            continue

        a, b = active[i], active[j]
        imp = merge_improvement(a, b, merged)
        if imp is None:
            continue
        candidate_total = _replace_metrics(total, [a, b], [merged])
        if not lex_better(candidate_total, total):
            continue

        del active[i]
        del active[j]
        new_id = next_id
        next_id += 1
        active[new_id] = merged
        merges += 1

        save_t = total.time_min - candidate_total.time_min
        save_it = total.intransit_min - candidate_total.intransit_min
        save_f = total.fuel_kg - candidate_total.fuel_kg
        gain_u = candidate_total.seat_util - total.seat_util
        total = candidate_total
        print(
            f"  merge {merges:3d}: save_time={save_t:4d} min, "
            f"save_fuel={save_f:8.1f} kg, util_delta={gain_u:+.4f}, "
            f"#sorties={len(active)}"
        )

        for other_id in list(active):
            if other_id != new_id:
                push_candidate(new_id, other_id)

    result = list(active.values())
    result.sort(
        key=lambda s: (
            s.airport,
            s.aircraft_type,
            s.stop_fids[0] if s.stop_fids else "",
            len(s.demands),
            tuple(sorted(d["pid"] for d in s.demands)),
        )
    )
    return result


# Stage 3：二架次重划分

def _replace_metrics(
    total: AggregateMetrics,
    old_sorties: Sequence[Sortie],
    replacement: Sequence[Sortie],
) -> AggregateMetrics:
    old = aggregate_sorties(old_sorties)
    new = aggregate_sorties(replacement)
    return AggregateMetrics(
        time_min=total.time_min - old.time_min + new.time_min,
        fuel_kg=total.fuel_kg - old.fuel_kg + new.fuel_kg,
        pass_km=total.pass_km - old.pass_km + new.pass_km,
        avail_km=total.avail_km - old.avail_km + new.avail_km,
        intransit_min=total.intransit_min - old.intransit_min + new.intransit_min,
    )


def _replace_pair_metrics(
    total: AggregateMetrics,
    old_a: Sortie,
    old_b: Sortie,
    replacement: Sequence[Sortie],
) -> AggregateMetrics:
    return _replace_metrics(total, [old_a, old_b], replacement)


def _demand_classes(demands: Sequence[dict]) -> List[List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for d in demands:
        groups[(d["origin"], d["dest"])].append(d)
    return [groups[k] for k in sorted(groups)]


def _ordered_split_values(c: int, current_left: int, lo: int, hi: int) -> List[int]:
    raw = [current_left]
    for delta in range(1, c + 1):
        raw.extend((current_left - delta, current_left + delta))
    raw.extend((0, c, c // 2))
    raw.extend(range(lo, hi + 1))
    out = []
    seen = set()
    for x in raw:
        if lo <= x <= hi and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _deadline_reached(deadline: Optional[float]) -> bool:
    return deadline is not None and time.time() >= deadline


def _batch_deadline(global_deadline: Optional[float], seconds: float) -> float:
    local_deadline = time.time() + max(0.1, seconds)
    if global_deadline is None:
        return local_deadline
    return min(global_deadline, local_deadline)


def best_two_sortie_repartition(
    a: Sortie,
    b: Sortie,
    total_metrics: AggregateMetrics,
    split_limit: int,
    deadline: Optional[float] = None,
) -> Tuple[List[Sortie], int, bool]:
    all_demands = list(a.demands) + list(b.demands)
    before = [a, b]
    best = before
    best_global = total_metrics
    timed_out = False

    if not _deadline_reached(deadline):
        merged = optimize_demands(all_demands)
        if merged is not None:
            merged_global = _replace_pair_metrics(total_metrics, a, b, [merged])
            if lex_better(merged_global, best_global):
                best, best_global = [merged], merged_global

    classes = _demand_classes(all_demands)
    counts = [len(g) for g in classes]
    current_left = [
        sum(1 for d in a.demands if d["origin"] == g[0]["origin"] and d["dest"] == g[0]["dest"])
        for g in classes
    ]
    allocation = [0] * len(classes)
    evaluated = 0
    visited = 0

    def evaluate_split() -> None:
        nonlocal best, best_global, evaluated, timed_out
        if evaluated >= split_limit or _deadline_reached(deadline):
            timed_out = timed_out or _deadline_reached(deadline)
            return
        complement = tuple(counts[i] - allocation[i] for i in range(len(classes)))
        alloc_tuple = tuple(allocation)
        # 两个新架次没有标签，镜像划分只需计算一次。
        if alloc_tuple > complement:
            return
        evaluated += 1
        left: List[dict] = []
        right: List[dict] = []
        for group, take in zip(classes, allocation):
            left.extend(group[:take])
            right.extend(group[take:])
        if not left or not right or len(left) > MAX_SEATS or len(right) > MAX_SEATS:
            return
        if len({d["dest"] for d in left}) > MAX_LANDINGS:
            return
        if len({d["dest"] for d in right}) > MAX_LANDINGS:
            return
        sa = optimize_demands(left)
        if sa is None:
            return
        sb = optimize_demands(right)
        if sb is None:
            return
        cand = [sa, sb]
        cand_global = _replace_pair_metrics(total_metrics, a, b, cand)
        if lex_better(cand_global, best_global):
            best, best_global = cand, cand_global

    def dfs(
        k: int,
        left_size: int,
        right_size: int,
        left_fixed: Optional[str],
        right_fixed: Optional[str],
        left_dests: frozenset,
        right_dests: frozenset,
    ) -> None:
        nonlocal visited, timed_out
        visited += 1
        if evaluated >= split_limit:
            return
        if visited % 256 == 0 and _deadline_reached(deadline):
            timed_out = True
            return
        if k == len(classes):
            evaluate_split()
            return
        remaining_after = sum(counts[k + 1:])
        c = counts[k]
        lo = max(0, len(all_demands) - MAX_SEATS - left_size - remaining_after)
        hi = min(c, MAX_SEATS - left_size)
        group = classes[k]
        origin = group[0]["origin"]
        dest = group[0]["dest"]
        values = _ordered_split_values(c, current_left[k], lo, hi)
        for x in values:
            to_right = c - x
            new_left_size = left_size + x
            new_right_size = right_size + to_right
            if new_left_size > MAX_SEATS or new_right_size > MAX_SEATS:
                continue

            new_left_fixed = left_fixed
            new_right_fixed = right_fixed
            if origin in AIRPORTS and x:
                if left_fixed is not None and left_fixed != origin:
                    continue
                new_left_fixed = origin
            if origin in AIRPORTS and to_right:
                if right_fixed is not None and right_fixed != origin:
                    continue
                new_right_fixed = origin

            new_left_dests = left_dests | {dest} if x else left_dests
            new_right_dests = right_dests | {dest} if to_right else right_dests
            if len(new_left_dests) > MAX_LANDINGS or len(new_right_dests) > MAX_LANDINGS:
                continue

            allocation[k] = x
            dfs(
                k + 1, new_left_size, new_right_size,
                new_left_fixed, new_right_fixed,
                frozenset(new_left_dests), frozenset(new_right_dests),
            )
            if evaluated >= split_limit or timed_out:
                break

    dfs(0, 0, 0, None, None, frozenset(), frozenset())
    return best, evaluated, timed_out


def _pair_search_priority(a: Sortie, b: Sortie) -> Tuple[float, int, str, str]:
    dest_a = {d["dest"] for d in a.demands}
    dest_b = {d["dest"] for d in b.demands}
    proximity = min(D[x][y] for x in dest_a for y in dest_b)
    return (
        proximity,
        -(a.time_min + b.time_min),
        min(d["pid"] for d in a.demands),
        min(d["pid"] for d in b.demands),
    )


def _pair_records(sorties: Sequence[Sortie]) -> List[dict]:
    records = []
    for i in range(len(sorties)):
        for j in range(i + 1, len(sorties)):
            a, b = sorties[i], sorties[j]
            da = {d["dest"] for d in a.demands}
            db = {d["dest"] for d in b.demands}
            proximity = min(D[x][y] for x in da for y in db)
            shared = len(da & db)
            total_people = len(a.demands) + len(b.demands)
            land_people = sum(d["origin"] == "LAND" for d in a.demands + b.demands)
            current_slack = (
                AC[a.aircraft_type]["seats"] - len(a.demands)
                + AC[b.aircraft_type]["seats"] - len(b.demands)
            )
            type_boundary = sum(s.aircraft_type == "T3" for s in (a, b))
            capacity_potential = 4 * land_people + 3 * type_boundary + current_slack - abs(31 - total_people)
            records.append({
                "i": i, "j": j, "proximity": proximity, "shared": shared,
                "time_sum": a.time_min + b.time_min,
                "capacity": capacity_potential,
            })
    return records


def _multi_source_pair_candidates(sorties: Sequence[Sortie], pair_budget: int) -> List[Tuple[int, int]]:
    records = _pair_records(sorties)
    if not records:
        return []
    budget = min(pair_budget, len(records))
    quotas = (
        max(1, int(math.ceil(budget * 0.40))),
        max(1, int(math.ceil(budget * 0.20))),
        max(1, int(math.ceil(budget * 0.20))),
        max(1, int(math.ceil(budget * 0.20))),
    )
    rankings = (
        sorted(records, key=lambda r: (r["proximity"], -r["shared"], -r["time_sum"], r["i"], r["j"])),
        sorted(records, key=lambda r: (-r["shared"], r["proximity"], -r["time_sum"], r["i"], r["j"])),
        sorted(records, key=lambda r: (-r["time_sum"], r["proximity"], -r["shared"], r["i"], r["j"])),
        sorted(records, key=lambda r: (-r["capacity"], -r["shared"], r["proximity"], r["i"], r["j"])),
    )
    chosen = []
    seen = set()
    for ranking, quota in zip(rankings, quotas):
        for r in ranking[:quota]:
            key = (r["i"], r["j"])
            if key not in seen:
                seen.add(key)
                chosen.append(key)
    rank_pos = [0, 0, 0, 0]
    while len(chosen) < budget:
        added = False
        for q, ranking in enumerate(rankings):
            while rank_pos[q] < len(ranking):
                r = ranking[rank_pos[q]]
                rank_pos[q] += 1
                key = (r["i"], r["j"])
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(key)
                added = True
                break
            if len(chosen) >= budget:
                break
        if not added:
            break
    return chosen[:budget]


def stage3_large_neighborhood(
    sorties: Sequence[Sortie],
    max_moves: int = 30,
    pair_budget: int = 250,
    split_limit: int = 20000,
    deadline: Optional[float] = None,
) -> Tuple[List[Sortie], Dict[str, int]]:
    current = list(sorties)
    total = aggregate_sorties(current)
    moves = 0
    checked = 0
    split_evaluated = 0
    timed_out = False

    while (max_moves <= 0 or moves < max_moves) and not _deadline_reached(deadline):
        pairs = _multi_source_pair_candidates(current, pair_budget)

        best_choice = None
        best_total = total
        for i, j in pairs:
            if _deadline_reached(deadline):
                timed_out = True
                break
            checked += 1
            if _PROGRESS is not None:
                _PROGRESS.note_neighborhood()
            replacement, evaluated, hit_limit = best_two_sortie_repartition(
                current[i], current[j], total_metrics=total,
                split_limit=split_limit, deadline=deadline,
            )
            split_evaluated += evaluated
            timed_out = timed_out or hit_limit
            cand_total = _replace_pair_metrics(
                total, current[i], current[j], replacement,
            )
            if lex_better(cand_total, best_total):
                best_total = cand_total
                best_choice = (i, j, replacement)
            if timed_out:
                break

        if best_choice is None:
            break
        i, j, replacement = best_choice
        old_time = total.time_min
        current = [s for k, s in enumerate(current) if k not in (i, j)] + replacement
        total = best_total
        moves += 1
        print(
            f"  LNS {moves:3d}: total_time {old_time} -> {total.time_min} min, "
            f"intransit={total.intransit_min}, #sorties={len(current)}"
        )

    current.sort(
        key=lambda s: (
            s.airport, s.aircraft_type,
            s.stop_fids[0] if s.stop_fids else "",
            len(s.demands), tuple(sorted(d["pid"] for d in s.demands)),
        )
    )
    return current, {
        "pairs_checked": checked,
        "accepted": moves,
        "splits_evaluated": split_evaluated,
        "timed_out": int(timed_out or _deadline_reached(deadline)),
    }


# Stage 4：三架次到二架次重组

def _triple_candidates(sorties: Sequence[Sortie], triple_budget: int) -> List[Tuple[int, int, int]]:
    n = len(sorties)
    if n < 3:
        return []
    top_k = max(2, min(n - 1, int(math.ceil(math.sqrt(max(2, 2 * triple_budget))))))
    neighbors: Dict[int, List[int]] = {}
    for i in range(n):
        ranked = []
        for j in range(n):
            if i == j:
                continue
            a, b = sorties[i], sorties[j]
            da = {d["dest"] for d in a.demands}
            db = {d["dest"] for d in b.demands}
            proximity = min(D[x][y] for x in da for y in db)
            shared = len(da & db)
            land = sum(d["origin"] == "LAND" for d in a.demands + b.demands)
            ranked.append((proximity, -shared, -land, -(a.time_min + b.time_min), j))
        ranked.sort()
        neighbors[i] = [x[-1] for x in ranked[:top_k]]

    triples = set()
    for i in range(n):
        near = neighbors[i]
        for x in range(len(near)):
            for y in range(x + 1, len(near)):
                triples.add(tuple(sorted((i, near[x], near[y]))))

    ranked_triples = []
    for triple in triples:
        selected = [sorties[i] for i in triple]
        all_demands = [d for s in selected for d in s.demands]
        total_people = len(all_demands)
        if not 20 <= total_people <= 38:
            continue
        fixed = {d["origin"] for d in all_demands if d["origin"] in AIRPORTS}
        if len(fixed) == 3:
            continue
        dest_sets = [{d["dest"] for d in s.demands} for s in selected]
        pair_prox = []
        shared = 0
        for x, y in ((0, 1), (0, 2), (1, 2)):
            pair_prox.append(min(D[u][v] for u in dest_sets[x] for v in dest_sets[y]))
            shared += len(dest_sets[x] & dest_sets[y])
        land = sum(d["origin"] == "LAND" for d in all_demands)
        poor_util = sum(1.0 - s.seat_util for s in selected)
        time_sum = sum(s.time_min for s in selected)
        score = (sum(pair_prox), -shared, -land, -poor_util, -time_sum, triple)
        ranked_triples.append((score, triple))
    ranked_triples.sort(key=lambda x: x[0])
    return [triple for _, triple in ranked_triples[:triple_budget]]


def _best_three_to_two_repartition(
    old: Sequence[Sortie],
    total_metrics: AggregateMetrics,
    beam_width: int,
    deadline: Optional[float],
) -> Tuple[List[Sortie], int, bool]:
    all_demands = [d for s in old for d in s.demands]
    classes = _demand_classes(all_demands)
    counts = [len(g) for g in classes]
    anchor = old[0]
    current_left = [
        sum(1 for d in anchor.demands if d["origin"] == g[0]["origin"] and d["dest"] == g[0]["dest"])
        for g in classes
    ]
    states = [(tuple(), 0, 0, None, None, frozenset(), frozenset(), 0)]
    total_people = len(all_demands)
    timed_out = False

    for k, group in enumerate(classes):
        if _deadline_reached(deadline):
            timed_out = True
            break
        c = counts[k]
        origin, dest = group[0]["origin"], group[0]["dest"]
        next_states = []
        for state in states:
            alloc, left_size, right_size, left_fixed, right_fixed, left_dests, right_dests, disp = state
            lo = max(0, c - (MAX_SEATS - right_size))
            hi = min(c, MAX_SEATS - left_size)
            for x in _ordered_split_values(c, current_left[k], lo, hi):
                to_right = c - x
                new_left_fixed, new_right_fixed = left_fixed, right_fixed
                if origin in AIRPORTS and x:
                    if left_fixed is not None and left_fixed != origin:
                        continue
                    new_left_fixed = origin
                if origin in AIRPORTS and to_right:
                    if right_fixed is not None and right_fixed != origin:
                        continue
                    new_right_fixed = origin
                new_left_dests = left_dests | {dest} if x else left_dests
                new_right_dests = right_dests | {dest} if to_right else right_dests
                if len(new_left_dests) > MAX_LANDINGS or len(new_right_dests) > MAX_LANDINGS:
                    continue
                next_states.append((
                    alloc + (x,), left_size + x, right_size + to_right,
                    new_left_fixed, new_right_fixed,
                    frozenset(new_left_dests), frozenset(new_right_dests),
                    disp + abs(x - current_left[k]),
                ))
        if not next_states:
            states = []
            break
        remaining = sum(counts[k + 1:])
        next_states.sort(key=lambda s: (
            abs(s[1] - s[2]),
            max(s[1], s[2]) - min(MAX_SEATS, (total_people + 1) // 2),
            s[7], len(s[5]) + len(s[6]), s[0],
        ))
        states = next_states[:beam_width]
        if remaining == 0:
            break

    best = list(old)
    best_global = total_metrics
    evaluated = 0
    seen_partitions = set()
    for state in states:
        if _deadline_reached(deadline):
            timed_out = True
            break
        allocation = state[0]
        complement = tuple(counts[i] - allocation[i] for i in range(len(classes)))
        canonical = min(allocation, complement)
        if canonical in seen_partitions:
            continue
        seen_partitions.add(canonical)
        if state[1] == 0 or state[2] == 0:
            continue
        evaluated += 1
        left, right = [], []
        for group, take in zip(classes, allocation):
            left.extend(group[:take])
            right.extend(group[take:])
        sa = optimize_demands(left)
        if sa is None:
            continue
        sb = optimize_demands(right)
        if sb is None:
            continue
        cand = [sa, sb]
        cand_global = _replace_metrics(total_metrics, old, cand)
        if lex_better(cand_global, best_global):
            best, best_global = cand, cand_global
    return best, evaluated, timed_out


def stage4_three_to_two(
    sorties: Sequence[Sortie],
    triple_budget: int = 100,
    triple_beam: int = 250,
    deadline: Optional[float] = None,
) -> Tuple[List[Sortie], Dict[str, int]]:
    current = list(sorties)
    total = aggregate_sorties(current)
    checked = 0
    entered = 0
    accepted = 0
    partitions = 0
    timed_out = False

    while not _deadline_reached(deadline):
        triples = _triple_candidates(current, triple_budget)
        best_choice = None
        best_total = total
        for triple in triples:
            if _deadline_reached(deadline):
                timed_out = True
                break
            checked += 1
            if _PROGRESS is not None:
                _PROGRESS.note_neighborhood()
            old = [current[i] for i in triple]
            entered += 1
            replacement, evaluated, hit_limit = _best_three_to_two_repartition(
                old, total, max(1, triple_beam), deadline,
            )
            partitions += evaluated
            timed_out = timed_out or hit_limit
            if len(replacement) == 2:
                cand_total = _replace_metrics(total, old, replacement)
                if lex_better(cand_total, best_total):
                    best_total = cand_total
                    best_choice = (triple, replacement)
            if timed_out:
                break
        if best_choice is None:
            break
        triple, replacement = best_choice
        old_time = total.time_min
        remove = set(triple)
        current = [s for i, s in enumerate(current) if i not in remove] + replacement
        total = best_total
        accepted += 1
        print(
            f"  3→2 {accepted:3d}: total_time {old_time} -> {total.time_min} min, "
            f"improve={old_time - total.time_min} min, #sorties={len(current)}"
        )

    current.sort(key=lambda s: (
        s.airport, s.aircraft_type, s.stop_fids[0] if s.stop_fids else "",
        len(s.demands), tuple(sorted(d["pid"] for d in s.demands)),
    ))
    return current, {
        "triples_checked": checked,
        "entered": entered,
        "accepted": accepted,
        "partitions_evaluated": partitions,
        "timed_out": int(timed_out or _deadline_reached(deadline)),
    }


# Stage 6/7：多架次重划分

def _composition_candidates(
    count: int,
    target_routes: int,
    current: Tuple[int, ...],
    branch_limit: int,
) -> List[Tuple[int, ...]]:
    raw: List[Tuple[int, ...]] = []

    def add(values: Sequence[int]) -> None:
        values = tuple(int(v) for v in values)
        if len(values) == target_routes and min(values) >= 0 and sum(values) == count:
            raw.append(values)

    add(current)
    for j in range(target_routes):
        whole = [0] * target_routes
        whole[j] = count
        add(whole)

    q, rem = divmod(count, target_routes)
    for shift in range(target_routes):
        balanced = [q] * target_routes
        for k in range(rem):
            balanced[(shift + k) % target_routes] += 1
        add(balanced)

    base = list(current)
    for delta in (1, 2, 3):
        for src in range(target_routes):
            for dst in range(target_routes):
                if src == dst or base[src] < delta:
                    continue
                moved = list(base)
                moved[src] -= delta
                moved[dst] += delta
                add(moved)

    if count <= 8:
        def enumerate_compositions(pos: int, remain: int, prefix: Tuple[int, ...]) -> None:
            if pos == target_routes - 1:
                add(prefix + (remain,))
                return
            for take in range(remain + 1):
                enumerate_compositions(pos + 1, remain - take, prefix + (take,))
        enumerate_compositions(0, count, tuple())

    unique = list(dict.fromkeys(raw))
    unique.sort(key=lambda v: (
        sum(abs(v[j] - current[j]) for j in range(target_routes)),
        sum(x > 0 for x in v),
        max(v) - min(v),
        v,
    ))
    return unique[:max(1, branch_limit)]


def _current_multi_allocation(
    group: Sequence[dict],
    old: Sequence[Sortie],
    target_routes: int,
) -> Tuple[int, ...]:
    origin, dest = group[0]["origin"], group[0]["dest"]
    values = [
        sum(1 for d in s.demands if d["origin"] == origin and d["dest"] == dest)
        for s in old
    ]
    base = values[:target_routes] + [0] * max(0, target_routes - len(values))
    for extra in values[target_routes:]:
        target = min(range(target_routes), key=lambda j: (base[j], j))
        base[target] += extra
    return tuple(base[:target_routes])


def best_multi_sortie_repartition(
    old: Sequence[Sortie],
    target_routes: int,
    total_metrics: AggregateMetrics,
    beam_width: int,
    branch_limit: int = 40,
    deadline: Optional[float] = None,
) -> Tuple[List[Sortie], int, bool]:
    all_demands = [d for s in old for d in s.demands]
    if target_routes <= 0 or len(all_demands) > target_routes * MAX_SEATS:
        return list(old), 0, False
    classes = _demand_classes(all_demands)
    counts = [len(group) for group in classes]
    current = [
        _current_multi_allocation(group, old, target_routes)
        for group in classes
    ]
    states = [(
        tuple(),
        (0,) * target_routes,
        (None,) * target_routes,
        tuple(frozenset() for _ in range(target_routes)),
        0,
    )]
    compactness_cache: Dict[Tuple[str, ...], float] = {}
    timed_out = False

    def compactness(destinations: frozenset) -> float:
        key = tuple(sorted(destinations))
        if key in compactness_cache:
            return compactness_cache[key]
        if len(key) <= 1:
            value = 0.0
        else:
            value = sum(
                min(D[x][y] for y in key if y != x)
                for x in key
            ) / 2.0
        compactness_cache[key] = value
        return value

    for k, group in enumerate(classes):
        if _deadline_reached(deadline):
            timed_out = True
            break
        origin, dest = group[0]["origin"], group[0]["dest"]
        next_states = []
        for allocations, sizes, fixed, destinations, displacement in states:
            for composition in _composition_candidates(
                counts[k], target_routes, current[k], branch_limit,
            ):
                new_sizes = tuple(sizes[j] + composition[j] for j in range(target_routes))
                if max(new_sizes) > MAX_SEATS:
                    continue
                new_fixed = list(fixed)
                new_destinations = list(destinations)
                feasible = True
                for j, take in enumerate(composition):
                    if not take:
                        continue
                    if origin in AIRPORTS:
                        if fixed[j] is not None and fixed[j] != origin:
                            feasible = False
                            break
                        new_fixed[j] = origin
                    new_destinations[j] = destinations[j] | {dest}
                    if len(new_destinations[j]) > MAX_LANDINGS:
                        feasible = False
                        break
                if not feasible:
                    continue
                remaining = sum(counts[k + 1:])
                if sum(MAX_SEATS - size for size in new_sizes) < remaining:
                    continue
                next_states.append((
                    allocations + (composition,),
                    new_sizes,
                    tuple(new_fixed),
                    tuple(frozenset(x) for x in new_destinations),
                    displacement + sum(
                        abs(composition[j] - current[k][j])
                        for j in range(target_routes)
                    ),
                ))
        if not next_states:
            states = []
            break
        processed = sum(counts[:k + 1])
        target_load = processed / target_routes
        next_states.sort(key=lambda state: (
            max(state[1]),
            sum(compactness(dests) for dests in state[3]),
            sum(abs(size - target_load) for size in state[1]),
            state[4],
            sum(len(dests) for dests in state[3]),
            state[0],
        ))
        states = next_states[:max(1, beam_width)]

    best = list(old)
    best_global = total_metrics
    evaluated = 0
    seen = set()
    for allocations, sizes, _, _, _ in states:
        if _deadline_reached(deadline):
            timed_out = True
            break
        if len(allocations) != len(classes) or any(size == 0 for size in sizes):
            continue
        canonical = tuple(sorted(tuple(a[j] for a in allocations) for j in range(target_routes)))
        if canonical in seen:
            continue
        seen.add(canonical)
        evaluated += 1
        groups = [[] for _ in range(target_routes)]
        for demand_group, composition in zip(classes, allocations):
            cursor = 0
            for j, take in enumerate(composition):
                groups[j].extend(demand_group[cursor:cursor + take])
                cursor += take
        candidate = []
        for demands_group in groups:
            sortie = optimize_demands(demands_group)
            if sortie is None:
                candidate = []
                break
            candidate.append(sortie)
        if not candidate:
            continue
        candidate_global = _replace_metrics(total_metrics, old, candidate)
        if lex_better(candidate_global, best_global):
            best = candidate
            best_global = candidate_global
    return best, evaluated, timed_out


def _neighbor_subsets(
    sorties: Sequence[Sortie],
    subset_size: int,
    candidate_budget: int,
    min_people: int = 0,
    max_people: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    n = len(sorties)
    if n < subset_size:
        return []
    top_k = min(n - 1, max(subset_size - 1, 8 + subset_size))
    neighbors: Dict[int, List[int]] = {}
    for i in range(n):
        ranked = []
        for j in range(n):
            if i == j:
                continue
            a, b = sorties[i], sorties[j]
            da = {d["dest"] for d in a.demands}
            db = {d["dest"] for d in b.demands}
            proximity = min(D[x][y] for x in da for y in db)
            shared = len(da & db)
            land = sum(d["origin"] == "LAND" for d in a.demands + b.demands)
            ranked.append((proximity, -shared, -land, -(a.time_min + b.time_min), j))
        ranked.sort()
        neighbors[i] = [row[-1] for row in ranked[:top_k]]

    subsets = set()
    raw_limit = max(candidate_budget * 20, candidate_budget)
    for anchor in range(n):
        for chosen in combinations(neighbors[anchor], subset_size - 1):
            indices = tuple(sorted((anchor,) + chosen))
            people = sum(len(sorties[i].demands) for i in indices)
            if people < min_people:
                continue
            if max_people is not None and people > max_people:
                continue
            subsets.add(indices)
            if len(subsets) >= raw_limit:
                break
        if len(subsets) >= raw_limit:
            break

    ranked_subsets = []
    for indices in subsets:
        selected = [sorties[i] for i in indices]
        all_demands = [d for s in selected for d in s.demands]
        if len(all_demands) < min_people:
            continue
        if max_people is not None and len(all_demands) > max_people:
            continue
        dest_sets = [{d["dest"] for d in s.demands} for s in selected]
        pair_distance = 0.0
        shared = 0
        for x in range(len(selected)):
            for y in range(x + 1, len(selected)):
                pair_distance += min(D[u][v] for u in dest_sets[x] for v in dest_sets[y])
                shared += len(dest_sets[x] & dest_sets[y])
        land = sum(d["origin"] == "LAND" for d in all_demands)
        poor_util = sum(1.0 - s.seat_util for s in selected)
        time_sum = sum(s.time_min for s in selected)
        ranked_subsets.append(((
            pair_distance, -shared, -land, -poor_util, -time_sum, indices,
        ), indices))
    ranked_subsets.sort(key=lambda item: item[0])
    return [indices for _, indices in ranked_subsets[:max(1, candidate_budget)]]


def stage6_four_to_three(
    sorties: Sequence[Sortie],
    candidate_budget: int = 120,
    beam_width: int = 350,
    branch_limit: int = 40,
    max_moves: int = 0,
    deadline: Optional[float] = None,
) -> Tuple[List[Sortie], Dict[str, int]]:
    current = list(sorties)
    total = aggregate_sorties(current)
    checked = entered = accepted = evaluated = 0
    timed_out = False
    while (max_moves <= 0 or accepted < max_moves) and not _deadline_reached(deadline):
        best_choice = None
        best_total = total
        for indices in _neighbor_subsets(
            current, 4, candidate_budget, min_people=39, max_people=57,
        ):
            checked += 1
            if _PROGRESS is not None:
                _PROGRESS.note_neighborhood()
            old = [current[i] for i in indices]
            demands = [d for s in old for d in s.demands]
            if not 39 <= len(demands) <= 57:
                continue
            if len({d["dest"] for d in demands}) > 3 * MAX_LANDINGS:
                continue
            entered += 1
            replacement, count, hit_limit = best_multi_sortie_repartition(
                old, 3, total, beam_width, branch_limit, deadline,
            )
            evaluated += count
            timed_out = timed_out or hit_limit
            if len(replacement) == 3:
                candidate_total = _replace_metrics(total, old, replacement)
                if lex_better(candidate_total, best_total):
                    best_total = candidate_total
                    best_choice = (indices, replacement)
            if timed_out:
                break
        if best_choice is None:
            break
        indices, replacement = best_choice
        old_time = total.time_min
        remove = set(indices)
        current = [s for i, s in enumerate(current) if i not in remove] + replacement
        total = best_total
        accepted += 1
        print(
            f"  4→3 {accepted:3d}: total_time {old_time} -> {total.time_min} min, "
            f"improve={old_time - total.time_min} min, #sorties={len(current)}"
        )
    current.sort(key=lambda s: (
        s.airport, s.aircraft_type, s.stop_fids[0] if s.stop_fids else "",
        len(s.demands), tuple(sorted(d["pid"] for d in s.demands)),
    ))
    return current, {
        "checked": checked, "entered": entered, "accepted": accepted,
        "evaluated": evaluated, "timed_out": int(timed_out),
    }


def stage7_three_to_three(
    sorties: Sequence[Sortie],
    candidate_budget: int = 160,
    beam_width: int = 350,
    branch_limit: int = 40,
    max_moves: int = 0,
    deadline: Optional[float] = None,
) -> Tuple[List[Sortie], Dict[str, int]]:
    current = list(sorties)
    total = aggregate_sorties(current)
    checked = entered = accepted = evaluated = 0
    timed_out = False
    while (max_moves <= 0 or accepted < max_moves) and not _deadline_reached(deadline):
        best_choice = None
        best_total = total
        for indices in _neighbor_subsets(
            current, 3, candidate_budget, min_people=20, max_people=57,
        ):
            checked += 1
            if _PROGRESS is not None:
                _PROGRESS.note_neighborhood()
            old = [current[i] for i in indices]
            demands = [d for s in old for d in s.demands]
            if not 20 <= len(demands) <= 57:
                continue
            if len({d["dest"] for d in demands}) > 3 * MAX_LANDINGS:
                continue
            entered += 1
            replacement, count, hit_limit = best_multi_sortie_repartition(
                old, 3, total, beam_width, branch_limit, deadline,
            )
            evaluated += count
            timed_out = timed_out or hit_limit
            candidate_total = _replace_metrics(total, old, replacement)
            if len(replacement) == 3 and lex_better(candidate_total, best_total):
                best_total = candidate_total
                best_choice = (indices, replacement)
            if timed_out:
                break
        if best_choice is None:
            break
        indices, replacement = best_choice
        old_time = total.time_min
        remove = set(indices)
        current = [s for i, s in enumerate(current) if i not in remove] + replacement
        total = best_total
        accepted += 1
        print(
            f"  3→3 {accepted:3d}: total_time {old_time} -> {total.time_min} min, "
            f"intransit={total.intransit_min}, #sorties={len(current)}"
        )
    current.sort(key=lambda s: (
        s.airport, s.aircraft_type, s.stop_fids[0] if s.stop_fids else "",
        len(s.demands), tuple(sorted(d["pid"] for d in s.demands)),
    ))
    return current, {
        "checked": checked, "entered": entered, "accepted": accepted,
        "evaluated": evaluated, "timed_out": int(timed_out),
    }


def _destroy_target_routes(total_people: int, destroy_size: int) -> Tuple[int, ...]:
    min_routes = int(math.ceil(total_people / MAX_SEATS))
    return tuple(range(min_routes, destroy_size))


def stage8_destroy_repair(
    sorties: Sequence[Sortie],
    destroy_sizes: Sequence[int] = (4, 5, 6),
    candidate_budget: int = 50,
    beam_width: int = 500,
    branch_limit: int = 50,
    max_moves: int = 0,
    deadline: Optional[float] = None,
) -> Tuple[List[Sortie], Dict[str, int]]:
    current = list(sorties)
    total = aggregate_sorties(current)
    checked = entered = accepted = evaluated = 0
    target_attempts: Counter = Counter()
    target_evaluated: Counter = Counter()
    timed_out = False
    while (max_moves <= 0 or accepted < max_moves) and not _deadline_reached(deadline):
        best_choice = None
        best_total = total
        for destroy_size in destroy_sizes:
            for indices in _neighbor_subsets(
                current,
                destroy_size,
                candidate_budget,
                min_people=20,
                max_people=MAX_SEATS * (destroy_size - 1),
            ):
                checked += 1
                if _PROGRESS is not None:
                    _PROGRESS.note_neighborhood()
                old = [current[i] for i in indices]
                demands = [d for s in old for d in s.demands]
                unique_destinations = {d["dest"] for d in demands}
                for target_routes in _destroy_target_routes(len(demands), destroy_size):
                    transition = (destroy_size, target_routes)
                    target_attempts[transition] += 1
                    if len(unique_destinations) > target_routes * MAX_LANDINGS:
                        continue
                    entered += 1
                    replacement, count, hit_limit = best_multi_sortie_repartition(
                        old, target_routes, total, beam_width, branch_limit, deadline,
                    )
                    evaluated += count
                    target_evaluated[transition] += count
                    timed_out = timed_out or hit_limit
                    if len(replacement) == target_routes:
                        candidate_total = _replace_metrics(total, old, replacement)
                        if lex_better(candidate_total, best_total):
                            best_total = candidate_total
                            best_choice = (indices, replacement, destroy_size, target_routes)
                    if timed_out:
                        break
                if timed_out:
                    break
            if timed_out:
                break
        if best_choice is None:
            break
        indices, replacement, destroy_size, target_routes = best_choice
        old_time = total.time_min
        remove = set(indices)
        current = [s for i, s in enumerate(current) if i not in remove] + replacement
        total = best_total
        accepted += 1
        print(
            f"  destroy-repair {destroy_size}→{target_routes} #{accepted}: "
            f"total_time {old_time} -> {total.time_min} min, "
            f"improve={old_time - total.time_min} min, #sorties={len(current)}"
        )
    current.sort(key=lambda s: (
        s.airport, s.aircraft_type, s.stop_fids[0] if s.stop_fids else "",
        len(s.demands), tuple(sorted(d["pid"] for d in s.demands)),
    ))
    for destroy_size in sorted({key[0] for key in target_attempts}):
        print(f"  destroy-repair {destroy_size} routes:")
        for _, target_routes in sorted(
            key for key in target_attempts if key[0] == destroy_size
        ):
            print(
                f"    try {destroy_size}→{target_routes}: "
                f"attempts={target_attempts[(destroy_size, target_routes)]}, "
                f"evaluated={target_evaluated[(destroy_size, target_routes)]}"
            )
    stats = {
        "checked": checked, "entered": entered, "accepted": accepted,
        "evaluated": evaluated, "timed_out": int(timed_out),
    }
    for destroy_size, target_routes in sorted(target_attempts):
        key = f"{destroy_size}_to_{target_routes}"
        stats[f"tried_{key}"] = target_attempts[(destroy_size, target_routes)]
        stats[f"evaluated_{key}"] = target_evaluated[(destroy_size, target_routes)]
    return current, stats


# 结果统计与校验

def simulate_sortie(sortie: Sortie) -> RouteTemplate:
    """从头模拟单架次并重新计算各项指标。"""
    t = sortie.aircraft_type
    ac = AC[t]
    seats = int(ac["seats"])
    full = float(ac["tank"])
    reserve = float(ac["reserve"])
    burn = float(ac["burn"])

    if len(sortie.stop_fids) != len(sortie.refuels):
        raise ValueError("stop_fids 与 refuels 长度不一致")
    if len(sortie.stop_fids) > MAX_LANDINGS:
        raise ValueError("海上着陆次数超过 5")
    if len(sortie.demands) > seats:
        raise ValueError("起飞时载客人数超过机型座位数")

    fixed = fixed_airport_of_demands(sortie.demands)
    if fixed == "__CONFLICT__":
        raise ValueError("同一架次包含来自不同固定机场的人员")
    if fixed is not None and fixed != sortie.airport:
        raise ValueError(f"固定机场需求应从 {fixed} 出发，但架次使用 {sortie.airport}")

    first_delivery = {}
    for i, fid in enumerate(sortie.stop_fids, start=1):
        if fid not in first_delivery:
            first_delivery[fid] = i
    for d in sortie.demands:
        if d["dest"] not in first_delivery:
            raise ValueError(f"人员 {d['pid']} 的目的地 {d['dest']} 不在路线中")

    remain = full
    elapsed = 0
    fuel_used = 0.0
    pass_km = 0.0
    avail_km = 0.0
    intransit = 0
    delivered = set()
    current = sortie.airport

    for idx, (fid, do_refuel) in enumerate(zip(sortie.stop_fids, sortie.refuels), start=1):
        if do_refuel and fid not in GAS_SET:
            raise ValueError(f"非加油设施 {fid} 被标记 refuel=1")

        onboard = sum(1 for d in sortie.demands if d["dest"] not in delivered)
        dist = D[current][fid]
        need = dist * burn
        remain -= need
        if remain < reserve - EPS:
            raise ValueError(f"{current}->{fid} 到达后余油低于安全余量")

        elapsed += flight_min(dist, t)
        fuel_used += need
        pass_km += onboard * dist
        avail_km += seats * dist

        if fid not in delivered:
            cnt = sum(1 for d in sortie.demands if d["dest"] == fid)
            if cnt:
                intransit += cnt * elapsed
                delivered.add(fid)

        elapsed += 20 if do_refuel else 10
        if do_refuel:
            remain = full
        current = fid

    onboard = sum(1 for d in sortie.demands if d["dest"] not in delivered)
    if onboard != 0:
        raise ValueError("返场前仍有人员未送达")

    dist = D[current][sortie.airport]
    need = dist * burn
    remain -= need
    if remain < reserve - EPS:
        raise ValueError(f"{current}->{sortie.airport} 返场后余油低于安全余量")
    elapsed += flight_min(dist, t)
    fuel_used += need
    avail_km += seats * dist

    return RouteTemplate(
        airport=sortie.airport,
        aircraft_type=t,
        stop_fids=tuple(sortie.stop_fids),
        refuels=tuple(sortie.refuels),
        time_min=elapsed,
        fuel_kg=fuel_used,
        pass_km=pass_km,
        avail_km=avail_km,
        intransit_min=intransit,
    )


def validate_solution(sorties: Sequence[Sortie], demands: Sequence[dict]) -> None:
    expected = {d["pid"] for d in demands}
    seen = []

    for s in sorties:
        sim = simulate_sortie(s)
        if s.time_min != sim.time_min:
            raise ValueError(f"架次时间缓存不一致: {s.time_min} vs {sim.time_min}")
        if abs(s.fuel_kg - sim.fuel_kg) > 1e-6:
            raise ValueError("架次燃油缓存不一致")
        if abs(s.pass_km - sim.pass_km) > 1e-6 or abs(s.avail_km - sim.avail_km) > 1e-6:
            raise ValueError("架次座位利用率缓存不一致")
        if s.intransit_min != sim.intransit_min:
            raise ValueError("架次人员在途时间缓存不一致")
        seen.extend(d["pid"] for d in s.demands)

    if len(seen) != len(set(seen)):
        dup = [pid for pid, c in Counter(seen).items() if c > 1]
        raise ValueError(f"人员被重复分配: {dup[:10]}")
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        raise ValueError(f"人员分配不完整，missing={missing[:10]}, extra={extra[:10]}")


def compute_stats(sorties: Sequence[Sortie]) -> dict:
    m = aggregate_sorties(sorties)
    return {
        "total_time_min": m.time_min,
        "total_sorties": len(sorties),
        "total_fuel_kg": m.fuel_kg,
        "total_intransit_min": m.intransit_min,
        "seat_util": m.seat_util,
    }


def generate_paper_figures(
    sorties: Sequence[Sortie],
    reporter: ProgressReporter,
    output_dir: str = FIGURE_DIR,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in available_fonts:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    colors = {"T1": "#3B6FB6", "T2": "#D9822B", "T3": "#3E8E63"}
    paths: List[str] = []

    best_rows = [row for row in reporter.solution_history if row["is_best"]]
    final_stats = compute_stats(sorties)
    final_keys = ("total_time_min", "total_intransit_min", "total_fuel_kg", "seat_util")
    if not best_rows or any(
        abs(best_rows[-1][key] - final_stats[key]) > 1e-8 for key in final_keys
    ):
        best_rows.append({
            "wall_sec": time.time() - reporter.start_time,
            "source": "Final",
            "is_best": True,
            **final_stats,
        })

    x = [row["wall_sec"] for row in best_rows]
    series = (
        ("total_time_min", "总飞机使用时间", "min"),
        ("total_intransit_min", "人员总在途时间", "min"),
        ("total_fuel_kg", "总燃油消耗量", "kg"),
        ("seat_util", "全局座位利用率", "比例"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (key, title, unit) in zip(axes.flat, series):
        y = [row[key] for row in best_rows]
        ax.step(x, y, where="post", color="#2F5D8C", linewidth=1.8)
        ax.scatter(x, y, color="#2F5D8C", s=20, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("运行时间（s）")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.suptitle("Q1 当前最好合法解指标演化")
    path = os.path.join(output_dir, "q1-convergence.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    status_rows = list(reporter.status_history)
    if not status_rows:
        status_rows.append({
            "wall_sec": time.time() - reporter.start_time,
            "pool_generated": reporter.pool.total_generated,
            "pool_nondominated": len(reporter.pool),
            "neighborhoods_checked": reporter.neighborhoods_checked,
            "stagnation_rounds": reporter.stagnation_rounds,
            "level": reporter.level,
        })
    sx = [row["wall_sec"] for row in status_rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    axes[0].plot(
        sx,
        [row["pool_generated"] for row in status_rows],
        label="累计候选列尝试数",
        color="#3B6FB6",
    )
    axes[0].plot(sx, [row["pool_nondominated"] for row in status_rows], label="非支配列", color="#D9822B")
    axes[0].set_ylabel("路线列数")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].plot(sx, [row["neighborhoods_checked"] for row in status_rows], color="#3E8E63")
    axes[1].set_xlabel("运行时间（s）")
    axes[1].set_ylabel("累计检查邻域数")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    level_ax = axes[1].twinx()
    level_ax.step(sx, [row["level"] for row in status_rows], where="post", color="#8C5A9E", alpha=0.8)
    level_ax.set_ylabel("邻域等级")
    level_ax.set_yticks(range(5))
    fig.suptitle("Q1 路线池与邻域搜索过程")
    path = os.path.join(output_dir, "q1-search-process.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    aircraft_types = [t for t in ("T1", "T2", "T3") if any(s.aircraft_type == t for s in sorties)]
    counts = [sum(s.aircraft_type == t for s in sorties) for t in aircraft_types]
    times = [sum(s.time_min for s in sorties if s.aircraft_type == t) for t in aircraft_types]
    utils = [[s.seat_util for s in sorties if s.aircraft_type == t] for t in aircraft_types]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    bar_colors = [colors[t] for t in aircraft_types]
    axes[0, 0].bar(aircraft_types, counts, color=bar_colors)
    axes[0, 0].set_title("各机型架次数")
    axes[0, 0].set_ylabel("架次")
    axes[0, 1].bar(aircraft_types, times, color=bar_colors)
    axes[0, 1].set_title("各机型飞机使用时间")
    axes[0, 1].set_ylabel("min")
    box = axes[1, 0].boxplot(utils, patch_artist=True)
    axes[1, 0].set_xticks(range(1, len(aircraft_types) + 1), aircraft_types)
    for patch, color in zip(box["boxes"], bar_colors):
        patch.set_facecolor(color)
    axes[1, 0].set_title("单架次座位利用率分布")
    axes[1, 0].set_ylabel("比例")
    for t in aircraft_types:
        subset = [s for s in sorties if s.aircraft_type == t]
        axes[1, 1].scatter(
            [len(s.demands) for s in subset],
            [s.time_min for s in subset],
            label=t,
            color=colors[t],
            s=28,
            alpha=0.8,
        )
    axes[1, 1].set_title("载客人数与架次时间")
    axes[1, 1].set_xlabel("运输人数")
    axes[1, 1].set_ylabel("min")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.suptitle("Q1 最终运输方案结构")
    path = os.path.join(output_dir, "q1-route-profile.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


# CSV 输出

def write_output(sorties: Sequence[Sortie], routes_path: str, assign_path: str) -> None:
    os.makedirs(os.path.dirname(routes_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(assign_path) or ".", exist_ok=True)

    flight_no = {"T1": 0, "T2": 0, "T3": 0}
    sortie_fn = []
    for s in sorties:
        flight_no[s.aircraft_type] += 1
        sortie_fn.append(flight_no[s.aircraft_type])

    with open(routes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["aircraft_type", "flight_no", "stop_order", "facility_id", "refuel"])
        for idx, s in enumerate(sorties):
            t = s.aircraft_type
            fn = sortie_fn[idx]
            w.writerow([t, fn, 0, s.airport, 0])
            for i, (fid, r) in enumerate(zip(s.stop_fids, s.refuels), start=1):
                w.writerow([t, fn, i, fid, 1 if r else 0])
            w.writerow([t, fn, len(s.stop_fids) + 1, s.airport, 0])

    with open(assign_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "person_id", "aircraft_type", "flight_no",
            "pickup_stop_order", "delivery_stop_order",
        ])
        for idx, s in enumerate(sorties):
            t = s.aircraft_type
            fn = sortie_fn[idx]
            first_stop = {}
            for i, fid in enumerate(s.stop_fids, start=1):
                first_stop.setdefault(fid, i)
            for d in s.demands:
                # Q1 中所有人的 pickup_stop_order 均为 0。
                delivery = first_stop[d["dest"]]
                w.writerow([d["pid"], t, fn, 0, delivery])


def read_output_solution(
    routes_path: str,
    assign_path: str,
    demands: Sequence[dict],
) -> List[Sortie]:
    """从正式 CSV 恢复并验证历史可行解。"""
    by_pid = {d["pid"]: d for d in demands}
    with open(routes_path, encoding="utf-8-sig", newline="") as f:
        route_rows = list(csv.DictReader(f))
    with open(assign_path, encoding="utf-8-sig", newline="") as f:
        assignment_rows = list(csv.DictReader(f))
    routes: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    assigned: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for row in route_rows:
        key = (row["aircraft_type"].strip(), int(row["flight_no"]))
        routes[key].append(row)
    for row in assignment_rows:
        pid = row["person_id"].strip()
        if pid not in by_pid:
            raise ValueError(f"历史 assignments 含未知 person_id: {pid}")
        key = (row["aircraft_type"].strip(), int(row["flight_no"]))
        assigned[key].append(by_pid[pid])
    if set(routes) != set(assigned):
        raise ValueError("历史 routes 与 assignments 的架次集合不一致")

    sorties = []
    for key, rows in routes.items():
        rows.sort(key=lambda row: int(row["stop_order"]))
        orders = [int(row["stop_order"]) for row in rows]
        if orders != list(range(len(rows))):
            raise ValueError(f"历史路线 {key} 的 stop_order 不连续")
        if len(rows) < 2 or rows[0]["facility_id"] != rows[-1]["facility_id"]:
            raise ValueError(f"历史路线 {key} 未返回起飞机场")
        provisional = Sortie(
            airport=rows[0]["facility_id"].strip(),
            aircraft_type=key[0],
            stop_fids=[row["facility_id"].strip() for row in rows[1:-1]],
            refuels=[row["refuel"].strip() == "1" for row in rows[1:-1]],
            demands=assigned[key],
            time_min=0,
            fuel_kg=0.0,
            pass_km=0.0,
            avail_km=0.0,
            intransit_min=0,
        )
        simulated = simulate_sortie(provisional)
        sorties.append(template_to_sortie(simulated, assigned[key]))
    validate_solution(sorties, demands)
    sorties.sort(key=lambda s: (
        s.airport, s.aircraft_type, s.stop_fids[0] if s.stop_fids else "",
        len(s.demands), tuple(sorted(d["pid"] for d in s.demands)),
    ))
    return sorties


# 主程序

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q1 Route Pool + Restricted Master question1_ver6")
    p.add_argument("--data-dir", default=DATA_DIR, help="输入 CSV 目录")
    p.add_argument("--dist", default=None, help="distances.csv 路径；默认 data-dir/distances.csv")
    p.add_argument("--demand", default=None, help="peopleQ1.csv 路径；默认 data-dir/peopleQ1.csv")
    p.add_argument("--routes-out", default=OUT_ROUTES)
    p.add_argument("--assign-out", default=OUT_ASSIGN)
    p.add_argument("--lns-moves", type=int, default=0, help="最多接受的大邻域改进次数；0 表示直到局部收敛")
    p.add_argument("--pair-budget", type=int, default=250, help="每轮精查的二架次邻域数")
    p.add_argument("--split-limit", type=int, default=20000, help="每个二架次邻域的最大划分枚举数")
    p.add_argument("--triple-budget", type=int, default=100, help="每轮精查的三架次候选数")
    p.add_argument("--triple-beam", type=int, default=250, help="每个 3→2 邻域保留的 Beam 状态数")
    p.add_argument("--time-limit", type=float, default=0.0, help="可选全局安全时限（秒）；0 表示无限时")
    p.add_argument("--route-pool-max", type=int, default=100000, help="非支配路线池清理上限")
    p.add_argument("--route-pool-per-coverage", type=int, default=12, help="同 coverage 最多保留列数")
    p.add_argument("--master-time", type=float, default=5.0, help="Restricted Master 单次 MILP 基础时限（秒）")
    p.add_argument("--master-retries", type=int, default=3, help="每级 MILP 逐步扩时调用次数")
    p.add_argument("--master-columns", type=int, default=1500, help="单次 Restricted Master 最大候选列数")
    p.add_argument("--portfolio-rounds", type=int, default=1, help="每个阶段使用的不同 Master 子集数")
    p.add_argument("--four-budget", type=int, default=120, help="4→3 每轮候选四元组数")
    p.add_argument("--three-three-budget", type=int, default=160, help="3→3 每轮候选三元组数")
    p.add_argument("--destroy-budget", type=int, default=50, help="每种 destroy size 的候选数")
    p.add_argument("--multi-beam", type=int, default=350, help="多路线重划分 Beam 宽度")
    p.add_argument("--multi-branch", type=int, default=40, help="每需求类的分配分支上限")
    p.add_argument("--adaptive-rounds", type=int, default=0, help="自适应轮数；0 表示持续运行到 Ctrl+C")
    p.add_argument("--stagnation-step", type=int, default=2, help="每停滞多少轮提升一个邻域等级")
    p.add_argument("--neighborhood-time", type=float, default=90.0, help="单个邻域批次的基础时限（秒）；anytime 模式会轮转后继续搜索")
    p.add_argument("--resume", action="store_true", help="显式允许读取已有输出作为初始解；默认严格从原始数据冷启动")
    p.add_argument("--progress-interval", type=float, default=7.0, help="完整状态输出间隔（秒）")
    return p.parse_args()


def print_stats(title: str, sorties: Sequence[Sortie]) -> None:
    s = compute_stats(sorties)
    print(title)
    print(f"  总飞机使用时间: {s['total_time_min']} min ({s['total_time_min']/60:.2f} h)")
    print(f"  人员总在途时间: {s['total_intransit_min']} min ({s['total_intransit_min']/60:.2f} h)")
    print(f"  总架次数:       {s['total_sorties']}")
    print(f"  总燃油消耗量:   {s['total_fuel_kg']:.2f} kg")
    print(f"  座位利用率:     {s['seat_util']:.6f}")


def main() -> None:
    global _ACTIVE_ROUTE_POOL, _PROGRESS
    args = parse_args()
    dist_path = args.dist or os.path.join(args.data_dir, "distances.csv")
    demand_path = args.demand or os.path.join(args.data_dir, "peopleQ1.csv")
    routes_out = args.routes_out or os.path.join(args.data_dir, "q1-routes.csv")
    assign_out = args.assign_out or os.path.join(args.data_dir, "q1-assignments.csv")

    t0 = time.time()
    if args.time_limit > 0:
        deadline: Optional[float] = t0 + args.time_limit
    else:
        deadline = None
    pool = RoutePool(
        max_columns=max(1, args.route_pool_max),
        max_per_coverage=max(1, args.route_pool_per_coverage),
    )
    _ACTIVE_ROUTE_POOL = pool
    _PROGRESS = ProgressReporter(pool, t0, args.progress_interval)
    demands: List[dict] = []
    best_valid_solution: List[Sortie] = []

    def accept_valid(candidate: Sequence[Sortie], source: str) -> bool:
        nonlocal best_valid_solution
        validate_solution(candidate, demands)
        pool.add_many(candidate)
        cand_metrics = aggregate_sorties(candidate)
        old_metrics = aggregate_sorties(best_valid_solution) if best_valid_solution else None
        improved = old_metrics is None or lex_better(cand_metrics, old_metrics)
        if improved:
            old_time = None if old_metrics is None else old_metrics.time_min
            best_valid_solution = list(candidate)
            _PROGRESS.set_best(best_valid_solution)
            if old_time is not None:
                print(
                    f"  NEW BEST: {old_time} -> {cand_metrics.time_min} min, "
                    f"source={source}, intransit={cand_metrics.intransit_min}, "
                    f"fuel={cand_metrics.fuel_kg:.1f}, util={cand_metrics.seat_util:.6f}"
                )
        _PROGRESS.record_solution(source, candidate, improved)
        return improved

    try:
        print("加载数据...")
        load_distances(dist_path)
        demands = load_demands(demand_path)
        print(
            f"  {len(demands)} 条 Q1 出海需求，"
            f"{len(aggregate_demand_totals(demands))} 个聚合需求类别，{len(D)} 个地点"
        )

        if args.resume and os.path.exists(routes_out) and os.path.exists(assign_out):
            try:
                seed = read_output_solution(routes_out, assign_out, demands)
                accept_valid(seed, "Existing-CSV-Seed")
                print_stats("已加载并验证历史 incumbent:", seed)
            except (OSError, ValueError, KeyError) as exc:
                print(f"  历史输出未作为 seed 使用: {exc}")
        elif not args.resume:
            print("  冷启动模式：不读取已有 q1 输出，只使用原始数据求解。")

        print("\nStage 1: ver4 同目的地打包 + LAND 自动分配机场...")
        sorties = stage1_pack(demands)
        accept_valid(sorties, "Stage1")
        print_stats("Stage 1 结果:", sorties)
        _PROGRESS.maybe_print(force=True)

        print("\nStage 2: ver4 节约合并 + 路线池收集...")
        sorties = stage2_merge(sorties)
        accept_valid(sorties, "Stage2-Merge")
        print_stats("Stage 2 结果:", sorties)

        print("\nStage 3: ver4 二架次重划分 + 路线池收集...")
        sorties, stage3_stats = stage3_large_neighborhood(
            sorties,
            max_moves=max(0, args.lns_moves),
            pair_budget=max(1, args.pair_budget),
            split_limit=max(1, args.split_limit),
            deadline=_batch_deadline(deadline, args.neighborhood_time),
        )
        accept_valid(sorties, "Stage3-2to2")
        print_stats("Stage 3 结果:", sorties)
        print(
            f"  Pair={stage3_stats['pairs_checked']}，"
            f"accepted={stage3_stats['accepted']}，"
            f"splits={stage3_stats['splits_evaluated']}"
        )

        print("\nStage 4: ver4 三架次到二架次 + 路线池收集...")
        sorties, stage4_stats = stage4_three_to_two(
            sorties,
            triple_budget=max(1, args.triple_budget),
            triple_beam=max(1, args.triple_beam),
            deadline=_batch_deadline(deadline, args.neighborhood_time),
        )
        accept_valid(sorties, "Stage4-3to2")
        print_stats("Stage 4 结果:", sorties)
        print(
            f"  Triple={stage4_stats['triples_checked']}，"
            f"entered={stage4_stats['entered']}，accepted={stage4_stats['accepted']}"
        )

        print("\nStage 5: ver4 Merge + 小规模二架次 Cleanup...")
        sorties = stage2_merge(sorties)
        accept_valid(sorties, "Stage5-Merge")
        cleanup_moves = 0 if args.lns_moves <= 0 else max(1, args.lns_moves // 3)
        sorties, cleanup_stats = stage3_large_neighborhood(
            sorties,
            max_moves=cleanup_moves,
            pair_budget=max(1, args.pair_budget // 3),
            split_limit=max(1, args.split_limit // 3),
            deadline=_batch_deadline(deadline, args.neighborhood_time * 0.5),
        )
        accept_valid(sorties, "Stage5-2to2-Cleanup")
        print_stats("Stage 5 结果:", sorties)
        print(
            f"  Cleanup Pair={cleanup_stats['pairs_checked']}，"
            f"accepted={cleanup_stats['accepted']}"
        )

        _PROGRESS.maybe_print(force=True)
        print("\nRestricted Set Partitioning Master Portfolio...")
        print(
            "  注意：Master 的最优性范围以本次实际选中列为准；"
            "只有全池列全部进入时才称 current Route Pool 内最优，绝不声称 Q1 global optimum。"
        )
        portfolio_cursor = 0

        def run_portfolio(level: int = 0) -> int:
            nonlocal portfolio_cursor
            pool.clean()
            scale = 2 if level >= 4 else 1
            portfolio_best, results, improvements = solve_master_portfolio(
                pool,
                demands,
                best_valid_solution,
                rounds=max(1, args.portfolio_rounds),
                start_round=portfolio_cursor,
                max_master_columns=max(1, args.master_columns * scale),
                per_call_time=max(0.1, args.master_time * scale),
                retries=max(1, args.master_retries),
            )
            portfolio_cursor += max(1, args.portfolio_rounds)
            accept_valid(portfolio_best, f"Master-Portfolio-L{level}")
            if results:
                last = results[-1]
                print(
                    f"  Portfolio status={last.status}, stage={last.completed_stage}, "
                    f"bound={last.last_bound}, gap={last.last_gap}, improvements={improvements}"
                )
            return improvements

        run_portfolio(level=0)

        print("\nStage 6: 4 routes → 3 routes...")
        sorties, four_stats = stage6_four_to_three(
            best_valid_solution,
            candidate_budget=max(1, args.four_budget),
            beam_width=max(1, args.multi_beam),
            branch_limit=max(1, args.multi_branch),
            max_moves=0,
            deadline=_batch_deadline(deadline, args.neighborhood_time),
        )
        accept_valid(sorties, "Stage6-4to3")
        print(
            f"  4→3 checked={four_stats['checked']}, entered={four_stats['entered']}, "
            f"accepted={four_stats['accepted']}, evaluated={four_stats['evaluated']}"
        )
        run_portfolio(level=1)

        print("\nStage 7: 3 routes → 3 routes...")
        sorties, three_stats = stage7_three_to_three(
            best_valid_solution,
            candidate_budget=max(1, args.three_three_budget),
            beam_width=max(1, args.multi_beam),
            branch_limit=max(1, args.multi_branch),
            max_moves=0,
            deadline=_batch_deadline(deadline, args.neighborhood_time),
        )
        accept_valid(sorties, "Stage7-3to3")
        print(
            f"  3→3 checked={three_stats['checked']}, entered={three_stats['entered']}, "
            f"accepted={three_stats['accepted']}, evaluated={three_stats['evaluated']}"
        )
        run_portfolio(level=2)

        print("\nAdaptive anytime search 已启动；默认持续到 Ctrl+C。")
        stagnation_rounds = 0
        adaptive_round = 0
        while (
            (args.adaptive_rounds <= 0 or adaptive_round < args.adaptive_rounds)
            and not _deadline_reached(deadline)
        ):
            adaptive_round += 1
            level = min(4, stagnation_rounds // max(1, args.stagnation_step))
            _PROGRESS.stagnation_rounds = stagnation_rounds
            _PROGRESS.level = level
            before = aggregate_sorties(best_valid_solution)
            print(
                f"\n=== Adaptive round {adaptive_round}, level={level}, "
                f"stagnation={stagnation_rounds}, best_time={before.time_min} ==="
            )
            working = stage2_merge(best_valid_solution)
            accept_valid(working, f"Adaptive-L{level}-Merge")
            working, pair_stats = stage3_large_neighborhood(
                best_valid_solution,
                max_moves=max(0, args.lns_moves),
                pair_budget=max(1, args.pair_budget * (2 if level >= 4 else 1)),
                split_limit=max(1, args.split_limit * (2 if level >= 4 else 1)),
                deadline=_batch_deadline(deadline, args.neighborhood_time),
            )
            accept_valid(working, f"Adaptive-L{level}-2to2")

            if level >= 1:
                working, stats4 = stage6_four_to_three(
                    best_valid_solution,
                    candidate_budget=max(1, args.four_budget * (2 if level >= 4 else 1)),
                    beam_width=max(1, args.multi_beam * (2 if level >= 4 else 1)),
                    branch_limit=max(1, args.multi_branch * (2 if level >= 4 else 1)),
                    max_moves=0,
                    deadline=_batch_deadline(deadline, args.neighborhood_time),
                )
                accept_valid(working, f"Adaptive-L{level}-4to3")
            if level >= 2:
                working, stats33 = stage7_three_to_three(
                    best_valid_solution,
                    candidate_budget=max(1, args.three_three_budget * (2 if level >= 4 else 1)),
                    beam_width=max(1, args.multi_beam * (2 if level >= 4 else 1)),
                    branch_limit=max(1, args.multi_branch * (2 if level >= 4 else 1)),
                    max_moves=0,
                    deadline=_batch_deadline(deadline, args.neighborhood_time),
                )
                accept_valid(working, f"Adaptive-L{level}-3to3")
            if level >= 3:
                working, destroy_stats = stage8_destroy_repair(
                    best_valid_solution,
                    destroy_sizes=(4, 5, 6),
                    candidate_budget=max(1, args.destroy_budget * (2 if level >= 4 else 1)),
                    beam_width=max(1, args.multi_beam * (2 if level >= 4 else 1)),
                    branch_limit=max(1, args.multi_branch * (2 if level >= 4 else 1)),
                    max_moves=0,
                    deadline=_batch_deadline(deadline, args.neighborhood_time * (2 if level >= 4 else 1)),
                )
                accept_valid(working, f"Adaptive-L{level}-DestroyRepair")

            run_portfolio(level=level)
            after = aggregate_sorties(best_valid_solution)
            if after.time_min < before.time_min:
                stagnation_rounds = 0
                print(f"  Primary improvement: {before.time_min} -> {after.time_min}; reset level.")
            else:
                stagnation_rounds += 1
            _PROGRESS.stagnation_rounds = stagnation_rounds
            _PROGRESS.level = min(4, stagnation_rounds // max(1, args.stagnation_step))
            _PROGRESS.maybe_print(force=True)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C：停止继续搜索，保存上一份已验证合法最好解。")
    finally:
        _ACTIVE_ROUTE_POOL = None
        pool.clean()
        if _PROGRESS is not None:
            _PROGRESS.maybe_print(force=True)
        if best_valid_solution and demands:
            print("\n最终保护性校验与输出...")
            validate_solution(best_valid_solution, demands)
            write_output(best_valid_solution, routes_out, assign_out)
            print_stats("question1 当前最好合法结果", best_valid_solution)
            print(f"  -> {routes_out}")
            print(f"  -> {assign_out}")
            try:
                figure_paths = generate_paper_figures(best_valid_solution, _PROGRESS, FIGURE_DIR)
                print("论文图表已生成：")
                for figure_path in figure_paths:
                    print(f"  -> {figure_path}")
            except Exception as exc:
                print(f"图表生成失败，不影响合法解与 CSV 输出：{exc}")
            print("VALIDATION PASSED")
        else:
            print("未形成可验证初始解，因此没有覆盖正式输出。")
        print(f"总运行时间: {time.time() - t0:.2f} s")
        print(
            f"Route Pool: generated={pool.total_generated}, "
            f"non-dominated={len(pool)}, duplicates={pool.duplicates}, "
            f"dominated_rejected={pool.dominated_rejected}, "
            f"dominated_removed={pool.dominated_removed}, "
            f"capacity_removed={pool.capacity_removed}"
        )
        print(
            f"路线缓存: {len(_ROUTE_CACHE)}，模板缓存: {len(_TEMPLATE_CACHE)}，"
            f"打包缓存: {len(_PACK_DP_CACHE)}"
        )


if __name__ == "__main__":
    main()
