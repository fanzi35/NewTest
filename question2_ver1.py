"""
问题2：出海、海返与穿梭联合运输（Python 3.11）。

模型：多机场取送路径 + 聚合路线池 + 严格字典序集合划分。

严格目标顺序：
1. 总飞机使用时间最小；
2. 第一层相同时，人员总在途时间最小；
3. 前两层相同时，总燃油最小；
4. 前三层相同时，总体座位利用率最大。

单架次显式处理：
- 同一机场起降；
- 最多5次海上着陆；
- 出海、海返和穿梭人员的上下机先后；
- 每一航段动态载客量不超过机型座位数；
- 同一停靠先下后上；
- 满油起飞、安全余油、指定设施选择性加油；
- LAND 自动绑定到该架次机场；
- 人员不得换乘。

算法说明：服务设施不重复时完整枚举其顺序；若简单顺序均不可行，再受控枚举最多
5次着陆内的设施重复，以处理相反方向穿梭等环。路线池主问题只证明当前已生成路线
池内最优，不声称原问题全部可行路线空间的全局最优。

依赖：python -m pip install ortools
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Windows/PyCharm 控制台统一采用 UTF-8，避免中文进度信息乱码。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 OR-Tools，请在当前 Python 3.11 环境执行：python -m pip install ortools"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "docs" / "reference_formats"
DIST_PATH = DATA_DIR / "distances.csv"
DEMAND_PATH = DATA_DIR / "peopleQ2.csv"
ROUTES_OUT = OUTPUT_DIR / "q2-routes.csv"
ASSIGN_OUT = OUTPUT_DIR / "q2-assignments.csv"
SUMMARY_OUT = OUTPUT_DIR / "q2-summary.json"

AIRPORTS = ("A01", "A02", "A03")
AIRPORT_SET = set(AIRPORTS)
LAND = "LAND"
GAS_STATIONS = ("F006", "F011", "F018", "F024", "F031", "F038", "F044", "F050")
GAS_SET = set(GAS_STATIONS)
AC = {
    "T1": {"seats": 12, "speed": 250.0, "burn": 3.4, "tank": 1000.0, "reserve": 150.0},
    "T2": {"seats": 16, "speed": 220.0, "burn": 2.5, "tank": 1150.0, "reserve": 150.0},
    "T3": {"seats": 19, "speed": 190.0, "burn": 2.9, "tank": 1600.0, "reserve": 200.0},
}
MAX_LANDINGS = 5
MAX_SEATS = 19
EPS = 1e-9
SCALE = 1000

D: Dict[str, Dict[str, float]] = {}
CLASSES: Tuple[Tuple[str, str], ...] = tuple()
CLASS_INDEX: Dict[Tuple[str, str], int] = {}


@dataclass(frozen=True)
class Metrics:
    time_min: int = 0
    intransit_min: int = 0
    fuel_kg: float = 0.0
    pass_km: float = 0.0
    avail_km: float = 0.0

    @property
    def seat_util(self) -> float:
        return 0.0 if self.avail_km <= EPS else self.pass_km / self.avail_km


@dataclass(frozen=True)
class Route:
    airport: str
    aircraft_type: str
    stops: Tuple[str, ...]
    refuels: Tuple[bool, ...]
    coverage: Tuple[Tuple[int, int], ...]
    pairs: Tuple[Tuple[int, int, int], ...]  # (class index, pickup order, delivery order)
    metrics: Metrics

    @property
    def people(self) -> int:
        return sum(n for _, n in self.coverage)

    @property
    def signature(self) -> Tuple:
        return (
            self.coverage, self.airport, self.aircraft_type,
            self.stops, self.refuels, self.pairs,
        )


@dataclass
class OperatedRoute:
    template: Route
    people: List[dict]


@dataclass
class Pattern:
    pid: int
    route: Route
    sources: set[str] = field(default_factory=set)


@dataclass
class MasterResult:
    selected: Counter
    metrics: Metrics
    stages: List[dict]
    optimal_within_pool: bool


def flight_min(distance: float, aircraft_type: str) -> int:
    return math.ceil(60.0 * distance / AC[aircraft_type]["speed"] - EPS)


def add_metrics(a: Metrics, b: Metrics) -> Metrics:
    return Metrics(
        a.time_min + b.time_min,
        a.intransit_min + b.intransit_min,
        a.fuel_kg + b.fuel_kg,
        a.pass_km + b.pass_km,
        a.avail_km + b.avail_km,
    )


def sub_add_metrics(total: Metrics, old: Sequence[Route], new: Sequence[Route]) -> Metrics:
    return Metrics(
        total.time_min - sum(r.metrics.time_min for r in old) + sum(r.metrics.time_min for r in new),
        total.intransit_min - sum(r.metrics.intransit_min for r in old)
        + sum(r.metrics.intransit_min for r in new),
        total.fuel_kg - sum(r.metrics.fuel_kg for r in old) + sum(r.metrics.fuel_kg for r in new),
        total.pass_km - sum(r.metrics.pass_km for r in old) + sum(r.metrics.pass_km for r in new),
        total.avail_km - sum(r.metrics.avail_km for r in old) + sum(r.metrics.avail_km for r in new),
    )


def aggregate_routes(routes: Iterable[Route]) -> Metrics:
    result = Metrics()
    for route in routes:
        result = add_metrics(result, route.metrics)
    return result


def lex_better(a: Metrics, b: Optional[Metrics]) -> bool:
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


def route_key(route: Route) -> Tuple:
    m = route.metrics
    return (
        m.time_min, m.intransit_min, round(m.fuel_kg, 9), -m.seat_util,
        route.aircraft_type, route.airport, route.stops, route.refuels, route.pairs,
    )


def load_distances(path: Path) -> None:
    global D
    D = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [x.strip() for x in header[1:]]
        for row in reader:
            if not row:
                continue
            origin = row[0].strip()
            D[origin] = {dest: float(raw.strip()) for dest, raw in zip(columns, row[1:])}
    nodes = set(D)
    if not AIRPORT_SET.issubset(nodes):
        raise ValueError("distances.csv 缺少 A01/A02/A03")
    for origin in nodes:
        missing = nodes - set(D[origin])
        if missing:
            raise ValueError(f"距离矩阵 {origin} 行缺少地点 {sorted(missing)[:5]}")


def load_demands(path: Path):
    rows = []
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"person_id", "origin_id", "destination_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"peopleQ2.csv 必须包含 {sorted(required)}")
        for row in reader:
            demand = {
                "pid": row["person_id"].strip(),
                "origin": row["origin_id"].strip(),
                "dest": row["destination_id"].strip(),
            }
            rows.append(demand)
            grouped[(demand["origin"], demand["dest"])].append(demand)
    if len({d["pid"] for d in rows}) != len(rows):
        raise ValueError("peopleQ2.csv 存在重复 person_id")
    nodes = set(D)
    for demand in rows:
        for key in ("origin", "dest"):
            value = demand[key]
            if value != LAND and value not in nodes:
                raise ValueError(f"{demand['pid']} 的 {key}={value} 非法")
        if demand["origin"] == demand["dest"]:
            raise ValueError(f"{demand['pid']} 起点终点相同")
        if demand["origin"] in AIRPORT_SET | {LAND} and demand["dest"] in AIRPORT_SET | {LAND}:
            raise ValueError(f"{demand['pid']} 是陆地到陆地需求，不属于本题运输结构")
    classes = tuple(sorted(grouped))
    totals = tuple(len(grouped[c]) for c in classes)
    return rows, classes, totals, grouped


def sparse_to_counter(coverage: Sequence[Tuple[int, int]]) -> Counter:
    return Counter({int(g): int(n) for g, n in coverage if n})


def canonical_coverage(counts: Counter | Dict[int, int] | Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    if isinstance(counts, (Counter, dict)):
        items = counts.items()
    else:
        items = counts
    return tuple(sorted((int(g), int(n)) for g, n in items if n > 0))


def combine_coverage(*coverages: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    counter = Counter()
    for coverage in coverages:
        counter.update(dict(coverage))
    return canonical_coverage(counter)


def fixed_airport(coverage: Sequence[Tuple[int, int]]) -> Optional[str]:
    fixed = set()
    for g, n in coverage:
        if not n:
            continue
        origin, dest = CLASSES[g]
        if origin in AIRPORT_SET:
            fixed.add(origin)
        if dest in AIRPORT_SET:
            fixed.add(dest)
    if len(fixed) > 1:
        return "__CONFLICT__"
    return next(iter(fixed)) if fixed else None


def required_nodes(coverage: Sequence[Tuple[int, int]]) -> Tuple[str, ...]:
    nodes = set()
    for g, n in coverage:
        if not n:
            continue
        origin, dest = CLASSES[g]
        if origin not in AIRPORT_SET and origin != LAND:
            nodes.add(origin)
        if dest not in AIRPORT_SET and dest != LAND:
            nodes.add(dest)
    return tuple(sorted(nodes))


def shuttle_edges(coverage: Sequence[Tuple[int, int]]) -> Tuple[Tuple[str, str], ...]:
    edges = set()
    for g, n in coverage:
        if not n:
            continue
        origin, dest = CLASSES[g]
        if origin.startswith("F") and dest.startswith("F"):
            edges.add((origin, dest))
    return tuple(sorted(edges))


def quick_coverage_possible(coverage: Sequence[Tuple[int, int]], max_route_people: int) -> bool:
    if not coverage or sum(n for _, n in coverage) > max_route_people:
        return False
    if fixed_airport(coverage) == "__CONFLICT__":
        return False
    return len(required_nodes(coverage)) <= MAX_LANDINGS


def sequence_supports(sequence: Sequence[str], edges: Sequence[Tuple[str, str]]) -> bool:
    positions: Dict[str, List[int]] = defaultdict(list)
    for i, node in enumerate(sequence):
        positions[node].append(i)
    return all(any(i < j for i in positions[o] for j in positions[d]) for o, d in edges)


@lru_cache(maxsize=None)
def simple_service_sequences(
    required: Tuple[str, ...], edges: Tuple[Tuple[str, str], ...],
) -> Tuple[Tuple[str, ...], ...]:
    return tuple(
        permutation
        for permutation in itertools.permutations(required)
        if sequence_supports(permutation, edges)
    )


@lru_cache(maxsize=None)
def repeated_service_sequences(
    required: Tuple[str, ...], edges: Tuple[Tuple[str, str], ...],
) -> Tuple[Tuple[str, ...], ...]:
    if len(required) <= 1 or len(required) >= MAX_LANDINGS:
        return tuple()
    out = []
    required_set = set(required)
    for length in range(len(required) + 1, MAX_LANDINGS + 1):
        for sequence in itertools.product(required, repeat=length):
            if any(sequence[i] == sequence[i + 1] for i in range(length - 1)):
                continue
            if set(sequence) != required_set:
                continue
            if sequence_supports(sequence, edges):
                out.append(tuple(sequence))
    return tuple(out)


def fuel_feasible(
    airport: str,
    aircraft_type: str,
    stops: Sequence[str],
    refuels: Sequence[bool],
) -> Tuple[bool, float, List[float], List[int], List[int], List[float]]:
    ac = AC[aircraft_type]
    remain = float(ac["tank"])
    reserve = float(ac["reserve"])
    burn = float(ac["burn"])
    current = airport
    elapsed = 0
    fuel = 0.0
    arrivals = [0]
    departures = [0]
    leg_distances = []
    for stop, refuel in zip(stops, refuels):
        distance = D[current][stop]
        need = distance * burn
        remain -= need
        if remain < reserve - EPS:
            return False, 0.0, [], [], [], []
        elapsed += flight_min(distance, aircraft_type)
        fuel += need
        leg_distances.append(distance)
        arrivals.append(elapsed)
        elapsed += 20 if refuel else 10
        departures.append(elapsed)
        if refuel:
            if stop not in GAS_SET:
                return False, 0.0, [], [], [], []
            remain = float(ac["tank"])
        current = stop
    distance = D[current][airport]
    need = distance * burn
    remain -= need
    if remain < reserve - EPS:
        return False, 0.0, [], [], [], []
    elapsed += flight_min(distance, aircraft_type)
    fuel += need
    leg_distances.append(distance)
    arrivals.append(elapsed)
    departures.append(elapsed)
    return True, fuel, leg_distances, arrivals, departures, [remain]


def refuel_variants(
    stops: Tuple[str, ...], forced_refuel: frozenset[int], airport: str, aircraft_type: str,
) -> List[Tuple[Tuple[bool, ...], Tuple]]:
    optional = [i for i, node in enumerate(stops) if node in GAS_SET and i not in forced_refuel]
    candidates = []
    for bits in itertools.product((False, True), repeat=len(optional)):
        flags = [False] * len(stops)
        for i in forced_refuel:
            flags[i] = True
        for i, bit in zip(optional, bits):
            flags[i] = bit
        result = fuel_feasible(airport, aircraft_type, stops, flags)
        if result[0]:
            candidates.append((tuple(flags), result))
    if not candidates:
        return []
    # 更多加油必定多耗停靠时间；只保留最少加油次数的可行方案。
    minimum = min(sum(flags) for flags, _ in candidates)
    return [(flags, result) for flags, result in candidates if sum(flags) == minimum]


def insert_gas_paths(base: Tuple[str, ...], extra: int) -> List[Tuple[Tuple[str, ...], frozenset[int]]]:
    states = [(tuple((node, False) for node in base))]
    for _ in range(extra):
        next_states = set()
        for state in states:
            for position in range(len(state) + 1):
                for gas in GAS_STATIONS:
                    before = state[position - 1][0] if position else None
                    after = state[position][0] if position < len(state) else None
                    if gas == before or gas == after:
                        continue
                    new_state = state[:position] + ((gas, True),) + state[position:]
                    next_states.add(new_state)
        states = list(next_states)
    out = []
    for state in states:
        stops = tuple(node for node, _ in state)
        forced = frozenset(i for i, (_, inserted) in enumerate(state) if inserted)
        out.append((stops, forced))
    return out


def path_variants(
    service_sequence: Tuple[str, ...], airport: str, aircraft_type: str,
) -> List[Tuple[Tuple[str, ...], Tuple[bool, ...], Tuple]]:
    direct = refuel_variants(service_sequence, frozenset(), airport, aircraft_type)
    if direct:
        return [(service_sequence, flags, result) for flags, result in direct]
    remaining = MAX_LANDINGS - len(service_sequence)
    for extra in range(1, remaining + 1):
        feasible = []
        for stops, forced in insert_gas_paths(service_sequence, extra):
            for flags, result in refuel_variants(stops, forced, airport, aircraft_type):
                feasible.append((stops, flags, result))
        if feasible:
            return feasible
    return []


def class_options(
    g: int, stops: Sequence[str], airport: str,
) -> List[Tuple[int, int]]:
    origin, dest = CLASSES[g]
    final = len(stops) + 1
    if origin in AIRPORT_SET and origin != airport:
        return []
    if dest in AIRPORT_SET and dest != airport:
        return []
    if origin in AIRPORT_SET or origin == LAND:
        pickup_positions = [0]
    else:
        pickup_positions = [i for i, node in enumerate(stops, start=1) if node == origin]
    options = []
    for pickup in pickup_positions:
        if dest in AIRPORT_SET or dest == LAND:
            options.append((pickup, final))
            continue
        # 附录要求：下机点是上机以后第一次停靠该终点的位置。
        later = [i for i, node in enumerate(stops, start=1) if i > pickup and node == dest]
        if later:
            options.append((pickup, later[0]))
    # 对海返，最后一次在起点上机同时减少容量占用和在途时间，支配更早上机。
    if (dest in AIRPORT_SET or dest == LAND) and options:
        options = [max(options, key=lambda x: x[0])]
    return list(dict.fromkeys(options))


def best_assignment(
    coverage: Tuple[Tuple[int, int], ...],
    stops: Tuple[str, ...],
    airport: str,
    aircraft_type: str,
    leg_distances: Sequence[float],
    arrivals: Sequence[int],
    departures: Sequence[int],
    assignment_limit: int,
) -> Optional[Tuple[Tuple[Tuple[int, int, int], ...], int, float]]:
    seats = int(AC[aircraft_type]["seats"])
    entries = []
    for g, count in coverage:
        options = class_options(g, stops, airport)
        if not options:
            return None
        entries.append((g, count, options))
    entries.sort(key=lambda x: (len(x[2]), -x[1], x[0]))
    loads = [0] * len(leg_distances)
    chosen: List[Tuple[int, int, int]] = []
    best = None
    evaluated = 0

    def dfs(k: int, transit: int) -> None:
        nonlocal best, evaluated
        if evaluated >= assignment_limit:
            return
        if best is not None and transit > best[0]:
            return
        if k == len(entries):
            evaluated += 1
            pass_km = sum(load * distance for load, distance in zip(loads, leg_distances))
            pairs = tuple(sorted(chosen))
            candidate = (transit, -pass_km, pairs)
            if best is None or candidate < best:
                best = candidate
            return
        g, count, options = entries[k]
        ordered = sorted(options, key=lambda pair: (
            arrivals[pair[1]] - departures[pair[0]], pair,
        ))
        for pickup, delivery in ordered:
            if any(loads[leg] + count > seats for leg in range(pickup, delivery)):
                continue
            for leg in range(pickup, delivery):
                loads[leg] += count
            chosen.append((g, pickup, delivery))
            dfs(k + 1, transit + count * (arrivals[delivery] - departures[pickup]))
            chosen.pop()
            for leg in range(pickup, delivery):
                loads[leg] -= count

    dfs(0, 0)
    if best is None:
        return None
    return best[2], best[0], -best[1]


class RouteOracle:
    def __init__(self, max_route_people: int = 38, assignment_limit: int = 5000) -> None:
        self.max_route_people = max_route_people
        self.assignment_limit = assignment_limit
        self.cache: Dict[Tuple[Tuple[int, int], ...], Optional[Route]] = {}
        self.calls = 0
        self.cache_hits = 0

    def optimize(self, coverage: Sequence[Tuple[int, int]]) -> Optional[Route]:
        coverage = canonical_coverage(coverage)
        if coverage in self.cache:
            self.cache_hits += 1
            return self.cache[coverage]
        self.calls += 1
        if not quick_coverage_possible(coverage, self.max_route_people):
            self.cache[coverage] = None
            return None
        fixed = fixed_airport(coverage)
        required = required_nodes(coverage)
        edges = shuttle_edges(coverage)
        airports = (fixed,) if fixed else AIRPORTS
        best: Optional[Route] = None

        def examine(sequences: Sequence[Tuple[str, ...]]) -> None:
            nonlocal best
            for service_sequence in sequences:
                for airport in airports:
                    for aircraft_type in AC:
                        for stops, refuels, fuel_result in path_variants(
                            service_sequence, airport, aircraft_type,
                        ):
                            _, fuel, leg_distances, arrivals, departures, _ = fuel_result
                            assignment = best_assignment(
                                coverage, stops, airport, aircraft_type,
                                leg_distances, arrivals, departures,
                                self.assignment_limit,
                            )
                            if assignment is None:
                                continue
                            pairs, intransit, pass_km = assignment
                            metrics = Metrics(
                                time_min=arrivals[-1],
                                intransit_min=intransit,
                                fuel_kg=fuel,
                                pass_km=pass_km,
                                avail_km=AC[aircraft_type]["seats"] * sum(leg_distances),
                            )
                            candidate = Route(
                                airport, aircraft_type, stops, refuels,
                                coverage, pairs, metrics,
                            )
                            if best is None or route_key(candidate) < route_key(best):
                                best = candidate

        simple = simple_service_sequences(required, edges)
        examine(simple)
        # 重复服务设施主要用于有向穿梭环或简单顺序下容量不可行的情况。
        if best is None:
            examine(repeated_service_sequences(required, edges))
        self.cache[coverage] = best
        return best


class RoutePool:
    def __init__(self, max_patterns: int) -> None:
        self.max_patterns = max(1, max_patterns)
        self.patterns: List[Pattern] = []
        self.by_signature: Dict[Tuple, int] = {}

    def __len__(self) -> int:
        return len(self.patterns)

    def add(self, route: Route, source: str) -> Tuple[int, bool]:
        signature = route.signature
        if signature in self.by_signature:
            pid = self.by_signature[signature]
            self.patterns[pid].sources.add(source)
            return pid, False
        if len(self.patterns) >= self.max_patterns:
            return -1, False
        pid = len(self.patterns)
        self.patterns.append(Pattern(pid, route, {source}))
        self.by_signature[signature] = pid
        return pid, True


def best_pack_one_class(g: int, total: int, oracle: RouteOracle):
    options = {}
    for size in range(1, min(MAX_SEATS, total) + 1):
        route = oracle.optimize(((g, size),))
        if route is not None:
            options[size] = route
    dp: List[Optional[Tuple[Metrics, List[Route]]]] = [None] * (total + 1)
    dp[0] = (Metrics(), [])
    for n in range(1, total + 1):
        best = None
        for size, route in options.items():
            if size > n or dp[n - size] is None:
                continue
            previous, plan = dp[n - size]
            metrics = add_metrics(previous, route.metrics)
            if best is None or lex_better(metrics, best[0]):
                best = (metrics, plan + [route])
        dp[n] = best
    if dp[total] is None:
        raise RuntimeError(f"需求类别 {CLASSES[g]} 无法构造直接运输初解")
    return dp[total][1]


def build_initial_routes(totals: Sequence[int], oracle: RouteOracle) -> List[Route]:
    routes = []
    for g, total in enumerate(totals):
        routes.extend(best_pack_one_class(g, total, oracle))
        if (g + 1) % 25 == 0 or g + 1 == len(totals):
            print(f"  初始打包 {g + 1}/{len(totals)} 类，当前架次={len(routes)}")
    return routes


def route_nodes(route: Route) -> set[str]:
    return set(route.stops)


def quick_pair_possible(a: Route, b: Route, oracle: RouteOracle) -> bool:
    coverage = combine_coverage(a.coverage, b.coverage)
    return quick_coverage_possible(coverage, oracle.max_route_people)


def pair_priority(a: Route, b: Route) -> Tuple:
    nodes_a, nodes_b = route_nodes(a), route_nodes(b)
    proximity = min(D[x][y] for x in nodes_a for y in nodes_b)
    shared = len(nodes_a & nodes_b)
    return (proximity, -shared, -(a.metrics.time_min + b.metrics.time_min), a.signature, b.signature)


def improve_by_merging(
    routes: Sequence[Route],
    oracle: RouteOracle,
    pool: RoutePool,
    neighbor_count: int,
    max_merges: int,
    deadline: Optional[float] = None,
) -> List[Route]:
    active: Dict[int, Route] = {i: route for i, route in enumerate(routes)}
    next_id = len(active)
    total = aggregate_routes(active.values())
    accepted = 0
    while (max_merges <= 0 or accepted < max_merges) and (
        deadline is None or time.monotonic() < deadline
    ):
        candidate_pairs = set()
        active_ids = sorted(active)
        for i in active_ids:
            if deadline is not None and time.monotonic() >= deadline:
                break
            ranked = []
            for j in active_ids:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if i >= j or not quick_pair_possible(active[i], active[j], oracle):
                    continue
                ranked.append((pair_priority(active[i], active[j]), i, j))
            ranked.sort(key=lambda x: x[0])
            candidate_pairs.update((i, j) for _, i, j in ranked[:neighbor_count])
        best_choice = None
        best_total = total
        for i, j in candidate_pairs:
            if deadline is not None and time.monotonic() >= deadline:
                break
            merged = oracle.optimize(combine_coverage(active[i].coverage, active[j].coverage))
            if merged is None:
                continue
            pool.add(merged, "merge_candidate")
            candidate_total = sub_add_metrics(total, [active[i], active[j]], [merged])
            if lex_better(candidate_total, best_total):
                best_total = candidate_total
                best_choice = (i, j, merged)
        if best_choice is None:
            break
        i, j, merged = best_choice
        del active[i]
        del active[j]
        active[next_id] = merged
        next_id += 1
        accepted += 1
        old_time = total.time_min
        total = best_total
        print(
            f"  merge {accepted:3d}: time {old_time} -> {total.time_min}, "
            f"sorties={len(active)}"
        )
    return list(active.values())


def scaled(value: float) -> int:
    return int(round(value * SCALE))


def selected_metrics(pool: RoutePool, selected: Counter) -> Metrics:
    return aggregate_routes(
        pool.patterns[pid].route
        for pid, multiplicity in selected.items()
        for _ in range(multiplicity)
    )


def solve_master(
    pool: RoutePool,
    totals: Sequence[int],
    incumbent: Counter,
    time_limit: float,
    workers: int,
    seed: int,
) -> MasterResult:
    model = cp_model.CpModel()
    variables = []
    for pattern in pool.patterns:
        bounds = [totals[g] // count for g, count in pattern.route.coverage]
        variables.append(model.NewIntVar(0, min(bounds), f"x_{pattern.pid}"))
    for g, required in enumerate(totals):
        terms = []
        for pattern in pool.patterns:
            count = dict(pattern.route.coverage).get(g, 0)
            if count:
                terms.append(count * variables[pattern.pid])
        if not terms:
            raise RuntimeError(f"路线池没有覆盖需求类别 {CLASSES[g]}")
        model.Add(sum(terms) == required)
    for pattern in pool.patterns:
        model.AddHint(variables[pattern.pid], int(incumbent.get(pattern.pid, 0)))

    time_expr = sum(p.route.metrics.time_min * variables[p.pid] for p in pool.patterns)
    transit_expr = sum(p.route.metrics.intransit_min * variables[p.pid] for p in pool.patterns)
    fuel_expr = sum(scaled(p.route.metrics.fuel_kg) * variables[p.pid] for p in pool.patterns)
    pass_expr = sum(scaled(p.route.metrics.pass_km) * variables[p.pid] for p in pool.patterns)
    avail_expr = sum(scaled(p.route.metrics.avail_km) * variables[p.pid] for p in pool.patterns)

    deadline = time.monotonic() + max(0.1, time_limit)
    stages = []
    best_selected = Counter(incumbent)
    all_optimal = True

    def run(expr, maximize: bool, name: str):
        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            stages.append({"stage": name, "status": "SKIPPED_TIME_LIMIT"})
            return None, None
        model.Maximize(expr) if maximize else model.Minimize(expr)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.05, remaining)
        solver.parameters.num_search_workers = max(1, workers)
        solver.parameters.random_seed = seed
        t0 = time.monotonic()
        status = solver.Solve(model)
        info = {
            "stage": name,
            "status": solver.StatusName(status),
            "wall_time_s": round(time.monotonic() - t0, 3),
        }
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            info["objective"] = int(round(solver.ObjectiveValue()))
        stages.append(info)
        return solver, status

    for name, expr in (
        ("total_aircraft_time", time_expr),
        ("total_person_intransit", transit_expr),
        ("total_fuel_scaled", fuel_expr),
    ):
        solver, status = run(expr, False, name)
        if solver is None or status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            all_optimal = False
            break
        best_selected = Counter({
            p.pid: solver.Value(variables[p.pid])
            for p in pool.patterns if solver.Value(variables[p.pid])
        })
        if status != cp_model.OPTIMAL:
            all_optimal = False
            break
        model.Add(expr == int(round(solver.ObjectiveValue())))
    else:
        current_pass = sum(
            scaled(pool.patterns[pid].route.metrics.pass_km) * n
            for pid, n in best_selected.items()
        )
        current_avail = sum(
            scaled(pool.patterns[pid].route.metrics.avail_km) * n
            for pid, n in best_selected.items()
        )
        ratio_optimal = False
        for iteration in range(20):
            divisor = math.gcd(abs(current_pass), abs(current_avail)) or 1
            q_num, q_den = current_pass // divisor, current_avail // divisor
            solver, status = run(
                q_den * pass_expr - q_num * avail_expr,
                True,
                f"seat_utilization_{iteration + 1}",
            )
            if solver is None or status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                all_optimal = False
                break
            candidate = Counter({
                p.pid: solver.Value(variables[p.pid])
                for p in pool.patterns if solver.Value(variables[p.pid])
            })
            candidate_pass = sum(
                scaled(pool.patterns[pid].route.metrics.pass_km) * n
                for pid, n in candidate.items()
            )
            candidate_avail = sum(
                scaled(pool.patterns[pid].route.metrics.avail_km) * n
                for pid, n in candidate.items()
            )
            best_selected = candidate
            if status != cp_model.OPTIMAL:
                all_optimal = False
                break
            residual = q_den * candidate_pass - q_num * candidate_avail
            if residual <= 0:
                ratio_optimal = True
                break
            current_pass, current_avail = candidate_pass, candidate_avail
        if not ratio_optimal:
            all_optimal = False

    return MasterResult(
        best_selected,
        selected_metrics(pool, best_selected),
        stages,
        all_optimal,
    )


def selected_ids(selected: Counter) -> List[int]:
    return [pid for pid, n in selected.items() for _ in range(n)]


def random_split_coverage(
    combined: Sequence[Tuple[int, int]],
    bins: int,
    rng: random.Random,
    max_route_people: int,
) -> Optional[List[Tuple[Tuple[int, int], ...]]]:
    items = [g for g, count in combined for _ in range(count)]
    if len(items) < bins or len(items) > bins * max_route_people:
        return None
    rng.shuffle(items)
    groups = [Counter() for _ in range(bins)]
    for j in range(bins):
        groups[j][items.pop()] += 1
    for g in items:
        choices = list(range(bins))
        rng.shuffle(choices)
        choices.sort(key=lambda j: sum(groups[j].values()))
        placed = False
        for j in choices:
            trial = groups[j].copy()
            trial[g] += 1
            coverage = canonical_coverage(trial)
            if quick_coverage_possible(coverage, max_route_people):
                groups[j] = trial
                placed = True
                break
        if not placed:
            return None
    return [canonical_coverage(group) for group in groups]


class CandidateGenerator:
    OPS = ("merge", "relocate", "swap", "repartition2", "three_to_two")

    def __init__(self, oracle: RouteOracle, pool: RoutePool, rng: random.Random) -> None:
        self.oracle = oracle
        self.pool = pool
        self.rng = rng
        self.weights = {"merge": 0.8, "relocate": 1.0, "swap": 1.0,
                        "repartition2": 1.4, "three_to_two": 1.4}
        self.uses = Counter()
        self.added = Counter()

    def choose(self) -> str:
        return self.rng.choices(
            list(self.weights), weights=[self.weights[x] for x in self.weights], k=1,
        )[0]

    def generate_routes(self, coverages, op: str) -> Tuple[int, set[int]]:
        generated = []
        for coverage in coverages:
            route = self.oracle.optimize(coverage)
            if route is None:
                return 0, set()
            generated.append(route)
        count = 0
        ids = set()
        for route in generated:
            pid, added = self.pool.add(route, op)
            count += int(added)
            if added:
                ids.add(pid)
        self.added[op] += count
        return count, ids

    def apply(self, selected: Counter) -> Tuple[str, int, set[int]]:
        ids = selected_ids(selected)
        op = self.choose()
        self.uses[op] += 1
        needed = 3 if op == "three_to_two" else 2
        if len(ids) < needed:
            return op, 0, set()
        chosen = self.rng.sample(ids, needed)
        routes = [self.pool.patterns[pid].route for pid in chosen]
        if op == "three_to_two":
            combined = combine_coverage(*(r.coverage for r in routes))
            split = random_split_coverage(
                combined, 2, self.rng, self.oracle.max_route_people,
            )
            return (op, 0, set()) if split is None else (op, *self.generate_routes(split, op))
        a, b = routes
        ca, cb = sparse_to_counter(a.coverage), sparse_to_counter(b.coverage)
        if op == "merge":
            return op, *self.generate_routes([combine_coverage(a.coverage, b.coverage)], op)
        if op == "relocate":
            movable = [g for g, n in ca.items() if n]
            if not movable:
                return op, 0, set()
            g = self.rng.choice(movable)
            amount = self.rng.randint(1, min(3, ca[g]))
            ca[g] -= amount
            cb[g] += amount
            coverages = [canonical_coverage(ca), canonical_coverage(cb)]
            coverages = [x for x in coverages if x]
            return op, *self.generate_routes(coverages, op)
        if op == "swap":
            ga, gb = list(ca), list(cb)
            if not ga or not gb:
                return op, 0, set()
            x, y = self.rng.choice(ga), self.rng.choice(gb)
            nx = self.rng.randint(1, min(2, ca[x]))
            ny = self.rng.randint(1, min(2, cb[y]))
            ca[x] -= nx
            cb[x] += nx
            cb[y] -= ny
            ca[y] += ny
            return op, *self.generate_routes(
                [canonical_coverage(ca), canonical_coverage(cb)], op,
            )
        combined = combine_coverage(a.coverage, b.coverage)
        split = random_split_coverage(combined, 2, self.rng, self.oracle.max_route_people)
        return (op, 0, set()) if split is None else (op, *self.generate_routes(split, op))

    def adapt(self, generated: Dict[str, set[int]], selected: Counter, improved: bool) -> None:
        used = set(selected)
        for op in self.OPS:
            chosen = len(generated.get(op, set()) & used)
            reward = 0.2 + 0.4 * len(generated.get(op, set())) + 3.0 * chosen
            if improved and chosen:
                reward += 6.0
            self.weights[op] = 0.85 * self.weights[op] + 0.15 * reward
        mean = sum(self.weights.values()) / len(self.weights)
        for op in self.weights:
            self.weights[op] = max(0.05, self.weights[op] / mean)


def reconstruct(
    pool: RoutePool,
    selected: Counter,
    grouped: Dict[Tuple[str, str], List[dict]],
) -> List[OperatedRoute]:
    buckets = {c: deque(sorted(grouped[c], key=lambda d: d["pid"])) for c in CLASSES}
    routes = []
    for pid in sorted(selected):
        route = pool.patterns[pid].route
        for _ in range(selected[pid]):
            people = []
            for g, count in route.coverage:
                for _ in range(count):
                    if not buckets[CLASSES[g]]:
                        raise RuntimeError(f"Master 对 {CLASSES[g]} 超额覆盖")
                    people.append(buckets[CLASSES[g]].popleft())
            routes.append(OperatedRoute(route, people))
    leftovers = {c: len(q) for c, q in buckets.items() if q}
    if leftovers:
        raise RuntimeError(f"Master 未覆盖全部人员：{list(leftovers.items())[:5]}")
    return routes


def simulate_operated(operated: OperatedRoute) -> Metrics:
    route = operated.template
    if route.airport not in AIRPORT_SET or len(route.stops) > MAX_LANDINGS:
        raise ValueError("架次机场或海上着陆次数非法")
    if len(route.stops) != len(route.refuels):
        raise ValueError("stops/refuels 长度不一致")
    feasible, fuel, leg_distances, arrivals, departures, _ = fuel_feasible(
        route.airport, route.aircraft_type, route.stops, route.refuels,
    )
    if not feasible:
        raise ValueError("架次燃油或安全余油约束不满足")
    pair_map = {g: (pickup, delivery) for g, pickup, delivery in route.pairs}
    loads = [0] * len(leg_distances)
    transit = 0
    for person in operated.people:
        g = CLASS_INDEX[(person["origin"], person["dest"])]
        if g not in pair_map:
            raise ValueError(f"人员 {person['pid']} 所属OD类不在路线中")
        pickup, delivery = pair_map[g]
        if not 0 <= pickup < delivery <= len(route.stops) + 1:
            raise ValueError(f"人员 {person['pid']} 上下机顺序非法")
        expected_pickup = route.airport if pickup == 0 else route.stops[pickup - 1]
        expected_delivery = route.airport if delivery == len(route.stops) + 1 else route.stops[delivery - 1]
        if person["origin"] == LAND:
            if pickup != 0:
                raise ValueError(f"LAND出海人员 {person['pid']} 未在机场上机")
        elif expected_pickup != person["origin"]:
            raise ValueError(f"人员 {person['pid']} 上机地点错误")
        if person["dest"] == LAND:
            if delivery != len(route.stops) + 1:
                raise ValueError(f"LAND海返人员 {person['pid']} 未在机场下机")
        elif expected_delivery != person["dest"]:
            raise ValueError(f"人员 {person['pid']} 下机地点错误")
        if person["dest"] not in AIRPORT_SET | {LAND}:
            first_after = next(
                (i for i, node in enumerate(route.stops, start=1)
                 if i > pickup and node == person["dest"]),
                None,
            )
            if delivery != first_after:
                raise ValueError(f"人员 {person['pid']} 未在上机后首次到达终点时下机")
        for leg in range(pickup, delivery):
            loads[leg] += 1
        transit += arrivals[delivery] - departures[pickup]
    seats = int(AC[route.aircraft_type]["seats"])
    if any(load > seats for load in loads):
        raise ValueError(f"架次动态载客量超过 {route.aircraft_type} 座位数")
    metrics = Metrics(
        arrivals[-1], transit, fuel,
        sum(load * distance for load, distance in zip(loads, leg_distances)),
        seats * sum(leg_distances),
    )
    cached = route.metrics
    if (
        metrics.time_min != cached.time_min
        or metrics.intransit_min != cached.intransit_min
        or abs(metrics.fuel_kg - cached.fuel_kg) > 1e-6
        or abs(metrics.pass_km - cached.pass_km) > 1e-6
        or abs(metrics.avail_km - cached.avail_km) > 1e-6
    ):
        raise ValueError("路线缓存指标与独立仿真不一致")
    return metrics


def validate_solution(routes: Sequence[OperatedRoute], demands: Sequence[dict]) -> Metrics:
    expected = {d["pid"] for d in demands}
    seen = []
    total = Metrics()
    for operated in routes:
        total = add_metrics(total, simulate_operated(operated))
        seen.extend(person["pid"] for person in operated.people)
    if len(seen) != len(set(seen)):
        duplicates = [pid for pid, count in Counter(seen).items() if count > 1]
        raise ValueError(f"人员重复安排：{duplicates[:10]}")
    if set(seen) != expected:
        raise ValueError(
            f"人员覆盖不完整，missing={sorted(expected-set(seen))[:10]}, "
            f"extra={sorted(set(seen)-expected)[:10]}"
        )
    return total


def write_output(routes: Sequence[OperatedRoute], routes_path: Path, assign_path: Path) -> None:
    routes_path.parent.mkdir(parents=True, exist_ok=True)
    assign_path.parent.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    flight_numbers = []
    for operated in routes:
        aircraft_type = operated.template.aircraft_type
        counters[aircraft_type] += 1
        flight_numbers.append(counters[aircraft_type])
    with routes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["aircraft_type", "flight_no", "stop_order", "facility_id", "refuel"])
        for operated, flight_no in zip(routes, flight_numbers):
            route = operated.template
            writer.writerow([route.aircraft_type, flight_no, 0, route.airport, 0])
            for i, (stop, refuel) in enumerate(zip(route.stops, route.refuels), start=1):
                writer.writerow([route.aircraft_type, flight_no, i, stop, int(refuel)])
            writer.writerow([
                route.aircraft_type, flight_no, len(route.stops) + 1, route.airport, 0,
            ])
    with assign_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "person_id", "aircraft_type", "flight_no",
            "pickup_stop_order", "delivery_stop_order",
        ])
        for operated, flight_no in zip(routes, flight_numbers):
            route = operated.template
            pair_map = {g: (pickup, delivery) for g, pickup, delivery in route.pairs}
            for person in operated.people:
                g = CLASS_INDEX[(person["origin"], person["dest"])]
                pickup, delivery = pair_map[g]
                writer.writerow([
                    person["pid"], route.aircraft_type, flight_no, pickup, delivery,
                ])


def print_metrics(title: str, metrics: Metrics, sorties: int, pool_size: int) -> None:
    print(title)
    print(f"  总飞机使用时间: {metrics.time_min} min ({metrics.time_min/60:.2f} h)")
    print(f"  人员总在途时间: {metrics.intransit_min} min ({metrics.intransit_min/60:.2f} h)")
    print(f"  总架次数:       {sorties}")
    print(f"  总燃油:         {metrics.fuel_kg:.2f} kg")
    print(f"  座位利用率:     {metrics.seat_util:.6f}")
    print(f"  路线池:         {pool_size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q2 pickup-delivery route pool solver (Python 3.11)"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dist", type=Path, default=None)
    parser.add_argument("--demand", type=Path, default=None)
    parser.add_argument("--routes-out", type=Path, default=ROUTES_OUT)
    parser.add_argument("--assign-out", type=Path, default=ASSIGN_OUT)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--merge-neighbors", type=int, default=10)
    parser.add_argument("--max-merges", type=int, default=0, help="0表示合并到局部收敛")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ops-per-epoch", type=int, default=120)
    parser.add_argument("--master-time", type=float, default=60.0)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-patterns", type=int, default=20000)
    parser.add_argument("--max-route-people", type=int, default=38)
    parser.add_argument("--assignment-limit", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    global CLASSES, CLASS_INDEX
    args = parse_args()
    start = time.monotonic()
    rng = random.Random(args.seed)
    dist_path = (args.dist or args.data_dir / "distances.csv").resolve()
    demand_path = (args.demand or args.data_dir / "peopleQ2.csv").resolve()
    print(f"加载距离：{dist_path}")
    load_distances(dist_path)
    demands, classes, totals, grouped = load_demands(demand_path)
    CLASSES = classes
    CLASS_INDEX = {c: i for i, c in enumerate(classes)}
    kinds = Counter(
        "outbound" if o in AIRPORT_SET | {LAND}
        else "inbound" if d in AIRPORT_SET | {LAND}
        else "shuttle"
        for o, d in classes
        for _ in range(len(grouped[(o, d)]))
    )
    print(
        f"people={len(demands)}, classes={len(classes)}, facilities={len(D)}, "
        f"outbound={kinds['outbound']}, inbound={kinds['inbound']}, shuttle={kinds['shuttle']}"
    )

    oracle = RouteOracle(
        max_route_people=max(MAX_SEATS, args.max_route_people),
        assignment_limit=max(1, args.assignment_limit),
    )
    pool = RoutePool(max(1, args.max_patterns))

    print("\nStage 1: 单OD类别精确批量打包……")
    routes = build_initial_routes(totals, oracle)
    for route in routes:
        pool.add(route, "initial")
    initial_metrics = aggregate_routes(routes)
    print_metrics("Stage 1：", initial_metrics, len(routes), len(pool))

    print("\nStage 2: 相关路线节约合并……")
    routes = improve_by_merging(
        routes,
        oracle,
        pool,
        max(1, args.merge_neighbors),
        max(0, args.max_merges),
        start + args.time_limit,
    )
    for route in routes:
        pool.add(route, "merge_solution")
    current = Counter(pool.by_signature[route.signature] for route in routes)
    current_metrics = selected_metrics(pool, current)
    print_metrics("Stage 2：", current_metrics, len(routes), len(pool))

    first = solve_master(
        pool, totals, current,
        min(args.master_time, max(0.1, args.time_limit - (time.monotonic() - start))),
        args.workers, args.seed,
    )
    history = [{
        "epoch": 0, "pool_size": len(pool),
        "optimal_within_pool": first.optimal_within_pool,
        "stages": first.stages,
    }]
    if lex_better(first.metrics, current_metrics):
        current, current_metrics = first.selected, first.metrics
    elif not lex_better(current_metrics, first.metrics):
        current = first.selected
    print_metrics("首次路线池重组：", current_metrics, sum(current.values()), len(pool))

    generator = CandidateGenerator(oracle, pool, rng)
    last_optimal = first.optimal_within_pool
    epochs_done = 0
    for epoch in range(1, max(0, args.epochs) + 1):
        if time.monotonic() - start >= args.time_limit or len(pool) >= pool.max_patterns:
            break
        generated: Dict[str, set[int]] = defaultdict(set)
        added = 0
        for _ in range(max(0, args.ops_per_epoch)):
            if time.monotonic() - start >= args.time_limit:
                break
            op, count, ids = generator.apply(current)
            added += count
            generated[op].update(ids)
        if added == 0:
            print(f"epoch {epoch}: 没有新路线，停止ALNS。")
            break
        remaining = args.time_limit - (time.monotonic() - start)
        if remaining <= 0.05:
            break
        result = solve_master(
            pool, totals, current, min(args.master_time, remaining),
            args.workers, args.seed + epoch,
        )
        improved = lex_better(result.metrics, current_metrics)
        if improved:
            current, current_metrics = result.selected, result.metrics
        elif not lex_better(current_metrics, result.metrics):
            current = result.selected
        generator.adapt(generated, current, improved)
        last_optimal = result.optimal_within_pool
        epochs_done = epoch
        history.append({
            "epoch": epoch, "pool_size": len(pool), "new_patterns": added,
            "improved": improved, "optimal_within_pool": result.optimal_within_pool,
            "stages": result.stages,
        })
        print_metrics(
            f"epoch {epoch:02d}（新增{added}, pool-opt={last_optimal}）：",
            current_metrics, sum(current.values()), len(pool),
        )

    operated = reconstruct(pool, current, grouped)
    final_metrics = validate_solution(operated, demands)
    write_output(operated, args.routes_out, args.assign_out)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": "Q2 pickup-delivery route pool + ALNS + lexicographic set partitioning",
        "python": sys.version.split()[0],
        "people": len(demands),
        "classes": len(classes),
        "route_pool_size": len(pool),
        "epochs_completed": epochs_done,
        "sorties": len(operated),
        "metrics": {
            "total_aircraft_time_min": final_metrics.time_min,
            "total_person_intransit_min": final_metrics.intransit_min,
            "total_fuel_kg": round(final_metrics.fuel_kg, 6),
            "seat_utilization": final_metrics.seat_util,
        },
        "objective_order": [
            "total_aircraft_time_min", "total_person_intransit_min",
            "total_fuel_kg", "seat_utilization_max",
        ],
        "claim_scope": (
            "optimal_over_current_generated_route_pool" if last_optimal
            else "best_valid_solution_found; final route-pool proof incomplete"
        ),
        "original_problem_global_optimum_claimed": False,
        "route_oracle_calls": oracle.calls,
        "route_oracle_cache_hits": oracle.cache_hits,
        "operator_uses": dict(generator.uses),
        "operator_new_patterns": dict(generator.added),
        "master_history": history,
        "elapsed_s": round(time.monotonic() - start, 3),
        "routes_out": str(args.routes_out.resolve()),
        "assignments_out": str(args.assign_out.resolve()),
    }
    with args.summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print_metrics("\n=== Q2 最终结果 ===", final_metrics, len(operated), len(pool))
    print("完整取送、动态容量、燃油、人员覆盖与首次到达终点校验通过。")
    print(f"routes -> {args.routes_out.resolve()}")
    print(f"assignments -> {args.assign_out.resolve()}")
    print(f"summary -> {args.summary_out.resolve()}")
    print(
        "当前生成路线池内四层最优性已证明。" if last_optimal
        else "最后一次路线池重组未完成全部证明；输出仍是完整可行解。"
    )


if __name__ == "__main__":
    main()
