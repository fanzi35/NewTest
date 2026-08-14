from __future__ import annotations

import argparse
import itertools
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import (
    classical_mds,
    ensure_dir,
    load_distance_matrix,
    load_people_q1,
    save_dataframe_excel,
    save_multi_sheet_excel,
    setup_matplotlib_chinese,
)


# 全局距离矩阵，测试时可直接替换
D: dict[str, dict[str, float]] = {}


@dataclass(frozen=True)
class DemandGroup:
    """聚合后的需求组。"""

    group_id: str
    origin_requirement: str
    destination: str
    count: int
    person_ids: list[str]


@dataclass(frozen=True)
class RouteTemplate:
    """不含具体人员编号的架次模板。"""

    airport: str
    aircraft_type: str
    stop_fids: tuple[str, ...]
    refuels: tuple[bool, ...]
    time_min: int
    fuel_kg: float
    pass_km: float
    avail_km: float
    intransit_min: int

    @property
    def seat_util(self) -> float:
        if self.avail_km <= config.EPS:
            return 0.0
        return self.pass_km / self.avail_km


@dataclass
class Sortie:
    """带具体人员的架次。"""

    airport: str
    aircraft_type: str
    stop_fids: list[str]
    refuels: list[bool]
    demands: list[dict]
    time_min: int
    fuel_kg: float
    pass_km: float
    avail_km: float
    intransit_min: int

    @property
    def seat_util(self) -> float:
        if self.avail_km <= config.EPS:
            return 0.0
        return self.pass_km / self.avail_km


@dataclass(frozen=True)
class Metrics:
    """方案评价指标。"""

    total_time_min: int = 0
    total_intransit_min: int = 0
    total_fuel_kg: float = 0.0
    total_sorties: int = 0
    pass_km: float = 0.0
    avail_km: float = 0.0

    @property
    def seat_util(self) -> float:
        if self.avail_km <= config.EPS:
            return 0.0
        return self.pass_km / self.avail_km


@dataclass
class ALNSTrace:
    """ALNS 收敛轨迹。"""

    seed: int
    iteration: int
    elapsed_sec: float
    current_time: int
    best_time: int


@dataclass
class SolveResult:
    """最终结果。"""

    sorties: list[Sortie]
    routes_df: pd.DataFrame
    assignments_df: pd.DataFrame
    summary_df: pd.DataFrame
    convergence_df: pd.DataFrame
    compare_df: pd.DataFrame
    seed_df: pd.DataFrame


_ROUTE_CACHE: dict[tuple[str, str, tuple[tuple[str, int], ...]], Optional[RouteTemplate]] = {}
_TEMPLATE_CACHE: dict[tuple[Optional[str], tuple[tuple[str, int], ...]], Optional[RouteTemplate]] = {}
_SEQUENCE_CACHE: dict[tuple[tuple[str, ...], int], list[tuple[str, ...]]] = {}


def clear_internal_caches() -> None:
    """清空内部缓存。"""
    _ROUTE_CACHE.clear()
    _TEMPLATE_CACHE.clear()
    _SEQUENCE_CACHE.clear()


def flight_minutes(distance_km: float, aircraft_type: str) -> int:
    """单航段飞行时间，向上取整。"""
    speed = config.AIRCRAFT[aircraft_type]["speed"]
    return math.ceil(60.0 * distance_km / speed - config.EPS)


def load_distances(path: Path) -> None:
    """读取距离矩阵。"""
    global D
    df = load_distance_matrix(path)
    cols = [col for col in df.columns if col != "from_id"]
    D = {}
    for row in df.itertuples(index=False):
        origin = row[0]
        D[origin] = {}
        for idx, dest in enumerate(cols, start=1):
            D[origin][dest] = float(row[idx])


def load_demands(path: Path) -> list[dict]:
    """读取第一问需求。"""
    df = load_people_q1(path)
    return [{"pid": row.person_id, "origin": row.origin_id, "dest": row.destination_id} for row in df.itertuples(index=False)]


def build_groups(demands: list[dict]) -> dict[str, DemandGroup]:
    """按起点要求和目的设施聚合。"""
    bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in demands:
        bucket[(item["origin"], item["dest"])].append(item["pid"])
    groups = {}
    for (origin, dest), person_ids in sorted(bucket.items()):
        group_id = f"{origin}->{dest}"
        groups[group_id] = DemandGroup(group_id, origin, dest, len(person_ids), person_ids)
    return groups


def metrics_from_sorties(sorties: list[Sortie]) -> Metrics:
    """统计方案总指标。"""
    return Metrics(
        total_time_min=sum(item.time_min for item in sorties),
        total_intransit_min=sum(item.intransit_min for item in sorties),
        total_fuel_kg=sum(item.fuel_kg for item in sorties),
        total_sorties=len(sorties),
        pass_km=sum(item.pass_km for item in sorties),
        avail_km=sum(item.avail_km for item in sorties),
    )


def solution_key(sorties: list[Sortie]) -> tuple:
    """方案字典序评价。"""
    metrics = metrics_from_sorties(sorties)
    return (
        metrics.total_time_min,
        metrics.total_intransit_min,
        round(metrics.total_fuel_kg, 6),
        -round(metrics.seat_util, 6),
    )


def is_better_solution(left: list[Sortie], right: Optional[list[Sortie]]) -> bool:
    """比较两套方案优劣。"""
    if right is None:
        return True
    return solution_key(left) < solution_key(right)


def route_key(template: RouteTemplate) -> tuple:
    """模板评价关键字。"""
    return (
        template.time_min,
        template.intransit_min,
        round(template.fuel_kg, 6),
        -round(template.seat_util, 6),
        template.airport,
        template.aircraft_type,
        template.stop_fids,
        template.refuels,
    )


def fixed_airport_of_demands(demands: list[dict]) -> Optional[str]:
    """识别是否存在固定机场要求。"""
    fixed = {item["origin"] for item in demands if item["origin"] in config.AIRPORTS}
    if len(fixed) > 1:
        return "__CONFLICT__"
    return next(iter(fixed)) if fixed else None


def demand_signature(demands: list[dict]) -> tuple[tuple[str, int], ...]:
    """将需求列表转为设施计数。"""
    return tuple(sorted(Counter(item["dest"] for item in demands).items()))


def build_stop_sequences(delivery_nodes: tuple[str, ...], max_landings: int) -> list[tuple[str, ...]]:
    """枚举访问设施集合下的所有停靠序列。"""
    cache_key = (delivery_nodes, max_landings)
    if cache_key in _SEQUENCE_CACHE:
        return _SEQUENCE_CACHE[cache_key]

    available_gas = [node for node in config.GAS_STATIONS if node in D]
    sequences: set[tuple[str, ...]] = set()
    base_permutations = list(itertools.permutations(delivery_nodes))
    max_extra = min(2, max_landings - len(delivery_nodes))
    for perm in base_permutations:
        sequences.add(tuple(perm))
        for extra in range(1, max_extra + 1):
            for gas_nodes in itertools.product(available_gas, repeat=extra):
                total_len = len(perm) + extra
                for slots in itertools.combinations(range(total_len), extra):
                    seq = []
                    perm_idx = 0
                    gas_idx = 0
                    for pos in range(total_len):
                        if pos in slots:
                            seq.append(gas_nodes[gas_idx])
                            gas_idx += 1
                        else:
                            seq.append(perm[perm_idx])
                            perm_idx += 1
                    sequences.add(tuple(seq))
    result = sorted(sequences)
    _SEQUENCE_CACHE[cache_key] = result
    return result


def evaluate_stop_sequence(
    airport: str,
    aircraft_type: str,
    delivery_counts: dict[str, int],
    stop_fids: tuple[str, ...],
) -> Optional[RouteTemplate]:
    """给定停靠序列后精确评估单架次。"""
    aircraft = config.AIRCRAFT[aircraft_type]
    total_pax = sum(delivery_counts.values())
    if total_pax > aircraft["seats"]:
        return None
    if len(stop_fids) > config.MAX_LANDINGS:
        return None
    if not set(delivery_counts).issubset(set(stop_fids)):
        return None

    gas_positions = [idx for idx, node in enumerate(stop_fids) if node in config.GAS_SET]
    best_template = None
    for mask in range(1 << len(gas_positions)):
        refuels = []
        for idx, node in enumerate(stop_fids):
            if node in config.GAS_SET:
                local_pos = gas_positions.index(idx)
                decision = ((mask >> local_pos) & 1) == 1
                if node not in delivery_counts:
                    decision = True
                refuels.append(decision)
            else:
                refuels.append(False)

        remain = aircraft["tank"]
        elapsed = 0
        fuel_used = 0.0
        pass_km = 0.0
        avail_km = 0.0
        intransit = 0
        delivered = set()
        current = airport
        feasible = True

        for idx, stop in enumerate(stop_fids):
            onboard = sum(cnt for fid, cnt in delivery_counts.items() if fid not in delivered)
            distance = D[current][stop]
            need = distance * aircraft["burn"]
            remain -= need
            if remain < aircraft["reserve"] - config.EPS:
                feasible = False
                break

            elapsed += flight_minutes(distance, aircraft_type)
            fuel_used += need
            pass_km += onboard * distance
            avail_km += aircraft["seats"] * distance

            if stop in delivery_counts and stop not in delivered:
                intransit += delivery_counts[stop] * elapsed
                delivered.add(stop)

            elapsed += 20 if refuels[idx] else 10
            if refuels[idx]:
                remain = aircraft["tank"]
            current = stop

        if not feasible or delivered != set(delivery_counts):
            continue

        back_distance = D[current][airport]
        back_need = back_distance * aircraft["burn"]
        remain -= back_need
        if remain < aircraft["reserve"] - config.EPS:
            continue

        elapsed += flight_minutes(back_distance, aircraft_type)
        fuel_used += back_need
        avail_km += aircraft["seats"] * back_distance

        template = RouteTemplate(
            airport=airport,
            aircraft_type=aircraft_type,
            stop_fids=tuple(stop_fids),
            refuels=tuple(refuels),
            time_min=int(elapsed),
            fuel_kg=float(fuel_used),
            pass_km=float(pass_km),
            avail_km=float(avail_km),
            intransit_min=int(intransit),
        )
        if best_template is None or route_key(template) < route_key(best_template):
            best_template = template
    return best_template


def search_route_for_aircraft(
    airport: str,
    aircraft_type: str,
    dest_sig: tuple[tuple[str, int], ...],
) -> Optional[RouteTemplate]:
    """固定机场和机型后搜索最优路线。"""
    cache_key = (airport, aircraft_type, dest_sig)
    if cache_key in _ROUTE_CACHE:
        return _ROUTE_CACHE[cache_key]

    delivery_counts = dict(dest_sig)
    if sum(delivery_counts.values()) > config.AIRCRAFT[aircraft_type]["seats"]:
        _ROUTE_CACHE[cache_key] = None
        return None
    if len(delivery_counts) > config.MAX_LANDINGS:
        _ROUTE_CACHE[cache_key] = None
        return None

    best = None
    for sequence in build_stop_sequences(tuple(delivery_counts), config.MAX_LANDINGS):
        candidate = evaluate_stop_sequence(airport, aircraft_type, delivery_counts, sequence)
        if candidate is None:
            continue
        if best is None or route_key(candidate) < route_key(best):
            best = candidate

    _ROUTE_CACHE[cache_key] = best
    return best


def optimize_template(dest_sig: tuple[tuple[str, int], ...], fixed_airport: Optional[str]) -> Optional[RouteTemplate]:
    """联立优化机场、机型、访问顺序和加油。"""
    cache_key = (fixed_airport, dest_sig)
    if cache_key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[cache_key]

    total_pax = sum(cnt for _, cnt in dest_sig)
    if total_pax > config.MAX_SEATS or len(dest_sig) > config.MAX_LANDINGS:
        _TEMPLATE_CACHE[cache_key] = None
        return None

    airports = [fixed_airport] if fixed_airport else [airport for airport in config.AIRPORTS if airport in D]
    best = None
    for airport in airports:
        for aircraft_type in config.AIRCRAFT:
            if total_pax > config.AIRCRAFT[aircraft_type]["seats"]:
                continue
            candidate = search_route_for_aircraft(airport, aircraft_type, dest_sig)
            if candidate is None:
                continue
            if best is None or route_key(candidate) < route_key(best):
                best = candidate

    _TEMPLATE_CACHE[cache_key] = best
    return best


def template_to_sortie(template: RouteTemplate, demands: list[dict]) -> Sortie:
    """模板恢复成具体架次。"""
    return Sortie(
        airport=template.airport,
        aircraft_type=template.aircraft_type,
        stop_fids=list(template.stop_fids),
        refuels=list(template.refuels),
        demands=list(demands),
        time_min=template.time_min,
        fuel_kg=template.fuel_kg,
        pass_km=template.pass_km,
        avail_km=template.avail_km,
        intransit_min=template.intransit_min,
    )


def optimize_demands(demands: list[dict]) -> Optional[Sortie]:
    """对一组人员构造最优单架次。"""
    fixed_airport = fixed_airport_of_demands(demands)
    if fixed_airport == "__CONFLICT__":
        return None
    template = optimize_template(demand_signature(demands), fixed_airport)
    if template is None:
        return None
    return template_to_sortie(template, demands)


def build_direct_baseline(groups: dict[str, DemandGroup]) -> list[Sortie]:
    """构造直飞基线方案。"""
    sorties = []
    for group in groups.values():
        pool = [{"pid": pid, "origin": group.origin_requirement, "dest": group.destination} for pid in group.person_ids]
        while pool:
            best = None
            best_chunk = None
            for size in range(min(len(pool), config.MAX_SEATS), 0, -1):
                chunk = pool[:size]
                candidate = optimize_demands(chunk)
                if candidate is not None:
                    best = candidate
                    best_chunk = chunk
                    break
            if best is None or best_chunk is None:
                raise RuntimeError(f"需求组 {group.group_id} 无法构造可行直飞架次")
            sorties.append(best)
            del pool[: len(best_chunk)]
    sorties.sort(key=lambda item: (item.airport, item.aircraft_type, len(item.demands), item.stop_fids))
    return sorties


def try_merge_sorties(left: Sortie, right: Sortie) -> Optional[Sortie]:
    """尝试合并两个架次。"""
    demands = left.demands + right.demands
    if len(demands) > config.MAX_SEATS:
        return None
    merged = optimize_demands(demands)
    if merged is None:
        return None
    if merged.time_min < left.time_min + right.time_min:
        return merged
    if merged.time_min == left.time_min + right.time_min:
        current = [left, right]
        if solution_key([merged]) < solution_key(current):
            return merged
    return None


def savings_candidates(sorties: list[Sortie]) -> list[tuple[float, int, int, Sortie]]:
    """计算当前所有可行合并及节约量。"""
    out = []
    for i, j in itertools.combinations(range(len(sorties)), 2):
        merged = try_merge_sorties(sorties[i], sorties[j])
        if merged is None:
            continue
        saving = sorties[i].time_min + sorties[j].time_min - merged.time_min
        if saving > 0 or merged.time_min == sorties[i].time_min + sorties[j].time_min:
            out.append((saving, i, j, merged))
    out.sort(key=lambda item: item[0], reverse=True)
    return out


def randomized_savings(sorties: list[Sortie], seed: int) -> list[Sortie]:
    """随机化 Clarke-Wright Savings。"""
    rng = random.Random(seed)
    current = list(sorties)
    while True:
        candidates = savings_candidates(current)
        if not candidates:
            break
        top_k = min(config.SAVINGS_RANDOM_TOP_K, len(candidates))
        chosen = rng.choice(candidates[:top_k])
        _, i, j, merged = chosen
        new_solution = [current[idx] for idx in range(len(current)) if idx not in (i, j)]
        new_solution.append(merged)
        current = sorted(new_solution, key=lambda item: (item.airport, item.aircraft_type, len(item.demands), item.stop_fids))
    return current


def build_savings_initial_solution(groups: dict[str, DemandGroup]) -> tuple[list[Sortie], list[dict]]:
    """多次随机化 Savings，返回最好初始解和过程对比。"""
    baseline = build_direct_baseline(groups)
    compare_rows = [
        {
            "算法": "简单直飞",
            "总飞机使用时间": metrics_from_sorties(baseline).total_time_min,
            "人员总在途时间": metrics_from_sorties(baseline).total_intransit_min,
            "总架次数": len(baseline),
            "总燃油消耗": round(metrics_from_sorties(baseline).total_fuel_kg, 3),
            "座位利用率": round(metrics_from_sorties(baseline).seat_util, 6),
        }
    ]
    best = baseline
    for offset in range(config.SAVINGS_MULTI_STARTS):
        candidate = randomized_savings(baseline, config.RANDOM_SEED + offset)
        if is_better_solution(candidate, best):
            best = candidate
    compare_rows.append(
        {
            "算法": "改进Savings",
            "总飞机使用时间": metrics_from_sorties(best).total_time_min,
            "人员总在途时间": metrics_from_sorties(best).total_intransit_min,
            "总架次数": len(best),
            "总燃油消耗": round(metrics_from_sorties(best).total_fuel_kg, 3),
            "座位利用率": round(metrics_from_sorties(best).seat_util, 6),
        }
    )
    return best, compare_rows


def random_destroy(sorties: list[Sortie], rng: random.Random) -> tuple[list[Sortie], list[dict]]:
    """随机删除若干架次。"""
    remove_count = max(1, int(len(sorties) * config.ALNS_REMOVE_RATIO))
    remove_ids = set(rng.sample(range(len(sorties)), remove_count))
    survivors = []
    removed = []
    for idx, sortie in enumerate(sorties):
        if idx in remove_ids:
            removed.extend(sortie.demands)
        else:
            survivors.append(sortie)
    return survivors, removed


def low_util_destroy(sorties: list[Sortie], rng: random.Random) -> tuple[list[Sortie], list[dict]]:
    """优先删除座位利用率低的架次。"""
    remove_count = max(1, int(len(sorties) * config.ALNS_REMOVE_RATIO))
    order = sorted(range(len(sorties)), key=lambda idx: sorties[idx].seat_util)
    top = order[: max(remove_count, min(len(order), remove_count * 2))]
    remove_ids = set(rng.sample(top, remove_count))
    survivors = []
    removed = []
    for idx, sortie in enumerate(sorties):
        if idx in remove_ids:
            removed.extend(sortie.demands)
        else:
            survivors.append(sortie)
    return survivors, removed


def cluster_destroy(sorties: list[Sortie], rng: random.Random) -> tuple[list[Sortie], list[dict]]:
    """按相近设施团簇删除。"""
    center_sortie = rng.choice(sorties)
    center_nodes = [item["dest"] for item in center_sortie.demands]
    center = rng.choice(center_nodes)
    scored = []
    for idx, sortie in enumerate(sorties):
        distance = min(D[center][item["dest"]] for item in sortie.demands)
        scored.append((distance, idx))
    remove_count = max(1, int(len(sorties) * config.ALNS_REMOVE_RATIO))
    remove_ids = {idx for _, idx in sorted(scored)[:remove_count]}
    survivors = []
    removed = []
    for idx, sortie in enumerate(sorties):
        if idx in remove_ids:
            removed.extend(sortie.demands)
        else:
            survivors.append(sortie)
    return survivors, removed


def detour_destroy(sorties: list[Sortie], rng: random.Random) -> tuple[list[Sortie], list[dict]]:
    """删除平均人均飞行时间较高的架次。"""
    remove_count = max(1, int(len(sorties) * config.ALNS_REMOVE_RATIO))
    scored = []
    for idx, sortie in enumerate(sorties):
        avg = sortie.time_min / max(len(sortie.demands), 1)
        scored.append((avg, idx))
    candidates = [idx for _, idx in sorted(scored, reverse=True)[: max(remove_count, min(len(scored), remove_count * 2))]]
    remove_ids = set(rng.sample(candidates, remove_count))
    survivors = []
    removed = []
    for idx, sortie in enumerate(sorties):
        if idx in remove_ids:
            removed.extend(sortie.demands)
        else:
            survivors.append(sortie)
    return survivors, removed


def best_batch_insertion(target_sorties: list[Sortie], batch: list[dict]) -> bool:
    """把一个小批量需求插入现有解中。"""
    best_index = None
    best_candidate = None
    best_delta = None
    for idx, sortie in enumerate(target_sorties):
        trial_demands = sortie.demands + batch
        candidate = optimize_demands(trial_demands)
        if candidate is None:
            continue
        delta = candidate.time_min - sortie.time_min
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_index = idx
            best_candidate = candidate
    if best_index is not None and best_candidate is not None:
        target_sorties[best_index] = best_candidate
        return True
    candidate = optimize_demands(batch)
    if candidate is None:
        return False
    target_sorties.append(candidate)
    return True


def repair_solution(base_sorties: list[Sortie], removed_demands: list[dict], rng: random.Random) -> list[Sortie]:
    """修复被破坏的方案。"""
    repaired = list(base_sorties)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in removed_demands:
        grouped[(item["origin"], item["dest"])].append(item)

    keys = list(grouped.keys())
    rng.shuffle(keys)
    for key in keys:
        people = grouped[key]
        while people:
            best_batch = None
            for size in range(min(len(people), config.MAX_SEATS), 0, -1):
                batch = people[:size]
                if optimize_demands(batch) is not None:
                    best_batch = batch
                    break
            if best_batch is None:
                raise RuntimeError("存在无法单独运输的小批量需求")
            best_batch_insertion(repaired, best_batch)
            del people[: len(best_batch)]
    repaired = randomized_savings(repaired, rng.randint(1, 10**9))
    return repaired


def select_operator(weights: dict[str, float], rng: random.Random) -> str:
    """按权重抽取算子。"""
    total = sum(weights.values())
    hit = rng.random() * total
    acc = 0.0
    for name, weight in weights.items():
        acc += weight
        if hit <= acc:
            return name
    return next(iter(weights))


def update_weight(weights: dict[str, float], name: str, reward: float) -> None:
    """更新算子权重。"""
    weights[name] = (1.0 - config.ALNS_WEIGHT_DECAY) * weights[name] + config.ALNS_WEIGHT_DECAY * reward


def run_alns(initial_sorties: list[Sortie], seed: int) -> tuple[list[Sortie], list[ALNSTrace]]:
    """运行一轮 ALNS。"""
    rng = random.Random(seed)
    destroy_ops: dict[str, Callable[[list[Sortie], random.Random], tuple[list[Sortie], list[dict]]]] = {
        "random": random_destroy,
        "low_util": low_util_destroy,
        "cluster": cluster_destroy,
        "detour": detour_destroy,
    }
    weights = {name: 1.0 for name in destroy_ops}
    current = list(initial_sorties)
    best = list(initial_sorties)
    temperature = config.ALNS_INITIAL_TEMPERATURE
    start = time.time()
    trace = [ALNSTrace(seed, 0, 0.0, metrics_from_sorties(current).total_time_min, metrics_from_sorties(best).total_time_min)]

    for iteration in range(1, config.ALNS_ITERATIONS + 1):
        op_name = select_operator(weights, rng)
        survivors, removed = destroy_ops[op_name](current, rng)
        candidate = repair_solution(survivors, removed, rng)

        current_time = metrics_from_sorties(current).total_time_min
        candidate_time = metrics_from_sorties(candidate).total_time_min
        accept = False
        reward = 0.0

        if is_better_solution(candidate, best):
            best = candidate
            current = candidate
            accept = True
            reward = config.ALNS_REWARD_GLOBAL
        elif is_better_solution(candidate, current):
            current = candidate
            accept = True
            reward = config.ALNS_REWARD_CURRENT
        else:
            delta = candidate_time - current_time
            prob = math.exp(-max(delta, 0) / max(temperature, config.EPS))
            if rng.random() < prob:
                current = candidate
                accept = True
                reward = config.ALNS_REWARD_ACCEPT

        if accept:
            update_weight(weights, op_name, reward)
        else:
            update_weight(weights, op_name, 0.1)

        temperature *= config.ALNS_COOLING_RATE
        trace.append(
            ALNSTrace(
                seed=seed,
                iteration=iteration,
                elapsed_sec=time.time() - start,
                current_time=metrics_from_sorties(current).total_time_min,
                best_time=metrics_from_sorties(best).total_time_min,
            )
        )
    return best, trace


def validate_solution(sorties: list[Sortie], demands: list[dict]) -> None:
    """独立验证最终方案。"""
    expected = sorted(item["pid"] for item in demands)
    seen = []
    for sortie in sorties:
        if len(sortie.stop_fids) > config.MAX_LANDINGS:
            raise ValueError("存在超过 5 次海上着陆的架次")
        if len(sortie.demands) > config.AIRCRAFT[sortie.aircraft_type]["seats"]:
            raise ValueError("存在超载架次")
        fixed = fixed_airport_of_demands(sortie.demands)
        if fixed == "__CONFLICT__":
            raise ValueError("同架次包含多个固定机场要求")
        if fixed not in (None, sortie.airport):
            raise ValueError("固定机场需求被错误分配")
        recomputed = optimize_demands(sortie.demands)
        if recomputed is None:
            raise ValueError("存在不可行架次")
        if recomputed.time_min != sortie.time_min or recomputed.airport != sortie.airport:
            raise ValueError("架次缓存结果与重新计算不一致")
        seen.extend(item["pid"] for item in sortie.demands)
    if sorted(seen) != expected:
        raise ValueError("人员分配存在遗漏或重复")


def build_output_tables(sorties: list[Sortie]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造题目要求的路线表和分配表。"""
    flight_no = {"T1": 0, "T2": 0, "T3": 0}
    route_rows = []
    assign_rows = []
    for sortie in sorties:
        flight_no[sortie.aircraft_type] += 1
        fn = flight_no[sortie.aircraft_type]
        route_rows.append([sortie.aircraft_type, fn, 0, sortie.airport, 0])
        for order, (fid, refuel) in enumerate(zip(sortie.stop_fids, sortie.refuels), start=1):
            route_rows.append([sortie.aircraft_type, fn, order, fid, int(refuel)])
        route_rows.append([sortie.aircraft_type, fn, len(sortie.stop_fids) + 1, sortie.airport, 0])

        first_stop = {}
        for order, fid in enumerate(sortie.stop_fids, start=1):
            first_stop.setdefault(fid, order)
        for item in sortie.demands:
            assign_rows.append([item["pid"], sortie.aircraft_type, fn, 0, first_stop[item["dest"]]])

    routes_df = pd.DataFrame(route_rows, columns=["aircraft_type", "flight_no", "stop_order", "facility_id", "refuel"])
    assignments_df = pd.DataFrame(
        assign_rows,
        columns=["person_id", "aircraft_type", "flight_no", "pickup_stop_order", "delivery_stop_order"],
    )
    return routes_df, assignments_df


def plot_alns_convergence(convergence_df: pd.DataFrame, path: Path) -> None:
    """绘制 ALNS 收敛曲线。"""
    setup_matplotlib_chinese()
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 4.5))
    for seed, group in convergence_df.groupby("seed"):
        plt.plot(group["iteration"], group["best_time"], linewidth=1.0, alpha=0.55, label=str(seed))
    best_seed = convergence_df.groupby("seed")["best_time"].min().idxmin()
    best_group = convergence_df[convergence_df["seed"] == best_seed]
    plt.plot(best_group["iteration"], best_group["best_time"], color="#d62728", linewidth=2.2, label="最佳种子")
    plt.xlabel("迭代次数")
    plt.ylabel("当前最好总飞机使用时间")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_method_compare(compare_df: pd.DataFrame, path: Path) -> None:
    """绘制方法对比图。"""
    setup_matplotlib_chinese()
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.bar(compare_df["算法"], compare_df["总飞机使用时间"], color=["#9ecae1", "#6baed6", "#3182bd"])
    plt.xlabel("方法")
    plt.ylabel("总飞机使用时间")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_network(distance_df: pd.DataFrame, sorties: list[Sortie], path: Path) -> None:
    """绘制最终航线空间分布。"""
    setup_matplotlib_chinese()
    ensure_dir(path.parent)
    coords = classical_mds(distance_df)
    coord_map = {row.node_id: (row.x, row.y) for row in coords.itertuples(index=False)}
    plt.figure(figsize=(9, 7))
    for node_id, (x, y) in coord_map.items():
        if node_id in config.AIRPORTS:
            plt.scatter(x, y, marker="s", s=90, color="#1f77b4")
            plt.text(x + 1.3, y + 1.3, node_id, fontsize=8)
        elif node_id in config.GAS_SET:
            plt.scatter(x, y, marker="^", s=45, color="#d62728")
        else:
            plt.scatter(x, y, marker="o", s=20, color="#7f7f7f", alpha=0.7)
    for sortie in sorties:
        route = [sortie.airport] + sortie.stop_fids + [sortie.airport]
        xs = [coord_map[node][0] for node in route]
        ys = [coord_map[node][1] for node in route]
        plt.plot(xs, ys, linewidth=0.8, alpha=0.30, color="#2ca02c")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_stop_distribution(sorties: list[Sortie], path: Path) -> None:
    """绘制停靠次数分布。"""
    setup_matplotlib_chinese()
    ensure_dir(path.parent)
    counter = Counter(len(item.stop_fids) for item in sorties)
    xs = [1, 2, 3, 4, 5]
    ys = [counter.get(x, 0) for x in xs]
    plt.figure(figsize=(7, 4.5))
    plt.bar(xs, ys, color="#4c78a8")
    plt.xlabel("海上停靠次数")
    plt.ylabel("架次数量")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_seed_compare(seed_df: pd.DataFrame, path: Path) -> None:
    """绘制多随机种子结果对比。"""
    setup_matplotlib_chinese()
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.plot(seed_df["seed"], seed_df["总飞机使用时间"], marker="o", linewidth=1.6)
    plt.xlabel("随机种子")
    plt.ylabel("总飞机使用时间")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_results(
    distance_path: Path,
    sorties: list[Sortie],
    routes_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    seed_df: pd.DataFrame,
) -> None:
    """保存表格与图像。"""
    ensure_dir(config.OUTPUT_TABLE_DIR)
    ensure_dir(config.OUTPUT_FIGURE_DIR)

    routes_df.to_csv(config.ROUTES_CSV, index=False, encoding="utf-8-sig")
    assignments_df.to_csv(config.ASSIGNMENTS_CSV, index=False, encoding="utf-8-sig")
    save_dataframe_excel(routes_df, config.ROUTES_XLSX, "q1_routes")
    save_dataframe_excel(assignments_df, config.ASSIGNMENTS_XLSX, "q1_assignments")
    save_multi_sheet_excel(
        {
            "summary": summary_df,
            "method_compare": compare_df,
            "seed_compare": seed_df,
            "alns_trace": convergence_df,
        },
        config.SUMMARY_XLSX,
    )

    distance_df = load_distance_matrix(distance_path)
    plot_alns_convergence(convergence_df, config.ALNS_CONVERGENCE_PNG)
    plot_method_compare(compare_df, config.METHOD_COMPARE_PNG)
    plot_network(distance_df, sorties, config.NETWORK_PNG)
    plot_stop_distribution(sorties, config.STOP_COUNT_PNG)
    plot_seed_compare(seed_df, config.SEED_COMPARE_PNG)


def run_solver(distance_path: Path = config.DISTANCE_FILE, demand_path: Path = config.Q1_DEMAND_FILE) -> SolveResult:
    """运行第一问求解器。"""
    clear_internal_caches()
    load_distances(distance_path)
    demands = load_demands(demand_path)
    groups = build_groups(demands)

    savings_solution, compare_rows = build_savings_initial_solution(groups)

    all_traces: list[ALNSTrace] = []
    seed_rows = []
    best_solution = savings_solution
    best_seed = None
    for seed in config.MULTI_SEEDS:
        candidate, trace = run_alns(savings_solution, seed)
        all_traces.extend(trace)
        metrics = metrics_from_sorties(candidate)
        seed_rows.append(
            {
                "seed": seed,
                "总飞机使用时间": metrics.total_time_min,
                "人员总在途时间": metrics.total_intransit_min,
                "总架次数": metrics.total_sorties,
                "总燃油消耗": round(metrics.total_fuel_kg, 3),
                "座位利用率": round(metrics.seat_util, 6),
            }
        )
        if is_better_solution(candidate, best_solution):
            best_solution = candidate
            best_seed = seed

    final_metrics = metrics_from_sorties(best_solution)
    compare_rows.append(
        {
            "算法": "改进Savings+ALNS",
            "总飞机使用时间": final_metrics.total_time_min,
            "人员总在途时间": final_metrics.total_intransit_min,
            "总架次数": final_metrics.total_sorties,
            "总燃油消耗": round(final_metrics.total_fuel_kg, 3),
            "座位利用率": round(final_metrics.seat_util, 6),
        }
    )

    validate_solution(best_solution, demands)
    routes_df, assignments_df = build_output_tables(best_solution)
    summary_df = pd.DataFrame(
        [
            ["总飞机使用时间(分钟)", final_metrics.total_time_min],
            ["人员总在途时间(分钟)", final_metrics.total_intransit_min],
            ["总架次数", final_metrics.total_sorties],
            ["总燃油消耗(kg)", round(final_metrics.total_fuel_kg, 3)],
            ["座位利用率", round(final_metrics.seat_util, 6)],
            ["最佳随机种子", best_seed if best_seed is not None else config.MULTI_SEEDS[0]],
        ],
        columns=["指标", "数值"],
    )
    convergence_df = pd.DataFrame(
        [
            {
                "seed": item.seed,
                "iteration": item.iteration,
                "elapsed_sec": item.elapsed_sec,
                "current_time": item.current_time,
                "best_time": item.best_time,
            }
            for item in all_traces
        ]
    )
    compare_df = pd.DataFrame(compare_rows)
    seed_df = pd.DataFrame(seed_rows)

    save_results(distance_path, best_solution, routes_df, assignments_df, summary_df, convergence_df, compare_df, seed_df)
    return SolveResult(best_solution, routes_df, assignments_df, summary_df, convergence_df, compare_df, seed_df)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="第一问：改进 Savings + ALNS 求解")
    parser.add_argument("--distance", default=str(config.DISTANCE_FILE), help="距离矩阵路径")
    parser.add_argument("--demand", default=str(config.Q1_DEMAND_FILE), help="第一问需求路径")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    result = run_solver(Path(args.distance), Path(args.demand))
    print("第一问求解完成")
    for row in result.summary_df.itertuples(index=False):
        print(f"{row.指标}: {row.数值}")
    print(f"结果表格目录: {config.OUTPUT_TABLE_DIR}")
    print(f"结果图像目录: {config.OUTPUT_FIGURE_DIR}")


if __name__ == "__main__":
    main()
