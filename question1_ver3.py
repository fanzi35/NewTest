"""
Q1: 单向出海运输 — question1_ver3
==================================
目标采用字典序：
1) 最小化总飞机使用时间；
2) 只有总飞机使用时间相同时，才降低人员总在途时间；
3) 再降低总燃油消耗；
4) 再提高座位利用率。

Q1 的关键特征：
- 仅有“出海”需求，人员均在架次起飞机场登机，在各自目的设施下机；
- LAND 可由算法在 A01/A02/A03 中自动选择；若 origin_id 是具体机场，则必须从该机场出发；
- 一个架次从机场出发，最多 5 次海上着陆，最后返回同一机场；
- 途中可在 8 个指定设施加油，加油时加满；
- 所有人在同一架次内不换乘。

算法：
Stage 1  同一目的设施的精确小规模打包：
         - 允许 LAND 与固定机场人员拼载；
         - 用动态规划选择每个机场上的批量/机型组合；
         - LAND 在三个机场间做枚举分配，主目标按总飞机使用时间比较。
Stage 2  基于“节约”的聚合合并，快速构造高质量初解：
         - 对任意可合并的两个架次重新全局优化机场、机型、交付顺序和加油决策；
         - 使用堆维护候选，不再用固定 150 km 阈值硬剪枝；
         - 合并后只更新与新架次有关的候选，避免每轮 O(n^2) 重扫。

路线优化：
- 对一个候选架次（人数 <= 19、不同交付设施 <= 5），DFS 穷举最多 5 次海上着陆内的
  “下一交付设施 / 必要的加油设施”组合；
- 到达加油设施时显式比较“加油/不加油”，因此可以处理“当前下一段能飞，但为了后续航段
  必须提前加油”的情况；
- 每个完整路线按 (飞机使用时间, 人员在途时间, 燃油, -座位利用率) 严格字典序选优。

Stage 3  可逆二架次大邻域重划分：
         - 同时拆开两条现有架次并重新分配其全部人员；
         - 重新联合优化机场、机型、访问顺序和加油，不受早期贪心合并锁定；
         - 每次只接受完整解在严格字典序上确实更优的替换。

默认数据路径与 ver1 保持一致，也可通过命令行参数覆盖：
python question1_ver3.py --data-dir data/raw
"""

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =========================
# 路径配置
# 输入数据:
#   data/raw/distances.csv
#   data/raw/peopleQ1.csv
#
# 输出结果:
#   docs/reference_formats/q1-routes.csv
#   docs/reference_formats/q1-assignments.csv
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "reference_formats")

DIST_PATH = os.path.join(DATA_DIR, "distances.csv")
DEMAND_PATH = os.path.join(DATA_DIR, "peopleQ1.csv")
OUT_ROUTES = os.path.join(OUTPUT_DIR, "q1-routes.csv")
OUT_ASSIGN = os.path.join(OUTPUT_DIR, "q1-assignments.csv")


# ============================================================
# 常量
# ============================================================

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

# 距离矩阵，load_distances() 后填充
D: Dict[str, Dict[str, float]] = {}


# ============================================================
# 数据结构
# ============================================================

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


# ============================================================
# 基础工具
# ============================================================

def flight_min(dist_km: float, aircraft_type: str) -> int:
    """单航段飞行分钟数，向上取整。"""
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
    """
    a 是否按字典序优于 b：
    严格分层：time -> intransit -> fuel -> -seat_util。
    后一层永远不能补偿前一层的任何恶化。
    """
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
    """用于稳定排序；前四项对应字典序目标，后面仅做确定性 tie-break。"""
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


# ============================================================
# 数据加载与输入校验
# ============================================================

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

    # 基本完整性检查
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


# ============================================================
# 单架次路线优化
# ============================================================

# 缓存：key=(airport, type, dest_signature)
_ROUTE_CACHE: Dict[Tuple[str, str, Tuple[Tuple[str, int], ...]], Optional[RouteTemplate]] = {}
# 缓存：key=(fixed_airport_or_None, dest_signature)
_TEMPLATE_CACHE: Dict[Tuple[Optional[str], Tuple[Tuple[str, int], ...]], Optional[RouteTemplate]] = {}


def _search_route_for_aircraft(
    airport: str,
    aircraft_type: str,
    dest_sig: Tuple[Tuple[str, int], ...],
) -> Optional[RouteTemplate]:
    """
    固定机场、机型和“各目的设施人数”后，在最多 5 次海上着陆内搜索最优路线。

    搜索允许：
    - 任意交付顺序；
    - 额外插入可加油设施；
    - 在作为交付点的加油设施上选择提前加油或不加油；
    - 必要时重访已经服务过的加油设施（重访仍计一次海上着陆）。
    """
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

    # 轻量 dominance：只在“时间严格更早且余油不少”时剪枝。
    # 这样主目标（总飞机使用时间）已经严格占优，不会因为次级指标而误剪。
    frontier: Dict[Tuple[str, Tuple[str, ...], int], List[Tuple[int, float]]] = defaultdict(list)

    def dominated(state_key: Tuple[str, Tuple[str, ...], int], elapsed: int, remain: float) -> bool:
        labels = frontier[state_key]
        for old_t, old_r in labels:
            if old_t < elapsed and old_r + EPS >= remain:
                return True
        # 仅删除被新标签在主目标上严格支配的旧标签
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
            pass_km=pass_km,  # 此时已无人，返场段客座公里为 0
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

        # 最少还要为每个未交付设施各落地一次
        if landings_used + len(unserved) > MAX_LANDINGS:
            return

        # 已找到更短完整路线时，可做一个非常保守的时间下界剪枝
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
            # 即使直返不可行，也可能需要额外落地加油，因此不能 return。

        if landings_used >= MAX_LANDINGS:
            return

        onboard = sum(counts[f] for f in unserved)
        unserved_set = set(unserved)

        # ----------------------------------------------------
        # 1) 下一站选择一个尚未交付的目的设施
        # ----------------------------------------------------
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

            # 非加油设施：固定停靠 10 分钟
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
                # 加油站作为交付点：两种决策都要考虑。
                # 不加油 10 min
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
                # 加油 20 min；允许“提前加油”，不要求下一段立刻油量不足
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

        # ----------------------------------------------------
        # 2) 插入“纯加油”停靠
        # ----------------------------------------------------
        # 必须至少留出 len(unserved) 个着陆位置给真正的交付设施。
        if allow_pure_gas and landings_used + 1 + len(unserved) <= MAX_LANDINGS:
            for g in GAS_STATIONS:
                if g == current:
                    continue
                # 若 g 是尚未交付的目的地，则到达它时必须立即完成交付，
                # 应由上面的“交付设施”分支处理，不能把它当纯加油站跳过交付。
                if g in unserved_set:
                    continue

                dist = D[current][g]
                need = dist * burn
                after = remain - need
                if after < reserve - EPS:
                    continue

                # 纯加油停靠的唯一意义就是加满；若加满后连任何“下一必要目标”都到不了，直接剪掉。
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

    # 第一遍只允许真正的交付设施停靠（交付点若可加油，仍会比较加/不加油）。
    # 若已有可行路线，则额外插入纯加油站只会增加着陆和停靠时间，通常没有必要。
    dfs(**initial_kwargs)

    # 只有交付设施本身无法支撑燃油约束时，才开放额外纯加油停靠。
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
    """给定目的地人数分布和机场约束，联合优化机场、机型、路线与加油。"""
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
    """重新优化一组 Q1 人员能否由一个架次完成，以及该架次的最佳方案。"""
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
    return template_to_sortie(t, demands)


# ============================================================
# Stage 1：同目的地打包 + LAND 机场分配
# ============================================================

_PACK_DP_CACHE: Dict[Tuple[str, str, int], Optional[List[Tuple[int, RouteTemplate]]]] = {}


def best_pack_one_airport(airport: str, dest: str, n_people: int) -> Optional[List[Tuple[int, RouteTemplate]]]:
    """
    将 n_people 名、同一目的地 dest、固定从 airport 出发的人员分成若干架次。
    用 DP 在“人数分批 + 每批最优机型/路线”上最小化字典序目标。
    """
    key = (airport, dest, n_people)
    if key in _PACK_DP_CACHE:
        return _PACK_DP_CACHE[key]
    if n_people == 0:
        _PACK_DP_CACHE[key] = []
        return []

    # 每种批量 p (1..19) 的单架次最优模板
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
    """
    对每个目的设施独立构造初始解。

    与 ver1 的关键区别：
    - 不再把 LAND 与固定机场人员永久拆开；
    - 对 LAND 在三个机场之间进行精确枚举分配；
    - 每个机场内部用 DP 决定分几架、每架载多少人、选什么机型。
    """
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

        # 为每个机场预计算“固定人数 + 0..L 个 LAND”对应的最佳打包方案
        plans: Dict[str, Dict[int, Optional[List[Tuple[int, RouteTemplate]]]]] = {A: {} for A in AIRPORTS}
        for A in AIRPORTS:
            for extra in range(L + 1):
                n = fixed_n[A] + extra
                plans[A][extra] = best_pack_one_airport(A, dest, n)

        best_alloc = None
        best_metrics = None

        # 三机场，枚举 x1+x2+x3=L；O(L^2)，每个设施人数通常很小。
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

        # 按选中的 LAND 分配，绑定真实 person_id 到各模板
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


# ============================================================
# Stage 2：节约合并（堆增量更新）
# ============================================================

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
    """
    若 merged 按字典序优于拆开的 a+b，返回“越大越好”的 improvement tuple；否则 None。
    tuple = (节省时间, 节省在途时间, 节省燃油, 座位利用率提升)
    """
    before = pair_metrics(a, b)
    after = metrics_of_sortie(merged)

    if not lex_better(after, before):
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
    """
    贪心节约合并。

    与 ver1 不同：
    - 不设置固定 CLOSE_THRESHOLD，避免把真正有节约的组合提前剪掉；
    - 初始候选两两计算一次；每次合并后只计算“新架次 vs 其余活动架次”；
    - 主目标优先，只有总飞机使用时间相同才比较在途时间/燃油/座位利用率。
    """
    active: Dict[int, Sortie] = {i: s for i, s in enumerate(sorties)}
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
        merged = try_merge(a, b)
        if merged is None:
            return
        imp = merge_improvement(a, b, merged)
        if imp is None:
            return
        save_t, save_it, save_f, gain_u = imp
        # heapq 为最小堆，因此全部取负；counter 用于稳定排序。
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
            continue  # stale

        # 活动对象从未原地修改，因此该候选仍然有效。
        a, b = active[i], active[j]
        imp = merge_improvement(a, b, merged)
        if imp is None:
            continue

        del active[i]
        del active[j]
        new_id = next_id
        next_id += 1
        active[new_id] = merged
        merges += 1

        save_t, save_it, save_f, gain_u = imp
        print(
            f"  merge {merges:3d}: save_time={save_t:4d} min, "
            f"save_fuel={save_f:8.1f} kg, util_delta={gain_u:+.4f}, "
            f"#sorties={len(active)}"
        )

        # 只更新新架次与其他活动架次之间的候选
        for other_id in list(active):
            if other_id != new_id:
                push_candidate(new_id, other_id)

    # 为输出稳定，先按机场、机型、首个目的地、人数排序
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


# ============================================================
# Stage 3：可逆二架次大邻域搜索
# ============================================================

def _replace_pair_metrics(
    total: AggregateMetrics,
    old_a: Sortie,
    old_b: Sortie,
    replacement: Sequence[Sortie],
) -> AggregateMetrics:
    """在不使用加权和的情况下计算替换后的完整字典序指标。"""
    old = pair_metrics(old_a, old_b)
    new = aggregate_sorties(replacement)
    return AggregateMetrics(
        time_min=total.time_min - old.time_min + new.time_min,
        fuel_kg=total.fuel_kg - old.fuel_kg + new.fuel_kg,
        pass_km=total.pass_km - old.pass_km + new.pass_km,
        avail_km=total.avail_km - old.avail_km + new.avail_km,
        intransit_min=total.intransit_min - old.intransit_min + new.intransit_min,
    )


def _demand_classes(demands: Sequence[dict]) -> List[List[dict]]:
    """同 origin、destination 的人员在 Q1 中对路线可行性完全等价。"""
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for d in demands:
        groups[(d["origin"], d["dest"])].append(d)
    return [groups[k] for k in sorted(groups)]


def best_two_sortie_repartition(
    a: Sortie,
    b: Sortie,
    split_limit: int,
) -> List[Sortie]:
    """
    对两条架次的全部人员做可逆重划分。

    与贪心 merge 不同，这里允许两条旧路线先被完全拆开，再把人员交换后
    重新联合优化机场、机型、访问顺序与加油。若组合数不超过 split_limit，
    该二架次邻域被完整枚举；更大邻域采用确定性的受控枚举。
    """
    all_demands = list(a.demands) + list(b.demands)
    before = [a, b]
    best = before
    best_m = aggregate_sorties(before)

    merged = optimize_demands(all_demands)
    if merged is not None:
        merged_m = metrics_of_sortie(merged)
        if lex_better(merged_m, best_m):
            best, best_m = [merged], merged_m

    classes = _demand_classes(all_demands)
    counts = [len(g) for g in classes]
    allocation = [0] * len(classes)
    evaluated = 0

    def evaluate_split() -> None:
        nonlocal best, best_m, evaluated
        if evaluated >= split_limit:
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
        cand_m = aggregate_sorties(cand)
        if lex_better(cand_m, best_m):
            best, best_m = cand, cand_m

    def dfs(k: int, left_size: int) -> None:
        if evaluated >= split_limit:
            return
        if k == len(classes):
            evaluate_split()
            return
        remaining_after = sum(counts[k + 1:])
        c = counts[k]
        lo = max(0, len(all_demands) - MAX_SEATS - left_size - remaining_after)
        hi = min(c, MAX_SEATS - left_size)
        # 中间分配优先，尽早覆盖交换而非只覆盖整类搬移。
        values = list(range(lo, hi + 1))
        values.sort(key=lambda x: (abs(x - c / 2.0), x))
        for x in values:
            allocation[k] = x
            dfs(k + 1, left_size + x)
            if evaluated >= split_limit:
                break

    dfs(0, 0)
    return best


def _pair_search_priority(a: Sortie, b: Sortie) -> Tuple[float, int, str, str]:
    """只用于安排计算顺序，不作为可行性硬剪枝。"""
    proximity = min(D[x][y] for x in set(a.stop_fids) for y in set(b.stop_fids))
    return (
        proximity,
        -(a.time_min + b.time_min),
        min(d["pid"] for d in a.demands),
        min(d["pid"] for d in b.demands),
    )


def stage3_large_neighborhood(
    sorties: Sequence[Sortie],
    max_moves: int = 30,
    pair_budget: int = 250,
    split_limit: int = 20000,
) -> List[Sortie]:
    """
    变量邻域下降：每轮检查一批最有希望的二架次邻域并采用全局最佳改进。

    接受判据始终是完整解的严格字典序，因此绝不允许用在途时间、燃油或
    利用率换取哪怕 1 分钟的总飞机使用时间。邻域操作可拆分旧合并，克服
    Stage 2 只能单向合并的核心缺陷。
    """
    current = list(sorties)
    total = aggregate_sorties(current)
    moves = 0

    while moves < max_moves:
        pairs = []
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                # 两条路线合计最多 38 人；固定机场冲突仍可能通过重新拆分解决。
                pairs.append((_pair_search_priority(current[i], current[j]), i, j))
        pairs.sort(key=lambda x: x[0])

        best_choice = None
        best_total = total
        for _, i, j in pairs[:pair_budget]:
            replacement = best_two_sortie_repartition(
                current[i], current[j], split_limit=split_limit,
            )
            cand_total = _replace_pair_metrics(
                total, current[i], current[j], replacement,
            )
            if lex_better(cand_total, best_total):
                best_total = cand_total
                best_choice = (i, j, replacement)

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
    return current


# ============================================================
# 结果统计与完整校验
# ============================================================

def simulate_sortie(sortie: Sortie) -> RouteTemplate:
    """按最终输出路线重新模拟，独立校验时间/燃油/客座公里/在途时间。"""
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

    # 返场段
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
        # 校验存储指标与重新模拟结果一致
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


# ============================================================
# CSV 输出
# ============================================================

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
                # Q1：所有人都在起飞机场上机，因此 pickup_stop_order 恒为 0。
                delivery = first_stop[d["dest"]]
                w.writerow([d["pid"], t, fn, 0, delivery])


# ============================================================
# 命令行与主程序
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q1 严格字典序大邻域优化 question1_ver3")
    p.add_argument("--data-dir", default=DATA_DIR, help="输入 CSV 目录")
    p.add_argument("--dist", default=None, help="distances.csv 路径；默认 data-dir/distances.csv")
    p.add_argument("--demand", default=None, help="peopleQ1.csv 路径；默认 data-dir/peopleQ1.csv")
    p.add_argument("--routes-out", default=OUT_ROUTES)
    p.add_argument("--assign-out", default=OUT_ASSIGN)
    p.add_argument("--lns-moves", type=int, default=30, help="最多接受的大邻域改进次数")
    p.add_argument("--pair-budget", type=int, default=250, help="每轮精查的二架次邻域数")
    p.add_argument("--split-limit", type=int, default=20000, help="每个二架次邻域的最大划分枚举数")
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
    args = parse_args()
    dist_path = args.dist or os.path.join(args.data_dir, "distances.csv")
    demand_path = args.demand or os.path.join(args.data_dir, "peopleQ1.csv")
    routes_out = args.routes_out or os.path.join(args.data_dir, "q1-routes.csv")
    assign_out = args.assign_out or os.path.join(args.data_dir, "q1-assignments.csv")

    t0 = time.time()
    print("加载数据...")
    load_distances(dist_path)
    demands = load_demands(demand_path)
    print(f"  {len(demands)} 条 Q1 出海需求，{len(D)} 个地点")

    print("\nStage 1: 同目的地打包 + LAND 自动分配机场...")
    sorties = stage1_pack(demands)
    validate_solution(sorties, demands)
    print_stats("Stage 1 结果:", sorties)

    print("\nStage 2: 节约合并 + 架次内部全局重优化...")
    sorties = stage2_merge(sorties)
    validate_solution(sorties, demands)
    print_stats("Stage 2 结果:", sorties)

    print("\nStage 3: 可逆二架次大邻域重划分...")
    sorties = stage3_large_neighborhood(
        sorties,
        max_moves=max(0, args.lns_moves),
        pair_budget=max(1, args.pair_budget),
        split_limit=max(1, args.split_limit),
    )
    validate_solution(sorties, demands)
    print_stats("\n=== 最终结果 ===", sorties)

    print("\n写出结果...")
    write_output(sorties, routes_out, assign_out)
    print(f"  -> {routes_out}")
    print(f"  -> {assign_out}")
    print(f"\n总运行时间: {time.time() - t0:.2f} s")
    print(f"路线缓存: {len(_ROUTE_CACHE)}，模板缓存: {len(_TEMPLATE_CACHE)}")


if __name__ == "__main__":
    main()
