"""
Q1: 单向出海运输 — 节约算法 (Clarke-Wright Savings)
============================================================
基础解: 每条需求一个架次 (闭环 A -> F -> A)
合并判据: s = T(k1) + T(k2) - T(merged) > 0
目标: 最小化总飞机使用时间 (字典序第一)，其次优化座位利用率、油耗、在途时间

两阶段:
  Stage 1: 按目的设施分组打包 (同目的地合并，机型容量约束)
  Stage 2: Clarke-Wright 节约合并 (跨目的地合并，近邻剪枝)
"""

import csv
import math
import os
import time

# === 配置 ===
DATA_DIR = 'docs/reference_formats'
DIST_PATH = os.path.join(DATA_DIR, 'distances.csv')
DEMAND_PATH = os.path.join(DATA_DIR, 'peopleQ1.csv')
OUT_ROUTES = os.path.join(DATA_DIR, 'q1-routes.csv')
OUT_ASSIGN = os.path.join(DATA_DIR, 'q1-assignments.csv')

AIRPORTS = ['A01', 'A02', 'A03']
GAS_STATIONS = {'F006', 'F011', 'F018', 'F024', 'F031', 'F038', 'F044', 'F050'}
AC = {
    'T1': {'seats': 12, 'speed': 250, 'burn': 3.4, 'tank': 1000, 'reserve': 150},
    'T2': {'seats': 16, 'speed': 220, 'burn': 2.5, 'tank': 1150, 'reserve': 150},
    'T3': {'seats': 19, 'speed': 190, 'burn': 2.9, 'tank': 1600, 'reserve': 200},
}
MAX_LANDINGS = 5
CLOSE_THRESHOLD = 150  # 设施间距离 ≤ 此值才视为合并候选 (km)

D = {}

# === 数据加载 ===
def load_distances():
    global D
    with open(DIST_PATH, encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r)
        cols = header[1:]
        for row in r:
            u = row[0]
            D[u] = {v: int(row[j + 1]) for j, v in enumerate(cols)}


def load_demands():
    out = []
    with open(DEMAND_PATH, encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                'pid': row['person_id'],
                'origin': row['origin_id'],
                'dest': row['destination_id'],
            })
    return out


# === 辅助函数 ===
def flight_min(dist, t):
    """飞行时间（分钟，向上取整）"""
    return math.ceil(60 * dist / AC[t]['speed'])


def check_fuel(airport, stop_fids, t):
    """
    贪心加油可行性: 满油起飞, 途中可在加油站加满。
    返回 (feasible, refuels_list)
    """
    ac = AC[t]
    full, res, burn = ac['tank'], ac['reserve'], ac['burn']
    pts = [airport] + list(stop_fids) + [airport]
    refuels = [False] * len(stop_fids)
    remain = full
    for i in range(len(pts) - 1):
        d = D[pts[i]][pts[i + 1]]
        need = d * burn
        if remain - need < res:
            if i == 0:
                return False, None  # 满油都不够飞第一段
            stop_idx = i - 1
            if stop_fids[stop_idx] in GAS_STATIONS:
                refuels[stop_idx] = True
                remain = full
                if remain - need < res:
                    return False, None
            else:
                return False, None
        remain -= need
    return True, refuels


def route_time(airport, stop_fids, t):
    """计算架次总时间（飞行 + 停靠），返回 (time, refuels) 或 (None, None)"""
    feas, refuels = check_fuel(airport, stop_fids, t)
    if not feas:
        return None, None
    pts = [airport] + list(stop_fids) + [airport]
    flight = sum(flight_min(D[pts[i]][pts[i + 1]], t) for i in range(len(pts) - 1))
    stop = sum(20 if r else 10 for r in refuels)
    return flight + stop, refuels


def best_route_with_gas(airport, deliv_fids, t):
    """
    尝试直达；若不可行，插入一个加油站；仍不可行且位置允许则插入两个。
    返回 (time, stop_fids, refuels) 或 None。
    """
    # 直达
    time, refuels = route_time(airport, deliv_fids, t)
    if time is not None:
        return (time, list(deliv_fids), refuels)

    # 插入一个加油站
    if len(deliv_fids) + 1 <= MAX_LANDINGS:
        for G in GAS_STATIONS:
            if G in deliv_fids:
                continue
            for pos in range(len(deliv_fids) + 1):
                cand = list(deliv_fids[:pos]) + [G] + list(deliv_fids[pos:])
                time, refuels = route_time(airport, cand, t)
                if time is not None:
                    return (time, cand, refuels)

    # 插入两个加油站（仅在交付设施 ≤ 3 时尝试，避免组合爆炸）
    if len(deliv_fids) <= 3 and len(deliv_fids) + 2 <= MAX_LANDINGS:
        gas_list = list(GAS_STATIONS)
        for G1 in gas_list:
            if G1 in deliv_fids:
                continue
            for G2 in gas_list:
                if G2 == G1 or G2 in deliv_fids:
                    continue
                for p1 in range(len(deliv_fids) + 1):
                    cand1 = list(deliv_fids[:p1]) + [G1] + list(deliv_fids[p1:])
                    for p2 in range(len(cand1) + 1):
                        cand2 = list(cand1[:p2]) + [G2] + list(cand1[p2:])
                        time, refuels = route_time(airport, cand2, t)
                        if time is not None:
                            return (time, cand2, refuels)
    return None


def capacity_ok(airport, stop_fids, t, demands):
    """逐航段容量校验: 每段机上人数 ≤ seats"""
    pts = [airport] + list(stop_fids) + [airport]
    seats = AC[t]['seats']
    for i in range(len(pts) - 1):
        remaining = set(pts[i + 1:-1])
        count = sum(1 for d in demands if d['dest'] in remaining)
        if count > seats:
            return False
    return True


def best_sortie(deliv_fids, demands, airport_fixed=None):
    """给定交付设施集合和人员，找最优 (A, t)。返回 (time, A, t, stop_fids, refuels) 或 None。"""
    cand_A = AIRPORTS if airport_fixed is None else [airport_fixed]
    best = None
    for A in cand_A:
        for t in AC:
            if AC[t]['seats'] < len(demands):
                continue
            res = best_route_with_gas(A, deliv_fids, t)
            if res is None:
                continue
            time, stop_fids, refuels = res
            if not capacity_ok(A, stop_fids, t, demands):
                continue
            if best is None or time < best[0]:
                best = (time, A, t, stop_fids, refuels)
    return best


# === Stage 1: 按目的设施打包 ===
def stage1_pack(demands):
    """按目的设施分组，再按起点机场约束分组，每组按机型容量打包。"""
    by_dest = {}
    for d in demands:
        by_dest.setdefault(d['dest'], []).append(d)
    sorties = []
    for F, group in by_dest.items():
        # 按起点机场约束分组: 不同固定机场的需求不能同架次
        sub_groups = {}
        for d in group:
            key = d['origin'] if d['origin'] in AIRPORTS else 'LAND'
            sub_groups.setdefault(key, []).append(d)
        for key, subgroup in sub_groups.items():
            af = key if key != 'LAND' else None
            pack_one_facility(F, subgroup, sorties, af)
    return sorties


def pack_one_facility(F, persons, sorties, airport_fixed):
    """将去往 F 的 persons 打包成若干架次。每架次优先用能装下全部剩余的最小可行机型。"""
    i = 0
    n = len(persons)
    while i < n:
        remaining = n - i
        best = None
        # 优先用能装下全部剩余的最小机型
        for t_name in ['T1', 'T2', 'T3']:
            cap = AC[t_name]['seats']
            if cap < remaining:
                continue
            batch = persons[i:i + remaining]
            res = best_sortie([F], batch, airport_fixed)
            if res is None:
                continue
            if best is None or res[0] < best[0]:
                best = (res[0], batch, res)
        if best is None:
            # 任何机型都装不下 remaining，用 T3 装满一批
            t_name = 'T3'
            cap = AC[t_name]['seats']
            batch = persons[i:i + cap]
            res = best_sortie([F], batch, airport_fixed)
            best = (res[0], batch, res)
        _, batch, res = best
        time, A, t, stop_fids, refuels = res
        sorties.append({
            'airport': A, 'type': t, 'stop_fids': list(stop_fids),
            'refuels': list(refuels), 'demands': batch, 'time': time
        })
        i += len(batch)


# === Stage 2: Clarke-Wright 节约合并 ===
def precompute_close_pairs():
    """预计算设施间近邻对"""
    facs = [f'F{i:03d}' for i in range(1, 53)]
    close = set()
    for i in range(len(facs)):
        for j in range(i + 1, len(facs)):
            if D[facs[i]][facs[j]] <= CLOSE_THRESHOLD:
                close.add((facs[i], facs[j]))
                close.add((facs[j], facs[i]))
    return close


def delivery_order(sortie):
    """返回该架次的交付设施访问顺序（去掉纯加油站停靠）"""
    dests = {d['dest'] for d in sortie['demands']}
    return [s for s in sortie['stop_fids'] if s in dests]


def try_merge(s1, s2):
    """尝试合并两个架次，返回合并后的架次或 None。"""
    A1, A2 = s1['airport'], s2['airport']
    f1 = any(d['origin'] in AIRPORTS for d in s1['demands'])
    f2 = any(d['origin'] in AIRPORTS for d in s2['demands'])
    if f1 and f2 and A1 != A2:
        return None
    if f1:
        cand_A = [A1]
    elif f2:
        cand_A = [A2]
    else:
        cand_A = AIRPORTS

    all_demands = s1['demands'] + s2['demands']
    if len(all_demands) > max(AC[t]['seats'] for t in AC):
        return None

    deliv1 = delivery_order(s1)
    deliv2 = delivery_order(s2)
    # 两种拼接顺序
    o1 = list(deliv1)
    for f in deliv2:
        if f not in o1:
            o1.append(f)
    o2 = list(deliv2)
    for f in deliv1:
        if f not in o2:
            o2.append(f)
    orders = [o1] if o1 == o2 else [o1, o2]

    if len(orders[0]) > MAX_LANDINGS:
        return None

    best = None
    for order in orders:
        for A in cand_A:
            for t in AC:
                if AC[t]['seats'] < len(all_demands):
                    continue
                res = best_route_with_gas(A, order, t)
                if res is None:
                    continue
                time, stop_fids, refuels = res
                if not capacity_ok(A, stop_fids, t, all_demands):
                    continue
                if best is None or time < best['time']:
                    best = {
                        'airport': A, 'type': t, 'stop_fids': list(stop_fids),
                        'refuels': list(refuels), 'demands': all_demands, 'time': time
                    }
    return best


def stage2_merge(sorties, close_pairs):
    """迭代节约合并: 每轮选节约最大的对合并，直到无正节约。"""
    iteration = 0
    while True:
        iteration += 1
        best_save = 0
        best_pair = None
        best_merged = None
        n = len(sorties)
        for i in range(n):
            s1 = sorties[i]
            facs1 = {d['dest'] for d in s1['demands']}
            for j in range(i + 1, n):
                s2 = sorties[j]
                facs2 = {d['dest'] for d in s2['demands']}
                # 近邻剪枝
                if not any((a, b) in close_pairs for a in facs1 for b in facs2):
                    continue
                merged = try_merge(s1, s2)
                if merged is None:
                    continue
                save = s1['time'] + s2['time'] - merged['time']
                if save > best_save:
                    best_save = save
                    best_pair = (i, j)
                    best_merged = merged
        if best_pair is None:
            break
        i, j = best_pair
        sorties = [s for k, s in enumerate(sorties) if k != i and k != j]
        sorties.append(best_merged)
        print(f'  iter {iteration}: merge ({i},{j}) save={best_save}min  #sorties={len(sorties)}')
    return sorties


# === 输出 ===
def write_output(sorties):
    flight_no = {'T1': 0, 'T2': 0, 'T3': 0}
    sortie_fn = []
    for s in sorties:
        flight_no[s['type']] += 1
        sortie_fn.append(flight_no[s['type']])

    with open(OUT_ROUTES, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['aircraft_type', 'flight_no', 'stop_order', 'facility_id', 'refuel'])
        for s_idx, s in enumerate(sorties):
            t = s['type']
            fn = sortie_fn[s_idx]
            w.writerow([t, fn, 0, s['airport'], 0])
            for i, fid in enumerate(s['stop_fids']):
                w.writerow([t, fn, i + 1, fid, 1 if s['refuels'][i] else 0])
            w.writerow([t, fn, len(s['stop_fids']) + 1, s['airport'], 0])

    with open(OUT_ASSIGN, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['person_id', 'aircraft_type', 'flight_no', 'pickup_stop_order', 'delivery_stop_order'])
        for s_idx, s in enumerate(sorties):
            t = s['type']
            fn = sortie_fn[s_idx]
            for d in s['demands']:
                pickup = 0
                delivery = None
                for i, fid in enumerate(s['stop_fids']):
                    if fid == d['dest']:
                        delivery = i + 1
                        break
                w.writerow([d['pid'], t, fn, pickup, delivery])


def compute_stats(sorties):
    total_time = sum(s['time'] for s in sorties)
    total_sorties = len(sorties)
    total_fuel = 0.0
    total_pass_km = 0
    total_avail_km = 0
    total_intransit = 0
    for s in sorties:
        t = s['type']
        ac = AC[t]
        pts = [s['airport']] + s['stop_fids'] + [s['airport']]
        # 各站到达时刻（用于在途时间）
        cur_time = 0
        arrival = {}
        for i in range(len(pts) - 1):
            cur_time += flight_min(D[pts[i]][pts[i + 1]], t)
            arrival[pts[i + 1]] = cur_time
            if i < len(pts) - 2:
                cur_time += 20 if s['refuels'][i] else 10
        # 油耗、客座公里、可用座公里
        for i in range(len(pts) - 1):
            d = D[pts[i]][pts[i + 1]]
            total_fuel += d * ac['burn']
            remaining_dests = set(pts[i + 1:-1])
            n_pax = sum(1 for dd in s['demands'] if dd['dest'] in remaining_dests)
            total_pass_km += n_pax * d
            total_avail_km += ac['seats'] * d
        for d in s['demands']:
            total_intransit += arrival.get(d['dest'], 0)
    seat_util = total_pass_km / total_avail_km if total_avail_km > 0 else 0
    return {
        'total_time_min': total_time,
        'total_sorties': total_sorties,
        'total_fuel_kg': total_fuel,
        'total_intransit_min': total_intransit,
        'seat_util': seat_util,
    }


# === 主程序 ===
def main():
    t0 = time.time()
    print('加载数据...')
    load_distances()
    demands = load_demands()
    print(f'  {len(demands)} 条需求, {len(D)} 个地点')

    print('\nStage 1: 按目的设施打包...')
    sorties = stage1_pack(demands)
    s1 = compute_stats(sorties)
    print(f'  架次数={s1["total_sorties"]}, 总时间={s1["total_time_min"]}min ({s1["total_time_min"]/60:.1f}h), '
          f'油耗={s1["total_fuel_kg"]:.0f}kg, 座位利用率={s1["seat_util"]:.3f}')

    print('\nStage 2: Clarke-Wright 节约合并...')
    close_pairs = precompute_close_pairs()
    print(f'  {len(close_pairs)//2} 对近邻设施 (≤{CLOSE_THRESHOLD}km)')
    sorties = stage2_merge(sorties, close_pairs)

    print('\n计算最终统计...')
    stats = compute_stats(sorties)
    print('=== 最终结果 ===')
    print(f'  总飞机使用时间: {stats["total_time_min"]} min ({stats["total_time_min"]/60:.1f} h)')
    print(f'  人员总在途时间: {stats["total_intransit_min"]} min ({stats["total_intransit_min"]/60:.1f} h)')
    print(f'  总架次数:       {stats["total_sorties"]}')
    print(f'  总燃油消耗量:   {stats["total_fuel_kg"]:.0f} kg')
    print(f'  座位利用率:     {stats["seat_util"]:.4f}')

    # 校验: 每条需求都被分配
    assigned = sum(len(s['demands']) for s in sorties)
    print(f'  分配人员数:     {assigned} / {len(demands)}')

    print('\n写出结果...')
    write_output(sorties)
    print(f'  -> {OUT_ROUTES}')
    print(f'  -> {OUT_ASSIGN}')
    print(f'\n总耗时: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
