#!/usr/bin/env python3
"""
02b_extract_steps.py — пошаговый датасет (state, action) пар из replay-логов.

Каждая строка = один ход одного игрока.

═══════════════════════════════════════════════════════════════════
СОСТОЯНИЕ (что видел игрок перед ходом):
═══════════════════════════════════════════════════════════════════

  Фаза игры:
    step, phase              — абс. шаг и нормализованный [0..1]
    n_planets                — всего планет на карте

  Баланс сил (resource ratio):
    own_ship_ratio           — наши корабли / все корабли на планетах
    own_prod_ratio           — наше производство / всё производство
    own_planet_ratio         — наши планеты / все планеты
    ship_gap                 — own_ships - enemy_ships (абс. разница)
    prod_gap                 — own_prod - enemy_prod

  Детали своих планет:
    own_ships_total          — сумма кораблей на наших планетах
    own_prod_total           — суммарное производство
    n_own_planets            — количество наших планет
    mean_own_ships           — среднее кораблей на планете
    max_own_ships            — максимум (самая большая планета)
    min_own_ships            — минимум (самая слабая планета)
    std_own_ships            — разброс (насколько неравномерно накоплено)

  Враг и нейтралы:
    n_neutral_planets        — нейтральных планет осталось
    n_enemy_planets          — вражеских планет
    enemy_ships_total        — все корабли врага на планетах
    mean_neutral_ships       — среднее кораблей у нейтралов (насколько дорого захватить)
    min_neutral_ships        — самый дешёвый нейтрал (лёгкий захват)

  Флоты в воздухе:
    own_fleets_count         — наших флотов летит
    own_fleets_ships         — суммарно кораблей в наших флотах
    enemy_fleets_count       — вражеских флотов летит
    enemy_fleets_ships       — суммарно кораблей во вражеских флотах
    fleet_balance            — own_fleets_ships - enemy_fleets_ships

  Проецированное состояние (после приземления всех флотов):
    proj_own_ship_ratio      — проецированная доля наших кораблей
    proj_own_planet_ratio    — проецированная доля наших планет
    proj_own_ships           — проецированное кол-во наших кораблей на планетах
    proj_ship_ratio_delta    — улучшение ship_ratio (proj - current)
    proj_planet_ratio_delta  — улучшение planet_ratio (proj - current)
    own_under_threat_count   — наших планет с нетто угрозой захвата
    own_under_threat_ships   — кораблей под угрозой потери
    enemy_contested_count    — вражеских планет куда мы уже летим
    neutral_contested_count  — нейтральных планет куда мы уже летим
    incoming_enemy_to_own    — враги летят на наши планеты (всего кораблей)
    incoming_own_reinforce   — наши подкрепления летят на наши планеты
    net_incoming_own         — incoming_own_reinforce - incoming_enemy_to_own
    closest_threat_dist      — норм. дистанция ближайшего вражеского флота к нам

═══════════════════════════════════════════════════════════════════
ДЕЙСТВИЕ (что игрок сделал):
═══════════════════════════════════════════════════════════════════

  Общее:
    did_act                  — 1 если хоть что-то запустил
    n_launches               — всего флотов запущено
    ships_sent_total         — всего кораблей отправлено
    ships_fraction_sent      — ships_sent_total / own_ships_total

  По типу цели:
    n_attack_enemy           — атаки на вражеские планеты
    n_attack_neutral         — захваты нейтральных
    n_reinforce_own          — подкрепления своих планет (!)
    attack_enemy_ratio       — n_attack_enemy / n_launches
    attack_neutral_ratio     — n_attack_neutral / n_launches
    reinforce_ratio          — n_reinforce_own / n_launches

  Характеристики атак (только реальные атаки, без reinforce):
    avg_overkill             — среднее ships_sent / target_ships
    avg_target_dist_frac     — средняя дистанция до цели (норм. 0..1)
    avg_target_ships         — среднее кораблей у атакованных планет
    avg_target_prod          — среднее производство атакованных планет
    max_single_launch        — максимальный одиночный флот (агрессивность пика)
    multi_launch             — 1 если >1 флота за ход

  Финансовая эффективность (только для enemy/neutral):
    ships_per_prod_gained    — ships_sent / sum(target_prod) — «цена за производство»

═══════════════════════════════════════════════════════════════════

Запуск:
  python3 opponent_learning/scripts/02b_extract_steps.py [--limit N]
"""

import json
import math
import argparse
import warnings
from pathlib import Path
from statistics import stdev

import pandas as pd

warnings.filterwarnings('ignore')

HERE     = Path(__file__).parent
RAW_DIR  = HERE.parent / "data" / "raw"
PROC_DIR = HERE.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

NEUTRAL_OWNER = -1


# ── Planet / fleet parsers ───────────────────────────────────────────────────

def _parse_planet(p):
    if isinstance(p, list) and len(p) >= 7:
        return {
            'id':         int(p[0]),
            'owner':      int(p[1]) if p[1] is not None else NEUTRAL_OWNER,
            'x':          float(p[2]),
            'y':          float(p[3]),
            'radius':     float(p[4]),
            'ships':      float(p[5]),
            'production': float(p[6]),
        }
    if isinstance(p, dict):
        return p
    return None


def _parse_fleet(f):
    """[id, owner, x, y, angle, ships, source_id]
    Сохраняем позицию и угол чтобы найти цель и оценить расстояние.
    """
    if isinstance(f, list) and len(f) >= 6:
        return {
            'id':    int(f[0]),
            'owner': int(f[1]),
            'x':     float(f[2]),
            'y':     float(f[3]),
            'angle': float(f[4]),
            'ships': float(f[5]),
            'src_id': int(f[6]) if len(f) >= 7 else -1,
        }
    if isinstance(f, dict):
        return f
    return None


def _planet_map(raw_planets):
    pm = {}
    for p in raw_planets:
        pd_ = _parse_planet(p)
        if pd_ is not None:
            pm[pd_['id']] = pd_
    return pm


def _max_dist(pm):
    coords = [(p['x'], p['y']) for p in pm.values()]
    md = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            md = max(md, d)
    return max(md, 1.0)


def _find_target(src, angle, pm):
    """Определяет планету-цель по углу запуска."""
    sx, sy   = src['x'], src['y']
    dx_aim   = math.cos(angle)
    dy_aim   = math.sin(angle)
    best, bs = None, float('inf')
    for tgt in pm.values():
        if tgt['id'] == src['id']:
            continue
        tdx = tgt['x'] - sx
        tdy = tgt['y'] - sy
        dist = math.hypot(tdx, tdy)
        if dist < 1e-6:
            continue
        dot = (dx_aim * tdx + dy_aim * tdy) / dist
        if dot < 0.2:
            continue
        t_angle  = math.atan2(tdy, tdx)
        ang_diff = abs(math.atan2(math.sin(angle - t_angle), math.cos(angle - t_angle)))
        if ang_diff < bs:
            bs   = ang_diff
            best = tgt
    return best


# ── Fleet destination inference ─────────────────────────────────────────────

def _find_fleet_dest(fleet, pm):
    """Находит планету-цель летящего флота по его позиции и углу.

    Аналогично _find_target, но источник — позиция флота, не планеты.
    Дополнительно: возвращает расстояние до цели чтобы оценить
    сколько ходов до приземления (для приоритизации угроз).
    """
    fx, fy   = fleet['x'], fleet['y']
    angle    = fleet['angle']
    dx_aim   = math.cos(angle)
    dy_aim   = math.sin(angle)

    best, bs, best_dist = None, float('inf'), 0.0
    for tgt in pm.values():
        tdx  = tgt['x'] - fx
        tdy  = tgt['y'] - fy
        dist = math.hypot(tdx, tdy)
        if dist < 1e-6:
            continue
        dot = (dx_aim * tdx + dy_aim * tdy) / dist
        if dot < 0.15:   # флот летит в сторону цели
            continue
        t_angle  = math.atan2(tdy, tdx)
        ang_diff = abs(math.atan2(math.sin(angle - t_angle), math.cos(angle - t_angle)))
        if ang_diff < bs:
            bs        = ang_diff
            best      = tgt
            best_dist = dist
    return best, best_dist


def _compute_projected(pm, fleets, pidx, max_d):
    """Вычисляет проецированное состояние доски после приземления всех флотов.

    Алгоритм (упрощённый — без точного порядка прибытия, без производства во время полёта):
      1. Для каждого флота определяем цель (angle-matching).
      2. Аккумулируем incoming_own[planet_id] и incoming_enemy[planet_id].
      3. Проецируем: proj_ships = current_ships + incoming_own - incoming_enemy
         - если proj_ships > 0 → планета остаётся/переходит к owner
         - если proj_ships <= 0 → «угроза» для текущего owner

    Возвращает dict с проецированными фичами.
    """
    # incoming флоты по планете-цели
    inc_own   = {}   # planet_id → суммарные свои корабли летящие туда
    inc_enemy = {}   # planet_id → суммарные вражеские корабли летящие туда
    # для own planet: угроза = incoming_enemy > (current_ships + incoming_own)
    # для enemy/neutral: шанс захвата = incoming_own > current_ships

    fleet_dest_resolved = 0
    for fl in fleets:
        dest, dist = _find_fleet_dest(fl, pm)
        if dest is None:
            continue
        fleet_dest_resolved += 1
        pid = dest['id']
        if fl['owner'] == pidx:
            inc_own[pid]   = inc_own.get(pid, 0)   + fl['ships']
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + fl['ships']

    # Проецируем каждую планету
    proj_own_ships  = 0.0
    proj_own_count  = 0
    proj_all_ships  = 0.0
    proj_all_count  = 0

    own_threatened     = 0    # наших планет с нетто-угрозой
    own_threatened_ships = 0.0  # сколько кораблей под угрозой
    enemy_contested    = 0    # вражеских планет куда мы летим
    neutral_contested  = 0    # нейтральных планет куда мы летим
    incoming_enemy_to_own = 0.0  # враги летят на наши
    incoming_own_to_own   = 0.0  # подкрепления на наши

    for pid, p in pm.items():
        cur_ships = max(0, p['ships'])
        o_in  = inc_own.get(pid, 0)
        e_in  = inc_enemy.get(pid, 0)

        if p['owner'] == pidx:
            # наша планета: net_defense = current + own_reinforcements - enemy_attack
            net = cur_ships + o_in - e_in
            incoming_own_to_own   += o_in
            incoming_enemy_to_own += e_in

            if net > 0:
                proj_own_ships += net
                proj_own_count += 1
            else:
                # под угрозой потери
                own_threatened += 1
                own_threatened_ships += cur_ships   # что потенциально потеряем

            proj_all_ships += max(0, net)
            proj_all_count += 1

        elif p['owner'] == NEUTRAL_OWNER:
            if o_in > 0:
                neutral_contested += 1
            # нейтрал останется нейтральным если никто не летит
            # иначе достанется тому кто больше отправил (упрощённо)
            if o_in > e_in and o_in > cur_ships:
                # скорее всего захватим
                net_ships = o_in - cur_ships - e_in
                proj_own_ships += max(0, net_ships)
                proj_own_count += 1
                proj_all_ships += max(0, net_ships)
            else:
                proj_all_ships += cur_ships
            proj_all_count += 1

        else:
            # вражеская планета
            if o_in > 0:
                enemy_contested += 1
            net = cur_ships + e_in - o_in   # с точки зрения врага
            # с нашей точки зрения: если o_in > cur + e_in → захватим
            if o_in > cur_ships + e_in:
                cap_ships = o_in - cur_ships - e_in
                proj_own_ships += cap_ships
                proj_own_count += 1
                proj_all_ships += cap_ships
            else:
                proj_all_ships += max(0, net)
            proj_all_count += 1

    # Угрозы: ближайший вражеский флот к нашим планетам (расстояние → срочность)
    closest_threat_dist = 1.0   # норм. 0..1, 0=уже почти прилетел
    for fl in fleets:
        if fl['owner'] == pidx:
            continue
        dest, dist = _find_fleet_dest(fl, pm)
        if dest is None:
            continue
        if dest['owner'] == pidx:
            nd = dist / max_d
            if nd < closest_threat_dist:
                closest_threat_dist = nd

    return {
        'proj_own_ship_ratio':      round(proj_own_ships / max(1, proj_all_ships), 4),
        'proj_own_planet_ratio':    round(proj_own_count / max(1, proj_all_count), 4),
        'proj_own_ships':           round(proj_own_ships, 1),
        'own_under_threat_count':   own_threatened,
        'own_under_threat_ships':   round(own_threatened_ships, 1),
        'enemy_contested_count':    enemy_contested,
        'neutral_contested_count':  neutral_contested,
        'incoming_enemy_to_own':    round(incoming_enemy_to_own, 1),
        'incoming_own_reinforce':   round(incoming_own_to_own, 1),
        'net_incoming_own':         round(incoming_own_to_own - incoming_enemy_to_own, 1),
        'closest_threat_dist':      round(closest_threat_dist, 4),
        'fleet_dest_resolved':      fleet_dest_resolved,
    }


# ── Per-step extraction ──────────────────────────────────────────────────────

def extract_steps(episode_id: int, data: dict) -> list:
    steps = data.get('steps', [])
    if len(steps) < 2:
        return []

    n_steps   = len(steps)
    n_players = len(steps[0])
    if n_players < 2:
        return []

    final_rewards = {}
    for pidx in range(n_players):
        try:
            final_rewards[pidx] = float(steps[-1][pidx].get('reward') or 0)
        except Exception:
            final_rewards[pidx] = 0.0

    # max_dist из первых нескольких шагов
    max_d = 1.0
    for step in steps[:5]:
        for pd_ in step:
            if isinstance(pd_, dict):
                obs = pd_.get('observation', {})
                pl  = obs.get('planets', []) if isinstance(obs, dict) else []
                pm  = _planet_map(pl)
                if pm:
                    max_d = _max_dist(pm)
                    break
        if max_d > 1.0:
            break

    rows = []

    for step_idx, step in enumerate(steps):
        if not isinstance(step, list) or not step:
            continue

        # Observation (берём из первого игрока — одинакова у обоих)
        obs0        = step[0].get('observation', {}) if isinstance(step[0], dict) else {}
        raw_planets = obs0.get('planets', []) if isinstance(obs0, dict) else []
        raw_fleets  = obs0.get('fleets',  []) if isinstance(obs0, dict) else []

        pm     = _planet_map(raw_planets)
        fleets = [f for f in [_parse_fleet(x) for x in raw_fleets] if f]

        if not pm:
            continue

        # ── Глобальные фичи состояния ─────────────────────────────────────
        n_planets    = len(pm)
        all_ships    = sum(max(0, p['ships'])      for p in pm.values())
        all_prod     = sum(max(0, p['production']) for p in pm.values())
        phase        = step_idx / max(1, n_steps - 1)

        neutral_pl  = [p for p in pm.values() if p['owner'] == NEUTRAL_OWNER]
        n_neutral   = len(neutral_pl)
        mean_neutral_ships = (sum(p['ships'] for p in neutral_pl) / n_neutral
                              if neutral_pl else 0.0)
        min_neutral_ships  = (min(p['ships'] for p in neutral_pl)
                              if neutral_pl else 0.0)

        for pidx in range(n_players):
            if pidx >= len(step) or not isinstance(step[pidx], dict):
                continue

            player_data = step[pidx]

            # ── Своё состояние ────────────────────────────────────────────
            own_pl   = [p for p in pm.values() if p['owner'] == pidx]
            enemy_pl = [p for p in pm.values()
                        if p['owner'] != pidx and p['owner'] != NEUTRAL_OWNER]

            n_own      = len(own_pl)
            own_ships_list = [max(0, p['ships']) for p in own_pl]
            own_ships  = sum(own_ships_list)
            own_prod   = sum(max(0, p['production']) for p in own_pl)

            enemy_ships = sum(max(0, p['ships']) for p in enemy_pl)
            enemy_prod  = sum(max(0, p['production']) for p in enemy_pl)

            # Флоты
            own_fl     = [f for f in fleets if f['owner'] == pidx]
            enemy_fl   = [f for f in fleets
                          if f['owner'] != pidx and f['owner'] != NEUTRAL_OWNER]
            own_fl_ships   = sum(f['ships'] for f in own_fl)
            enemy_fl_ships = sum(f['ships'] for f in enemy_fl)

            # ── Projected state ──────────────────────────────────────────
            proj = _compute_projected(pm, fleets, pidx, max_d)

            # ── Действие ─────────────────────────────────────────────────
            action_raw = player_data.get('action', [])
            launches   = []
            if isinstance(action_raw, list):
                for a in action_raw:
                    if isinstance(a, (list, tuple)) and len(a) >= 3:
                        try:
                            launches.append({
                                'source_id': int(a[0]),
                                'angle':     float(a[1]),
                                'ships':     int(a[2]),
                            })
                        except (TypeError, ValueError):
                            pass

            # Классифицируем каждый запуск
            n_attack_enemy   = 0
            n_attack_neutral = 0
            n_reinforce_own  = 0
            n_unknown        = 0

            ships_sent = 0
            overkill_list   = []
            dist_list       = []
            target_ships_l  = []
            target_prod_l   = []
            launch_sizes    = []

            for l in launches:
                ships_sent += l['ships']
                launch_sizes.append(l['ships'])

                src = pm.get(l['source_id'])
                if src is None:
                    n_unknown += 1
                    continue

                tgt = _find_target(src, l['angle'], pm)
                if tgt is None:
                    n_unknown += 1
                    continue

                tgt_owner = tgt['owner']
                tgt_ships = max(1.0, tgt['ships'])
                tgt_prod  = tgt['production']

                if tgt_owner == pidx:
                    # подкрепление своей планеты
                    n_reinforce_own += 1
                elif tgt_owner == NEUTRAL_OWNER:
                    n_attack_neutral += 1
                    d = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
                    dist_list.append(d / max_d)
                    overkill_list.append(l['ships'] / tgt_ships)
                    target_ships_l.append(tgt_ships)
                    target_prod_l.append(tgt_prod)
                else:
                    n_attack_enemy += 1
                    d = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
                    dist_list.append(d / max_d)
                    overkill_list.append(l['ships'] / tgt_ships)
                    target_ships_l.append(tgt_ships)
                    target_prod_l.append(tgt_prod)

            n_real_attacks = n_attack_enemy + n_attack_neutral
            did_act        = int(len(launches) > 0)
            n_launches     = len(launches)

            row = {
                # Мета
                'episode_id':   episode_id,
                'player_idx':   pidx,
                'step':         step_idx,
                'reward':       final_rewards.get(pidx, 0),

                # Фаза
                'phase':         round(phase, 4),
                'n_planets':     n_planets,

                # Баланс сил
                'own_ship_ratio':    round(own_ships   / max(1, all_ships), 4),
                'own_prod_ratio':    round(own_prod    / max(1, all_prod),  4),
                'own_planet_ratio':  round(n_own       / max(1, n_planets), 4),
                'ship_gap':          round(own_ships   - enemy_ships, 1),
                'prod_gap':          round(own_prod    - enemy_prod,  1),

                # Своих
                'own_ships_total':   round(own_ships, 1),
                'own_prod_total':    round(own_prod, 1),
                'n_own_planets':     n_own,
                'mean_own_ships':    round(own_ships / max(1, n_own), 2),
                'max_own_ships':     round(max(own_ships_list, default=0), 1),
                'min_own_ships':     round(min(own_ships_list, default=0), 1),
                'std_own_ships':     round(stdev(own_ships_list) if len(own_ships_list) > 1 else 0, 2),

                # Враги/нейтралы
                'n_neutral_planets':  n_neutral,
                'n_enemy_planets':    len(enemy_pl),
                'enemy_ships_total':  round(enemy_ships, 1),
                'mean_neutral_ships': round(mean_neutral_ships, 2),
                'min_neutral_ships':  round(min_neutral_ships, 2),

                # Флоты
                'own_fleets_count':    len(own_fl),
                'own_fleets_ships':    round(own_fl_ships, 1),
                'enemy_fleets_count':  len(enemy_fl),
                'enemy_fleets_ships':  round(enemy_fl_ships, 1),
                'fleet_balance':       round(own_fl_ships - enemy_fl_ships, 1),

                # Действие — общее
                'did_act':             did_act,
                'n_launches':          n_launches,
                'ships_sent_total':    ships_sent,
                'ships_fraction_sent': round(ships_sent / max(1, own_ships), 4) if own_ships else 0,

                # Действие — по типу цели
                'n_attack_enemy':      n_attack_enemy,
                'n_attack_neutral':    n_attack_neutral,
                'n_reinforce_own':     n_reinforce_own,
                'n_unknown_target':    n_unknown,
                'attack_enemy_ratio':  round(n_attack_enemy   / max(1, n_launches), 4) if n_launches else 0,
                'attack_neutral_ratio':round(n_attack_neutral / max(1, n_launches), 4) if n_launches else 0,
                'reinforce_ratio':     round(n_reinforce_own  / max(1, n_launches), 4) if n_launches else 0,

                # Характеристики атак
                'avg_overkill':         round(sum(overkill_list)  / len(overkill_list),  3) if overkill_list  else 0,
                'avg_target_dist_frac': round(sum(dist_list)      / len(dist_list),      4) if dist_list      else 0,
                'avg_target_ships':     round(sum(target_ships_l) / len(target_ships_l), 2) if target_ships_l else 0,
                'avg_target_prod':      round(sum(target_prod_l)  / len(target_prod_l),  3) if target_prod_l  else 0,
                'max_single_launch':    max(launch_sizes, default=0),
                'multi_launch':         int(n_launches > 1),

                # Эффективность
                'ships_per_prod_gained': round(
                    ships_sent / max(1, sum(target_prod_l)), 2
                ) if target_prod_l else 0,

                # ── Projected state (после приземления всех флотов) ──────
                # Разница текущего и проецированного баланса сил
                'proj_own_ship_ratio':      proj['proj_own_ship_ratio'],
                'proj_own_planet_ratio':    proj['proj_own_planet_ratio'],
                'proj_own_ships':           proj['proj_own_ships'],
                # delta: насколько позиция улучшается/ухудшается
                'proj_ship_ratio_delta':    round(
                    proj['proj_own_ship_ratio'] -
                    round(own_ships / max(1, all_ships), 4), 4),
                'proj_planet_ratio_delta':  round(
                    proj['proj_own_planet_ratio'] -
                    round(n_own / max(1, n_planets), 4), 4),
                # угрозы нашим планетам
                'own_under_threat_count':   proj['own_under_threat_count'],
                'own_under_threat_ships':   proj['own_under_threat_ships'],
                # наши атаки в воздухе
                'enemy_contested_count':    proj['enemy_contested_count'],
                'neutral_contested_count':  proj['neutral_contested_count'],
                # чистый поток кораблей к нашим планетам
                'incoming_enemy_to_own':    proj['incoming_enemy_to_own'],
                'incoming_own_reinforce':   proj['incoming_own_reinforce'],
                'net_incoming_own':         proj['net_incoming_own'],
                # расстояние до ближайшей угрозы (0=сейчас прилетает, 1=далеко)
                'closest_threat_dist':      proj['closest_threat_dist'],
            }
            rows.append(row)

    return rows


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',   default=str(RAW_DIR))
    parser.add_argument('--out',   default=str(PROC_DIR / 'steps.csv'))
    parser.add_argument('--limit', type=int, default=0,
                        help='Обработать только первые N файлов (0=все)')
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    files   = sorted(raw_dir.glob('[0-9]*.json'))
    if args.limit:
        files = files[:args.limit]

    print(f"Найдено {len(files)} replay-файлов  ({len(files)*300//1000}k строк ожидаем)")

    all_rows, errors = [], 0
    for i, fpath in enumerate(files, 1):
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
            rows = extract_steps(int(fpath.stem), data)
            all_rows.extend(rows)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠ {fpath.name}: {e}")
        if i % 100 == 0 or i == len(files):
            print(f"  [{i:>4}/{len(files)}]  строк: {len(all_rows):>7,}  ошибок: {errors}")

    if not all_rows:
        print("Нет данных.")
        return

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    print(f"\n✓ {out_path}")
    print(f"  Строк: {len(df):,}  Эпизодов: {df['episode_id'].nunique()}  "
          f"Ошибок: {errors}  Колонок: {len(df.columns)}")
    print(f"  Размер файла: {out_path.stat().st_size // 1024 / 1024:.1f} MB")

    # ── Быстрый анализ по фазам ──────────────────────────────────────────────
    df['phase_bin'] = pd.cut(df['phase'], bins=5,
                              labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])

    print("\n── Активность по фазам игры ────────────────────────────────")
    print(df.groupby('phase_bin', observed=True).agg(
        attack_rate   =('did_act',              'mean'),
        neutral_ratio =('attack_neutral_ratio', 'mean'),
        enemy_ratio   =('attack_enemy_ratio',   'mean'),
        reinf_ratio   =('reinforce_ratio',      'mean'),
        sf_sent       =('ships_fraction_sent',  'mean'),
        n             =('did_act',              'count'),
    ).round(3).to_string())

    print("\n── Только атакующие ходы: что атакуют в разных фазах ──────")
    att = df[df['did_act'] == 1]
    if len(att):
        print(att.groupby('phase_bin', observed=True).agg(
            avg_overkill  =('avg_overkill',       'mean'),
            avg_dist      =('avg_target_dist_frac','mean'),
            avg_tgt_ships =('avg_target_ships',   'mean'),
            avg_tgt_prod  =('avg_target_prod',    'mean'),
            reinforce_r   =('reinforce_ratio',    'mean'),
            n             =('did_act',            'count'),
        ).round(3).to_string())

    print("\n── Projected state: насколько позиция улучшается за ход ───")
    print(df.groupby('phase_bin', observed=True).agg(
        proj_ship_delta  =('proj_ship_ratio_delta',  'mean'),
        proj_plan_delta  =('proj_planet_ratio_delta','mean'),
        under_threat_avg =('own_under_threat_count', 'mean'),
        contested_enemy  =('enemy_contested_count',  'mean'),
        closest_threat   =('closest_threat_dist',    'mean'),
        net_incoming     =('net_incoming_own',        'mean'),
        n                =('did_act',               'count'),
    ).round(3).to_string())


if __name__ == '__main__':
    main()
