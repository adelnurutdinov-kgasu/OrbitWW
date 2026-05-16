#!/usr/bin/env python3
"""
02c_extract_launches.py — per-launch датасет с zone-value контекстом выбора.

Философия
─────────
Каждый запуск флота — дискретный ВЫБОР из множества доступных вариантов.
Мы хотим знать не только ЧТО выбрал игрок, но и ПОЧЕМУ:
  - какие zone-value были у выбранной цели?
  - что было доступно но не выбрано?
  - в каком zone-состоянии находился источник атаки?

Zone-фичи (из zones.py нашего агента)
──────────────────────────────────────
  Наш агент вычисляет 7 фич на каждую планету:
    area_inv        — силовой интеграл (требует симуляции → ПРОПУСКАЕМ)
    wnn_close_res   — взвешенный score соседства ← ВЫЧИСЛЯЕМ
    mean_dist_all   — удалённость от центра карты ← ВЫЧИСЛЯЕМ
    prod            — производство ← ЕСТЬ
    ships           — гарнизон ← ЕСТЬ
    n_cross         — число пересечений силовой кривой (требует симуляции → ПРОПУСКАЕМ)
    late_aggression — ripeness × deepness ← ВЫЧИСЛЯЕМ

  Зональные метки целей нашего агента:
    easy_target     — wnn > 0.5 И ships слабый
    priority_target — prod высокий И не далеко
    hard_far        — далеко И не слабый
    contested       — много силовых пересечений (→ приближение)
    periphery       — остальные

Структура строки (одна строка = один запуск флота)
───────────────────────────────────────────────────
  [МЕТА]
    episode_id, player_idx, step, launch_idx, reward

  [КОНТЕКСТ: состояние мира]
    phase, own_ship_ratio, own_prod_ratio, own_planet_ratio
    own_ships_total, own_prod_total, n_own_planets
    n_neutral_planets, n_enemy_planets, enemy_ships_total
    expansion_space, own_fleets_count, enemy_fleets_count
    under_threat_count
    own_territory_spread   — среднее расст. между своими планетами (компактность)
    own_frontier_frac      — доля наших планет с wnn < 0 (приграничные)

  [ИСТОЧНИК]
    src_ships, src_prod, src_ships_ratio
    src_wnn               — zone-score соседства источника
    src_ripeness          — prod/(1+ships): источник «спелый» сам по себе?
    src_nearest_enemy_dist— ближайший враг к источнику (fronline proxy)
    src_zone_approx       — frontline/bastion/rear/mid

  [ВЫБРАННАЯ ЦЕЛЬ: zone-values]
    tgt_owner_type        — 0=neutral, 1=enemy, 2=own
    tgt_prod, tgt_ships
    tgt_dist_frac         — норм. расст. от источника до цели
    tgt_ripeness          — prod/(1+ships): ценность/цена захвата
    tgt_wnn               — zone-score соседства цели (+= рядом с нами, -= у врага)
    tgt_centrality        — mean dist до всех планет (маленький = центральная)
    tgt_deepness          — mean dist от цели до НАШИХ планет (глубина в тылу)
    tgt_late_aggression   — ripeness × deepness (как в W_TARGETS нашего агента)
    tgt_nearest_own_dist  — ближайшая своя планета к цели
    tgt_nearest_enemy_dist— ближайшая вражеская планета к цели
    tgt_priority_approx   — приближённый priority score (без area_inv, n_cross)
    tgt_zone_approx       — easy_target/priority_target/hard_far/periphery
    ships_sent, overkill, fleet_size_ratio

  [АЛЬТЕРНАТИВЫ: что МОЖНО было выбрать]
    n_alternatives, n_neutral_avail, n_enemy_avail, n_own_avail
    # Нейтральные:
    neutral_{prod,ships,ripeness,wnn,late_agg}_{mean,max/min}
    neutral_dist_min
    # Вражеские:
    enemy_{prod,ships,ripeness,wnn,late_agg}_{mean,max/min}
    enemy_dist_min
    own_ships_min         — уязвимая своя планета (reinforce target)

  [ОТНОСИТЕЛЬНЫЙ ВЫБОР vs альтернативы того же типа]
    rel_prod, rel_ships, rel_dist, rel_ripeness, rel_wnn, rel_late_agg
    chose_cheapest        — выбрал самого слабого из доступных
    chose_richest_prod    — выбрал самого богатого по производству
    chose_best_ripeness   — выбрал самый выгодный по prod/ships
    chose_best_wnn        — выбрал самый доступный по proximity
    tgt_priority_rank     — ранг цели по priority_approx среди доступных (1=лучший)
    n_better_alternatives — сколько альтернатив лучше выбранной по priority

Запуск:
  python3 opponent_learning/scripts/02c_extract_launches.py
  python3 opponent_learning/scripts/02c_extract_launches.py --limit 50
  python3 opponent_learning/scripts/02c_extract_launches.py --min-reward 0.5
"""

import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE     = Path(__file__).parent
RAW_DIR  = HERE.parent / "data" / "raw"
PROC_DIR = HERE.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

NEUTRAL_OWNER = -1

# Веса zone-priority для целей (из W_TARGETS zones.py, без area_inv и n_cross)
# Нормируем чтобы сумма abs весов = 1
_W = {
    'wnn_close_res':   +0.8,
    'mean_dist_all':   -0.2,
    'prod':            +0.5,
    'ships':           -0.5,
    'late_aggression': +0.9,   # умножается на phase
}
_W_TOTAL = sum(abs(v) for v in _W.values())   # 2.9


# ═══════════════════════════════════════════════════════════════════
# Парсеры
# ═══════════════════════════════════════════════════════════════════

def _parse_planet(p) -> Optional[dict]:
    if isinstance(p, list) and len(p) >= 7:
        return {
            'id':    int(p[0]),
            'owner': int(p[1]) if p[1] is not None else NEUTRAL_OWNER,
            'x':     float(p[2]),
            'y':     float(p[3]),
            'ships': max(0.0, float(p[5])),
            'prod':  float(p[6]),
        }
    if isinstance(p, dict):
        owner = p.get('owner', NEUTRAL_OWNER)
        return {
            'id':    p['id'],
            'owner': owner if owner is not None else NEUTRAL_OWNER,
            'x':     float(p.get('x', 0)),
            'y':     float(p.get('y', 0)),
            'ships': max(0.0, float(p.get('ships', 0))),
            'prod':  float(p.get('production', p.get('prod', 0))),
        }
    return None


def _parse_fleet(f) -> Optional[dict]:
    if isinstance(f, list) and len(f) >= 6:
        return {'owner': int(f[1]), 'x': float(f[2]), 'y': float(f[3]),
                'angle': float(f[4]), 'ships': float(f[5])}
    if isinstance(f, dict):
        return f
    return None


def _planet_map(raw: list) -> dict:
    pm = {}
    for p in raw:
        pd_ = _parse_planet(p)
        if pd_ is not None:
            pm[pd_['id']] = pd_
    return pm


def _find_target(src: dict, angle: float, pm: dict) -> Optional[dict]:
    sx, sy   = src['x'], src['y']
    dx, dy   = math.cos(angle), math.sin(angle)
    best, bs = None, float('inf')
    for t in pm.values():
        if t['id'] == src['id']:
            continue
        tdx, tdy = t['x'] - sx, t['y'] - sy
        dist = math.hypot(tdx, tdy)
        if dist < 1e-6:
            continue
        dot = (dx * tdx + dy * tdy) / dist
        if dot < 0.2:
            continue
        ang_diff = abs(math.atan2(math.sin(angle - math.atan2(tdy, tdx)),
                                   math.cos(angle - math.atan2(tdy, tdx))))
        if ang_diff < bs:
            bs, best = ang_diff, t
    return best


# ═══════════════════════════════════════════════════════════════════
# Zone-фичи (вычисляются один раз для всех планет на шаге)
# ═══════════════════════════════════════════════════════════════════

class ZoneCache:
    """Предвычисляет zone-фичи для всех планет за один векторизованный проход."""

    def __init__(self, pm: dict, pidx: int, max_d: float, phase: float):
        planets = list(pm.values())
        n = len(planets)
        if n == 0:
            self._cache = {}
            return

        ids     = np.array([p['id']    for p in planets])
        xs      = np.array([p['x']     for p in planets], dtype=np.float32)
        ys      = np.array([p['y']     for p in planets], dtype=np.float32)
        ships   = np.array([p['ships'] for p in planets], dtype=np.float32)
        prods   = np.array([p['prod']  for p in planets], dtype=np.float32)
        owners  = np.array([p['owner'] for p in planets])

        # Матрица расстояний [n × n]
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        D  = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(D, np.inf)

        # wnn_close_res: взвешенная сумма (sign × (ships + 5*prod) / dist)
        # sign: +1 = наша, -1 = вражеская, 0 = нейтральная
        signs = np.where(owners == pidx, 1.0,
                np.where(owners == NEUTRAL_OWNER, 0.0, -1.0))
        weights  = (ships[None, :] + 5.0 * prods[None, :]) / D   # [n × n]
        wnn_num  = (signs[None, :] * weights).sum(axis=1)         # [n]
        wnn_den  = weights.sum(axis=1)
        wnn      = np.where(wnn_den > 0, wnn_num / wnn_den, 0.0)

        # mean_dist_all: средняя дистанция до всех других
        D_finite = np.where(D == np.inf, 0.0, D)
        mean_dist_all = D_finite.sum(axis=1) / max(1, n - 1)

        # Дистанции до ближайших own / enemy
        own_mask   = (owners == pidx)
        enemy_mask = (owners != pidx) & (owners != NEUTRAL_OWNER)

        def _min_dist(mask):
            if mask.sum() == 0:
                return np.full(n, max_d)
            d = D.copy()
            d[:, ~mask] = np.inf
            return d.min(axis=1)

        nearest_own   = _min_dist(own_mask)
        nearest_enemy = _min_dist(enemy_mask)

        # deepness для каждой планеты (mean dist до наших)
        if own_mask.sum() > 0:
            own_dists = D_finite[:, own_mask]
            deepness  = own_dists.mean(axis=1)
        else:
            deepness  = np.zeros(n)

        # ripeness и late_aggression
        ripeness        = prods / (1.0 + ships)
        late_aggression = ripeness * deepness

        # z-score для priority_approx (отдельно по не-своим планетам)
        def _zscore_safe(arr):
            s = arr.std()
            return (arr - arr.mean()) / s if s > 1e-9 else arr * 0

        # Только для не-своих планет (потенциальные цели)
        tgt_mask = ~own_mask
        wnn_z        = _zscore_safe(wnn)
        mda_z        = _zscore_safe(mean_dist_all)
        prod_z       = _zscore_safe(prods)
        ships_z      = _zscore_safe(ships)
        late_agg_z   = _zscore_safe(late_aggression)

        priority_approx = (
            _W['wnn_close_res']   * wnn_z +
            _W['mean_dist_all']   * mda_z +
            _W['prod']            * prod_z +
            _W['ships']           * ships_z +
            _W['late_aggression'] * phase * late_agg_z
        ) / _W_TOTAL

        # Zone label (приближение без area_inv и n_cross)
        def _zone_label(i):
            if own_mask[i]:
                # наша планета
                in_them = wnn[i] < -0.5
                in_us   = wnn[i] >  0.5
                far     = mean_dist_all[i] > mean_dist_all.mean()
                if in_them:  return 'frontline'
                if in_us and prods[i] > prods.mean(): return 'bastion'
                if in_us:    return 'rear'
                if far:      return 'isolated'
                return 'mid'
            else:
                in_us  = wnn[i] >  0.5
                weak   = ships[i] < ships.mean()
                rich   = prods[i] > prods.mean()
                far    = mean_dist_all[i] > mean_dist_all.mean() + mean_dist_all.std()
                if in_us and weak:      return 'easy_target'
                if rich and not far:    return 'priority_target'
                if far and not weak:    return 'hard_far'
                return 'periphery'

        zones = [_zone_label(i) for i in range(n)]

        self._cache = {}
        for i, pid in enumerate(ids):
            self._cache[int(pid)] = {
                'wnn':              float(wnn[i]),
                'mean_dist_all':    float(mean_dist_all[i]),
                'deepness':         float(deepness[i]),
                'ripeness':         float(ripeness[i]),
                'late_aggression':  float(late_aggression[i]),
                'nearest_own_dist': float(nearest_own[i] / max_d),
                'nearest_enemy_dist': float(nearest_enemy[i] / max_d),
                'priority_approx':  float(priority_approx[i]),
                'zone_approx':      zones[i],
            }

    def get(self, pid: int) -> dict:
        return self._cache.get(int(pid), {
            'wnn': 0.0, 'mean_dist_all': 0.5, 'deepness': 0.5,
            'ripeness': 0.0, 'late_aggression': 0.0,
            'nearest_own_dist': 1.0, 'nearest_enemy_dist': 1.0,
            'priority_approx': 0.0, 'zone_approx': 'periphery',
        })


# ═══════════════════════════════════════════════════════════════════
# Контекст состояния мира
# ═══════════════════════════════════════════════════════════════════

def _context(pm: dict, fleets: list, pidx: int,
             step_idx: int, n_steps: int, zc: ZoneCache) -> dict:
    n_pl     = len(pm)
    phase    = step_idx / max(1, n_steps - 1)

    own_pl    = [p for p in pm.values() if p['owner'] == pidx]
    enemy_pl  = [p for p in pm.values()
                 if p['owner'] != pidx and p['owner'] != NEUTRAL_OWNER]
    neutral_pl = [p for p in pm.values() if p['owner'] == NEUTRAL_OWNER]

    own_ships  = sum(p['ships'] for p in own_pl)
    own_prod   = sum(p['prod']  for p in own_pl)
    all_ships  = sum(p['ships'] for p in pm.values())
    all_prod   = sum(p['prod']  for p in pm.values())

    own_fl    = [f for f in fleets if f['owner'] == pidx]
    enemy_fl  = [f for f in fleets
                 if f['owner'] != pidx and f['owner'] != NEUTRAL_OWNER]

    # under_threat через вражеские флоты летящие к нашим планетам
    own_planet_ids = {p['id'] for p in own_pl}
    under_threat = 0
    for f in enemy_fl:
        tgt = _find_target({'id': -999, 'x': f['x'], 'y': f['y']},
                           f['angle'], pm)
        if tgt and tgt['id'] in own_planet_ids:
            under_threat += 1

    # own_territory_spread: среднее расст. между своими планетами
    if len(own_pl) >= 2:
        coords = [(p['x'], p['y']) for p in own_pl]
        dists  = [math.hypot(coords[i][0] - coords[j][0],
                             coords[i][1] - coords[j][1])
                  for i in range(len(coords))
                  for j in range(i + 1, len(coords))]
        territory_spread = sum(dists) / len(dists)
    else:
        territory_spread = 0.0

    # own_frontier_frac: доля наших планет с wnn < 0
    frontier_count = sum(1 for p in own_pl if zc.get(p['id'])['wnn'] < 0)

    return {
        'phase':                round(phase, 4),
        'own_ship_ratio':       round(own_ships  / max(1, all_ships), 4),
        'own_prod_ratio':       round(own_prod   / max(1, all_prod),  4),
        'own_planet_ratio':     round(len(own_pl) / max(1, n_pl),     4),
        'own_ships_total':      round(own_ships,  1),
        'own_prod_total':       round(own_prod,   1),
        'n_own_planets':        len(own_pl),
        'n_neutral_planets':    len(neutral_pl),
        'n_enemy_planets':      len(enemy_pl),
        'enemy_ships_total':    round(sum(p['ships'] for p in enemy_pl), 1),
        'expansion_space':      round(len(neutral_pl) / max(1, n_pl), 4),
        'own_fleets_count':     len(own_fl),
        'enemy_fleets_count':   len(enemy_fl),
        'under_threat_count':   under_threat,
        'own_territory_spread': round(territory_spread, 1),
        'own_frontier_frac':    round(frontier_count / max(1, len(own_pl)), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# Фичи источника
# ═══════════════════════════════════════════════════════════════════

def _src_features(src: dict, own_ships: float, zc: ZoneCache) -> dict:
    z = zc.get(src['id'])
    return {
        'src_ships':              round(src['ships'], 1),
        'src_prod':               src['prod'],
        'src_ships_ratio':        round(src['ships'] / max(1, own_ships), 4),
        'src_wnn':                round(z['wnn'], 4),
        'src_ripeness':           round(z['ripeness'], 4),
        'src_nearest_enemy_dist': round(z['nearest_enemy_dist'], 4),
        'src_zone_approx':        z['zone_approx'],
    }


# ═══════════════════════════════════════════════════════════════════
# Фичи выбранной цели
# ═══════════════════════════════════════════════════════════════════

def _tgt_features(tgt: dict, src: dict, ships_sent: int,
                  own_ships: float, max_d: float,
                  pidx: int, zc: ZoneCache) -> dict:
    dist      = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
    dist_frac = dist / max_d
    owner     = tgt['owner']
    owner_type = 2 if owner == pidx else (0 if owner == NEUTRAL_OWNER else 1)
    z         = zc.get(tgt['id'])
    tgt_ships = max(1.0, tgt['ships'])

    return {
        'tgt_owner_type':         owner_type,
        'tgt_prod':               tgt['prod'],
        'tgt_ships':              tgt['ships'],
        'tgt_dist_frac':          round(dist_frac, 4),
        'tgt_ripeness':           round(z['ripeness'], 4),
        'tgt_wnn':                round(z['wnn'], 4),
        'tgt_centrality':         round(z['mean_dist_all'] / max_d, 4),
        'tgt_deepness':           round(z['deepness'] / max_d, 4),
        'tgt_late_aggression':    round(z['late_aggression'], 4),
        'tgt_nearest_own_dist':   round(z['nearest_own_dist'], 4),
        'tgt_nearest_enemy_dist': round(z['nearest_enemy_dist'], 4),
        'tgt_priority_approx':    round(z['priority_approx'], 4),
        'tgt_zone_approx':        z['zone_approx'],
        'ships_sent':             ships_sent,
        'overkill':               round(ships_sent / tgt_ships, 3),
        'fleet_size_ratio':       round(ships_sent / max(1, src['ships']), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# Фичи альтернатив
# ═══════════════════════════════════════════════════════════════════

def _alt_features(src: dict, tgt: dict, pm: dict,
                  pidx: int, max_d: float, own_ships: float,
                  zc: ZoneCache) -> dict:

    neutrals, enemies, owns = [], [], []

    for p in pm.values():
        if p['id'] == src['id']:
            continue
        dist_frac = math.hypot(p['x'] - src['x'], p['y'] - src['y']) / max_d
        z = zc.get(p['id'])
        entry = {
            'prod':           p['prod'],
            'ships':          p['ships'],
            'dist_frac':      dist_frac,
            'ripeness':       z['ripeness'],
            'wnn':            z['wnn'],
            'late_aggression': z['late_aggression'],
            'priority_approx': z['priority_approx'],
        }
        if p['owner'] == pidx:
            owns.append(entry)
        elif p['owner'] == NEUTRAL_OWNER:
            neutrals.append(entry)
        else:
            enemies.append(entry)

    def _stats(lst, name):
        if not lst:
            return {f'{name}_{k}': 0.0
                    for k in ['prod_mean', 'prod_max', 'ships_mean', 'ships_min',
                               'ripeness_mean', 'ripeness_max',
                               'wnn_mean', 'wnn_max',
                               'late_agg_mean', 'late_agg_max',
                               'priority_mean', 'priority_max', 'dist_min']}
        def _m(key):  return sum(e[key] for e in lst) / len(lst)
        def _mx(key): return max(e[key] for e in lst)
        def _mn(key): return min(e[key] for e in lst)
        return {
            f'{name}_prod_mean':     round(_m('prod'),           3),
            f'{name}_prod_max':      round(_mx('prod'),          3),
            f'{name}_ships_mean':    round(_m('ships'),          3),
            f'{name}_ships_min':     round(_mn('ships'),         3),
            f'{name}_ripeness_mean': round(_m('ripeness'),       4),
            f'{name}_ripeness_max':  round(_mx('ripeness'),      4),
            f'{name}_wnn_mean':      round(_m('wnn'),            4),
            f'{name}_wnn_max':       round(_mx('wnn'),           4),
            f'{name}_late_agg_mean': round(_m('late_aggression'),4),
            f'{name}_late_agg_max':  round(_mx('late_aggression'),4),
            f'{name}_priority_mean': round(_m('priority_approx'),4),
            f'{name}_priority_max':  round(_mx('priority_approx'),4),
            f'{name}_dist_min':      round(_mn('dist_frac'),     4),
        }

    out = {
        'n_alternatives':  len(neutrals) + len(enemies) + len(owns),
        'n_neutral_avail': len(neutrals),
        'n_enemy_avail':   len(enemies),
        'n_own_avail':     len(owns),
        'own_ships_min':   round(min((e['ships'] for e in owns), default=0), 1),
    }
    out.update(_stats(neutrals, 'neutral'))
    out.update(_stats(enemies,  'enemy'))
    return out


# ═══════════════════════════════════════════════════════════════════
# Относительный выбор
# ═══════════════════════════════════════════════════════════════════

def _relative_choice(tgt_f: dict, alt_f: dict) -> dict:
    owner_type = tgt_f['tgt_owner_type']
    prefix = {0: 'neutral', 1: 'enemy'}.get(owner_type, None)

    if prefix is None:   # reinforce
        return {
            'rel_prod': 0.0, 'rel_ships': 0.0, 'rel_dist': 0.0,
            'rel_ripeness': 0.0, 'rel_wnn': 0.0, 'rel_late_agg': 0.0,
            'chose_cheapest': 0, 'chose_richest_prod': 0,
            'chose_best_ripeness': 0, 'chose_best_wnn': 0,
            'tgt_priority_rank': 1, 'n_better_alternatives': 0,
        }

    mean_prod    = alt_f.get(f'{prefix}_prod_mean',     tgt_f['tgt_prod'])
    mean_ships   = alt_f.get(f'{prefix}_ships_mean',    tgt_f['tgt_ships'])
    mean_dist    = alt_f.get(f'{prefix}_dist_min',      tgt_f['tgt_dist_frac'])
    mean_rip     = alt_f.get(f'{prefix}_ripeness_mean', tgt_f['tgt_ripeness'])
    mean_wnn     = alt_f.get(f'{prefix}_wnn_mean',      tgt_f['tgt_wnn'])
    mean_la      = alt_f.get(f'{prefix}_late_agg_mean', tgt_f['tgt_late_aggression'])

    min_ships    = alt_f.get(f'{prefix}_ships_min',     tgt_f['tgt_ships'])
    max_prod     = alt_f.get(f'{prefix}_prod_max',      tgt_f['tgt_prod'])
    max_rip      = alt_f.get(f'{prefix}_ripeness_max',  tgt_f['tgt_ripeness'])
    max_wnn      = alt_f.get(f'{prefix}_wnn_max',       tgt_f['tgt_wnn'])
    max_prio     = alt_f.get(f'{prefix}_priority_max',  tgt_f['tgt_priority_approx'])
    mean_prio    = alt_f.get(f'{prefix}_priority_mean', tgt_f['tgt_priority_approx'])

    tgt_prio     = tgt_f['tgt_priority_approx']
    n_avail      = alt_f.get(f'n_{prefix}_avail', 1)

    # Ранг: сколько альтернатив лучше выбранной
    # Грубая оценка: если tgt_prio < max → есть лучшие
    n_better = max(0, round((max_prio - tgt_prio) / max(0.01, max_prio - mean_prio + 1e-6)
                            * max(0, n_avail - 1)))
    n_better = min(n_better, n_avail)
    prio_rank = 1 + n_better

    return {
        'rel_prod':             round(tgt_f['tgt_prod']               - mean_prod,  3),
        'rel_ships':            round(tgt_f['tgt_ships']              - mean_ships, 3),
        'rel_dist':             round(tgt_f['tgt_dist_frac']          - mean_dist,  4),
        'rel_ripeness':         round(tgt_f['tgt_ripeness']           - mean_rip,   4),
        'rel_wnn':              round(tgt_f['tgt_wnn']                - mean_wnn,   4),
        'rel_late_agg':         round(tgt_f['tgt_late_aggression']    - mean_la,    4),
        'chose_cheapest':       int(abs(tgt_f['tgt_ships'] - min_ships) < 0.5),
        'chose_richest_prod':   int(abs(tgt_f['tgt_prod']  - max_prod)  < 0.5),
        'chose_best_ripeness':  int(abs(tgt_f['tgt_ripeness'] - max_rip) < 1e-4),
        'chose_best_wnn':       int(abs(tgt_f['tgt_wnn']  - max_wnn)  < 1e-4),
        'tgt_priority_rank':    prio_rank,
        'n_better_alternatives': n_better,
    }


# ═══════════════════════════════════════════════════════════════════
# Основная функция
# ═══════════════════════════════════════════════════════════════════

def extract_launches(episode_id: int, data: dict) -> list:
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

    max_d = 1.0
    for step in steps[:5]:
        for sp in step:
            if isinstance(sp, dict):
                obs = sp.get('observation', {})
                pl  = obs.get('planets', []) if isinstance(obs, dict) else []
                pm  = _planet_map(pl)
                if pm:
                    xs = [p['x'] for p in pm.values()]
                    ys = [p['y'] for p in pm.values()]
                    d  = max((math.hypot(xs[i]-xs[j], ys[i]-ys[j])
                              for i in range(len(xs))
                              for j in range(i+1, len(xs))), default=1.0)
                    max_d = max(max_d, d)
                    break
        if max_d > 1.0:
            break

    rows = []

    for step_idx, step in enumerate(steps):
        if not isinstance(step, list) or not step:
            continue

        obs0        = step[0].get('observation', {}) if isinstance(step[0], dict) else {}
        raw_planets = obs0.get('planets', []) if isinstance(obs0, dict) else []
        raw_fleets  = obs0.get('fleets',  []) if isinstance(obs0, dict) else []

        pm     = _planet_map(raw_planets)
        fleets = [f for f in [_parse_fleet(x) for x in raw_fleets] if f]
        if not pm:
            continue

        phase = step_idx / max(1, n_steps - 1)

        for pidx in range(n_players):
            if pidx >= len(step) or not isinstance(step[pidx], dict):
                continue

            action_raw = step[pidx].get('action', [])
            if not isinstance(action_raw, list) or not action_raw:
                continue

            launches = []
            for a in action_raw:
                if isinstance(a, (list, tuple)) and len(a) >= 3:
                    try:
                        launches.append({'source_id': int(a[0]),
                                         'angle': float(a[1]), 'ships': int(a[2])})
                    except (TypeError, ValueError):
                        pass
            if not launches:
                continue

            # Zone cache — один раз на (step, pidx)
            zc = ZoneCache(pm, pidx, max_d, phase)

            ctx_f = _context(pm, fleets, pidx, step_idx, n_steps, zc)
            own_ships = ctx_f['own_ships_total']

            for launch_idx, launch in enumerate(launches):
                src = pm.get(launch['source_id'])
                if src is None:
                    continue
                tgt = _find_target(src, launch['angle'], pm)
                if tgt is None:
                    continue

                src_f = _src_features(src, own_ships, zc)
                tgt_f = _tgt_features(tgt, src, launch['ships'], own_ships, max_d, pidx, zc)
                alt_f = _alt_features(src, tgt, pm, pidx, max_d, own_ships, zc)
                rel_f = _relative_choice(tgt_f, alt_f)

                rows.append({
                    'episode_id':  episode_id,
                    'player_idx':  pidx,
                    'step':        step_idx,
                    'launch_idx':  launch_idx,
                    'reward':      final_rewards.get(pidx, 0.0),
                    **ctx_f,
                    **src_f,
                    **tgt_f,
                    **alt_f,
                    **rel_f,
                })

    return rows


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',        default=str(RAW_DIR))
    parser.add_argument('--out',        default=str(PROC_DIR / 'launches.csv'))
    parser.add_argument('--limit',      type=int,   default=0)
    parser.add_argument('--min-reward', type=float, default=0.0)
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    files   = sorted(raw_dir.glob('[0-9]*.json'))
    if args.limit:
        files = files[:args.limit]

    n = len(files)
    print(f"Найдено {n} replay-файлов")
    print(f"Фичи: контекст + источник (zone) + цель (zone) + альтернативы (zone) + rel-choice")
    print(f"Ожидаем: ~{n * 550 // 1000}k строк")

    all_rows, errors = [], 0
    for i, fpath in enumerate(files, 1):
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
            rows = extract_launches(int(fpath.stem), data)
            if args.min_reward > 0:
                rows = [r for r in rows if r['reward'] >= args.min_reward]
            all_rows.extend(rows)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠ {fpath.name}: {e}")

        if i % 50 == 0 or i == n:
            print(f"  [{i:>4}/{n}]  запусков: {len(all_rows):>8,}  ошибок: {errors}")

    if not all_rows:
        print("Нет данных.")
        return

    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    df.to_csv(out, index=False)

    print(f"\n✓ {out}")
    print(f"  Строк:   {len(df):,}  |  Колонок: {len(df.columns)}")
    print(f"  Размер:  {out.stat().st_size / 1024 / 1024:.1f} MB  |  Ошибок: {errors}")

    # ── Быстрый анализ ────────────────────────────────────────────────────────
    type_map = {0: 'neutral', 1: 'enemy', 2: 'reinforce'}
    df['tgt_type'] = df['tgt_owner_type'].map(type_map)

    print("\n── Распределение типов целей ───")
    print(df['tgt_type'].value_counts(normalize=True).round(3).to_string())

    print("\n── Zone-label выбранных целей ──")
    print(df['tgt_zone_approx'].value_counts(normalize=True).round(3).to_string())

    print("\n── Zone-label источников ────────")
    print(df['src_zone_approx'].value_counts(normalize=True).round(3).to_string())

    print("\n── Относительный выбор (vs альтернативы) ──")
    for ttype in ['neutral', 'enemy']:
        sub = df[df['tgt_type'] == ttype]
        if not len(sub): continue
        print(f"\n  [{ttype}]  n={len(sub):,}")
        print(f"    rel_prod       {sub['rel_prod'].median():+.3f}  (+=богатые цели)")
        print(f"    rel_ships      {sub['rel_ships'].median():+.3f}  (-=слабые цели)")
        print(f"    rel_ripeness   {sub['rel_ripeness'].median():+.4f}  (+=выгодные)")
        print(f"    rel_wnn        {sub['rel_wnn'].median():+.4f}  (+=близкие к нам)")
        print(f"    rel_late_agg   {sub['rel_late_agg'].median():+.4f}  (+=глубокие цели)")
        print(f"    chose_cheapest        {sub['chose_cheapest'].mean():.1%}")
        print(f"    chose_richest_prod    {sub['chose_richest_prod'].mean():.1%}")
        print(f"    chose_best_ripeness   {sub['chose_best_ripeness'].mean():.1%}")
        print(f"    chose_best_wnn        {sub['chose_best_wnn'].mean():.1%}")
        print(f"    prio_rank=1 (optimal) {(sub['tgt_priority_rank']==1).mean():.1%}")

    print("\n── Все колонки ─────────────────")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:>3}. {c}")


if __name__ == '__main__':
    main()
