#!/usr/bin/env python3
"""
02d_extract_choice_sets.py — Long-format choice sets для conditional logit / GNN.

Отличие от 02c_extract_launches.py
────────────────────────────────────
02c: одна строка = один запуск (chosen + агрегаты по альтернативам).
02d: одна строка = одна планета в момент запуска (chosen + все альтернативы).

Структура реплея (Kaggle-формат orbit_wars):
  d['steps']          — список шагов
  d['steps'][si]      — список obs на шаге [player0_dict, player1_dict, ...]
  d['steps'][si][pi]['action']      = [[src_id, angle, ships], ...]
  d['steps'][si][pi]['observation'] = {'planets': [...], 'fleets': [...], ...}
  planet = [id, owner, x, y, radius, ships, production]
  d['rewards']        = [r_p0, r_p1]  (−1 или 1)

Выходной формат (одна строка = одна планета-вариант):
──────────────────────────────────────────────────────
  [КЛЮЧИ]
    episode_id, player_idx, step, launch_idx
    choice_set_id          — уникальный ключ набора (для groupby при обучении)
    planet_id
    is_chosen              — 1 = игрок выбрал именно эту планету целью

  [ФИЧИ ВАРИАНТА]
    owner_type    — 0=neutral, 1=enemy, 2=own
    prod, ships
    dist_frac     — расстояние до цели / max_map_dist
    ripeness      — prod/(1+ships)
    wnn           — zone-score соседства (ручной 1-hop GNN)
    deepness      — mean_dist до наших планет (норм.)
    late_aggression — ripeness × deepness (без phase)
    priority_approx — взвешенный score (без area_inv/n_cross)

  [КОНТЕКСТ ЗАПУСКА — одинаков для всей группы выбора]
    phase
    own_ship_ratio, own_prod_ratio, own_planet_ratio
    n_own_planets, n_neutral_planets, n_enemy_planets
    src_ships, src_prod, src_wnn, src_ripeness

  [ИСХОД]
    reward         — 1 = победа, -1 = поражение

Применение:
  - conditional_logit: logit(is_chosen ~ фичи | choice_set_id)
  - GNN training:      choice_set_id → граф (планеты = ноды)
  - XGBoost pointwise: бинарный классификатор is_chosen

Размер: ~8–12M строк для 500 реплеев (≈15–25 вариантов на запуск).

Запуск:
  python3 opponent_learning/scripts/02d_extract_choice_sets.py
  python3 opponent_learning/scripts/02d_extract_choice_sets.py --limit 50
  python3 opponent_learning/scripts/02d_extract_choice_sets.py --out custom.csv
"""

import json
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE     = Path(__file__).parent
RAW_DIR  = HERE.parent / "data" / "raw"
PROC_DIR = HERE.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

NEUTRAL_OWNER = -1


# ═══════════════════════════════════════════════════════════════════
# Парсеры
# ═══════════════════════════════════════════════════════════════════

def _parse_planet(p) -> dict | None:
    """planet = [id, owner, x, y, radius, ships, production]"""
    if isinstance(p, (list, tuple)) and len(p) >= 7:
        owner = int(p[1]) if p[1] is not None else NEUTRAL_OWNER
        return {'id': int(p[0]), 'owner': owner,
                'x': float(p[2]), 'y': float(p[3]),
                'ships': max(0.0, float(p[5])),
                'prod': float(p[6])}
    if isinstance(p, dict):
        owner = p.get('owner', NEUTRAL_OWNER)
        return {'id': int(p['id']),
                'owner': owner if owner is not None else NEUTRAL_OWNER,
                'x': float(p.get('x', 0)), 'y': float(p.get('y', 0)),
                'ships': max(0.0, float(p.get('ships', 0))),
                'prod': float(p.get('production', p.get('prod', 0)))}
    return None


def _planet_map(raw: list) -> dict:
    pm = {}
    for p in raw:
        pd_ = _parse_planet(p)
        if pd_ is not None:
            pm[pd_['id']] = pd_
    return pm


def _find_target_by_angle(src: dict, angle: float, pm: dict) -> dict | None:
    """Ищем планету, ближайшую по направлению к angle от src."""
    sx, sy = src['x'], src['y']
    dx, dy = math.cos(angle), math.sin(angle)
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
        ang = math.atan2(tdy, tdx)
        ang_diff = abs(math.atan2(math.sin(angle - ang), math.cos(angle - ang)))
        if ang_diff < bs:
            bs, best = ang_diff, t
    return best


# ═══════════════════════════════════════════════════════════════════
# ZoneCache — предвычисляет planet-level фичи за один проход
# ═══════════════════════════════════════════════════════════════════

class ZoneCache:
    def __init__(self, pm: dict, pidx: int, max_d: float, phase: float):
        planets = list(pm.values())
        n = len(planets)
        if n == 0:
            self._cache = {}
            return

        ids    = np.array([p['id']    for p in planets])
        xs     = np.array([p['x']     for p in planets], dtype=np.float32)
        ys     = np.array([p['y']     for p in planets], dtype=np.float32)
        ships  = np.array([p['ships'] for p in planets], dtype=np.float32)
        prods  = np.array([p['prod']  for p in planets], dtype=np.float32)
        owners = np.array([p['owner'] for p in planets])

        # Матрица расстояний
        D = np.sqrt((xs[:, None] - xs[None, :])**2 +
                    (ys[:, None] - ys[None, :])**2)
        np.fill_diagonal(D, np.inf)

        # wnn_close_res: ручной 1-hop GNN aggregation
        signs   = np.where(owners == pidx, 1.0,
                  np.where(owners == NEUTRAL_OWNER, 0.0, -1.0))
        W       = (ships[None, :] + 5.0 * prods[None, :]) / D
        wnn_num = (signs[None, :] * W).sum(axis=1)
        wnn_den = W.sum(axis=1)
        wnn     = np.where(wnn_den > 0, wnn_num / wnn_den, 0.0)

        # mean_dist_all
        D_fin  = np.where(np.isinf(D), 0.0, D)
        mda    = D_fin.sum(axis=1) / max(1, n - 1)

        # deepness = mean dist до своих планет
        own_mask = (owners == pidx)
        if own_mask.sum() > 0:
            deepness = D_fin[:, own_mask].mean(axis=1)
        else:
            deepness = np.zeros(n)

        # nearest_enemy
        enemy_mask = (owners != pidx) & (owners != NEUTRAL_OWNER)

        def _min_dist(mask):
            if mask.sum() == 0:
                return np.full(n, max_d)
            d = D.copy()
            d[:, ~mask] = np.inf
            return d.min(axis=1)

        nearest_enemy = _min_dist(enemy_mask)

        ripeness        = prods / (1.0 + ships)
        late_aggression = ripeness * deepness

        # priority_approx (без area_inv, n_cross; phase на late_aggression)
        _W = {'wnn': 0.8, 'mda': -0.2, 'prod': 0.5, 'ships': -0.5, 'la': 0.9}
        W_tot = sum(abs(v) for v in _W.values())

        def _z(arr):
            s = arr.std()
            return (arr - arr.mean()) / s if s > 1e-9 else arr * 0.0

        priority = (_W['wnn'] * _z(wnn) + _W['mda'] * _z(mda) +
                    _W['prod'] * _z(prods) + _W['ships'] * _z(ships) +
                    _W['la'] * phase * _z(late_aggression)) / W_tot

        self._cache = {}
        md = max(max_d, 1.0)
        for i, pid in enumerate(ids):
            self._cache[int(pid)] = {
                'wnn':              float(wnn[i]),
                'mean_dist_all':    float(mda[i]),
                'deepness':         float(deepness[i] / md),
                'ripeness':         float(ripeness[i]),
                'late_aggression':  float(late_aggression[i]),
                'nearest_enemy_dist': float(nearest_enemy[i] / md),
                'priority_approx':  float(priority[i]),
            }

    def get(self, pid: int) -> dict:
        return self._cache.get(int(pid), {
            'wnn': 0.0, 'mean_dist_all': 0.5, 'deepness': 0.5,
            'ripeness': 0.0, 'late_aggression': 0.0,
            'nearest_enemy_dist': 1.0, 'priority_approx': 0.0,
        })


# ═══════════════════════════════════════════════════════════════════
# Основная обработка одного эпизода
# ═══════════════════════════════════════════════════════════════════

def _process_episode(ep_data: dict) -> list[dict]:
    """
    Принимает dict реплея Kaggle-формата.
    Возвращает список строк (long-format choice sets).

    Структура:
      ep_data['steps'][si][pi]['action']      = [[src_id, angle, ships], ...]
      ep_data['steps'][si][pi]['observation'] = {'planets': [...], ...}
      ep_data['rewards']                      = [r_p0, r_p1]
    """
    eid     = ep_data.get('id') or ep_data.get('episode_id', 'unknown')
    steps   = ep_data.get('steps', [])
    rewards = ep_data.get('rewards', [])
    n_steps = len(steps)
    if n_steps == 0:
        return []

    rows = []

    # Определяем число игроков из первого шага
    n_players = len(steps[0]) if steps else 2

    for pidx in range(n_players):
        reward = float(rewards[pidx]) if pidx < len(rewards) else 0.0
        launch_idx_global = 0

        for si, step in enumerate(steps):
            if pidx >= len(step):
                continue

            player_data = step[pidx]
            actions     = player_data.get('action') or []
            obs         = player_data.get('observation', {})
            planets_raw = obs.get('planets', [])

            if not planets_raw or not actions:
                continue

            pm    = _planet_map(planets_raw)
            phase = si / max(1, n_steps - 1)

            # max_d на карте
            coords = [(p['x'], p['y']) for p in pm.values()]
            if len(coords) < 2:
                continue
            max_d = max(
                math.hypot(a[0]-b[0], a[1]-b[1])
                for i, a in enumerate(coords) for b in coords[i+1:]
            ) or 1.0

            zc = ZoneCache(pm, pidx, max_d, phase)

            # Контекст шага
            own_pl  = [p for p in pm.values() if p['owner'] == pidx]
            neut_pl = [p for p in pm.values() if p['owner'] == NEUTRAL_OWNER]
            enm_pl  = [p for p in pm.values()
                       if p['owner'] != pidx and p['owner'] != NEUTRAL_OWNER]

            all_ships = sum(p['ships'] for p in pm.values()) or 1.0
            all_prod  = sum(p['prod']  for p in pm.values()) or 1.0
            own_ships = sum(p['ships'] for p in own_pl) or 1.0
            own_prod  = sum(p['prod']  for p in own_pl)

            ctx = {
                'phase':             round(phase, 4),
                'own_ship_ratio':    round(own_ships / all_ships, 4),
                'own_prod_ratio':    round(own_prod  / all_prod,  4),
                'own_planet_ratio':  round(len(own_pl) / max(1, len(pm)), 4),
                'n_own_planets':     len(own_pl),
                'n_neutral_planets': len(neut_pl),
                'n_enemy_planets':   len(enm_pl),
            }

            # Обрабатываем каждый запуск на шаге
            for action in actions:
                # action = [src_id, angle, ships_sent]
                if not isinstance(action, (list, tuple)) or len(action) < 3:
                    continue
                src_id, angle, ships_sent = int(action[0]), float(action[1]), int(action[2])

                src = pm.get(src_id)
                if src is None:
                    continue

                # Находим цель по углу
                tgt = _find_target_by_angle(src, angle, pm)
                if tgt is None:
                    continue

                zs = zc.get(src_id)
                src_ctx = {
                    'src_ships':    round(src['ships'], 1),
                    'src_prod':     src['prod'],
                    'src_wnn':      round(zs['wnn'], 4),
                    'src_ripeness': round(zs['ripeness'], 4),
                }

                # Набор вариантов = все планеты кроме источника
                alternatives = [p for p in pm.values() if p['id'] != src_id]
                if not alternatives:
                    continue

                choice_set_id = f"{eid}_{pidx}_{si}_{launch_idx_global}"
                launch_idx_global += 1

                for planet in alternatives:
                    z = zc.get(planet['id'])
                    dist = math.hypot(planet['x'] - src['x'],
                                      planet['y'] - src['y'])
                    dist_frac  = dist / max_d
                    owner      = planet['owner']
                    owner_type = (2 if owner == pidx else
                                  0 if owner == NEUTRAL_OWNER else 1)

                    rows.append({
                        # Ключи
                        'episode_id':    eid,
                        'player_idx':    pidx,
                        'step':          si,
                        'launch_idx':    launch_idx_global - 1,
                        'choice_set_id': choice_set_id,
                        'planet_id':     planet['id'],
                        'is_chosen':     int(planet['id'] == tgt['id']),
                        # Фичи варианта
                        'owner_type':        owner_type,
                        'prod':              round(planet['prod'], 3),
                        'ships':             round(planet['ships'], 1),
                        'dist_frac':         round(dist_frac, 4),
                        'ripeness':          round(z['ripeness'], 4),
                        'wnn':               round(z['wnn'], 4),
                        'deepness':          round(z['deepness'], 4),
                        'late_aggression':   round(z['late_aggression'], 4),
                        'priority_approx':   round(z['priority_approx'], 4),
                        # Контекст (одинаков для всей группы)
                        **ctx,
                        **src_ctx,
                        # Исход эпизода
                        'reward': reward,
                    })

    return rows


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Extract long-format choice sets from orbit_wars replays')
    parser.add_argument('--limit', type=int, default=None,
                        help='Ограничить число реплеев (для отладки)')
    parser.add_argument('--out',   type=str, default=None,
                        help='Путь к выходному CSV (default: data/processed/choice_sets.csv)')
    parser.add_argument('--chunk', type=int, default=50,
                        help='Сохранять каждые N реплеев (контроль памяти)')
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else PROC_DIR / 'choice_sets.csv'

    files = sorted(RAW_DIR.glob('*.json'))
    if args.limit:
        files = files[:args.limit]

    print(f"Processing {len(files)} replay files")
    print(f"Output: {out_path}\n")

    first_write  = True
    total_rows   = 0
    buf          = []

    for i, fp in enumerate(files):
        try:
            ep_data = json.loads(fp.read_text())
        except Exception as e:
            print(f"  ⚠ {fp.name}: {e}")
            continue

        try:
            rows = _process_episode(ep_data)
        except Exception as e:
            print(f"  ⚠ {fp.name} (processing): {e}")
            continue

        buf.extend(rows)

        if (i + 1) % args.chunk == 0 or (i + 1) == len(files):
            if buf:
                df_chunk = pd.DataFrame(buf)
                df_chunk.to_csv(out_path,
                                mode='a' if not first_write else 'w',
                                header=first_write,
                                index=False)
                total_rows  += len(df_chunk)
                first_write  = False
                buf.clear()
                print(f"  [{i+1:>3}/{len(files)}]  {total_rows:>10,} rows written …")

    print(f"\n✓ Done: {total_rows:,} rows → {out_path}")

    if total_rows > 0:
        sample = pd.read_csv(out_path, nrows=3)
        print(f"  Columns ({len(sample.columns)}): {sample.columns.tolist()}")
        print(f"\n  is_chosen rate: "
              f"{pd.read_csv(out_path, usecols=['is_chosen'])['is_chosen'].mean():.3f}")
        cs_size = pd.read_csv(out_path, usecols=['choice_set_id'])['choice_set_id'].nunique()
        print(f"  Unique choice sets: {cs_size:,}")

    print(f"\nNext steps:")
    print(f"  python3 opponent_learning/scripts/10_conditional_logit.py")
    print(f"  python3 opponent_learning/scripts/11_gnn_base.py")


if __name__ == '__main__':
    main()
