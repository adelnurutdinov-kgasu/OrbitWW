#!/usr/bin/env python3
"""
02_extract_features.py — извлекает поведенческие фичи оппонентов из replay-файлов.

Реальный формат orbit_wars replay:
  planets: список списков [id, owner, x, y, radius, ships, production]
    owner = -1 для нейтральных, 0/1 для игроков
  action:  [[source_planet_id, aim_angle_radians, ships], ...]
    ВНИМАНИЕ: нет target_id — только угол запуска!
  fleets:  [fleet_id, owner, x, y, vx, vy, ships, ?] (структура уточняется)

Для определения цели атаки используем angle-matching:
  из позиции source_planet вычисляем угол до каждой другой планеты,
  выбираем ближайшую к aim_angle (вперёд, dot > 0).

Фичи которые извлекаем:
  attack_rate        — доля ходов с хоть одним запуском
  expansion_bias     — доля атак на нейтральные планеты (0=только враг, 1=только нейтралы)
  ships_fraction     — средняя доля отправленных кораблей от доступных
  avg_distance_frac  — средняя нормализованная дистанция до цели
  overkill_ratio     — среднее ships_sent / target_ships
  early_rate         — attack_rate на первых 100 ходах
  late_rate          — attack_rate на последних 100 ходах
  multi_launch_rate  — доля ходов с >1 одновременным запуском
  idle_peak_frac     — пик доли кораблей игрока без атаки

Запуск:
  python3 opponent_learning/scripts/02_extract_features.py [--limit N]
"""

import json
import math
import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

HERE     = Path(__file__).parent
RAW_DIR  = HERE.parent / "data" / "raw"
PROC_DIR = HERE.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

EARLY_CUTOFF = 100
LATE_WINDOW  = 100

# ── Planet helpers (формат: список [id, owner, x, y, radius, ships, prod]) ──

def _parse_planet(p):
    """Нормализует планету в dict независимо от формата (list или dict)."""
    if isinstance(p, list) and len(p) >= 7:
        return {'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3],
                'radius': p[4], 'ships': p[5], 'production': p[6]}
    if isinstance(p, dict):
        return p
    return None


def _planet_map(planets):
    """id → planet dict."""
    out = {}
    for p in planets:
        pd_ = _parse_planet(p)
        if pd_ and pd_.get('id') is not None:
            out[pd_['id']] = pd_
    return out


def _max_dist(planets):
    """Максимальная попарная дистанция — для нормализации."""
    coords = []
    for p in planets:
        pd_ = _parse_planet(p)
        if pd_:
            coords.append((pd_['x'], pd_['y']))
    if len(coords) < 2:
        return 1.0
    md = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = math.hypot(coords[i][0] - coords[j][0],
                           coords[i][1] - coords[j][1])
            md = max(md, d)
    return max(md, 1.0)


def _find_target_by_angle(src: dict, angle: float, pl_map: dict) -> dict | None:
    """Определяет цель атаки по углу запуска.

    Вычисляем угол от src до каждой другой планеты,
    выбираем ту у которой минимальная угловая разница с aim_angle
    (и которая находится «впереди» — dot > 0).
    """
    sx, sy = src['x'], src['y']
    dx_aim = math.cos(angle)
    dy_aim = math.sin(angle)

    best, best_score = None, float('inf')
    for pid, tgt in pl_map.items():
        if pid == src['id']:
            continue
        tdx = tgt['x'] - sx
        tdy = tgt['y'] - sy
        dist = math.hypot(tdx, tdy)
        if dist < 1e-6:
            continue
        # dot product — цель должна быть "впереди"
        dot = (dx_aim * tdx + dy_aim * tdy) / dist
        if dot < 0.2:
            continue
        # угловая разница
        t_angle  = math.atan2(tdy, tdx)
        ang_diff = abs(math.atan2(
            math.sin(angle - t_angle),
            math.cos(angle - t_angle)
        ))
        if ang_diff < best_score:
            best_score = ang_diff
            best = tgt
    return best


# ── Action parsing ────────────────────────────────────────────────────────────

def _parse_action(action_raw):
    """action может быть list of lists или list of dicts.

    Формат orbit_wars: [[source_planet_id, aim_angle, ships], ...]
    Возвращает list of dict с ключами: source_id, angle, ships
    """
    result = []
    if not isinstance(action_raw, list):
        return result
    for a in action_raw:
        if isinstance(a, (list, tuple)) and len(a) >= 3:
            try:
                result.append({
                    'source_id': int(a[0]),
                    'angle':     float(a[1]),
                    'ships':     int(a[2]),
                })
            except (TypeError, ValueError):
                pass
        elif isinstance(a, dict):
            # на случай альтернативного формата
            if 'source_id' in a and 'angle' in a:
                result.append(a)
            elif 'from_id' in a and 'target_id' in a:
                result.append({
                    'source_id': a['from_id'],
                    'angle':     None,
                    'ships':     a.get('ships', 0),
                    '_target_id': a['target_id'],
                })
    return result


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_from_replay(episode_id: int, data: dict) -> list:
    """Возвращает list[dict] — по одной записи на (episode_id, player_idx)."""
    steps = data.get('steps', [])
    if not steps:
        return []

    n_steps   = len(steps)
    n_players = len(steps[0]) if steps else 0
    if n_players < 2:
        return []

    # Финальные rewards
    final_rewards = {}
    for pidx in range(n_players):
        try:
            final_rewards[pidx] = float(steps[-1][pidx].get('reward') or 0)
        except (IndexError, TypeError):
            final_rewards[pidx] = 0.0

    # Максимальная дистанция из первого шага с планетами
    max_d = 1.0
    for step in steps[:5]:
        for pd_ in step:
            if isinstance(pd_, dict):
                obs = pd_.get('observation', {})
                pl  = obs.get('planets', []) if isinstance(obs, dict) else []
                if pl and _max_dist(pl) > 1.0:
                    max_d = _max_dist(pl)
                    break
        if max_d > 1.0:
            break

    # Per-player накопители
    records = {pidx: {
        'episode_id':       episode_id,
        'player_idx':       pidx,
        'n_steps':          n_steps,
        'reward':           final_rewards.get(pidx, 0),
        '_turns_active':    0,
        '_turns_total':     0,
        '_attacks_neutral': 0,
        '_attacks_enemy':   0,
        '_ships_frac_list': [],
        '_dist_frac_list':  [],
        '_overkill_list':   [],
        '_early_active':    0,
        '_early_total':     0,
        '_late_active':     0,
        '_late_total':      0,
        '_multi_launches':  0,
        '_idle_ship_frac':  [],
    } for pidx in range(n_players)}

    for step_idx, step in enumerate(steps):
        if not isinstance(step, list):
            continue

        # Берём планеты из первого игрока шага (одинаковы для всех)
        all_pl = []
        for sp in step:
            if isinstance(sp, dict):
                obs = sp.get('observation', {})
                pl  = obs.get('planets', []) if isinstance(obs, dict) else []
                if pl:
                    all_pl = pl
                    break

        pl_map = _planet_map(all_pl)

        # Суммарные корабли по owner
        total_ships_by_owner: dict = {}
        for p in pl_map.values():
            o = p.get('owner', -1)
            total_ships_by_owner[o] = (
                total_ships_by_owner.get(o, 0) + max(0, p.get('ships', 0))
            )
        all_ships = sum(total_ships_by_owner.values())

        for pidx, player_data in enumerate(step):
            if not isinstance(player_data, dict):
                continue
            if pidx not in records:
                continue

            rec    = records[pidx]
            action_raw = player_data.get('action', [])
            launches   = _parse_action(action_raw)

            own_ships = total_ships_by_owner.get(pidx, 0)

            rec['_turns_total'] += 1
            is_early = step_idx < EARLY_CUTOFF
            is_late  = step_idx >= (n_steps - LATE_WINDOW)
            if is_early:
                rec['_early_total'] += 1
            if is_late:
                rec['_late_total'] += 1

            if launches:
                rec['_turns_active'] += 1
                if is_early:
                    rec['_early_active'] += 1
                if is_late:
                    rec['_late_active'] += 1
                if len(launches) > 1:
                    rec['_multi_launches'] += 1

                total_sent = sum(max(0, l['ships']) for l in launches)
                avail = max(1, own_ships + total_sent)
                rec['_ships_frac_list'].append(min(1.0, total_sent / avail))

                for l in launches:
                    src = pl_map.get(l['source_id'])
                    if src is None:
                        continue

                    # Определяем цель
                    tgt = None
                    if l.get('_target_id') is not None:
                        tgt = pl_map.get(l['_target_id'])
                    elif l.get('angle') is not None:
                        tgt = _find_target_by_angle(src, l['angle'], pl_map)

                    if tgt:
                        # Дистанция
                        d = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
                        rec['_dist_frac_list'].append(d / max_d)

                        # Тип цели
                        tgt_owner = tgt.get('owner', -1)
                        if tgt_owner == -1 or tgt_owner is None:
                            rec['_attacks_neutral'] += 1
                        elif tgt_owner != pidx:
                            rec['_attacks_enemy'] += 1
                        # else: own planet = reinforce, не считаем

                        # Overkill
                        tgt_ships = max(1.0, tgt.get('ships', 1))
                        rec['_overkill_list'].append(l['ships'] / tgt_ships)

            else:
                # Idle: игрок не атакует
                if all_ships > 0 and own_ships > 0:
                    rec['_idle_ship_frac'].append(own_ships / all_ships)

    # ── Финализируем ─────────────────────────────────────────────────────────
    out = []
    for pidx, rec in records.items():
        T = max(1, rec['_turns_total'])
        a = max(0, rec['_turns_active'])
        total_attacks = rec['_attacks_neutral'] + rec['_attacks_enemy']

        n_sf   = len(rec['_ships_frac_list'])
        n_df   = len(rec['_dist_frac_list'])
        n_ok   = len(rec['_overkill_list'])
        n_idle = len(rec['_idle_ship_frac'])

        row = {
            'episode_id':        episode_id,
            'player_idx':        pidx,
            'n_steps':           rec['n_steps'],
            'reward':            rec['reward'],

            'attack_rate':       a / T,
            'expansion_bias':    rec['_attacks_neutral'] / max(1, total_attacks),
            'ships_fraction':    sum(rec['_ships_frac_list']) / n_sf if n_sf else 0.0,
            'avg_distance_frac': sum(rec['_dist_frac_list']) / n_df if n_df else 0.5,
            'overkill_ratio':    sum(rec['_overkill_list']) / n_ok if n_ok else 1.0,
            'early_rate':        rec['_early_active'] / max(1, rec['_early_total']),
            'late_rate':         rec['_late_active'] / max(1, rec['_late_total']),
            'multi_launch_rate': rec['_multi_launches'] / max(1, a),
            'idle_peak_frac':    max(rec['_idle_ship_frac']) if n_idle else 0.5,
        }
        out.append(row)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',   default=str(RAW_DIR))
    parser.add_argument('--out',   default=str(PROC_DIR / 'features.csv'))
    parser.add_argument('--limit', type=int, default=0,
                        help='Обработать только первые N файлов (0=все)')
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    files   = sorted(raw_dir.glob('[0-9]*.json'))
    if args.limit:
        files = files[:args.limit]

    print(f"Найдено {len(files)} replay-файлов в {raw_dir}")

    all_rows, errors = [], 0
    for i, fpath in enumerate(files, 1):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rows = extract_features_from_replay(int(fpath.stem), data)
            all_rows.extend(rows)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ⚠ {fpath.name}: {e}")

        if i % 100 == 0 or i == len(files):
            print(f"  [{i:>4}/{len(files)}]  строк: {len(all_rows)}  ошибок: {errors}")

    if not all_rows:
        print("Нет данных.")
        return

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    print(f"\n✓ Сохранено: {out_path}")
    print(f"  Эпизодов: {df['episode_id'].nunique()}  "
          f"Записей: {len(df)}  Ошибок: {errors}")

    feat_cols = ['attack_rate','expansion_bias','ships_fraction',
                 'avg_distance_frac','overkill_ratio',
                 'early_rate','late_rate','multi_launch_rate','idle_peak_frac']
    print("\nСтатистика фич:")
    print(df[feat_cols].describe().round(3).to_string())

    # Проверка: сколько записей с ненулевым expansion_bias
    nonzero_eb = (df['expansion_bias'] > 0).sum()
    nonzero_dist = (df['avg_distance_frac'] != 0.5).sum()
    print(f"\nСанити-чек:")
    print(f"  expansion_bias > 0:     {nonzero_eb}/{len(df)} записей")
    print(f"  avg_distance_frac ≠ 0.5: {nonzero_dist}/{len(df)} записей")


if __name__ == '__main__':
    main()
