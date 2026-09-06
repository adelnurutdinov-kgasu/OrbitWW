#!/usr/bin/env python3
"""
13_state_reconstruction_debug.py — почему наш simulate_step не точно
воспроизводит реальность в активных шагах с launches.

КОНТЕКСТ
────────
12_pro_vs_heuristic.py показал:
  - Mean owner match:    0.996  (хорошо)
  - Mean ship match:     0.731  (27% планет с неверным ship count)
  - Fleet count match:   0.22   (катастрофически плохо)
  - Perfect state match: 0/59   (никогда)

В 04_diagnostics было 96% совпадение на random шагах 1v1. Здесь — на
шагах с launches (которые смещены к активным моментам, и FFA dominates) —
сильно хуже. Похоже наш simulate_step расходится на multi-launch / FFA combat.

ЧТО ДЕЛАЕМ
──────────
Берём sample launches, применяем все pro-actions, и для **тех** где
реконструкция плохая — распечатываем подробный diff: какие планеты
расходятся, какие флоты пропущены/лишние, в чём паттерн.

ВЫХОД
─────
  rebuild/reports/state_recon_debug.md  — отчёт с примерами
  rebuild/data/state_recon_diffs.csv    — детали по 200+ launches
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
PROC_DIR   = OL_DIR / "data" / "processed"
RAW_DIR    = OL_DIR / "data" / "raw"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LAUNCHES_CSV = PROC_DIR / "launches.csv"
DIFFS_OUT    = DATA_DIR / "state_recon_diffs.csv"
REPORT_MD    = REPORT_DIR / "state_recon_debug.md"

AGENT_DIR = OL_DIR.parent / "agent_bundle_swarm 2"
sys.path.insert(0, str(AGENT_DIR))
from orbit_sim import GameState, simulate_step


def diff_states(obs_now: dict, all_actions: list, obs_next: dict,
                step: int, n_players: int) -> dict:
    """Симулируем next state, считаем разницу с реальным."""
    state = GameState.from_kaggle_obs(obs_now, step=step)
    state.n_players = n_players
    moves = {i: (a if isinstance(a, list) else [])
             for i, a in enumerate(all_actions)}
    try:
        predicted = simulate_step(state, moves)
    except Exception as e:
        return {'error': repr(e)[:100]}

    real = GameState.from_kaggle_obs(obs_next, step=step+1)
    p_by_id = {p.id: p for p in predicted.planets}
    r_by_id = {p.id: p for p in real.planets}
    common = set(p_by_id) & set(r_by_id)

    # Per-planet diffs
    planet_diffs = []
    for pid in common:
        pp, rp = p_by_id[pid], r_by_id[pid]
        if pp.owner != rp.owner or pp.ships != rp.ships:
            planet_diffs.append({
                'pid': pid,
                'predicted_owner': pp.owner, 'real_owner': rp.owner,
                'predicted_ships': pp.ships, 'real_ships': rp.ships,
                'ship_diff': pp.ships - rp.ships,
                'orbital_predicted': (math.hypot(pp.x-50, pp.y-50) + pp.radius) <= 30,
            })

    # Fleet diffs
    pred_fleets = [(f.owner, f.ships, round(f.angle, 2)) for f in predicted.fleets]
    real_fleets = [(f.owner, f.ships, round(f.angle, 2)) for f in real.fleets]
    pred_set = Counter(pred_fleets)
    real_set = Counter(real_fleets)
    extra_in_pred = [k for k in pred_set if pred_set[k] > real_set.get(k, 0)]
    missing_in_pred = [k for k in real_set if real_set[k] > pred_set.get(k, 0)]

    n_total_launches = sum(len(a) if isinstance(a, list) else 0 for a in all_actions)
    return {
        'n_planets':           len(common),
        'n_planet_diffs':      len(planet_diffs),
        'pct_planets_match':   round(1 - len(planet_diffs)/max(len(common),1), 4),
        'n_pred_fleets':       len(pred_fleets),
        'n_real_fleets':       len(real_fleets),
        'fleet_count_match':   len(pred_fleets) == len(real_fleets),
        'n_extra_pred_fleets': len(extra_in_pred),
        'n_missing_pred_fleets': len(missing_in_pred),
        'n_total_launches':    n_total_launches,
        'n_active_players':    sum(1 for a in all_actions if isinstance(a, list) and len(a) > 0),
        'planet_diffs':        planet_diffs,
        'extra_in_pred':       extra_in_pred[:5],
        'missing_in_pred':     missing_in_pred[:5],
    }


def main():
    print(f'Loading {LAUNCHES_CSV} …')
    launches = pd.read_csv(LAUNCHES_CSV,
                            usecols=['episode_id', 'player_idx', 'step', 'launch_idx', 'phase'])
    launches['episode_id'] = launches['episode_id'].astype(str)

    # Берём sample 400 случайных
    rng = random.Random(42)
    launches = launches.sample(400, random_state=42).reset_index(drop=True)
    print(f'  sample: {len(launches)}')

    results = []
    examples = []  # для отчёта — конкретные примеры
    t0 = time.time()

    by_ep = launches.groupby('episode_id')
    for eid, group in by_ep:
        json_path = RAW_DIR / f'{eid}.json'
        if not json_path.exists():
            continue
        with open(json_path) as f:
            ep = json.load(f)
        steps = ep.get('steps') or []
        n_players = len(steps[0]) if steps and isinstance(steps[0], list) else 2

        for _, row in group.iterrows():
            step = int(row['step'])
            if step >= len(steps) - 1:
                continue
            s_now, s_next = steps[step], steps[step+1]
            if not (isinstance(s_now, list) and isinstance(s_next, list)):
                continue
            obs_now = (s_now[0] or {}).get('observation') or {}
            obs_next = (s_next[0] or {}).get('observation') or {}
            if not obs_now.get('planets') or not obs_next.get('planets'):
                continue
            # ВАЖНО: action который привёл к obs[t+1] хранится в JSON в steps[t+1].action,
            # не в steps[t].action (см. 04_diagnostics smoke-test где это эмпирически найдено)
            all_actions = [(s_next[pid] or {}).get('action') or [] for pid in range(n_players)]

            d = diff_states(obs_now, all_actions, obs_next, step, n_players)
            if 'error' in d:
                continue
            d['episode_id'] = eid
            d['step'] = step
            d['n_players'] = n_players
            d['phase'] = float(row['phase'])
            results.append({k: v for k, v in d.items()
                            if not isinstance(v, list)})  # для csv без listов
            if d['n_planet_diffs'] > 0 or not d['fleet_count_match']:
                examples.append(d)

    df = pd.DataFrame(results)
    df.to_csv(DIFFS_OUT, index=False)
    print(f'\n✓ Сохранено: {DIFFS_OUT}  ({len(df)} проверок за {time.time()-t0:.1f}s)')

    print('\n' + '═'*70)
    print('СВОДКА')
    print('═'*70)
    print(f'\n  Проверок всего: {len(df)}')
    print(f'  Perfect state match (planets + fleets):   '
          f'{((df["n_planet_diffs"] == 0) & df["fleet_count_match"]).sum()}/{len(df)}')
    print(f'  Planets perfect (n_planet_diffs == 0):    '
          f'{(df["n_planet_diffs"] == 0).sum()}/{len(df)}  '
          f'({100*(df["n_planet_diffs"]==0).mean():.1f}%)')
    print(f'  Fleets perfect (fleet_count_match):       '
          f'{df["fleet_count_match"].sum()}/{len(df)}  '
          f'({100*df["fleet_count_match"].mean():.1f}%)')

    print(f'\n  Распределение n_planet_diffs:')
    print(df['n_planet_diffs'].describe().round(2).to_string())
    print(f'\n  Распределение n_extra_pred_fleets (наши лишние флоты):')
    print(df['n_extra_pred_fleets'].describe().round(2).to_string())
    print(f'\n  Распределение n_missing_pred_fleets (мы пропустили флоты):')
    print(df['n_missing_pred_fleets'].describe().round(2).to_string())

    # По числу launches на шаге
    print(f'\n  Совпадение по n_total_launches на шаге:')
    by_n = df.groupby('n_total_launches').agg(
        n=('fleet_count_match', 'count'),
        fleet_ok=('fleet_count_match', 'mean'),
        planets_ok=('n_planet_diffs', lambda x: (x == 0).mean()),
    ).round(3)
    print(by_n.head(15).to_string())

    # По n_players
    print(f'\n  Совпадение по n_players:')
    by_np = df.groupby('n_players').agg(
        n=('fleet_count_match', 'count'),
        fleet_ok=('fleet_count_match', 'mean'),
        planets_ok=('n_planet_diffs', lambda x: (x == 0).mean()),
    ).round(3)
    print(by_np.to_string())

    # ── Конкретные примеры из failures ───────────────────────────────
    print('\n' + '═'*70)
    print('ПРИМЕРЫ FAILURE CASES')
    print('═'*70)
    rng.shuffle(examples)
    for i, ex in enumerate(examples[:5]):
        print(f'\n--- Example {i+1}: {ex["episode_id"]} step={ex["step"]} '
              f'n_players={ex["n_players"]} ---')
        print(f'  total_launches_on_step: {ex["n_total_launches"]}, '
              f'active_players: {ex["n_active_players"]}')
        print(f'  planets diff: {ex["n_planet_diffs"]}/{ex["n_planets"]}, '
              f'fleets pred/real: {ex["n_pred_fleets"]}/{ex["n_real_fleets"]}')
        if ex['planet_diffs']:
            print(f'  planet diffs (first 3):')
            for pd_ in ex['planet_diffs'][:3]:
                orb = ' ORBITAL' if pd_['orbital_predicted'] else ''
                print(f'    p{pd_["pid"]:>3}{orb}: '
                      f'pred owner={pd_["predicted_owner"]} ships={pd_["predicted_ships"]} | '
                      f'real owner={pd_["real_owner"]} ships={pd_["real_ships"]} '
                      f'(diff={pd_["ship_diff"]:+d})')
        if ex['extra_in_pred']:
            print(f'  extra fleets in pred: {ex["extra_in_pred"]}')
        if ex['missing_in_pred']:
            print(f'  missing fleets in pred: {ex["missing_in_pred"]}')

    # ── Markdown отчёт ────────────────────────────────────────────────
    md = []
    md.append('# State Reconstruction Debug\n\n')
    md.append(f'**Проверок**: {len(df)} launches\n\n')
    md.append('## Главные числа\n\n')
    md.append('| Metric | Value |\n|---|---|\n')
    md.append(f'| Planets perfect | {(df["n_planet_diffs"]==0).sum()}/{len(df)} '
              f'({100*(df["n_planet_diffs"]==0).mean():.1f}%) |\n')
    md.append(f'| Fleets count perfect | {df["fleet_count_match"].sum()}/{len(df)} '
              f'({100*df["fleet_count_match"].mean():.1f}%) |\n')
    md.append(f'| State fully perfect | {((df["n_planet_diffs"]==0) & df["fleet_count_match"]).sum()}/{len(df)} |\n')

    md.append('\n## По числу launches на шаге\n\n')
    md.append(by_n.head(15).to_markdown())

    md.append('\n\n## По n_players\n\n')
    md.append(by_np.to_markdown())

    md.append('\n\n## Примеры failure cases\n\n')
    for i, ex in enumerate(examples[:5]):
        md.append(f'\n### Example {i+1}: {ex["episode_id"]} step={ex["step"]}\n\n')
        md.append(f'- n_players: {ex["n_players"]}\n')
        md.append(f'- launches на шаге: {ex["n_total_launches"]} '
                  f'({ex["n_active_players"]} активных игроков)\n')
        md.append(f'- planets diff: {ex["n_planet_diffs"]}/{ex["n_planets"]}\n')
        md.append(f'- fleets pred/real: {ex["n_pred_fleets"]}/{ex["n_real_fleets"]}\n')
        if ex['planet_diffs']:
            md.append('\nPlanet diffs:\n```\n')
            for pd_ in ex['planet_diffs'][:5]:
                orb = ' ORBITAL' if pd_['orbital_predicted'] else ''
                md.append(f'  p{pd_["pid"]:>3}{orb}: '
                          f'pred owner={pd_["predicted_owner"]} ships={pd_["predicted_ships"]} '
                          f'| real owner={pd_["real_owner"]} ships={pd_["real_ships"]}\n')
            md.append('```\n')

    REPORT_MD.write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_MD}')


if __name__ == '__main__':
    main()
