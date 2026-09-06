#!/usr/bin/env python3
"""
14_perspective_analysis.py — три задачи в одном скрипте:

A. ПРАВИЛЬНАЯ ОЦЕНКА PROJECTION
   Сравнение projection не с "реальностью через PROJECTION_HORIZON шагов"
   (там новые launches искажают), а с реальностью **в момент projected_at
   каждой конкретной планеты**. Если projection говорит "планета сменит
   владельца на шаге t+5" — проверяем реальный owner в obs at t+5.
   Также: точность через 1, 3, 5, 10 фиксированных горизонтов.

B. СИСТЕМАТИЧЕСКИЕ ПЕРЕКОСЫ ПРИОРИТЕТОВ
   Для launches где pro_target != heur_top1 — какие свойства pro_target
   отличаются от heur_top1? Профи берёт дальние цели? Высокопроизводственные?
   Вражеские против нейтральных? Усреднённая разница даёт направление
   "перекос" приоритетов профи vs эвристики.

C. ЧТО ЭВРИСТИКА УПУСКАЕТ (heur_idle)
   В 31% случаев профи стрельнул, эвристика бы — нет. Что в этих ситуациях
   характерно? Phase? Resource ratio? Тип pro-атаки?

ВХОД
────
  opponent_learning/data/processed/launches.csv
  opponent_learning/data/raw/*.json

ВЫХОД
─────
  rebuild/data/perspective_*.csv
  rebuild/reports/perspective_analysis.md
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict, Counter
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

AGENT_DIR = OL_DIR.parent / "agent_bundle_swarm 2"
sys.path.insert(0, str(AGENT_DIR))

import agent as agent_mod
agent_mod.USE_MCTS = False
from agent import agent
from zones import compute_zones_from_state
from orbit_sim import GameState
from projection import project_state, PROJECTION_HORIZON, NEUTRAL_OWNER


# ──────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────

def find_target_planet(src_x, src_y, angle, planets, exclude_src_id=None):
    dx, dy = math.cos(angle), math.sin(angle)
    best_id, best_ang_dev = None, float('inf')
    for p in planets:
        pid = p[0]
        if pid == exclude_src_id:
            continue
        tdx, tdy = p[2] - src_x, p[3] - src_y
        dist = math.hypot(tdx, tdy)
        if dist < 1e-6:
            continue
        if (dx * tdx + dy * tdy) / dist < 0.3:
            continue
        ang = math.atan2(tdy, tdx)
        ad = abs(math.atan2(math.sin(angle - ang), math.cos(angle - ang)))
        if ad < best_ang_dev:
            best_ang_dev = ad
            best_id = pid
    return best_id


def classify(player_idx, owner):
    if owner == NEUTRAL_OWNER or owner == -1:
        return 'neutral'
    if owner == player_idx:
        return 'reinforce'
    return 'enemy'


def planet_dist_normalized(src_p, tgt_p, board_diag=100*math.sqrt(2)):
    return math.hypot(src_p[2] - tgt_p[2], src_p[3] - tgt_p[3]) / board_diag


# ──────────────────────────────────────────────────────────────────────────
# A. PROJECTION CORRECTNESS
# ──────────────────────────────────────────────────────────────────────────

def evaluate_projection(obs_t: dict, steps: list, t: int, player_idx: int) -> dict:
    """Сравнить projected состояние с реальностью через 1/3/5/10 шагов +
    adaptive horizon (в момент projected_at каждой планеты)."""
    try:
        state = GameState.from_kaggle_obs(obs_t, step=t)
        projected = project_state(state, horizon=PROJECTION_HORIZON, player=player_idx)
    except Exception:
        return {}
    proj_at = getattr(projected, 'projected_at', {}) or {}
    proj_by_id = {p.id: p for p in projected.planets}

    # Fixed horizons
    horizons = [1, 3, 5, 10]
    fixed_results = {h: {'correct': 0, 'total': 0} for h in horizons}
    for h in horizons:
        if t + h >= len(steps):
            continue
        s_h = steps[t + h]
        if not isinstance(s_h, list) or player_idx >= len(s_h):
            continue
        obs_h = (s_h[player_idx] or {}).get('observation') or {}
        real_planets = obs_h.get('planets', []) or []
        if not real_planets:
            continue
        real_by_id = {p[0]: p for p in real_planets}
        for pid, pp in proj_by_id.items():
            if pid in real_by_id:
                fixed_results[h]['total'] += 1
                if pp.owner == real_by_id[pid][1]:
                    fixed_results[h]['correct'] += 1

    # Adaptive horizon: для каждой планеты с projected_at > 0 проверяем
    # реальность в момент именно projected_at
    adaptive_correct, adaptive_total = 0, 0
    adaptive_active_only_correct, adaptive_active_total = 0, 0
    for pid, pp in proj_by_id.items():
        pat = int(proj_at.get(pid, 0))
        adaptive_total += 1
        # Если projected_at = 0 — планета не менялась в projection, должна совпасть с obs_t
        if pat == 0:
            obs_t_planets = {p[0]: p for p in obs_t.get('planets', [])}
            if pid in obs_t_planets and obs_t_planets[pid][1] == pp.owner:
                adaptive_correct += 1
            continue
        # projected_at > 0: смотрим obs в момент t+pat
        target_step = t + pat
        if target_step >= len(steps):
            continue
        s_pat = steps[target_step]
        if not isinstance(s_pat, list) or player_idx >= len(s_pat):
            continue
        obs_pat = (s_pat[player_idx] or {}).get('observation') or {}
        real_planets = {p[0]: p for p in obs_pat.get('planets', [])}
        if pid in real_planets:
            adaptive_active_total += 1
            if real_planets[pid][1] == pp.owner:
                adaptive_correct += 1
                adaptive_active_only_correct += 1

    return {
        'fixed_h1_correct':   fixed_results[1]['correct'],
        'fixed_h1_total':     fixed_results[1]['total'],
        'fixed_h3_correct':   fixed_results[3]['correct'],
        'fixed_h3_total':     fixed_results[3]['total'],
        'fixed_h5_correct':   fixed_results[5]['correct'],
        'fixed_h5_total':     fixed_results[5]['total'],
        'fixed_h10_correct':  fixed_results[10]['correct'],
        'fixed_h10_total':    fixed_results[10]['total'],
        'adaptive_correct':   adaptive_correct,
        'adaptive_total':     adaptive_total,
        'adaptive_active_correct': adaptive_active_only_correct,
        'adaptive_active_total':   adaptive_active_total,
    }


# ──────────────────────────────────────────────────────────────────────────
# B+C. Сбор фич pro_target и heur_top1 для каждого launch
# ──────────────────────────────────────────────────────────────────────────

def collect_features(obs: dict, all_actions: list, action_launch: list,
                     player_idx: int) -> dict | None:
    if not isinstance(action_launch, list) or len(action_launch) != 3:
        return None
    pro_src_id, pro_angle, pro_ships = int(action_launch[0]), float(action_launch[1]), int(action_launch[2])
    planets_raw = obs.get('planets', [])
    if not planets_raw:
        return None
    src_p = next((p for p in planets_raw if p[0] == pro_src_id), None)
    if not src_p:
        return None
    pro_target_id = find_target_planet(src_p[2], src_p[3], pro_angle,
                                        planets_raw, exclude_src_id=pro_src_id)
    if pro_target_id is None:
        return None
    pro_target_p = next((p for p in planets_raw if p[0] == pro_target_id), None)
    if not pro_target_p:
        return None

    # Запускаем эвристику
    try:
        heur_moves = agent(dict(obs))
    except Exception:
        heur_moves = []

    # zones для получения priorities и features
    try:
        state = GameState.from_kaggle_obs(obs)
        state_proj = project_state(state, horizon=PROJECTION_HORIZON, player=player_idx)
        df_zones, _ = compute_zones_from_state(state_proj, player=player_idx)
    except Exception:
        return None

    df_targets = df_zones[df_zones['owner'] != player_idx].copy()
    df_targets = df_targets.sort_values('priority', ascending=False).reset_index(drop=True)

    pro_zone_row = df_zones[df_zones['pid'] == pro_target_id]
    pro_features = pro_zone_row.iloc[0].to_dict() if len(pro_zone_row) else {}

    # heur_top1
    heur_top_id = int(df_targets.iloc[0]['pid']) if len(df_targets) else None
    heur_top_features = df_targets.iloc[0].to_dict() if len(df_targets) else {}

    # pro rank
    pro_rank = None
    if pro_target_id in df_targets['pid'].values:
        pro_rank = int(df_targets.index[df_targets['pid'] == pro_target_id][0]) + 1

    pro_dist = planet_dist_normalized(src_p, pro_target_p)
    heur_top_dist = None
    if heur_top_id is not None and heur_top_id != pro_target_id:
        ht_p = next((p for p in planets_raw if p[0] == heur_top_id), None)
        if ht_p:
            # эвристика бы выбрала какой source? возьмём ближайший
            our_planets = [p for p in planets_raw if p[1] == player_idx]
            if our_planets:
                heur_top_dist = min(planet_dist_normalized(p, ht_p) for p in our_planets)

    return {
        'pro_target_id':     pro_target_id,
        'pro_target_owner':  int(pro_target_p[1]),
        'pro_target_ships':  int(pro_target_p[5]),
        'pro_target_prod':   int(pro_target_p[6]),
        'pro_target_priority': float(pro_features.get('priority', 0.0)) if pro_features else None,
        'pro_target_zone':   str(pro_features.get('zone', '')) if pro_features else None,
        'pro_target_dist_frac': pro_dist,
        'pro_action_type':   classify(player_idx, int(pro_target_p[1])),
        'pro_rank':          pro_rank,

        'heur_top_id':       heur_top_id,
        'heur_top_owner':    int(heur_top_features.get('owner', 0)) if heur_top_features else None,
        'heur_top_ships':    int(heur_top_features.get('ships', 0)) if heur_top_features else None,
        'heur_top_prod':     int(heur_top_features.get('prod', 0)) if heur_top_features else None,
        'heur_top_priority': float(heur_top_features.get('priority', 0.0)) if heur_top_features else None,
        'heur_top_zone':     str(heur_top_features.get('zone', '')) if heur_top_features else None,
        'heur_top_dist_frac': heur_top_dist,

        'heur_n_moves':      len(heur_moves) if isinstance(heur_moves, list) else 0,
        'heur_idle':         not heur_moves,

        'targets_match':     heur_top_id == pro_target_id,
        'pro_ships_sent':    pro_ships,
        'pro_src_owner_ships_total': sum(p[5] for p in planets_raw if p[1] == player_idx),
        'pro_src_ships':     int(src_p[5]),
        'pro_src_prod':      int(src_p[6]),
    }


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f'Loading {LAUNCHES_CSV} …')
    launches = pd.read_csv(LAUNCHES_CSV,
                            usecols=['episode_id', 'player_idx', 'step', 'launch_idx', 'phase', 'reward'])
    launches['episode_id'] = launches['episode_id'].astype(str)
    launches = launches.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    print(f'  sample: {len(launches)}')

    proj_records = []
    feat_records = []
    t0 = time.time()
    processed = 0
    by_ep = launches.groupby('episode_id')

    for eid, group in by_ep:
        json_path = RAW_DIR / f'{eid}.json'
        if not json_path.exists():
            continue
        try:
            with open(json_path) as f:
                ep = json.load(f)
        except Exception:
            continue
        steps = ep.get('steps') or []
        n_players = len(steps[0]) if steps and isinstance(steps[0], list) else 2

        for _, row in group.iterrows():
            step = int(row['step'])
            player_idx = int(row['player_idx'])
            launch_idx = int(row['launch_idx'])
            if step >= len(steps) - 1:
                continue
            s_now, s_next = steps[step], steps[step + 1]
            if not (isinstance(s_now, list) and isinstance(s_next, list)):
                continue
            obs = (s_now[player_idx] or {}).get('observation') or {}
            if not obs.get('planets'):
                continue
            actions = (s_next[player_idx] or {}).get('action') or []
            if launch_idx >= len(actions):
                continue
            pro_action = actions[launch_idx]

            # A: projection (только каждый 5-й launch чтобы не замедлять)
            if processed % 5 == 0:
                pr = evaluate_projection(obs, steps, step, player_idx)
                if pr:
                    pr['episode_id'] = eid
                    pr['step'] = step
                    pr['phase'] = float(row['phase'])
                    pr['n_players'] = n_players
                    proj_records.append(pr)

            # B+C: features
            try:
                feats = collect_features(obs, actions, pro_action, player_idx)
            except Exception:
                feats = None
            if feats is not None:
                feats['episode_id'] = eid
                feats['step'] = step
                feats['phase'] = float(row['phase'])
                feats['n_players'] = n_players
                feats['launch_idx'] = launch_idx
                feat_records.append(feats)

            processed += 1
            if processed % 100 == 0:
                print(f'  {processed}/{len(launches)} ({time.time()-t0:.1f}s)')

    proj_df = pd.DataFrame(proj_records)
    feat_df = pd.DataFrame(feat_records)
    proj_df.to_csv(DATA_DIR / 'perspective_projection.csv', index=False)
    feat_df.to_csv(DATA_DIR / 'perspective_features.csv', index=False)
    print(f'\n✓ Сохранено: projection={len(proj_df)}, features={len(feat_df)}')

    # ════════════════════════════════════════════════════════════════════
    # A. PROJECTION CORRECTNESS
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '═'*70)
    print('A. PROJECTION ACCURACY')
    print('═'*70)
    if len(proj_df) > 0:
        for h in [1, 3, 5, 10]:
            c = proj_df[f'fixed_h{h}_correct'].sum()
            t = proj_df[f'fixed_h{h}_total'].sum()
            if t > 0:
                print(f'  fixed horizon={h:>2}:  '
                      f'{c}/{t} = {100*c/t:.2f}% owner accuracy')
        c = proj_df['adaptive_correct'].sum()
        t = proj_df['adaptive_total'].sum()
        if t > 0:
            print(f'  adaptive (в момент projected_at):  '
                  f'{c}/{t} = {100*c/t:.2f}% (включая planet с pat=0)')
        c = proj_df['adaptive_active_correct'].sum()
        t = proj_df['adaptive_active_total'].sum()
        if t > 0:
            print(f'  adaptive active only (pat>0):  '
                  f'{c}/{t} = {100*c/t:.2f}% (только активные projections)')

    # ════════════════════════════════════════════════════════════════════
    # B. ПЕРЕКОСЫ ПРИОРИТЕТОВ
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '═'*70)
    print('B. ПЕРЕКОСЫ ПРИОРИТЕТОВ — pro_target vs heur_top1')
    print('═'*70)
    if len(feat_df) > 0:
        disagree = feat_df[~feat_df['targets_match']].copy()
        print(f'\n  Disagreements: {len(disagree)} / {len(feat_df)} ({100*len(disagree)/len(feat_df):.1f}%)')

        # Mean features comparison
        pro_cols  = ['pro_target_ships', 'pro_target_prod',
                     'pro_target_dist_frac', 'pro_target_priority']
        heur_cols = ['heur_top_ships', 'heur_top_prod',
                     'heur_top_dist_frac', 'heur_top_priority']
        short_names = ['ships', 'prod', 'dist_frac', 'priority']
        pro_mean  = disagree[pro_cols].mean()
        heur_mean = disagree[heur_cols].mean()
        pro_mean.index = short_names
        heur_mean.index = short_names
        compare = pd.DataFrame({
            'pro_target': pro_mean.round(3),
            'heur_top':   heur_mean.round(3),
            'diff':       (pro_mean - heur_mean).round(3),
        })
        print('\n  Средние значения фич в disagreement-cases:')
        print(compare.to_string())

        # Тип pro_target vs heur_top
        print('\n  Тип pro_target (когда heur выбирал по-другому):')
        print(disagree['pro_action_type'].value_counts(normalize=True).round(3).to_string())
        print('\n  Тип heur_top (когда сам по-другому):')
        print(disagree['heur_top_zone'].value_counts(normalize=True).round(3).to_string())

        # Профи vs эвристика по zone
        print('\n  Cross-tab: pro_target_zone × heur_top_zone:')
        ct = pd.crosstab(disagree['pro_target_zone'], disagree['heur_top_zone'], normalize='index').round(3)
        print(ct.to_string())

    # ════════════════════════════════════════════════════════════════════
    # C. ЧТО ЭВРИСТИКА УПУСКАЕТ
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '═'*70)
    print('C. ЧТО ЭВРИСТИКА УПУСКАЕТ (heur_idle)')
    print('═'*70)
    if len(feat_df) > 0:
        idle = feat_df[feat_df['heur_idle']].copy()
        active = feat_df[~feat_df['heur_idle']].copy()
        print(f'\n  heur_idle: {len(idle)} / {len(feat_df)} ({100*len(idle)/len(feat_df):.1f}%)')
        print(f'  heur_active: {len(active)}')

        print('\n  Сравнение характеристик state (idle vs active):')
        idle_active = pd.DataFrame({
            'idle':   idle[['phase', 'pro_src_ships', 'pro_target_priority',
                            'pro_rank', 'pro_target_dist_frac',
                            'pro_src_owner_ships_total']].mean(),
            'active': active[['phase', 'pro_src_ships', 'pro_target_priority',
                              'pro_rank', 'pro_target_dist_frac',
                              'pro_src_owner_ships_total']].mean(),
        }).round(3)
        idle_active['diff'] = idle_active['idle'] - idle_active['active']
        print(idle_active.to_string())

        print('\n  Тип pro-атаки когда эвристика бы idle:')
        print(idle['pro_action_type'].value_counts(normalize=True).round(3).to_string())

        print('\n  Phase distribution для heur_idle:')
        if len(idle):
            print(idle['phase'].describe().round(3).to_string())

    # ════════════════════════════════════════════════════════════════════
    # Markdown отчёт
    # ════════════════════════════════════════════════════════════════════
    md = []
    md.append('# Perspective Analysis: projection + перекос + idle\n\n')

    md.append('## A. Projection accuracy (правильное измерение)\n\n')
    if len(proj_df) > 0:
        md.append('| Horizon | n | Accuracy |\n|---|---|---|\n')
        for h in [1, 3, 5, 10]:
            c = proj_df[f'fixed_h{h}_correct'].sum()
            t = proj_df[f'fixed_h{h}_total'].sum()
            if t > 0:
                md.append(f'| fixed h={h} | {t} | {100*c/t:.1f}% |\n')
        c = proj_df['adaptive_correct'].sum()
        t = proj_df['adaptive_total'].sum()
        if t > 0:
            md.append(f'| adaptive (at projected_at) | {t} | {100*c/t:.1f}% |\n')
        c = proj_df['adaptive_active_correct'].sum()
        t = proj_df['adaptive_active_total'].sum()
        if t > 0:
            md.append(f'| adaptive active only | {t} | {100*c/t:.1f}% |\n')

    md.append('\n## B. Перекос priorities — pro_target vs heur_top\n\n')
    if len(feat_df) > 0:
        disagree = feat_df[~feat_df['targets_match']].copy()
        md.append(f'Disagreements: {len(disagree)} / {len(feat_df)} '
                  f'({100*len(disagree)/len(feat_df):.1f}%)\n\n')
        md.append('### Средние фичи в disagreement-cases\n\n')
        md.append('| Feature | mean(pro_target) | mean(heur_top) | diff |\n|---|---|---|---|\n')
        for f, ft in [('pro_target_ships', 'heur_top_ships'),
                       ('pro_target_prod', 'heur_top_prod'),
                       ('pro_target_dist_frac', 'heur_top_dist_frac'),
                       ('pro_target_priority', 'heur_top_priority')]:
            if ft in disagree.columns and f in disagree.columns:
                pv, hv = disagree[f].mean(), disagree[ft].mean()
                md.append(f'| {f.replace("pro_target_","")} | {pv:.3f} | {hv:.3f} | {pv-hv:+.3f} |\n')

        md.append('\n### Тип pro_target в disagreements\n\n```\n')
        md.append(disagree['pro_action_type'].value_counts(normalize=True).round(3).to_string())
        md.append('\n```\n')
        md.append('\n### Zone pro_target vs heur_top (crosstab, row-normalized)\n\n')
        ct = pd.crosstab(disagree['pro_target_zone'], disagree['heur_top_zone'], normalize='index').round(3)
        md.append(ct.to_markdown())

    md.append('\n\n## C. heur_idle: где эвристика пассивна\n\n')
    if len(feat_df) > 0:
        idle = feat_df[feat_df['heur_idle']].copy()
        md.append(f'heur_idle: {len(idle)} / {len(feat_df)} '
                  f'({100*len(idle)/len(feat_df):.1f}%)\n\n')

        if len(idle):
            md.append('### Тип pro-атаки когда эвристика idle\n\n```\n')
            md.append(idle['pro_action_type'].value_counts(normalize=True).round(3).to_string())
            md.append('\n```\n')

            md.append('\n### Phase distribution в idle cases\n\n```\n')
            md.append(idle['phase'].describe().round(3).to_string())
            md.append('\n```\n')

    (REPORT_DIR / 'perspective_analysis.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/perspective_analysis.md')


if __name__ == '__main__':
    main()
