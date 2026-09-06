#!/usr/bin/env python3
"""
12_pro_vs_heuristic.py — сравнение эвристики agent_bundle_swarm 2 с действиями профи.

ЦЕЛЬ
────
Перед тем как тюнить параметры эвристики, нужно понять:
  1. Насколько уже совпадает её приоритезация с выбором профи?
  2. Правильно ли мы вызываем агента (state, projection, zones)?
  3. Где она наиболее далека от профи — там и есть пространство для тюнинга.

ЧТО ДЕЛАЕМ
──────────
Для sample pro-launches:
  1. Загружаем obs из replay (что профи видел перед решением)
  2. Запускаем agent(obs) — что бы выбрала наша эвристика
  3. Запускаем compute_zones_from_state — получаем priority всех планет
  4. Сравниваем:
     • Совпадает ли pro_target с топ-1 эвристики?
     • Какую позицию в эвристическом ranking занимает pro_target?
     • Совпадает ли тип атаки (neutral/enemy/reinforce)?
     • Стрельнул ли agent на этом шаге?

Плюс sanity check на reconstruction state:
  для подвыборки — применяем pro_action к state, симулируем,
  сравниваем с реальным next obs. Если расходится — состояние нечестно собрано.

ВХОД
────
  opponent_learning/data/processed/launches.csv  (270K записей)
  opponent_learning/data/raw/*.json              (replay'и)

ВЫХОД
─────
  rebuild/data/pro_vs_heur.csv
  rebuild/reports/pro_vs_heur.md
  rebuild/reports/pro_vs_heur_*.png

СРЕДНИЙ скрипт. 1000 launches ≈ 3-5 минут (зависит от MCTS).
USE_MCTS отключаем — нужны детерминистичные heuristic решения.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
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
OUT_CSV      = DATA_DIR / "pro_vs_heur.csv"
REPORT_MD    = REPORT_DIR / "pro_vs_heur.md"

# ── Импорт agent_bundle_swarm 2 (с пробелом в имени папки) ───────────────
AGENT_DIR = OL_DIR.parent / "agent_bundle_swarm 2"
if not AGENT_DIR.exists():
    print(f'⚠ {AGENT_DIR} не найдена')
    sys.exit(1)
sys.path.insert(0, str(AGENT_DIR))

try:
    import agent as agent_mod
    # Отключаем MCTS для воспроизводимых heuristic решений
    agent_mod.USE_MCTS = False
    from agent import agent
    from zones import compute_zones_from_state
    from orbit_sim import GameState, simulate_step
    from projection import project_state, PROJECTION_HORIZON, NEUTRAL_OWNER
    print(f'✓ agent_bundle_swarm 2 импортирован (USE_MCTS=False)')
except Exception as e:
    print(f'✗ import error: {e!r}')
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────
# Утилиты: target detection через kinematics
# ──────────────────────────────────────────────────────────────────────────

def find_target_planet(src_x: float, src_y: float, angle: float,
                       planets: list, exclude_src_id: int = None) -> int | None:
    """Какая планета цель данного действия? Берём с минимальным угловым отклонением."""
    dx, dy = math.cos(angle), math.sin(angle)
    best_id = None
    best_ang_dev = float('inf')
    for p in planets:
        pid = p[0]
        if pid == exclude_src_id:
            continue
        tdx, tdy = p[2] - src_x, p[3] - src_y
        dist = math.hypot(tdx, tdy)
        if dist < 1e-6:
            continue
        # Только планеты ВПЕРЕДИ направления полёта
        if (dx * tdx + dy * tdy) / dist < 0.3:
            continue
        ang = math.atan2(tdy, tdx)
        ad = abs(math.atan2(math.sin(angle - ang), math.cos(angle - ang)))
        if ad < best_ang_dev:
            best_ang_dev = ad
            best_id = pid
    return best_id


def classify_action_type(player_idx: int, target_owner: int) -> str:
    """neutral / enemy / reinforce."""
    if target_owner == NEUTRAL_OWNER or target_owner == -1:
        return 'neutral'
    if target_owner == player_idx:
        return 'reinforce'
    return 'enemy'


# ──────────────────────────────────────────────────────────────────────────
# Sanity checks — правильно ли мы подаём данные модели?
# ──────────────────────────────────────────────────────────────────────────

def state_consistency_check(obs_now: dict, all_actions: list,
                             obs_next: dict, step: int) -> dict:
    """Применяем ВСЕ pro-actions к нашему state и сравниваем с реальным next obs.
    Если сходится — значит мы правильно реконструируем state и эвристика видит
    те же планеты что и движок Kaggle. Если нет — мы что-то теряем при парсинге."""
    try:
        state = GameState.from_kaggle_obs(obs_now, step=step)
        moves = {i: (a if isinstance(a, list) else [])
                 for i, a in enumerate(all_actions)}
        predicted = simulate_step(state, moves)
        real = GameState.from_kaggle_obs(obs_next, step=step + 1)
        p_by_id = {p.id: p for p in predicted.planets}
        r_by_id = {p.id: p for p in real.planets}
        common = set(p_by_id) & set(r_by_id)
        if not common:
            return {'sanity_owner_match': None, 'sanity_ship_match': None,
                    'sanity_fleet_count_match': None}
        owner_match = sum(1 for pid in common
                          if p_by_id[pid].owner == r_by_id[pid].owner) / len(common)
        ship_match = sum(1 for pid in common
                         if p_by_id[pid].ships == r_by_id[pid].ships) / len(common)
        fleet_match = (len(predicted.fleets) == len(real.fleets))
        return {
            'sanity_owner_match': round(float(owner_match), 4),
            'sanity_ship_match':  round(float(ship_match), 4),
            'sanity_fleet_count_match': bool(fleet_match),
            'sanity_n_planets_compared': len(common),
        }
    except Exception as e:
        return {'sanity_owner_match': None, 'sanity_ship_match': None,
                'sanity_fleet_count_match': None,
                'sanity_error': repr(e)[:100]}


def projection_sanity(obs_now: dict, steps_data: list, step: int,
                      player_idx: int,
                      horizon: int = PROJECTION_HORIZON) -> dict:
    """Сравниваем что projection предсказывает через horizon шагов с тем
    что реально произошло. Это говорит: даёт ли projection полезный прогноз."""
    try:
        state = GameState.from_kaggle_obs(obs_now, step=step)
        projected = project_state(state, horizon=horizon, player=player_idx)
        proj_by_id = {p.id: p for p in projected.planets}

        target_step = step + horizon
        if target_step >= len(steps_data):
            return {}
        s_fut = steps_data[target_step]
        if not isinstance(s_fut, list) or player_idx >= len(s_fut):
            return {}
        obs_fut = (s_fut[player_idx] or {}).get('observation') or {}
        if not obs_fut.get('planets'):
            return {}
        real_owners = {p[0]: p[1] for p in obs_fut['planets']}
        common = set(proj_by_id) & set(real_owners)
        if not common:
            return {}
        owner_acc = sum(1 for pid in common
                        if proj_by_id[pid].owner == real_owners[pid]) / len(common)
        # Также: для planets с projected_at > 0, как часто реальный исход совпал?
        proj_at = getattr(projected, 'projected_at', {}) or {}
        active_proj = [pid for pid in common if proj_at.get(pid, 0) > 0]
        active_acc = None
        if active_proj:
            active_acc = sum(1 for pid in active_proj
                             if proj_by_id[pid].owner == real_owners[pid]) / len(active_proj)
        return {
            'proj_owner_accuracy': round(float(owner_acc), 4),
            'proj_active_accuracy': round(float(active_acc), 4) if active_acc is not None else None,
            'proj_n_active': len(active_proj),
            'proj_horizon': horizon,
        }
    except Exception as e:
        return {'proj_error': repr(e)[:100]}


# ──────────────────────────────────────────────────────────────────────────
# Главная обработка одного launch
# ──────────────────────────────────────────────────────────────────────────

def process_launch(obs: dict, action_launch: list, player_idx: int,
                   episode_id: str, step: int, launch_idx: int) -> dict | None:
    """Сравнить один pro launch с тем что бы сделала эвристика."""
    if not isinstance(action_launch, list) or len(action_launch) != 3:
        return None
    pro_src_id, pro_angle, pro_ships = int(action_launch[0]), float(action_launch[1]), int(action_launch[2])

    planets_raw = obs.get('planets', [])
    if not planets_raw:
        return None

    # Найти src planet и определить target
    src_p = next((p for p in planets_raw if p[0] == pro_src_id), None)
    if src_p is None:
        return None

    pro_target_id = find_target_planet(src_p[2], src_p[3], pro_angle,
                                        planets_raw, exclude_src_id=pro_src_id)
    if pro_target_id is None:
        return None

    pro_target_p = next((p for p in planets_raw if p[0] == pro_target_id), None)
    if pro_target_p is None:
        return None

    pro_action_type = classify_action_type(player_idx, pro_target_p[1])

    # ── Запускаем эвристику ──────────────────────────────────────────────
    try:
        heur_moves = agent(dict(obs))  # копия — agent может что-то менять
    except Exception as e:
        heur_moves = []
        heur_error = repr(e)[:120]
    else:
        heur_error = None

    # ── Запускаем zones отдельно для priorities ──────────────────────────
    try:
        state = GameState.from_kaggle_obs(obs, step=step)
        state_proj = project_state(state, horizon=PROJECTION_HORIZON, player=player_idx)
        df_zones, _ = compute_zones_from_state(state_proj, player=player_idx)
    except Exception as e:
        return {'episode_id': episode_id, 'step': step, 'player_idx': player_idx,
                'launch_idx': launch_idx, 'error': f'zones: {repr(e)[:120]}'}

    # Сортируем по priority (descending)
    df_zones_sorted = df_zones.sort_values('priority', ascending=False).reset_index(drop=True)
    # Не-наши планеты как кандидаты на target
    not_ours = df_zones_sorted[df_zones_sorted['owner'] != player_idx].reset_index(drop=True)

    # Позиция pro_target в ranking (среди не-наших)
    pro_rank_in_targets = None
    pro_target_priority = None
    pro_target_zone = None
    if pro_target_id in not_ours['pid'].values:
        idx = not_ours.index[not_ours['pid'] == pro_target_id][0]
        pro_rank_in_targets = int(idx) + 1
        pro_target_priority = float(not_ours.iloc[idx]['priority'])
        pro_target_zone = str(not_ours.iloc[idx]['zone'])
    elif pro_target_id in df_zones['pid'].values:
        # pro_target — наша планета (reinforce); считаем как rank=-1 в targets ranking
        idx = df_zones.index[df_zones['pid'] == pro_target_id][0]
        pro_target_priority = float(df_zones.iloc[idx]['priority'])
        pro_target_zone = str(df_zones.iloc[idx]['zone'])

    # Топ эвристического выбора
    heur_top_target_id = int(not_ours.iloc[0]['pid']) if len(not_ours) else None
    heur_top_priority = float(not_ours.iloc[0]['priority']) if len(not_ours) else None

    # Топ-K совпадение
    heur_top1 = pro_rank_in_targets == 1 if pro_rank_in_targets is not None else False
    heur_top3 = (pro_rank_in_targets is not None) and (pro_rank_in_targets <= 3)
    heur_top5 = (pro_rank_in_targets is not None) and (pro_rank_in_targets <= 5)
    heur_top10 = (pro_rank_in_targets is not None) and (pro_rank_in_targets <= 10)

    # Что реально сделала эвристика
    heur_n_moves = len(heur_moves) if isinstance(heur_moves, list) else 0
    heur_action_src_id = None
    heur_action_target_id = None
    heur_action_type = None
    heur_action_ships = None
    if heur_n_moves > 0:
        m = heur_moves[0]
        heur_action_src_id = int(m[0])
        heur_action_target_id = find_target_planet(
            *[p for p in planets_raw if p[0] == heur_action_src_id][0][2:4],
            float(m[1]), planets_raw, exclude_src_id=heur_action_src_id
        )
        heur_action_ships = int(m[2])
        if heur_action_target_id is not None:
            ht = next((p for p in planets_raw if p[0] == heur_action_target_id), None)
            if ht:
                heur_action_type = classify_action_type(player_idx, ht[1])

    return {
        'episode_id':         episode_id,
        'step':               step,
        'player_idx':         player_idx,
        'launch_idx':         launch_idx,
        'n_players':          obs.get('player') is not None and 2 or None,  # ставим позже корректно
        'n_planets':          len(planets_raw),
        'n_fleets':           len(obs.get('fleets', [])),

        # Про-выбор
        'pro_src_id':         pro_src_id,
        'pro_target_id':      pro_target_id,
        'pro_ships':          pro_ships,
        'pro_angle':          round(pro_angle, 4),
        'pro_action_type':    pro_action_type,
        'pro_target_owner':   int(pro_target_p[1]),

        # Эвристика приоритезация
        'pro_target_priority':  round(pro_target_priority, 4) if pro_target_priority is not None else None,
        'pro_target_zone':      pro_target_zone,
        'pro_rank_in_targets':  pro_rank_in_targets,
        'n_targets':            len(not_ours),

        'heur_top_target_id':   heur_top_target_id,
        'heur_top_priority':    round(heur_top_priority, 4) if heur_top_priority is not None else None,
        'heur_top1':            heur_top1,
        'heur_top3':            heur_top3,
        'heur_top5':            heur_top5,
        'heur_top10':           heur_top10,

        # Что эвристика реально сделала
        'heur_n_moves':         heur_n_moves,
        'heur_action_src_id':   heur_action_src_id,
        'heur_action_target_id': heur_action_target_id,
        'heur_action_type':     heur_action_type,
        'heur_action_ships':    heur_action_ships,

        # Match flags
        'src_match':            heur_action_src_id == pro_src_id if heur_action_src_id is not None else False,
        'target_match':         heur_action_target_id == pro_target_id if heur_action_target_id is not None else False,
        'type_match':           heur_action_type == pro_action_type if heur_action_type is not None else False,
        'heur_idle':            heur_n_moves == 0,

        'error':                heur_error,
    }


# ──────────────────────────────────────────────────────────────────────────
# Главный цикл
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=1000,
                    help='Сколько случайных pro-launches проверить (0 = все)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f'Loading {LAUNCHES_CSV} …')
    cols_needed = ['episode_id', 'player_idx', 'step', 'launch_idx', 'phase', 'reward']
    launches = pd.read_csv(LAUNCHES_CSV, usecols=cols_needed)
    print(f'  {len(launches):,} pro-launches')

    if args.sample > 0 and args.sample < len(launches):
        launches = launches.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f'  sample: {len(launches)}')

    launches['episode_id'] = launches['episode_id'].astype(str)

    # Группировка по эпизоду чтобы открыть каждый JSON один раз
    by_episode = launches.groupby('episode_id')

    results = []
    t0 = time.time()
    processed = 0
    errors_load = 0

    for eid, group in by_episode:
        json_path = RAW_DIR / f'{eid}.json'
        if not json_path.exists():
            errors_load += 1
            continue
        try:
            with open(json_path) as f:
                ep = json.load(f)
        except Exception:
            errors_load += 1
            continue

        steps = ep.get('steps') or []
        n_players = len(steps[0]) if steps and isinstance(steps[0], list) else 2

        for _, row in group.iterrows():
            step = int(row['step'])
            player_idx = int(row['player_idx'])
            launch_idx = int(row['launch_idx'])

            # State: obs at step, action: из step+1 для player_idx
            if step >= len(steps) - 1:
                continue
            s_now = steps[step]
            s_next = steps[step + 1]
            if not isinstance(s_now, list) or not isinstance(s_next, list):
                continue
            if player_idx >= len(s_now) or player_idx >= len(s_next):
                continue

            obs = (s_now[player_idx] or {}).get('observation') or {}
            if not obs.get('planets'):
                continue

            actions = (s_next[player_idx] or {}).get('action') or []
            if launch_idx >= len(actions):
                continue
            pro_action = actions[launch_idx]

            try:
                rec = process_launch(obs, pro_action, player_idx,
                                      eid, step, launch_idx)
            except Exception as e:
                rec = {'episode_id': eid, 'step': step, 'player_idx': player_idx,
                       'launch_idx': launch_idx, 'error': repr(e)[:120]}
            if rec is None:
                continue
            rec['n_players'] = n_players
            rec['phase'] = float(row['phase'])
            rec['final_reward'] = float(row['reward'])

            # Sanity checks — каждый 10-й launch (чтобы не замедлять сильно)
            if processed % 10 == 0:
                # ВАЖНО: actions которые перевели s_now → s_next хранятся в JSON в
                # steps[t+1].action (т.е. s_next), не в steps[t].action (см. 04_diagnostics)
                all_actions_for_transition = [
                    (s_next[pid_] or {}).get('action') or []
                    for pid_ in range(n_players)
                ]
                obs_next = (s_next[player_idx] or {}).get('observation') or {}
                if obs_next.get('planets'):
                    rec.update(state_consistency_check(
                        obs, all_actions_for_transition, obs_next, step))
                rec.update(projection_sanity(obs, steps, step, player_idx))

            results.append(rec)
            processed += 1
            if processed % 100 == 0:
                dt = time.time() - t0
                print(f'  {processed}/{len(launches)} обработано ({dt:.1f}s, {processed/dt:.1f}/s)')

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f'\n✓ Сохранено: {OUT_CSV}  ({len(df)} записей)')
    print(f'  errors_load: {errors_load}')

    # ── СВОДКА ───────────────────────────────────────────────────────────
    print('\n' + '═'*70)
    print('СВОДКА')
    print('═'*70)
    valid = df[df['pro_rank_in_targets'].notna()].copy()
    print(f'\n  Валидных записей (pro_target найден в targets): {len(valid)} / {len(df)}')

    # Топ-K совпадение target
    print(f'\n  Top-K совпадение pro_target в эвристическом ranking:')
    for k_name, k_col in [('top-1', 'heur_top1'), ('top-3', 'heur_top3'),
                           ('top-5', 'heur_top5'), ('top-10', 'heur_top10')]:
        pct = 100 * valid[k_col].mean()
        print(f'    {k_name}: {pct:.1f}%')

    print(f'\n  Median rank pro_target среди targets: {valid["pro_rank_in_targets"].median():.0f}')
    print(f'  Mean rank: {valid["pro_rank_in_targets"].mean():.1f}')
    print(f'  90-perc rank: {valid["pro_rank_in_targets"].quantile(0.9):.0f}')

    print(f'\n  Type match (neutral/enemy/reinforce): '
          f'{100*df["type_match"].mean():.1f}%')
    print(f'  Source match: {100*df["src_match"].mean():.1f}%')
    print(f'  Target match (heur stuck the same): {100*df["target_match"].mean():.1f}%')

    print(f'\n  heur_idle (агент не стрелял): {100*df["heur_idle"].mean():.1f}%')
    print(f'  Это значит профи стрельнул, эвристика бы не сделала ничего.')

    # По типу действия
    print(f'\n  Type-K match by pro_action_type:')
    for t in ['neutral', 'enemy', 'reinforce']:
        sub = valid[valid['pro_action_type'] == t]
        if len(sub) < 10: continue
        print(f'    {t:<10} n={len(sub):>4}  top1={100*sub["heur_top1"].mean():.1f}%  '
              f'top5={100*sub["heur_top5"].mean():.1f}%  '
              f'median_rank={sub["pro_rank_in_targets"].median():.0f}')

    # По фазе
    print(f'\n  Top-3 match by phase quartile:')
    valid['phase_q'] = pd.cut(valid['phase'], bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
                               labels=['q1', 'q2', 'q3', 'q4'])
    by_phase = valid.groupby('phase_q', observed=True).agg(
        n=('heur_top3', 'count'),
        top1=('heur_top1', 'mean'),
        top3=('heur_top3', 'mean'),
        med_rank=('pro_rank_in_targets', 'median'),
    ).round(3)
    print(by_phase.to_string())

    # По n_players
    print(f'\n  Top-3 match by n_players:')
    by_np = valid.groupby('n_players').agg(
        n=('heur_top3', 'count'),
        top1=('heur_top1', 'mean'),
        top3=('heur_top3', 'mean'),
        med_rank=('pro_rank_in_targets', 'median'),
    ).round(3)
    print(by_np.to_string())

    # ── SANITY CHECKS — правильно ли мы подаём данные модели ─────────
    print('\n' + '═'*70)
    print('SANITY CHECKS: правильно ли мы подаём данные модели?')
    print('═'*70)
    sanity = df[df['sanity_owner_match'].notna()].copy() if 'sanity_owner_match' in df.columns else pd.DataFrame()
    if len(sanity):
        print(f'\n  State consistency (sample {len(sanity)} launches):')
        print(f'    Mean owner match:  {sanity["sanity_owner_match"].mean():.4f}')
        print(f'    Mean ship  match:  {sanity["sanity_ship_match"].mean():.4f}')
        print(f'    Fleet count match: {100*sanity["sanity_fleet_count_match"].mean():.1f}%')
        # сколько launches с perfect match
        perfect = ((sanity["sanity_owner_match"] == 1.0) &
                   (sanity["sanity_ship_match"] == 1.0)).sum()
        print(f'    Perfect state match: {perfect}/{len(sanity)} ({100*perfect/len(sanity):.1f}%)')
    else:
        print('  Нет sanity данных')

    proj = df[df['proj_owner_accuracy'].notna()].copy() if 'proj_owner_accuracy' in df.columns else pd.DataFrame()
    if len(proj):
        print(f'\n  Projection sanity (sample {len(proj)} launches):')
        print(f'    Mean owner_accuracy через PROJECTION_HORIZON шагов: '
              f'{proj["proj_owner_accuracy"].mean():.4f}')
        if 'proj_active_accuracy' in proj.columns:
            active = proj[proj['proj_active_accuracy'].notna()]
            if len(active):
                print(f'    Mean active_accuracy (только планеты с projected_at>0): '
                      f'{active["proj_active_accuracy"].mean():.4f}  '
                      f'(n_active records={len(active)})')
    else:
        print('  Нет projection данных')

    # ── Markdown отчёт ──────────────────────────────────────────────────
    md = []
    md.append('# Pro vs Heuristic — comparison report\n\n')
    md.append(f'**Данные**: {len(df)} pro-launches, валидных {len(valid)}\n')
    md.append(f'**Эвристика**: `agent_bundle_swarm 2`, USE_MCTS=False\n\n')

    md.append('## Главные числа\n\n')
    md.append('| Metric | Value |\n|---|---|\n')
    md.append(f'| Top-1 target match | {100*valid["heur_top1"].mean():.1f}% |\n')
    md.append(f'| Top-3 target match | {100*valid["heur_top3"].mean():.1f}% |\n')
    md.append(f'| Top-5 target match | {100*valid["heur_top5"].mean():.1f}% |\n')
    md.append(f'| Top-10 target match | {100*valid["heur_top10"].mean():.1f}% |\n')
    md.append(f'| Type match (n/e/r) | {100*df["type_match"].mean():.1f}% |\n')
    md.append(f'| Source match | {100*df["src_match"].mean():.1f}% |\n')
    md.append(f'| Heur idle when pro fired | {100*df["heur_idle"].mean():.1f}% |\n')
    md.append(f'| Median pro_rank | {valid["pro_rank_in_targets"].median():.0f} |\n')

    md.append('\n## По типу действия профи\n\n')
    md.append('| type | n | top1 | top3 | median_rank |\n|---|---|---|---|---|\n')
    for t in ['neutral', 'enemy', 'reinforce']:
        sub = valid[valid['pro_action_type'] == t]
        if len(sub) < 10: continue
        md.append(f'| {t} | {len(sub)} | {100*sub["heur_top1"].mean():.1f}% | '
                  f'{100*sub["heur_top5"].mean():.1f}% | '
                  f'{sub["pro_rank_in_targets"].median():.0f} |\n')

    md.append('\n## По фазе\n\n```\n')
    md.append(by_phase.to_string())
    md.append('\n```\n\n## По n_players\n\n```\n')
    md.append(by_np.to_string())
    md.append('\n```\n')

    md.append('\n## Графики\n\n')
    md.append('- `pro_vs_heur_rank_hist.png` — распределение rank pro_target\n')
    md.append('- `pro_vs_heur_match_by_phase.png` — top-K match по фазам\n')

    REPORT_MD.write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_MD}')

    # Графики
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ranks = valid['pro_rank_in_targets'].clip(upper=20)
        ax.hist(ranks, bins=20, edgecolor='k', alpha=0.7)
        ax.axvline(1, c='green', ls='--', label='top-1')
        ax.axvline(3, c='orange', ls='--', label='top-3')
        ax.axvline(5, c='red', ls='--', label='top-5')
        ax.set_xlabel('rank pro_target в эвристическом ranking (clip @20)')
        ax.set_ylabel('число launches')
        ax.set_title('Где профи-цель в эвристическом ranking?')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'pro_vs_heur_rank_hist.png', dpi=120)
        plt.close()

        # Match by phase
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(4)
        for col, label, c in [('top1', 'Top-1', 'green'),
                               ('top3', 'Top-3', 'orange'),
                               ('top5', 'Top-5', 'red')]:
            vals = by_phase[col].values if col in by_phase.columns else \
                   valid.groupby('phase_q', observed=True)[f'heur_{col}'].mean().values
            ax.plot(x, vals, 'o-', label=label, color=c, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(['q1', 'q2', 'q3', 'q4'])
        ax.set_xlabel('phase quartile')
        ax.set_ylabel('match rate')
        ax.set_title('Top-K match эвристики с pro-выбором по фазам')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'pro_vs_heur_match_by_phase.png', dpi=120)
        plt.close()
        print('  графики сохранены')
    except Exception as e:
        print(f'  графики: {e!r}')


if __name__ == '__main__':
    main()
