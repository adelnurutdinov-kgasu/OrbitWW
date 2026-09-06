#!/usr/bin/env python3
"""
15_bo_zones_promatch.py — Bayesian optimization W_OURS / W_TARGETS под pro-matching.

ИДЕЯ
────
zones_bayesian.py оптимизирует веса под winrate в турнирах. Это требует
прогонять полные матчи на каждый trial. Здесь мы оптимизируем те же веса,
но под другой loss: для sample pro-launches считаем долю случаев когда
pro_target попал в топ-K эвристического ranking. Это:
  - быстрее (zones-only, без полного матча)
  - целенаправленно к согласованию с профи

ОПТИМИЗИРУЕМ
────────────
14 весов: W_OURS (7 фич) + W_TARGETS (7 фич), диапазон [-1.5, +1.5].
Фичи: area_inv, wnn_close_res, mean_dist_all, prod, ships, n_cross, late_aggression.

LOSS
────
Maximize top-3 accuracy на sample pro-launches.
Также трекаем top-1 / top-5 / median rank для отчёта.

КЭШИРОВАНИЕ
───────────
Sample states грузим из JSON ОДИН раз (parsing медленный). В каждый trial:
  apply weights → compute_zones для каждого state → подсчитать rank pro_target.
Один trial на ~500 launches ≈ 5-10 секунд. 200 trials ≈ 20-30 минут.

ВХОД
────
  opponent_learning/data/processed/launches.csv

ВЫХОД
─────
  rebuild/data/bo_promatch_trials.csv
  rebuild/data/bo_promatch_best.json
  rebuild/reports/bo_promatch_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    sys.exit('Optuna не установлен. Установи: pip install optuna --break-system-packages')

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
PROC_DIR   = OL_DIR / "data" / "processed"
RAW_DIR    = OL_DIR / "data" / "raw"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

LAUNCHES_CSV = PROC_DIR / "launches.csv"

# Импорт agent_bundle_swarm 2
AGENT_DIR = OL_DIR.parent / "agent_bundle_swarm 2"
sys.path.insert(0, str(AGENT_DIR))

import zones as zones_mod
from zones import compute_zones_from_state, W_OURS as DEFAULT_W_OURS, W_TARGETS as DEFAULT_W_TARGETS
from orbit_sim import GameState
from projection import project_state, PROJECTION_HORIZON, NEUTRAL_OWNER

FEATURES = ['area_inv', 'wnn_close_res', 'mean_dist_all',
            'prod', 'ships', 'n_cross', 'late_aggression']

# Сохраним baseline (для restore)
BASELINE_W_OURS = dict(DEFAULT_W_OURS)
BASELINE_W_TARGETS = dict(DEFAULT_W_TARGETS)


# ──────────────────────────────────────────────────────────────────────────
# Кэш states
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


def build_cache(sample_size: int, seed: int = 42, prj_horizon=PROJECTION_HORIZON):
    """Собираем sample launches и pre-compute states.
    Возвращает список (state_proj, player_idx, pro_target_id, pro_target_is_ours_planet)."""
    print(f'Loading {LAUNCHES_CSV} …')
    launches = pd.read_csv(LAUNCHES_CSV,
                            usecols=['episode_id', 'player_idx', 'step', 'launch_idx', 'phase'])
    launches['episode_id'] = launches['episode_id'].astype(str)
    launches = launches.sample(sample_size, random_state=seed).reset_index(drop=True)
    print(f'  sample: {len(launches)}')

    cache = []
    t0 = time.time()
    by_ep = launches.groupby('episode_id')
    skipped = 0
    for eid, group in by_ep:
        json_path = RAW_DIR / f'{eid}.json'
        if not json_path.exists():
            skipped += len(group)
            continue
        try:
            with open(json_path) as f:
                ep = json.load(f)
        except Exception:
            skipped += len(group)
            continue
        steps = ep.get('steps') or []
        for _, row in group.iterrows():
            step = int(row['step'])
            player_idx = int(row['player_idx'])
            launch_idx = int(row['launch_idx'])
            if step >= len(steps) - 1:
                skipped += 1
                continue
            s_now, s_next = steps[step], steps[step + 1]
            if not (isinstance(s_now, list) and isinstance(s_next, list)):
                skipped += 1
                continue
            obs = (s_now[player_idx] or {}).get('observation') or {}
            if not obs.get('planets'):
                skipped += 1
                continue
            actions = (s_next[player_idx] or {}).get('action') or []
            if launch_idx >= len(actions):
                skipped += 1
                continue
            pro_action = actions[launch_idx]
            if not isinstance(pro_action, list) or len(pro_action) != 3:
                skipped += 1
                continue

            pro_src_id, pro_angle = int(pro_action[0]), float(pro_action[1])
            planets_raw = obs['planets']
            src_p = next((p for p in planets_raw if p[0] == pro_src_id), None)
            if src_p is None:
                skipped += 1
                continue
            pro_target_id = find_target_planet(src_p[2], src_p[3], pro_angle,
                                                planets_raw, exclude_src_id=pro_src_id)
            if pro_target_id is None:
                skipped += 1
                continue
            target_p = next((p for p in planets_raw if p[0] == pro_target_id), None)
            if target_p is None:
                skipped += 1
                continue
            is_ours = (target_p[1] == player_idx)

            try:
                state = GameState.from_kaggle_obs(obs, step=step)
                state_proj = project_state(state, horizon=prj_horizon, player=player_idx)
            except Exception:
                skipped += 1
                continue

            cache.append({
                'state_proj':     state_proj,
                'player_idx':     player_idx,
                'pro_target_id':  pro_target_id,
                'pro_is_ours':    is_ours,
                'step':           step,
                'phase':          float(row['phase']),
            })

    dt = time.time() - t0
    print(f'  кэш: {len(cache)} states за {dt:.1f}s  (skipped: {skipped})')
    return cache


# ──────────────────────────────────────────────────────────────────────────
# Оценка trial'а
# ──────────────────────────────────────────────────────────────────────────

def evaluate_weights(w_ours: dict, w_targets: dict, cache: list,
                     return_full: bool = False) -> dict:
    """Применить веса, прогнать zones на cache, посчитать pro-match metrics."""
    # Мутация defaults (compute_zones_from_state использует module-level)
    for k, v in w_ours.items():
        zones_mod.W_OURS[k] = v
    for k, v in w_targets.items():
        zones_mod.W_TARGETS[k] = v

    top1 = top3 = top5 = top10 = 0
    total_attack = 0  # cases где pro_target — не наша планета
    total_reinforce = 0
    ranks_attack = []
    ranks_reinforce = []

    for entry in cache:
        state = entry['state_proj']
        player_idx = entry['player_idx']
        pro_id = entry['pro_target_id']
        try:
            df, _ = compute_zones_from_state(state, player=player_idx)
        except Exception:
            continue

        if entry['pro_is_ours']:
            # Reinforce: ранжируем среди наших по priority
            df_own = df[df['owner'] == player_idx].sort_values('priority', ascending=False).reset_index(drop=True)
            if pro_id in df_own['pid'].values:
                rank = int(df_own.index[df_own['pid'] == pro_id][0]) + 1
                ranks_reinforce.append(rank)
                total_reinforce += 1
        else:
            df_targets = df[df['owner'] != player_idx].sort_values('priority', ascending=False).reset_index(drop=True)
            if pro_id in df_targets['pid'].values:
                rank = int(df_targets.index[df_targets['pid'] == pro_id][0]) + 1
                ranks_attack.append(rank)
                total_attack += 1
                if rank <= 1:  top1  += 1
                if rank <= 3:  top3  += 1
                if rank <= 5:  top5  += 1
                if rank <= 10: top10 += 1

    if total_attack == 0:
        return {'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0,
                'total_attack': 0, 'median_rank': 99}

    metrics = {
        'top1':  top1 / total_attack,
        'top3':  top3 / total_attack,
        'top5':  top5 / total_attack,
        'top10': top10 / total_attack,
        'total_attack':    total_attack,
        'total_reinforce': total_reinforce,
        'median_rank':     float(np.median(ranks_attack)) if ranks_attack else 99,
        'mean_rank':       float(np.mean(ranks_attack)) if ranks_attack else 99,
    }
    if return_full:
        metrics['ranks_attack'] = ranks_attack
        metrics['ranks_reinforce'] = ranks_reinforce
    return metrics


def restore_baseline():
    """Восстановить дефолтные веса в модуле zones."""
    for k, v in BASELINE_W_OURS.items():
        zones_mod.W_OURS[k] = v
    for k, v in BASELINE_W_TARGETS.items():
        zones_mod.W_TARGETS[k] = v


# ──────────────────────────────────────────────────────────────────────────
# Optuna study
# ──────────────────────────────────────────────────────────────────────────

def make_objective(cache, target_metric: str = 'top3'):
    def objective(trial: optuna.Trial) -> float:
        w_ours = {f: trial.suggest_float(f'ours_{f}', -1.5, 1.5) for f in FEATURES}
        w_targets = {f: trial.suggest_float(f'tgt_{f}', -1.5, 1.5) for f in FEATURES}
        m = evaluate_weights(w_ours, w_targets, cache)
        # Логируем все метрики в user_attrs для отчёта
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m[target_metric]
    return objective


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=400,
                    help='Сколько launches брать для каждого trial (cache size)')
    ap.add_argument('--trials', type=int, default=150,
                    help='Сколько BO trials')
    ap.add_argument('--target', default='top3',
                    choices=['top1', 'top3', 'top5', 'top10'],
                    help='Какую метрику максимизировать')
    ap.add_argument('--startup', type=int, default=20,
                    help='Сколько random startup trials до TPE')
    args = ap.parse_args()

    cache = build_cache(args.sample)
    if not cache:
        print('Кэш пустой')
        sys.exit(1)

    # Baseline (текущие веса)
    print('\n── Baseline (текущие веса в zones.py) ────────────────')
    restore_baseline()
    baseline = evaluate_weights(BASELINE_W_OURS, BASELINE_W_TARGETS, cache)
    print(f'  top1={baseline["top1"]:.4f}  top3={baseline["top3"]:.4f}  '
          f'top5={baseline["top5"]:.4f}  median_rank={baseline["median_rank"]:.1f}  '
          f'n_attack={baseline["total_attack"]}')

    # Study
    print(f'\n── Optuna TPE study (target={args.target}, {args.trials} trials) ──')
    sampler = TPESampler(n_startup_trials=args.startup, seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    objective = make_objective(cache, target_metric=args.target)

    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    dt = time.time() - t0
    print(f'  выполнено за {dt:.0f}s ({dt/args.trials:.1f}s/trial)')

    best = study.best_trial
    best_w_ours   = {f: best.params[f'ours_{f}'] for f in FEATURES}
    best_w_targets = {f: best.params[f'tgt_{f}']  for f in FEATURES}

    # Финальная оценка с большим расширенным cache (если возможно)
    print(f'\n── Best trial: {args.target}={best.value:.4f} ──────────────────')
    final_m = evaluate_weights(best_w_ours, best_w_targets, cache, return_full=True)
    print(f'  Полные метрики на cache:')
    for k in ['top1', 'top3', 'top5', 'top10', 'median_rank', 'mean_rank',
              'total_attack', 'total_reinforce']:
        print(f'    {k}: {final_m.get(k):.4f}' if isinstance(final_m.get(k), float)
              else f'    {k}: {final_m.get(k)}')

    print(f'\n  Прибавка vs baseline:')
    for k in ['top1', 'top3', 'top5', 'top10']:
        d = final_m[k] - baseline[k]
        print(f'    {k}: {baseline[k]:.4f} → {final_m[k]:.4f}  (Δ={d:+.4f})')
    print(f'    median_rank: {baseline["median_rank"]:.1f} → {final_m["median_rank"]:.1f}')

    print(f'\n  Найденные веса (W_OURS):')
    for f in FEATURES:
        d = best_w_ours[f] - BASELINE_W_OURS[f]
        print(f'    {f:18s}  {BASELINE_W_OURS[f]:+.3f} → {best_w_ours[f]:+.3f}  (Δ={d:+.3f})')
    print(f'\n  Найденные веса (W_TARGETS):')
    for f in FEATURES:
        d = best_w_targets[f] - BASELINE_W_TARGETS[f]
        print(f'    {f:18s}  {BASELINE_W_TARGETS[f]:+.3f} → {best_w_targets[f]:+.3f}  (Δ={d:+.3f})')

    # Сохранение
    trials_df = study.trials_dataframe()
    trials_df.to_csv(DATA_DIR / 'bo_promatch_trials.csv', index=False)

    best_json = {
        'baseline': {
            'W_OURS':   dict(BASELINE_W_OURS),
            'W_TARGETS': dict(BASELINE_W_TARGETS),
            'metrics':  {k: baseline[k] for k in ['top1', 'top3', 'top5', 'top10',
                                                    'median_rank', 'total_attack']},
        },
        'best': {
            'W_OURS':   best_w_ours,
            'W_TARGETS': best_w_targets,
            'metrics':  {k: final_m[k] for k in ['top1', 'top3', 'top5', 'top10',
                                                  'median_rank', 'mean_rank',
                                                  'total_attack', 'total_reinforce']},
        },
        'config': {
            'sample': args.sample,
            'trials': args.trials,
            'target': args.target,
            'startup': args.startup,
        },
    }
    (DATA_DIR / 'bo_promatch_best.json').write_text(json.dumps(best_json, indent=2))

    # Markdown отчёт
    md = []
    md.append(f'# BO результат: zones-веса под pro-matching\n\n')
    md.append(f'**Target metric**: {args.target}\n')
    md.append(f'**Trials**: {args.trials}, sample: {args.sample}\n\n')

    md.append(f'## Метрики baseline vs best\n\n')
    md.append('| metric | baseline | best | Δ |\n|---|---|---|---|\n')
    for k in ['top1', 'top3', 'top5', 'top10']:
        md.append(f'| {k} | {baseline[k]:.4f} | {final_m[k]:.4f} | '
                  f'{final_m[k]-baseline[k]:+.4f} |\n')
    md.append(f'| median_rank | {baseline["median_rank"]:.1f} | '
              f'{final_m["median_rank"]:.1f} | '
              f'{final_m["median_rank"]-baseline["median_rank"]:+.1f} |\n')

    md.append(f'\n## Найденные веса W_OURS\n\n')
    md.append('| feature | baseline | best | Δ |\n|---|---|---|---|\n')
    for f in FEATURES:
        md.append(f'| {f} | {BASELINE_W_OURS[f]:+.3f} | {best_w_ours[f]:+.3f} | '
                  f'{best_w_ours[f]-BASELINE_W_OURS[f]:+.3f} |\n')

    md.append(f'\n## Найденные веса W_TARGETS\n\n')
    md.append('| feature | baseline | best | Δ |\n|---|---|---|---|\n')
    for f in FEATURES:
        md.append(f'| {f} | {BASELINE_W_TARGETS[f]:+.3f} | {best_w_targets[f]:+.3f} | '
                  f'{best_w_targets[f]-BASELINE_W_TARGETS[f]:+.3f} |\n')

    md.append(f'\n## Готовые подставки в zones.py\n\n')
    md.append('```python\nW_OURS = {\n')
    for f in FEATURES:
        md.append(f'    \'{f}\': {best_w_ours[f]:+.3f},\n')
    md.append('}\n\nW_TARGETS = {\n')
    for f in FEATURES:
        md.append(f'    \'{f}\': {best_w_targets[f]:+.3f},\n')
    md.append('}\n```\n')

    (REPORT_DIR / 'bo_promatch_report.md').write_text(''.join(md))
    print(f'\n✓ Сохранено:')
    print(f'  trials → {DATA_DIR / "bo_promatch_trials.csv"}')
    print(f'  best   → {DATA_DIR / "bo_promatch_best.json"}')
    print(f'  отчёт  → {REPORT_DIR / "bo_promatch_report.md"}')

    # Восстановим baseline в модуле
    restore_baseline()


if __name__ == '__main__':
    main()
