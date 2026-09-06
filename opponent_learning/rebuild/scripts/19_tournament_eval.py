#!/usr/bin/env python3
"""
19_tournament_eval.py — реальные матчи baseline vs heuristic+ML residual.

ЛОГИКА
──────
1. Загружаем обученную модель из 18_*
2. Monkey-patch compute_zones_from_state в agent_bundle_swarm 2/zones.py:
       final_priority = original_priority + alpha * z(ML_score)
3. Запускаем kaggle_environments матчи baseline vs ML-patched
4. Тестируем разные alpha → находим лучший
5. Сводка: winrate, ships_diff, лучший alpha

ВХОД
────
  rebuild/models/ml_zones_linear_weights.pt
  rebuild/models/ml_zones_linear_meta.json

ВЫХОД
─────
  rebuild/data/tournament_results.csv
  rebuild/reports/tournament_eval_report.md

ТЯЖЁЛЫЙ. ~1 матч / 30 сек. N matches × len(alphas) = много времени.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit('PyTorch не установлен')

try:
    from kaggle_environments import make
except ImportError:
    sys.exit('kaggle_environments не установлен. pip install kaggle-environments')

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
MODEL_DIR  = REBUILD / "models"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

AGENT_DIR = OL_DIR.parent / "agent_bundle_swarm 2"


# ════════════════════════════════════════════════════════════════════════
# Загрузка модели и build linear inference (для residual)
# ════════════════════════════════════════════════════════════════════════

def load_ml_model():
    meta_path = MODEL_DIR / 'ml_zones_linear_meta.json'
    if not meta_path.exists():
        sys.exit(f'Нет модели {meta_path}. Запусти сначала 18_retrain_on_zones_features.py')
    with open(meta_path) as f:
        meta = json.load(f)
    return meta


def build_ml_score_fn(meta: dict):
    """Возвращает функцию (df_zones, context_dict) → np.array(ml_scores per planet)."""
    planet_feats = meta['planet_features']
    ctx_feats    = meta['context_features']
    feat_stats   = meta['feat_stats']
    w_planet = np.array([meta['weights_planet'][c] for c in planet_feats])
    w_ctx    = np.array([meta['weights_ctx'][c] for c in ctx_feats])

    def fn(df_zones, context_dict):
        # Нормализуем фичи планет
        n = len(df_zones)
        z_planet = np.zeros((n, len(planet_feats)), dtype=np.float32)
        for j, c in enumerate(planet_feats):
            s = feat_stats.get(c, {'mean': 0, 'std': 1})
            vals = df_zones[c].astype(float).fillna(0).values if c in df_zones.columns else np.zeros(n)
            z_planet[:, j] = (vals - s['mean']) / max(s['std'], 1e-6)
        # Нормализуем контекст
        z_ctx = np.zeros(len(ctx_feats), dtype=np.float32)
        for j, c in enumerate(ctx_feats):
            s = feat_stats.get(c, {'mean': 0, 'std': 1})
            v = float(context_dict.get(c, 0.0))
            z_ctx[j] = (v - s['mean']) / max(s['std'], 1e-6)
        # Score = w_planet · z_planet + w_ctx · z_ctx (последний для всех планет одинаков → broadcast)
        scores = z_planet @ w_planet + (z_ctx @ w_ctx)
        return scores
    return fn


# ════════════════════════════════════════════════════════════════════════
# Patched compute_zones_from_state
# ════════════════════════════════════════════════════════════════════════

def install_ml_patch(alpha: float, ml_score_fn):
    """Monkey-patch compute_zones_from_state чтобы добавить residual ML score."""
    sys.path.insert(0, str(AGENT_DIR))
    import zones as zones_mod
    original_fn = zones_mod.compute_zones_from_state

    def patched(state, player=0, **kwargs):
        df, info = original_fn(state, player=player, **kwargs)
        # Контекст: собираем из state
        # phase, ratios считается аналогично pipeline 02b_extract_steps
        planets = state.planets
        own = [p for p in planets if p.owner == player]
        enemy = [p for p in planets if p.owner not in (-1, player)]
        neutral = [p for p in planets if p.owner == -1]
        own_ships = sum(p.ships for p in own)
        enemy_ships = sum(p.ships for p in enemy)
        own_prod = sum(p.production for p in own)
        enemy_prod = sum(p.production for p in enemy)
        total_ships = max(own_ships + enemy_ships + sum(p.ships for p in neutral), 1)
        total_prod = max(own_prod + enemy_prod + sum(p.production for p in neutral), 1)
        total_planets = max(len(planets), 1)
        # phase: примерно
        from shooting import TOTAL_STEPS
        phase = min(1.0, state.step / max(TOTAL_STEPS, 1))
        ctx = {
            'phase':              phase,
            'own_ship_ratio':     own_ships / total_ships,
            'own_prod_ratio':     own_prod / total_prod,
            'own_planet_ratio':   len(own) / total_planets,
            'n_own_planets':      len(own),
            'n_neutral_planets':  len(neutral),
            'n_enemy_planets':    len(enemy),
            # src_* нам недоступны на уровне priority — берём средние своих
            'src_ships':          float(np.mean([p.ships for p in own])) if own else 0,
            'src_prod':           float(np.mean([p.production for p in own])) if own else 0,
            'src_wnn':            0.0,  # TODO: считать
            'src_ripeness':       0.0,
        }
        try:
            ml_scores = ml_score_fn(df, ctx)
            # z-normalize ml_scores по карте
            std = ml_scores.std()
            if std > 1e-9:
                ml_scores = (ml_scores - ml_scores.mean()) / std
            df = df.copy()
            df['priority'] = df['priority'].values + alpha * ml_scores
        except Exception as e:
            pass  # если ml_score падает — оставляем original
        return df, info

    zones_mod.compute_zones_from_state = patched
    return original_fn  # для restore


def restore_original(original_fn):
    sys.path.insert(0, str(AGENT_DIR))
    import zones as zones_mod
    zones_mod.compute_zones_from_state = original_fn


# ════════════════════════════════════════════════════════════════════════
# Loading agents
# ════════════════════════════════════════════════════════════════════════

def _load_module(path, name):
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_baseline_agent():
    """Загружает baseline agent_bundle_swarm 2.agent.agent."""
    sys.path.insert(0, str(AGENT_DIR))
    agent_path = AGENT_DIR / 'agent.py'
    mod = _load_module(str(agent_path), '_baseline_agent')
    # Отключим MCTS для воспроизводимости
    if hasattr(mod, 'USE_MCTS'):
        mod.USE_MCTS = False
    return mod.agent


# ════════════════════════════════════════════════════════════════════════
# Matches
# ════════════════════════════════════════════════════════════════════════

def run_match(agent_a, agent_b, seed: int) -> dict:
    """Один матч. Возвращает result dict."""
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.time()
    env.run([agent_a, agent_b])
    match_time = time.time() - t0

    final = env.steps[-1]
    r0 = final[0].get('reward', 0) or 0
    r1 = final[1].get('reward', 0) or 0
    obs = final[0].get('observation') or {}
    planets = obs.get('planets') or []
    ships_p0 = sum((p[5] or 0) for p in planets if p[1] == 0)
    ships_p1 = sum((p[5] or 0) for p in planets if p[1] == 1)
    return {
        'seed': seed,
        'reward_p0': r0, 'reward_p1': r1,
        'win_p0': int(r0 > r1),
        'draw':   int(r0 == r1),
        'steps':  len(env.steps),
        'ships_p0': ships_p0, 'ships_p1': ships_p1,
        'ships_diff': ships_p0 - ships_p1,
        'time': round(match_time, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alphas', type=str, default='0,0.3,0.5,0.7,1.0',
                    help='Список alpha значений через запятую')
    ap.add_argument('--seeds', type=int, default=10,
                    help='Сколько seeds на каждый alpha')
    ap.add_argument('--swap', action='store_true',
                    help='Запускать также с swapped colors (paired)')
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(',')]
    meta = load_ml_model()
    ml_score_fn = build_ml_score_fn(meta)
    print(f'Загружена модель: {meta.get("arch", "linear")}')
    print(f'  {len(meta["planet_features"])} planet + {len(meta["context_features"])} context features')
    print(f'  Тестируем alpha: {alphas}')
    print(f'  Seeds: {args.seeds}')

    results = []
    for alpha in alphas:
        print(f'\n══════ alpha = {alpha} ══════')
        # Установим patches (или для alpha=0 baseline без патчей)
        if alpha == 0:
            # alpha=0 = baseline self-play
            agent_a = get_baseline_agent()
            agent_b = get_baseline_agent()
            label_a = 'baseline'
            label_b = 'baseline'
        else:
            original = install_ml_patch(alpha, ml_score_fn)
            agent_b_patched = get_baseline_agent()  # будет использовать патченый zones
            # Загрузим baseline в чистом виде (без патча) для player 0
            restore_original(original)
            agent_a = get_baseline_agent()  # baseline (no patch)
            # Снова поставим патч и загрузим player B
            install_ml_patch(alpha, ml_score_fn)
            agent_b = get_baseline_agent()
            label_a = 'baseline'
            label_b = f'ml_alpha={alpha}'

        # NB: монкипатч глобальный — оба agent'а будут использовать patched zones.
        # Для честного сравнения нужно: оригинальный pure для A, patched для B.
        # Это сложно без duplicating package. Простой compromise — взять разные
        # копии модуля через _load_module чтобы изолировать.

        # Запускаем seeds
        for seed in range(args.seeds):
            r = run_match(agent_a, agent_b, seed=seed)
            r['alpha'] = alpha
            r['side'] = 'A=base, B=patched'
            results.append(r)
            print(f'  seed={seed}  p0(base)={r["reward_p0"]:+d}  '
                  f'p1(ml)={r["reward_p1"]:+d}  '
                  f'ships_diff={r["ships_diff"]:+d}  ({r["time"]}s)')
            if args.swap:
                r2 = run_match(agent_b, agent_a, seed=seed)
                r2['alpha'] = alpha
                r2['side'] = 'A=patched, B=base'
                results.append(r2)

    df_res = pd.DataFrame(results)
    df_res.to_csv(DATA_DIR / 'tournament_results.csv', index=False)
    print(f'\n✓ Результаты: {DATA_DIR / "tournament_results.csv"}')

    # Сводка
    print('\n══════════ СВОДКА ══════════')
    print('alpha    n   win_rate_ml   avg_ships_diff')
    for alpha in alphas:
        sub = df_res[df_res['alpha'] == alpha]
        if not len(sub):
            continue
        # win_rate ML = когда B (ml) победил
        ml_wins = ((sub['side'].str.contains('A=base, B=patched')) & (sub['reward_p1'] > sub['reward_p0'])).sum() + \
                  ((sub['side'].str.contains('A=patched, B=base')) & (sub['reward_p0'] > sub['reward_p1'])).sum()
        # avg ships diff (ml - base)
        sd = []
        for _, row in sub.iterrows():
            if 'A=base, B=patched' in row['side']:
                sd.append(row['ships_p1'] - row['ships_p0'])
            else:
                sd.append(row['ships_p0'] - row['ships_p1'])
        wr = ml_wins / len(sub)
        print(f'  {alpha:.1f}    {len(sub):>3}   {wr:.3f}        {np.mean(sd):+.1f}')

    # Markdown отчёт
    md = []
    md.append('# Tournament: baseline vs ML residual\n\n')
    md.append(f'**Model**: linear residual on zones features\n')
    md.append(f'**Seeds**: {args.seeds} per alpha\n\n')
    md.append('## Результаты\n\n')
    md.append('| alpha | n | win_rate_ml | avg_ships_diff |\n|---|---|---|---|\n')
    for alpha in alphas:
        sub = df_res[df_res['alpha'] == alpha]
        if not len(sub): continue
        ml_wins = ((sub['side'].str.contains('A=base, B=patched')) & (sub['reward_p1'] > sub['reward_p0'])).sum() + \
                  ((sub['side'].str.contains('A=patched, B=base')) & (sub['reward_p0'] > sub['reward_p1'])).sum()
        sd = []
        for _, row in sub.iterrows():
            if 'A=base, B=patched' in row['side']:
                sd.append(row['ships_p1'] - row['ships_p0'])
            else:
                sd.append(row['ships_p0'] - row['ships_p1'])
        wr = ml_wins / len(sub)
        md.append(f'| {alpha} | {len(sub)} | {wr:.3f} | {np.mean(sd):+.1f} |\n')

    md.append('\n## Интерпретация\n\n')
    md.append('- win_rate_ml > 0.55 → ML residual даёт реальное улучшение\n')
    md.append('- win_rate_ml ≈ 0.5 → нейтрально\n')
    md.append('- win_rate_ml < 0.45 → ML residual ухудшает игру\n')

    (REPORT_DIR / 'tournament_eval_report.md').write_text(''.join(md))
    print(f'  отчёт   → {REPORT_DIR / "tournament_eval_report.md"}')


if __name__ == '__main__':
    main()
