#!/usr/bin/env python3
"""
21_tournament_proper.py — честный турнир baseline vs ML.

ОТЛИЧИЕ ОТ 19_*
───────────────
19 использовал monkey-patch который применялся ОБОИМ агентам глобально.
Тут baseline и ML — это **разные папки** с **разными модулями**, грузятся
через _load_module из разных путей. Изоляция гарантирована.

ЛОГИКА
──────
- baseline = agent_bundle_swarm 2 (не модифицирован)
- ml       = agent_bundle_ml (создан через 20_setup_ml_bundle.py)
- Запускаем N seeds × 2 swap (paired) — стабильнее
- Сводка: winrate ML, avg ships_diff, p-value

ВЫЗОВ
─────
  python3 21_tournament_proper.py --seeds 10 --swap

ВЫХОД
─────
  rebuild/data/tournament_proper_results.csv
  rebuild/reports/tournament_proper_report.md
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from kaggle_environments import make
except ImportError:
    sys.exit('kaggle_environments не установлен')

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
PROJECT_ROOT = OL_DIR.parent
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = PROJECT_ROOT / "agent_bundle_swarm 2"
ML_DIR       = PROJECT_ROOT / "agent_bundle_ml"


def _load_agent(bundle_dir: Path, name: str):
    """Загружает agent из указанной папки.
    Возвращает agent function. Параметр name делает уникальное module имя."""
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
    agent_path = bundle_dir / 'agent.py'

    # Уникальное имя для каждого загружения
    spec = importlib.util.spec_from_file_location(name, agent_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, 'USE_MCTS'):
        mod.USE_MCTS = False
    return mod.agent


def run_match(agent_a, agent_b, seed: int) -> dict:
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
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--swap', action='store_true',
                    help='Запускать также с swapped colors для paired test')
    ap.add_argument('--start-seed', type=int, default=0)
    args = ap.parse_args()

    if not ML_DIR.exists():
        sys.exit(f'⚠ Нет {ML_DIR}. Запусти 20_setup_ml_bundle.py')
    if not BASELINE_DIR.exists():
        sys.exit(f'⚠ Нет {BASELINE_DIR}')

    print('Загружаю агентов из изолированных папок …')
    baseline_agent = _load_agent(BASELINE_DIR, '_baseline_mod')
    print(f'  baseline ← {BASELINE_DIR}')
    ml_agent = _load_agent(ML_DIR, '_ml_mod')
    print(f'  ml ← {ML_DIR}')

    results = []
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    print(f'\nЗапуск {len(seeds)} seeds {("× 2 (swap)" if args.swap else "")} матчей …')

    for seed in seeds:
        # baseline играет p0, ml — p1
        r1 = run_match(baseline_agent, ml_agent, seed=seed)
        r1['side'] = 'base_p0'
        r1['ml_wins'] = int(r1['reward_p1'] > r1['reward_p0'])
        r1['ml_ships_advantage'] = r1['ships_p1'] - r1['ships_p0']
        results.append(r1)
        print(f'  seed={seed} (base/p0 vs ml/p1)  '
              f'p0={r1["reward_p0"]:+d}  p1={r1["reward_p1"]:+d}  '
              f'ml_diff={r1["ml_ships_advantage"]:+d}  ({r1["time"]}s)')
        if args.swap:
            # ml играет p0, baseline — p1
            r2 = run_match(ml_agent, baseline_agent, seed=seed)
            r2['side'] = 'ml_p0'
            r2['ml_wins'] = int(r2['reward_p0'] > r2['reward_p1'])
            r2['ml_ships_advantage'] = r2['ships_p0'] - r2['ships_p1']
            results.append(r2)
            print(f'  seed={seed} (ml/p0 vs base/p1)  '
                  f'p0={r2["reward_p0"]:+d}  p1={r2["reward_p1"]:+d}  '
                  f'ml_diff={r2["ml_ships_advantage"]:+d}  ({r2["time"]}s)')

    df_res = pd.DataFrame(results)
    df_res.to_csv(DATA_DIR / 'tournament_proper_results.csv', index=False)

    # Сводка
    print('\n══════════ СВОДКА ══════════')
    n = len(df_res)
    ml_wins = df_res['ml_wins'].sum()
    ml_winrate = ml_wins / max(n, 1)
    avg_diff = df_res['ml_ships_advantage'].mean()

    # Welch t-test без scipy
    diffs = df_res['ml_ships_advantage'].values
    n_diffs = len(diffs)
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1) if n_diffs > 1 else 0
    se = std_d / math.sqrt(n_diffs) if n_diffs > 1 else 0
    t_stat = mean_d / se if se > 0 else 0
    from math import erf, sqrt
    p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))) if se > 0 else 1.0

    print(f'  Матчей: {n}')
    print(f'  ML winrate:                  {ml_winrate:.3f}  ({ml_wins}/{n})')
    print(f'  Средний ML ships advantage:  {avg_diff:+.1f}')
    print(f'  Std:                         {std_d:.1f}')
    print(f'  t-stat:                      {t_stat:+.2f}')
    print(f'  p-value (двусторон, t-tест): {p_val:.4f}')
    print(f'  Среднее время матча:         {df_res["time"].mean():.1f}s')

    # Per side breakdown
    if args.swap:
        print('\n  По стороне (paired):')
        for side in ['base_p0', 'ml_p0']:
            sub = df_res[df_res['side'] == side]
            wr = sub['ml_wins'].mean() if len(sub) else 0
            ad = sub['ml_ships_advantage'].mean() if len(sub) else 0
            print(f'    {side:<10}  n={len(sub):>3}  ml_wr={wr:.3f}  avg_diff={ad:+.1f}')

    # Markdown
    md = []
    md.append('# Tournament proper: baseline vs ML (изолированные bundles)\n\n')
    md.append(f'**Baseline**: agent_bundle_swarm 2\n')
    md.append(f'**ML**:       agent_bundle_ml (zones с inline ML residual)\n')
    md.append(f'**Seeds**: {args.seeds}{" × 2 (swap)" if args.swap else ""}\n\n')

    md.append('## Главные числа\n\n')
    md.append('| Metric | Value |\n|---|---|\n')
    md.append(f'| матчей | {n} |\n')
    md.append(f'| ML winrate | {ml_winrate:.3f} ({ml_wins}/{n}) |\n')
    md.append(f'| avg ml_ships_advantage | {avg_diff:+.1f} |\n')
    md.append(f'| std | {std_d:.1f} |\n')
    md.append(f'| t-stat | {t_stat:+.2f} |\n')
    md.append(f'| p-value | {p_val:.4f} |\n')

    md.append('\n## Интерпретация\n\n')
    if ml_winrate > 0.55 and p_val < 0.1:
        md.append('✅ **ML значимо лучше** baseline (winrate > 55%, p < 0.10)\n')
    elif ml_winrate < 0.45 and p_val < 0.1:
        md.append('❌ **ML значимо ХУЖЕ** baseline (winrate < 45%, p < 0.10).\n')
        md.append('   Residual интеграция ломает эвристику.\n')
    else:
        md.append('⚖️ **Нейтрально** или статистика недостаточна (нужно больше seeds)\n')

    (REPORT_DIR / 'tournament_proper_report.md').write_text(''.join(md))
    print(f'\n✓ Сохранено:')
    print(f'  results → {DATA_DIR / "tournament_proper_results.csv"}')
    print(f'  отчёт   → {REPORT_DIR / "tournament_proper_report.md"}')


if __name__ == '__main__':
    main()
