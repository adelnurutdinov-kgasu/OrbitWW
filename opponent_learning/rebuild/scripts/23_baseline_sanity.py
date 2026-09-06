#!/usr/bin/env python3
"""
23_baseline_sanity.py — sanity check: есть ли асимметрия p0 vs p1 у baseline?

ЦЕЛЬ
────
В 21_tournament_proper мы увидели:
  ML как p1: winrate 53.3%
  ML как p0: winrate 26.7%   ← сильная асимметрия
Это либо (A) асимметрия игры (first-move advantage), либо (B) проблема в
нашей ML integration.

ПРОВЕРКА
────────
Запускаем baseline_A vs baseline_B где оба — копии agent_bundle_swarm 2.
Через изолированные модули.

  - Если winrate ≈ 50/50 → асимметрия специфична для ML (B), нужно чинить
  - Если winrate ≈ 53/27 (как у ML) → асимметрия самой игры (A)

ВЫЗОВ
─────
  python3 23_baseline_sanity.py --seeds 15
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import time
from math import erf, sqrt
from pathlib import Path

import pandas as pd

try:
    from kaggle_environments import make
except ImportError:
    sys.exit('kaggle_environments не установлен')

HERE = Path(__file__).resolve().parent
REBUILD = HERE.parent
OL_DIR = REBUILD.parent
PROJECT_ROOT = OL_DIR.parent
DATA_DIR = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = PROJECT_ROOT / "agent_bundle_swarm 2"
COPY_DIR     = PROJECT_ROOT / "agent_bundle_swarm_copy"


def _load_agent(bundle_dir: Path, name: str):
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
    spec = importlib.util.spec_from_file_location(name, bundle_dir / 'agent.py')
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
    t = time.time() - t0
    final = env.steps[-1]
    r0 = final[0].get('reward', 0) or 0
    r1 = final[1].get('reward', 0) or 0
    obs = final[0].get('observation') or {}
    planets = obs.get('planets') or []
    ships_p0 = sum((p[5] or 0) for p in planets if p[1] == 0)
    ships_p1 = sum((p[5] or 0) for p in planets if p[1] == 1)
    return {
        'seed': seed, 'reward_p0': r0, 'reward_p1': r1,
        'win_p0': int(r0 > r1), 'draw': int(r0 == r1),
        'ships_p0': ships_p0, 'ships_p1': ships_p1,
        'time': round(t, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    args = ap.parse_args()

    # Копируем baseline в "agent_bundle_swarm_copy" чтобы было два разных модуля
    if not COPY_DIR.exists():
        print(f'Копирую {BASELINE_DIR.name} → {COPY_DIR.name} …')
        shutil.copytree(BASELINE_DIR, COPY_DIR,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc',
                                                        '.DS_Store',
                                                        '.ipynb_checkpoints'))

    print('Загружаю агентов из обеих папок (одинаковая логика, разные модули) …')
    agent_a = _load_agent(BASELINE_DIR, '_baseline_a')
    agent_b = _load_agent(COPY_DIR, '_baseline_b')

    results = []
    print(f'\nЗапуск {args.seeds} seeds × 2 swap …')
    for seed in range(args.seeds):
        # A на p0, B на p1
        r1 = run_match(agent_a, agent_b, seed=seed)
        r1['side'] = 'A_p0'
        results.append(r1)
        print(f'  seed={seed} (A/p0 vs B/p1)  '
              f'p0={r1["reward_p0"]:+d}  p1={r1["reward_p1"]:+d}  '
              f'({r1["time"]}s)')
        # swap: B на p0, A на p1
        r2 = run_match(agent_b, agent_a, seed=seed)
        r2['side'] = 'B_p0'
        results.append(r2)
        print(f'  seed={seed} (B/p0 vs A/p1)  '
              f'p0={r2["reward_p0"]:+d}  p1={r2["reward_p1"]:+d}  '
              f'({r2["time"]}s)')

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / 'baseline_sanity_results.csv', index=False)

    print('\n══════════ СВОДКА ══════════')
    n = len(df)
    p0_wins = df['win_p0'].sum()
    p0_wr = p0_wins / n
    print(f'  Матчей: {n}')
    print(f'  Всего p0 wins: {p0_wins}/{n}  ({p0_wr:.3f})')
    print(f'  Если игра симметрична — должно быть ~ 0.5 (с шумом)')

    # По стороне (где A и B)
    for side in ['A_p0', 'B_p0']:
        sub = df[df['side'] == side]
        a_wins = sub['win_p0'].sum()  # когда side='A_p0', p0=A; когда side='B_p0', p0=B
        print(f'  {side:<8}  n={len(sub):>3}  p0_wr={a_wins/len(sub):.3f}')

    # ships diff
    print(f'\n  Среднее ships_p0 - ships_p1:    {(df["ships_p0"] - df["ships_p1"]).mean():+.1f}')
    diffs = df['ships_p0'].values - df['ships_p1'].values
    if len(diffs) > 1:
        mean_d = diffs.mean()
        std_d = diffs.std(ddof=1)
        se = std_d / (n ** 0.5)
        t_stat = mean_d / se if se > 0 else 0
        p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))) if se > 0 else 1
        print(f'  t-stat (p0 advantage): {t_stat:+.2f}  p-value: {p_val:.4f}')

    print(f'\n  ИНТЕРПРЕТАЦИЯ:')
    if abs(p0_wr - 0.5) < 0.10:
        print(f'  ✅ Игра симметрична (winrate близко к 0.5)')
        print(f'     → Асимметрия в 21_tournament_proper исходит из ML integration')
        print(f'     → Нужно отлаживать residual')
    else:
        print(f'  ⚠ Есть first-move/side advantage (winrate далеко от 0.5)')
        print(f'     → Асимметрия в 21 может быть частью самой игры')
        print(f'     → ML результат нужно интерпретировать с этим знанием')

    md = []
    md.append('# Baseline sanity: есть ли first-move advantage?\n\n')
    md.append(f'**Матчей**: {n}\n\n')
    md.append('## Главные числа\n\n')
    md.append('| Metric | Value |\n|---|---|\n')
    md.append(f'| p0 winrate (всего) | {p0_wr:.3f} |\n')
    md.append(f'| mean ships_p0 - ships_p1 | {diffs.mean():+.1f} |\n')
    if len(diffs) > 1:
        md.append(f'| t-stat | {t_stat:+.2f} |\n')
        md.append(f'| p-value | {p_val:.4f} |\n')
    md.append('\n## По стороне\n\n')
    for side in ['A_p0', 'B_p0']:
        sub = df[df['side'] == side]
        wr = sub['win_p0'].mean() if len(sub) else 0
        md.append(f'- `{side}`: p0_wr = {wr:.3f}, n={len(sub)}\n')

    (REPORT_DIR / 'baseline_sanity_report.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR / "baseline_sanity_report.md"}')


if __name__ == '__main__':
    main()
