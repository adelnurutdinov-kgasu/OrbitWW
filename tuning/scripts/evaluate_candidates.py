#!/usr/bin/env python3
"""
evaluate_candidates.py — оценка кандидатов SwarmWeights на пачке сидов.

Зачем: после reverse_tournament у нас есть 8+ «победных» комбинаций весов
(по одной на проигранный сид). Median по ним — переобучение или шум, потому
что разные победители решают разные задачи и взаимно противоречат.

Что делаем:
  1. Берём CSV от reverse_tournament.
  2. Строим список КАНДИДАТОВ:
       - default SwarmWeights
       - каждый победный trial-row (по одному на seed)
       - агрегаты: median-of-winners, mean-top-K-by-ships_diff, ridge-argmax,
         soft-weighted (exp-нормированные top-30)
  3. Прогоняем КАЖДОГО кандидата на одинаковом наборе EVAL_SEEDS.
  4. Отчёт: winrate, avg ships_diff, кто решил какие сиды.

Это даёт честное сравнение: какой РЕАЛЬНО обобщается, а какой — переобучен.

ЗАПУСК:
    python3 tuning/scripts/evaluate_candidates.py [PATH_TO_CSV]
"""

import os, sys, importlib.util, time, math, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from dataclasses import asdict, fields as dc_fields

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ══════════════════════════════════════════════════════════════════════════

# На каких сидах оценивать. Лучше расширить за рамки train (smaller bias).
EVAL_SEEDS = list(range(1, 51))   # 50 сидов — компромисс между шумом и временем
RUN_BACKEND = 'kaggle'            # 'kaggle' | 'local' (см. reverse_tournament.py)
N_WORKERS   = max(1, os.cpu_count() // 2)

# Сколько top по ships_diff брать в mean-top, ridge, soft
TOP_K_TRIALS = 20

# ══════════════════════════════════════════════════════════════════════════

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "agent_bundle_swarm"))

from reverse_tournament import run_one, _load_module          # noqa
from swarm import SwarmWeights, DEFAULT_WEIGHTS               # noqa

W_NAMES = [f.name for f in dc_fields(SwarmWeights)]
SEARCH_KEYS = ['activity_weight','idle_floor','distance_comfort','risk_tolerance',
               'ships_weight','eta_bonus','priority_bonus','stress_top_k','stress_gamma']
INT_KEYS = {'idle_floor','risk_tolerance','stress_top_k'}


def _row_to_overrides(row):
    """trial-row → overrides dict (только колонки w_*)."""
    out = {}
    for k in SEARCH_KEYS:
        col = f'w_{k}'
        if col in row and not pd.isna(row[col]):
            v = row[col]
            if k in INT_KEYS:
                v = int(round(v))
            else:
                v = float(v)
            out[k] = v
    return out


def _build_aggregates(trials, top_k=TOP_K_TRIALS):
    """Несколько агрегатов из таблицы 'phase∈{1,2}' trial'ов."""
    out = {}
    wins = trials[trials['win'] == 1]
    if len(wins) > 0:
        out['median_winners'] = {k: float(wins[f'w_{k}'].median()) for k in SEARCH_KEYS}

    top = trials.nlargest(top_k, 'ships_diff')
    if len(top) > 0:
        out[f'mean_top{top_k}'] = {k: float(top[f'w_{k}'].mean()) for k in SEARCH_KEYS}
        # soft-weighted (exp z-score по ships_diff)
        sd = top['ships_diff'].values
        z = (sd - sd.mean()) / (sd.std() + 1e-9)
        w = np.exp(np.clip(z, -3, 3)); w /= w.sum()
        out[f'soft_top{top_k}'] = {k: float((top[f'w_{k}'].values * w).sum()) for k in SEARCH_KEYS}

    # ridge-argmax (берёт края диапазона по знакам коэф.)
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        X = trials[[f'w_{k}' for k in SEARCH_KEYS]].values
        y = trials['ships_diff'].values
        sc = StandardScaler().fit(X)
        r = Ridge(alpha=1.0).fit(sc.transform(X), y)
        bounds = {
            'activity_weight':(0,8),'idle_floor':(10,40),'distance_comfort':(0,0.7),
            'risk_tolerance':(0,2),'ships_weight':(0,8),'eta_bonus':(5,100),
            'priority_bonus':(0,6),'stress_top_k':(3,6),'stress_gamma':(0.1,1.9),
        }
        ridge = {}
        for k, c in zip(SEARCH_KEYS, r.coef_):
            lo, hi = bounds[k]
            ridge[k] = (hi if c > 0 else lo)
        out['ridge_argmax'] = ridge
    except Exception as e:
        print(f"(ridge skipped: {e})", file=sys.stderr)

    # типизируем int-поля
    for name, w in out.items():
        for k in INT_KEYS:
            if k in w:
                w[k] = int(round(w[k]))
    return out


def _candidates_from_csv(csv_path, top_k=TOP_K_TRIALS, fallback_top_n=8):
    """Кандидатов в порядке: default, per-winning-seed, top-by-diff (fallback), агрегаты."""
    df = pd.read_csv(csv_path)
    trials = df[df['phase'].isin([1, 2])].copy()
    print(f"  CSV trials (phase∈{{1,2}}): {len(trials)}  wins: {(trials['win']==1).sum()}")

    cands = {'default': {}}

    # 1. уникальные победители: для каждого seed первый winning trial
    wins = trials[trials['win'] == 1].sort_values('seed')
    seen_seeds = set()
    for _, row in wins.iterrows():
        sd = int(row['seed'])
        if sd in seen_seeds: continue
        seen_seeds.add(sd)
        cands[f'winner_seed{sd}'] = _row_to_overrides(row)

    # 2. fallback: если победителей мало (<3), берём top-N по ships_diff
    if len(cands) - 1 < 3 and len(trials) > 0:
        print(f"  ⚠ победителей мало ({len(cands)-1}), добавляю top-{fallback_top_n} по ships_diff")
        top = trials.nlargest(fallback_top_n, 'ships_diff')
        for i, (_, row) in enumerate(top.iterrows()):
            sd = int(row['seed'])
            diff = int(row['ships_diff'])
            name = f'top_seed{sd}_diff{diff:+d}_{i}'
            if name not in cands:
                cands[name] = _row_to_overrides(row)

    # 3. агрегаты
    for name, ov in _build_aggregates(trials, top_k=top_k).items():
        cands[name] = ov
    return cands


def _eval_one(task):
    """Выполняем run_one из reverse_tournament (тот же backend, тот же конфиг)."""
    return run_one(task)


def evaluate_candidate(name, overrides, seeds):
    """Прогоняет кандидата на всех seeds (паралельно), возвращает aggregated stats."""
    tasks = [(s, dict(overrides)) for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_eval_one, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                seed, _ = futures[fut]
                print(f"  [{name}] seed={seed} ERROR: {e}", file=sys.stderr)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        'name':         name,
        'overrides':    overrides,
        'n':            len(df),
        'wins':         int(df['win'].sum()),
        'winrate':      df['win'].mean(),
        'avg_diff':     df['ships_diff'].mean(),
        'median_diff':  df['ships_diff'].median(),
        'min_diff':     int(df['ships_diff'].min()),
        'max_diff':     int(df['ships_diff'].max()),
        'won_seeds':    sorted(df[df['win'] == 1]['seed'].tolist()),
    }


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # последний reverse_tour CSV
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "reverse_tour_*.csv")))
        if not files:
            print("Нет reverse_tour_*.csv в", RESULTS_DIR); return
        csv_path = files[-1]
    print(f"CSV: {csv_path}")

    cands = _candidates_from_csv(csv_path)
    print(f"кандидатов: {len(cands)}")
    for name in cands:
        print(f"  - {name}")
    print(f"\nоцениваем на {len(EVAL_SEEDS)} сидах: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}")
    print(f"backend: {RUN_BACKEND}, workers: {N_WORKERS}")

    rows = []
    t0 = time.time()
    for i, (name, ov) in enumerate(cands.items(), 1):
        t1 = time.time()
        res = evaluate_candidate(name, ov, EVAL_SEEDS)
        if res is None: continue
        rows.append(res)
        print(f"\n[{i}/{len(cands)}] {name}: WR={res['winrate']:.3f} ({res['wins']}/{res['n']})  "
              f"avg_diff={res['avg_diff']:+.0f}  med_diff={res['median_diff']:+.0f}  "
              f"({time.time()-t1:.0f}s)")

    if not rows:
        print("ничего не оценилось"); return

    df = pd.DataFrame(rows).sort_values(['winrate','avg_diff'], ascending=False)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f"candidates_eval_{ts}.csv")

    # Раскладываем overrides по колонкам для CSV
    flat_rows = []
    for r in rows:
        ov = r.pop('overrides')
        wons = r.pop('won_seeds')
        r2 = dict(r)
        for k, v in ov.items():
            r2[f'w_{k}'] = v
        r2['won_seeds'] = ','.join(map(str, wons))
        flat_rows.append(r2)
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print(f"\n=== РАНЖИРОВАНИЕ (по winrate, потом avg_diff) ===")
    print(df[['name','wins','n','winrate','avg_diff','median_diff']].to_string(index=False))
    print(f"\nsaved: {out_csv}")
    print(f"всего: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
