#!/usr/bin/env python3
"""
eval_8_winners.py — оценка 8 ЗАХАРДКОЖЕННЫХ победных конфигов на пачке сидов.

Никакого CSV loading: 8 победных конфигов из reverse_tour_20260501_235618
вписаны прямо здесь. Плюс 'default' для baseline-сравнения.

Запуск:  python3 tuning/scripts/eval_8_winners.py
"""

import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ══════════════════════════════════════════════════════════════════════════

EVAL_SEEDS  = list(range(1, 101))   # 50 сидов
N_WORKERS   = max(1, os.cpu_count() // 2)

# 8 победителей из reverse_tour_20260501_235618.csv. Каждый победил
# на «своём» seed где baseline проигрывал.
WINNERS = {
    'winner_seed4': dict(  # diff=+1680
        activity_weight=2.27, idle_floor=13, distance_comfort=0.26,
        risk_tolerance=0, ships_weight=2.93, eta_bonus=76.37,
        priority_bonus=3.41, stress_top_k=4, stress_gamma=1.40,
    ),
    'winner_seed11': dict(  # diff=+2694
        activity_weight=4.94, idle_floor=38, distance_comfort=0.15,
        risk_tolerance=2, ships_weight=5.69, eta_bonus=94.01,
        priority_bonus=3.12, stress_top_k=6, stress_gamma=1.41,
    ),
    'winner_seed19': dict(  # diff=+5048
        activity_weight=2.14, idle_floor=38, distance_comfort=0.28,
        risk_tolerance=0, ships_weight=0.57, eta_bonus=51.96,
        priority_bonus=5.63, stress_top_k=5, stress_gamma=1.27,
    ),
    'winner_seed20': dict(  # diff=+2365
        activity_weight=1.29, idle_floor=40, distance_comfort=0.18,
        risk_tolerance=2, ships_weight=1.37, eta_bonus=16.66,
        priority_bonus=2.84, stress_top_k=3, stress_gamma=1.50,
    ),
    'winner_seed22': dict(  # diff=+4220
        activity_weight=3.18, idle_floor=16, distance_comfort=0.21,
        risk_tolerance=0, ships_weight=4.40, eta_bonus=39.09,
        priority_bonus=1.84, stress_top_k=5, stress_gamma=1.34,
    ),
    'winner_seed36': dict(  # diff=+2418
        activity_weight=7.08, idle_floor=22, distance_comfort=0.54,
        risk_tolerance=0, ships_weight=7.38, eta_bonus=49.76,
        priority_bonus=5.85, stress_top_k=3, stress_gamma=1.60,
    ),
    'winner_seed39': dict(  # diff=+4716
        activity_weight=2.79, idle_floor=37, distance_comfort=0.19,
        risk_tolerance=1, ships_weight=5.76, eta_bonus=14.33,
        priority_bonus=0.37, stress_top_k=6, stress_gamma=1.29,
    ),
    'winner_seed47': dict(  # diff=+4783
        activity_weight=1.32, idle_floor=26, distance_comfort=0.21,
        risk_tolerance=0, ships_weight=2.94, eta_bonus=58.06,
        priority_bonus=2.71, stress_top_k=5, stress_gamma=0.64,
    ),
}

# 'default' = пустой override → SwarmWeights() дефолты.
CANDIDATES = {'default': {}, **WINNERS}

# ══════════════════════════════════════════════════════════════════════════

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "agent_bundle_swarm"))

# Берём run_one из reverse_tournament — он уже умеет инжектить SwarmWeights
# и вызывать kaggle backend (RUN_BACKEND='kaggle' по дефолту).
from reverse_tournament import run_one


def evaluate(name, ov, seeds):
    tasks = [(s, dict(ov)) for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                seed, _ = futs[fut]
                print(f"  [{name}] seed={seed} ERROR: {e}", file=sys.stderr)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        'name':        name,
        'overrides':   ov,
        'n':           len(df),
        'wins':        int(df['win'].sum()),
        'winrate':     df['win'].mean(),
        'avg_diff':    df['ships_diff'].mean(),
        'median_diff': df['ships_diff'].median(),
        'min_diff':    int(df['ships_diff'].min()),
        'max_diff':    int(df['ships_diff'].max()),
        'won_seeds':   sorted(df[df['win'] == 1]['seed'].tolist()),
    }


def main():
    print(f"кандидатов: {len(CANDIDATES)}")
    for n in CANDIDATES: print(f"  - {n}")
    print(f"\nоцениваем на {len(EVAL_SEEDS)} сидах: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}")
    print(f"workers: {N_WORKERS}\n")

    rows = []
    t0 = time.time()
    for i, (name, ov) in enumerate(CANDIDATES.items(), 1):
        t1 = time.time()
        res = evaluate(name, ov, EVAL_SEEDS)
        if res is None: continue
        rows.append(res)
        print(f"[{i}/{len(CANDIDATES)}] {name}: WR={res['winrate']:.3f} "
              f"({res['wins']}/{res['n']})  avg={res['avg_diff']:+.0f}  "
              f"med={res['median_diff']:+.0f}  ({time.time()-t1:.0f}s)")

    if not rows:
        print("ничего не оценилось"); return

    df = pd.DataFrame(rows).sort_values(['winrate','avg_diff'], ascending=False)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f"eval_8winners_{ts}.csv")

    flat = []
    for r in rows:
        ov = r.pop('overrides'); won = r.pop('won_seeds')
        r2 = dict(r)
        for k, v in ov.items(): r2[f'w_{k}'] = v
        r2['won_seeds'] = ','.join(map(str, won))
        flat.append(r2)
    pd.DataFrame(flat).to_csv(out_csv, index=False)

    print(f"\n=== РАНЖИРОВАНИЕ ===")
    print(df[['name','wins','n','winrate','avg_diff','median_diff']].to_string(index=False))
    print(f"\nsaved: {out_csv}")
    print(f"всего: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
