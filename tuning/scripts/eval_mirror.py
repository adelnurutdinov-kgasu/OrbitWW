#!/usr/bin/env python3
"""
eval_mirror.py — empirical game-theoretic analysis: 9 наших policies друг
против друга. Получаем 9×9 payoff matrix (winrate и avg ships_diff).

Цель: понять структуру наших policies.
  - Кто **доминирует** (всех бьёт).
  - Есть ли **rock-paper-scissors** циклы (A>B>C>A).
  - **Robust policy** (минимальный спред в исходах).
  - **Best-on-average** (стартовая для adaptive switcher).

Это empirical game-theoretic analysis (Wellman 2006).

Запуск:  python3 tuning/scripts/eval_mirror.py
Выход:   tuning/results/mirror_<ts>.csv + pivot summary в stdout.
"""

import os, sys, importlib.util, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields as dc_fields
from datetime import datetime

import pandas as pd
import numpy as np

# ── КОНФИГ ──────────────────────────────────────────────────────────────
EVAL_SEEDS  = list(range(1, 16))         # 15 сидов на ячейку (9×9×15 = 1215 матчей)
N_WORKERS   = max(1, os.cpu_count() // 2)
# ───────────────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BUNDLE_DIR = os.path.join(PROJECT_ROOT, "agent_bundle_swarm")
OUR_PATH   = os.path.join(BUNDLE_DIR, "agent.py")

sys.path.insert(0, BUNDLE_DIR)
sys.path.insert(0, HERE)

from eval_8_winners import CANDIDATES   # 9 policies (default + 8 winners)


def _load_module(path, name):
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _silence_debug():
    for k in ("ORBIT_AGENT_LOG", "ORBIT_AGENT_LOG_PLANS_ALL",
              "ORBIT_AGENT_LOG_FLEETS",
              "SUB_AGENT_LOG", "SUB_AGENT_LOG_MISSIONS_ALL",
              "SUB_AGENT_LOG_FLEETS"):
        os.environ.pop(k, None)


def _make_agent(suffix, overrides):
    """Загружает agent.py с инжектированными SwarmWeights — возвращает функцию."""
    mod = _load_module(OUR_PATH, f"_agent_{suffix}")
    from swarm import SwarmWeights
    base = asdict(SwarmWeights())
    base.update(overrides)
    for f in dc_fields(SwarmWeights):
        if f.type is int and f.name in base:
            base[f.name] = int(base[f.name])
    mod.SWARM_WEIGHTS = SwarmWeights(**base)
    return mod.agent


def run_one(task):
    """task = (seed, row_name, row_ov, col_name, col_ov) → результат."""
    seed, row_name, row_ov, col_name, col_ov = task
    _silence_debug()

    suf_row = f"row_{seed}_{row_name}_{abs(hash(frozenset(row_ov.items())))%10**6}"
    suf_col = f"col_{seed}_{col_name}_{abs(hash(frozenset(col_ov.items())))%10**6}"
    agent_row = _make_agent(suf_row, row_ov)
    agent_col = _make_agent(suf_col, col_ov)

    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.time()
    env.run([agent_row, agent_col])
    elapsed = time.time() - t0

    final = env.steps[-1]
    r0 = final[0].get('reward', 0) or 0
    r1 = final[1].get('reward', 0) or 0
    obs = final[0].get('observation') or {}
    planets = obs.get('planets') or []
    ships_p0 = sum((p[5] or 0) for p in planets if p[1] == 0)
    ships_p1 = sum((p[5] or 0) for p in planets if p[1] == 1)

    return {
        'seed':       seed,
        'row':        row_name,
        'col':        col_name,
        'win':        int(r0 > r1),       # выиграл ли row (player 0)
        'draw':       int(r0 == r1),
        'ships_row':  ships_p0,
        'ships_col':  ships_p1,
        'ships_diff': ships_p0 - ships_p1,
        'steps':      len(env.steps),
        'time_sec':   round(elapsed, 1),
    }


def main():
    names = list(CANDIDATES.keys())
    n = len(names)
    tasks = [
        (s, rn, dict(CANDIDATES[rn]), cn, dict(CANDIDATES[cn]))
        for rn in names
        for cn in names
        for s in EVAL_SEEDS
    ]

    print(f"Mirror tournament: {n}×{n} = {n*n} cells, {len(EVAL_SEEDS)} seeds/cell")
    print(f"Total tasks: {len(tasks)}, workers: {N_WORKERS}\n")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                rows.append(r)
                if done % 50 == 0 or done == len(tasks):
                    el = time.time() - t0
                    eta = el / done * (len(tasks) - done) / 60
                    print(f"  [{done}/{len(tasks)}] {r['row']:<14} vs {r['col']:<14} "
                          f"seed={r['seed']:>2} "
                          f"{'W' if r['win'] else ('D' if r['draw'] else 'L')} "
                          f"diff={r['ships_diff']:+5d} | "
                          f"elapsed={el:.0f}s eta={eta:.0f}min")
            except Exception as e:
                seed, rn, _, cn, _ = futs[fut]
                print(f"  ERROR seed={seed} {rn} vs {cn}: {e}", file=sys.stderr)

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f"mirror_{ts}.csv")
    df.to_csv(out_csv, index=False)

    # ── ANALYSIS ───────────────────────────────────────────────────────
    # 1. WR matrix
    wr = df.pivot_table(index='row', columns='col', values='win', aggfunc='mean').round(2)
    wr = wr.reindex(index=names, columns=names)
    print("\n=== WINRATE matrix (row vs col) ===")
    print(wr.to_string())

    # 2. Avg ships_diff matrix
    sd = df.pivot_table(index='row', columns='col',
                        values='ships_diff', aggfunc='mean').round(0)
    sd = sd.reindex(index=names, columns=names)
    print("\n=== AVG SHIPS_DIFF matrix (row vs col) ===")
    print(sd.to_string())

    # 3. Best-on-average: средний WR по строке = насколько хорош vs всех
    avg_wr = wr.mean(axis=1).sort_values(ascending=False)
    print("\n=== AVG WINRATE (row vs all cols) — best на старте ===")
    for name, v in avg_wr.items():
        print(f"  {name:<16}  {v:.3f}")

    # 4. Best counter — для каждой col кто её лучше всех бьёт
    print("\n=== BEST COUNTER per opponent column ===")
    for col in names:
        col_wr = wr[col].drop(col, errors='ignore')   # исключаем self vs self
        if col_wr.empty: continue
        best = col_wr.idxmax()
        worst = col_wr.idxmin()
        print(f"  vs {col:<16}  BEST counter={best} ({col_wr[best]:.2f})  "
              f"WORST={worst} ({col_wr[worst]:.2f})")

    # 5. Rock-paper-scissors detection: ищем тройки A→B→C→A
    print("\n=== ROCK-PAPER-SCISSORS cycles (если есть) ===")
    cycles = []
    for a in names:
        for b in names:
            if a >= b: continue
            for c in names:
                if c == a or c == b: continue
                # A бьёт B, B бьёт C, C бьёт A?
                if (wr.loc[a, b] > 0.55 and wr.loc[b, c] > 0.55
                        and wr.loc[c, a] > 0.55):
                    cycles.append((a, b, c))
    if cycles:
        for a, b, c in cycles[:10]:
            print(f"  {a} > {b} > {c} > {a}  "
                  f"(WR: {wr.loc[a,b]:.2f}, {wr.loc[b,c]:.2f}, {wr.loc[c,a]:.2f})")
        if len(cycles) > 10:
            print(f"  ... ещё {len(cycles)-10}")
    else:
        print("  нет циклов (один доминирует) → opponent-aware routing не критичен")

    # 6. Robust policy: минимальный spread
    print("\n=== ROBUSTNESS (spread по строке): меньше = устойчивее ===")
    spread = (wr.max(axis=1) - wr.min(axis=1)).sort_values()
    for name, v in spread.items():
        print(f"  {name:<16}  spread={v:.2f}  (best={wr.loc[name].max():.2f} "
              f"worst={wr.loc[name].min():.2f})")

    print(f"\nsaved: {out_csv}")
    print(f"total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
