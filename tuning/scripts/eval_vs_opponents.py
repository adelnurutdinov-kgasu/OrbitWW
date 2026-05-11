#!/usr/bin/env python3
"""
eval_vs_opponents.py — наши 9 candidates × {sub2, noop} на пачке сидов.

Зачем: проверить, **зависит ли** оптимальная SwarmWeights от типа оппонента.
Если winner_seed20 хорош и против sub2 (стратегический), и против noop —
opponent-aware routing бесполезен. Если разные кандидаты лучшие против
разных opp — есть смысл в type-based reasoning.

Запуск:  python3 tuning/scripts/eval_vs_opponents.py
Результат: tuning/results/opp_eval_<ts>.csv + pivot summary.
"""

import os, sys, importlib.util, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields as dc_fields
from datetime import datetime

import pandas as pd

# ── КОНФИГ ──────────────────────────────────────────────────────────────
EVAL_SEEDS  = list(range(1, 31))           # 30 сидов (компромисс времени)
OPPONENTS   = ['sub2', 'noop']             # типы оппонентов
N_WORKERS   = max(1, os.cpu_count() // 2)
# ───────────────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BUNDLE_DIR = os.path.join(PROJECT_ROOT, "agent_bundle_swarm")
OUR_PATH   = os.path.join(BUNDLE_DIR, "agent.py")
SUB_PATH   = os.path.join(PROJECT_ROOT, "sub2.py")

sys.path.insert(0, BUNDLE_DIR)
sys.path.insert(0, HERE)

# Импортируем WINNERS из eval_8_winners (не выполняет main, т.к. __name__ != '__main__')
from eval_8_winners import CANDIDATES


def _load_module(path, name):
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def noop_agent(obs, config=None):
    """Бот-бездействие: ничего не делает. Тест на 'способность капитализировать'."""
    return []


def _silence_debug_logs():
    for k in ("ORBIT_AGENT_LOG", "ORBIT_AGENT_LOG_PLANS_ALL",
              "ORBIT_AGENT_LOG_FLEETS",
              "SUB_AGENT_LOG", "SUB_AGENT_LOG_MISSIONS_ALL",
              "SUB_AGENT_LOG_FLEETS"):
        os.environ.pop(k, None)


def run_one(task):
    """task = (seed, cand_name, cand_overrides, opp_name) → dict с результатом."""
    seed, cname, cov, opp_name = task
    _silence_debug_logs()

    suf = f"{seed}_{cname}_{opp_name}_{abs(hash(frozenset(cov.items())))%10**6}"

    # Наш агент с инжекцией SwarmWeights
    our_mod = _load_module(OUR_PATH, f"_our_{suf}")
    from swarm import SwarmWeights
    base = asdict(SwarmWeights())
    base.update(cov)
    for f in dc_fields(SwarmWeights):
        if f.type is int and f.name in base:
            base[f.name] = int(base[f.name])
    our_mod.SWARM_WEIGHTS = SwarmWeights(**base)

    # Оппонент
    if opp_name == 'sub2':
        opp_mod = _load_module(SUB_PATH, f"_sub_{suf}")
        opp_fn = opp_mod.agent
    elif opp_name == 'noop':
        opp_fn = noop_agent
    else:
        raise ValueError(f"unknown opponent: {opp_name}")

    # Матч через kaggle
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.time()
    env.run([our_mod.agent, opp_fn])
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
        'candidate':  cname,
        'opponent':   opp_name,
        'win':        int(r0 > r1),
        'draw':       int(r0 == r1),
        'ships_our':  ships_p0,
        'ships_opp':  ships_p1,
        'ships_diff': ships_p0 - ships_p1,
        'steps':      len(env.steps),
        'time_sec':   round(elapsed, 1),
    }


def main():
    tasks = [(s, cname, dict(cov), opp)
             for opp in OPPONENTS
             for cname, cov in CANDIDATES.items()
             for s in EVAL_SEEDS]

    print(f"Tasks: {len(tasks)} = {len(EVAL_SEEDS)}seeds × "
          f"{len(CANDIDATES)}cand × {len(OPPONENTS)}opp")
    print(f"Workers: {N_WORKERS}\n")

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
                if done % 30 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    eta_min = elapsed / done * (len(tasks) - done) / 60
                    print(f"  [{done}/{len(tasks)}] {r['candidate']:<14} vs {r['opponent']:<5} "
                          f"seed={r['seed']:>2} "
                          f"{'W' if r['win'] else ('D' if r['draw'] else 'L')} "
                          f"diff={r['ships_diff']:+5d}  | "
                          f"elapsed={elapsed:.0f}s eta={eta_min:.0f}min")
            except Exception as e:
                seed, cname, _, opp = futs[fut]
                print(f"  ERROR seed={seed} cand={cname} opp={opp}: {e}", file=sys.stderr)

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f"opp_eval_{ts}.csv")
    df.to_csv(out_csv, index=False)

    # Pivots
    print("\n=== WINRATE: candidate × opponent ===")
    pivot = df.pivot_table(index='candidate', columns='opponent',
                           values='win', aggfunc='mean').round(3)
    pivot['Δ(sub−noop)'] = pivot.get('sub2', 0) - pivot.get('noop', 0)
    pivot = pivot.sort_values(by='sub2' if 'sub2' in pivot.columns else pivot.columns[0],
                              ascending=False)
    print(pivot.to_string())

    print("\n=== AVG SHIPS_DIFF ===")
    pivot_d = df.pivot_table(index='candidate', columns='opponent',
                             values='ships_diff', aggfunc='mean').round(0)
    print(pivot_d.to_string())

    # Какой кандидат лучший vs каждого оппонента
    print("\n=== BEST CANDIDATE per opponent ===")
    for opp in OPPONENTS:
        subset = df[df['opponent'] == opp]
        per_cand = subset.groupby('candidate')['win'].mean().sort_values(ascending=False)
        best = per_cand.idxmax()
        worst = per_cand.idxmin()
        print(f"  vs {opp:<5}:  BEST={best} ({per_cand[best]:.3f})  "
              f"WORST={worst} ({per_cand[worst]:.3f})  "
              f"spread={per_cand[best]-per_cand[worst]:.3f}")

    print(f"\nsaved: {out_csv}")
    print(f"total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
