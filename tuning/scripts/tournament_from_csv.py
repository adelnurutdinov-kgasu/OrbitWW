#!/usr/bin/env python3
"""
tournament_from_csv.py — круговой турнир между кандидатами (свои против своих).
Каждый кандидат загружается как agent.py с разными SwarmWeights.
"""

import os
import sys
import importlib.util
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields as dc_fields
from datetime import datetime

# ========== КОНФИГ ========================================================
CSV_CANDIDATES = "/Users/adel/Documents/GitHub/OrbitWW/tuning/results/eval_8winners_20260502_050430.csv"
N_MATCHES_PER_PAIR = 20
SEED_OFFSET = 1000
RUN_BACKEND = 'kaggle'          # 'kaggle' или 'local'
MAX_STEPS_PER_MATCH = 300
N_WORKERS = max(1, os.cpu_count() // 2)

# Путь к папке с агентами и к файлу agent.py
PROJECT_ROOT = "/Users/adel/Documents/GitHub/OrbitWW"
AGENT_BUNDLE_PATH = os.path.join(PROJECT_ROOT, "agent_bundle_swarm")
AGENT_FILE = os.path.join(AGENT_BUNDLE_PATH, "agent.py")
SWARM_FILE = os.path.join(AGENT_BUNDLE_PATH, "swarm.py")   # для импорта SwarmWeights

RESULTS_DIR = os.path.join(PROJECT_ROOT, "tuning", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_CSV = os.path.join(RESULTS_DIR, f"tournament_{TS}.csv")
SUMMARY_TXT = os.path.join(RESULTS_DIR, f"tournament_summary_{TS}.txt")
# ==========================================================================

def _silence_debug_logs():
    for k in ("ORBIT_AGENT_LOG", "ORBIT_AGENT_LOG_PLANS_ALL",
              "ORBIT_AGENT_LOG_FLEETS"):
        os.environ.pop(k, None)

def _load_module(path, name):
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def _load_agent_module(weights_override, suffix):
    """Загружает agent.py и инжектирует SwarmWeights."""
    # Добавляем bundle в путь, чтобы работали импорты внутри agent.py (например, from swarm import ...)
    if AGENT_BUNDLE_PATH not in sys.path:
        sys.path.insert(0, AGENT_BUNDLE_PATH)
    
    # Принудительно загружаем swarm.py, чтобы класс SwarmWeights был доступен
    # (agent.py, скорее всего, делает "from swarm import SwarmWeights")
    if os.path.exists(SWARM_FILE):
        _load_module(SWARM_FILE, "swarm")
    
    mod_name = f"_tourney_agent_{suffix}"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, AGENT_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    
    # Теперь берём класс SwarmWeights (из загруженного модуля swarm или из mod, если он там определён)
    try:
        from swarm import SwarmWeights
    except ImportError:
        # Возможно, SwarmWeights определён внутри agent.py
        if hasattr(mod, 'SwarmWeights'):
            SwarmWeights = mod.SwarmWeights
        else:
            raise RuntimeError("Не найден класс SwarmWeights. Проверьте agent_bundle_swarm.")
    
    base = asdict(SwarmWeights())
    base.update(weights_override)
    for f in dc_fields(SwarmWeights):
        if f.type is int and f.name in base:
            base[f.name] = int(base[f.name])
    mod.SWARM_WEIGHTS = SwarmWeights(**base)
    return mod

def _run_kaggle_match(seed, mod_a, mod_b):
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.time()
    env.run([mod_a.agent, mod_b.agent])
    match_time = time.time() - t0
    final = env.steps[-1]
    r0 = final[0].get('reward', 0) or 0
    r1 = final[1].get('reward', 0) or 0
    obs = final[0].get('observation') or {}
    planets = obs.get('planets') or []
    ships0 = sum((p[5] or 0) for p in planets if p[1] == 0)
    ships1 = sum((p[5] or 0) for p in planets if p[1] == 1)
    return {
        'reward_our': r0,
        'reward_sub': r1,
        'win': int(r0 > r1),
        'draw': int(r0 == r1),
        'steps': len(env.steps),
        'ships_our_final': ships0,
        'ships_sub_final': ships1,
        'ships_diff': ships0 - ships1,
        'match_time_sec': round(match_time, 1),
    }

def run_match(task):
    """task = (seed, idx_a, weights_a, name_a, idx_b, weights_b, name_b, match_id)"""
    seed, i_a, w_a, name_a, i_b, w_b, name_b, match_id = task
    _silence_debug_logs()
    suffix = f"{seed}_{i_a}_{i_b}_{match_id}"
    mod_a = _load_agent_module(w_a, suffix + "_a")
    mod_b = _load_agent_module(w_b, suffix + "_b")
    
    if RUN_BACKEND == 'kaggle':
        res = _run_kaggle_match(seed, mod_a, mod_b)
    else:
        # local backend — нужно определить local_match или использовать свою реализацию
        # (оставляем заглушку)
        raise NotImplementedError("local backend requires local_match module")
    
    points_a = 1.0 if res['win'] else (0.5 if res['draw'] else 0.0)
    points_b = 1.0 - points_a
    return {
        'candidate_a_idx': i_a,
        'candidate_a_name': name_a,
        'candidate_b_idx': i_b,
        'candidate_b_name': name_b,
        'match_id': match_id,
        'seed': seed,
        'points_a': points_a,
        'points_b': points_b,
        'win_a': res['win'],
        'draw': res['draw'],
        'win_b': 1 - res['win'] - res['draw'],
        'reward_our': res['reward_our'],
        'reward_sub': res['reward_sub'],
        'ships_our_final': res['ships_our_final'],
        'ships_sub_final': res['ships_sub_final'],
        'ships_diff': res['ships_diff'],
        'steps': res['steps'],
        'match_time_sec': res['match_time_sec'],
    }

def load_candidates_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    candidates = []
    for _, row in df.iterrows():
        name = row['name']
        weights = {}
        for col in df.columns:
            if col.startswith('w_'):
                val = row[col]
                if pd.notna(val):
                    weights[col[2:]] = val
        candidates.append((name, weights))
    return candidates

def main():
    candidates = load_candidates_from_csv(CSV_CANDIDATES)
    print(f"Загружено кандидатов: {len(candidates)}")
    for name, w in candidates:
        print(f"  {name}: {w}")
    
    n = len(candidates)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    tasks = []
    for i, j in pairs:
        name_i, w_i = candidates[i]
        name_j, w_j = candidates[j]
        for match_id in range(N_MATCHES_PER_PAIR):
            seed = SEED_OFFSET + i * 1000 + j * 100 + match_id
            tasks.append((seed, i, w_i, name_i, j, w_j, name_j, match_id))
    total_matches = len(tasks)
    print(f"Всего матчей: {total_matches} (пар {len(pairs)} × {N_MATCHES_PER_PAIR})")
    
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_match, t): t for t in tasks}
        for idx, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(f"[{idx}/{total_matches}] {r['candidate_a_name']} vs {r['candidate_b_name']} "
                  f"match {r['match_id']} -> {r['points_a']} : {r['points_b']}")
    print(f"Все матчи за {time.time()-t0:.1f} сек.")
    
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"Сохранено: {OUT_CSV}")
    
    points = {i: 0.0 for i in range(n)}
    for r in results:
        i, j = r['candidate_a_idx'], r['candidate_b_idx']
        points[i] += r['points_a']
        points[j] += r['points_b']
    
    total_matches_per_candidate = (n - 1) * N_MATCHES_PER_PAIR
    ranking = sorted([(i, points[i]) for i in range(n)], key=lambda x: -x[1])
    print("\n" + "="*60)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("="*60)
    print(f"{'Кандидат':<20} {'Очки':<8} {'Матчей':<8} {'Среднее':<8}")
    for i, pts in ranking:
        name, _ = candidates[i]
        avg = pts / total_matches_per_candidate
        print(f"{name:<20} {pts:<8.1f} {total_matches_per_candidate:<8} {avg:<8.3f}")
    
    with open(SUMMARY_TXT, 'w') as f:
        f.write(f"Турнир {TS}\n")
        f.write(f"Кандидатов: {n}, матчей на пару: {N_MATCHES_PER_PAIR}\n\n")
        f.write("Рейтинг:\n")
        for i, pts in ranking:
            name, _ = candidates[i]
            avg = pts / total_matches_per_candidate
            f.write(f"{name}: {pts} очков, среднее {avg:.3f}\n")
        f.write(f"\nДетали: {OUT_CSV}\n")
    print(f"Сводка: {SUMMARY_TXT}")

if __name__ == "__main__":
    main()