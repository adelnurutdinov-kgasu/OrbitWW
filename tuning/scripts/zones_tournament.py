#!/usr/bin/env python3
"""
zones_tournament.py — успешное половинение (Successive Halving) для zones-весов.

Альтернатива zones_grid_search.py: вместо «полного грида на N сидов»
делаем РАУНДЫ с эскалирующим бюджетом и отсевом худших combo на каждом
раунде. Топ-кандидаты получают всё больше сидов → узкий CI на финале.
Слабые отсеиваются рано → не тратим compute впустую.

КАК ПОЛЬЗОВАТЬСЯ:
  1. Поправь блок ── КОНФИГ ── ниже.
  2. Запусти:    python3 zones_tournament.py
  3. Получишь:
       • zones_tour_<ts>.csv  — все матчи со всех раундов (есть колонка `round`)
       • stdout                — таблицы по раундам + финальный пьедестал

УСТРОЙСТВО:
  • run_one — копия из zones_grid_search.py (один матч в воркере, инъекция
    весов через мутацию zones.W_TARGETS).
  • Раунды: combo × ROUND_BUDGETS[round] сидов параллельно; в конце каждого
    раунда отсекаем нижние по ELIM_METRIC. До MIN_SURVIVORS либо до
    последнего раунда (тогда оставшиеся = финал).
  • В финал даём FINAL_BUDGET сидов — обычно 30+ для устойчивого ранжирования.
  • Сиды в каждом раунде НОВЫЕ (round_id * 1000 + i), чтобы не дублировать
    выборки. Внутри раунда все combo на одних и тех же сидах → paired.

ПОЧЕМУ ELIM_METRIC = 'ships_diff' по умолчанию:
  В прошлом гриде Spearman ρ(winrate, ships_diff) = 0.898 — метрики
  согласованы. Но ships_diff — непрерывная, на малой выборке (2-5 сидов)
  значительно стабильнее чем бинарный winrate. Финальное ранжирование
  показываем по обеим метрикам.
"""

import os
import sys
import importlib.util
import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ══════════════════════════════════════════════════════════════════════════

# Какие веса W_TARGETS из zones.py перебирать.
PARAM_GRID = {
    'wnn_close_res':  [0.2, 0.4, 0.6, 0.8, 1.0],
    'mean_dist_all':  [-0.8, -0.6, -0.4, -0.2],
    'ships':          [-0.9, -0.7, -0.5, -0.3, -0.1],
    # 'late_aggression': [0.0, 0.3, 0.6, 0.9, 1.2],
}

# Бюджет сидов на combo по раундам. Длина = число «отсеивающих» раундов.
# После них идёт ОБЯЗАТЕЛЬНЫЙ финал на FINAL_BUDGET для оставшихся.
# Начинай с малого (2-3) — отсев очевидно слабых дёшев.
ROUND_BUDGETS = [2, 3, 5, 8]

# Доля выживающих после раунда (0.5 = половина). Меньше — агрессивнее
# режем, выше шанс убить «случайно» плохой combo. Больше — мягче, дольше.
SURVIVORS_PCT = 0.5

# Минимум combo'в, при достижении которого прекращаем отсев и идём в финал.
# Если у тебя сразу < этого числа — все идут в финал.
MIN_SURVIVORS = 4

# Бюджет сидов на КАЖДУЮ combo в финале. 30+ даёт CI ширины ~±0.18 на winrate.
FINAL_BUDGET = 30

# По чему отсекать: 'ships_diff' (avg ship-разница) или 'winrate'.
# ships_diff менее шумный на малых выборках, рекомендуется.
ELIM_METRIC = 'ships_diff'

N_WORKERS = max(1, os.cpu_count() // 2)

# ══════════════════════════════════════════════════════════════════════════

# Скрипт лежит в <project>/tuning/scripts/. См. комментарий в zones_grid_search.py.
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUR_PATH = os.path.join(PROJECT_ROOT, "agent_bundle", "agent.py")
SUB_PATH = os.path.join(PROJECT_ROOT, "submission.py")
TS       = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_CSV  = os.path.join(RESULTS_DIR, f"zones_tour_{TS}.csv")


# ──────────────────────────────────────────────────────────────────────────
# Воркер (идентичен zones_grid_search.run_one — копия чтобы не зависеть)
# ──────────────────────────────────────────────────────────────────────────

def _silence_debug_logs():
    for k in ("ORBIT_AGENT_LOG", "ORBIT_AGENT_LOG_PLANS_ALL",
              "ORBIT_AGENT_LOG_FLEETS",
              "SUB_AGENT_LOG", "SUB_AGENT_LOG_MISSIONS_ALL",
              "SUB_AGENT_LOG_FLEETS"):
        os.environ.pop(k, None)


def _load_module(path, name):
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(task):
    """Один матч: (seed, overrides_dict) → metrics dict."""
    seed, overrides = task
    _silence_debug_logs()

    bundle_dir = os.path.join(PROJECT_ROOT, "agent_bundle")
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    suffix = f"{seed}_{abs(hash(frozenset(overrides.items()))) % 10**8}"
    our_mod = _load_module(OUR_PATH, f"_our_tour_{suffix}")
    sub_mod = _load_module(SUB_PATH, f"_sub_tour_{suffix}")

    import zones as zones_mod
    for key, val in overrides.items():
        if key not in zones_mod.W_TARGETS:
            raise KeyError(f"'{key}' нет в zones.W_TARGETS")
        zones_mod.W_TARGETS[key] = val

    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.time()
    env.run([our_mod.agent, sub_mod.agent])
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
        **{f'w_{k}': v for k, v in overrides.items()},
        'reward_our': r0,
        'reward_sub': r1,
        'win': int(r0 > r1),
        'draw': int(r0 == r1),
        'steps': len(env.steps),
        'ships_our_final': ships_p0,
        'ships_sub_final': ships_p1,
        'ships_diff': ships_p0 - ships_p1,
        'match_time_sec': round(match_time, 1),
    }


# ──────────────────────────────────────────────────────────────────────────
# Раунд: запуск combos × seeds параллельно
# ──────────────────────────────────────────────────────────────────────────

def _combo_key(overrides):
    """Хешируемый ключ combo для группировки между раундами."""
    return tuple(sorted(overrides.items()))


def run_round(combos, seeds, round_id, all_results):
    """Прогоняет combos × seeds, возвращает DataFrame с результатами раунда.
    Дополнительно append'ит к all_results для финального CSV."""
    tasks = [(s, ov) for ov in combos for s in seeds]
    n_total = len(tasks)
    print(f"=== ROUND {round_id} ===  {len(combos)} combos × {len(seeds)} seeds = {n_total} матчей")
    print(f"    seeds: {seeds[0]}..{seeds[-1]}  ({'paired' if len(seeds)>1 else 'single'})")

    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            seed, ov = futs[fut]
            done += 1
            try:
                r = fut.result()
                r['round'] = round_id
                rows.append(r)
                all_results.append(r)
            except Exception as e:
                print(f"  [ERR] seed={seed} {ov}: {e}")
            if done % max(1, n_total // 10) == 0 or done == n_total:
                pct = 100 * done / n_total
                elapsed = time.time() - started
                eta = elapsed * (n_total - done) / max(1, done)
                print(f"  [progress] {done:>4}/{n_total} ({pct:.0f}%) "
                      f"elapsed {elapsed:.0f}s  eta {eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Сводка по combos из результатов раунда
# ──────────────────────────────────────────────────────────────────────────

def _aggregate(df, weight_cols):
    agg = df.groupby(weight_cols).agg(
        matches=('win', 'count'),
        wins=('win', 'sum'),
        winrate=('win', 'mean'),
        avg_ships_diff=('ships_diff', 'mean'),
        median_ships_diff=('ships_diff', 'median'),
    ).round(2)
    return agg


def _select_survivors(agg, n_target):
    """Возвращает топ-n_target combos по ELIM_METRIC."""
    sort_col = 'avg_ships_diff' if ELIM_METRIC == 'ships_diff' else 'winrate'
    sorted_agg = agg.sort_values(sort_col, ascending=False)
    return sorted_agg.head(n_target)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(OUR_PATH):
        sys.exit(f"Не найден агент: {OUR_PATH}")
    if not os.path.isfile(SUB_PATH):
        sys.exit(f"Не найден baseline: {SUB_PATH}")

    keys = list(PARAM_GRID.keys())
    value_lists = [PARAM_GRID[k] for k in keys]
    weight_cols = [f'w_{k}' for k in keys]
    initial_combos = [dict(zip(keys, vs)) for vs in itertools.product(*value_lists)]

    # Прикинуть бюджет
    cur_n = len(initial_combos)
    sched = []
    for budget in ROUND_BUDGETS:
        sched.append((cur_n, budget, cur_n * budget))
        survivors_n = max(MIN_SURVIVORS, math.ceil(cur_n * SURVIVORS_PCT))
        if survivors_n >= cur_n:
            break
        cur_n = survivors_n
    final_n = cur_n
    sched.append((final_n, FINAL_BUDGET, final_n * FINAL_BUDGET))
    total = sum(t for _, _, t in sched)

    print(f"=== zones tournament ===")
    print(f"  параметры:    {list(PARAM_GRID.keys())}")
    print(f"  стартовых combos: {len(initial_combos)}")
    print(f"  раундов отсева:   {len(ROUND_BUDGETS)}")
    print(f"  выживания:        {SURVIVORS_PCT*100:.0f}% per round, не менее {MIN_SURVIVORS}")
    print(f"  финальный бюджет: {FINAL_BUDGET} сидов на combo")
    print(f"  метрика отсева:   {ELIM_METRIC}")
    print(f"  воркеров:         {N_WORKERS}")
    print(f"  CSV:              {OUT_CSV}")
    print()
    print(f"  бюджет:")
    for i, (n, b, t) in enumerate(sched):
        label = "FINAL" if i == len(sched) - 1 else f"R{i}"
        print(f"    {label:>6}: {n:>3} combos × {b:>3} seeds = {t:>5} матчей")
    print(f"    {'ИТОГО':>6}: {total} матчей")
    print()

    all_results = []
    cur_combos = list(initial_combos)
    started = time.time()

    # ── Раунды отсева ────────────────────────────────────────────────────
    for round_id, budget in enumerate(ROUND_BUDGETS):
        seeds = [round_id * 1000 + i for i in range(budget)]
        df = run_round(cur_combos, seeds, round_id, all_results)
        agg = _aggregate(df, weight_cols)

        survivors_n = max(MIN_SURVIVORS, math.ceil(len(cur_combos) * SURVIVORS_PCT))
        if survivors_n >= len(cur_combos):
            print(f"\n  ⊘ выживание {survivors_n} ≥ текущих {len(cur_combos)}, "
                  f"переходим в финал без отсева")
            break

        survivors = _select_survivors(agg, survivors_n)
        eliminated = agg.drop(survivors.index)

        # Печать сводки
        print(f"\n  ── ИТОГИ ROUND {round_id} ──")
        print(f"  ВЫЖИЛИ ({len(survivors)}):")
        print(survivors.to_string())
        print(f"\n  ВЫБИЛИ ({len(eliminated)}, top-5 чтобы не засорять):")
        print(eliminated.sort_values(
            'avg_ships_diff' if ELIM_METRIC == 'ships_diff' else 'winrate',
            ascending=False
        ).head(5).to_string())
        print()

        # Восстановим список dict'ов для выживших
        cur_combos = []
        for idx in survivors.index:
            ov = dict(zip(keys, idx if isinstance(idx, tuple) else (idx,)))
            cur_combos.append(ov)

        if len(cur_combos) <= MIN_SURVIVORS:
            print(f"  достигли MIN_SURVIVORS={MIN_SURVIVORS}, переходим в финал")
            break

    # ── Финал ────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"ФИНАЛ: {len(cur_combos)} combos × {FINAL_BUDGET} seeds = "
          f"{len(cur_combos) * FINAL_BUDGET} матчей")
    print(f"{'═'*70}")

    final_round_id = len(ROUND_BUDGETS)
    final_seeds = [final_round_id * 1000 + i for i in range(FINAL_BUDGET)]
    final_df = run_round(cur_combos, final_seeds, final_round_id, all_results)

    final_agg = _aggregate(final_df, weight_cols)

    # Wilson 95% CI на winrate
    def wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        denom = 1 + z*z/n
        c = (p + z*z/(2*n)) / denom
        h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        return c - h, c + h

    final_agg['ci_lo'] = final_agg.apply(
        lambda r: round(wilson(int(r['wins']), int(r['matches']))[0], 3),
        axis=1)
    final_agg['ci_hi'] = final_agg.apply(
        lambda r: round(wilson(int(r['wins']), int(r['matches']))[1], 3),
        axis=1)

    final_agg = final_agg.sort_values(
        ['winrate', 'avg_ships_diff'], ascending=[False, False]
    )

    print(f"\n{'═'*70}")
    print(f"ФИНАЛЬНОЕ РАНЖИРОВАНИЕ (по winrate, потом avg_ships_diff)")
    print(f"{'═'*70}")
    print(final_agg.to_string())

    print(f"\n{'═'*70}")
    print(f"ПОБЕДИТЕЛЬ:")
    print(f"{'═'*70}")
    winner_idx = final_agg.index[0]
    winner_dict = dict(zip(weight_cols, winner_idx if isinstance(winner_idx, tuple) else (winner_idx,)))
    print(f"  веса:           {winner_dict}")
    w_row = final_agg.iloc[0]
    print(f"  winrate:        {w_row['winrate']:.3f}  CI95 [{w_row['ci_lo']}, {w_row['ci_hi']}]")
    print(f"  avg_ships_diff: {w_row['avg_ships_diff']:+.1f}")
    print(f"  median_ships:   {w_row['median_ships_diff']:+.1f}")

    elapsed = time.time() - started
    print(f"\nВремя: {elapsed:.0f}с ({elapsed/max(1,len(all_results)):.1f}с/матч)")
    print(f"Матчей всего: {len(all_results)}")
    print(f"CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
