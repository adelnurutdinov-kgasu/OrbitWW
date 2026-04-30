#!/usr/bin/env python3
"""
zones_grid_search.py — параллельный перебор весов из agent_bundle/zones.py.

Запускает много матчей "наш агент (agent_bundle/agent.py) vs strong (submission.py)",
варьируя указанные веса W_TARGETS, и пишет результаты в CSV.

КАК ПОЛЬЗОВАТЬСЯ:
  1. Поправь блок ── КОНФИГ ── ниже под нужный grid.
  2. Запусти:    python3 zones_grid_search.py
  3. Получишь:   zones_grid_<timestamp>.csv  + сводку по winrate в stdout.

УСТРОЙСТВО:
  • Каждый матч идёт в своём процессе (ProcessPoolExecutor).
  • Воркер сам загружает агентов через importlib (sys.modules изолированы).
  • Веса инжектируются мутацией zones.W_TARGETS in-place — defaults
    функций ссылаются на тот же dict, изменения подхватываются.
  • Логи agent_debug отключены принудительно (они нужны только в одиночных
    запусках; в grid они только тормозят запись на диск).

ПОЧЕМУ НЕ ТРОГАЕМ TEST.IPYNB:
  Скрипт автономен, можно прервать ctrl+C, можно запускать в фоне через
  nohup. Существующая логика test.ipynb (одиночный матч + N_SEEDS=10
  серия) живёт своей жизнью и продолжает работать как раньше.
"""

import os
import sys
import importlib.util
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# КОНФИГ — РЕДАКТИРУЙ ЭТОТ БЛОК
# ══════════════════════════════════════════════════════════════════════════

# Какие веса W_TARGETS из zones.py перебирать.
# Каждый ключ — фича из ZONE_FEATURES, список — значения grid'а.
# Веса, не указанные здесь, остаются дефолтными.
PARAM_GRID = {
    #'late_aggression': [0.0, 0.3, 0.6, 0.9, 1.2],
    # Раскомментируй чтобы тюнить пачкой (осторожно, число матчей
    # умножается комбинаторно):
    'wnn_close_res':  [0.2, 0.4, 0.6, 0.8, 1.0],
    'mean_dist_all':  [-0.8, -0.6, -0.4, -0.2],
    'ships':          [-0.9, -0.7, -0.5, -0.3, -0,1],
}

# Сколько разных seed'ов для каждой комбинации (статистическая мощность).
# 10-20 разумно: меньше — шум, больше — долго. Один и тот же набор seed'ов
# используется для всех комбинаций → paired comparison.
N_SEEDS = 10
SEED_OFFSET = 0   # начало диапазона seed'ов (0..N_SEEDS-1 по умолчанию)

# Параллельность. На маке: половина ядер — компромисс между скоростью и
# тепловым throttling. Если памяти мало (< 8 GB), уменьши до 2-3.
N_WORKERS = max(1, os.cpu_count() // 2)

# ══════════════════════════════════════════════════════════════════════════
# КОНЕЦ КОНФИГА
# ══════════════════════════════════════════════════════════════════════════

# Скрипт лежит в <project>/tuning/scripts/. Поднимаемся на 2 уровня к корню,
# где находится agent_bundle/ и submission.py. Результаты — в tuning/results/.
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUR_PATH = os.path.join(PROJECT_ROOT, "agent_bundle", "agent.py")
SUB_PATH = os.path.join(PROJECT_ROOT, "submission.py")
OUT_CSV  = os.path.join(
    RESULTS_DIR, f"zones_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)


def _silence_debug_logs():
    """Глушим все debug env vars — в grid они только мешают."""
    for k in ("ORBIT_AGENT_LOG", "ORBIT_AGENT_LOG_PLANS_ALL",
              "ORBIT_AGENT_LOG_FLEETS",
              "SUB_AGENT_LOG", "SUB_AGENT_LOG_MISSIONS_ALL",
              "SUB_AGENT_LOG_FLEETS"):
        os.environ.pop(k, None)


def _load_module(path, name):
    """Свежая загрузка модуля. В воркере sys.modules своё, конфликтов нет."""
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(task):
    """Воркер: один матч. Принимает (seed, overrides_dict).
    Возвращает компактный dict с метриками."""
    seed, overrides = task
    _silence_debug_logs()

    # agent_bundle на путях чтобы внутренние импорты zones/projection/etc. работали
    bundle_dir = os.path.join(PROJECT_ROOT, "agent_bundle")
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    # Каждый воркер грузит модули заново под уникальным именем
    suffix = f"{seed}_{abs(hash(frozenset(overrides.items()))) % 10**8}"
    our_mod = _load_module(OUR_PATH, f"_our_grid_{suffix}")
    sub_mod = _load_module(SUB_PATH, f"_sub_grid_{suffix}")

    # ── ИНЪЕКЦИЯ ВЕСОВ ──────────────────────────────────────────────────
    # zones импортнулся внутри agent.py при загрузке. Берём его dict
    # W_TARGETS и мутируем in-place: функции (compute_zones и т.п.)
    # держат ссылку на тот же объект через default-аргументы → изменения
    # подхватятся при первом же вызове.
    import zones as zones_mod
    for key, val in overrides.items():
        if key not in zones_mod.W_TARGETS:
            raise KeyError(
                f"Override '{key}' нет в zones.W_TARGETS "
                f"(доступные: {sorted(zones_mod.W_TARGETS)})"
            )
        zones_mod.W_TARGETS[key] = val

    # ── ЗАПУСК МАТЧА ────────────────────────────────────────────────────
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.time()
    env.run([our_mod.agent, sub_mod.agent])
    match_time = time.time() - t0

    final = env.steps[-1]
    r0 = final[0].get('reward', 0) or 0
    r1 = final[1].get('reward', 0) or 0

    # Финальный snapshot для дополнительных метрик
    obs = final[0].get('observation') or {}
    planets = obs.get('planets') or []
    ships_p0 = sum((p[5] or 0) for p in planets if p[1] == 0)
    ships_p1 = sum((p[5] or 0) for p in planets if p[1] == 1)
    n_p0 = sum(1 for p in planets if p[1] == 0)
    n_p1 = sum(1 for p in planets if p[1] == 1)

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
        'n_our_final': n_p0,
        'n_sub_final': n_p1,
        'match_time_sec': round(match_time, 1),
    }


def main():
    if not os.path.isfile(OUR_PATH):
        sys.exit(f"Не найден агент: {OUR_PATH}")
    if not os.path.isfile(SUB_PATH):
        sys.exit(f"Не найден baseline: {SUB_PATH}")

    # Раскрываем grid → список dict'ов overrides
    keys = list(PARAM_GRID.keys())
    value_lists = [PARAM_GRID[k] for k in keys]
    combos = [dict(zip(keys, vs)) for vs in itertools.product(*value_lists)]

    seeds = list(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))
    tasks = [(s, ov) for ov in combos for s in seeds]

    print(f"=== zones grid search ===")
    print(f"  параметры:    {PARAM_GRID}")
    print(f"  комбинаций:   {len(combos)}")
    print(f"  seed'ов:      {N_SEEDS} (диапазон {seeds[0]}..{seeds[-1]})")
    print(f"  всего матчей: {len(tasks)}")
    print(f"  воркеров:     {N_WORKERS}")
    print(f"  CSV:          {OUT_CSV}")
    print()

    results = []
    started = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            seed, overrides = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                ovr_str = ' '.join(f'{k}={v}' for k, v in overrides.items())
                outcome = 'WIN ' if r['win'] else ('DRAW' if r['draw'] else 'LOSS')
                print(f"[{i:>4}/{len(tasks)}] seed={r['seed']:>2} {ovr_str:<30} "
                      f"{outcome}  steps={r['steps']:>3}  "
                      f"final={r['ships_our_final']:>4}/{r['ships_sub_final']:<4}  "
                      f"({r['match_time_sec']}s)", flush=True)
            except Exception as e:
                print(f"[{i:>4}/{len(tasks)}] seed={seed} {overrides} -> FAILED: {e}",
                      flush=True)

            # Промежуточно сохраняем каждые 10 матчей — на случай ctrl+C
            if i % 10 == 0:
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    elapsed = time.time() - started
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    print()
    print(f"Готово за {elapsed:.0f}с  (avg {elapsed/max(1,len(tasks)):.1f}с/матч)")
    print(f"CSV: {OUT_CSV}")
    print()

    # Сводка по комбинациям параметров
    weight_cols = [c for c in df.columns if c.startswith('w_')]
    if weight_cols and not df.empty:
        df['ships_diff'] = df['ships_our_final'] - df['ships_sub_final']
        agg = df.groupby(weight_cols).agg(
            matches=('win', 'count'),
            wins=('win', 'sum'),
            draws=('draw', 'sum'),
            winrate=('win', 'mean'),
            avg_steps=('steps', 'mean'),
            avg_ships_diff=('ships_diff', 'mean'),
        ).round(3).sort_values('winrate', ascending=False)
        print("Сводка по комбинациям (отсортирована по winrate):")
        print(agg.to_string())
        print()
        best = agg.index[0]
        print(f"ЛУЧШАЯ комбинация: {dict(zip(weight_cols, best if isinstance(best, tuple) else (best,)))}")
        print(f"  winrate = {agg.iloc[0]['winrate']:.1%}  "
              f"avg_ships_diff = {agg.iloc[0]['avg_ships_diff']:+.1f}")


if __name__ == "__main__":
    main()
