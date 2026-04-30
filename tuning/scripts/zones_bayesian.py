#!/usr/bin/env python3
"""
zones_bayesian.py — Bayesian-оптимизация ВСЕХ весов zones.py через optuna (TPE).

ОТЛИЧИЕ ОТ TOURNAMENT:
  Tournament — full grid с отсевом, нужно вручную выбирать сетку значений.
  Bayesian — TPE-сэмплер сам направленно ищет в непрерывном пространстве,
  учится на предыдущих trial'ах. Эффективен в high-dimensional (14D) поиске,
  где full grid взрывается.

ЧТО ИЩЕТ:
  Все 14 весов сразу:
    W_OURS:    area_inv, wnn_close_res, mean_dist_all, prod, ships, n_cross, late_aggression
    W_TARGETS: те же 7 фич
  Диапазон каждого: [-1.5, +1.5] — широкий, включая отрицательные, ноль и
  положительные значения. TPE сам решит где копать.

УСТРОЙСТВО:
  • Sequential trials (TPE учится на каждом результате).
  • Внутри trial — параллельно прогоняем N_SEEDS_PER_TRIAL сидов.
  • Целевая функция: avg_ships_diff (стабильнее winrate на малых выборках).
  • Хранение: SQLite-storage → можно прерывать и возобновлять (`--resume`).

ЗАПУСК:
  python3 zones_bayesian.py            # новый study
  python3 zones_bayesian.py --resume   # продолжить предыдущий

ЗАВИСИМОСТИ:
  pip install optuna
  (Если не установлен — скрипт скажет команду установки и прервётся.)
"""

import os
import sys
import importlib.util
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    sys.exit("optuna не установлен. Запусти: pip install optuna")

# ══════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ══════════════════════════════════════════════════════════════════════════

# Сколько trial'ов TPE прогонит. Каждый trial = N_SEEDS_PER_TRIAL матчей.
# Рекомендация: 200-300 для 14D пространства. Меньше — мало для TPE,
# больше — diminishing returns.
N_TRIALS = 250

# Сколько сидов на каждый trial. Меньше — быстрее, но шумнее objective.
# 5 — компромисс. Для финального ранжирования топ-3 потом догоним до 30.
N_SEEDS_PER_TRIAL = 5

# Диапазон поиска для каждого веса. [-1.5, +1.5] широко покрывает текущие
# значения (-0.6..+0.9 в zones.py) и оставляет место для «контр-знаковых»
# гипотез. Если хочешь уже — поставь [-1.0, +1.0].
WEIGHT_RANGE = (-1.5, 1.5)

# Сколько TPE warmup-trial'ов делать со случайным сэмплингом (до того как
# включится TPE). Меньше 20 — TPE стартует в темноте.
N_STARTUP_TRIALS = 25

# Какие фичи zones перебирать. Если хочешь зафиксировать какие-то — убери
# их из списка, и они останутся с дефолтными значениями.
FEATURES = ['area_inv', 'wnn_close_res', 'mean_dist_all',
            'prod', 'ships', 'n_cross', 'late_aggression']

# Параллельность внутри одного trial.
N_WORKERS = max(1, os.cpu_count() // 2)

# Финальная валидация топ-N trial'ов на N_FINAL_SEEDS сидах для tight CI.
N_FINAL_TOP    = 5
N_FINAL_SEEDS  = 30

# ══════════════════════════════════════════════════════════════════════════

# Скрипт лежит в <project>/tuning/scripts/. См. комментарий в zones_grid_search.py.
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUR_PATH = os.path.join(PROJECT_ROOT, "agent_bundle", "agent.py")
SUB_PATH = os.path.join(PROJECT_ROOT, "submission.py")
TS       = datetime.now().strftime('%Y%m%d_%H%M%S')
STUDY_DB = os.path.join(RESULTS_DIR, "zones_bayesian_study.db")
OUT_CSV  = os.path.join(RESULTS_DIR, f"zones_bayesian_{TS}.csv")
STUDY_NAME = "zones_w_full"


# ──────────────────────────────────────────────────────────────────────────
# Worker
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
    """Один матч. Принимает (seed, w_ours_dict, w_targets_dict).
    Возвращает (seed, ships_diff, win, steps)."""
    seed, w_ours, w_targets = task
    _silence_debug_logs()

    bundle_dir = os.path.join(PROJECT_ROOT, "agent_bundle")
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    suffix = f"{seed}_{abs(hash(frozenset(w_ours.items()) | frozenset(w_targets.items()))) % 10**8}"
    our_mod = _load_module(OUR_PATH, f"_our_bay_{suffix}")
    sub_mod = _load_module(SUB_PATH, f"_sub_bay_{suffix}")

    # Инъекция в ОБА словаря одновременно.
    import zones as zones_mod
    for k, v in w_ours.items():
        zones_mod.W_OURS[k] = v
    for k, v in w_targets.items():
        zones_mod.W_TARGETS[k] = v

    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    env.run([our_mod.agent, sub_mod.agent])

    final = env.steps[-1]
    r0 = final[0].get('reward', 0) or 0
    r1 = final[1].get('reward', 0) or 0
    obs = final[0].get('observation') or {}
    planets = obs.get('planets') or []
    ships_p0 = sum((p[5] or 0) for p in planets if p[1] == 0)
    ships_p1 = sum((p[5] or 0) for p in planets if p[1] == 1)

    return {
        'seed': seed,
        'ships_diff': ships_p0 - ships_p1,
        'win': int(r0 > r1),
        'steps': len(env.steps),
    }


# ──────────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────────

def _make_objective(pool, seed_offset_fn, all_results):
    """Closure-фабрика для optuna objective. pool переиспользуется между trial'ами."""

    def objective(trial):
        # Suggest веса
        w_ours = {f: trial.suggest_float(f'ours_{f}', *WEIGHT_RANGE) for f in FEATURES}
        w_targets = {f: trial.suggest_float(f'tgt_{f}', *WEIGHT_RANGE) for f in FEATURES}

        # Сиды для этого trial — уникальные, чтобы не пересекались с предыдущими
        seed_offset = seed_offset_fn(trial.number)
        seeds = list(range(seed_offset, seed_offset + N_SEEDS_PER_TRIAL))
        tasks = [(s, w_ours, w_targets) for s in seeds]

        # Прогон параллельно
        t0 = time.time()
        results = list(pool.map(run_one, tasks))
        dt = time.time() - t0

        # Объективная — avg ships_diff
        avg_ships_diff = sum(r['ships_diff'] for r in results) / len(results)
        winrate = sum(r['win'] for r in results) / len(results)

        # Логируем для CSV
        for r in results:
            row = {
                'trial': trial.number,
                **{f'w_ours_{k}': v for k, v in w_ours.items()},
                **{f'w_tgt_{k}': v for k, v in w_targets.items()},
                **r,
            }
            all_results.append(row)

        # Прогресс
        print(f"[trial {trial.number:>4}] avg_ships_diff={avg_ships_diff:>+8.1f}  "
              f"winrate={winrate:.2f}  ({dt:.0f}s)", flush=True)

        # Промежуточное сохранение каждые 10 trial'ов
        if trial.number % 10 == 0 and all_results:
            pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)

        return avg_ships_diff

    return objective


# ──────────────────────────────────────────────────────────────────────────
# Финальная валидация топ-N trial'ов
# ──────────────────────────────────────────────────────────────────────────

def validate_top(study, pool, all_results):
    """Прогоняет топ-N_FINAL_TOP trial'ов на N_FINAL_SEEDS НОВЫХ сидах.
    Это снимает дисперсию N_SEEDS_PER_TRIAL=5 и даёт устойчивое ранжирование."""
    print(f"\n{'═'*70}")
    print(f"ФИНАЛЬНАЯ ВАЛИДАЦИЯ — топ-{N_FINAL_TOP} trial'ов × {N_FINAL_SEEDS} сидов")
    print(f"{'═'*70}")

    # Сортируем trials по value (avg_ships_diff)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: -t.value)[:N_FINAL_TOP]

    # Финальные сиды — далеко от использованных в trial'ах
    final_seeds = list(range(900_000, 900_000 + N_FINAL_SEEDS))

    final_rows = []
    for rank, t in enumerate(top, 1):
        w_ours = {k.replace('ours_', ''): v for k, v in t.params.items() if k.startswith('ours_')}
        w_targets = {k.replace('tgt_', ''): v for k, v in t.params.items() if k.startswith('tgt_')}

        tasks = [(s, w_ours, w_targets) for s in final_seeds]
        print(f"\n  [#{rank}] trial={t.number}  prelim={t.value:+.1f}  ...")
        t0 = time.time()
        results = list(pool.map(run_one, tasks))
        dt = time.time() - t0
        avg = sum(r['ships_diff'] for r in results) / len(results)
        wr = sum(r['win'] for r in results) / len(results)
        print(f"     final: avg_ships_diff={avg:+.1f}  winrate={wr:.3f}  ({dt:.0f}s)")

        final_rows.append({
            'rank': rank,
            'trial': t.number,
            'prelim_value': t.value,
            'final_avg_ships_diff': avg,
            'final_winrate': wr,
            **t.params,
        })

        # Также сохраним все матчи финальной валидации в основном CSV
        for r in results:
            all_results.append({
                'trial': t.number,
                'phase': 'final',
                **{f'w_ours_{k}': v for k, v in w_ours.items()},
                **{f'w_tgt_{k}': v for k, v in w_targets.items()},
                **r,
            })

    final_df = pd.DataFrame(final_rows).sort_values('final_winrate', ascending=False)
    return final_df


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true', help='Продолжить существующий study')
    ap.add_argument('--n-trials', type=int, default=N_TRIALS,
                    help=f'Сколько trial’ов запустить (default {N_TRIALS})')
    args = ap.parse_args()

    if not os.path.isfile(OUR_PATH):
        sys.exit(f"Не найден агент: {OUR_PATH}")
    if not os.path.isfile(SUB_PATH):
        sys.exit(f"Не найден baseline: {SUB_PATH}")

    storage_url = f"sqlite:///{STUDY_DB}"

    # Создаём или загружаем study
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=42)
    if args.resume and os.path.isfile(STUDY_DB):
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage_url, sampler=sampler)
        existing = len(study.trials)
        print(f"=== RESUME: study уже содержит {existing} trial'ов ===")
    else:
        if os.path.isfile(STUDY_DB) and not args.resume:
            print(f"  [warning] {STUDY_DB} существует, но --resume не передан → создаём заново")
            os.remove(STUDY_DB)
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage_url,
            sampler=sampler,
            direction='maximize',
        )

    print(f"=== zones Bayesian search ===")
    print(f"  features:           {FEATURES}")
    print(f"  всего параметров:   {len(FEATURES) * 2} (W_OURS + W_TARGETS)")
    print(f"  диапазон:           {WEIGHT_RANGE}")
    print(f"  trials:             {args.n_trials}  (warmup={N_STARTUP_TRIALS})")
    print(f"  seeds per trial:    {N_SEEDS_PER_TRIAL}")
    print(f"  всего матчей:       ~{args.n_trials * N_SEEDS_PER_TRIAL} (без финала)")
    print(f"  + финал:            {N_FINAL_TOP} × {N_FINAL_SEEDS} = {N_FINAL_TOP*N_FINAL_SEEDS} матчей")
    print(f"  воркеров:           {N_WORKERS}")
    print(f"  CSV:                {OUT_CSV}")
    print(f"  study DB:           {STUDY_DB}")
    print()

    # Сиды каждого trial далеко друг от друга, чтобы не пересекаться
    def seed_offset_fn(trial_number):
        return trial_number * 100

    all_results = []
    started = time.time()

    pool = ProcessPoolExecutor(max_workers=N_WORKERS)
    try:
        objective = _make_objective(pool, seed_offset_fn, all_results)
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        # Финальная валидация топ-N
        final_df = validate_top(study, pool, all_results)
    finally:
        pool.shutdown()

    # Финальный CSV
    pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)

    elapsed = time.time() - started
    print(f"\nВремя: {elapsed:.0f}с")

    # ── Финал: ранжирование топ-N ───────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"ИТОГОВОЕ РАНЖИРОВАНИЕ ТОП-{N_FINAL_TOP}")
    print(f"{'═'*70}")
    print(final_df[['rank','trial','prelim_value','final_avg_ships_diff','final_winrate']].to_string(index=False))

    # ── Победитель: вывод весов в готовом для вставки виде ──────────────
    print(f"\n{'═'*70}")
    print(f"ПОБЕДИТЕЛЬ — веса для вставки в zones.py")
    print(f"{'═'*70}")
    winner = final_df.iloc[0]
    w_ours = {k.replace('ours_', ''): v for k, v in winner.items() if k.startswith('ours_')}
    w_tgt  = {k.replace('tgt_', ''):  v for k, v in winner.items() if k.startswith('tgt_')}

    print(f"  trial:              {int(winner['trial'])}")
    print(f"  final winrate:      {winner['final_winrate']:.3f}")
    print(f"  final ships_diff:   {winner['final_avg_ships_diff']:+.1f}")
    print()
    print("W_OURS = {")
    for k in FEATURES:
        print(f"    '{k:<18}': {w_ours[k]:+.3f},")
    print("}")
    print()
    print("W_TARGETS = {")
    for k in FEATURES:
        print(f"    '{k:<18}': {w_tgt[k]:+.3f},")
    print("}")

    # ── Краткие insights ────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("ВАЖНОСТЬ ПАРАМЕТРОВ (по optuna importance)")
    print(f"{'═'*70}")
    try:
        imp = optuna.importance.get_param_importances(study)
        for name, val in list(imp.items())[:15]:
            bar = '█' * int(val * 50)
            print(f"  {name:<28}  {val:.3f}  {bar}")
    except Exception as e:
        print(f"  importance не удалось вычислить: {e}")

    print(f"\nВсе trial'ы и матчи: {OUT_CSV}")
    print(f"Study DB (можно --resume):   {STUDY_DB}")


if __name__ == "__main__":
    main()
