#!/usr/bin/env python3
"""
reverse_tournament.py — «реверс-турнир» по SwarmWeights.

ИДЕЯ
====
1. Прогоняем baseline-конфиг на всех сидах → собираем список ПРОИГРЫШЕЙ.
2. Для каждого проигранного сида ищем такие SwarmWeights, при которых
   тот же сид становится победой. Бюджет ограничен: random search +
   опциональный CEM-довод вокруг лидера.
3. Сохраняем КАЖДУЮ попытку (даже неудачную) — это датасет для анализа
   «какие веса в каких сидах работают».

АНТИ-ПЕРЕОБУЧЕНИЕ
================
Сиды делятся на TRAIN / HOLDOUT (по умолчанию 70/30). Поиск идёт только
на TRAIN. В конце берём «consensus weights» (медиана/режим по решённым
TRAIN-сидам) и проверяем винрейт на HOLDOUT — если он не упал, значит
найденный режим общеприменим, а не подгонялся под конкретные карты.

УМНЫЙ ПОИСК
===========
Фаза 1: чистый random (uniform) внутри SEARCH_SPACE.
Фаза 2 (опц.): CEM-довод. Если сид не решён за PHASE1_BUDGET попыток,
строим Гауссиан вокруг 5 лучших по `ships_diff` и сэмплим ещё PHASE2_BUDGET.
«Лучшие» = ближе к победе (даже если все loss, выбираем с наибольшим diff).

DEAD-END сиды
=============
Если за PHASE1+PHASE2 не нашлось ни одной победы — сид помечается как
dead-end. Это сигнал «здесь либо баг, либо структурно слабая позиция».
Печатается список — отдельный таск на разбор.

ВЫХОД
=====
  reverse_tour_<ts>.csv:
    seed, phase, trial, weights..., win, ships_diff, steps, time_sec
  reverse_tour_summary_<ts>.txt:
    train winrate (baseline → after-search), holdout winrate (consensus),
    список dead-ends, медианные/модальные веса по решённым.

ЗАПУСК
======
  python3 tuning/scripts/reverse_tournament.py
  (поправь блок ── КОНФИГ ── ниже)
"""

import os
import sys
import importlib.util
import random
import math
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields as dc_fields
from datetime import datetime

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ══════════════════════════════════════════════════════════════════════════

N_SEEDS         = 70           # сколько сидов всего гонять в baseline
SEED_OFFSET     = 1            # стартовый seed (сиды = OFFSET..OFFSET+N-1)
TRAIN_FRAC      = 0.7          # доля сидов для поиска; остальные — holdout
SPLIT_SEED      = 42           # детерминированное разбиение

PHASE1_BUDGET   = 100           # сколько random попыток на проигранный сид
PHASE2_BUDGET   = 0         # CEM-добор если фаза 1 не дала победы
CEM_ELITE       = 5            # сколько лучших попыток фазы 1 берём для CEM
EARLY_STOP      = True         # после первой победы на сиде — следующий сид

MAX_STEPS_PER_MATCH = 300      # лимит ходов в локальном детерминированном матче

N_WORKERS       = max(1, os.cpu_count() // 2)

# Пространство поиска: (lo, hi). Если оба int — сэмплим randint, иначе uniform.
SEARCH_SPACE = {
    'activity_weight':   (0.0,  8.0),
    'idle_floor':        (10,   40),       # int
    'distance_comfort':  (0.0,  0.7),
    'risk_tolerance':    (0,    2),        # int
    'ships_weight':      (0.0,  8.0),
    'eta_bonus':         (5.0, 100.0),
    'priority_bonus':    (0.0,  6.0),
    'stress_top_k':      (3,    6),        # int
    'stress_gamma':      (0.1,  1.9),
}

# ══════════════════════════════════════════════════════════════════════════

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUR_PATH    = os.path.join(PROJECT_ROOT, "agent_bundle_swarm", "agent.py")
SUB_PATH    = os.path.join(PROJECT_ROOT, "sub2.py")
TS          = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_CSV     = os.path.join(RESULTS_DIR, f"reverse_tour_{TS}.csv")
SUMMARY_TXT = os.path.join(RESULTS_DIR, f"reverse_tour_summary_{TS}.txt")


# ──────────────────────────────────────────────────────────────────────────
# Воркер
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
    """Один матч: (seed, weights_overrides_dict) → metrics dict.

    Использует local_match.run_match (обход kaggle_environments). Полностью
    детерминирован: тот же seed + те же weights → тот же исход.
    """
    seed, overrides = task
    _silence_debug_logs()

    bundle_dir = os.path.join(PROJECT_ROOT, "agent_bundle_swarm")
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    # уникальный suffix чтобы избежать кеша sys.modules
    suffix = f"{seed}_{abs(hash(frozenset(overrides.items()))) % 10**8}"
    our_mod = _load_module(OUR_PATH, f"_our_rev_{suffix}")
    sub_mod = _load_module(SUB_PATH, f"_sub_rev_{suffix}")

    # инъекция SwarmWeights
    from swarm import SwarmWeights
    base = asdict(SwarmWeights())   # дефолтный полный набор
    base.update(overrides)
    for f in dc_fields(SwarmWeights):
        if f.type is int and f.name in base:
            base[f.name] = int(base[f.name])
    our_mod.SWARM_WEIGHTS = SwarmWeights(**base)
    # sanity: убеждаемся что инъекция реально применилась
    assert our_mod.SWARM_WEIGHTS.activity_weight == base['activity_weight'], \
        "SWARM_WEIGHTS не инжектились в our_mod"

    # детерминированный локальный матч
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from local_match import run_match
    res = run_match(seed, our_mod.agent, sub_mod.agent,
                    bundle_dir=bundle_dir, max_steps=MAX_STEPS_PER_MATCH)

    return {
        'seed': seed,
        **{f'w_{k}': v for k, v in overrides.items()},
        'reward_our': +1 if res['win_a'] else (0 if res['draw'] else -1),
        'reward_sub': +1 if res['win_b'] else (0 if res['draw'] else -1),
        'win': res['win_a'],
        'draw': res['draw'],
        'steps': res['steps'],
        'ships_our_final': res['ships_a_final'],
        'ships_sub_final': res['ships_b_final'],
        'ships_diff': res['ships_diff'],
        'time_sec': res['time_sec'],
    }


# ──────────────────────────────────────────────────────────────────────────
# Сэмплирование весов
# ──────────────────────────────────────────────────────────────────────────

def _sample_uniform(rng):
    """Random uniform sample из SEARCH_SPACE."""
    out = {}
    for k, (lo, hi) in SEARCH_SPACE.items():
        if isinstance(lo, int) and isinstance(hi, int):
            out[k] = rng.randint(lo, hi)
        else:
            out[k] = round(rng.uniform(lo, hi), 4)
    return out


def _sample_cem(rng, elite_rows):
    """Сэмплим из Гауссиана вокруг элиты. Каждый параметр — независимый
    нормал N(mean_elite, std_elite + small_floor), клипнутый к диапазону.
    """
    out = {}
    for k, (lo, hi) in SEARCH_SPACE.items():
        col = f'w_{k}'
        vals = [r[col] for r in elite_rows if col in r]
        if not vals:
            # fallback на uniform если элиты нет
            out[k] = _sample_uniform(rng)[k]
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals)) + (hi - lo) * 0.08   # пол std чтобы не схлопывалось
        v = rng.gauss(m, s)
        v = max(lo, min(hi, v))
        out[k] = int(round(v)) if isinstance(lo, int) and isinstance(hi, int) else round(v, 4)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Главная логика
# ──────────────────────────────────────────────────────────────────────────

def split_train_holdout(seeds, frac, split_seed):
    rng = random.Random(split_seed)
    shuffled = list(seeds); rng.shuffle(shuffled)
    n_train = int(len(shuffled) * frac)
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def run_baseline(seeds):
    """Baseline на дефолтных весах (overrides={}). Возвращает list[dict]."""
    print(f"[baseline]  N={len(seeds)} seeds")
    tasks = [(s, {}) for s in seeds]
    return _run_tasks(tasks, label='baseline')


def _run_tasks(tasks, label='matches'):
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
                results.append(r)
                wf = '✓' if r['win'] else ('=' if r['draw'] else '✗')
                print(f"  [{label}] {i}/{len(tasks)} seed={r['seed']} {wf} "
                      f"diff={r['ships_diff']:+d} t={r['time_sec']:.1f}s")
            except Exception as e:
                seed, overrides = futures[fut]
                print(f"  [{label}] seed={seed} FAILED: {e}")
    print(f"[{label}] done in {time.time()-t0:.0f}s")
    return results


def _short_ov(ov):
    """Компактный вид override-словаря для лога."""
    parts = []
    for k, v in ov.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2g}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def search_for_seed(seed, rng_seed, baseline_row=None):
    """Random + опц. CEM поиск веса для одного проигранного сида.
    Возвращает (rows, won_at_phase, won_at_trial, best_overrides).
    """
    rng = random.Random(rng_seed)
    rows = []

    if baseline_row is not None:
        bs = baseline_row['steps']
        print(f"  baseline: ✗ diff={baseline_row['ships_diff']:+d} steps={bs}")
        if bs <= 80:
            print(f"  ⚠ матч закончился рано (steps={bs}) — мы потеряли всё. "
                  f"Веса не помогут если у агента нет жизнеспособных планов на этой карте.")
            print(f"  → попробуй включить enable_redistribute, расширить risk_tolerance до 2, или признай как структурный dead-end")

    # ── фаза 1: random ──────────────────────────────────────────────────
    phase1_tasks = [(seed, _sample_uniform(rng)) for _ in range(PHASE1_BUDGET)]
    for trial_idx, (s, ov) in enumerate(phase1_tasks):
        r = run_one((s, ov))
        r.update({'phase': 1, 'trial': trial_idx})
        rows.append(r)
        wf = '✓ WIN' if r['win'] else '✗'
        print(f"  P1.{trial_idx:>2}  diff={r['ships_diff']:+5d} steps={r['steps']:>3} "
              f"{wf}  | {_short_ov(ov)}")
        if r['win']:
            return rows, 1, trial_idx, ov

    # ── фаза 2: CEM ─────────────────────────────────────────────────────
    if PHASE2_BUDGET <= 0:
        return rows, None, None, None

    sorted_rows = sorted(rows, key=lambda x: -x['ships_diff'])
    elite = sorted_rows[:CEM_ELITE]
    print(f"  ── фаза 2 (CEM around top-{len(elite)} by ships_diff) ──")
    for trial_idx in range(PHASE2_BUDGET):
        ov = _sample_cem(rng, elite)
        r = run_one((seed, ov))
        r.update({'phase': 2, 'trial': PHASE1_BUDGET + trial_idx})
        rows.append(r)
        wf = '✓ WIN' if r['win'] else '✗'
        print(f"  P2.{trial_idx:>2}  diff={r['ships_diff']:+5d} steps={r['steps']:>3} "
              f"{wf}  | {_short_ov(ov)}")
        if r['win']:
            return rows, 2, PHASE1_BUDGET + trial_idx, ov

    return rows, None, None, None


def _consensus_weights(rows_won):
    """По строкам с win=1 берём медиану по каждому весу. Это «общеприменимый»
    режим, который мы потом проверим на holdout."""
    if not rows_won:
        return None
    consensus = {}
    for k, (lo, hi) in SEARCH_SPACE.items():
        col = f'w_{k}'
        vals = [r[col] for r in rows_won if col in r]
        if not vals:
            continue
        m = float(np.median(vals))
        if isinstance(lo, int) and isinstance(hi, int):
            m = int(round(m))
        else:
            m = round(m, 4)
        consensus[k] = m
    return consensus


def main():
    seeds = list(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))
    train_seeds, holdout_seeds = split_train_holdout(seeds, TRAIN_FRAC, SPLIT_SEED)
    print(f"split: train={len(train_seeds)} holdout={len(holdout_seeds)}")

    # ── 1. Baseline на TRAIN ────────────────────────────────────────────
    base_rows = run_baseline(train_seeds)
    for r in base_rows:
        r['phase'] = 0; r['trial'] = -1
    base_loss = [r['seed'] for r in base_rows if not r['win']]
    base_winrate = sum(r['win'] for r in base_rows) / max(1, len(base_rows))
    print(f"\nbaseline TRAIN winrate: {base_winrate:.3f} ({len(base_rows)-len(base_loss)}/{len(base_rows)})")
    print(f"проигранных сидов: {len(base_loss)} → начинаем поиск")

    # ── 2. Поиск по проигрышам ─────────────────────────────────────────
    all_rows = list(base_rows)
    won_seeds = []
    deadend_seeds = []
    base_by_seed = {r['seed']: r for r in base_rows}
    for i, seed in enumerate(base_loss, 1):
        print(f"\n[search] {i}/{len(base_loss)} seed={seed}")
        rows, phase, trial, best_ov = search_for_seed(
            seed, rng_seed=SPLIT_SEED + seed, baseline_row=base_by_seed.get(seed),
        )
        all_rows.extend(rows)
        if phase is not None:
            won_seeds.append((seed, phase, trial, best_ov))
            print(f"  → solved at phase={phase} trial={trial}")
        else:
            deadend_seeds.append(seed)
            print(f"  → DEAD-END after {len(rows)} trials")

    # ── 3. Сохраняем все матчи ──────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved: {OUT_CSV}  ({len(df)} rows)")

    # ── 4. Consensus weights → проверка на HOLDOUT ─────────────────────
    won_rows = [r for r in all_rows if r['win'] and r['phase'] in (1, 2)]
    consensus = _consensus_weights(won_rows)
    holdout_winrate_base = None
    holdout_winrate_cons = None

    if holdout_seeds:
        print(f"\n[holdout]  baseline на {len(holdout_seeds)} сидах")
        ho_base = _run_tasks([(s, {}) for s in holdout_seeds], label='holdout-base')
        for r in ho_base: r['phase'] = -1; r['trial'] = -1
        all_rows.extend(ho_base)
        holdout_winrate_base = sum(r['win'] for r in ho_base) / max(1, len(ho_base))

        if consensus:
            print(f"\n[holdout]  consensus weights = {consensus}")
            ho_cons = _run_tasks([(s, consensus) for s in holdout_seeds],
                                 label='holdout-consensus')
            for r in ho_cons: r['phase'] = -2; r['trial'] = -1
            all_rows.extend(ho_cons)
            holdout_winrate_cons = sum(r['win'] for r in ho_cons) / max(1, len(ho_cons))

        # перезаписываем CSV с holdout
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

    # ── 5. Summary ─────────────────────────────────────────────────────
    train_winrate_after = (
        len(base_rows) - len(deadend_seeds)
    ) / max(1, len(base_rows))
    summary = []
    summary.append(f"reverse_tournament summary — {TS}")
    summary.append(f"seeds: {SEED_OFFSET}..{SEED_OFFSET+N_SEEDS-1}  "
                   f"train={len(train_seeds)} holdout={len(holdout_seeds)}")
    summary.append("")
    summary.append(f"TRAIN baseline winrate:  {base_winrate:.3f}")
    summary.append(f"TRAIN after-search WR:   {train_winrate_after:.3f}  "
                   f"(solved {len(won_seeds)} of {len(base_loss)} losses)")
    if holdout_winrate_base is not None:
        summary.append(f"HOLDOUT baseline:        {holdout_winrate_base:.3f}")
    if holdout_winrate_cons is not None:
        summary.append(f"HOLDOUT consensus:       {holdout_winrate_cons:.3f}  "
                       f"(должно быть ≥ baseline; иначе переобучение)")
    summary.append("")
    summary.append(f"DEAD-END seeds ({len(deadend_seeds)}): {deadend_seeds}")
    summary.append("  → их разбираем отдельно: или баг агента, или структурно невозможные карты.")
    summary.append("")
    if consensus:
        summary.append("Consensus weights (median по всем победным trial-row):")
        for k, v in consensus.items():
            summary.append(f"  {k:<22} {v}")
    summary.append("")
    summary.append(f"data: {OUT_CSV}")
    summary_text = "\n".join(summary)
    with open(SUMMARY_TXT, 'w') as f:
        f.write(summary_text)
    print("\n" + summary_text)
    print(f"\nsaved summary: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
