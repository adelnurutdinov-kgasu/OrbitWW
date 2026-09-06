#!/usr/bin/env python3
"""
07b_map_features_v2.py — характеристики карт (исправлено).

ЧТО ИСПРАВЛЕНО vs 07
────────────────────
1. Кометы исключаем из подсчёта planets (через comet_planet_ids).
2. Orbital определяем ЭМПИРИЧЕСКИ: планета "движется" если её позиция
   изменилась более чем на 0.5 единиц между step=0 и step=10. Это
   робастнее чем использовать константу ROTATION_LIMIT, которая в
   shooting.py (=30) не совпадает с реальной границей Kaggle движка.
3. Главная ось стратификации — n_planets (size_bucket). Удаляем
   ложное orbital/static деление — на всех картах есть движущиеся
   и стационарные планеты, разница только в количествах.

СТРАТИФИКАЦИЯ
─────────────
size_bucket:
   small  → n_planets ≤ 24
   medium → 25 ≤ n_planets ≤ 32
   large  → n_planets ≥ 33

Continous фичи (для тонкого анализа):
   • omega
   • orbital_share (доля движущихся среди основных планет)
   • n_planets
   • mean_inter_dist

ВХОД
────
  opponent_learning/data/raw/*.json

ВЫХОД
─────
  rebuild/data/map_features_v2.csv

СРЕДНИЙ скрипт (~70 секунд: для каждого эпизода загружаем step=0 И step=10).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
RAW_DIR    = OL_DIR / "data" / "raw"
DATA_DIR   = REBUILD / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = DATA_DIR / "map_features_v2.csv"

CX, CY  = 50.0, 50.0
SUN_R   = 6.0
COMPARE_STEP = 10  # сравниваем step=0 с этим шагом для определения orbital


def segment_hits_sun(x1, y1, x2, y2, sun_r=SUN_R):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CX, y1 - CY
    a = dx*dx + dy*dy
    if a < 1e-9:
        return math.hypot(fx, fy) < sun_r
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - sun_r*sun_r
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1, t2 = (-b - sq) / (2*a), (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def extract_map_features(json_path: Path) -> dict | None:
    """Карта-фичи: step=0 (базовая структура) + step=10 (для определения orbital)."""
    try:
        with open(json_path) as f:
            ep = json.load(f)
    except Exception:
        return None

    steps = ep.get('steps') or []
    if len(steps) < COMPARE_STEP + 1:
        return None
    s0 = steps[0]
    s10 = steps[COMPARE_STEP]
    if not (isinstance(s0, list) and isinstance(s10, list)):
        return None
    obs0 = s0[0].get('observation', {})
    obs10 = s10[0].get('observation', {})

    planets0 = obs0.get('planets', [])
    planets10 = obs10.get('planets', [])
    if not planets0 or not planets10:
        return None

    # Кометы в obs0 (обычно [], кометы появляются позже)
    comet_ids = set(obs0.get('comet_planet_ids', []) or [])

    # Фильтр: только non-comet планеты
    base_planets0  = [p for p in planets0 if p[0] not in comet_ids]
    base_ids = {p[0] for p in base_planets0}

    # На step=10 берём ту же базу (не кометы из тогдашних)
    obs10_comet_ids = set(obs10.get('comet_planet_ids', []) or [])
    base_planets10 = [p for p in planets10 if p[0] in base_ids
                      and p[0] not in obs10_comet_ids]

    # Позиции
    pos0 = {p[0]: (float(p[2]), float(p[3])) for p in base_planets0}
    pos10 = {p[0]: (float(p[2]), float(p[3])) for p in base_planets10}

    # Эмпирическое orbital: двигалась ли планета между step=0 и step=10
    moved_ids = set()
    for pid, (x0, y0) in pos0.items():
        if pid in pos10:
            x1, y1 = pos10[pid]
            if math.hypot(x1 - x0, y1 - y0) > 0.5:
                moved_ids.add(pid)

    n_planets = len(base_planets0)
    n_orbital = len(moved_ids)
    orbital_share = n_orbital / max(n_planets, 1)

    # Геометрия (на step=0, только основные планеты)
    xs = np.array([float(p[2]) for p in base_planets0])
    ys = np.array([float(p[3]) for p in base_planets0])
    rs = np.array([float(p[4]) for p in base_planets0])
    owners = np.array([int(p[1]) for p in base_planets0])
    prods = np.array([float(p[6]) for p in base_planets0])

    prod_total = float(prods.sum())
    prod_mean  = float(prods.mean())
    prod_std   = float(prods.std())
    neutral_mask = owners == -1
    n_neutral = int(neutral_mask.sum())
    neutral_prod_share = (float(prods[neutral_mask].sum() / max(prod_total, 1e-9))
                          if n_neutral > 0 else 0.0)

    # Парные расстояния
    dists = []
    sun_blockages = 0
    pairs_total = 0
    for i, j in combinations(range(n_planets), 2):
        d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
        dists.append(d)
        if segment_hits_sun(xs[i], ys[i], xs[j], ys[j]):
            sun_blockages += 1
        pairs_total += 1
    dists = np.array(dists) if dists else np.array([0.0])

    # Дистанции движущихся от центра
    moved_dist_from_sun = []
    if moved_ids:
        for pid in moved_ids:
            x, y = pos0[pid]
            moved_dist_from_sun.append(math.hypot(x - CX, y - CY))
    moved_dist_from_sun = np.array(moved_dist_from_sun) if moved_dist_from_sun else np.array([0.0])

    # Домашние планеты (owner ≥ 0)
    home_mask = owners >= 0
    n_home = int(home_mask.sum())
    home_xs = xs[home_mask]
    home_ys = ys[home_mask]

    home_to_neutral_dists = []
    if n_neutral > 0 and n_home > 0:
        n_xs = xs[neutral_mask]
        n_ys = ys[neutral_mask]
        for hx, hy in zip(home_xs, home_ys):
            ds = np.hypot(n_xs - hx, n_ys - hy)
            home_to_neutral_dists.append(float(ds.min()))
    home_to_nearest_neutral_median = (
        float(np.median(home_to_neutral_dists))
        if home_to_neutral_dists else 0.0
    )

    home_to_home = []
    if n_home >= 2:
        for i, j in combinations(range(n_home), 2):
            home_to_home.append(math.hypot(home_xs[i] - home_xs[j],
                                            home_ys[i] - home_ys[j]))
    home_to_home_median = float(np.median(home_to_home)) if home_to_home else 0.0

    # Size bucket — только по n_planets
    if n_planets <= 24:
        size_bucket = 'small'
    elif n_planets <= 32:
        size_bucket = 'medium'
    else:
        size_bucket = 'large'

    return {
        'episode_id':        json_path.stem,
        'omega':             float(obs0.get('angular_velocity', 0.0) or 0.0),
        'n_planets':         n_planets,
        'n_orbital':         n_orbital,
        'orbital_share':     round(orbital_share, 4),
        'size_bucket':       size_bucket,
        'mean_orbital_dist_from_sun': round(float(moved_dist_from_sun.mean()), 3),
        'max_orbital_dist_from_sun':  round(float(moved_dist_from_sun.max()), 3),
        'prod_total':        prod_total,
        'prod_mean':         prod_mean,
        'prod_std':          prod_std,
        'n_neutral_start':   n_neutral,
        'neutral_prod_share': round(neutral_prod_share, 4),
        'mean_inter_dist':   round(float(dists.mean()), 3),
        'min_inter_dist':    round(float(dists.min()), 3),
        'max_inter_dist':    round(float(dists.max()), 3),
        'std_inter_dist':    round(float(dists.std()), 3),
        'sun_blockage_share': round(sun_blockages / max(pairs_total, 1), 4),
        'home_to_nearest_neutral_median': round(home_to_nearest_neutral_median, 3),
        'home_to_home_median': round(home_to_home_median, 3),
        'n_home':            n_home,
    }


def main():
    files = sorted(p for p in RAW_DIR.glob('*.json')
                   if p.name != 'episodes_index.json')
    if not files:
        print(f'Нет эпизодов в {RAW_DIR}')
        sys.exit(1)

    print(f'Эпизодов: {len(files)}')
    t0 = time.time()
    records = []
    errors = 0
    for i, path in enumerate(files):
        rec = extract_map_features(path)
        if rec is None:
            errors += 1
        else:
            records.append(rec)
        if (i + 1) % 100 == 0:
            print(f'  обработано {i+1}/{len(files)}  ({time.time()-t0:.1f}s)')

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    print(f'\n✓ Сохранено: {OUT_PATH}  ({len(df)} карт за {time.time()-t0:.1f}s, errors={errors})')

    print('\n── Сводка по характеристикам карт (исправленные) ─────')
    cols = ['omega', 'n_planets', 'n_orbital', 'orbital_share',
            'mean_orbital_dist_from_sun', 'max_orbital_dist_from_sun',
            'mean_inter_dist', 'neutral_prod_share']
    print(df[cols].describe().round(3).to_string())

    print(f'\n  size_bucket distribution:\n{df["size_bucket"].value_counts().to_string()}')
    print(f'\n  n_planets distribution: {df["n_planets"].value_counts().sort_index().to_dict()}')
    print(f'\n  n_orbital distribution: {df["n_orbital"].value_counts().sort_index().to_dict()}')
    print(f'  orbital_share: min={df["orbital_share"].min()}, max={df["orbital_share"].max()}, '
          f'median={df["orbital_share"].median()}')
    print(f'\n  Корреляция n_orbital ↔ n_planets: {df[["n_orbital", "n_planets"]].corr().iloc[0,1]:.3f}')


if __name__ == '__main__':
    main()
