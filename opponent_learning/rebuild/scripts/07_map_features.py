#!/usr/bin/env python3
"""
07_map_features.py — характеристики карты для каждого эпизода.

ЗАЧЕМ
─────
Карты разные: omega, число планет, плотность, доля нейтрального production,
начальные расстояния. Если различия карт сильные, то одинаковая стратегия
на разных картах ведёт к разным исходам — это размазывает сигнал в общем
анализе. Мы хотим иметь возможность стратифицировать по карте.

ЧТО СЧИТАЕМ (из initial_planets первого шага каждого эпизода)
─────────────────────────────────────────────────────────────
  • omega                 — скорость вращения орбитальных планет
  • n_planets             — общее число планет
  • n_orbital, n_static   — разбивка по типам (орбитальные внутри ROTATION_LIMIT)
  • prod_total            — суммарное производство (всех планет)
  • prod_mean, prod_std   — статистика по производству
  • neutral_prod_share    — доля производства у нейтралов (важная стратегическая мера)
  • n_neutral_planets     — нейтральных планет на старте
  • mean_inter_dist       — среднее попарное расстояние между планетами
  • min_inter_dist        — минимальное (плотность)
  • max_inter_dist        — диаметр карты
  • home_to_nearest_neutral — медиана расстояний от домашних к ближайшим нейтралам
  • home_to_enemy_home    — расстояние между домашними планетами игроков
  • sun_blockage_share    — доля попарных линий между планетами которые блокируются солнцем

ВХОД
────
  opponent_learning/data/raw/*.json   — replay JSONs

ВЫХОД
─────
  rebuild/data/map_features.csv

ЛЁГКИЙ скрипт (~1 минута: только initial_planets каждого JSON, не все шаги).
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
OUT_PATH   = DATA_DIR / "map_features.csv"

# Константы карты из shooting.py
BOARD = 100.0
CX, CY = 50.0, 50.0
SUN_R  = 6.0
ROTATION_LIMIT = 30.0  # планеты внутри этого радиуса — орбитальные

# Будем подгружать shooting.is_orbital если получится, иначе считаем сами
def is_orbital(x: float, y: float, radius: float) -> bool:
    """Орбитальная если центр + радиус помещается внутри ROTATION_LIMIT от солнца."""
    return math.hypot(x - CX, y - CY) + radius <= ROTATION_LIMIT


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float,
                     sun_r: float = SUN_R) -> bool:
    """Пересекает ли отрезок между двумя точками солнце."""
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
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def extract_map_features(json_path: Path) -> dict | None:
    """Извлекает map features из step=0 (initial_planets)."""
    try:
        with open(json_path) as f:
            ep = json.load(f)
    except Exception:
        return None

    steps = ep.get('steps') or []
    if not steps:
        return None
    s0 = steps[0]
    if not isinstance(s0, list) or len(s0) == 0:
        return None
    obs = s0[0].get('observation', {})
    omega = float(obs.get('angular_velocity', 0.0) or 0.0)

    # initial_planets: [id, owner, x, y, radius, ships, prod]
    init = obs.get('initial_planets') or obs.get('planets') or []
    if not init:
        return None

    # Структура планет
    xs = np.array([float(p[2]) for p in init])
    ys = np.array([float(p[3]) for p in init])
    rs = np.array([float(p[4]) for p in init])
    owners = np.array([int(p[1]) for p in init])
    prods = np.array([float(p[6]) for p in init])

    n_planets = len(init)
    orbital_flags = np.array([is_orbital(x, y, r)
                              for x, y, r in zip(xs, ys, rs)])
    n_orbital = int(orbital_flags.sum())
    n_static  = n_planets - n_orbital

    # Production
    prod_total = float(prods.sum())
    prod_mean  = float(prods.mean())
    prod_std   = float(prods.std())
    neutral_mask = owners == -1
    n_neutral  = int(neutral_mask.sum())
    neutral_prod_share = (float(prods[neutral_mask].sum() / max(prod_total, 1e-9))
                          if n_neutral > 0 else 0.0)

    # Парные расстояния
    dists = []
    sun_blockages = 0
    pairs_total = 0
    for i, j in combinations(range(n_planets), 2):
        d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
        dists.append(d)
        # с учётом радиусов планет — реальная "длина пути"
        if segment_hits_sun(xs[i], ys[i], xs[j], ys[j]):
            sun_blockages += 1
        pairs_total += 1
    dists = np.array(dists) if dists else np.array([0.0])
    mean_inter_dist = float(dists.mean())
    min_inter_dist  = float(dists.min())
    max_inter_dist  = float(dists.max())
    std_inter_dist  = float(dists.std())
    sun_blockage_share = sun_blockages / max(pairs_total, 1)

    # Домашние планеты — те у которых owner ∈ {0,1,2,3} на старте
    home_mask = owners >= 0
    n_home = int(home_mask.sum())
    home_xs = xs[home_mask]
    home_ys = ys[home_mask]

    # Расстояния домашних до нейтральных
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

    # Расстояние между домашними планетами (если 2+)
    home_to_home = []
    if n_home >= 2:
        for i, j in combinations(range(n_home), 2):
            home_to_home.append(math.hypot(home_xs[i] - home_xs[j],
                                            home_ys[i] - home_ys[j]))
    home_to_home_median = float(np.median(home_to_home)) if home_to_home else 0.0

    return {
        'episode_id':        json_path.stem,
        'omega':             omega,
        'n_planets':         n_planets,
        'n_orbital':         n_orbital,
        'n_static':          n_static,
        'orbital_share':     n_orbital / max(n_planets, 1),
        'prod_total':        prod_total,
        'prod_mean':         prod_mean,
        'prod_std':          prod_std,
        'n_neutral_start':   n_neutral,
        'neutral_prod_share': neutral_prod_share,
        'mean_inter_dist':   mean_inter_dist,
        'min_inter_dist':    min_inter_dist,
        'max_inter_dist':    max_inter_dist,
        'std_inter_dist':    std_inter_dist,
        'sun_blockage_share': sun_blockage_share,
        'home_to_nearest_neutral_median': home_to_nearest_neutral_median,
        'home_to_home_median': home_to_home_median,
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
    errors  = 0
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

    # Сводка
    print('\n── Сводка по характеристикам карт ────────────────────')
    cols = ['omega', 'n_planets', 'orbital_share', 'prod_total',
            'neutral_prod_share', 'mean_inter_dist', 'max_inter_dist',
            'sun_blockage_share', 'home_to_home_median']
    print(df[cols].describe().round(3).to_string())

    # Распределение n_planets — карты бывают разного размера?
    print(f'\n  n_planets distribution: {df["n_planets"].value_counts().sort_index().to_dict()}')
    print(f'  n_orbital distribution: {df["n_orbital"].value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
