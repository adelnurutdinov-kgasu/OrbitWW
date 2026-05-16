#!/usr/bin/env python3
"""
03_cluster_and_fit.py — кластеризует поведенческие профили оппонентов
и фиттит SwarmWeights-параметры к каждому кластеру.

Алгоритм:
  1. Читаем features.csv (вывод 02_extract_features.py).
  2. Нормализуем фичи (StandardScaler).
  3. K-Means с k=5 → 5 поведенческих архетипов.
  4. Для каждого кластера: маппинг центроида на SwarmWeights параметры.
  5. Вычисляем prior = частота каждого кластера.
  6. Сохраняем results/clusters.json и results/presets_generated.py.

Маппинг фич → SwarmWeights:
  attack_rate       → activity_weight  (высокая активность → высокий вес)
  expansion_bias    → prio_reclassify_thr (любит нейтралов → ниже порог)
  ships_fraction    → idle_floor  (отправляет много → низкий floor)
  avg_distance_frac → distance_comfort (любит дальние → выше комфорт)
  overkill_ratio    → risk_tolerance (экономит ships → 0, расточителен → 2)
  early_rate        → (влияет на activity_weight в ранней игре)
  idle_peak_frac    → idle_floor дополнительно (высокий idle → высокий floor)

Запуск:
  python3 opponent_learning/scripts/03_cluster_and_fit.py
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    raise ImportError("pip install scikit-learn --break-system-packages")

HERE      = Path(__file__).parent
PROC_DIR  = HERE.parent / "data" / "processed"
RES_DIR   = HERE.parent / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'attack_rate',
    'expansion_bias',
    'ships_fraction',
    'avg_distance_frac',
    'overkill_ratio',
    'early_rate',
    'late_rate',
    'multi_launch_rate',
    'idle_peak_frac',
]

# ── Маппинг центроида кластера → SwarmWeights параметры ─────────────────────
#
# Правило «линейного маппинга» для каждой фичи:
#   activity_weight:      attack_rate ∈ [0, 1]  → [0.2, 5.0]
#   idle_floor:           ships_fraction ∈ [0, 1] → [5, 40]  (обратный)
#   distance_comfort:     avg_distance_frac ∈ [0, 1] → [0.0, 0.7]
#   risk_tolerance:       overkill_ratio ∈ [0.5, 3.0] → [0, 2]
#   prio_reclassify_thr:  expansion_bias ∈ [0, 1] → [1.5, 0.4]  (обратный)
#   priority_bonus:       expansion_bias → [1.0, 5.0]  (прямой)
#
# Ограничения (clamp) применяются после маппинга.

def _lerp(v, in_lo, in_hi, out_lo, out_hi):
    t = (v - in_lo) / max(1e-9, in_hi - in_lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


def centroid_to_weights(c: dict) -> dict:
    """Центроид (dict фич) → dict параметров SwarmWeights."""
    ar   = c['attack_rate']          # 0..1
    eb   = c['expansion_bias']       # 0..1 (высокий = любит нейтралов)
    sf   = c['ships_fraction']       # 0..1 (высокий = расточителен)
    adf  = c['avg_distance_frac']    # 0..1 (высокий = дальние цели)
    ok   = c['overkill_ratio']       # 0.5..3+
    er   = c['early_rate']           # 0..1
    ipf  = c['idle_peak_frac']       # 0..1

    # activity_weight: активный игрок → высокий вес бездействия наказывается
    activity_weight = _lerp(ar, 0.0, 1.0, 0.3, 5.0)
    # Корректируем по early_rate: активен с самого начала → ещё выше
    activity_weight = activity_weight * (1.0 + 0.3 * er)
    activity_weight = max(0.3, min(6.0, activity_weight))

    # idle_floor: расточительный (высокий sf) оставляет мало → низкий floor
    # осторожный (низкий sf) держит много → высокий floor
    idle_floor = int(_lerp(sf, 0.0, 1.0, 40, 5))
    # Коррекция по idle_peak_frac (реальные залежи)
    idle_floor = int(idle_floor * (0.7 + 0.6 * ipf))
    idle_floor = max(3, min(45, idle_floor))

    # distance_comfort: любит дальние цели → выше comfort
    distance_comfort = _lerp(adf, 0.0, 1.0, 0.0, 0.6)
    distance_comfort = round(max(0.0, min(0.7, distance_comfort)), 2)

    # risk_tolerance: экономный (ok ~1.0, отправляет впритык) → агрессивный (2)
    # расточительный (ok >5, отправляет с большим запасом) → осторожный (0)
    # В нашей модели: risk_tolerance=2 = минимальный буфер (рискует), 0 = консерватор
    risk_tolerance = int(round(_lerp(ok, 1.0, 5.0, 2, 0)))
    risk_tolerance = max(0, min(2, risk_tolerance))

    # prio_reclassify_thr: любит нейтралов → агрессивная reclassify (низкий порог)
    prio_thr = _lerp(eb, 0.0, 1.0, 1.5, 0.4)
    prio_thr = round(max(0.4, min(1.6, prio_thr)), 2)

    # priority_bonus: важность production у целей
    priority_bonus = _lerp(eb, 0.0, 1.0, 0.5, 4.0)
    priority_bonus = round(max(0.5, min(5.0, priority_bonus)), 2)

    # ships_weight: осторожный оппонент держит много кораблей → выше вес
    ships_weight = _lerp(1.0 - sf, 0.0, 1.0, 1.0, 6.0)
    ships_weight = round(max(1.0, min(6.0, ships_weight)), 2)

    return {
        'activity_weight':       round(activity_weight, 2),
        'idle_floor':            idle_floor,
        'distance_comfort':      distance_comfort,
        'risk_tolerance':        risk_tolerance,
        'prio_reclassify_thr':   prio_thr,
        'priority_bonus':        priority_bonus,
        'ships_weight':          ships_weight,
    }


# ── Именование кластеров ─────────────────────────────────────────────────────

def name_cluster(c: dict, cluster_id: int) -> str:
    """Даём кластеру читаемое имя по доминирующим фичам."""
    ar  = c['attack_rate']
    eb  = c['expansion_bias']
    sf  = c['ships_fraction']
    er  = c['early_rate']
    ipf = c['idle_peak_frac']

    if ar < 0.25:
        return f'c{cluster_id}_turtle'       # очень пассивный
    if ar > 0.6 and er > 0.5:
        return f'c{cluster_id}_rusher'        # ранний агрессор
    if eb > 0.65:
        return f'c{cluster_id}_expander'      # любит нейтралов
    if sf > 0.55 and ar > 0.4:
        return f'c{cluster_id}_brawler'       # расточительный агрессор
    if ipf > 0.6:
        return f'c{cluster_id}_hoarder'       # накапливает корабли
    return f'c{cluster_id}_balanced'


# ── Python-код для opponent_presets.py ──────────────────────────────────────

def generate_presets_code(clusters: list) -> str:
    """Генерирует фрагмент кода для вставки в opponent_presets.py."""
    lines = [
        "# ── Data-driven presets (auto-generated by 03_cluster_and_fit.py) ──",
        "# Обучены на replay-логах топ-игроков Kaggle orbit_wars.",
        "# Обновить: python3 opponent_learning/scripts/03_cluster_and_fit.py",
        "",
    ]
    for cl in clusters:
        name   = cl['name']
        w      = cl['weights']
        prior  = cl['prior']
        desc   = cl['description']
        lines += [
            f"    # {name}  prior={prior:.3f}  —  {desc}",
            f"    '{name}': SwarmWeights(",
        ]
        for k, v in w.items():
            if isinstance(v, float):
                lines.append(f"        {k}={v},")
            else:
                lines.append(f"        {k}={v},")
        lines += ["    ),", ""]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default=str(PROC_DIR / 'features.csv'))
    parser.add_argument('--k', type=int, default=5, help='Число кластеров')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-reward', type=float, default=0.0,
                        help='Фильтр: берём только записи с reward >= порога '
                             '(0 = все игроки, 0.5 = только победители)')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    print(f"Загружено {len(df)} записей из {args.features}")

    # Фильтр: убираем строки с NaN в фичах
    df = df.dropna(subset=FEATURE_COLS)
    print(f"После дропа NaN: {len(df)} записей")

    if args.min_reward > 0:
        df = df[df['reward'] >= args.min_reward]
        print(f"После фильтра reward>={args.min_reward}: {len(df)} записей")

    if len(df) < args.k * 10:
        print(f"⚠ Слишком мало данных ({len(df)} записей) для k={args.k}. "
              f"Уменьши k или скачай больше эпизодов.")
        return

    # ── Клипинг выбросов перед кластеризацией ────────────────────────────────
    # overkill_ratio имеет long tail (max=384 при mean~3) — клипаем на 99-й перцентиль
    clip_cols = {
        'overkill_ratio':    df['overkill_ratio'].quantile(0.99),
        'ships_fraction':    1.0,
        'attack_rate':       1.0,
        'expansion_bias':    1.0,
        'avg_distance_frac': 1.0,
        'idle_peak_frac':    1.0,
    }
    for col, cap in clip_cols.items():
        if col in df.columns:
            before = df[col].max()
            df[col] = df[col].clip(upper=cap)
            if before > cap * 1.1:
                print(f"  clip {col}: {before:.2f} → {cap:.2f}")

    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ── K-Means ──────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=20)
    labels = km.fit_predict(Xs)
    df['cluster'] = labels

    sil = silhouette_score(Xs, labels)
    print(f"\nK-Means k={args.k}  silhouette={sil:.3f}  "
          f"(>0.3 хорошо, >0.5 отлично)")

    # ── Обратно в исходное пространство ─────────────────────────────────────
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    centroid_dicts = [
        {f: float(centroids_orig[i, j])
         for j, f in enumerate(FEATURE_COLS)}
        for i in range(args.k)
    ]

    # Частоты
    priors = {}
    for i in range(args.k):
        priors[i] = float((labels == i).mean())

    # ── Собираем кластеры ────────────────────────────────────────────────────
    clusters = []
    for i in range(args.k):
        c    = centroid_dicts[i]
        w    = centroid_to_weights(c)
        name = name_cluster(c, i)

        # Описание
        desc_parts = []
        if c['attack_rate'] > 0.55:
            desc_parts.append(f"active={c['attack_rate']:.2f}")
        if c['expansion_bias'] > 0.55:
            desc_parts.append(f"expands neutrals={c['expansion_bias']:.2f}")
        if c['idle_peak_frac'] > 0.6:
            desc_parts.append(f"hoards ships={c['idle_peak_frac']:.2f}")
        if c['overkill_ratio'] > 1.8:
            desc_parts.append(f"wasteful={c['overkill_ratio']:.1f}x")
        desc = ', '.join(desc_parts) or 'balanced'

        clusters.append({
            'id':           i,
            'name':         name,
            'prior':        priors[i],
            'centroid':     c,
            'weights':      w,
            'description':  desc,
            'n_samples':    int((labels == i).sum()),
        })

    # ── Вывод в консоль ──────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'Кластер':<30} {'prior':>6}  {'n':>5}  {'ar':>5}  "
          f"{'eb':>5}  {'sf':>5}  {'ok':>5}  {'idle':>5}")
    print('─'*80)
    for cl in sorted(clusters, key=lambda x: -x['prior']):
        c = cl['centroid']
        print(f"  {cl['name']:<28} {cl['prior']:>6.3f}  {cl['n_samples']:>5}  "
              f"{c['attack_rate']:>5.2f}  {c['expansion_bias']:>5.2f}  "
              f"{c['ships_fraction']:>5.2f}  {c['overkill_ratio']:>5.2f}  "
              f"{c['idle_peak_frac']:>5.2f}")
    print('─'*80)

    print("\nМаппинг → SwarmWeights:")
    print(f"{'Кластер':<30} act_w  floor  dist   risk  prio_thr  prio_b  sw")
    for cl in sorted(clusters, key=lambda x: -x['prior']):
        w = cl['weights']
        print(f"  {cl['name']:<28} "
              f"{w['activity_weight']:>5.2f}  {w['idle_floor']:>5}  "
              f"{w['distance_comfort']:>5.2f}  {w['risk_tolerance']:>5}  "
              f"{w['prio_reclassify_thr']:>8.2f}  {w['priority_bonus']:>6.2f}  "
              f"{w['ships_weight']:>4.2f}")

    # ── Сохраняем ────────────────────────────────────────────────────────────
    out_json = RES_DIR / 'clusters.json'
    with open(out_json, 'w') as f:
        json.dump(clusters, f, indent=2)
    print(f"\n✓ clusters.json → {out_json}")

    out_py = RES_DIR / 'presets_generated.py'
    code   = generate_presets_code(clusters)
    with open(out_py, 'w') as f:
        f.write("# Auto-generated — вставить в opponent_presets.py в словарь PRESETS\n")
        f.write(code)
    print(f"✓ presets_generated.py → {out_py}")

    # Добавляем метку кластера обратно в features
    out_feat = PROC_DIR / 'features_clustered.csv'
    df.to_csv(out_feat, index=False)
    print(f"✓ features_clustered.csv → {out_feat}")


if __name__ == '__main__':
    main()
