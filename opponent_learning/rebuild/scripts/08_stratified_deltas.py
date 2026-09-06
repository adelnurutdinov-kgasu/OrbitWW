#!/usr/bin/env python3
"""
08_stratified_deltas.py — delta-features в разрезе типов карты.

ГИПОТЕЗА (пользователя)
───────────────────────
Карты сильно различаются по структуре: число планет (20–36), есть ли
орбитальные планеты (0 или 4), omega, плотность. Стратегии победителей
на разных типах карт могут отличаться, и усреднение по всем картам
размазывает сигнал. Если стратифицировать — внутри каждого типа карты
эффект может быть резче.

ЧТО ДЕЛАЕМ
──────────
  1. Загружаем delta_features.csv (06) + map_features.csv (07)
  2. Кластеризуем карты на 4–6 типов (KMeans на ключевых параметрах)
  3. Для каждого типа карты пересчитываем Cohen's d для всех delta-фич
  4. Сравниваем общий d с d внутри страт:
       - усиливается ли эффект внутри страт (значит стратификация полезна)
       - меняется ли топ-список фич между типами карт (значит стратегии разные)

ВХОД
────
  rebuild/data/delta_features.csv  (из 06)
  rebuild/data/map_features.csv    (из 07)

ВЫХОД
─────
  rebuild/reports/stratified_deltas.md
  rebuild/reports/stratified_*.png
  rebuild/data/map_clusters.csv       — какая карта в каком кластере
"""

from __future__ import annotations

import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_PATH = DATA_DIR / "delta_features.csv"
MAP_PATH   = DATA_DIR / "map_features.csv"
CLUST_OUT  = DATA_DIR / "map_clusters.csv"


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va + vb == 0:
        return 0.0, 1.0
    se = sqrt(va/len(a) + vb/len(b))
    t = (a.mean() - b.mean()) / se
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


def cohens_d(a, b):
    """Стандартизованный эффект-размер."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sd = 0.5 * (a.std(ddof=1) + b.std(ddof=1))
    if sd < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / sd)


def cluster_maps(maps: pd.DataFrame) -> pd.DataFrame:
    """Кластеризуем карты. Сначала пробуем явную таксономию по n_planets и n_orbital,
    потом KMeans если нужно тоньше."""
    maps = maps.copy()

    # Явная стратификация: размер карты × наличие орбитальных
    def size_bucket(n):
        if n <= 24:   return 'small'
        elif n <= 32: return 'medium'
        else:         return 'large'

    maps['size_bucket'] = maps['n_planets'].apply(size_bucket)
    maps['has_orbital'] = (maps['n_orbital'] > 0).astype(int)
    maps['map_type'] = maps['size_bucket'] + '_' + maps['has_orbital'].map({0: 'static', 1: 'orbital'})

    print('\n── Явная стратификация (size × orbital) ──────────────')
    print(maps.groupby('map_type').size().to_string())

    # KMeans на нормализованных численных фичах — справочно
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        feat_cols = ['n_planets', 'n_orbital', 'omega', 'neutral_prod_share',
                     'mean_inter_dist', 'min_inter_dist', 'sun_blockage_share',
                     'home_to_home_median']
        feat_cols = [c for c in feat_cols if c in maps.columns]
        X = StandardScaler().fit_transform(maps[feat_cols].fillna(0).values)
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        maps['kmeans_cluster'] = km.fit_predict(X)
        print('\n── KMeans (k=5, справочно) ────────────────────────────')
        print(maps.groupby('kmeans_cluster').size().to_string())
        # Центроиды (на оригинальной шкале)
        centers_real = pd.DataFrame(
            km.cluster_centers_ @ np.diag(np.array([maps[c].std() for c in feat_cols])) +
            np.array([maps[c].mean() for c in feat_cols]),
            columns=feat_cols,
        )
        print('\nKMeans centroids:')
        print(centers_real.round(2).to_string())
    except ImportError:
        print('  (sklearn недоступен — kmeans пропущен)')

    return maps


def analyze_per_stratum(deltas: pd.DataFrame, maps: pd.DataFrame,
                       group_col: str = 'map_type') -> pd.DataFrame:
    """Внутри каждой страты — Cohen's d для всех delta-фич."""
    merged = deltas.merge(maps[['episode_id', group_col]],
                          on='episode_id', how='left')
    delta_cols = [c for c in deltas.columns if c.startswith('delta_')]

    rows = []
    strata = sorted(merged[group_col].dropna().unique())
    for s in strata:
        sub = merged[merged[group_col] == s]
        if len(sub) < 20:
            continue
        win = sub[sub['final_label'] == 1]
        los = sub[sub['final_label'] == 0]
        if len(win) < 5 or len(los) < 5:
            continue
        for f in delta_cols:
            d = cohens_d(win[f].values, los[f].values)
            t, p = welch_t(win[f].values, los[f].values)
            rows.append({
                'stratum':    s,
                'n_win':      len(win),
                'n_los':      len(los),
                'feature':    f,
                'cohens_d':   round(d, 3),
                'log10_p':    round(np.log10(max(p, 1e-300)), 2),
                'mean_win':   round(float(win[f].mean()), 4),
                'mean_los':   round(float(los[f].mean()), 4),
            })
    return pd.DataFrame(rows)


def baseline_overall(deltas: pd.DataFrame) -> pd.DataFrame:
    """Cohen's d на всём датасете (без стратификации) — для сравнения."""
    delta_cols = [c for c in deltas.columns if c.startswith('delta_')]
    win = deltas[deltas['final_label'] == 1]
    los = deltas[deltas['final_label'] == 0]
    rows = []
    for f in delta_cols:
        d = cohens_d(win[f].values, los[f].values)
        rows.append({'feature': f, 'cohens_d_overall': round(d, 3)})
    return pd.DataFrame(rows)


def compare_strata(per_stratum: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    """Для каждой фичи: насколько максимальный |d| по стратам больше |d| в среднем."""
    pivot = per_stratum.pivot_table(index='feature', columns='stratum',
                                    values='cohens_d', aggfunc='first')
    merged = pivot.merge(overall.set_index('feature'),
                          left_index=True, right_index=True, how='left')
    merged['max_abs_d_in_strata'] = pivot.abs().max(axis=1).round(3)
    merged['amplification'] = (merged['max_abs_d_in_strata'] -
                                merged['cohens_d_overall'].abs()).round(3)
    return merged.sort_values('amplification', ascending=False)


def main():
    if not DELTA_PATH.exists():
        print(f'⚠ нет {DELTA_PATH} — запустите 06_delta_features.py')
        sys.exit(1)
    if not MAP_PATH.exists():
        print(f'⚠ нет {MAP_PATH} — запустите 07_map_features.py')
        sys.exit(1)

    print(f'Loading {DELTA_PATH} …')
    deltas = pd.read_csv(DELTA_PATH)
    print(f'  {len(deltas)} партий')

    print(f'Loading {MAP_PATH} …')
    maps = pd.read_csv(MAP_PATH)
    maps['episode_id'] = maps['episode_id'].astype(str)
    deltas['episode_id'] = deltas['episode_id'].astype(str)
    print(f'  {len(maps)} карт')

    # Кластеризация карт
    maps_with_clusters = cluster_maps(maps)
    maps_with_clusters.to_csv(CLUST_OUT, index=False)
    print(f'\n✓ Сохранено: {CLUST_OUT}')

    # Анализ
    print('\n── Cohen\'s d на всём датасете (baseline) ─────────────')
    overall = baseline_overall(deltas)
    print(overall.assign(abs_d=overall['cohens_d_overall'].abs())
          .sort_values('abs_d', ascending=False).head(10)
          [['feature', 'cohens_d_overall']].to_string(index=False))

    # По явному типу карты
    print('\n── Cohen\'s d ВНУТРИ страт (явная таксономия) ─────────')
    per_strat = analyze_per_stratum(deltas, maps_with_clusters, 'map_type')
    if len(per_strat):
        # Топ-10 фич × все страты
        top_features = (per_strat.groupby('feature')['cohens_d']
                                  .apply(lambda x: x.abs().max())
                                  .sort_values(ascending=False).head(10).index.tolist())
        for f in top_features:
            sub = per_strat[per_strat['feature'] == f]
            print(f'\n  {f}:')
            for _, r in sub.iterrows():
                print(f"    {r['stratum']:<18} n_win={r['n_win']:>3}  n_los={r['n_los']:>3}  "
                      f"d={r['cohens_d']:+.3f}")

    # Сравнение усиления
    print('\n── Усиление эффекта при стратификации ────────────────')
    print('(max |d| в страте) - (|d| общий) — насколько лучше внутри страт')
    cmp_df = compare_strata(per_strat, overall)
    print(cmp_df[['cohens_d_overall', 'max_abs_d_in_strata', 'amplification']]
          .head(15).to_string())

    # Markdown отчёт
    md = []
    md.append('# Стратифицированный анализ delta-features по типам карт\n\n')
    md.append(f'**Данные**: {len(deltas)} партий, {len(maps)} карт\n\n')
    md.append('## Размеры страт (size × orbital)\n\n')
    md.append('```\n')
    md.append(maps_with_clusters.groupby('map_type').size().to_string())
    md.append('\n```\n\n')

    md.append('## Cohen\'s d общий vs максимальный внутри страты\n\n')
    md.append('Если max_in_strata > overall — стратификация выявляет более резкий сигнал.\n\n')
    md.append('| feature | overall d | max |d| в страте | усиление |\n|---|---|---|---|\n')
    for f, r in cmp_df.iterrows():
        md.append(f"| {f} | {r['cohens_d_overall']} | {r['max_abs_d_in_strata']} | "
                  f"{r['amplification']} |\n")

    md.append('\n## Топ-10 фич × все страты\n\n')
    md.append('| feature | страта | n_win | n_los | Cohen\'s d |\n|---|---|---|---|---|\n')
    if len(per_strat):
        top_features = (per_strat.groupby('feature')['cohens_d']
                                  .apply(lambda x: x.abs().max())
                                  .sort_values(ascending=False).head(10).index.tolist())
        for f in top_features:
            sub = per_strat[per_strat['feature'] == f].sort_values('cohens_d', ascending=False)
            for _, r in sub.iterrows():
                md.append(f"| {r['feature']} | {r['stratum']} | {r['n_win']} | "
                          f"{r['n_los']} | {r['cohens_d']} |\n")

    md.append('\n## Графики\n\n')
    md.append('- `stratified_amplification.png` — усиление по фичам\n')
    md.append('- `stratified_top_features.png` — топ фич по стратам\n')

    (REPORT_DIR / 'stratified_deltas.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/stratified_deltas.md')

    # Графики
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 1) Усиление: top-15 фич
        top15 = cmp_df.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(top15))
        ax.barh(y, top15['cohens_d_overall'].abs(), alpha=0.4, label='|overall d|', color='gray')
        ax.barh(y, top15['max_abs_d_in_strata'], alpha=0.6, label='max |d| in stratum', color='steelblue')
        ax.set_yticks(y)
        ax.set_yticklabels(top15.index, fontsize=8)
        ax.set_xlabel('|Cohen\'s d|')
        ax.set_title('Усиление эффекта при стратификации по типу карты')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'stratified_amplification.png', dpi=120)
        plt.close()

        # 2) Тепловая карта: топ-10 фич × страты
        top10 = (per_strat.groupby('feature')['cohens_d']
                          .apply(lambda x: x.abs().max())
                          .sort_values(ascending=False).head(10).index.tolist())
        heat = per_strat[per_strat['feature'].isin(top10)].pivot_table(
            index='feature', columns='stratum', values='cohens_d', aggfunc='first'
        ).loc[top10]
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heat.values, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index, fontsize=9)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                v = heat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                             fontsize=7, color='white' if abs(v) > 1.5 else 'black')
        plt.colorbar(im, ax=ax, label="Cohen's d")
        ax.set_title('Cohen\'s d по топ-10 фичам × тип карты')
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'stratified_top_features.png', dpi=120)
        plt.close()
        print('  графики сохранены')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
