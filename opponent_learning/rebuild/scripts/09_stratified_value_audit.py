#!/usr/bin/env python3
"""
09_stratified_value_audit.py — переаудит value_v2 в разрезе типов карты.

ЗАЧЕМ
─────
value_v2 показала AUC=0.906 в среднем, lift over xgb_5macro = +0.0009.
Мы интерпретировали это как "структурные фичи бесполезны". Но это
агрегат по всем картам — на разных типах могут быть **разные**
важные фичи. Аудит:
  1. AUC value_v2 внутри каждой страты — где модель сильна, где слаба
  2. Где slabе — там либо данных мало, либо текущие фичи неподходящие
  3. Кroots stratified vs global: на какой страте global модель
     системно ошибается?

ВХОД
────
  rebuild/data/val_predictions.csv  (из 03)
  rebuild/data/map_features.csv     (из 07)
  rebuild/data/map_clusters.csv     (из 08, для map_type)

ВЫХОД
─────
  rebuild/reports/value_v2_per_stratum.md
  rebuild/reports/value_v2_per_stratum.png

ЛЁГКИЙ скрипт (без xgboost, только predict уже загружены).
"""

from __future__ import annotations

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

PRED_PATH    = DATA_DIR / "val_predictions.csv"
MAP_PATH     = DATA_DIR / "map_features.csv"
CLUSTER_PATH = DATA_DIR / "map_clusters.csv"


def metrics_for(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """AUC / LogLoss / Brier / Accuracy."""
    from sklearn.metrics import (roc_auc_score, log_loss,
                                  brier_score_loss, accuracy_score)
    if len(y_true) < 30:
        return {'n': len(y_true), 'auc': None, 'logloss': None,
                'brier': None, 'acc': None, 'pos_rate': float(y_true.mean())
                if len(y_true) else None}
    if len(np.unique(y_true)) < 2:
        return {'n': len(y_true), 'auc': None, 'logloss': None,
                'brier': float(brier_score_loss(y_true, p_pred)),
                'acc': float(accuracy_score(y_true, (p_pred > 0.5).astype(int))),
                'pos_rate': float(y_true.mean())}
    return {
        'n': int(len(y_true)),
        'auc':     round(float(roc_auc_score(y_true, p_pred)), 4),
        'logloss': round(float(log_loss(y_true, p_pred, labels=[0, 1])), 4),
        'brier':   round(float(brier_score_loss(y_true, p_pred)), 4),
        'acc':     round(float(accuracy_score(y_true, (p_pred > 0.5).astype(int))), 4),
        'pos_rate': round(float(y_true.mean()), 4),
    }


def main():
    if not all(p.exists() for p in [PRED_PATH, MAP_PATH, CLUSTER_PATH]):
        print('⚠ Нужны val_predictions.csv, map_features.csv, map_clusters.csv')
        sys.exit(1)

    print(f'Loading {PRED_PATH} …')
    df = pd.read_csv(PRED_PATH)
    df['episode_id'] = df['episode_id'].astype(str)

    clusters = pd.read_csv(CLUSTER_PATH)[['episode_id', 'map_type',
                                            'n_planets', 'n_orbital',
                                            'size_bucket', 'has_orbital']]
    clusters['episode_id'] = clusters['episode_id'].astype(str)
    print(f'  val predictions: {len(df):,}  карт: {len(clusters)}')

    df = df.merge(clusters, on='episode_id', how='left')
    print(f'  merged: {df["map_type"].notna().sum()} строк с типом карты')

    # ── AUC value_v2 внутри каждой страты ──────────────────────────
    print('\n── AUC value_v2 по типу карты (1v1 + FFA) ────────────')
    strata = sorted(df['map_type'].dropna().unique())
    overall = metrics_for(df['label'].values, df['p_win'].values)
    print(f'  overall: AUC={overall["auc"]}  Brier={overall["brier"]}  '
          f'Acc={overall["acc"]}  n={overall["n"]}')

    per_stratum = {}
    for s in strata:
        sub = df[df['map_type'] == s]
        if len(sub) < 100:
            continue
        m = metrics_for(sub['label'].values, sub['p_win'].values)
        per_stratum[s] = m
        print(f'  {s:<18}  AUC={m["auc"]}  Brier={m["brier"]}  '
              f'Acc={m["acc"]}  pos_rate={m["pos_rate"]}  n={m["n"]}')

    # ── Стратификация по n_players × size_bucket × has_orbital ─────
    print('\n── AUC по тонкой стратификации (1v1 only) ─────────────')
    df_1v1 = df[df['n_players'] == 2].copy()
    for s in strata:
        sub = df_1v1[df_1v1['map_type'] == s]
        if len(sub) < 100:
            continue
        m = metrics_for(sub['label'].values, sub['p_win'].values)
        print(f'  1v1 / {s:<14}  AUC={m["auc"]}  Brier={m["brier"]}  '
              f'pos_rate={m["pos_rate"]}  n={m["n"]}')

    # ── AUC vs phase (внутри страт): где модель решает рано/поздно ─
    print('\n── AUC vs phase quartile, по стратам ──────────────────')
    phase_bins = [-0.01, 0.25, 0.5, 0.75, 1.01]
    phase_names = ['q1', 'q2', 'q3', 'q4']

    rows = []
    for s in strata:
        sub = df_1v1[df_1v1['map_type'] == s]
        if len(sub) < 200:
            continue
        for ql, qh, qn in zip(phase_bins[:-1], phase_bins[1:], phase_names):
            ss = sub[(sub['phase'] >= ql) & (sub['phase'] < qh)]
            if len(ss) < 50 or ss['label'].nunique() < 2:
                continue
            m = metrics_for(ss['label'].values, ss['p_win'].values)
            rows.append({
                'stratum': s, 'quartile': qn,
                'n': m['n'], 'auc': m['auc'],
                'pos_rate': m['pos_rate'],
            })
    auc_by_q = pd.DataFrame(rows)
    if len(auc_by_q):
        pivot = auc_by_q.pivot_table(index='stratum', columns='quartile',
                                       values='auc', aggfunc='first')
        print(pivot.round(3).to_string())

    # ── Where the model is weakest? ─────────────────────────────────
    print('\n── Где value_v2 ошибается чаще всего (в 1v1) ───────────')
    # Top-misses: высокая уверенность но ошибка
    miss = df_1v1.copy()
    miss['confidence'] = np.where(miss['p_win'] > 0.5, miss['p_win'], 1 - miss['p_win'])
    miss['correct'] = ((miss['p_win'] > 0.5) == (miss['label'] == 1)).astype(int)
    miss['high_conf_wrong'] = ((miss['confidence'] > 0.85) & (miss['correct'] == 0)).astype(int)
    print('\n  Доля high_confidence_wrong (>0.85 уверенность, но ошибся):')
    print(miss.groupby('map_type')['high_conf_wrong'].agg(['sum', 'count', 'mean']).round(4).to_string())

    # ── Markdown отчёт ──────────────────────────────────────────────
    md = []
    md.append('# Аудит value_v2 по типам карт\n\n')
    md.append('## Overall vs по стратам\n\n')
    md.append('| Stratum | n | AUC | Brier | Acc | pos_rate |\n|---|---|---|---|---|---|\n')
    md.append(f"| OVERALL (all) | {overall['n']} | {overall['auc']} | {overall['brier']} | "
              f"{overall['acc']} | {overall['pos_rate']} |\n")
    for s, m in per_stratum.items():
        md.append(f"| {s} | {m['n']} | {m['auc']} | {m['brier']} | {m['acc']} | {m['pos_rate']} |\n")

    md.append('\n## AUC vs phase, по стратам (1v1)\n\n')
    if len(auc_by_q):
        md.append(pivot.round(3).to_markdown())
        md.append('\n\n')
        md.append('Где модель плохо предсказывает в q1 (раннее), значит структурных\n')
        md.append('признаков (включая макро) недостаточно — нужен либо специфичный для\n')
        md.append('этой страты тип фич, либо ситуация реально competitive.\n\n')

    md.append('## Высокоуверенные ошибки (1v1)\n\n')
    md.append('Если в страте доля `high_conf_wrong` > 1% — модель регулярно ошибается с уверенностью.\n')
    md.append('Эти ошибки и есть подсказка, что универсальная функция упускает что-то специфичное.\n\n')
    md.append(miss.groupby('map_type')['high_conf_wrong']
              .agg(['sum', 'count', 'mean']).round(4).to_markdown())

    (REPORT_DIR / 'value_v2_per_stratum.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/value_v2_per_stratum.md')

    # ── График: AUC vs phase по стратам ──────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if len(auc_by_q):
            fig, ax = plt.subplots(figsize=(10, 6))
            for s in pivot.index:
                vals = pivot.loc[s].values
                ax.plot([1, 2, 3, 4], vals, 'o-', label=s, linewidth=2)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(['q1 (0-25%)', 'q2 (25-50%)', 'q3 (50-75%)', 'q4 (75-100%)'])
            ax.set_ylabel('AUC')
            ax.set_xlabel('phase quartile')
            ax.set_title('AUC value_v2 по фазам игры и типам карт (1v1 only)')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            ax.set_ylim(0.4, 1.0)
            plt.tight_layout()
            plt.savefig(REPORT_DIR / 'value_v2_per_stratum.png', dpi=120)
            plt.close()
            print(f'✓ График: {REPORT_DIR}/value_v2_per_stratum.png')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
