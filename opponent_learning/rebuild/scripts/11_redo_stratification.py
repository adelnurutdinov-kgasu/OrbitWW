#!/usr/bin/env python3
"""
11_redo_stratification.py — заново 08+09+10 со стратификацией ТОЛЬКО по размеру карты.

ИЗМЕНЕНИЯ vs 08/09/10
─────────────────────
Старая ось `map_type = size × has_orbital` была артефактом неверной
классификации orbital. После 07b у нас правильные данные:
все карты содержат и движущиеся, и стационарные планеты.
Используем только корректную ось — `size_bucket` (small/medium/large).

ЧТО ДЕЛАЕМ
──────────
A. Stratified delta-features  — Cohen's d по size_bucket
B. Stratified value-AUC       — AUC value_v2 по size_bucket
C. Stratified safe zones      — пороги prod_advantage по size_bucket

ВХОД
────
  rebuild/data/delta_features.csv     (из 06)
  rebuild/data/val_predictions.csv    (из 03)
  rebuild/data/map_features_v2.csv    (из 07b)
  rebuild/data/episodes_meta.csv      (из 01)
  data/processed/steps.csv

ВЫХОД
─────
  rebuild/reports/redo_stratification.md
  rebuild/reports/redo_*.png
  rebuild/data/map_clusters_v2.csv

ЛЁГКИЙ скрипт (~30 сек, без xgboost / sklearn).
"""

from __future__ import annotations

import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
PROC_DIR   = OL_DIR / "data" / "processed"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_PATH = DATA_DIR / "delta_features.csv"
PRED_PATH  = DATA_DIR / "val_predictions.csv"
MAP_PATH   = DATA_DIR / "map_features_v2.csv"
META_PATH  = DATA_DIR / "episodes_meta.csv"
STEPS_CSV  = PROC_DIR / "steps.csv"


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2: return 0.0, 1.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va + vb == 0: return 0.0, 1.0
    se = sqrt(va/len(a) + vb/len(b))
    t = (a.mean() - b.mean()) / se
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2: return 0.0
    sd = 0.5 * (a.std(ddof=1) + b.std(ddof=1))
    return float((a.mean() - b.mean()) / sd) if sd > 1e-9 else 0.0


def auc_simple(y_true, p_pred):
    """ROC AUC без sklearn (через сортировку)."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(p_pred)
    y = y_true[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Mann-Whitney style: count pairs where positive ranked above negative
    cum_neg = 0
    tp_above_fp = 0
    for yi in y:
        if yi == 0:
            cum_neg += 1
        else:
            tp_above_fp += cum_neg
    return tp_above_fp / (n_pos * n_neg)


# ════════════════════════════════════════════════════════════════════════
# A. Stratified delta-features
# ════════════════════════════════════════════════════════════════════════

def part_A_deltas(deltas, maps):
    print('\n' + '═'*70)
    print('A. Stratified DELTA-FEATURES по size_bucket')
    print('═'*70)
    merged = deltas.merge(maps[['episode_id', 'size_bucket']], on='episode_id', how='left')
    delta_cols = [c for c in deltas.columns if c.startswith('delta_')]
    print(f'\n  Размеры страт:')
    print(merged.groupby('size_bucket').size().to_string())

    # Overall
    win = merged[merged['final_label'] == 1]
    los = merged[merged['final_label'] == 0]
    overall = []
    for f in delta_cols:
        d = cohens_d(win[f].values, los[f].values)
        overall.append({'feature': f, 'd_overall': round(d, 3)})
    over_df = pd.DataFrame(overall)

    # Per stratum
    rows = []
    for s in sorted(merged['size_bucket'].dropna().unique()):
        sub = merged[merged['size_bucket'] == s]
        if len(sub) < 20: continue
        w, l = sub[sub['final_label']==1], sub[sub['final_label']==0]
        if len(w) < 5 or len(l) < 5: continue
        for f in delta_cols:
            d = cohens_d(w[f].values, l[f].values)
            t, p = welch_t(w[f].values, l[f].values)
            rows.append({'stratum': s, 'n_win': len(w), 'n_los': len(l),
                         'feature': f, 'cohens_d': round(d, 3),
                         'log10_p': round(np.log10(max(p, 1e-300)), 2)})
    per_strat = pd.DataFrame(rows)

    pivot = per_strat.pivot_table(index='feature', columns='stratum',
                                    values='cohens_d', aggfunc='first')
    cmp_df = pivot.merge(over_df.set_index('feature'),
                          left_index=True, right_index=True, how='left')
    cmp_df['max_abs_d'] = pivot.abs().max(axis=1).round(3)
    cmp_df['amplification'] = (cmp_df['max_abs_d'] - cmp_df['d_overall'].abs()).round(3)
    cmp_df = cmp_df.sort_values('amplification', ascending=False)

    print('\n  Топ-15 по усилению (max |d| в страте) - (|d| общий):')
    print(cmp_df.head(15).round(3).to_string())
    return cmp_df, per_strat


# ════════════════════════════════════════════════════════════════════════
# B. Stratified value AUC
# ════════════════════════════════════════════════════════════════════════

def part_B_value(preds, maps):
    print('\n' + '═'*70)
    print('B. Stratified VALUE-AUC по size_bucket')
    print('═'*70)
    df = preds.merge(maps[['episode_id', 'size_bucket']], on='episode_id', how='left')
    rows = []

    # Overall
    auc_all = auc_simple(df['label'].values, df['p_win'].values)
    print(f'\n  overall AUC = {auc_all:.4f}  n={len(df)}')

    for s in sorted(df['size_bucket'].dropna().unique()):
        sub = df[df['size_bucket'] == s]
        if len(sub) < 100: continue
        auc = auc_simple(sub['label'].values, sub['p_win'].values)
        acc = float(((sub['p_win'] > 0.5) == (sub['label'] == 1)).mean())
        pos_rate = float(sub['label'].mean())
        print(f'  {s:<8}  AUC={auc:.4f}  Acc={acc:.4f}  '
              f'pos_rate={pos_rate:.3f}  n={len(sub)}')
        rows.append({'stratum': s, 'auc': round(auc, 4), 'acc': round(acc, 4),
                     'pos_rate': round(pos_rate, 4), 'n': len(sub)})

    # 1v1 only
    print('\n  1v1 only:')
    df_1v1 = df[df['n_players'] == 2]
    rows_1v1 = []
    for s in sorted(df_1v1['size_bucket'].dropna().unique()):
        sub = df_1v1[df_1v1['size_bucket'] == s]
        if len(sub) < 50: continue
        auc = auc_simple(sub['label'].values, sub['p_win'].values)
        print(f'    {s:<8}  AUC={auc:.4f}  n={len(sub)}')
        rows_1v1.append({'stratum': s, 'auc': round(auc, 4), 'n': len(sub)})

    # AUC vs phase x stratum (1v1)
    print('\n  AUC vs phase × size_bucket (1v1):')
    phase_bins = [-0.01, 0.25, 0.5, 0.75, 1.01]
    phase_names = ['q1', 'q2', 'q3', 'q4']
    auc_phase = {}
    for s in sorted(df_1v1['size_bucket'].dropna().unique()):
        sub = df_1v1[df_1v1['size_bucket'] == s]
        if len(sub) < 200: continue
        row = {}
        for ql, qh, qn in zip(phase_bins[:-1], phase_bins[1:], phase_names):
            ss = sub[(sub['phase'] >= ql) & (sub['phase'] < qh)]
            if len(ss) < 50 or ss['label'].nunique() < 2:
                row[qn] = None
            else:
                row[qn] = round(auc_simple(ss['label'].values, ss['p_win'].values), 3)
        auc_phase[s] = row
        print(f'    {s:<8}  q1={row.get("q1")}  q2={row.get("q2")}  '
              f'q3={row.get("q3")}  q4={row.get("q4")}')

    return rows, rows_1v1, auc_phase


# ════════════════════════════════════════════════════════════════════════
# C. Stratified safe zones
# ════════════════════════════════════════════════════════════════════════

def find_safe_zones(df, param, n_bins=50, min_n=30, eps=0.10):
    bins = np.linspace(df[param].min(), df[param].max() + 1e-9, n_bins + 1)
    safe_low, safe_high = None, None
    valid_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sub = df[(df[param] >= lo) & (df[param] < hi)]
        if len(sub) >= min_n:
            valid_rows.append((lo, hi, sub['label'].mean()))
    if not valid_rows: return None, None
    for lo, hi, wr in valid_rows:
        if wr <= eps: safe_low = hi
        else: break
    for lo, hi, wr in valid_rows[::-1]:
        if wr >= 1 - eps: safe_high = lo
        else: break
    return safe_low, safe_high


def part_C_zones(steps, maps):
    print('\n' + '═'*70)
    print('C. Stratified SAFE ZONES по size_bucket')
    print('═'*70)
    cols = ['episode_id', 'player_idx', 'step', 'phase', 'reward',
            'own_prod_total', 'prod_gap']
    df = pd.read_csv(STEPS_CSV, usecols=cols)
    df['episode_id'] = df['episode_id'].astype(str)
    df['label'] = (df['reward'] > 0).astype(int)

    meta = pd.read_csv(META_PATH)[['episode_id', 'n_players']]
    meta['episode_id'] = meta['episode_id'].astype(str)
    df = df.merge(meta, on='episode_id', how='left')
    df = df[df['n_players'] == 2].copy()

    df['enemy_prod_total'] = (df['own_prod_total'] - df['prod_gap']).clip(lower=0)
    denom = (df['own_prod_total'] + df['enemy_prod_total']).clip(lower=1)
    df['prod_advantage'] = df['own_prod_total'] / denom

    df = df.merge(maps[['episode_id', 'size_bucket']], on='episode_id', how='left')

    print(f'\n  Размеры страт (1v1):')
    print(df.groupby('size_bucket').size().to_string())

    print('\n  Safe zones по size_bucket:')
    print('  size      n     soft_low   soft_high   hard_low   hard_high')
    print('  ' + '-'*70)
    rows = []
    for s in sorted(df['size_bucket'].dropna().unique()):
        sub = df[df['size_bucket'] == s]
        if len(sub) < 1000:
            print(f'  {s:<8}  {len(sub):>5}  (мало данных)')
            continue
        sl_soft, sh_soft = find_safe_zones(sub, 'prod_advantage', eps=0.10)
        sl_hard, sh_hard = find_safe_zones(sub, 'prod_advantage', eps=0.02)
        print(f'  {s:<8}  {len(sub):>5}    {sl_soft}      {sh_soft}     '
              f'{sl_hard}      {sh_hard}')
        rows.append({
            'stratum': s, 'n': len(sub),
            'soft_low': sl_soft, 'soft_high': sh_soft,
            'hard_low': sl_hard, 'hard_high': sh_hard,
        })
    # Сравнение с global
    sl_g_soft, sh_g_soft = find_safe_zones(df, 'prod_advantage', eps=0.10)
    sl_g_hard, sh_g_hard = find_safe_zones(df, 'prod_advantage', eps=0.02)
    print(f'\n  GLOBAL (all 1v1):  soft=[{sl_g_soft}, {sh_g_soft}]  '
          f'hard=[{sl_g_hard}, {sh_g_hard}]')
    return rows, {'soft': (sl_g_soft, sh_g_soft), 'hard': (sl_g_hard, sh_g_hard)}


# ════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════

def main():
    for p in [DELTA_PATH, PRED_PATH, MAP_PATH, META_PATH, STEPS_CSV]:
        if not p.exists():
            print(f'⚠ нет {p}')
            sys.exit(1)

    print('Loading data...')
    deltas = pd.read_csv(DELTA_PATH)
    preds  = pd.read_csv(PRED_PATH)
    maps   = pd.read_csv(MAP_PATH)
    for d in [deltas, preds, maps]:
        d['episode_id'] = d['episode_id'].astype(str)
    print(f'  deltas: {len(deltas)}  preds: {len(preds)}  maps: {len(maps)}')

    # Save clusters v2
    maps[['episode_id', 'size_bucket', 'n_planets', 'orbital_share', 'omega']].to_csv(
        DATA_DIR / 'map_clusters_v2.csv', index=False)

    # A
    cmp_df, per_strat_deltas = part_A_deltas(deltas, maps)

    # B
    auc_overall, auc_1v1, auc_phase = part_B_value(preds, maps)

    # C
    zones, global_zones = part_C_zones(None, maps)

    # ── Отчёт ────────────────────────────────────────────────────────
    md = []
    md.append('# Redo Stratification (по size_bucket only)\n\n')
    md.append('Перезапуск 08/09/10 после корректировки 07b: ось orbital/static\n')
    md.append('была артефактом. Используем только корректную ось — `size_bucket`.\n\n')

    md.append('## A. Delta-features: усиление при стратификации\n\n')
    md.append('| feature | d_overall | small | medium | large | max_abs_d | amplification |\n')
    md.append('|---|---|---|---|---|---|---|\n')
    for f, r in cmp_df.head(20).iterrows():
        md.append(f"| {f} | {r['d_overall']} | "
                  f"{r.get('small', '—')} | {r.get('medium', '—')} | "
                  f"{r.get('large', '—')} | {r['max_abs_d']} | "
                  f"{r['amplification']} |\n")

    md.append('\n## B. Value AUC по size_bucket\n\n')
    md.append('### Все партии (1v1+FFA)\n\n')
    md.append('| stratum | n | AUC | Acc | pos_rate |\n|---|---|---|---|---|\n')
    for r in auc_overall:
        md.append(f"| {r['stratum']} | {r['n']} | {r['auc']} | "
                  f"{r['acc']} | {r['pos_rate']} |\n")
    md.append('\n### 1v1 only\n\n')
    md.append('| stratum | n | AUC |\n|---|---|---|\n')
    for r in auc_1v1:
        md.append(f"| {r['stratum']} | {r['n']} | {r['auc']} |\n")

    md.append('\n### AUC vs phase quartile × size_bucket (1v1)\n\n')
    md.append('| stratum | q1 | q2 | q3 | q4 |\n|---|---|---|---|---|\n')
    for s, q in auc_phase.items():
        md.append(f"| {s} | {q.get('q1')} | {q.get('q2')} | "
                  f"{q.get('q3')} | {q.get('q4')} |\n")

    md.append('\n## C. Safe zones по size_bucket (1v1)\n\n')
    md.append('| stratum | n | soft_low | soft_high | hard_low | hard_high |\n')
    md.append('|---|---|---|---|---|---|\n')
    for r in zones:
        md.append(f"| {r['stratum']} | {r['n']} | {r['soft_low']} | {r['soft_high']} | "
                  f"{r['hard_low']} | {r['hard_high']} |\n")
    md.append(f"\nGLOBAL: soft={global_zones['soft']}  hard={global_zones['hard']}\n")

    (REPORT_DIR / 'redo_stratification.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/redo_stratification.md')

    # Heatmap delta-features
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        top10 = cmp_df.head(10).index.tolist()
        heat = per_strat_deltas[per_strat_deltas['feature'].isin(top10)].pivot_table(
            index='feature', columns='stratum', values='cohens_d', aggfunc='first'
        ).loc[top10]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heat.values, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns, fontsize=10)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index, fontsize=9)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                v = heat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                            fontsize=8, color='white' if abs(v) > 1.5 else 'black')
        plt.colorbar(im, ax=ax, label="Cohen's d")
        ax.set_title('Cohen\'s d по топ-10 фичам × size_bucket (corrected)')
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'redo_heatmap.png', dpi=120)
        plt.close()
        print(f'✓ График: {REPORT_DIR}/redo_heatmap.png')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
