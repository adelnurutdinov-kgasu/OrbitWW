#!/usr/bin/env python3
"""
06_delta_features.py — скалярные дельты в pre-resolution окне.

ЛОГИКА
──────
Для каждой 1v1 партии у нас уже есть t_soft (из 05). Окно изучения:
    window = [t_soft - 0.2*n_steps, t_soft]

В этом окне мы хотим понять: какие ИЗМЕНЕНИЯ макро-фич между началом
и концом окна разделяют победителей и проигравших? То есть delta-features:
    delta_X = X(t_soft) - X(t_window_start)

Если даже эти **скалярные** дельты разделяют исходы — есть смысл идти
в реальную геометрию (06+). Если не разделяют — нужно переосмыслить
прежде чем тратиться на тяжёлый reload raw JSON.

Это sanity test и одновременно baseline для будущей геометрии.

ВХОД
────
  data/processed/steps.csv
  rebuild/data/episodes_meta.csv
  rebuild/data/resolution_moments.csv  (из 05)

ВЫХОД
─────
  rebuild/data/delta_features.parquet      — фичи на каждую партию
  rebuild/reports/delta_features.md         — отчёт
  rebuild/reports/delta_features_*.png      — графики
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
OL_DIR     = REBUILD.parent
PROC_DIR   = OL_DIR / "data" / "processed"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STEPS_CSV  = PROC_DIR / "steps.csv"
META_PATH  = DATA_DIR / "episodes_meta.csv"
MOM_PATH   = DATA_DIR / "resolution_moments.csv"
OUT_PATH   = DATA_DIR / "delta_features.csv"

# Фичи которые будем дельтировать.
# Все симметричные / относительные.
TRACK_FEATURES = [
    'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
    'ship_gap', 'prod_gap',
    'n_own_planets', 'n_neutral_planets', 'n_enemy_planets',
    'own_ships_total', 'own_prod_total',
    'mean_own_ships', 'max_own_ships', 'std_own_ships',
    'own_fleets_count', 'own_fleets_ships',
    'enemy_fleets_count', 'enemy_fleets_ships', 'fleet_balance',
    'own_under_threat_count', 'own_under_threat_ships',
    'enemy_contested_count', 'neutral_contested_count',
    'incoming_enemy_to_own', 'incoming_own_reinforce', 'net_incoming_own',
    'closest_threat_dist',
]


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test, без scipy."""
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


def load():
    print('Loading steps.csv …')
    cols = ['episode_id', 'player_idx', 'step', 'phase', 'reward'] + TRACK_FEATURES
    df = pd.read_csv(STEPS_CSV, usecols=lambda c: c in cols)
    df['episode_id'] = df['episode_id'].astype(str)
    df['label'] = (df['reward'] > 0).astype(int)

    moments = pd.read_csv(MOM_PATH)
    moments['episode_id'] = moments['episode_id'].astype(str)
    print(f'  steps: {len(df):,} строк, moments: {len(moments)} партий')
    return df, moments


def extract_deltas(df: pd.DataFrame, moments: pd.DataFrame) -> pd.DataFrame:
    """Для каждой (episode, player) считает Δfeature = feature(t_soft) - feature(t_window_start)."""
    by_key = {(eid, pid): g.sort_values('step') for (eid, pid), g
              in df.groupby(['episode_id', 'player_idx'])}

    feats = [f for f in TRACK_FEATURES if f in df.columns]
    print(f'\n  отслеживаем {len(feats)} фич')

    rows = []
    skipped = 0
    for _, m in moments.iterrows():
        key = (m['episode_id'], int(m['player_idx']))
        if key not in by_key:
            skipped += 1
            continue
        g = by_key[key]

        # Берём строки в окне [t_window_start, t_soft]
        t_start = int(m['t_window_start'])
        t_end   = int(m['t_soft'])
        win_rows = g[(g['step'] >= t_start) & (g['step'] <= t_end)]
        if len(win_rows) < 2:
            skipped += 1
            continue

        first = win_rows.iloc[0]
        last  = win_rows.iloc[-1]

        rec = {
            'episode_id':     m['episode_id'],
            'player_idx':     int(m['player_idx']),
            'final_label':    int(m['final_label']),
            'window_size':    int(m['window_size']),
            't_soft':         int(m['t_soft']),
            'phase_t_soft':   float(m['phase_t_soft']),
        }
        for f in feats:
            v0 = float(first[f]) if not pd.isna(first[f]) else 0.0
            v1 = float(last[f])  if not pd.isna(last[f])  else 0.0
            rec[f'delta_{f}'] = v1 - v0
            # также абсолютные начальные значения - они важны как контекст
            rec[f'start_{f}'] = v0
        rows.append(rec)

    print(f'  skipped: {skipped}, итого записей: {len(rows)}')
    return pd.DataFrame(rows)


def analyze_separation(deltas: pd.DataFrame) -> pd.DataFrame:
    """Для каждой delta-фичи: t-test winners vs losers."""
    feats = [c for c in deltas.columns if c.startswith('delta_')]
    win = deltas[deltas['final_label'] == 1]
    los = deltas[deltas['final_label'] == 0]
    print(f'\n  winners: {len(win)}  losers: {len(los)}')

    rows = []
    for f in feats:
        t, p = welch_t(win[f].values, los[f].values)
        rows.append({
            'feature':       f,
            'mean_winners':  round(float(win[f].mean()), 4),
            'mean_losers':   round(float(los[f].mean()), 4),
            'std_winners':   round(float(win[f].std()), 4),
            'std_losers':    round(float(los[f].std()), 4),
            'gap':           round(float(win[f].mean() - los[f].mean()), 4),
            't':             round(t, 3),
            'p':             p,
            'log10_p':       round(np.log10(max(p, 1e-300)), 2),
            # эффект-размер (Cohen's d)
            'cohens_d':      round(float(
                (win[f].mean() - los[f].mean()) /
                max(0.5*(win[f].std() + los[f].std()), 1e-9)
            ), 3),
        })
    res = pd.DataFrame(rows).sort_values('p')
    return res


def classifier_test(deltas: pd.DataFrame) -> dict:
    """Логистическая регрессия на delta-фичах: разделяют ли они исход?"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        print('  sklearn недоступен — classifier_test пропущен')
        return {}

    feat_cols = [c for c in deltas.columns
                 if c.startswith('delta_') or c.startswith('start_')]
    delta_only = [c for c in feat_cols if c.startswith('delta_')]
    start_only = [c for c in feat_cols if c.startswith('start_')]

    y = deltas['final_label'].values
    groups = deltas['episode_id'].values  # split by episode

    results = {}
    for name, cols in [('delta_only', delta_only),
                        ('start_only', start_only),
                        ('delta+start', feat_cols)]:
        X = deltas[cols].fillna(0).replace([np.inf, -np.inf], 0).values

        gkf = GroupKFold(n_splits=5)
        aucs, accs = [], []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = LogisticRegression(max_iter=300, C=1.0)
            model.fit(X_tr, y_tr)
            p_val = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, p_val))
            accs.append(accuracy_score(y_val, (p_val > 0.5).astype(int)))
        results[name] = {
            'auc_mean': round(float(np.mean(aucs)), 4),
            'auc_std':  round(float(np.std(aucs)), 4),
            'acc_mean': round(float(np.mean(accs)), 4),
            'n_features': len(cols),
        }
        print(f'  {name:<14} ({len(cols):>3} фич): AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    return results


def plot_top(separation: pd.DataFrame, deltas: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Топ-12 по Cohen's d
    top = separation.assign(abs_d=separation['cohens_d'].abs()).sort_values('abs_d', ascending=False).head(12)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i, (_, r) in enumerate(top.iterrows()):
        ax = axes[i // 4, i % 4]
        win = deltas[deltas['final_label'] == 1][r['feature']].values
        los = deltas[deltas['final_label'] == 0][r['feature']].values
        # Совместный bin range
        lo, hi = np.percentile(np.concatenate([win, los]), [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(win, bins=bins, alpha=0.5, label=f'winners ({len(win)})', color='green')
        ax.hist(los, bins=bins, alpha=0.5, label=f'losers ({len(los)})', color='red')
        ax.axvline(win.mean(), c='green', ls='--', alpha=0.7)
        ax.axvline(los.mean(), c='red',   ls='--', alpha=0.7)
        ax.set_title(f"{r['feature']}\nd={r['cohens_d']:+.2f}  log10_p={r['log10_p']:.1f}",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle('Топ-12 delta-фич по эффект-размеру (Cohen\'s d)')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'delta_features_top12.png', dpi=120)
    plt.close()


def write_report(separation: pd.DataFrame, clf_results: dict, deltas: pd.DataFrame):
    md = []
    md.append('# Delta-features в pre-resolution окне\n\n')
    md.append(f'**Партий-игроков**: {len(deltas)} (winners={deltas["final_label"].sum()}, '
              f'losers={(deltas["final_label"]==0).sum()})\n\n')
    md.append('## Главная цифра\n\n')
    md.append('| Модель | n_features | AUC mean | AUC std | Acc mean |\n')
    md.append('|---|---|---|---|---|\n')
    for name, r in clf_results.items():
        md.append(f"| {name} | {r['n_features']} | {r['auc_mean']} | "
                  f"{r['auc_std']} | {r['acc_mean']} |\n")
    md.append('\nПорог осмысленности: AUC ≥ 0.65 для delta_only — значит изменения в окне разделяют исходы.\n')
    md.append('Если delta+start = start_only — изменения ничего не добавляют, только текущая позиция.\n\n')

    md.append('## Топ-20 delta-фич по эффект-размеру\n\n')
    top = separation.assign(abs_d=separation['cohens_d'].abs()).sort_values('abs_d', ascending=False).head(20)
    md.append('| feature | gap | d (Cohen) | log10_p |\n|---|---|---|---|\n')
    for _, r in top.iterrows():
        md.append(f"| {r['feature']} | {r['gap']} | {r['cohens_d']} | {r['log10_p']} |\n")

    md.append('\n## Графики\n\n')
    md.append('- `delta_features_top12.png` — гистограммы winners vs losers по топ-12 фич\n')

    (REPORT_DIR / 'delta_features.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/delta_features.md')


def main():
    if not MOM_PATH.exists():
        print(f'⚠ Нет {MOM_PATH} — запустите сначала 05_resolution_moments.py')
        sys.exit(1)

    df, moments = load()
    deltas = extract_deltas(df, moments)
    if len(deltas) == 0:
        print('⚠ нет валидных дельт')
        sys.exit(1)

    deltas.to_csv(OUT_PATH, index=False)
    print(f'\n✓ Сохранено: {OUT_PATH}')

    print('\n── Сравнение winners vs losers по дельтам ────────────')
    sep = analyze_separation(deltas)
    print(sep.head(20)[['feature', 'mean_winners', 'mean_losers',
                         'gap', 'cohens_d', 'log10_p']].to_string(index=False))

    print('\n── Логистическая регрессия (5-fold GroupKFold by episode) ──')
    clf = classifier_test(deltas)

    plot_top(sep, deltas)
    write_report(sep, clf, deltas)


if __name__ == '__main__':
    main()
