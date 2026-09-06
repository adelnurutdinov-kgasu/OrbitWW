#!/usr/bin/env python3
"""
03_phase_curve.py — анатомия фазового перехода.

Что измеряем (по результатам value_v2)
──────────────────────────────────────
Мы уже знаем что AUC растёт от 0.69 (q1) до 0.99 (q4). Это значит игра
становится решённой по ходу партии. Этот скрипт раскапывает СТРУКТУРУ
этого перехода с разрешением 10-20 децилей и проверяет три гипотезы:

  H1: Переход быстрый — AUC скачкообразно растёт в узкой полосе phase.
      Если так — есть характерный момент решения.

  H2: V(t) для победителей и проигравших ВНУТРИ партии расходится
      нелинейно (бифуркация после общего среднего ~0.5).

  H3: Дисперсия V среди партий в данной фазе peak'ует в момент перехода
      (critical slowing down / увеличение восприимчивости).

ВХОД
────
  data/processed/steps.csv             — per-step датасет
  rebuild/data/episodes_meta.csv       — n_players по эпизодам (01)
  rebuild/models/value_function_v2.json — обученная модель (02)

ВЫХОД
─────
  rebuild/data/val_predictions.csv     — predictions на val split (для повтораного анализа)
  rebuild/reports/phase_curve.md       — отчёт с числами
  rebuild/reports/phase_curve_*.png    — графики (если matplotlib доступен)

СРЕДНИЙ скрипт (~30 секунд predict + сохранение CSV; графики ещё ~10с).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE        = Path(__file__).resolve().parent
REBUILD     = HERE.parent
OL_DIR      = REBUILD.parent
PROC_DIR    = OL_DIR / "data" / "processed"
DATA_DIR    = REBUILD / "data"
MODEL_DIR   = REBUILD / "models"
REPORT_DIR  = REBUILD / "reports"
for d in (DATA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

STEPS_CSV   = PROC_DIR / "steps.csv"
META_PATH   = DATA_DIR / "episodes_meta.csv"
MODEL_PATH  = MODEL_DIR / "value_function_v2.json"
META_JSON   = MODEL_DIR / "value_function_v2_meta.json"
PRED_PATH   = DATA_DIR / "val_predictions.csv"

N_BINS      = 20  # phase deciles → twenties — лучше резолюция


# ── шаги анализа ──────────────────────────────────────────────────────────

def load_val_with_predictions(force_repredict: bool = False) -> pd.DataFrame:
    """Загружает val split, при необходимости пересчитывает predictions."""
    if PRED_PATH.exists() and not force_repredict:
        print(f'Found cached predictions: {PRED_PATH}')
        return pd.read_csv(PRED_PATH)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print('pip install xgboost --break-system-packages')
        sys.exit(1)

    print(f'Loading model: {MODEL_PATH}')
    meta = json.load(open(META_JSON))
    feats = meta['features']
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print(f'Loading {STEPS_CSV} …')
    df = pd.read_csv(STEPS_CSV)
    df['label'] = (df['reward'] > 0).astype(int)
    df['episode_id'] = df['episode_id'].astype(str)

    # Merge с meta (для n_players)
    if META_PATH.exists():
        meta_df = pd.read_csv(META_PATH)[['episode_id', 'n_players']]
        meta_df['episode_id'] = meta_df['episode_id'].astype(str)
        df = df.merge(meta_df, on='episode_id', how='left')

    # Тот же split_by_episode, что в 02_value_function_v2.py (seed=42)
    eids = df['episode_id'].unique()
    rng  = np.random.default_rng(42)
    rng.shuffle(eids)
    n_val = max(1, int(len(eids) * 0.15))
    val_set = set(eids[:n_val])
    is_val = df['episode_id'].isin(val_set)
    df_val = df[is_val].copy()
    print(f'  val: {len(df_val):,} строк, {df_val["episode_id"].nunique()} эпизодов')

    # Predict
    X = (df_val[feats].fillna(0).replace([np.inf, -np.inf], 0)
                       .values.astype(np.float32))
    print('  predicting …')
    p_win = model.predict_proba(X)[:, 1]

    out = df_val[['episode_id', 'player_idx', 'step', 'phase',
                  'label', 'reward', 'n_players']].copy()
    out['p_win'] = p_win
    out.to_csv(PRED_PATH, index=False)
    print(f'  saved: {PRED_PATH}')
    return out


# ── анализ #1: AUC по N бинам phase ──────────────────────────────────────

def auc_curve(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """AUC в каждом фазном бине; возвращает таблицу."""
    from sklearn.metrics import roc_auc_score, brier_score_loss

    rows = []
    edges = np.linspace(0, 1.001, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (df['phase'] >= lo) & (df['phase'] < hi)
        sub = df[mask]
        if len(sub) < 50 or sub['label'].nunique() < 2:
            rows.append({'bin': i, 'lo': lo, 'hi': hi, 'n': len(sub),
                          'auc': None, 'pos_rate': sub['label'].mean()
                          if len(sub) else None,
                          'brier': None})
            continue
        auc = roc_auc_score(sub['label'].values, sub['p_win'].values)
        bri = brier_score_loss(sub['label'].values, sub['p_win'].values)
        rows.append({'bin': i, 'lo': round(lo, 3), 'hi': round(hi, 3),
                      'n': len(sub),
                      'auc': round(float(auc), 4),
                      'pos_rate': round(float(sub['label'].mean()), 4),
                      'brier': round(float(bri), 4)})
    return pd.DataFrame(rows)


# ── анализ #2: V(t) для победителей и проигравших ─────────────────────────

def mean_v_by_outcome(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Mean p_win в каждом phase-бине, отдельно для label=1 (победитель) и 0."""
    rows = []
    edges = np.linspace(0, 1.001, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (df['phase'] >= lo) & (df['phase'] < hi)
        sub = df[mask]
        if len(sub) < 50:
            continue
        win = sub[sub['label'] == 1]
        los = sub[sub['label'] == 0]
        rows.append({
            'bin': i, 'lo': round(lo, 3), 'hi': round(hi, 3),
            'n_win': len(win), 'n_los': len(los),
            'mean_v_winners': round(float(win['p_win'].mean()), 4) if len(win) else None,
            'mean_v_losers':  round(float(los['p_win'].mean()), 4) if len(los) else None,
            'var_v_all':      round(float(sub['p_win'].var()), 5),
        })
    return pd.DataFrame(rows)


# ── анализ #3: distribution of argmax |dV/dt| по партиям ──────────────────

def transition_moments(df: pd.DataFrame) -> pd.DataFrame:
    """Для каждой (episode, player): найти phase где |dp_win/dphase| максимально."""
    out = []
    for (eid, p), g in df.groupby(['episode_id', 'player_idx']):
        g = g.sort_values('step')
        if len(g) < 10:
            continue
        v = g['p_win'].values
        # сглаживание простым moving average шириной 5
        if len(v) >= 5:
            kernel = np.ones(5) / 5
            v_sm = np.convolve(v, kernel, mode='same')
        else:
            v_sm = v
        dv = np.abs(np.diff(v_sm))
        if dv.size == 0:
            continue
        idx = int(np.argmax(dv))
        phase_at_max = float(g['phase'].iloc[idx])
        max_dv = float(dv[idx])
        out.append({
            'episode_id':   eid,
            'player_idx':   int(p),
            'n_steps':      int(len(g)),
            'phase_argmax': round(phase_at_max, 4),
            'max_dv':       round(max_dv, 4),
            'final_label':  int(g['label'].iloc[-1]),
        })
    return pd.DataFrame(out)


# ── анализ #4: bimodality of V at each phase bin ──────────────────────────

def bimodality_test(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Простой тест бимодальности: отношение дисперсии к bimodal-coefficient.
    Используем Sarle bimodality coefficient: (skew^2 + 1) / kurtosis.
    BC > 5/9 ≈ 0.555 → bimodal.
    Если scipy установлен — используем Hartigan dip test через diptest пакет
    (опционально), иначе bimodality coefficient."""
    from scipy.stats import skew, kurtosis

    rows = []
    edges = np.linspace(0, 1.001, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        sub = df[(df['phase'] >= lo) & (df['phase'] < hi)]['p_win'].values
        if len(sub) < 100:
            continue
        sk = skew(sub)
        ku = kurtosis(sub, fisher=False)  # not-fisher: normal=3
        bc = (sk ** 2 + 1) / max(ku, 1e-9)
        rows.append({
            'bin': i, 'lo': round(lo, 3), 'hi': round(hi, 3),
            'n': len(sub),
            'mean':  round(float(np.mean(sub)), 4),
            'std':   round(float(np.std(sub)), 4),
            'skew':  round(float(sk), 4),
            'kurt':  round(float(ku), 4),
            'bimodality_coef': round(float(bc), 4),
            'is_bimodal_naive': bc > 5/9,
        })
    return pd.DataFrame(rows)


# ── графики ──────────────────────────────────────────────────────────────

def plot_all(auc_df, vbo_df, tm_df, bm_df):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib не установлен — графики пропущены)')
        return

    # 1) AUC vs phase
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = auc_df.dropna(subset=['auc'])
    ax.plot((valid['lo'] + valid['hi']) / 2, valid['auc'], 'o-', label='AUC')
    ax.axhline(0.5, ls='--', c='gray', alpha=0.5, label='random')
    ax.set_xlabel('phase (нормированное время партии)')
    ax.set_ylabel('AUC')
    ax.set_title('AUC value function как функция фазы')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'phase_curve_auc.png', dpi=120)
    plt.close()

    # 2) V(t) winners vs losers
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = (vbo_df['lo'] + vbo_df['hi']) / 2
    ax.plot(centers, vbo_df['mean_v_winners'], 'o-', c='green', label='mean V(t) победителей')
    ax.plot(centers, vbo_df['mean_v_losers'],  'o-', c='red',   label='mean V(t) проигравших')
    ax.fill_between(centers, vbo_df['mean_v_losers'],
                    vbo_df['mean_v_winners'], alpha=0.15)
    ax.set_xlabel('phase')
    ax.set_ylabel('mean P(win)')
    ax.set_title('Бифуркация value function по исходу')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'phase_curve_bifurcation.png', dpi=120)
    plt.close()

    # 3) Variance V across replays vs phase
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, vbo_df['var_v_all'], 'o-', c='purple')
    ax.set_xlabel('phase')
    ax.set_ylabel('Variance of V across replays')
    ax.set_title('Дисперсия V — peak = critical slowing down')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'phase_curve_variance.png', dpi=120)
    plt.close()

    # 4) Гистограмма transition moments
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tm_df['phase_argmax'], bins=20, edgecolor='k', alpha=0.7)
    ax.set_xlabel('phase argmax |dV/dt|')
    ax.set_ylabel('число партий')
    ax.set_title('Распределение момента максимального скачка V')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'phase_curve_transitions.png', dpi=120)
    plt.close()

    # 5) Bimodality coefficient
    fig, ax = plt.subplots(figsize=(8, 5))
    centers_b = (bm_df['lo'] + bm_df['hi']) / 2
    ax.plot(centers_b, bm_df['bimodality_coef'], 'o-', c='orange')
    ax.axhline(5/9, ls='--', c='red', alpha=0.7, label='порог бимодальности (5/9)')
    ax.set_xlabel('phase')
    ax.set_ylabel('Sarle bimodality coefficient')
    ax.set_title('Бимодальность распределения V — выше порога = два режима')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'phase_curve_bimodality.png', dpi=120)
    plt.close()

    print(f'  графики сохранены в {REPORT_DIR}/')


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repredict', action='store_true',
                    help='пересчитать predictions даже если кэш существует')
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f'⚠ Нет модели {MODEL_PATH} — запустите сначала 02_value_function_v2.py')
        sys.exit(1)

    df = load_val_with_predictions(force_repredict=args.repredict)
    print(f'\nVal predictions: {len(df):,} строк, {df["episode_id"].nunique()} эпизодов')

    print('\n[1/4] AUC curve по 20 фазным бинам …')
    auc_df = auc_curve(df, n_bins=N_BINS)
    print(auc_df.to_string(index=False))

    print('\n[2/4] V(t) победителей vs проигравших + variance …')
    vbo_df = mean_v_by_outcome(df, n_bins=N_BINS)
    print(vbo_df.to_string(index=False))

    print('\n[3/4] Moments of max |dV/dt| по каждой партии …')
    tm_df = transition_moments(df)
    print(f'  partий проанализировано: {len(tm_df)}')
    print(f'  median phase_argmax: {tm_df["phase_argmax"].median():.3f}')
    print(f'  std phase_argmax:    {tm_df["phase_argmax"].std():.3f}')
    # Quantiles
    q = tm_df['phase_argmax'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f'  quantiles: {dict(q.round(3))}')

    print('\n[4/4] Бимодальность V в каждом бине …')
    try:
        bm_df = bimodality_test(df, n_bins=N_BINS)
        print(bm_df[['lo','hi','n','mean','bimodality_coef','is_bimodal_naive']]
              .to_string(index=False))
    except ImportError:
        print('  scipy недоступен — bimodality пропущена')
        bm_df = pd.DataFrame()

    # Сохранения
    auc_df.to_csv(REPORT_DIR / 'phase_curve_auc.csv', index=False)
    vbo_df.to_csv(REPORT_DIR / 'phase_curve_vbo.csv', index=False)
    tm_df.to_csv(REPORT_DIR / 'phase_curve_transitions.csv', index=False)
    if not bm_df.empty:
        bm_df.to_csv(REPORT_DIR / 'phase_curve_bimodality.csv', index=False)

    plot_all(auc_df, vbo_df, tm_df, bm_df)

    # Markdown отчёт
    md = []
    md.append('# Фазовый переход — карта по результатам value_v2\n\n')
    md.append(f'**Datasamples**: {len(df):,} строк (val split, {df["episode_id"].nunique()} эпизодов)\n\n')
    md.append('## 1. AUC по фазе игры\n\n')
    md.append('| phase | n | AUC | pos_rate |\n|---|---|---|---|\n')
    for _, r in auc_df.iterrows():
        md.append(f"| [{r['lo']}, {r['hi']}) | {r['n']:,} | {r['auc']} | {r['pos_rate']} |\n")

    md.append('\n## 2. Бифуркация V(t)\n\n')
    md.append('| phase | mean V (победители) | mean V (проигравшие) | разрыв | var(V) |\n')
    md.append('|---|---|---|---|---|\n')
    for _, r in vbo_df.iterrows():
        gap = (r['mean_v_winners'] or 0) - (r['mean_v_losers'] or 0)
        md.append(f"| [{r['lo']}, {r['hi']}) | {r['mean_v_winners']} | {r['mean_v_losers']} | "
                  f"{round(gap,3)} | {r['var_v_all']} |\n")

    md.append('\n## 3. Момент максимального dV/dt\n\n')
    md.append(f'- partий: {len(tm_df)}\n')
    md.append(f'- median phase_argmax: {tm_df["phase_argmax"].median():.3f}\n')
    md.append(f'- 10/25/50/75/90 квантили: '
              f'{q.round(3).to_dict()}\n\n')
    md.append('Если распределение argmax узкое (std < 0.15) → есть характерный момент решения партии.\n')
    md.append('Если широкое (std > 0.25) → переход размазан.\n\n')

    if not bm_df.empty:
        md.append('## 4. Бимодальность распределения V\n\n')
        md.append('| phase | mean | bimodality_coef | bimodal? |\n|---|---|---|---|\n')
        for _, r in bm_df.iterrows():
            md.append(f"| [{r['lo']}, {r['hi']}) | {r['mean']} | "
                      f"{r['bimodality_coef']} | {r['is_bimodal_naive']} |\n")
        md.append('\nПорог bimodality (Sarle): bc > 5/9 ≈ 0.555.\n')

    md.append('\n## 5. Графики\n\n')
    md.append('- `phase_curve_auc.png` — AUC vs phase\n')
    md.append('- `phase_curve_bifurcation.png` — V(t) победителей vs проигравших\n')
    md.append('- `phase_curve_variance.png` — variance V across replays\n')
    md.append('- `phase_curve_transitions.png` — гистограмма моментов max |dV/dt|\n')
    md.append('- `phase_curve_bimodality.png` — bimodality coefficient vs phase\n')

    (REPORT_DIR / 'phase_curve.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/phase_curve.md')


if __name__ == '__main__':
    main()
