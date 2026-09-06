#!/usr/bin/env python3
"""
04_diagnostics.py — перепроверка двух выводов из 03_phase_curve.

Перед тем как опираться на эти выводы — убеждаемся, что они не артефакты.

ЧАСТЬ A: Реально ли стартовая позиция несёт сигнал
──────────────────────────────────────────────────
В 03 я говорил: V(winners) = 0.34, V(losers) = 0.30 в phase=[0, 0.05).
Разрыв +0.04 в самом начале — это сигнал в стартовой позиции, или
артефакт смешения 1v1 и FFA?

Гипотеза артефакта: в 1v1 winrate = 0.5 (модель предсказывает ~0.5),
в FFA winrate = 0.25 (модель предсказывает ~0.25). Смешали → получили
~0.31 со среднеквадратичным разбросом. Возможные победители тяготеют
к более высоким значениям просто из-за шума.

Проверка:
  A1) Сами raw features at step=1 симметричны? own_ship_ratio_p0
      и own_ship_ratio_p1 в одном эпизоде должны быть равны при
      симметричной карте.
  A2) V at phase=0 СТРАТИФИЦИРОВАННО по n_players: в 1v1 winrate
      должен быть 0.5, и V(winners) ≈ V(losers) ≈ 0.5 если сигнала нет.

ЧАСТЬ B: Артефакт ли бимодальность argmax |dV/dt|
──────────────────────────────────────────────────
В 03 я нашёл: 25% партий argmax в начале (phase ≈ 0), 25% в конце
(phase ≈ 0.99). Это может быть реальный сигнал ("early/late decisive
archetypes") или артефакт метода:
  - Edge effects в свёртке: kernel=5 + mode='same' даёт ослабленное
    усреднение на краях → ложно большие dV/dt
  - Endpoint effects: в конце партии V снаружи (0 или 1), что само
    по себе делает финальное dV/dt большим
  - Короткие партии: dV/dt очень шумный для малого числа точек

Проверка:
  B1) Пересчитать argmax с разными методами:
        - raw (без сглаживания)
        - smoothing kernel=5 (как было)
        - smoothing kernel=15
        - exclude first/last 10% steps
      Сравнить histograms — устойчива ли бимодальность?
  B2) Корреляция phase_argmax с длиной партии и c argmax-метрикой
      (max_dv) — что если короткие партии всегда дают пик в начале?
  B3) Для нескольких партий нарисовать V(t) с argmax — глазами
      проверить, реально ли там "событие".

ВХОД
────
  rebuild/data/val_predictions.csv  — predictions из 03_phase_curve
  data/processed/steps.csv          — для проверки симметрии (часть A)

ВЫХОД
─────
  rebuild/reports/diagnostics.md
  rebuild/reports/diagnostics_*.png

ЛЁГКИЙ скрипт (без xgboost, ~30 сек на 360K строк).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
PROC_DIR   = OL_DIR / "data" / "processed"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH  = DATA_DIR / "val_predictions.csv"
STEPS_CSV  = PROC_DIR / "steps.csv"


# ══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ A — стартовая симметрия
# ══════════════════════════════════════════════════════════════════════════

def check_raw_symmetry():
    """A1: на step=0/1, симметричны ли raw features между игроками?"""
    print('\n── A1: симметрия raw features ──────────────────────')
    cols = ['episode_id', 'player_idx', 'step', 'phase',
            'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
            'n_own_planets', 'own_ships_total', 'own_prod_total',
            'enemy_ships_total', 'n_enemy_planets']
    df = pd.read_csv(STEPS_CSV, usecols=cols)
    df['episode_id'] = df['episode_id'].astype(str)

    # Mergeс n_players
    meta = pd.read_csv(DATA_DIR / 'episodes_meta.csv')[['episode_id', 'n_players']]
    meta['episode_id'] = meta['episode_id'].astype(str)
    df = df.merge(meta, on='episode_id', how='left')

    # Только первый шаг
    first = df[df['step'] == 1].copy()
    print(f'  step=1 записей: {len(first)} ({first["episode_id"].nunique()} эпизодов)')

    # Группируем по эпизоду
    for n_p_val in [2, 4]:
        sub = first[first['n_players'] == n_p_val].copy()
        if not len(sub):
            continue
        print(f'\n  n_players={n_p_val}: {sub["episode_id"].nunique()} эпизодов')

        # Для каждого эпизода считаем разброс между игроками
        diffs = []
        for eid, g in sub.groupby('episode_id'):
            if len(g) < 2:
                continue
            for col in ['own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
                         'own_ships_total', 'own_prod_total']:
                values = g[col].values
                diff = values.max() - values.min()
                diffs.append({'episode_id': eid, 'feature': col, 'diff': diff,
                              'values': list(values)})
        diff_df = pd.DataFrame(diffs)

        print(f'  Различие между игроками (max - min внутри эпизода):')
        for feat in ['own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
                     'own_ships_total', 'own_prod_total']:
            sub_f = diff_df[diff_df['feature'] == feat]
            print(f'    {feat:24s}  median diff = {sub_f["diff"].median():.4f}, '
                  f'max diff = {sub_f["diff"].max():.4f}')

        # Сумма ratio'ов (должна быть ~1 - доля нейтралов)
        sums_ship = sub.groupby('episode_id')['own_ship_ratio'].sum()
        sums_prod = sub.groupby('episode_id')['own_prod_ratio'].sum()
        sums_pl   = sub.groupby('episode_id')['own_planet_ratio'].sum()
        print(f'  Сумма ratio\'ов по эпизоду (по всем {n_p_val} игрокам):')
        print(f'    own_ship_ratio   sum: median={sums_ship.median():.3f}')
        print(f'    own_prod_ratio   sum: median={sums_prod.median():.3f}')
        print(f'    own_planet_ratio sum: median={sums_pl.median():.3f}')


def check_start_v_symmetry():
    """A2: V at phase~0 стратифицированно по n_players. Если симметрия
    реальна, V(winners) ≈ V(losers) внутри каждой страты."""
    print('\n── A2: V at phase ≈ 0 по стратам ───────────────────')
    df = pd.read_csv(PRED_PATH)

    # phase < 0.02 (очень рано)
    early = df[df['phase'] < 0.02].copy()
    print(f'  записей с phase < 0.02: {len(early)}')

    results = {}
    for n_p_val in sorted(early['n_players'].dropna().unique()):
        sub = early[early['n_players'] == n_p_val]
        if len(sub) < 100:
            continue
        w = sub[sub['label'] == 1]['p_win']
        l = sub[sub['label'] == 0]['p_win']
        if len(w) < 10 or len(l) < 10:
            continue
        # Welch t-test (без scipy: при n > 50 t-распределение ≈ нормальное)
        from math import erf, sqrt
        a, b = np.asarray(w), np.asarray(l)
        if a.var(ddof=1) + b.var(ddof=1) > 0:
            se = sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
            t = (a.mean() - b.mean()) / se
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        else:
            t, p = 0.0, 1.0
        gap = w.mean() - l.mean()
        win_rate = sub['label'].mean()
        print(f'\n  n_players={int(n_p_val)}:')
        print(f'    win rate in stratum: {win_rate:.3f}')
        print(f'    V(winners) mean: {w.mean():.4f}  median: {w.median():.4f}  n={len(w)}')
        print(f'    V(losers)  mean: {l.mean():.4f}  median: {l.median():.4f}  n={len(l)}')
        print(f'    gap: {gap:+.4f}   t={t:.2f}  p={p:.2e}')
        results[int(n_p_val)] = {
            'win_rate': float(win_rate),
            'v_winners_mean': float(w.mean()),
            'v_losers_mean': float(l.mean()),
            'gap': float(gap),
            't_stat': float(t),
            'p_value': float(p),
            'n_winners': int(len(w)),
            'n_losers':  int(len(l)),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ B — артефакт ли бимодальность argmax |dV/dt|
# ══════════════════════════════════════════════════════════════════════════

def argmax_dv(v: np.ndarray, smoothing: int, drop_edges_frac: float = 0.0):
    """Найти phase позицию argmax |dV/dt|.
    drop_edges_frac > 0: игнорировать первые/последние N% точек.
    Возвращает (rel_index в [0, 1], max_dv)."""
    if len(v) < 3:
        return 0.0, 0.0
    if smoothing > 1 and len(v) >= smoothing:
        kernel = np.ones(smoothing) / smoothing
        v_sm = np.convolve(v, kernel, mode='same')
    else:
        v_sm = v
    dv = np.abs(np.diff(v_sm))
    if dv.size == 0:
        return 0.0, 0.0
    if drop_edges_frac > 0:
        n_drop = int(len(dv) * drop_edges_frac)
        if n_drop * 2 < len(dv):
            valid = np.zeros(len(dv), dtype=bool)
            valid[n_drop:len(dv) - n_drop] = True
            if not valid.any():
                return 0.0, 0.0
            # argmax только среди valid
            masked = np.where(valid, dv, -1)
            idx = int(np.argmax(masked))
        else:
            idx = int(np.argmax(dv))
    else:
        idx = int(np.argmax(dv))
    rel = idx / max(1, len(dv) - 1)  # в [0, 1]
    return rel, float(dv[idx])


def compare_argmax_methods():
    """B1+B2: argmax по разным методам, корреляция с длиной партии."""
    print('\n── B1: argmax |dV/dt| с разными методами ──────────')
    df = pd.read_csv(PRED_PATH)

    results = []
    methods = [
        ('raw',              1, 0.0),
        ('smooth_k5',        5, 0.0),
        ('smooth_k15',      15, 0.0),
        ('k5_drop10pct',     5, 0.10),
        ('k5_drop20pct',     5, 0.20),
    ]

    rows = []
    for (eid, p), g in df.groupby(['episode_id', 'player_idx']):
        g = g.sort_values('step')
        if len(g) < 10:
            continue
        v = g['p_win'].values
        n_steps = len(g)
        label = int(g['label'].iloc[-1])
        for name, k, drop in methods:
            rel, max_dv = argmax_dv(v, smoothing=k, drop_edges_frac=drop)
            rows.append({
                'episode_id': eid, 'player_idx': int(p),
                'n_steps': n_steps, 'final_label': label,
                'method': name,
                'argmax_rel': rel,
                'max_dv':    max_dv,
            })
    res = pd.DataFrame(rows)

    print(f'\n  partий проанализировано: {res["episode_id"].nunique()*2} '
          f'(по 2 игрока в среднем)')
    print(f'\n  Распределение argmax_rel по методам:')
    print(f'  {"method":<16} median   std    q25    q75    %<0.1   %>0.9')
    print(f'  {"-"*70}')
    for name, _, _ in methods:
        sub = res[res['method'] == name]['argmax_rel']
        pct_early = (sub < 0.1).mean() * 100
        pct_late  = (sub > 0.9).mean() * 100
        print(f'  {name:<16} {sub.median():.3f}   {sub.std():.3f}  '
              f'{sub.quantile(0.25):.3f}  {sub.quantile(0.75):.3f}  '
              f'{pct_early:5.1f}%  {pct_late:5.1f}%')

    # B2: корреляция argmax с длиной партии для каждого метода
    print(f'\n── B2: корреляция argmax_rel с длиной партии ──────')
    for name, _, _ in methods:
        sub = res[res['method'] == name]
        corr = sub[['n_steps', 'argmax_rel']].corr().iloc[0, 1]
        print(f'  {name:<16}  corr(argmax_rel, n_steps) = {corr:+.3f}')

    res.to_csv(REPORT_DIR / 'diagnostics_argmax.csv', index=False)
    return res


def plot_diagnostics(argmax_df, start_v_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib недоступен')
        return

    # 1) Гистограммы argmax_rel по методам
    methods = ['raw', 'smooth_k5', 'smooth_k15', 'k5_drop10pct', 'k5_drop20pct']
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for ax, name in zip(axes, methods):
        sub = argmax_df[argmax_df['method'] == name]['argmax_rel']
        ax.hist(sub, bins=20, edgecolor='k', alpha=0.7)
        ax.set_title(f'{name}\nn={len(sub)}, med={sub.median():.2f}')
        ax.set_xlabel('argmax_rel')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('partий')
    plt.suptitle('Distribution of argmax |dV/dt| — by method (если бимодальность артефакт, она исчезнет на drop_edges)')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'diagnostics_argmax_histograms.png', dpi=120)
    plt.close()

    # 2) V(t) для нескольких партий с argmax в начале и в конце (метод smooth_k5)
    df_pred = pd.read_csv(PRED_PATH)
    sub_method = argmax_df[argmax_df['method'] == 'smooth_k5']
    early = sub_method[sub_method['argmax_rel'] < 0.1].head(4)
    late  = sub_method[sub_method['argmax_rel'] > 0.9].head(4)
    middle = sub_method[(sub_method['argmax_rel'] > 0.4) & (sub_method['argmax_rel'] < 0.6)].head(4)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for row_idx, (label, group) in enumerate([
        ('early (argmax<0.1)', early),
        ('middle (~0.5)', middle),
        ('late (>0.9)', late),
    ]):
        for col_idx, (_, r) in enumerate(group.iterrows()):
            ax = axes[row_idx, col_idx]
            g = df_pred[(df_pred['episode_id'] == r['episode_id']) &
                         (df_pred['player_idx'] == r['player_idx'])].sort_values('step')
            ax.plot(g['phase'], g['p_win'], '-', lw=1)
            ax.axvline(r['argmax_rel'], c='red', ls='--', alpha=0.7,
                        label=f'argmax')
            ax.set_title(f"{str(r['episode_id'])[:8]}/p{r['player_idx']} "
                         f"n={r['n_steps']} lbl={r['final_label']}", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.3)
        axes[row_idx, 0].set_ylabel(label, fontsize=10)
    plt.suptitle('Glance: V(t) на партиях с разными argmax — реальные ли там события?')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'diagnostics_trajectory_glance.png', dpi=120)
    plt.close()

    # 3) Раскладка V в начале по n_players
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    early_pred = df_pred[df_pred['phase'] < 0.02]
    for ax, n_p in zip(axes, [2, 4]):
        sub = early_pred[early_pred['n_players'] == n_p]
        if not len(sub):
            continue
        w = sub[sub['label'] == 1]['p_win']
        l = sub[sub['label'] == 0]['p_win']
        ax.hist(w, bins=30, alpha=0.5, label=f'winners n={len(w)}', color='green')
        ax.hist(l, bins=30, alpha=0.5, label=f'losers n={len(l)}', color='red')
        ax.axvline(w.mean(), c='green', ls='--')
        ax.axvline(l.mean(), c='red', ls='--')
        ax.set_title(f'V at phase<0.02, n_players={n_p}\n'
                     f'gap = {w.mean()-l.mean():+.3f}')
        ax.set_xlabel('V')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle('Стартовая позиция: что в V(winners) vs V(losers) внутри страты')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'diagnostics_start_v.png', dpi=120)
    plt.close()
    print(f'  графики: {REPORT_DIR}/diagnostics_*.png')


def write_report(start_v_results, argmax_df):
    md = []
    md.append('# Diagnostics — перепроверка выводов phase_curve\n\n')
    md.append('Перед опорой на наблюдения из 03_phase_curve проверяем что они не артефакты.\n\n')

    md.append('## A. Симметрия стартовой позиции\n\n')
    md.append('### A2: V(winners) vs V(losers) at phase < 0.02, стратифицированно\n\n')
    md.append('| n_players | win_rate | V(winners) | V(losers) | gap | p-value |\n')
    md.append('|---|---|---|---|---|---|\n')
    for n_p, r in start_v_results.items():
        md.append(f"| {n_p} | {r['win_rate']:.3f} | {r['v_winners_mean']:.4f} | "
                  f"{r['v_losers_mean']:.4f} | {r['gap']:+.4f} | {r['p_value']:.1e} |\n")
    md.append('\n**Интерпретация**:\n')
    md.append('- Если gap внутри страты ≈ 0 (или p-value > 0.05) → симметрия выполнена,\n')
    md.append('  моё прежнее наблюдение "разрыв 0.04 в самом начале" было артефактом смешения\n')
    md.append('  1v1 и FFA с разным base rate.\n')
    md.append('- Если gap значимый — модель видит реальный (хоть и слабый) сигнал в стартовой позиции.\n\n')

    md.append('## B. Артефакт ли бимодальность argmax |dV/dt|\n\n')
    methods = ['raw', 'smooth_k5', 'smooth_k15', 'k5_drop10pct', 'k5_drop20pct']
    md.append('### B1: распределение argmax_rel по методам\n\n')
    md.append('| method | median | std | % argmax<0.1 | % argmax>0.9 |\n')
    md.append('|---|---|---|---|---|\n')
    for name in methods:
        sub = argmax_df[argmax_df['method'] == name]['argmax_rel']
        pct_e = (sub < 0.1).mean() * 100
        pct_l = (sub > 0.9).mean() * 100
        md.append(f'| {name} | {sub.median():.3f} | {sub.std():.3f} | '
                  f'{pct_e:.1f}% | {pct_l:.1f}% |\n')
    md.append('\n')
    md.append('### B2: корреляция argmax_rel с n_steps\n\n')
    md.append('| method | corr(argmax_rel, n_steps) |\n|---|---|\n')
    for name in methods:
        sub = argmax_df[argmax_df['method'] == name]
        corr = sub[['n_steps', 'argmax_rel']].corr().iloc[0, 1]
        md.append(f'| {name} | {corr:+.3f} |\n')
    md.append('\n**Интерпретация**:\n')
    md.append('- Если в `k5_drop20pct` % argmax<0.1 и % argmax>0.9 резко падают → бимодальность артефакт.\n')
    md.append('- Если бимодальность сохраняется во всех методах → это реальный сигнал.\n')
    md.append('- Сильная отрицательная корреляция corr(argmax_rel, n_steps) указывает на проблему: короткие партии всегда дают пик в начале.\n')
    md.append('\n## Графики\n\n')
    md.append('- `diagnostics_argmax_histograms.png` — гистограммы argmax_rel по методам\n')
    md.append('- `diagnostics_trajectory_glance.png` — V(t) на партиях с разными argmax\n')
    md.append('- `diagnostics_start_v.png` — V в начале партии по стратам\n')

    (REPORT_DIR / 'diagnostics.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/diagnostics.md')


def main():
    if not PRED_PATH.exists():
        print(f'⚠ нет val_predictions.csv — запустите 03_phase_curve.py')
        sys.exit(1)

    # ── A ──
    check_raw_symmetry()
    start_v_results = check_start_v_symmetry()

    # ── B ──
    argmax_df = compare_argmax_methods()

    # ── Plots + Report ──
    plot_diagnostics(argmax_df, start_v_results)
    write_report(start_v_results, argmax_df)


if __name__ == '__main__':
    main()
