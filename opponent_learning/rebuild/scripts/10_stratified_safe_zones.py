#!/usr/bin/env python3
"""
10_stratified_safe_zones.py — safe zones и t_soft/t_hard ВНУТРИ страт карт.

ЗАЧЕМ
─────
В 05 мы нашли глобальные safe zones по prod_advantage:
  soft:  [0.38, 0.60]   hard:  [0.16, 0.84]
И получили median t_soft = 0.676.

Но на разных картах "точка невозврата" может быть разной:
  • На small_static (мало планет, статичные) — отыграться сложно,
    safe zone может быть **уже** → t_soft наступает **раньше**
  • На large_orbital (много планет, динамика) — отыграться легче,
    safe zone **шире** → t_soft позже

Если найти **stratified** safe zones — получим более ранние t_soft
на одних картах, что расширит pre-resolution окно и даст больше материала
для leading indicator анализа.

ВХОД
────
  data/processed/steps.csv
  rebuild/data/episodes_meta.csv
  rebuild/data/map_clusters.csv     (из 08)

ВЫХОД
─────
  rebuild/data/resolution_moments_v2.csv
  rebuild/reports/stratified_safe_zones.md
  rebuild/reports/stratified_safe_zones.png

ЛЁГКИЙ скрипт (~30 сек, без xgboost / sklearn).
"""

from __future__ import annotations

import sys
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

STEPS_CSV    = PROC_DIR / "steps.csv"
META_PATH    = DATA_DIR / "episodes_meta.csv"
CLUSTER_PATH = DATA_DIR / "map_clusters.csv"
OLD_MOM_PATH = DATA_DIR / "resolution_moments.csv"
OUT_PATH     = DATA_DIR / "resolution_moments_v2.csv"


def load_1v1():
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

    clusters = pd.read_csv(CLUSTER_PATH)[['episode_id', 'map_type']]
    clusters['episode_id'] = clusters['episode_id'].astype(str)
    df = df.merge(clusters, on='episode_id', how='left')
    print(f'1v1 строк: {len(df):,}  типов карт: {df["map_type"].nunique()}')
    return df


def find_safe_zones(df: pd.DataFrame, param: str,
                    n_bins: int = 50, min_n: int = 30,
                    eps: float = 0.02) -> dict:
    bins = np.linspace(df[param].min(), df[param].max() + 1e-9, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sub = df[(df[param] >= lo) & (df[param] < hi)]
        if len(sub) < min_n:
            rows.append((lo, hi, len(sub), None))
        else:
            wr = sub['label'].mean()
            rows.append((lo, hi, len(sub), float(wr)))
    table = pd.DataFrame(rows, columns=['lo', 'hi', 'n', 'winrate'])
    valid = table.dropna(subset=['winrate']).reset_index(drop=True)

    safe_low, safe_high = None, None
    for _, r in valid.iterrows():
        if r['winrate'] <= eps:
            safe_low = r['hi']
        else:
            break
    for _, r in valid.iloc[::-1].iterrows():
        if r['winrate'] >= 1 - eps:
            safe_high = r['lo']
        else:
            break
    return {'safe_low': safe_low, 'safe_high': safe_high, 'table': table}


def find_t_star(g: pd.DataFrame, sl, sh, param='prod_advantage'):
    g = g.sort_values('step').reset_index(drop=True)
    vals = g[param].values
    steps = g['step'].values
    n = len(vals)
    if sh is None and sl is None:
        return None, None
    in_win  = (vals >= sh) if sh is not None else np.zeros(n, bool)
    in_loss = (vals <= sl) if sl is not None else np.zeros(n, bool)
    sw, sl_arr = np.zeros(n, bool), np.zeros(n, bool)
    sw[-1] = in_win[-1]; sl_arr[-1] = in_loss[-1]
    for i in range(n-2, -1, -1):
        sw[i] = in_win[i] and sw[i+1]
        sl_arr[i] = in_loss[i] and sl_arr[i+1]
    cand = []
    if sw.any():
        i = int(np.argmax(sw)); cand.append((int(steps[i]), 'win'))
    if sl_arr.any():
        i = int(np.argmax(sl_arr)); cand.append((int(steps[i]), 'loss'))
    if not cand:
        return None, None
    cand.sort(key=lambda x: x[0])
    return cand[0]


def main():
    if not all(p.exists() for p in [STEPS_CSV, META_PATH, CLUSTER_PATH]):
        print('⚠ Нужны steps.csv, episodes_meta.csv, map_clusters.csv')
        sys.exit(1)

    df = load_1v1()
    strata = sorted(df['map_type'].dropna().unique())
    print(f'Страты: {strata}')

    # ── Глобальные safe zones (для сравнения) ──────────────────────
    print('\n── Глобальные safe zones (как в 05) ───────────────────')
    g_soft = find_safe_zones(df, 'prod_advantage', eps=0.10, min_n=100)
    g_hard = find_safe_zones(df, 'prod_advantage', eps=0.02, min_n=100)
    print(f'  soft (eps=0.10): [{g_soft["safe_low"]}, {g_soft["safe_high"]}]')
    print(f'  hard (eps=0.02): [{g_hard["safe_low"]}, {g_hard["safe_high"]}]')

    # ── Stratified safe zones ──────────────────────────────────────
    print('\n── Stratified safe zones по типу карты ────────────────')
    stratum_zones = {}
    for s in strata:
        sub = df[df['map_type'] == s]
        if len(sub) < 1000:
            print(f'  {s:<18}  пропускаем (n={len(sub)} мало)')
            continue
        soft = find_safe_zones(sub, 'prod_advantage', eps=0.10, min_n=30)
        hard = find_safe_zones(sub, 'prod_advantage', eps=0.02, min_n=30)
        stratum_zones[s] = {'soft': soft, 'hard': hard, 'n': len(sub)}
        print(f'  {s:<18}  n={len(sub):>6}  '
              f'soft=[{soft["safe_low"]}, {soft["safe_high"]}]  '
              f'hard=[{hard["safe_low"]}, {hard["safe_high"]}]')

    # ── Считаем t_soft / t_hard со stratified zones ────────────────
    print('\n── t_soft / t_hard со stratified zones ────────────────')
    out = []
    no_res = 0
    for (eid, pid), g in df.groupby(['episode_id', 'player_idx']):
        s = g['map_type'].iloc[0]
        if s not in stratum_zones:
            # fallback к глобальным
            zs = {'soft': g_soft, 'hard': g_hard}
        else:
            zs = stratum_zones[s]
        n_steps_total = int(g['step'].max() + 1)
        t_soft, dir_soft = find_t_star(g, zs['soft']['safe_low'],
                                        zs['soft']['safe_high'])
        t_hard, dir_hard = find_t_star(g, zs['hard']['safe_low'],
                                        zs['hard']['safe_high'])
        if t_soft is None:
            no_res += 1
            continue
        phase_t_soft = float(g[g['step'] == t_soft]['phase'].iloc[0])
        phase_t_hard = (float(g[g['step'] == t_hard]['phase'].iloc[0])
                        if t_hard is not None else None)
        window_size = max(5, int(0.2 * n_steps_total))
        t_window_start = max(0, t_soft - window_size)
        actual_window = t_soft - t_window_start
        final_label = int(g['label'].iloc[-1])
        out.append({
            'episode_id': eid, 'player_idx': int(pid),
            'map_type': s, 'n_steps_total': n_steps_total,
            't_soft': t_soft, 't_hard': t_hard,
            'phase_t_soft': round(phase_t_soft, 4),
            'phase_t_hard': round(phase_t_hard, 4) if phase_t_hard is not None else None,
            'direction': dir_soft,
            't_window_start': int(t_window_start),
            'window_size': int(actual_window),
            'final_label': final_label,
            'consistent': int(dir_soft == ('win' if final_label == 1 else 'loss')),
        })

    moments = pd.DataFrame(out)
    moments.to_csv(OUT_PATH, index=False)
    print(f'  партий: {len(moments)} из {df.groupby(["episode_id", "player_idx"]).ngroups}')
    print(f'  без resolution: {no_res}')
    print(f'  consistency: {moments["consistent"].mean():.3f}')
    print(f'\n✓ Сохранено: {OUT_PATH}')

    # ── Сравнение с глобальной версией ─────────────────────────────
    if OLD_MOM_PATH.exists():
        old = pd.read_csv(OLD_MOM_PATH)
        old['episode_id'] = old['episode_id'].astype(str)
        cmp = moments.merge(
            old[['episode_id', 'player_idx', 'phase_t_soft']].rename(
                columns={'phase_t_soft': 'phase_t_soft_global'}),
            on=['episode_id', 'player_idx'], how='inner')
        cmp['phase_shift'] = cmp['phase_t_soft'] - cmp['phase_t_soft_global']
        print('\n── Сравнение global vs stratified t_soft ─────────')
        for s in strata:
            sub = cmp[cmp['map_type'] == s]
            if len(sub) < 5: continue
            print(f'  {s:<18}  n={len(sub):>3}  '
                  f'med shift={sub["phase_shift"].median():+.3f}  '
                  f'mean shift={sub["phase_shift"].mean():+.3f}  '
                  f'(меньше=раньше t_soft в страт. версии)')

    # ── По стратам: распределение phase_t_soft ──────────────────────
    print('\n── Распределение phase_t_soft по стратам ──────────────')
    print(moments.groupby('map_type')['phase_t_soft'].describe().round(3).to_string())
    print('\n── Распределение window_size по стратам ────────────────')
    print(moments.groupby('map_type')['window_size'].describe().round(1).to_string())

    # Markdown
    md = []
    md.append('# Stratified safe zones и t_soft\n\n')
    md.append('## Safe zones по типу карты (1v1)\n\n')
    md.append('| Тип карты | n | soft_low | soft_high | hard_low | hard_high |\n')
    md.append('|---|---|---|---|---|---|\n')
    md.append(f"| GLOBAL (all) | {len(df)} | {g_soft['safe_low']} | {g_soft['safe_high']} | "
              f"{g_hard['safe_low']} | {g_hard['safe_high']} |\n")
    for s, z in stratum_zones.items():
        md.append(f"| {s} | {z['n']} | {z['soft']['safe_low']} | {z['soft']['safe_high']} | "
                  f"{z['hard']['safe_low']} | {z['hard']['safe_high']} |\n")

    md.append('\n## Распределение `phase_t_soft` по стратам\n\n```\n')
    md.append(moments.groupby('map_type')['phase_t_soft'].describe().round(3).to_string())
    md.append('\n```\n\n## Распределение `window_size` по стратам\n\n```\n')
    md.append(moments.groupby('map_type')['window_size'].describe().round(1).to_string())
    md.append('\n```\n')

    (REPORT_DIR / 'stratified_safe_zones.md').write_text(''.join(md))
    print(f'\n✓ Отчёт: {REPORT_DIR}/stratified_safe_zones.md')

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for s in sorted(moments['map_type'].dropna().unique()):
            sub = moments[moments['map_type'] == s]
            if len(sub) < 5: continue
            axes[0].hist(sub['phase_t_soft'], bins=20, alpha=0.5, label=f'{s} (n={len(sub)})')
            axes[1].hist(sub['window_size'], bins=20, alpha=0.5, label=f'{s} (n={len(sub)})')
        axes[0].set_xlabel('phase_t_soft')
        axes[0].set_ylabel('число партий')
        axes[0].set_title('Распределение phase_t_soft по типам карт (stratified zones)')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel('window_size (шагов)')
        axes[1].set_title('Распределение window_size по типам карт')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'stratified_safe_zones.png', dpi=120)
        plt.close()
        print(f'✓ График: {REPORT_DIR}/stratified_safe_zones.png')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
