#!/usr/bin/env python3
"""
05_resolution_moments.py — эмпирический момент предрешённости t* для каждой 1v1 партии.

ИДЕЯ (от пользователя)
──────────────────────
Не использовать произвольный порог value function. Вместо этого: найти ЭМПИРИЧЕСКИ
по всем 1v1 шагам зоны в пространстве относительных параметров, где партии
никогда не переворачивались. t* = самый ранний шаг где партия зашла в такую
"safe zone" И остаётся там до конца.

Параметры РЕЛЯТИВНЫЕ (одни и те же между картами):
  • own_prod_ratio   — доля производства игрока (primary)
  • own_ship_ratio   — доля кораблей
  • own_planet_ratio — доля планет

ОКНО для последующего анализа: 20% от длины партии перед t*.

ВХОД
────
  data/processed/steps.csv
  rebuild/data/episodes_meta.csv

ВЫХОД
─────
  rebuild/data/resolution_moments.csv  — t* для каждой (episode_id, player_idx)
  rebuild/reports/resolution_moments.md — отчёт со статистикой и safe-zone границами
  rebuild/reports/resolution_*.png      — графики

ЛЁГКИЙ скрипт (~30 секунд, без xgboost).
"""

from __future__ import annotations

import json
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

STEPS_CSV  = PROC_DIR / "steps.csv"
META_PATH  = DATA_DIR / "episodes_meta.csv"
OUT_PATH   = DATA_DIR / "resolution_moments.csv"


def load_1v1_steps() -> pd.DataFrame:
    """Загружает только 1v1 партии и добавляет симметричные относительные параметры."""
    cols = ['episode_id', 'player_idx', 'step', 'phase', 'reward',
            'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
            'own_ships_total', 'own_prod_total', 'n_own_planets',
            'enemy_ships_total', 'n_enemy_planets',
            'ship_gap', 'prod_gap', 'n_planets']
    df = pd.read_csv(STEPS_CSV, usecols=cols)
    df['episode_id'] = df['episode_id'].astype(str)
    df['label'] = (df['reward'] > 0).astype(int)

    meta = pd.read_csv(META_PATH)[['episode_id', 'n_players', 'n_steps']]
    meta['episode_id'] = meta['episode_id'].astype(str)
    df = df.merge(meta, on='episode_id', how='left')

    df_1v1 = df[df['n_players'] == 2].copy()

    # ── Симметричные относительные параметры (1v1) ────────────────────────
    # enemy_prod_total = own_prod_total - prod_gap
    df_1v1['enemy_prod_total'] = (df_1v1['own_prod_total']
                                  - df_1v1['prod_gap']).clip(lower=0)
    # prod_advantage в [0, 1], 0.5 = паритет, 1 = enemy исчез
    denom_p = (df_1v1['own_prod_total'] + df_1v1['enemy_prod_total']).clip(lower=1)
    df_1v1['prod_advantage'] = df_1v1['own_prod_total'] / denom_p

    # ship_advantage аналогично
    denom_s = (df_1v1['own_ships_total'] + df_1v1['enemy_ships_total']).clip(lower=1)
    df_1v1['ship_advantage'] = df_1v1['own_ships_total'] / denom_s

    # planet_advantage (только мои/мои+вражеские, игнорируем нейтралов)
    denom_pl = (df_1v1['n_own_planets'] + df_1v1['n_enemy_planets']).clip(lower=1)
    df_1v1['planet_advantage'] = df_1v1['n_own_planets'] / denom_pl

    print(f'1v1 эпизодов: {df_1v1["episode_id"].nunique()}')
    print(f'1v1 строк:    {len(df_1v1):,}')
    return df_1v1


def find_safe_zones(df: pd.DataFrame, param: str,
                    n_bins: int = 50, min_n: int = 30,
                    eps: float = 0.02) -> dict:
    """Бинит параметр, считает winrate в каждом бине, находит границы safe zones.

    Возвращает: {'safe_low': x, 'safe_high': y} — границы где winrate ≤ eps или ≥ 1-eps.
    """
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

    # Safe-low: самый правый bin где winrate ≤ eps подряд от лева
    # Safe-high: самый левый bin где winrate ≥ 1-eps подряд до права
    safe_low = None
    safe_high = None
    valid = table.dropna(subset=['winrate']).reset_index(drop=True)

    # Идём слева: пока winrate ≤ eps, продвигаем safe_low до hi бина
    for _, r in valid.iterrows():
        if r['winrate'] <= eps:
            safe_low = r['hi']  # верхняя граница "безнадёжной" зоны
        else:
            break
    # Идём справа: пока winrate ≥ 1-eps, продвигаем safe_high
    for _, r in valid.iloc[::-1].iterrows():
        if r['winrate'] >= 1 - eps:
            safe_high = r['lo']  # нижняя граница "выигрышной" зоны
        else:
            break

    return {
        'param': param,
        'safe_low': safe_low,   # ниже этого = почти точно проиграет
        'safe_high': safe_high, # выше этого = почти точно выиграет
        'table': table,
        'eps': eps,
    }


def find_t_star(df_partition: pd.DataFrame, safe_low: float,
                safe_high: float, param: str) -> tuple[int | None, str]:
    """Для одной (episode, player) партии — t*.

    Партия "в safe zone выигрыша" если param ≥ safe_high.
    Партия "в safe zone проигрыша" если param ≤ safe_low.
    t* = самый ранний шаг такой что:
      - param[t*] в safe zone (любой из двух),
      - И param[t*+1:] остаётся в той же safe zone до конца партии.
    Возвращает (t_star, direction) где direction in {'win','loss',None}.
    """
    df_partition = df_partition.sort_values('step').reset_index(drop=True)
    vals = df_partition[param].values
    steps = df_partition['step'].values

    if safe_high is None and safe_low is None:
        return None, None

    # для каждого индекса i проверяем — все ли последующие в той же зоне
    in_win  = (vals >= safe_high) if safe_high is not None else np.zeros_like(vals, dtype=bool)
    in_loss = (vals <= safe_low)  if safe_low  is not None else np.zeros_like(vals, dtype=bool)

    # сzaдней суффиксный AND
    n = len(vals)
    suffix_in_win  = np.zeros(n, dtype=bool)
    suffix_in_loss = np.zeros(n, dtype=bool)
    suffix_in_win[-1]  = in_win[-1]
    suffix_in_loss[-1] = in_loss[-1]
    for i in range(n - 2, -1, -1):
        suffix_in_win[i]  = in_win[i] and suffix_in_win[i+1]
        suffix_in_loss[i] = in_loss[i] and suffix_in_loss[i+1]

    # самый ранний t где он стартует suffix_in_win/loss
    win_idx  = np.argmax(suffix_in_win)  if suffix_in_win.any()  else -1
    loss_idx = np.argmax(suffix_in_loss) if suffix_in_loss.any() else -1

    cand = []
    if suffix_in_win.any():
        cand.append((steps[win_idx], 'win', win_idx))
    if suffix_in_loss.any():
        cand.append((steps[loss_idx], 'loss', loss_idx))
    if not cand:
        return None, None
    cand.sort(key=lambda x: x[0])  # самый ранний
    return int(cand[0][0]), cand[0][1]


def main():
    df = load_1v1_steps()

    print('\n── Поиск safe zones для own_prod_ratio (primary) ─────')
    zones = find_safe_zones(df, 'prod_advantage', n_bins=50, min_n=100, eps=0.02)
    print(f'  safe_low  (winrate ≤ 0.02): own_prod_ratio ≤ {zones["safe_low"]}')
    print(f'  safe_high (winrate ≥ 0.98): own_prod_ratio ≥ {zones["safe_high"]}')

    # Так же для других симметричных параметров — справочно
    print('\n── Справочно: safe zones по другим симметричным advantage ───')
    for p in ['ship_advantage', 'planet_advantage', 'own_prod_ratio']:
        z = find_safe_zones(df, p, n_bins=50, min_n=100, eps=0.02)
        print(f'  {p:20s}: safe_low={z["safe_low"]}, safe_high={z["safe_high"]}')

    # Печатаем таблицу winrate vs prod_ratio для понимания
    print('\n── winrate vs own_prod_ratio (детально) ─────────────')
    t = zones['table'].dropna(subset=['winrate'])
    for _, r in t.iterrows():
        flag = ''
        if zones['safe_low'] is not None and r['hi'] <= zones['safe_low']:
            flag = ' [SAFE LOSS]'
        elif zones['safe_high'] is not None and r['lo'] >= zones['safe_high']:
            flag = ' [SAFE WIN]'
        print(f'  prod_ratio [{r["lo"]:.3f}, {r["hi"]:.3f})  n={int(r["n"]):>5}  winrate={r["winrate"]:.3f}{flag}')

    # Двухпорог: soft (eps=0.10, изучаем) и hard (eps=0.02, валидация)
    print('\n── Двухпорог: t_soft (eps=0.10) и t_hard (eps=0.02) ──')
    zones_soft = find_safe_zones(df, 'prod_advantage', n_bins=50, min_n=100, eps=0.10)
    zones_hard = zones  # уже посчитан с eps=0.02
    sl_soft = zones_soft['safe_low']
    sh_soft = zones_soft['safe_high']
    sl_hard = zones_hard['safe_low']
    sh_hard = zones_hard['safe_high']
    print(f'  t_soft порог: safe_low ≤ {sl_soft}, safe_high ≥ {sh_soft}')
    print(f'  t_hard порог: safe_low ≤ {sl_hard}, safe_high ≥ {sh_hard}')

    # Находим t_soft и t_hard для каждой (episode, player) партии
    print('\n── Находим t_soft / t_hard для всех 1v1 партий ───────')

    out = []
    no_resolution = 0
    sanity_violations = 0
    for (eid, pid), g in df.groupby(['episode_id', 'player_idx']):
        t_soft, dir_soft = find_t_star(g, sl_soft, sh_soft, 'prod_advantage')
        t_hard, dir_hard = find_t_star(g, sl_hard, sh_hard, 'prod_advantage')

        n_steps_total = g['step'].max() + 1
        if t_soft is None:
            no_resolution += 1
            continue

        phase_t_soft = float(g[g['step'] == t_soft]['phase'].iloc[0])
        phase_t_hard = (float(g[g['step'] == t_hard]['phase'].iloc[0])
                        if t_hard is not None else None)

        # sanity: t_hard должен быть >= t_soft (партия сначала становится "почти решённой", потом "точно решённой")
        sanity_ok = (t_hard is None) or (t_hard >= t_soft)
        if not sanity_ok:
            sanity_violations += 1

        # окно: 20% от длины ОТ t_soft
        window_size = max(5, int(0.2 * n_steps_total))
        t_window_start = max(0, t_soft - window_size)
        actual_window = t_soft - t_window_start
        final_label = int(g['label'].iloc[-1])

        out.append({
            'episode_id':       eid,
            'player_idx':       int(pid),
            'n_steps_total':    int(n_steps_total),
            't_soft':           int(t_soft),
            't_hard':           int(t_hard) if t_hard is not None else None,
            'phase_t_soft':     round(phase_t_soft, 4),
            'phase_t_hard':     round(phase_t_hard, 4) if phase_t_hard is not None else None,
            'direction':        dir_soft,
            't_window_start':   int(t_window_start),
            'window_size':      int(actual_window),
            'final_label':      final_label,
            'consistent':       int(dir_soft == ('win' if final_label == 1 else 'loss')),
            'sanity_ok':        int(sanity_ok),
        })
    print(f'  sanity violations (t_hard < t_soft): {sanity_violations}')

    moments = pd.DataFrame(out)
    print(f'  партий с t*:        {len(moments)} из {df.groupby(["episode_id", "player_idx"]).ngroups}')
    print(f'  без resolution:      {no_resolution}')
    print(f'  consistent (direction = final outcome): {moments["consistent"].sum()} '
          f'({100*moments["consistent"].mean():.1f}%)')

    print(f'\n  Распределение phase_at_t_star:')
    print(moments['phase_t_soft'].describe().round(3).to_string())
    print(f'\n  Распределение window_size:')
    print(moments['window_size'].describe().round(1).to_string())
    print(f'\n  Quantiles phase_at_t_star:')
    print(moments['phase_t_soft'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_string())

    # Сохраняем
    moments.to_csv(OUT_PATH, index=False)
    print(f'\n✓ Сохранено: {OUT_PATH}')

    # ── Графики ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 1) winrate vs own_prod_ratio
        fig, ax = plt.subplots(figsize=(9, 5))
        t = zones['table'].dropna(subset=['winrate'])
        x = (t['lo'] + t['hi']) / 2
        ax.plot(x, t['winrate'], 'o-')
        if sl_soft is not None:
            ax.axvline(sl_soft, c='orange', ls='--', label=f'soft_low={sl_soft:.3f}')
        if sh_soft is not None:
            ax.axvline(sh_soft, c='lime', ls='--', label=f'soft_high={sh_soft:.3f}')
        if sl_hard is not None:
            ax.axvline(sl_hard, c='red', ls='--', label=f'hard_low={sl_hard:.3f}')
        if sh_hard is not None:
            ax.axvline(sh_hard, c='green', ls='--', label=f'hard_high={sh_hard:.3f}')
        ax.axhline(0.02, c='red', alpha=0.3)
        ax.axhline(0.98, c='green', alpha=0.3)
        ax.set_xlabel('prod_advantage')
        ax.set_ylabel('эмпирический winrate')
        ax.set_title('Safe zones в пространстве own_prod_ratio (1v1)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'resolution_safe_zones.png', dpi=120)
        plt.close()

        # 2) Распределение phase_at_t_star
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].hist(moments['phase_t_soft'], bins=30, edgecolor='k', alpha=0.7)
        axes[0].set_xlabel('phase at t*')
        axes[0].set_ylabel('число партий')
        axes[0].set_title(f'Распределение момента предрешённости\n(n={len(moments)} player-partий)')
        axes[0].grid(alpha=0.3)

        axes[1].hist(moments['window_size'], bins=30, edgecolor='k', alpha=0.7, color='orange')
        axes[1].set_xlabel('размер окна (число шагов в [t*-N, t*])')
        axes[1].set_ylabel('число партий')
        axes[1].set_title('Размер pre-resolution окна')
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'resolution_distribution.png', dpi=120)
        plt.close()
    except ImportError:
        print('  matplotlib не доступен — графики пропущены')

    # ── Markdown отчёт ──
    md = []
    md.append('# Resolution Moments — эмпирический t* для 1v1 партий\n\n')
    md.append('## Метод\n\n')
    md.append('Определяем `t*` как самый ранний шаг где партия зашла в "safe zone"\n')
    md.append('по `own_prod_ratio` и оставалась там до конца. Safe zones найдены эмпирически:\n')
    md.append('диапазоны параметра где winrate ≤ 2% (точно проиграет) или ≥ 98% (точно выиграет).\n\n')
    md.append('## Найденные safe zones\n\n')
    md.append('| Параметр | safe_low (≤2% winrate) | safe_high (≥98% winrate) |\n')
    md.append('|----------|------------------------|--------------------------|\n')
    for p in ['prod_advantage', 'ship_advantage', 'planet_advantage', 'own_prod_ratio']:
        z = find_safe_zones(df, p, n_bins=50, min_n=100, eps=0.02)
        md.append(f"| {p} | {z['safe_low']} | {z['safe_high']} |\n")
    md.append('\n## Статистика t*\n\n')
    md.append(f'- Партий-игроков с найденным t*: **{len(moments)}** / {df.groupby(["episode_id", "player_idx"]).ngroups}\n')
    md.append(f'- Без resolution в данных: {no_resolution}\n')
    md.append(f'- Direction at t* совпадает с финальным исходом: {moments["consistent"].sum()} ({100*moments["consistent"].mean():.1f}%)\n\n')
    md.append('### Распределение `phase_at_t_star`\n\n')
    md.append('```\n')
    md.append(moments['phase_t_soft'].describe().round(3).to_string())
    md.append('\n```\n\n')
    md.append('### Распределение `window_size` (20% длины партии)\n\n')
    md.append('```\n')
    md.append(moments['window_size'].describe().round(1).to_string())
    md.append('\n```\n\n')
    md.append('## Графики\n\n')
    md.append('- `resolution_safe_zones.png` — winrate vs own_prod_ratio с границами safe zones\n')
    md.append('- `resolution_distribution.png` — распределение phase_at_t_star и window_size\n\n')
    md.append('## Следующий шаг\n\n')
    md.append('`06_pre_resolution_geometry.py` — для каждой (episode, player) загрузить raw obs\n')
    md.append('для шагов в окне `[t_window_start, t_star]`, посчитать геометрические признаки\n')
    md.append('поля контроля (площадь, длина фронта, кривизна, race-pair flip count),\n')
    md.append('и сравнить их **изменения** между победителями и проигравшими.\n')
    (REPORT_DIR / 'resolution_moments.md').write_text(''.join(md))
    print(f'✓ Отчёт: {REPORT_DIR}/resolution_moments.md')


if __name__ == '__main__':
    main()
