#!/usr/bin/env python3
"""
coverage_diagram.py — визуализация покрытия по победителям.

Читает последний eval_8winners_*.csv, строит:
  1. Heatmap [9 кандидатов × 100 сидов] win/loss.
  2. Сверху bar: сколько кандидатов выигрывают каждый сид (0..9).
  3. Справа bar: всего побед на кандидата.
  4. В правом нижнем углу — текстовая сводка (oracle vs best single).

Сохраняет PNG в tuning/results/coverage_<ts>.png.

Запуск:  python3 tuning/scripts/coverage_diagram.py
"""

import os, sys, glob
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if csv_path is None:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_8winners_*.csv")))
        if not files:
            print("Нет eval_8winners_*.csv"); return
        csv_path = files[-1]
    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.set_index('name')

    # Парсим won_seeds → set per кандидата
    wins_by = {}
    all_seeds = set()
    for name, row in df.iterrows():
        s = str(row.get('won_seeds') or '')
        wins = set(int(x) for x in s.split(',') if x.strip())
        wins_by[name] = wins
        all_seeds |= wins

    # Диапазон сидов: 1..N где N = max известных (не только победных, всех протестированных)
    n_test = int(df['n'].iloc[0]) if 'n' in df.columns else max(all_seeds) if all_seeds else 100
    seeds_range = list(range(1, n_test + 1))

    # Сортируем кандидатов: default первым, потом по убыванию побед
    sorted_names = ['default'] + sorted(
        [n for n in df.index if n != 'default'],
        key=lambda n: -len(wins_by.get(n, set()))
    )
    sorted_names = [n for n in sorted_names if n in df.index]

    # Матрица [n_cand × n_seeds], 1 если выиграл
    M = np.zeros((len(sorted_names), len(seeds_range)), dtype=int)
    for i, name in enumerate(sorted_names):
        for j, s in enumerate(seeds_range):
            if s in wins_by.get(name, set()):
                M[i, j] = 1

    # Сортируем сиды по тому, сколько кандидатов их выиграло (oracle-friendly view):
    # слева — «лёгкие» (все выигрывают), справа — «трудные» (никто).
    seed_difficulty = M.sum(axis=0)   # 0..len(sorted_names)
    order = np.argsort(-seed_difficulty)   # desc: лёгкие слева
    M_sorted = M[:, order]
    seeds_sorted = [seeds_range[i] for i in order]

    # Oracle stats
    union_won = set().union(*wins_by.values())
    best_single = max((len(s), n) for n, s in wins_by.items())
    print(f"oracle потенциал: {len(union_won)}/{n_test} = {len(union_won)/n_test*100:.1f}%")
    print(f"best single:      {best_single[0]}/{n_test} ({best_single[1]})")
    print(f"never_won:        {n_test - len(union_won)} сидов где НИКТО из 9 не выиграл")

    # ── рисуем ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 2, height_ratios=[1, 5], width_ratios=[20, 2],
                  hspace=0.05, wspace=0.04)

    # Top: bar по сидам
    ax_top = fig.add_subplot(gs[0, 0])
    coverage = M_sorted.sum(axis=0)
    colors_top = ['#2eccaa' if c == len(sorted_names) else
                  '#4a90d9' if c >= len(sorted_names)//2 else
                  '#f0b04a' if c >= 1 else
                  '#e05c3a' for c in coverage]
    ax_top.bar(range(len(seeds_sorted)), coverage, color=colors_top, width=1.0)
    ax_top.set_xlim(-0.5, len(seeds_sorted) - 0.5)
    ax_top.set_ylim(0, len(sorted_names) + 0.5)
    ax_top.set_ylabel('# winners', fontsize=8)
    ax_top.set_xticks([])
    ax_top.set_yticks([0, len(sorted_names)//2, len(sorted_names)])
    ax_top.tick_params(labelsize=7)
    ax_top.set_title(f"Coverage: {len(union_won)}/{n_test} oracle  |  "
                     f"best single = {best_single[0]} ({best_single[1]})  |  "
                     f"never won = {n_test - len(union_won)}",
                     fontsize=10)

    # Heatmap
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cmap = plt.matplotlib.colors.ListedColormap(['#1a1a1a', '#2eccaa'])
    ax_main.imshow(M_sorted, aspect='auto', cmap=cmap, interpolation='nearest')
    ax_main.set_yticks(range(len(sorted_names)))
    ax_main.set_yticklabels(sorted_names, fontsize=8)
    # Подписи сидов: показываем каждый 10-й
    n_ticks = 10
    tick_idx = np.linspace(0, len(seeds_sorted) - 1, n_ticks, dtype=int)
    ax_main.set_xticks(tick_idx)
    ax_main.set_xticklabels([seeds_sorted[i] for i in tick_idx], fontsize=7, rotation=90)
    ax_main.set_xlabel(f'seeds (отсортированы по сложности: лёгкие слева, трудные справа)',
                       fontsize=8)

    # Right: bar по кандидатам
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    wins_per = M_sorted.sum(axis=1)
    bar_colors = ['#888' if n == 'default' else
                  ('#2eccaa' if w >= np.median(wins_per) else '#4a90d9')
                  for n, w in zip(sorted_names, wins_per)]
    ax_right.barh(range(len(sorted_names)), wins_per, color=bar_colors)
    for i, w in enumerate(wins_per):
        ax_right.text(w + 0.5, i, str(int(w)), va='center', fontsize=7)
    ax_right.set_xlim(0, max(wins_per) * 1.15)
    ax_right.set_yticks([])
    ax_right.set_xlabel('wins', fontsize=8)
    ax_right.tick_params(labelsize=7)
    ax_right.invert_yaxis()
    ax_main.invert_yaxis()

    plt.suptitle(f"Coverage диаграмма — {os.path.basename(csv_path)}",
                 fontsize=11, y=0.995)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_png = os.path.join(RESULTS_DIR, f"coverage_{ts}.png")
    plt.savefig(out_png, dpi=130, bbox_inches='tight', facecolor='white')
    print(f"saved: {out_png}")

    # ── текстовая выкладка по уникальности ──────────────────────────────
    print("\n=== Уникальность побед (выигрывают только этот кандидат) ===")
    union_others = {}
    for n, s in wins_by.items():
        others = set().union(*[w for nn, w in wins_by.items() if nn != n])
        unique = s - others
        union_others[n] = len(unique)
    for n in sorted_names:
        print(f"  {n:<16}  total={len(wins_by[n]):>3}  unique={union_others.get(n, 0):>3}")

    plt.close()


if __name__ == "__main__":
    main()
