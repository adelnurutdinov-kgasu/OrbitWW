#!/usr/bin/env python3
"""
visualize_tournament.py — строит тепловую карту и графики по результатам кругового турнира.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ====== НАСТРОЙКИ ======
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = PROJECT_ROOT / "tuning" / "results" / "tournament_20260511_091806.csv"  # замените на актуальный файл
SAVE_FIGURES = True
OUTPUT_DIR = Path(".")  # куда сохранить графики
# =======================

def load_and_compute_pairs(df):
    """Строит матрицы:
       - win_rate[i,j] = доля побед i над j (только матчи i vs j, где i<j)
       - score_diff[i,j] = средняя разница кораблей (ships_diff) в матчах i против j
       Плюс итоговые очки.
    """
    # Уникальные имена кандидатов
    candidates = sorted(set(df['candidate_a_name'].unique()) | set(df['candidate_b_name'].unique()))
    n = len(candidates)
    win_mat = np.zeros((n, n))
    diff_mat = np.zeros((n, n))
    counts_mat = np.zeros((n, n), dtype=int)

    # Заполняем
    for _, row in df.iterrows():
        a = row['candidate_a_name']
        b = row['candidate_b_name']
        i = candidates.index(a)
        j = candidates.index(b)
        win_a = row['win_a']
        win_b = row['win_b']
        draw = row['draw']
        diff = row['ships_diff']
        # Победы i над j
        if win_a:
            win_mat[i, j] += 1
            # По умолчанию win_mat[j,i] хранит победы j над i (заполним симметрично)
            # Но для тепловой карты удобнее win_rate[i][j] = доля побед i над j
        elif win_b:
            win_mat[j, i] += 1
        # Если ничья - никому не добавляем
        # Считаем количество матчей
        counts_mat[i, j] += 1
        counts_mat[j, i] += 1
        diff_mat[i, j] += diff      # положительный diff означает преимущество i (если a=i)
        diff_mat[j, i] += -diff     # для j diff будет противоположный

    # Превратим в проценты
    win_rate = np.zeros((n, n))
    avg_diff = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and counts_mat[i, j] > 0:
                win_rate[i, j] = win_mat[i, j] / counts_mat[i, j] * 100
                avg_diff[i, j] = diff_mat[i, j] / counts_mat[i, j]

    return candidates, win_rate, avg_diff, counts_mat

def plot_heatmap(mat, candidates, title, cmap='RdYlGn', center=50, fmt='.1f', cbar_label='% побед'):
    """Рисует тепловую карту матрицы."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, center=center,
                xticklabels=candidates, yticklabels=candidates,
                square=True, cbar_kws={'label': cbar_label})
    plt.title(title)
    plt.ylabel('Агент A (победы над B)')
    plt.xlabel('Агент B')
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(OUTPUT_DIR / f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

def plot_ranking(df, candidates):
    """Строит bar chart суммарных очков."""
    # Подсчёт очков: суммируем points_a по каждому кандидату в роли a и points_b в роли b
    points = {}
    for cand in candidates:
        total = df[df['candidate_a_name'] == cand]['points_a'].sum() + \
                df[df['candidate_b_name'] == cand]['points_b'].sum()
        points[cand] = total
    sorted_cands = sorted(points.items(), key=lambda x: -x[1])
    names, scores = zip(*sorted_cands)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, scores, color='steelblue')
    plt.title('Итоговое количество очков (победа=1, ничья=0.5)')
    plt.ylabel('Очки')
    plt.xticks(rotation=45, ha='right')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{score:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(OUTPUT_DIR / "ranking.png", dpi=150)
    plt.show()

def main():
    # Загрузка CSV
    df = pd.read_csv(CSV_FILE)
    print(f"Загружено {len(df)} матчей")
    print(df.head())

    # Вычисление матриц
    candidates, win_rate, avg_diff, counts = load_and_compute_pairs(df)
    print("Кандидаты:", candidates)

    # Тепловая карта процентов побед
    plot_heatmap(win_rate, candidates,
                 title="Турнир 9x9: процент побед (строка побеждает столбец)",
                 cmap='RdYlGn', center=50, cbar_label='Победы A над B (%)')

    # Тепловая карта средней разницы кораблей (ships_diff)
    # Для наглядности закрасим от отрицательной (проигрыш) к положительной (выигрыш)
    plot_heatmap(avg_diff, candidates,
                 title="Средняя разница кораблей (ships_A - ships_B)",
                 cmap='coolwarm', center=0, fmt='.1f', cbar_label='Δ ships')

    # Итоговый рейтинг
    plot_ranking(df, candidates)

    # Дополнительно: можно вывести таблицу win_rate в консоль
    print("\nМатрица побед в процентах:")
    win_df = pd.DataFrame(win_rate, index=candidates, columns=candidates)
    print(win_df.round(1))

    print("\nМатрица средней разницы кораблей:")
    diff_df = pd.DataFrame(avg_diff, index=candidates, columns=candidates)
    print(diff_df.round(1))

if __name__ == "__main__":
    main()