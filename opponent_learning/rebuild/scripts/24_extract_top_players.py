#!/usr/bin/env python3
"""
24_extract_top_players.py — извлекает топ-игроков из уже скачанных replay JSON.

ЛОГИКА
──────
Каждый replay JSON содержит `info.TeamNames` — имена двух команд (и для FFA
ещё имена 3-4 команд). И `rewards` — кто победил.

Проходим по всем replay JSON, для каждой команды считаем:
  • n_games — в скольких партиях она участвовала
  • n_wins — сколько раз победила
  • winrate — n_wins / n_games

Топ-N по winrate (с минимум N_MIN_GAMES для статистической надёжности) даёт
нам реальных топ-игроков среди тех чьи эпизоды у нас уже есть.

ВЫХОД
─────
  rebuild/data/top_players.csv — все команды с метриками
  rebuild/reports/top_players_report.md — топ-30 + распределение
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
RAW_DIR    = OL_DIR / "data" / "raw"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def extract_one(json_path: Path) -> list[dict]:
    try:
        with open(json_path) as f:
            ep = json.load(f)
    except Exception:
        return []

    info = ep.get('info') or {}
    team_names = info.get('TeamNames') or []
    rewards = ep.get('rewards') or []

    if not team_names or not rewards:
        return []

    # winners: индексы тех у кого max reward
    if not all(r is not None for r in rewards):
        return []
    try:
        max_r = max(rewards)
        winners = [i for i, r in enumerate(rewards) if r == max_r]
    except (TypeError, ValueError):
        return []

    n_players = len(team_names)
    out = []
    for i, name in enumerate(team_names):
        if not name:
            continue
        out.append({
            'team_name':   name,
            'episode_id':  json_path.stem,
            'won':         int(i in winners),
            'n_players':   n_players,
            'reward':      rewards[i] if i < len(rewards) else 0,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-games', type=int, default=10,
                    help='Минимум партий чтобы команда вошла в топ')
    ap.add_argument('--top-n', type=int, default=30)
    args = ap.parse_args()

    files = sorted(p for p in RAW_DIR.glob('*.json')
                   if p.name != 'episodes_index.json'
                   and p.stem != 'current_leaderboard'
                   and p.stem != 'leaderboard_raw'
                   and p.stem.isdigit())
    if not files:
        sys.exit(f'⚠ Нет replay JSON в {RAW_DIR}')

    print(f'Анализирую {len(files)} replay JSON …')
    t0 = time.time()
    records = []
    for i, fp in enumerate(files):
        records.extend(extract_one(fp))
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(files)} ({time.time()-t0:.0f}s, {len(records)} records)')

    df = pd.DataFrame(records)
    print(f'\nВсего team-participations: {len(df)}')

    # Группируем по команде
    grp = df.groupby('team_name').agg(
        n_games=('won', 'count'),
        n_wins=('won', 'sum'),
        avg_reward=('reward', 'mean'),
    ).reset_index()
    grp['winrate'] = grp['n_wins'] / grp['n_games']

    # Топ по winrate (с фильтром min_games)
    qualified = grp[grp['n_games'] >= args.min_games].copy()
    qualified = qualified.sort_values(['winrate', 'n_games'], ascending=[False, False])

    print(f'\nКоманд всего: {len(grp)}')
    print(f'Команд с ≥ {args.min_games} играми: {len(qualified)}')

    print(f'\n══ Топ-{args.top_n} по winrate (min {args.min_games} игр) ══')
    print(f'  {"#":<4}{"name":<35}  {"games":>6}  {"wins":>5}  {"winrate":>8}')
    for i, (_, r) in enumerate(qualified.head(args.top_n).iterrows()):
        name = r['team_name'][:32]
        print(f'  {i+1:<4}{name:<35}  {r["n_games"]:>6}  {r["n_wins"]:>5}  {r["winrate"]:>8.3f}')

    # Сохранение
    grp.to_csv(DATA_DIR / 'top_players.csv', index=False)
    print(f'\n✓ CSV: {DATA_DIR / "top_players.csv"}')

    # Markdown
    md = []
    md.append('# Топ игроков из скачанных реплеев\n\n')
    md.append(f'**Анализировано**: {len(files)} эпизодов\n')
    md.append(f'**Команд найдено**: {len(grp)}\n')
    md.append(f'**С ≥ {args.min_games} играми**: {len(qualified)}\n\n')

    md.append(f'## Топ-{args.top_n} по winrate\n\n')
    md.append('| # | name | games | wins | winrate |\n|---|---|---|---|---|\n')
    for i, (_, r) in enumerate(qualified.head(args.top_n).iterrows()):
        md.append(f'| {i+1} | {r["team_name"]} | {r["n_games"]} | '
                  f'{r["n_wins"]} | {r["winrate"]:.3f} |\n')

    md.append(f'\n## Топ-{args.top_n} по количеству игр\n\n')
    by_games = grp.sort_values('n_games', ascending=False).head(args.top_n)
    md.append('| name | games | wins | winrate |\n|---|---|---|---|\n')
    for _, r in by_games.iterrows():
        md.append(f'| {r["team_name"]} | {r["n_games"]} | '
                  f'{r["n_wins"]} | {r["winrate"]:.3f} |\n')

    md.append('\n## Как использовать дальше\n\n')
    md.append('Если хочешь скачивать эпизоды от ИМЕННО этих игроков:\n')
    md.append('1. Найди их submission IDs вручную на странице соревнования\n')
    md.append('2. Передай в 22b_download_v2.py через --submission-ids\n\n')
    md.append('Однако наша текущая загрузка (22b с DEFAULT_TOP_SUBMISSION_IDS)\n')
    md.append('уже захватывает большинство топов — у них matches накапливаются у нас\n')
    md.append('в реплеях натурально (топ-1 играет в основном с другими топами).\n')

    (REPORT_DIR / 'top_players_report.md').write_text(''.join(md))
    print(f'✓ Отчёт: {REPORT_DIR / "top_players_report.md"}')


if __name__ == '__main__':
    main()
