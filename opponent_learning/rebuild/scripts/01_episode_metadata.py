#!/usr/bin/env python3
"""
01_episode_metadata.py — извлекает метаданные каждого эпизода в одну CSV.

ЦЕЛЬ
────
Существующий `data/processed/steps.csv` не содержит `n_players` — поэтому
1v1 и 4-player FFA партии смешиваются в обучении. Это пункт #4 в AUDIT.md.

Этот скрипт пробегает 500 JSON и собирает _только_ метаданные эпизода:
    episode_id, n_players, n_steps, rewards (как json), team_names, seed
Каждое последующее место в pipeline может смержить с этим по episode_id
и стратифицировать или фильтровать (только 1v1, например).

Парсить тяжёлый JSON не нужно — мы трогаем только верх и первый шаг.

ЛЁГКИЙ скрипт (~30 секунд на 500 эпизодов).

ВХОД
────
    opponent_learning/data/raw/*.json

ВЫХОД
─────
    opponent_learning/rebuild/data/episodes_meta.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE     = Path(__file__).resolve().parent
RAW_DIR  = HERE.parent.parent / "data" / "raw"
OUT_DIR  = HERE.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "episodes_meta.csv"


def extract_meta(json_path: Path) -> dict:
    """Читает один эпизод, возвращает словарь metadata. Никаких step-данных."""
    with open(json_path) as fh:
        ep = json.load(fh)

    info        = ep.get('info') or {}
    config      = ep.get('configuration') or {}
    steps       = ep.get('steps') or []
    rewards     = ep.get('rewards') or []
    statuses    = ep.get('statuses') or []
    team_names  = info.get('TeamNames') or []

    # n_players: число agents в первом шаге (более надёжно, чем info)
    n_players = len(steps[0]) if steps and isinstance(steps[0], list) else len(rewards)

    # outcome: индексы победителей (могут быть нескольких в случае ничьей)
    if rewards:
        max_r   = max(rewards)
        winners = [i for i, r in enumerate(rewards) if r == max_r]
    else:
        winners = []

    return {
        'episode_id':    json_path.stem,
        'n_players':     int(n_players),
        'n_steps':       len(steps),
        'rewards_json':  json.dumps(rewards),
        'statuses_json': json.dumps(statuses),
        'team_names_json': json.dumps(team_names),
        'winners_idx':   json.dumps(winners),
        'is_1v1':        n_players == 2,
        'is_ffa4':       n_players == 4,
        'is_draw':       len(winners) > 1,
        'seed':          config.get('seed'),
        'episode_steps_max': config.get('episodeSteps'),
        'shipSpeed':     config.get('shipSpeed'),
        'cometSpeed':    config.get('cometSpeed'),
    }


def main():
    files = sorted(p for p in RAW_DIR.glob('*.json')
                   if p.name != 'episodes_index.json')
    if not files:
        print(f'⚠ No episodes in {RAW_DIR}')
        sys.exit(1)

    print(f'Эпизодов: {len(files)}')
    t0 = time.time()
    records = []
    errors  = []
    for i, path in enumerate(files):
        try:
            records.append(extract_meta(path))
        except Exception as e:
            errors.append((path.name, repr(e)))
        if (i + 1) % 100 == 0:
            print(f'  обработано {i+1}/{len(files)}  ({time.time()-t0:.1f}s)')

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    dt = time.time() - t0

    print('-' * 60)
    print(f'Сохранено: {OUT_PATH}  ({len(df)} строк за {dt:.1f}s)')
    print(f'\n=== Сводка ===')
    print(f'  n_players distribution:')
    print(df['n_players'].value_counts().sort_index().to_string())
    print(f'\n  1v1 эпизодов:    {df["is_1v1"].sum()}')
    print(f'  FFA-4 эпизодов:  {df["is_ffa4"].sum()}')
    print(f'  ничьих:          {df["is_draw"].sum()}')
    print(f'  средняя длина:   {df["n_steps"].mean():.0f} шагов')
    print(f'  min/max steps:   {df["n_steps"].min()} / {df["n_steps"].max()}')

    # Распределение реальных исходов в 1v1
    df_1v1 = df[df['is_1v1']].copy()
    if len(df_1v1):
        # Парсим rewards: 1 элемент = индекс победителя (или -1 если ничья)
        df_1v1['win_p0'] = df_1v1['rewards_json'].apply(
            lambda s: json.loads(s)[0] > json.loads(s)[1]
            if not pd.isna(s) else None
        )
        print(f'\n  1v1: победа p0 в {df_1v1["win_p0"].sum()} '
              f'({100*df_1v1["win_p0"].mean():.1f}%) — should be ~50%')

    if errors:
        print(f'\n⚠ Ошибок: {len(errors)}')
        for name, err in errors[:5]:
            print(f'  {name}: {err[:100]}')


if __name__ == '__main__':
    main()
