#!/usr/bin/env python3
"""
30_via_kaggle_cli.py — использует kaggle CLI для обхода API проблем.

ЛОГИКА
──────
kaggle CLI работает с KAGGLE_API_TOKEN. Он может делать "kaggle api ..."
команды которые проксируют любой Kaggle API endpoint. Пробуем разные
варианты получить submission IDs для teamIDs из CSV.

Если CLI помогает — построим episodes_index и скачаем стандартным путём.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REBUILD = HERE.parent
OL_DIR = REBUILD.parent
DATA_DIR = OL_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REBUILD_DATA = REBUILD / "data"


def find_csv() -> Path | None:
    cands = list(REBUILD_DATA.glob('*publicleaderboard*.csv'))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def kaggle_cli_call(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Запускает kaggle CLI. Возвращает (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(['kaggle'] + args, capture_output=True,
                            text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, '', 'kaggle CLI не установлен'
    except subprocess.TimeoutExpired:
        return -1, '', 'timeout'


def try_cli_methods(team_id: int) -> list[int]:
    """Пробуем разные kaggle CLI команды для получения submissions."""
    print(f'\n  team_id={team_id}: пробую CLI методы …')

    # Метод 1: api команда напрямую (proxy через kaggle CLI)
    methods = [
        ['api', 'list-submissions', 'orbit-wars', f'--team-id', str(team_id)],
        ['api', 'get-submissions', 'orbit-wars', f'--team-id', str(team_id)],
        ['competitions', 'submissions', 'orbit-wars', '--team-id', str(team_id)],
        ['competitions', 'submissions', 'orbit-wars'],
    ]
    for args in methods:
        rc, out, err = kaggle_cli_call(args)
        out_preview = out[:200] if out else ''
        err_preview = err[:200] if err else ''
        print(f'    {" ".join(args)}: rc={rc}')
        if out_preview:
            print(f'      stdout: {out_preview}')
        if err_preview and rc != 0:
            print(f'      stderr: {err_preview}')
        if rc == 0 and out:
            # Парсим numbers из output
            import re
            ids = re.findall(r'\b\d{8,10}\b', out)
            if ids:
                ids = [int(i) for i in ids if 10_000_000 < int(i) < 99_999_999]
                if ids:
                    return list(set(ids))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-n', type=int, default=5,
                    help='Сколько команд из CSV проверить')
    args = ap.parse_args()

    # Проверим что CLI работает
    print('Проверяю kaggle CLI …')
    rc, out, err = kaggle_cli_call(['--version'])
    if rc != 0:
        sys.exit(f'⚠ kaggle CLI не работает: {err}')
    print(f'  ✓ {out.strip()}')

    # Helper команда: что доступно?
    print('\nДоступные subcommands:')
    rc, out, err = kaggle_cli_call(['competitions', '--help'])
    print(out[:1500] if out else err[:500])

    # CSV
    csv_path = find_csv()
    if not csv_path:
        sys.exit('⚠ CSV leaderboard не найден')

    teams = []
    with open(csv_path, encoding='utf-8-sig') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= args.top_n:
                break
            teams.append({'rank': int(r['Rank']),
                           'team_id': int(r['TeamId']),
                           'team_name': r['TeamName']})

    print(f'\nТестируем CLI на топ-{len(teams)} команд:')
    for t in teams:
        ids = try_cli_methods(t['team_id'])
        print(f'  #{t["rank"]:>2} {t["team_name"]:<30} → {len(ids)} subs '
              f'{ids[:5] if ids else ""}')

    print(f'\n💡 Если все 0 — CLI не предоставляет доступ к чужим submissions.')
    print(f'   Это RESTRICTED API. Нужно либо:')
    print(f'   1. Использовать сайт kaggle.com вручную для каждой команды')
    print(f'   2. Запустить старый 01_download_episodes.py (DEFAULT submission IDs)')
    print(f'   3. Сфокусироваться на анализе уже скачанных данных')


if __name__ == '__main__':
    main()
