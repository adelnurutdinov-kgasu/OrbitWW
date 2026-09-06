#!/usr/bin/env python3
"""
34_find_sub_ids.py — finding submission IDs for top players (3 paths).

ПОДХОДЫ
───────
1. kaggle CLI: `kaggle competitions submissions orbit-wars --help` — есть ли
   флаг для запроса чужих submissions?
2. HTTP GET страницы команды с auth: парсим HTML на submission IDs
3. HTTP GET профиля пользователя

Идём от простого. Если найдём способ — пробуем на одном пользователе
(pressman1 = Isaiah @ Tufa Labs #1).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import requests


def main():
    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = (os.environ.get('KAGGLE_KEY', '') or
                os.environ.get('KAGGLE_API_TOKEN', ''))
    if not username or not api_key:
        sys.exit('Set KAGGLE_USERNAME + KAGGLE_KEY/KAGGLE_API_TOKEN')

    # === ПУТЬ 1: kaggle CLI help для submissions ===
    print('═' * 70)
    print('ПУТЬ 1: что есть в kaggle competitions submissions')
    print('═' * 70)
    try:
        r = subprocess.run(['kaggle', 'competitions', 'submissions', '--help'],
                            capture_output=True, text=True, timeout=15)
        print(r.stdout)
        if r.stderr:
            print('STDERR:', r.stderr[:500])
    except Exception as e:
        print(f'CLI error: {e!r}')

    # === ПУТЬ 2: HTML страница команды ===
    print('\n' + '═' * 70)
    print('ПУТЬ 2: HTML страница команды Isaiah (teamId=15654628)')
    print('═' * 70)

    s = requests.Session()
    s.auth = (username, api_key)
    s.headers['User-Agent'] = 'Mozilla/5.0 Chrome/120'

    urls = [
        'https://www.kaggle.com/competitions/orbit-wars/team/15654628',
        'https://www.kaggle.com/c/orbit-wars/team/15654628',
        'https://www.kaggle.com/competitions/orbit-wars/leaderboard?team=15654628',
    ]
    for url in urls:
        print(f'\n  GET {url}')
        try:
            r = s.get(url, timeout=20, allow_redirects=True)
            print(f'    status: {r.status_code}, length: {len(r.text)}, final_url: {r.url}')
            if r.status_code != 200:
                print(f'    body (200 chars): {r.text[:200]}')
                continue
            # Ищем submission IDs в разных pattern'ах
            patterns = {
                '/submissions/NNN':      re.findall(r'/submissions/(\d{7,10})', r.text),
                'submissionId":NNN':     re.findall(r'"submissionId"\s*:\s*(\d{7,10})', r.text),
                'subId=NNN':             re.findall(r'subId=(\d{7,10})', r.text),
                'submissionId=NNN':      re.findall(r'submissionId=(\d{7,10})', r.text),
                '"id":NNN(near submit)': re.findall(r'submit\w{0,30}?"id"\s*:\s*(\d{7,10})', r.text),
            }
            any_found = False
            for name, ids in patterns.items():
                ids = list(set(int(i) for i in ids if 10_000_000 < int(i) < 99_999_999))
                if ids:
                    print(f'    ★ pattern "{name}": {ids[:10]}')
                    any_found = True
            if not any_found:
                # Распечатать первые JSON блоки
                jsons = re.findall(r'(?s)\{[^{]{300,5000}?\}', r.text[:200000])
                print(f'    Нет паттернов. JSON-блоков в page: {len(jsons)}')
                if jsons:
                    print(f'    First JSON sample (500 chars): {jsons[0][:500]}')
                # Также — поиск любого упоминания "submission" в тексте
                idx = r.text.lower().find('submission')
                if idx >= 0:
                    print(f'    Mentions of "submission" at offset {idx}:')
                    print(f'    {r.text[max(0,idx-50):idx+300]}')
        except Exception as e:
            print(f'    EXC: {e!r}')

    # === ПУТЬ 3: профиль пользователя ===
    print('\n' + '═' * 70)
    print('ПУТЬ 3: профиль pressman1')
    print('═' * 70)
    for url in [
        'https://www.kaggle.com/pressman1',
        'https://www.kaggle.com/pressman1/competitions',
    ]:
        print(f'\n  GET {url}')
        try:
            r = s.get(url, timeout=20)
            print(f'    status: {r.status_code}, length: {len(r.text)}')
            if r.status_code == 200:
                ids = list(set(int(m) for m in re.findall(r'/submissions/(\d{7,10})', r.text)
                                if 10_000_000 < int(m) < 99_999_999))
                if ids:
                    print(f'    ★ найдено submission IDs: {ids[:10]}')
        except Exception as e:
            print(f'    EXC: {e!r}')


if __name__ == '__main__':
    main()
