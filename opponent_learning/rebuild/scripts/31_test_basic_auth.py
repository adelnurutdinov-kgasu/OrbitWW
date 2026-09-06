#!/usr/bin/env python3
"""
31_test_basic_auth.py — тестирует auth как в оригинальном notebook.

Notebook автора: `requests.post(url, auth=(username, api_key), ...)` — Basic auth.
Наши скрипты делают Bearer auth для KGAT_ token. Возможно это и есть причина
почему ListSubmissions API возвращало пустоту.

Тестируем оба варианта на одном submission ID который ТОЧНО работал в notebook
(bowwowforeach=52318886) и сравниваем.
"""

from __future__ import annotations

import json
import os
import sys

import requests


def main():
    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = (os.environ.get('KAGGLE_KEY', '') or
               os.environ.get('KAGGLE_API_TOKEN', ''))
    if not username or not api_key:
        sys.exit('⚠ KAGGLE_USERNAME + (KAGGLE_KEY или KAGGLE_API_TOKEN)')

    print(f'USERNAME: {username}')
    print(f'API_KEY:  {api_key[:8]}... ({len(api_key)} chars)')
    print(f'  startswith KGAT_: {api_key.startswith("KGAT_")}\n')

    SUB_ID = 52318886  # bowwowforeach из notebook
    url = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
    body = {"submissionId": SUB_ID}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # ── Тест 1: Basic auth (как в notebook) ──
    print('═══ Тест 1: Basic auth (username, key) ═══')
    r = requests.post(url, auth=(username, api_key), json=body,
                       headers=headers, timeout=30)
    print(f'  status: {r.status_code}')
    print(f'  body (first 500): {r.text[:500]}')
    try:
        d = r.json()
        eps = d.get('episodes', [])
        print(f'  episodes: {len(eps)}')
        if eps:
            print(f'  first episode: {eps[0]}')
    except Exception:
        pass

    print()
    # ── Тест 2: Bearer auth (наш текущий подход) ──
    print('═══ Тест 2: Bearer auth ═══')
    r = requests.post(url, json=body,
                       headers={**headers, 'Authorization': f'Bearer {api_key}'},
                       timeout=30)
    print(f'  status: {r.status_code}')
    print(f'  body (first 500): {r.text[:500]}')
    try:
        d = r.json()
        eps = d.get('episodes', [])
        print(f'  episodes: {len(eps)}')
    except Exception:
        pass

    # ── Тест 3: Basic auth для ListSubmissions ──
    print('\n═══ Тест 3: Basic auth для ListSubmissions (другой API) ═══')
    sub_url = "https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions"
    r = requests.post(sub_url, auth=(username, api_key),
                       json={"teamId": 15654628},
                       headers=headers, timeout=30)
    print(f'  status: {r.status_code}')
    print(f'  body (first 800): {r.text[:800]}')

    print('\n💡 Если Тест 1 == 200 с episodes — Basic auth работает с KGAT_ token!')
    print('   Тогда поправим все наши скрипты на Basic auth и повторим попытки.')


if __name__ == '__main__':
    main()
