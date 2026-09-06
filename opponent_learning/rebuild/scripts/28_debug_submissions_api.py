#!/usr/bin/env python3
"""
28_debug_submissions_api.py — verbose diagnostic для ListSubmissions API.

ЦЕЛЬ
────
27_* возвращал 0 subs для всех команд. Нужно понять что именно отвечает Kaggle:
  • status code?
  • тело ответа (JSON, HTML, empty)?
  • правильный ли body нашего запроса?
  • работает ли endpoint вообще?

ЛОГИКА
──────
1. Берём teamId=15654628 (Isaiah @ Tufa Labs #1)
2. Пробуем разные API endpoints и body variants
3. Для каждого printим: URL, body, status, headers, response (первые 2000 байт)
4. Также пробуем HTML scrape страницы команды

Запуск:
  python3 28_debug_submissions_api.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REBUILD = HERE.parent
DATA_DIR = REBUILD / "data"

TEST_TEAM_ID = 15654628  # Isaiah @ Tufa Labs

# Тестируемые endpoints (URL, method, body_factory)
ENDPOINTS = [
    # ListSubmissions варианты body
    ('https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions',
     'POST', lambda tid: {'teamId': tid}),
    ('https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions',
     'POST', lambda tid: {'teamId': tid, 'competitionId': 86535}),
    ('https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions',
     'POST', lambda tid: {'teamId': tid, 'competitionName': 'orbit-wars'}),
    ('https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions',
     'POST', lambda tid: {'TeamId': tid}),
    # Альтернативные API
    ('https://www.kaggle.com/api/v1/competitions/submissions/list',
     'POST', lambda tid: {'teamId': tid}),
    ('https://www.kaggle.com/api/i/competitions.TeamApiService/GetTeam',
     'POST', lambda tid: {'teamId': tid}),
    ('https://www.kaggle.com/api/i/competitions.TeamApiService/GetTeamSubmissions',
     'POST', lambda tid: {'teamId': tid}),
    # HTML страница команды (без API)
    ('https://www.kaggle.com/competitions/orbit-wars/team/{tid}',
     'GET', None),
    ('https://www.kaggle.com/c/orbit-wars/team/{tid}',
     'GET', None),
]


def make_session(username: str, api_key: str) -> requests.Session:
    s = requests.Session()
    if api_key.startswith('KGAT_') or api_key.startswith('kkk_'):
        s.headers.update({'Authorization': f'Bearer {api_key}'})
    else:
        s.auth = (username, api_key)
    s.headers['User-Agent'] = 'Kaggle/Python'
    return s


def try_endpoint(session, url_tmpl, method, body_factory, team_id):
    """Вызывает endpoint, возвращает (status, body_text, headers_snapshot)."""
    url = url_tmpl.replace('{tid}', str(team_id))
    body = body_factory(team_id) if body_factory else None

    print(f'\n{"─" * 70}')
    print(f'{method} {url}')
    if body:
        print(f'body: {json.dumps(body)}')

    try:
        if method == 'POST':
            r = session.post(url, json=body,
                              headers={'Content-Type': 'application/json',
                                       'Accept': 'application/json'},
                              timeout=30)
        else:
            r = session.get(url, timeout=30)
    except Exception as e:
        print(f'  EXCEPTION: {e!r}')
        return None, str(e), {}

    print(f'status: {r.status_code}')
    # Печатаем важные headers
    important = ['Content-Type', 'Server', 'X-Frame-Options', 'WWW-Authenticate']
    for h in important:
        if h in r.headers:
            print(f'  header {h}: {r.headers[h]}')

    body_text = r.text
    # Если это JSON — pretty
    try:
        as_json = r.json()
        body_text = json.dumps(as_json, indent=2)[:3000]
        print(f'\n=== JSON response (first 3000 chars) ===')
        print(body_text)
        # Особенно нас интересует поле 'submissions'
        if isinstance(as_json, dict):
            subs = as_json.get('submissions') or as_json.get('Submissions')
            if subs:
                print(f'\n  ★ submissions field: len={len(subs)}')
                if subs:
                    print(f'    first sub keys: {list(subs[0].keys())[:15]}')
                    print(f'    first sub: {json.dumps(subs[0], indent=2)[:500]}')
    except json.JSONDecodeError:
        # HTML / plain text
        print(f'\n=== TEXT response (first 1500 chars) ===')
        print(body_text[:1500])
        # Ищем submission IDs в HTML регулярками
        ids = re.findall(r'submissionId["\':\s]+(\d{8,12})', body_text)
        ids += re.findall(r'/submissions/(\d{8,12})', body_text)
        ids += re.findall(r'"id":\s*(\d{8,12})', body_text)
        ids = list(set(ids))
        if ids:
            print(f'\n  ★ Найдены ID-кандидаты в тексте: {ids[:10]}')

    return r.status_code, body_text, dict(r.headers)


def main():
    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = os.environ.get('KAGGLE_KEY', '')
    if not username or not api_key:
        sys.exit('Установи KAGGLE_USERNAME и KAGGLE_KEY')

    print(f'KAGGLE_USERNAME: {username}')
    print(f'KAGGLE_KEY: {api_key[:8]}... ({len(api_key)} chars)  '
          f'starts_with_KGAT={api_key.startswith("KGAT_")}')
    print(f'\nТестируем на teamId={TEST_TEAM_ID} (Isaiah @ Tufa Labs #1)')

    session = make_session(username, api_key)

    results = []
    for url, method, body_factory in ENDPOINTS:
        status, body, headers = try_endpoint(session, url, method,
                                              body_factory, TEST_TEAM_ID)
        results.append({
            'url': url, 'method': method,
            'status': status,
            'body_preview': body[:1500] if body else '',
            'content_type': headers.get('Content-Type', '?'),
        })

    # Сохраняем для анализа
    out = DATA_DIR / 'submissions_api_debug.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False)[:200000])
    print(f'\n{"═" * 70}')
    print(f'✓ Все результаты сохранены: {out}')
    print(f'\nИтог:')
    for r in results:
        emoji = '✓' if r['status'] == 200 else '✗'
        print(f'  {emoji} {r["method"]:<4} status={r["status"]}  '
              f'ct={r["content_type"][:30]}  '
              f'{r["url"][:60]}')

    # Подсказка
    print(f'\n💡 Если ВСЕ endpoints возвращают 404/401:')
    print(f'   Kaggle ограничил доступ к этому API из-за нового token type.')
    print(f'   Альтернативы:')
    print(f'   1. Использовать классический kaggle.json (не KGAT_)')
    print(f'   2. Скопировать submission IDs со страницы команды вручную')
    print(f'   3. Использовать только уже скачанные эпизоды')


if __name__ == '__main__':
    main()
