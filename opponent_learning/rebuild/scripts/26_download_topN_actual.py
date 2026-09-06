#!/usr/bin/env python3
"""
26_download_topN_actual.py — попытка автоматизированно получить submission IDs
для топ-40 актуальных игроков и скачать их эпизоды.

ИДЕЯ
────
1. Пробуем несколько Kaggle API endpoints для получения leaderboard или
   поиска команд по имени
2. Если хотя бы один endpoint работает с нашим auth — получаем teamIDs →
   submission IDs → episode IDs → скачиваем
3. Если ВСЕ endpoints 404/auth fail — даём пользователю browser-инструкцию

ВЫЗОВ
─────
  export KAGGLE_USERNAME=... KAGGLE_KEY=...
  python3 26_download_topN_actual.py --max-total 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REBUILD = HERE.parent
OL_DIR = REBUILD.parent
DATA_DIR = OL_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Топ-40 актуальных
TOP_40_NAMES = [
    'Isaiah @ Tufa Labs', 'Jake Will', 'TonyK', 'Hober Malloc',
    'Felix M Neumann', 'flg', 'Audun Ljone Henriksen', 'Boey', 'Ender',
    'Xiangyu Liu', 'Luca', 'One Man Wrecking Machine', 'M & J & M.ver2',
    'Azat Akhtyamov', 'moriiiiiiiiim', 'Gregor Lied', 'Vadasz & Ascalon',
    'dragon warrior', '213tubo', 'Luke Li', 'Slawek Biel', 'Orbit Goblins',
    'Scool', 'LuckyXC&me', 'Yuki Okumura', 'Sheeesh---', 'Roche Overflow',
    'Artem', 'jonathan breitgand', 'kosmostars @ pseudolabeling',
    'Alan C52', 'Tufaben', 'This waiting will kill me!', 'bowwowforeach',
    'vkhydras', 'skalermo', 'typeIIIfairy', 'XtraLearning', 'Ebi', 'ma name',
]

# Endpoints к попытке
ENDPOINTS_TO_TRY = [
    # API leaderboard endpoints
    ('POST', 'https://www.kaggle.com/api/i/competitions.LeaderboardApiService/GetLeaderboard',
     {'competitionId': 86535}),
    ('POST', 'https://www.kaggle.com/api/i/competitions.LeaderboardApiService/GetLeaderboard',
     {'competitionName': 'orbit-wars'}),
    ('GET', 'https://www.kaggle.com/api/v1/competitions/orbit-wars/leaderboard', None),
    ('GET', 'https://www.kaggle.com/api/v1/competitions/leaderboards/orbit-wars', None),
    # HTML страница leaderboard (часто там JSON в JSON-LD или data-attribute)
    ('GET', 'https://www.kaggle.com/competitions/orbit-wars/leaderboard.json', None),
]


def make_session(username: str, api_key: str) -> requests.Session:
    s = requests.Session()
    if api_key.startswith('KGAT_') or api_key.startswith('kkk_'):
        s.headers.update({'Authorization': f'Bearer {api_key}'})
    else:
        s.auth = (username, api_key)
    s.headers['User-Agent'] = 'Kaggle/Python'
    return s


def try_get_leaderboard(session: requests.Session) -> dict | None:
    """Перебираем endpoints, возвращаем первый valid JSON."""
    print('\nПробуем endpoints для leaderboard …')
    for method, url, body in ENDPOINTS_TO_TRY:
        try:
            if method == 'POST':
                r = session.post(url, json=body,
                                 headers={'Content-Type': 'application/json',
                                          'Accept': 'application/json'},
                                 timeout=20)
            else:
                r = session.get(url, timeout=20)
            print(f'  {method} {url[:70]}…  → {r.status_code}')
            if r.status_code == 200:
                try:
                    data = r.json()
                    print(f'    ✓ JSON получен, keys: {list(data.keys())[:10] if isinstance(data, dict) else "list"}')
                    return data
                except json.JSONDecodeError:
                    print(f'    ⚠ не JSON ({r.headers.get("Content-Type", "?")})')
        except Exception as e:
            print(f'  {method} {url[:70]}… → exception {e!r}')
    return None


def extract_team_info(data: dict | list) -> list[dict]:
    """Из leaderboard data достаём (teamName, teamId, submissionId)."""
    results = []
    if isinstance(data, dict):
        # Возможные ключи где лежат rows
        for k in ('submissions', 'results', 'leaderboard', 'entries',
                   'teams', 'topUsers'):
            if k in data and isinstance(data[k], list):
                rows = data[k]
                print(f'  Найдено {len(rows)} строк в "{k}"')
                break
        else:
            rows = []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get('teamName') or row.get('TeamName') or row.get('name')
        tid = row.get('teamId') or row.get('TeamId') or row.get('id')
        sid = row.get('submissionId') or row.get('SubmissionId')
        results.append({'teamName': name, 'teamId': tid, 'submissionId': sid,
                        '_raw': row})
    return results


def fetch_episodes_for_submission(session: requests.Session, sub_id: int) -> list[int]:
    try:
        r = session.post(
            'https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes',
            json={'submissionId': sub_id},
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            timeout=30)
        if r.status_code == 200:
            return [ep['id'] for ep in r.json().get('episodes', []) if ep.get('id')]
    except Exception:
        pass
    return []


def download_replay(session: requests.Session, episode_id: int) -> str:
    out = DATA_DIR / f'{episode_id}.json'
    if out.exists():
        return 'exists'
    for attempt in range(3):
        try:
            r = session.get(
                f'https://www.kaggle.com/api/v1/competitions/episodes/{episode_id}/replay',
                timeout=60)
            if r.status_code == 200:
                out.write_bytes(r.content)
                return 'ok'
            elif r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            else:
                return f'fail_{r.status_code}'
        except Exception:
            return 'fail_exc'
    return 'fail_retries'


def manual_instructions():
    """Если автомат не работает — даём чёткие шаги."""
    print('\n' + '═' * 70)
    print('АВТОМАТИЧЕСКИЙ ПУТЬ НЕ СРАБОТАЛ')
    print('═' * 70)
    print('\nKaggle API не отвечает на наши endpoints для leaderboard.')
    print('Нужно получить submission IDs топ-40 ВРУЧНУЮ. Простейший способ:')
    print('')
    print('1. Открыть в браузере:')
    print('   https://www.kaggle.com/competitions/orbit-wars/leaderboard')
    print('')
    print('2. F12 → Network → перезагрузить страницу')
    print('')
    print('3. В фильтре найти запрос с "leaderboard" или "GetLeaderboard"')
    print('   → правый клик → Copy → Copy as cURL')
    print('   → или → Response → скопировать JSON')
    print('')
    print('4. Прислать мне скопированный JSON или curl-команду')
    print('   → я извлеку teamIDs / submissionIDs автоматически')
    print('')
    print('АЛЬТЕРНАТИВА: установить kaggle CLI с классическим API key:')
    print('   1. https://www.kaggle.com/settings → Create New Token')
    print('   2. mkdir -p ~/.kaggle')
    print('   3. mv ~/Downloads/kaggle.json ~/.kaggle/')
    print('   4. chmod 600 ~/.kaggle/kaggle.json')
    print('   5. kaggle competitions leaderboard orbit-wars --show --csv')
    print('   6. Прислать CSV')
    print('═' * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-total', type=int, default=1000)
    ap.add_argument('--top-n', type=int, default=40)
    args = ap.parse_args()

    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = os.environ.get('KAGGLE_KEY', '')
    if not username or not api_key:
        sys.exit('Установи KAGGLE_USERNAME и KAGGLE_KEY')

    session = make_session(username, api_key)

    # 1. Получаем leaderboard
    lb_data = try_get_leaderboard(session)
    if not lb_data:
        manual_instructions()
        sys.exit(1)

    # Сохраняем raw
    (DATA_DIR / 'leaderboard_raw.json').write_text(
        json.dumps(lb_data, indent=2)[:50000]
    )
    print(f'\nRaw saved: {DATA_DIR / "leaderboard_raw.json"}')

    teams = extract_team_info(lb_data)[:args.top_n]
    if not teams:
        print('⚠ Не смог распарсить структуру leaderboard. Сохранил raw.')
        manual_instructions()
        sys.exit(1)

    print(f'\nИзвлечено {len(teams)} команд:')
    for i, t in enumerate(teams[:5]):
        print(f'  [{i+1}] {t["teamName"]}  teamId={t["teamId"]}  subId={t["submissionId"]}')

    # 2. Собираем episode IDs
    print(f'\nСобираю episode IDs от {len(teams)} команд …')
    sub_ids = [t['submissionId'] for t in teams if t['submissionId']]
    if not sub_ids:
        print('⚠ Нет submissionIDs в данных. Нужны teamIDs?')
        manual_instructions()
        sys.exit(1)

    all_ep_ids = set()
    for i, sid in enumerate(sub_ids):
        eps = fetch_episodes_for_submission(session, sid)
        new = [x for x in eps if x not in all_ep_ids]
        all_ep_ids.update(eps)
        print(f'  [{i+1}/{len(sub_ids)}] sub={sid}: {len(eps)} eps ({len(new)} new) '
              f'total={len(all_ep_ids)}')
        time.sleep(0.5)

    # 3. Скачиваем (свежие первыми)
    existing = {int(p.stem) for p in DATA_DIR.glob('[0-9]*.json') if p.stem.isdigit()}
    to_download = sorted(all_ep_ids - existing, reverse=True)[:args.max_total]
    print(f'\nК скачиванию: {len(to_download)} (limit {args.max_total})')

    stats = {'ok': 0, 'exists': 0, 'fails': 0}
    t0 = time.time()
    for i, eid in enumerate(to_download, 1):
        s = download_replay(session, eid)
        if s == 'ok': stats['ok'] += 1
        elif s == 'exists': stats['exists'] += 1
        else: stats['fails'] += 1
        if i % 20 == 0:
            print(f'  [{i}/{len(to_download)}] ok={stats["ok"]} '
                  f'fails={stats["fails"]} ({time.time()-t0:.0f}s)')
        time.sleep(1.0)

    print(f'\n══════ ИТОГ ══════')
    print(f'  ok:     {stats["ok"]}')
    print(f'  fails:  {stats["fails"]}')


if __name__ == '__main__':
    main()
