#!/usr/bin/env python3
"""
29_simple_download.py — простой подход: scrape submission IDs со страниц команд.

ЛОГИКА (как в старом 01_*)
──────────────────────────
1. Из CSV берём топ-N teamIDs
2. Для каждой teamID — GET публичной страницы команды на kaggle.com
3. Regex'ом из HTML вытаскиваем submission IDs
4. Через рабочий EpisodeService/ListEpisodes (как и раньше) → episode IDs
5. Скачиваем эпизоды (тот же replay endpoint что и в 01_*)

ВЫЗОВ
─────
  export KAGGLE_USERNAME=... KAGGLE_KEY=...
  python3 29_simple_download.py --top-n 40 --max-total 1500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL   = "https://www.kaggle.com/api/v1/competitions/episodes/{id}/replay"


def make_session(username: str, api_key: str) -> requests.Session:
    s = requests.Session()
    if api_key.startswith('KGAT_') or api_key.startswith('kkk_'):
        s.headers.update({'Authorization': f'Bearer {api_key}'})
    else:
        s.auth = (username, api_key)
    s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh) Chrome/120'
    return s


def find_csv() -> Path | None:
    candidates = list(REBUILD_DATA.glob('*publicleaderboard*.csv'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_leaderboard(csv_path: Path, top_n: int) -> list[dict]:
    rows = []
    with open(csv_path, encoding='utf-8-sig') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= top_n:
                break
            rows.append({
                'rank': int(r['Rank']),
                'team_id': int(r['TeamId']),
                'team_name': r['TeamName'],
            })
    return rows


def scrape_submission_ids(session: requests.Session, team_id: int) -> list[int]:
    """GET страницы команды, ищем submission IDs в HTML."""
    # Несколько вариантов URL
    urls = [
        f'https://www.kaggle.com/competitions/orbit-wars/team/{team_id}',
        f'https://www.kaggle.com/c/orbit-wars/team/{team_id}',
        f'https://www.kaggle.com/competitions/orbit-wars/leaderboard?team={team_id}',
    ]
    for url in urls:
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                continue
            html = r.text
            # Поиск submission IDs в разных форматах
            ids = set()
            # Pattern 1: /submissions/NNNNNNN
            ids.update(int(m) for m in re.findall(r'/submissions/(\d{6,10})', html))
            # Pattern 2: "submissionId": NNN или submissionId: NNN
            ids.update(int(m) for m in re.findall(r'submissionId["\':\s]+(\d{6,10})', html))
            # Pattern 3: "id": NNN в submission-like контексте
            ids.update(int(m) for m in re.findall(r'submission["\'].*?"id":\s*(\d{6,10})', html))
            # Очистка — submission IDs обычно 52xxx-55xxx
            ids = [i for i in ids if 10_000_000 < i < 99_999_999]
            if ids:
                return sorted(set(ids), reverse=True)[:5]  # топ-5 свежих
        except Exception as e:
            print(f'    GET {url[:60]}… → {e!r}')
    return []


def list_episodes(session: requests.Session, sub_id: int) -> list[int]:
    try:
        r = session.post(EPISODES_URL,
                         json={'submissionId': sub_id},
                         headers={'Content-Type': 'application/json',
                                  'Accept': 'application/json'},
                         timeout=30)
        if r.status_code == 200:
            return [ep['id'] for ep in r.json().get('episodes', []) if ep.get('id')]
    except Exception:
        pass
    return []


def download_replay(session: requests.Session, eid: int) -> str:
    out = DATA_DIR / f'{eid}.json'
    if out.exists():
        return 'exists'
    for attempt in range(3):
        try:
            r = session.get(REPLAY_URL.format(id=eid), timeout=60)
            if r.status_code == 200:
                out.write_bytes(r.content)
                return 'ok'
            elif r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            return f'fail_{r.status_code}'
        except Exception:
            return 'fail_exc'
    return 'fail_retries'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-n', type=int, default=40)
    ap.add_argument('--max-total', type=int, default=1500)
    ap.add_argument('--sleep', type=float, default=1.0)
    args = ap.parse_args()

    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = os.environ.get('KAGGLE_KEY', '')
    if not username or not api_key:
        sys.exit('Установи KAGGLE_USERNAME и KAGGLE_KEY')

    csv_path = find_csv()
    if not csv_path:
        sys.exit('⚠ CSV leaderboard не найден в rebuild/data/')
    print(f'CSV: {csv_path.name}')

    teams = parse_leaderboard(csv_path, args.top_n)
    print(f'Топ-{len(teams)} команд (выбраны по Rank)')

    session = make_session(username, api_key)

    # Шаг 1: scrape submission IDs
    print(f'\n══ ШАГ 1: scrape submission IDs со страниц команд ══')
    team_subs = []
    for i, t in enumerate(teams):
        sub_ids = scrape_submission_ids(session, t['team_id'])
        team_subs.append({**t, 'sub_ids': sub_ids})
        print(f'  [{i+1}/{len(teams)}] #{t["rank"]} {t["team_name"]:<32} '
              f'teamId={t["team_id"]} → {len(sub_ids)} subs')
        if sub_ids:
            print(f'    subs: {sub_ids}')
        time.sleep(0.5)

    # Сохраняем mapping
    (REBUILD_DATA / 'team_to_subs.json').write_text(
        json.dumps(team_subs, indent=2, ensure_ascii=False))

    total = sum(len(t['sub_ids']) for t in team_subs)
    print(f'\n  Всего subs: {total}')
    if total == 0:
        print('\n⚠ HTML scrape не нашёл submissions. Возможно:')
        print('  - Страница требует auth → fallback на API leaderboard')
        print('  - Kaggle отдаёт только JS-shell без данных (SSR через JS)')
        print('  - Регулярки не подходят к их HTML')
        print(f'\n  Сохрани одну страницу для анализа:')
        print(f'  curl -H "Authorization: Bearer $KAGGLE_KEY" \\')
        print(f'    "https://www.kaggle.com/competitions/orbit-wars/team/{teams[0]["team_id"]}" \\')
        print(f'    -o /tmp/team_page.html')
        print(f'  Открой /tmp/team_page.html и поищи submission IDs.')
        sys.exit(1)

    # Шаг 2: episode IDs
    print(f'\n══ ШАГ 2: получаю episode IDs через рабочий ListEpisodes ══')
    all_eps = set()
    for t in team_subs:
        for sid in t['sub_ids']:
            eps = list_episodes(session, sid)
            new = [x for x in eps if x not in all_eps]
            all_eps.update(eps)
            print(f'  team #{t["rank"]} sub={sid}: {len(eps)} eps ({len(new)} new) total={len(all_eps)}')
            time.sleep(0.5)

    # Шаг 3: скачать
    existing = {int(p.stem) for p in DATA_DIR.glob('[0-9]*.json') if p.stem.isdigit()}
    to_dl = sorted(all_eps - existing, reverse=True)[:args.max_total]
    print(f'\n══ ШАГ 3: скачиваю {len(to_dl)} эпизодов ══')

    stats = {'ok': 0, 'fails': 0, 'exists': 0}
    consec_fails = 0
    t0 = time.time()
    for i, eid in enumerate(to_dl, 1):
        s = download_replay(session, eid)
        if s == 'ok':
            stats['ok'] += 1
            consec_fails = 0
        elif s == 'exists':
            stats['exists'] += 1
        else:
            stats['fails'] += 1
            consec_fails += 1
            if consec_fails >= 25:
                print(f'\n  {consec_fails} consecutive fails — стоп')
                break
        if i % 25 == 0:
            print(f'  [{i}/{len(to_dl)}] ok={stats["ok"]} fails={stats["fails"]} '
                  f'({time.time()-t0:.0f}s)')
        time.sleep(args.sleep)

    print(f'\n══════ ИТОГ ══════')
    print(f'  ok:     {stats["ok"]}')
    print(f'  exists: {stats["exists"]}')
    print(f'  fails:  {stats["fails"]}')


if __name__ == '__main__':
    main()
