#!/usr/bin/env python3
"""
27_download_from_leaderboard_csv.py — скачивание эпизодов от АКТУАЛЬНЫХ топ-N
через официальный CSV-экспорт leaderboard.

ИДЕЯ
────
Пользователь загрузил `orbit-wars-publicleaderboard-*.csv` (от Kaggle). У него
есть Rank, TeamId, TeamName, Score. По TeamId через ListSubmissions API
получаем submissionIDs, далее ListEpisodes → episode IDs → скачать.

ВХОД
────
  rebuild/data/orbit-wars-publicleaderboard-*.csv (любой CSV с TeamId column)

ВЫХОД
─────
  data/raw/{episode_id}.json — новые эпизоды от топ-N
  rebuild/data/leaderboard_top_subs.json — mapping (rank, teamName, teamId, subIds)

ВЫЗОВ
─────
  export KAGGLE_USERNAME=... KAGGLE_KEY=...
  python3 27_download_from_leaderboard_csv.py --top-n 40 --max-total 1500
"""

from __future__ import annotations

import argparse
import csv
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
REBUILD_DATA = REBUILD / "data"

SUBMISSIONS_URL = "https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions"
EPISODES_URL    = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL      = "https://www.kaggle.com/api/v1/competitions/episodes/{id}/replay"

# competitionId orbit-wars — обычно 86535 или нужно подобрать; ListSubmissions
# может требовать competitionId. Попробуем оба пути: с competitionId и без.
COMPETITION_ID = 86535


def make_session(username: str, api_key: str) -> requests.Session:
    s = requests.Session()
    if api_key.startswith('KGAT_') or api_key.startswith('kkk_'):
        s.headers.update({'Authorization': f'Bearer {api_key}'})
    else:
        s.auth = (username, api_key)
    s.headers['User-Agent'] = 'Kaggle/Python'
    return s


def find_leaderboard_csv() -> Path | None:
    """Найти последний CSV в rebuild/data/."""
    candidates = list(REBUILD_DATA.glob('*publicleaderboard*.csv'))
    if not candidates:
        return None
    # Берём самый свежий по mtime
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_leaderboard(csv_path: Path, top_n: int = 40) -> list[dict]:
    """Извлекает топ-N строк из CSV."""
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = []
        for i, r in enumerate(reader):
            if i >= top_n:
                break
            rows.append({
                'rank':      int(r.get('Rank', i + 1)),
                'team_id':   int(r['TeamId']),
                'team_name': r['TeamName'],
                'score':     float(r.get('Score', 0)) if r.get('Score') else None,
            })
    return rows


def list_submissions_for_team(session: requests.Session, team_id: int) -> list[int]:
    """Через ListSubmissions API получаем все submission ID для команды."""
    # Пробуем несколько вариантов body
    for body in [
        {'teamId': team_id, 'competitionId': COMPETITION_ID},
        {'teamId': team_id},
        {'TeamId': team_id},
    ]:
        try:
            r = session.post(
                SUBMISSIONS_URL, json=body,
                headers={'Content-Type': 'application/json',
                         'Accept': 'application/json'},
                timeout=30)
            if r.status_code == 200:
                data = r.json()
                subs = data.get('submissions') or data.get('Submissions') or []
                # Берём только submissions с computed score (играли в матчах)
                ids = []
                for s in subs:
                    sid = s.get('id') or s.get('Id') or s.get('submissionId')
                    has_score = s.get('hasComputedScore') or s.get('score') is not None
                    if sid and has_score:
                        ids.append(sid)
                if ids:
                    return ids
        except Exception:
            continue
    return []


def list_episodes_for_submission(session: requests.Session, sub_id: int) -> list[int]:
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


def download_replay(session: requests.Session, eid: int,
                    base_sleep: float = 1.0) -> tuple[str, float]:
    out = DATA_DIR / f'{eid}.json'
    if out.exists():
        return 'exists', base_sleep
    for attempt in range(3):
        try:
            r = session.get(REPLAY_URL.format(id=eid), timeout=60)
            if r.status_code == 200:
                out.write_bytes(r.content)
                return 'ok', max(base_sleep, base_sleep * 0.9)
            elif r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f'    [429] жду {wait}s …')
                time.sleep(wait)
                continue
            return f'fail_{r.status_code}', base_sleep * 1.5
        except Exception:
            return 'fail_exc', base_sleep * 2.0
    return 'fail_retries', base_sleep * 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None,
                    help='Путь к CSV leaderboard (auto-detect если не задан)')
    ap.add_argument('--top-n', type=int, default=40)
    ap.add_argument('--max-total', type=int, default=1500)
    ap.add_argument('--base-sleep', type=float, default=1.0)
    args = ap.parse_args()

    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = os.environ.get('KAGGLE_KEY', '')
    if not username or not api_key:
        sys.exit('Установи KAGGLE_USERNAME и KAGGLE_KEY')

    # Найти CSV
    csv_path = Path(args.csv) if args.csv else find_leaderboard_csv()
    if not csv_path or not csv_path.exists():
        sys.exit(f'⚠ CSV не найден. Положи orbit-wars-publicleaderboard-*.csv '
                  f'в {REBUILD_DATA}/ или укажи --csv')
    print(f'CSV: {csv_path.name}')

    teams = parse_leaderboard(csv_path, top_n=args.top_n)
    print(f'Топ-{len(teams)} команд:')
    for t in teams[:5]:
        print(f'  #{t["rank"]:>2}  {t["team_name"]:<30}  teamId={t["team_id"]}')
    if len(teams) > 5:
        print(f'  ... ещё {len(teams)-5}')

    session = make_session(username, api_key)

    # Шаг 1: для каждой teamID — submission IDs
    print(f'\n══ ШАГ 1: получаю submission IDs ══')
    all_team_subs = []
    for i, t in enumerate(teams):
        subs = list_submissions_for_team(session, t['team_id'])
        all_team_subs.append({**t, 'submission_ids': subs})
        print(f'  [{i+1}/{len(teams)}] #{t["rank"]} {t["team_name"]:<30}  '
              f'teamId={t["team_id"]}  → {len(subs)} subs')
        time.sleep(0.5)
    # Сохраняем mapping
    (REBUILD_DATA / 'leaderboard_top_subs.json').write_text(
        json.dumps(all_team_subs, indent=2, ensure_ascii=False))
    print(f'  Сохранён mapping: {REBUILD_DATA / "leaderboard_top_subs.json"}')

    total_subs = sum(len(t['submission_ids']) for t in all_team_subs)
    print(f'  Всего submissions: {total_subs}')
    if total_subs == 0:
        print('⚠ Ни одного submission не получено. API не отвечает корректно.')
        print('  Проверь response через curl или DevTools.')
        sys.exit(1)

    # Шаг 2: для каждого submission ID — episode IDs
    print(f'\n══ ШАГ 2: получаю episode IDs ══')
    all_ep_ids = set()
    for i, t in enumerate(all_team_subs):
        for sid in t['submission_ids']:
            eps = list_episodes_for_submission(session, sid)
            new = [x for x in eps if x not in all_ep_ids]
            all_ep_ids.update(eps)
            print(f'  team #{t["rank"]}  sub={sid}: {len(eps)} eps ({len(new)} new)  '
                  f'total={len(all_ep_ids)}')
            time.sleep(0.5)

    # Сохраняем index
    index_path = DATA_DIR / 'episodes_index_top40.json'
    index_path.write_text(json.dumps(sorted(all_ep_ids, reverse=True)))
    print(f'  Index сохранён: {index_path}')

    # Шаг 3: скачать (свежие первыми, skip existing)
    existing = {int(p.stem) for p in DATA_DIR.glob('[0-9]*.json')
                if p.stem.isdigit()}
    to_download = sorted(all_ep_ids - existing, reverse=True)
    print(f'\nСуществующие: {len(existing)}')
    print(f'К скачиванию (всего): {len(to_download)}')

    if args.max_total and len(to_download) > args.max_total:
        to_download = to_download[:args.max_total]
        print(f'Лимит --max-total: {len(to_download)}')

    if not to_download:
        print('Нечего скачивать.')
        return

    print(f'\n══ ШАГ 3: скачиваю {len(to_download)} эпизодов '
          f'(serial, retry on 429) ══')
    stats = {'ok': 0, 'exists': 0, 'fails': 0}
    consec_fails = 0
    sleep = args.base_sleep
    t0 = time.time()
    for i, eid in enumerate(to_download, 1):
        status, sleep = download_replay(session, eid, sleep)
        if status == 'ok':
            stats['ok'] += 1
            consec_fails = 0
        elif status == 'exists':
            stats['exists'] += 1
        else:
            stats['fails'] += 1
            consec_fails += 1
            if consec_fails >= 25:
                print(f'\n  ⚠ {consec_fails} consecutive fails — стоп')
                break
        if i % 25 == 0:
            dt = time.time() - t0
            print(f'  [{i:>4}/{len(to_download)}] ok={stats["ok"]} '
                  f'fails={stats["fails"]} sleep={sleep:.1f}s '
                  f'({dt:.0f}s, {i/dt:.1f}/s)')
        time.sleep(sleep)

    print(f'\n══════ ИТОГ ══════')
    print(f'  ok:     {stats["ok"]}')
    print(f'  exists: {stats["exists"]}')
    print(f'  fails:  {stats["fails"]}')
    total = len(list(DATA_DIR.glob('[0-9]*.json')))
    print(f'  всего файлов в {DATA_DIR}: {total}')


if __name__ == '__main__':
    main()
