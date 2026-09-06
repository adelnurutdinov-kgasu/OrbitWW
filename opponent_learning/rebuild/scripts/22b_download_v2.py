#!/usr/bin/env python3
"""
22b_download_v2.py — улучшенная загрузка реплеев с retry и leaderboard refresh.

ЧТО ИСПРАВЛЕНО vs 22_*
──────────────────────
- workers=1 по умолчанию (Kaggle тяжело терпит >1)
- Retry on 429 с exponential backoff
- Адаптивный sleep (увеличивается при ошибках, снижается при успехах)
- Опция --refresh-leaderboard для получения актуальных submissions через
  Kaggle CLI (если установлен kaggle package)

КАК ОПРЕДЕЛЯЕМ ТОПОВ
────────────────────
Kaggle CLI:  kaggle competitions leaderboard orbit-wars --csv --download
→ CSV с teamId/teamName, нужно получить их submission IDs (отдельный API call)

Альтернативно: получить leaderboard JSON через прямой API
  https://www.kaggle.com/api/v1/competitions/leaderboards/orbit-wars

После refresh_leaderboard → новый submission_ids → ListEpisodes → новый
episodes_index.json с свежими эпизодами топ-N игроков.

ВЫЗОВ
─────
  # Базовый — используй existing index, slow download
  python3 22b_download_v2.py --max-total 1000

  # Обновить leaderboard и получить эпизоды от текущих топ-30
  python3 22b_download_v2.py --refresh-leaderboard --top-n 30 --max-total 1500

ВЫХОД
─────
  data/raw/{episode_id}.json (новые)
  data/raw/episodes_index.json (обновлённый)
  data/raw/current_leaderboard.csv (если --refresh-leaderboard)
"""

from __future__ import annotations

import argparse
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

EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL   = "https://www.kaggle.com/api/v1/competitions/episodes/{id}/replay"
LEADERBOARD_URL = "https://www.kaggle.com/api/v1/competitions/leaderboards/orbit-wars"
SUBMISSIONS_URL = "https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions"

# Fallback submissions если не получится refresh
DEFAULT_TOP_SUBMISSION_IDS = [
    52318886, 52292204, 51987365, 52300620, 52266125, 52335742,
    52334987, 52358342, 52266849, 52276006, 52346769, 52317967,
    52312579, 52103846, 52214689, 52357823, 52334402, 52176489,
    52279326,
]


def _auth_headers(api_key: str) -> dict:
    if api_key.startswith('KGAT_') or api_key.startswith('kkk_'):
        return {'Authorization': f'Bearer {api_key}'}
    return {}


def _make_session(username: str, api_key: str) -> requests.Session:
    s = requests.Session()
    headers = _auth_headers(api_key)
    if headers:
        s.headers.update(headers)
    else:
        s.auth = (username, api_key)
    s.headers['User-Agent'] = 'Kaggle/Python-Custom'
    return s


# ── Refresh leaderboard ──────────────────────────────────────────────────
def refresh_leaderboard_via_api(session: requests.Session, top_n: int = 30) -> list[int]:
    """Прямой API call (минуя kaggle CLI). Возвращает teamIDs топ-N."""
    # Internal endpoint для leaderboard
    endpoint = "https://www.kaggle.com/api/i/competitions.CompetitionApiService/GetLeaderboard"
    try:
        r = session.post(endpoint,
                         json={"competitionName": "orbit-wars"},
                         headers={'Content-Type': 'application/json',
                                   'Accept': 'application/json'},
                         timeout=30)
    except Exception as e:
        print(f'  ⚠ leaderboard API ошибка: {e!r}')
        return []
    if r.status_code != 200:
        print(f'  ⚠ leaderboard API status {r.status_code}: {r.text[:200]}')
        # Альтернативный endpoint
        alt = "https://www.kaggle.com/api/v1/competitions/leaderboards/orbit-wars"
        try:
            r = session.get(alt, timeout=30)
        except Exception as e:
            print(f'  ⚠ alt API: {e!r}')
            return []
        if r.status_code != 200:
            print(f'  ⚠ alt status {r.status_code}: {r.text[:200]}')
            return []

    try:
        data = r.json()
    except Exception:
        print(f'  ⚠ не JSON: {r.text[:200]}')
        return []

    # Parse — структура может быть {"submissions": [...]} или {"results": [...]}
    rows = []
    for k in ('submissions', 'results', 'leaderboard', 'entries'):
        if k in data and isinstance(data[k], list):
            rows = data[k]
            print(f'  ключ ответа: {k}, длина: {len(rows)}')
            break
    if not rows and isinstance(data, list):
        rows = data
    if not rows:
        print(f'  ⚠ не нашёл результаты, keys: {list(data.keys())[:10] if isinstance(data, dict) else "list"}')
        # Сохраним для диагностики
        (DATA_DIR / 'leaderboard_raw.json').write_text(json.dumps(data, indent=2)[:5000])
        print(f'  raw сохранён: {DATA_DIR / "leaderboard_raw.json"}')
        return []

    team_ids = []
    for i, row in enumerate(rows):
        if i >= top_n:
            break
        # Возможные имена поля team ID
        tid = row.get('teamId') or row.get('teamID') or row.get('TeamId')
        if tid:
            try:
                team_ids.append(int(tid))
            except (TypeError, ValueError):
                pass
    print(f'  Получено {len(team_ids)} teamIDs из топ-{top_n}')

    # Сохраняем для следующих запусков
    (DATA_DIR / 'current_leaderboard.json').write_text(json.dumps(rows[:top_n], indent=2))
    return team_ids


def get_submission_ids_for_teams(session: requests.Session,
                                  team_ids: list[int]) -> list[int]:
    """Для списка teamIDs получает свежие submission IDs через API."""
    sub_ids = []
    for i, tid in enumerate(team_ids):
        try:
            r = session.post(SUBMISSIONS_URL,
                              json={"teamId": tid, "competitionId": 86535},
                              headers={'Content-Type': 'application/json',
                                       'Accept': 'application/json'},
                              timeout=30)
            if r.status_code == 200:
                subs = r.json().get('submissions', [])
                # Берём самый свежий active submission
                for s in subs:
                    if s.get('hasComputedScore'):
                        sub_ids.append(s.get('id'))
                        break
            else:
                print(f'    team {tid}: {r.status_code}')
        except Exception as e:
            print(f'    team {tid}: {e!r}')
        time.sleep(0.5)
        if (i + 1) % 5 == 0:
            print(f'  {i+1}/{len(team_ids)} обработано, {len(sub_ids)} subs')
    return sub_ids


def fetch_episode_ids_for_submission(session: requests.Session, sub_id: int) -> list:
    try:
        r = session.post(EPISODES_URL,
                          json={"submissionId": sub_id},
                          headers={'Content-Type': 'application/json',
                                    'Accept': 'application/json'},
                          timeout=30)
        if r.status_code == 200:
            episodes = r.json().get('episodes', [])
            return [ep['id'] for ep in episodes if ep.get('id')]
    except Exception:
        pass
    return []


def build_episodes_index(session: requests.Session, sub_ids: list[int]) -> list[int]:
    all_ids = set()
    for i, sub_id in enumerate(sub_ids):
        ids = fetch_episode_ids_for_submission(session, sub_id)
        new = [x for x in ids if x not in all_ids]
        all_ids.update(ids)
        print(f'  [{i+1}/{len(sub_ids)}] sub={sub_id} → {len(ids)} eps ({len(new)} new) total={len(all_ids)}')
        time.sleep(0.5)
    return sorted(all_ids, reverse=True)


# ── Download with retry ──────────────────────────────────────────────────
def download_with_retry(session: requests.Session, episode_id: int,
                         max_retries: int = 3) -> tuple[str, float]:
    """Скачивает один replay. Возвращает (status, suggested_next_sleep)."""
    out_path = DATA_DIR / f'{episode_id}.json'
    if out_path.exists():
        return 'exists', 0.5

    url = REPLAY_URL.format(id=episode_id)
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                out_path.write_bytes(r.content)
                return 'ok', 0.5  # успех — можно сбросить sleep
            elif r.status_code == 429:
                # rate limit — wait более долго
                wait = 30 * (attempt + 1)
                print(f'    [429 rate limit] жду {wait}s …')
                time.sleep(wait)
                continue
            elif r.status_code in (404,):
                return f'fail_{r.status_code}', 1.0
            else:
                return f'fail_{r.status_code}', 2.0
        except requests.RequestException as e:
            return 'fail_exc', 3.0
    return 'fail_retries', 5.0


def download_serial(episode_ids: list[int], session: requests.Session,
                     base_sleep: float = 1.0, max_consecutive_fails: int = 30):
    stats = {'ok': 0, 'exists': 0, 'fails': 0}
    consecutive_fails = 0
    current_sleep = base_sleep
    t0 = time.time()
    n = len(episode_ids)

    for i, eid in enumerate(episode_ids, 1):
        status, suggested_sleep = download_with_retry(session, eid)
        if status == 'ok':
            stats['ok'] += 1
            consecutive_fails = 0
            current_sleep = max(base_sleep, current_sleep * 0.9)
        elif status == 'exists':
            stats['exists'] += 1
        else:
            stats['fails'] += 1
            consecutive_fails += 1
            current_sleep = min(10.0, current_sleep * 1.5)
            if consecutive_fails % 5 == 0:
                print(f'    {consecutive_fails} consecutive fails, sleep={current_sleep:.1f}s, last={status}')
            if consecutive_fails >= max_consecutive_fails:
                print(f'\n  ⚠ {consecutive_fails} consecutive fails — стоп')
                break

        if i % 25 == 0:
            dt = time.time() - t0
            print(f'  [{i:>5}/{n}] ok={stats["ok"]} exists={stats["exists"]} '
                  f'fails={stats["fails"]} sleep={current_sleep:.1f}s '
                  f'({dt:.0f}s, {i/dt:.1f}/s)')

        time.sleep(current_sleep)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-total', type=int, default=1000)
    ap.add_argument('--refresh-leaderboard', action='store_true',
                    help='Получить актуальный leaderboard через kaggle CLI')
    ap.add_argument('--top-n', type=int, default=30,
                    help='Top-N игроков с leaderboard для refresh')
    ap.add_argument('--base-sleep', type=float, default=1.0)
    args = ap.parse_args()

    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = os.environ.get('KAGGLE_KEY', '')
    if not username or not api_key:
        sys.exit('Установи KAGGLE_USERNAME и KAGGLE_KEY')

    session = _make_session(username, api_key)
    index_path = DATA_DIR / 'episodes_index.json'

    # Опционально обновить leaderboard → submissions → episodes_index
    if args.refresh_leaderboard:
        print('\n══ ШАГ 1: refresh leaderboard (через API) ══')
        team_ids = refresh_leaderboard_via_api(session, top_n=args.top_n)
        if not team_ids:
            print('  Падаем обратно на DEFAULT_TOP_SUBMISSION_IDS')
            sub_ids = DEFAULT_TOP_SUBMISSION_IDS
        else:
            print(f'  Получаю submission IDs для {len(team_ids)} команд …')
            sub_ids = get_submission_ids_for_teams(session, team_ids)
            print(f'  Найдено {len(sub_ids)} submissions')

        print('\n══ ШАГ 2: получаю episode IDs ══')
        episode_ids = build_episodes_index(session, sub_ids)
        with open(index_path, 'w') as f:
            json.dump(episode_ids, f)
        print(f'  Index обновлён: {len(episode_ids)} IDs')
    else:
        if not index_path.exists():
            print('  ⚠ episodes_index.json нет, запусти с --refresh-leaderboard')
            sys.exit(1)
        with open(index_path) as f:
            episode_ids = json.load(f)
        episode_ids = sorted(episode_ids, reverse=True)
        print(f'Используем existing index: {len(episode_ids)} IDs')

    existing = {int(p.stem) for p in DATA_DIR.glob('[0-9]*.json') if p.stem.isdigit()}
    to_download = [eid for eid in episode_ids if eid not in existing]
    print(f'\nСкачано уже: {len(existing)}')
    print(f'К скачиванию: {len(to_download)}')

    if args.max_total and len(to_download) > args.max_total:
        to_download = to_download[:args.max_total]
        print(f'Лимит: {len(to_download)}')

    if not to_download:
        print('Нечего качать.')
        return

    print(f'\n══ ШАГ 3: serial download (workers=1, retry 429, base_sleep={args.base_sleep}s) ══')
    stats = download_serial(to_download, session, base_sleep=args.base_sleep)

    print(f'\n══════ ИТОГ ══════')
    print(f'  ok:     {stats["ok"]}')
    print(f'  exists: {stats["exists"]}')
    print(f'  fails:  {stats["fails"]}')
    total = len(list(DATA_DIR.glob('[0-9]*.json')))
    print(f'  всего в {DATA_DIR}: {total} файлов')


if __name__ == '__main__':
    main()
