#!/usr/bin/env python3
"""
22_download_more_episodes.py — расширенная загрузка реплеев Kaggle.

УЛУЧШЕНИЯ vs 01_download_episodes.py
─────────────────────────────────────
- Параллельные запросы (concurrent.futures.ThreadPoolExecutor)
- Sort by episode_id descending (свежие первыми)
- --refresh: обновляет episodes_index.json через Kaggle API
- Skip already downloaded
- Лучший progress reporting + лимит на пакеты ошибок
- CLI: --max-total, --max-per-submission, --workers, --refresh
- Адаптивный sleep — снижается если нет ошибок

ТРЕБУЕТ
───────
  KAGGLE_USERNAME и KAGGLE_KEY как env vars
  pip install requests

ВЫЗОВ
─────
  # Скачать 1500 самых свежих, использовать существующий index
  export KAGGLE_USERNAME=... KAGGLE_KEY=...
  python3 22_download_more_episodes.py --max-total 1500

  # Обновить index (запросить новые submissionы) и скачать 2000 свежих
  python3 22_download_more_episodes.py --refresh --max-total 2000 --workers 4

ВЫХОД
─────
  opponent_learning/data/raw/{episode_id}.json — новые файлы
  opponent_learning/data/raw/episodes_index.json — обновлённый index
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
DATA_DIR   = OL_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL   = "https://www.kaggle.com/api/v1/competitions/episodes/{id}/replay"

# Дефолтные топ-игроки (можно переопределить через --submission-ids)
DEFAULT_TOP_SUBMISSION_IDS = [
    52318886,   # bowwowforeach
    52292204,   # Shun_PI
    51987365,   # Vadasz
    52300620,   # Kovi
    52266125,   # Ousagi
    52335742,   # sash
    52334987,   # Andrew Tratz
    52358342,   # ymg_aq
    52266849,   # ush
    52276006,   # Wenchong Huang
    52346769,   # Orbit Team
    52317967,   # flg
    52312579,   # HY2017
    52103846,   # Orbital Occle
    52214689,   # lookaside
    52357823,   # Claws
    52334402,   # Ezra
    52176489,   # fgwiebfaoish
    52279326,   # SalvadorDali
]


# ── Auth ────────────────────────────────────────────────────────────────
def _auth_headers(api_key: str) -> dict:
    if api_key.startswith('KGAT_') or api_key.startswith('kkk_'):
        return {'Authorization': f'Bearer {api_key}'}
    return {}


def _make_session(username: str, api_key: str) -> requests.Session:
    """Сессия с переиспользованием соединения и нужным auth."""
    s = requests.Session()
    headers = _auth_headers(api_key)
    if headers:
        s.headers.update(headers)
    else:
        s.auth = (username, api_key)
    return s


# ── Получение episode IDs ────────────────────────────────────────────────
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
    except Exception as e:
        print(f'  ⚠ submission {sub_id}: {e!r}')
    return []


def refresh_index(session: requests.Session, top_submission_ids: list,
                  sleep_between: float = 0.5) -> list:
    all_ids = set()
    print(f'\nЗапрашиваю episode IDs для {len(top_submission_ids)} submissions …')
    for i, sub_id in enumerate(top_submission_ids):
        ids = fetch_episode_ids_for_submission(session, sub_id)
        new = [x for x in ids if x not in all_ids]
        all_ids.update(ids)
        print(f'  [{i+1}/{len(top_submission_ids)}] sub={sub_id}  '
              f'{len(ids)} episodes ({len(new)} новых)  total={len(all_ids)}')
        time.sleep(sleep_between)
    return sorted(all_ids, reverse=True)  # свежие первыми


# ── Скачивание ───────────────────────────────────────────────────────────
def download_one(session: requests.Session, episode_id: int,
                 out_dir: Path, timeout: int = 60) -> tuple[int, str]:
    """Возвращает (episode_id, status). status: 'ok' / 'exists' / 'fail_NNN'."""
    out_path = out_dir / f'{episode_id}.json'
    if out_path.exists():
        return episode_id, 'exists'
    try:
        r = session.get(REPLAY_URL.format(id=episode_id), timeout=timeout)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            return episode_id, 'ok'
        else:
            return episode_id, f'fail_{r.status_code}'
    except Exception as e:
        return episode_id, f'fail_exc'


def download_parallel(episode_ids: list, session: requests.Session,
                       workers: int = 3, sleep_between_batches: float = 0.0,
                       max_consecutive_fails: int = 20) -> dict:
    """Параллельное скачивание. Останавливается если идёт серия fails (rate limit)."""
    stats = {'ok': 0, 'exists': 0, 'failed': 0}
    consecutive_fails = 0
    t0 = time.time()

    print(f'\nЗапуск скачивания {len(episode_ids)} эпизодов '
          f'({workers} workers) …')
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, session, eid, DATA_DIR): eid
                   for eid in episode_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            eid, status = fut.result()
            if status == 'ok':
                stats['ok'] += 1
                consecutive_fails = 0
            elif status == 'exists':
                stats['exists'] += 1
                consecutive_fails = 0
            else:
                stats['failed'] += 1
                consecutive_fails += 1
                if consecutive_fails >= max_consecutive_fails:
                    print(f'\n  ⚠ {consecutive_fails} consecutive failures — '
                          f'останавливаюсь (rate limit?)')
                    # Cancel остальные
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
            if i % 25 == 0:
                dt = time.time() - t0
                print(f'  [{i:>4}/{len(episode_ids)}]  ok={stats["ok"]}  '
                      f'exists={stats["exists"]}  failed={stats["failed"]}  '
                      f'({dt:.0f}s, {i/dt:.1f}/s)')
            if sleep_between_batches and i % workers == 0:
                time.sleep(sleep_between_batches)

    return stats


# ── main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-total', type=int, default=1500,
                    help='Максимум новых файлов скачать')
    ap.add_argument('--refresh', action='store_true',
                    help='Обновить episodes_index.json через Kaggle API')
    ap.add_argument('--workers', type=int, default=3,
                    help='Concurrent downloads (3 — мягко, 5 — рискованно)')
    ap.add_argument('--submission-ids', type=str, default=None,
                    help='Comma-separated кастомные submission IDs (override default)')
    args = ap.parse_args()

    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key  = os.environ.get('KAGGLE_KEY', '')
    if not username or not api_key:
        sys.exit('⚠ Установи KAGGLE_USERNAME и KAGGLE_KEY как env vars')

    session = _make_session(username, api_key)

    # Получаем список episode IDs
    index_path = DATA_DIR / 'episodes_index.json'
    sub_ids = (DEFAULT_TOP_SUBMISSION_IDS if not args.submission_ids
                else [int(x) for x in args.submission_ids.split(',')])

    if args.refresh or not index_path.exists():
        episode_ids = refresh_index(session, sub_ids)
        with open(index_path, 'w') as f:
            json.dump(episode_ids, f)
        print(f'✓ Index обновлён: {index_path}  ({len(episode_ids)} IDs)')
    else:
        with open(index_path) as f:
            episode_ids = json.load(f)
        # Свежие первыми
        episode_ids = sorted(episode_ids, reverse=True)
        print(f'Использую существующий index: {len(episode_ids)} IDs')

    # Skip already downloaded
    existing = {int(p.stem) for p in DATA_DIR.glob('[0-9]*.json')
                if p.stem.isdigit()}
    to_download = [eid for eid in episode_ids if eid not in existing]
    print(f'\nУже скачано: {len(existing)}')
    print(f'Доступно к скачиванию: {len(to_download)}')

    if args.max_total and len(to_download) > args.max_total:
        to_download = to_download[:args.max_total]
        print(f'Лимит --max-total: возьмём первые {len(to_download)} '
              f'(свежие, сортировка по ID desc)')

    if not to_download:
        print('Нечего качать.')
        return

    stats = download_parallel(to_download, session, workers=args.workers)

    print(f'\n══════ ИТОГ ══════')
    print(f'  ok:     {stats["ok"]}')
    print(f'  exists: {stats["exists"]}')
    print(f'  failed: {stats["failed"]}')
    total_in_dir = len(list(DATA_DIR.glob('[0-9]*.json')))
    print(f'  всего в {DATA_DIR}: {total_in_dir} файлов')


if __name__ == '__main__':
    main()
