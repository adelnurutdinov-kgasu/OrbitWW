#!/usr/bin/env python3
"""
01_download_episodes.py — скачивает replay-файлы эпизодов через Kaggle API.

Адаптировано из api-download-replay-orbit-wars.ipynb.

Конфигурация:
  KAGGLE_USERNAME и KAGGLE_KEY — переменные окружения (или вставить напрямую).
  TOP_SUBMISSION_IDS — submission IDs топ-игроков лидерборда.

Вывод:
  opponent_learning/data/raw/*.json       — replay-файлы эпизодов
  opponent_learning/data/raw/episodes_merged.json  — индекс всех эпизодов

Запуск:
  export KAGGLE_USERNAME=твой_логин
  export KAGGLE_KEY=твой_ключ
  python3 opponent_learning/scripts/01_download_episodes.py

Или без env (вставь токен прямо сюда):
  USERNAME = "твой_логин"
  API_KEY  = "твой_ключ"
"""

import os
import json
import time
import requests
from pathlib import Path

# ── Auth ────────────────────────────────────────────────────────────────────
USERNAME = os.environ.get('KAGGLE_USERNAME', '')
API_KEY  = os.environ.get('KAGGLE_KEY', '')

# Если env не заданы — вставь сюда (не коммитить!):
# USERNAME = "your_kaggle_username"
# API_KEY  = "your_kaggle_api_key"

if not USERNAME or not API_KEY:
    raise RuntimeError(
        "Задай KAGGLE_USERNAME и KAGGLE_KEY как переменные окружения,\n"
        "или вставь напрямую в начале скрипта."
    )

# ── Пути ────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Топ-игроки (submission IDs из лидерборда) ────────────────────────────
# Первый — bowwowforeach (#1 на момент сбора).
# Добавляй/убирай по мере обновления лидерборда.
TOP_SUBMISSION_IDS = [
    52318886,   # bowwowforeach  (#1)
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

EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL   = "https://www.kaggle.com/api/v1/competitions/episodes/{id}/replay"

SLEEP_LIST   = 1.0   # секунд между запросами списка
SLEEP_REPLAY = 1.0   # секунд между скачиванием replay

# Новый формат токена KGAT_xxx требует Bearer, старый username:key — Basic auth
def _auth_headers():
    """Возвращает headers с правильной аутентификацией."""
    if API_KEY.startswith('KGAT_') or API_KEY.startswith('kkk_'):
        return {'Authorization': f'Bearer {API_KEY}'}
    return {}  # используем auth= параметр (Basic auth)

def _make_request_get(url):
    """GET с автоопределением типа аутентификации."""
    h = _auth_headers()
    if h:
        return requests.get(url, headers=h, timeout=60)
    return requests.get(url, auth=(USERNAME, API_KEY), timeout=60)

def _make_request_post(url, json_body):
    """POST с автоопределением типа аутентификации."""
    h = {**_auth_headers(), 'Content-Type': 'application/json', 'Accept': 'application/json'}
    if _auth_headers():
        return requests.post(url, json=json_body, headers=h, timeout=30)
    return requests.post(url, auth=(USERNAME, API_KEY), json=json_body, headers=h, timeout=30)


# ── Шаг 1: собрать список эпизодов ──────────────────────────────────────────

def fetch_episode_ids(sub_id: int) -> list:
    """Возвращает список episode_id для данного submission."""
    r = _make_request_post(EPISODES_URL, {"submissionId": sub_id})
    if r.status_code == 200:
        episodes = r.json().get('episodes', [])
        return [ep['id'] for ep in episodes if ep.get('id')]
    else:
        print(f"  ⚠ ListEpisodes failed for {sub_id}: {r.status_code} {r.text[:200]}")
        return []


def collect_all_ids() -> list:
    """Собирает уникальные episode_id от всех TOP_SUBMISSION_IDS."""
    all_ids: set = set()
    for sub_id in TOP_SUBMISSION_IDS:
        print(f"Запрашиваем список для sub_id={sub_id} …", end=' ', flush=True)
        ids = fetch_episode_ids(sub_id)
        new = [i for i in ids if i not in all_ids]
        all_ids.update(ids)
        print(f"{len(ids)} эпизодов, новых: {len(new)}")
        time.sleep(SLEEP_LIST)
    return list(all_ids)


# ── Шаг 2: скачать replay-файлы ─────────────────────────────────────────────

MAX_DOWNLOAD = 500   # максимум файлов скачать за один запуск (0 = без ограничений)
                     # 500 ≈ 750 MB, хватит для кластеризации


def download_replays(episode_ids: list):
    """Скачивает replay JSON для каждого episode_id в DATA_DIR."""
    existing = {int(p.stem) for p in DATA_DIR.glob("*.json")
                if p.stem.isdigit()}
    to_download = [eid for eid in episode_ids if eid not in existing]

    # Лимит: берём равномерно от разных игроков (перемешиваем)
    import random as _rnd
    _rnd.shuffle(to_download)
    if MAX_DOWNLOAD and len(to_download) > MAX_DOWNLOAD:
        to_download = to_download[:MAX_DOWNLOAD]
        print(f"Лимит MAX_DOWNLOAD={MAX_DOWNLOAD}: скачаем {len(to_download)} файлов "
              f"≈ {len(to_download)*1.5/1024:.1f} GB")

    print(f"\nВсего уникальных эпизодов: {len(episode_ids)}")
    print(f"Уже скачано: {len(existing)}  |  К загрузке: {len(to_download)}")

    ok, fail = 0, 0
    for i, eid in enumerate(to_download, 1):
        url = REPLAY_URL.format(id=eid)
        r = _make_request_get(url)
        if r.status_code == 200:
            out = DATA_DIR / f"{eid}.json"
            out.write_bytes(r.content)
            ok += 1
        else:
            print(f"  ✗ [{i}/{len(to_download)}] {eid}: {r.status_code}")
            fail += 1

        if i % 50 == 0 or i == len(to_download):
            print(f"  [{i:>4}/{len(to_download)}]  ok={ok}  fail={fail}")

        time.sleep(SLEEP_REPLAY)

    print(f"\nСкачано: {ok}  Ошибок: {fail}")
    return ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Шаг 1: собираем списки эпизодов")
    print("=" * 60)
    all_ids = collect_all_ids()

    # сохраняем индекс
    idx_path = DATA_DIR / "episodes_index.json"
    with open(idx_path, 'w') as f:
        json.dump(all_ids, f)
    print(f"\nИндекс сохранён: {idx_path}  ({len(all_ids)} уникальных)")

    print("\n" + "=" * 60)
    print("Шаг 2: скачиваем replay-файлы")
    print("=" * 60)
    download_replays(all_ids)

    # итог
    downloaded = list(DATA_DIR.glob("[0-9]*.json"))
    print(f"\n✓ Итого в {DATA_DIR}: {len(downloaded)} файлов")


if __name__ == '__main__':
    main()
