#!/usr/bin/env python3
"""
32_test_team_episodes.py — тест: принимает ли ListEpisodes teamId напрямую?

ГИПОТЕЗА (пользователя)
───────────────────────
SubmissionIDs могут быть **private** — Kaggle их не отдаёт. А teamIDs
**публичные** (есть в CSV экспорте leaderboard). Может EpisodeService
принимает teamId напрямую или есть отдельный endpoint.

ТЕСТ
────
Для teamId=15654628 (Isaiah @ Tufa Labs #1) пробуем разные body и
endpoints. Если хоть один возвращает episodes — проблема решена.
"""

from __future__ import annotations

import json
import os
import sys

import requests

TEAM_ID = 15654628  # Isaiah @ Tufa Labs #1

TESTS = [
    # endpoint, body
    ('https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes',
     {'teamId': TEAM_ID}),
    ('https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes',
     {'teamId': TEAM_ID, 'competitionId': 86535}),
    ('https://www.kaggle.com/api/i/competitions.EpisodeService/ListTeamEpisodes',
     {'teamId': TEAM_ID}),
    ('https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes',
     {'TeamId': TEAM_ID}),
    ('https://www.kaggle.com/api/i/competitions.EpisodeService/GetTeamEpisodes',
     {'teamId': TEAM_ID}),
    ('https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodesByTeam',
     {'teamId': TEAM_ID}),
]


def main():
    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = (os.environ.get('KAGGLE_KEY', '') or
               os.environ.get('KAGGLE_API_TOKEN', ''))
    if not username or not api_key:
        sys.exit('⚠ KAGGLE_USERNAME + KAGGLE_KEY или KAGGLE_API_TOKEN')

    print(f'Testing teamId={TEAM_ID} (Isaiah @ Tufa Labs #1)')
    print(f'Auth: Basic ({username}, {api_key[:8]}...)\n')

    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    for url, body in TESTS:
        print(f'\n{"─" * 70}')
        endpoint = url.split('/')[-1]
        print(f'POST {endpoint}')
        print(f'body: {body}')

        try:
            r = requests.post(url, auth=(username, api_key), json=body,
                              headers=headers, timeout=30)
        except Exception as e:
            print(f'  EXCEPTION: {e!r}')
            continue

        print(f'  status: {r.status_code}')
        body_text = r.text[:600]
        print(f'  body (first 600): {body_text}')
        try:
            d = r.json()
            for k in ('episodes', 'Episodes', 'results', 'data'):
                if k in d and isinstance(d[k], list):
                    print(f'  ★ field "{k}": {len(d[k])} items')
                    if d[k]:
                        print(f'    first: {json.dumps(d[k][0], indent=2)[:300]}')
                    break
        except json.JSONDecodeError:
            pass


if __name__ == '__main__':
    main()
