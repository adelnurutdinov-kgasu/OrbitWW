#!/usr/bin/env python3
"""
33_test_all_filters.py — find ID filter that works for ListEpisodes.

Kaggle сказал "specify at least one ID filter" — значит существуют разные.
Пробуем все возможные. Также testing SubmissionService через valid filters.

Isaiah @ Tufa Labs:
  teamId: 15654628
  TeamMemberUserNames: pressman1
"""

from __future__ import annotations

import json
import os
import sys

import requests

ENDPOINTS = [
    'https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes',
    'https://www.kaggle.com/api/i/competitions.SubmissionService/ListSubmissions',
]

# Body вариации для ListEpisodes
EPISODE_BODIES = [
    {'submissionId': 52318886},  # known working: bowwowforeach
    {'userId': 15654628},          # как будто userId == teamId?
    {'userName': 'pressman1'},
    {'teamId': 15654628},
    {'userIds': [15654628]},
    {'userNames': ['pressman1']},
    {'submissionIds': [52318886]},
    {'episodeId': 75608555},       # one of our existing
    {'competitionId': 86535},
    {'competitionId': 86535, 'userName': 'pressman1'},
]

# Body вариации для ListSubmissions
SUB_BODIES = [
    {'userName': 'pressman1'},
    {'userId': 15654628},
    {'teamId': 15654628},
    {'competitionId': 86535, 'userName': 'pressman1'},
    {'competitionName': 'orbit-wars', 'userName': 'pressman1'},
]


def main():
    username = os.environ.get('KAGGLE_USERNAME', '')
    api_key = (os.environ.get('KAGGLE_KEY', '') or
                os.environ.get('KAGGLE_API_TOKEN', ''))
    if not username or not api_key:
        sys.exit('Set KAGGLE_USERNAME + KAGGLE_KEY/KAGGLE_API_TOKEN')

    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    for url in ENDPOINTS:
        endpoint = url.split('/')[-1]
        bodies = EPISODE_BODIES if 'Episode' in endpoint else SUB_BODIES
        print(f'\n══════ {endpoint} ══════')
        for body in bodies:
            try:
                r = requests.post(url, auth=(username, api_key),
                                   json=body, headers=headers, timeout=20)
            except Exception as e:
                print(f'  {body}: EXC {e!r}')
                continue
            status = r.status_code
            short = r.text[:300]
            # Check for success indicator
            star = ''
            try:
                d = r.json()
                if isinstance(d, dict):
                    for k in ('episodes', 'submissions', 'Episodes', 'Submissions'):
                        if d.get(k):
                            star = f'  ★★★ {k}: {len(d[k])}'
                            short = json.dumps(d, indent=2)[:400]
                            break
            except Exception:
                pass
            print(f'\n  body: {body}')
            print(f'  status: {status}{star}')
            print(f'  resp: {short[:300]}')


if __name__ == '__main__':
    main()
