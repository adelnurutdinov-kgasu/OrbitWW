#!/usr/bin/env python3
"""
Check how many replays (episodes) are available for a Kaggle Orbit Wars submission.

Usage:
    python check_replays.py                # uses the default SUBMISSION_ID below
    python check_replays.py 15654628       # check a specific submissionId
    python check_replays.py 15654628 -v    # also list the most recent episode IDs

Credentials (tried in this order):
    1. Environment variables  KAGGLE_USERNAME  and  KAGGLE_KEY
    2. kaggle.json  in the current folder or in  ~/.kaggle/
    3. No auth (the internal endpoint sometimes answers without it)

Get your key at: kaggle.com -> Settings -> API -> "Create New Token"  (downloads kaggle.json)
"""

import os
import sys
import json
from pathlib import Path

import requests

# ── Config ───────────────────────────────────────────────────────────
SUBMISSION_ID = 15654628  # default; override by passing an id on the command line
ENDPOINT = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def load_credentials():
    """Return (username, key) tuple, or None if nothing was found."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return (user, key)

    for path in (Path("kaggle.json"), Path.home() / ".kaggle" / "kaggle.json"):
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return (data["username"], data["key"])
            except (KeyError, json.JSONDecodeError):
                print(f"WARNING: {path} exists but is not valid kaggle.json")
    return None


def list_episodes(submission_id, auth):
    """Call ListEpisodes and return the parsed JSON response."""
    resp = requests.post(
        ENDPOINT,
        json={"submissionId": submission_id},
        auth=auth,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    # ── parse arguments: optional submissionId, optional -v / --verbose ──
    args = sys.argv[1:]

    # Drop an inline "# comment ..." - zsh does NOT strip these by default,
    # so a pasted command like  `check_replays.py 123  # note`  would otherwise
    # pass '#' and the note text as arguments.
    for i, tok in enumerate(args):
        if tok.startswith("#"):
            args = args[:i]
            break

    verbose = False
    for flag in ("-v", "--verbose"):
        while flag in args:
            verbose = True
            args.remove(flag)

    if args:
        if not args[0].lstrip("-").isdigit():
            print(f"Not a valid submissionId: {args[0]!r}")
            print("Usage: python check_replays.py <submissionId> [-v]")
            sys.exit(2)
        sub_id = int(args[0])
    else:
        sub_id = SUBMISSION_ID

    # ── credentials ──
    auth = load_credentials()
    if auth is None:
        print("WARNING: no Kaggle credentials found - trying without auth.")
        print("  Set KAGGLE_USERNAME / KAGGLE_KEY, or put kaggle.json in this folder or ~/.kaggle/")
    else:
        print(f"Authenticated as: {auth[0]}")

    # ── request ──
    print(f"Checking submission {sub_id} ...\n")
    try:
        data = list_episodes(sub_id, auth)
    except requests.HTTPError as e:
        print(f"Request failed: HTTP {e.response.status_code}")
        print(e.response.text[:500])
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Network error: {e}")
        sys.exit(1)

    episodes = data.get("episodes", [])
    print(f"  Submission {sub_id}: {len(episodes)} replays (episodes) available")

    # ── optional extra detail ──
    if verbose and episodes:
        recent = sorted(episodes, key=lambda e: e.get("createTime", ""), reverse=True)
        print("\n  Most recent episode IDs:")
        for ep in recent[:10]:
            print(f"    {ep.get('id')}   {ep.get('createTime', '')[:19]}")

    if not episodes:
        print("\n  (0 episodes - check the submissionId is correct and currently on the leaderboard,")
        print("   or that your credentials are valid.)")


if __name__ == "__main__":
    main()
