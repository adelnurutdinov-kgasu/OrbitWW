"""
opponent_presets.py -- Opponent presets based on SwarmWeights grid-search winners.

Each preset is a named SwarmWeights configuration from eval_8winners.csv.
Action prediction uses the same scoring logic as swarm._action_value but
simplified (no zones needed) so it runs in microseconds per call.

Score(src, tgt, W) =
    margin
    + W.eta_bonus / eta^(1 - W.distance_comfort)
    + W.ships_weight * log1p(surplus)
    + W.activity_weight * max(0, surplus - W.idle_floor)
    + W.priority_bonus * tgt.production

Public interface:
  PRESETS              : dict {name: SwarmWeights}
  get_preset_actions(preset_name, state, opp_id) -> List[dict]
"""

import math
from typing import List, Dict

from swarm import SwarmWeights, DEFAULT_WEIGHTS
from shooting import fleet_speed_correct, segment_hits_sun, SUN_SAFETY
from projection import NEUTRAL_OWNER

# ── Preset definitions (from eval_8winners_20260502_050430.csv) ────────────
# Columns: activity_weight, idle_floor, distance_comfort, risk_tolerance,
#          ships_weight, eta_bonus, priority_bonus, stress_top_k, stress_gamma

PRESETS: Dict[str, SwarmWeights] = {
    # default SwarmWeights (baseline, winrate=0.38)
    'default': DEFAULT_WEIGHTS,

    # winner_seed19  winrate=0.45
    'w_seed19': SwarmWeights(
        activity_weight=2.14, idle_floor=38,  distance_comfort=0.28,
        risk_tolerance=0,     ships_weight=0.57, eta_bonus=51.96,
        priority_bonus=5.63,  stress_top_k=5,  stress_gamma=1.27,
    ),

    # winner_seed20  winrate=0.46
    'w_seed20': SwarmWeights(
        activity_weight=1.29, idle_floor=40,  distance_comfort=0.18,
        risk_tolerance=2,     ships_weight=1.37, eta_bonus=16.66,
        priority_bonus=2.84,  stress_top_k=3,  stress_gamma=1.50,
    ),

    # winner_seed39  winrate=0.46
    'w_seed39': SwarmWeights(
        activity_weight=2.79, idle_floor=37,  distance_comfort=0.19,
        risk_tolerance=1,     ships_weight=5.76, eta_bonus=14.33,
        priority_bonus=0.37,  stress_top_k=6,  stress_gamma=1.29,
    ),

    # winner_seed22  winrate=0.37
    'w_seed22': SwarmWeights(
        activity_weight=3.18, idle_floor=16,  distance_comfort=0.21,
        risk_tolerance=0,     ships_weight=4.40, eta_bonus=39.09,
        priority_bonus=1.84,  stress_top_k=5,  stress_gamma=1.34,
    ),
}

# ── Scoring parameters ─────────────────────────────────────────────────────
RESERVE_RATIO = 1.5    # keep production * ratio on source (mirror action_space.py)
MIN_SURPLUS   = 3      # minimum surplus to consider launching
MAX_ACTIONS   = 3      # max actions per preset
SPEED_SAMPLE  = 20     # reference ship count for speed estimate


def _surplus(p) -> int:
    return max(0, int(p.ships) - int(p.production * RESERVE_RATIO))


def _eta_est(src, tgt, ships: int) -> float:
    """Rough ETA: straight-line distance / fleet speed."""
    spd = fleet_speed_correct(max(1, ships))
    d   = math.hypot(tgt.x - src.x, tgt.y - src.y)
    d   = max(d - src.radius - tgt.radius, 1.0)
    return max(1.0, d / max(spd, 1e-6))


def _oppreset_overkill(tgt, risk_tolerance: int) -> int:
    """Minimum overkill buffer mirroring SAFETY_OVERKILL logic.

    risk_tolerance reduces the buffer (more aggressive, risker attacks).
    0 = conservative (+2 for owned, +1 for neutral)
    1 = slightly risky (+1 / +0)
    2+ = very aggressive (just > defender)
    """
    if tgt.owner == NEUTRAL_OWNER:
        return max(0, 1 - risk_tolerance)
    return max(0, 2 - risk_tolerance)


def _score(src, tgt, surplus: int, W: SwarmWeights) -> float:
    """Simplified action_value for (src->tgt) under SwarmWeights W."""
    ok      = _oppreset_overkill(tgt, W.risk_tolerance)
    margin  = surplus - float(tgt.ships) - ok
    eta     = _eta_est(src, tgt, surplus)
    comfort = max(0.0, min(1.0, W.distance_comfort))
    eta_term      = W.eta_bonus / (eta ** (1.0 - comfort))
    ships_term    = W.ships_weight * math.log1p(max(0, surplus))
    activity_term = W.activity_weight * max(0, surplus - W.idle_floor)
    prio_term     = W.priority_bonus * float(tgt.production)
    return margin + eta_term + ships_term + activity_term + prio_term


def _action_type(tgt, opp_id: int) -> str:
    if tgt.owner == opp_id:
        return 'reinforce'
    if tgt.owner == NEUTRAL_OWNER:
        return 'capture_neutral'
    return 'attack_enemy'


# ══════════════════════════════════════════════════════════════════════════
# Core generator
# ══════════════════════════════════════════════════════════════════════════

def get_preset_actions(preset_name: str, state, opp_id: int) -> List[dict]:
    """Predict opponent actions under the given preset (SwarmWeights).

    Algorithm:
    1. Gather opponent planets with surplus ships.
    2. For each (src, tgt) pair not blocked by sun, compute score.
    3. Greedily pick top-MAX_ACTIONS pairs (each target used at most once).

    Returns list of dicts: {from_id, target_id, ships, action_type}.
    """
    W = PRESETS.get(preset_name)
    if W is None:
        return []

    try:
        raw_planets = getattr(state, 'raw_planets', state.planets)
        all_planets = state.planets

        srcs = [(p, _surplus(p)) for p in raw_planets
                if p.owner == opp_id and _surplus(p) >= MIN_SURPLUS]
        if not srcs:
            return []

        # Candidates: all scored (src, tgt, score) pairs
        scored = []
        for src, surp in srcs:
            for tgt in all_planets:
                if tgt.id == src.id:
                    continue
                if segment_hits_sun(src.x, src.y, tgt.x, tgt.y, safety=SUN_SAFETY):
                    continue
                ok      = _oppreset_overkill(tgt, W.risk_tolerance)
                needed  = int(tgt.ships) + ok
                send    = surp if tgt.owner == opp_id else min(surp, max(MIN_SURPLUS, needed))
                if send < MIN_SURPLUS:
                    continue
                # For capture: need at least needed ships
                if tgt.owner != opp_id and surp < needed:
                    continue
                sc = _score(src, tgt, surp, W)
                scored.append((sc, src, tgt, send))

        if not scored:
            return []

        scored.sort(key=lambda x: -x[0])

        # Greedy pick: each (src, tgt) used at most once
        used_src = set()
        used_tgt = set()
        actions  = []
        for sc, src, tgt, send in scored:
            if len(actions) >= MAX_ACTIONS:
                break
            if src.id in used_src or tgt.id in used_tgt:
                continue
            used_src.add(src.id)
            used_tgt.add(tgt.id)
            actions.append({
                'from_id':     src.id,
                'target_id':   tgt.id,
                'ships':       max(1, int(send)),
                'action_type': _action_type(tgt, opp_id),
            })

        return actions

    except Exception:
        return []


__all__ = ['PRESETS', 'get_preset_actions', 'MAX_ACTIONS']
