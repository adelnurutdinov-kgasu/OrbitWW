"""
transport_planner.py -- Greedy transport planner for ship distribution.

Finds the best fleet distribution plan via greedy bipartite matching:
each of our "surplus" planets (src) is assigned the most valuable target
by the criterion production / (defender + 1).

Used in action_space_v4 to generate a small focused PAR.

Public interface:
  TransportPlan   -- one src -> tgt assignment with details
  build_transport_plan(state, player_id) -> List[TransportPlan]
"""

import math
from dataclasses import dataclass
from typing import List, Set

from force import _rendezvous_eta
from shooting import segment_hits_sun, SUN_SAFETY
from action_space import (
    MIN_SEND, MAX_TRAVEL_TIME, RESERVE_FACTOR,
    _defender_at_arrival,
)
from projection import NEUTRAL_OWNER

# ── Planner parameters ─────────────────────────────────────────────────────
MAX_PLAN_SIZE  = 8     # maximum number of assignments in the plan
MIN_SURPLUS    = 5     # minimum surplus to include a planet as a source
PRIORITY_DECAY = 0.9   # priority decay for already-assigned targets


@dataclass
class TransportPlan:
    """One assignment in the transport plan."""
    src_id:   int
    tgt_id:   int
    ships:    int
    eta:      float
    priority: float    # production/defender -- "value" of the capture
    mode:     str      # 'capture' | 'reinforce'


def _target_priority(tgt, state, eta: float) -> float:
    """Target priority: production / (defender_at_arrival + 1).

    High priority = lots of production, few ships on planet.
    Neutral planets get a bonus (production is "free").
    """
    defender = _defender_at_arrival(state, tgt, eta)
    prod     = float(getattr(tgt, 'production', 1))
    bonus    = 1.2 if tgt.owner == NEUTRAL_OWNER else 1.0
    return bonus * prod / (defender + 1.0)


def build_transport_plan(state, player_id: int,
                          max_travel_time: int = MAX_TRAVEL_TIME) -> List[TransportPlan]:
    """Greedy bipartite matching: src (our surplus planets) -> tgt.

    Algorithm:
    1. Build list of source planets (ours with ships > production*RESERVE_FACTOR + MIN_SURPLUS).
    2. Build list of target planets (not ours, or ours with deficit).
    3. For each source, pick the best unassigned target by priority.
    4. Repeat while there are sources and targets, up to MAX_PLAN_SIZE iterations.

    Returns:
        List of TransportPlan in descending priority order.
    """
    raw_planets = getattr(state, 'raw_planets', state.planets)
    all_planets = state.planets
    omega       = state.omega

    # Sources: our planets with surplus ships
    sources = []
    for p in raw_planets:
        if p.owner != player_id:
            continue
        surplus = p.ships - int(p.production * RESERVE_FACTOR)
        if surplus < MIN_SURPLUS:
            continue
        sources.append((surplus, p))

    if not sources:
        return []

    # Targets: all non-player planets + our planets with low garrison
    targets = []
    for p in all_planets:
        if p.owner == player_id:
            # reinforce only if explicit deficit
            if p.ships > p.production * 2:
                continue
        targets.append(p)

    if not targets:
        return []

    # Sort sources by surplus (larger surplus = better attacker)
    sources.sort(key=lambda x: -x[0])

    used_targets: Set[int] = set()
    plan: List[TransportPlan] = []

    for surplus, src in sources:
        if len(plan) >= MAX_PLAN_SIZE:
            break

        best_prio  = -1.0
        best_tgt   = None
        best_eta   = 0.0
        best_ships = 0

        for tgt in targets:
            if tgt.id == src.id:
                continue
            if tgt.id in used_targets:
                continue

            if segment_hits_sun(src.x, src.y, tgt.x, tgt.y, safety=SUN_SAFETY):
                continue

            send = max(MIN_SEND, min(surplus, int(src.ships - 1)))
            eta, _ = _rendezvous_eta(src, tgt, send, omega)
            if eta > max_travel_time:
                continue

            prio = _target_priority(tgt, state, eta)
            if prio > best_prio:
                best_prio  = prio
                best_tgt   = tgt
                best_eta   = eta
                best_ships = send

        if best_tgt is None:
            continue

        mode = 'reinforce' if best_tgt.owner == player_id else 'capture'
        plan.append(TransportPlan(
            src_id=src.id,
            tgt_id=best_tgt.id,
            ships=best_ships,
            eta=best_eta,
            priority=best_prio,
            mode=mode,
        ))
        used_targets.add(best_tgt.id)

    return sorted(plan, key=lambda p: -p.priority)


__all__ = [
    'TransportPlan',
    'MAX_PLAN_SIZE', 'MIN_SURPLUS',
    'build_transport_plan',
]
