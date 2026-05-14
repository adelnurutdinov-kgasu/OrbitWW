"""
action_space_v3.py -- PAR v3: Two-wave attacks (Partial-Prepare).

Idea: a "preparation" first wave weakens the defense, second wave finishes.
Wave-1 (partial) -> flies with ships_1 < min_capture, intentionally insufficient.
Wave-2 (completing) -> launched next turn, finishes off the weakened defender.

MCTSNode tracks pending_partials -- list of PartialAction already launched
as first waves, waiting for completing actions.

Public interface:
  class PartialAction
  generate_partial_actions(state, player_id, ...) -> List[PartialAction]
  generate_completing_actions(pending, state, player_id, ...) -> List[Action]
  generate_actions_v3(state, player_id, pending, ...) -> List[Action|PartialAction]
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from action_space import (
    Action, generate_actions,
    MIN_SEND, MAX_TRAVEL_TIME, OVERKILL_NEUTRAL, OVERKILL_OWNED,
    _defender_at_arrival, _initial_by_id,
)
from shooting import aim_hybrid, segment_hits_sun, SUN_SAFETY
from force import _rendezvous_eta
from projection import NEUTRAL_OWNER

# ── Parameters v3 ──────────────────────────────────────────────────────────
WAVE1_FACTOR    = 0.6   # wave-1 sends factor * min_capture ships
WAVE2_MAX_DELAY = 5     # wave-2 must arrive no later than wave-1 + this many turns
MAX_PARTIALS    = 15    # max PartialAction per call


@dataclass
class PartialAction(Action):
    """First wave of a two-wave attack.

    Extends Action with additional fields for tracking the required second wave.
    Inherited Action fields: from_id, target_id, ships, angle, travel_time, mode
    """
    partial_ships_needed: int = 0   # how many more ships are needed to capture
    partial_eta:          int = 0   # ETA of the first wave

    def __repr__(self) -> str:
        return (f"PartialAction({self.from_id}->{self.target_id} "
                f"wave1={self.ships} need_more={self.partial_ships_needed} "
                f"eta={self.travel_time})")


def generate_partial_actions(state, player_id: int,
                              max_travel_time: int = MAX_TRAVEL_TIME) -> List[PartialAction]:
    """Generate first waves of two-wave attacks.

    Logic:
    - For each capture pair (src, tgt) compute min_capture.
    - Wave-1 sends WAVE1_FACTOR * min_capture ships (not enough alone to capture).
    - After wave-1 arrives, defender is weakened -> wave-2 needs fewer ships.

    Returns:
        List of PartialAction (first waves), empty if no candidates.
    """
    raw_planets = getattr(state, 'raw_planets', state.planets)
    src_planets = [p for p in raw_planets
                   if p.owner == player_id and p.ships >= MIN_SEND * 2]
    all_planets = state.planets
    omega       = state.omega
    ibd         = _initial_by_id(state)

    partials: List[PartialAction] = []

    for src in src_planets:
        for tgt in all_planets:
            if tgt.id == src.id:
                continue
            if tgt.owner == player_id:
                continue   # don't do partial reinforce

            if segment_hits_sun(src.x, src.y, tgt.x, tgt.y, safety=SUN_SAFETY):
                continue

            eta_est, _ = _rendezvous_eta(src, tgt, max(MIN_SEND, src.ships // 2), omega)
            if eta_est > max_travel_time:
                continue

            overkill = OVERKILL_NEUTRAL if tgt.owner == NEUTRAL_OWNER else OVERKILL_OWNED
            defender = _defender_at_arrival(state, tgt, eta_est)
            min_cap  = int(math.ceil(defender)) + overkill

            # Wave-1: deliberately less than min_capture
            wave1_ships = max(MIN_SEND, int(math.floor(min_cap * WAVE1_FACTOR)))
            if wave1_ships >= min_cap:
                continue   # no point splitting if wave1 already enough
            if wave1_ships > src.ships - 1:
                continue

            # Exact angle for wave-1
            result = aim_hybrid(src, tgt, wave1_ships, all_planets, ibd, omega)
            if result is None:
                continue
            angle, eta_actual, _, _ = result
            if int(eta_actual) > max_travel_time:
                continue

            # Ships still needed for wave-2 (wave-1 weakens by wave1_ships)
            defender_after = max(0.0, defender - wave1_ships)
            ships_needed   = int(math.ceil(defender_after)) + overkill

            partials.append(PartialAction(
                from_id=src.id,
                target_id=tgt.id,
                ships=wave1_ships,
                angle=angle,
                travel_time=int(eta_actual),
                mode='capture',
                partial_ships_needed=ships_needed,
                partial_eta=int(eta_actual),
            ))

            if len(partials) >= MAX_PARTIALS:
                return partials

    return partials


def generate_completing_actions(pending_partials: List[PartialAction],
                                state, player_id: int,
                                max_travel_time: int = MAX_TRAVEL_TIME) -> List[Action]:
    """Generate second waves for already-launched partial attacks.

    For each PartialAction in pending_partials, find planets that can send
    a second wave arriving in the window [eta_wave1, eta_wave1 + WAVE2_MAX_DELAY].

    Returns:
        List of plain Action objects (second waves).
    """
    if not pending_partials:
        return []

    raw_planets = getattr(state, 'raw_planets', state.planets)
    all_planets = state.planets
    omega       = state.omega
    ibd         = _initial_by_id(state)

    completing: List[Action] = []

    for partial in pending_partials:
        tgt_map = {p.id: p for p in all_planets}
        tgt = tgt_map.get(partial.target_id)
        if tgt is None:
            continue

        deadline_eta = partial.partial_eta + WAVE2_MAX_DELAY

        for src in raw_planets:
            if src.owner != player_id:
                continue
            if src.id == partial.from_id:
                continue   # same planet already sent wave-1
            if src.ships < MIN_SEND:
                continue

            if segment_hits_sun(src.x, src.y, tgt.x, tgt.y, safety=SUN_SAFETY):
                continue

            ships = min(src.ships - 1, partial.partial_ships_needed + 2)
            if ships < MIN_SEND:
                continue

            eta_est, _ = _rendezvous_eta(src, tgt, ships, omega)
            if eta_est > deadline_eta:
                continue

            result = aim_hybrid(src, tgt, ships, all_planets, ibd, omega)
            if result is None:
                continue
            angle, eta_actual, _, _ = result
            if int(eta_actual) > deadline_eta:
                continue

            completing.append(Action(
                from_id=src.id,
                target_id=partial.target_id,
                ships=ships,
                angle=angle,
                travel_time=int(eta_actual),
                mode='capture',
            ))

    return completing


def generate_actions_v3(state, player_id: int,
                        pending_partials: Optional[List[PartialAction]] = None,
                        max_travel_time: int = MAX_TRAVEL_TIME) -> List:
    """PAR v3 = v1 + partial first waves + second waves for pending partials.

    Args:
        pending_partials: PartialAction from previous turns (already in flight).

    Returns:
        Mixed list of Action and PartialAction.
    """
    v1         = generate_actions(state, player_id, max_travel_time)
    partials   = generate_partial_actions(state, player_id, max_travel_time)
    completing = []
    if pending_partials:
        completing = generate_completing_actions(
            pending_partials, state, player_id, max_travel_time
        )
    return v1 + partials + completing


__all__ = [
    'PartialAction',
    'WAVE1_FACTOR', 'WAVE2_MAX_DELAY', 'MAX_PARTIALS',
    'generate_partial_actions', 'generate_completing_actions', 'generate_actions_v3',
]
