"""
action_space_v2.py -- PAR v2: Paired attacks (Парные атаки).

Extends v1 with PairedAction -- simultaneous launch from two planets to one
target. The pair arrives almost synchronously (|etaA-etaB| <= MAX_ETA_DIFF),
creating a coordinated strike effect.

Public interface:
  class PairedAction
  generate_paired_actions(state, player_id, max_travel_time) -> List[PairedAction]
  generate_actions_v2(state, player_id, max_travel_time)    -> List[Action|PairedAction]
"""

import math
from dataclasses import dataclass
from typing import List

from action_space import (
    Action, generate_actions,
    MIN_SEND, MAX_TRAVEL_TIME, OVERKILL_NEUTRAL, OVERKILL_OWNED,
    _defender_at_arrival,
)
from projection import NEUTRAL_OWNER

# ── Parameters v2 ──────────────────────────────────────────────────────────
MAX_ETA_DIFF = 3     # max difference in arrival times between waves (turns)
MAX_PAIRS    = 20    # max PairedAction per call (speed guard)


@dataclass
class PairedAction:
    """Coordinated strike from two planets onto one target.

    action_a and action_b fly from different planets to the same target_id,
    arriving nearly simultaneously (|eta_a - eta_b| <= MAX_ETA_DIFF).
    """
    action_a:  Action
    action_b:  Action
    target_id: int
    ships:     int    # combined ships (a + b)
    mode:      str    # 'capture' | 'reinforce'

    def to_moves(self) -> list:
        """Two launches: [[from_a, angle_a, ships_a], [from_b, angle_b, ships_b]]."""
        return self.action_a.to_moves() + self.action_b.to_moves()

    def __repr__(self) -> str:
        return (f"PairedAction({self.mode} "
                f"{self.action_a.from_id}+{self.action_b.from_id}->{self.target_id} "
                f"ships={self.ships} eta=({self.action_a.travel_time},{self.action_b.travel_time}))")


def generate_paired_actions(state, player_id: int,
                             max_travel_time: int = MAX_TRAVEL_TIME) -> List[PairedAction]:
    """Generate PairedAction -- coordinated double strikes.

    Logic:
    1. Take v1 PAR, group by target_id.
    2. For each target, iterate pairs (act_a, act_b) from different planets.
    3. Check ETA compatibility and combined ship sufficiency.

    Returns:
        List of PairedAction sorted by ships (fewer = more economical).
    """
    v1 = generate_actions(state, player_id, max_travel_time)

    # Group by target
    by_target: dict = {}
    for a in v1:
        by_target.setdefault(a.target_id, []).append(a)

    all_planets = state.planets
    tgt_map = {p.id: p for p in all_planets}

    pairs: List[PairedAction] = []

    for tgt_id, acts in by_target.items():
        if len(acts) < 2:
            continue
        tgt = tgt_map.get(tgt_id)
        if tgt is None:
            continue

        overkill = OVERKILL_NEUTRAL if tgt.owner == NEUTRAL_OWNER else OVERKILL_OWNED

        seen = set()
        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                if a.from_id == b.from_id:
                    continue
                if abs(a.travel_time - b.travel_time) > MAX_ETA_DIFF:
                    continue

                # Deduplicate: one planet pair can yield multiple ship-count combos
                pair_key = (min(a.from_id, b.from_id), max(a.from_id, b.from_id))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # Check combined ships vs defender
                avg_eta  = (a.travel_time + b.travel_time) / 2.0
                defender = _defender_at_arrival(state, tgt, avg_eta)
                combined = a.ships + b.ships
                if combined < math.ceil(defender) + overkill:
                    continue

                mode = 'reinforce' if tgt.owner == player_id else 'capture'
                pairs.append(PairedAction(
                    action_a=a,
                    action_b=b,
                    target_id=tgt_id,
                    ships=combined,
                    mode=mode,
                ))

                if len(pairs) >= MAX_PAIRS:
                    return sorted(pairs, key=lambda p: p.ships)

    return sorted(pairs, key=lambda p: p.ships)


def generate_actions_v2(state, player_id: int,
                        max_travel_time: int = MAX_TRAVEL_TIME) -> List:
    """PAR v2 = v1 (single launches) + paired attacks.

    Returns:
        Mixed list of Action and PairedAction.
    """
    v1    = generate_actions(state, player_id, max_travel_time)
    pairs = generate_paired_actions(state, player_id, max_travel_time)
    return v1 + pairs


__all__ = [
    'PairedAction', 'MAX_ETA_DIFF', 'MAX_PAIRS',
    'generate_paired_actions', 'generate_actions_v2',
]
