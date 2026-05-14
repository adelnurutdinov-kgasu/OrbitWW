"""
action_space_v4.py -- PAR v4: Transport planner (Hybrid Net).

Uses build_transport_plan to create a small (~5-15 actions) but focused PAR.
Each plan assignment is expanded into 3 variants with different ship counts
(VARIATION_FACTORS).

Advantage: MCTS explores fewer nodes, but all are "meaningful" from a strategic
planning perspective.

Public interface:
  generate_actions_v4(state, player_id, max_travel_time) -> List[Action]
"""

from typing import List

from action_space import (
    Action, generate_actions,
    MIN_SEND, MAX_TRAVEL_TIME,
    _initial_by_id,
)
from transport_planner import build_transport_plan
from shooting import aim_hybrid

# ── Parameters v4 ──────────────────────────────────────────────────────────
VARIATION_FACTORS = [0.8, 1.0, 1.2]   # three ship-count variants per assignment
FALLBACK_TO_V1    = True               # fall back to v1 if plan is empty


def generate_actions_v4(state, player_id: int,
                        max_travel_time: int = MAX_TRAVEL_TIME) -> List[Action]:
    """PAR v4 = transport plan x VARIATION_FACTORS (+ v1 fallback if needed).

    For each assignment (src->tgt, ships) from build_transport_plan,
    generate 3 Action objects with ships * factor (0.8x, 1.0x, 1.2x).
    Total: ~5-15 Action (much smaller than v1, but strategically precise).

    Returns:
        List of Action (small but focused PAR).
    """
    plan = build_transport_plan(state, player_id, max_travel_time)

    raw_planets = getattr(state, 'raw_planets', state.planets)
    all_planets = state.planets
    omega       = state.omega
    ibd         = _initial_by_id(state)

    src_map = {p.id: p for p in raw_planets}
    tgt_map = {p.id: p for p in all_planets}

    actions: List[Action] = []

    for assignment in plan:
        src = src_map.get(assignment.src_id)
        tgt = tgt_map.get(assignment.tgt_id)
        if src is None or tgt is None:
            continue

        max_avail = max(0, src.ships - 1)

        for factor in VARIATION_FACTORS:
            ships = max(MIN_SEND, min(max_avail, int(round(assignment.ships * factor))))
            if ships > max_avail:
                continue

            result = aim_hybrid(src, tgt, ships, all_planets, ibd, omega)
            if result is None:
                continue
            angle, eta_actual, _, _ = result
            if int(eta_actual) > max_travel_time:
                continue

            actions.append(Action(
                from_id=src.id,
                target_id=tgt.id,
                ships=ships,
                angle=angle,
                travel_time=int(eta_actual),
                mode=assignment.mode,
            ))

    # Deduplicate by (from_id, target_id, ships)
    seen   = set()
    unique: List[Action] = []
    for a in actions:
        key = (a.from_id, a.target_id, a.ships)
        if key not in seen:
            seen.add(key)
            unique.append(a)

    if not unique and FALLBACK_TO_V1:
        return generate_actions(state, player_id, max_travel_time)

    return unique


__all__ = [
    'VARIATION_FACTORS', 'FALLBACK_TO_V1',
    'generate_actions_v4',
]
