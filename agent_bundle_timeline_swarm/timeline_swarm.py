"""
timeline_swarm.py — главная точка входа.

Склеивает все слои:
    state + player → build_timelines (L1)
                   → extract_opportunities (L2)
                   → compute_budgets (L3)
                   → run_auction (L4, многотиерный)
                   → validate (L5)
                   → emit_orders (L6)

Возвращает (plans, debug) в формате, совместимом с swarm_plan() из
agent_bundle_swarm, чтобы можно было подменить в агенте одной строкой.

Формат каждого плана соответствует тому, что ждёт
agent._execute_plan_atomically:
    {'mode': 'direct'|'transfer', 'sup_id': None, 'att_id': src,
     'tgt_id': dst, 'x_att': ships, 'x_sup': 0, 'success': True, ...}
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .timeline import PlanetTimeline, NEUTRAL
from .simulator import build_timelines
from .opportunities import OppParams, extract_opportunities, opportunities_summary
from .budgets import BudgetParams, compute_budgets
from .auction import AuctionParams, run_auction
from .validator import validate


@dataclass
class TimelineWeights:
    """Все параметры timeline-планировщика — для grid search."""
    horizon:             int   = 200
    accelerate_min_delay: int  = 8
    accelerate_value_discount: float = 0.8
    safety_threshold:    float = 5.0
    react_horizon:       int   = 30
    react_safety:        float = 1.5
    safety_floor:        int   = 2
    min_useful_strike:   int   = 3
    min_roi:             float = 0.05
    enable_validate:     bool  = True


DEFAULT_TIMELINE_WEIGHTS = TimelineWeights()


# ── Главная функция ───────────────────────────────────────────────────

def timeline_swarm_plan(state,
                        player: int,
                        targets: Optional[List[Any]] = None,
                        weights: Optional[TimelineWeights] = None,
                        current_turn: Optional[int] = None,
                        **kwargs) -> tuple:
    """Drop-in замена swarm_plan(): возвращает (plans, debug).

    Параметр `targets` сохранён для совместимости интерфейса, но
    игнорируется — мы рассматриваем ВСЕ планеты (а не только
    предсортированные zones). Цели возникают из таймлайнов естественно
    как функция владения и динамики.
    """
    if weights is None:
        weights = DEFAULT_TIMELINE_WEIGHTS
    if current_turn is None:
        current_turn = int(getattr(state, 'step', 0))

    raw = getattr(state, 'raw_planets', None) or state.planets
    our_planets = [p for p in raw if p.owner == player]
    if not our_planets:
        return [], {'reason': 'no_our_planets', 'planner': 'timeline_swarm'}

    our_by_id = {p.id: p for p in our_planets}
    all_by_id = {p.id: p for p in raw}

    # L1 ────────────────────────────────────────────────
    timelines = build_timelines(
        state, current_turn=current_turn,
        player=player, horizon=weights.horizon,
    )

    # L2 ────────────────────────────────────────────────
    opp_params = OppParams(
        horizon=weights.horizon,
        accelerate_min_delay=weights.accelerate_min_delay,
        safety_threshold=weights.safety_threshold,
        accelerate_value_discount=weights.accelerate_value_discount,
    )
    opportunities = extract_opportunities(
        timelines, state, current_turn, player, params=opp_params,
    )

    # L3 ────────────────────────────────────────────────
    budget_params = BudgetParams(
        horizon=weights.horizon,
        react_horizon=weights.react_horizon,
        safety_floor=weights.safety_floor,
        react_safety=weights.react_safety,
    )
    budgets = compute_budgets(
        timelines, list(our_by_id.keys()), current_turn, params=budget_params,
    )

    # L4 ────────────────────────────────────────────────
    auction_params = AuctionParams(
        horizon=weights.horizon,
        min_useful_strike=weights.min_useful_strike,
        min_roi=weights.min_roi,
    )
    bids, updated_timelines, updated_budgets, auction_stats = run_auction(
        opportunities=opportunities,
        timelines=timelines,
        budgets=budgets,
        our_planets_by_id=our_by_id,
        target_planets_by_id=all_by_id,
        current_turn=current_turn,
        player=player,
        auction_params=auction_params,
        opp_params=opp_params,
    )

    # L5 ────────────────────────────────────────────────
    if weights.enable_validate:
        bids, rolled_back = validate(
            bids, timelines, our_by_id, current_turn, player,
            horizon=weights.horizon,
        )
    else:
        rolled_back = []

    # L6 — Emit ─────────────────────────────────────────
    plans = []
    for b in bids:
        src = our_by_id.get(b.source_id)
        dst = all_by_id.get(b.target_id)
        if src is None or dst is None:
            continue
        plans.append(b.to_plan_dict(src, dst))

    debug = {
        'planner':           'timeline_swarm',
        'n_timelines':       len(timelines),
        'n_opportunities':   len(opportunities),
        'opp_summary':       opportunities_summary(opportunities),
        'budgets':           dict(budgets),
        'updated_budgets':   dict(updated_budgets),
        'auction_stats':     auction_stats,
        'n_committed':       len(plans),
        'n_rolled_back':     len(rolled_back),
        'committed_by_type': {},
        'plans_meta':        [{'type': b.opp_type, 'tier': b.opp_tier,
                               'src': b.source_id, 'tgt': b.target_id,
                               'ships': b.ships, 'eta': b.arrival_turn - current_turn,
                               'roi': round(b.roi, 3), 'value': round(b.value, 1)}
                              for b in bids],
    }
    for b in bids:
        debug['committed_by_type'][b.opp_type] = \
            debug['committed_by_type'].get(b.opp_type, 0) + 1

    return plans, debug


__all__ = ['TimelineWeights', 'DEFAULT_TIMELINE_WEIGHTS',
           'timeline_swarm_plan']
