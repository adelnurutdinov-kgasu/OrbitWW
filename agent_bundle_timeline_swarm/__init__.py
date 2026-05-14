"""
agent_bundle_timeline_swarm — экспериментальный планировщик на основе
полных таймлайнов владения планет.

Главная точка входа:
    from agent_bundle_timeline_swarm import timeline_swarm_plan
    plans, debug = timeline_swarm_plan(state, player, targets, weights)

Архитектура (см. README.md):
    L0 World          → текущее состояние + летящие флоты
    L1 Timelines      → simulator.build_timelines
    L2 Opportunities  → opportunities.extract_opportunities
    L3 Budgets        → budgets.compute_budgets
    L4 Auction        → auction.run_auction
    L5 Validate       → validator.validate
    L6 Orders         → timeline_swarm.emit_orders
"""

from .timeline import Event, PlanetTimeline
from .simulator import build_timelines
from .opportunities import Opportunity, extract_opportunities
from .budgets import compute_budgets
from .auction import Bid, run_auction
from .validator import validate
from .timeline_swarm import (
    TimelineWeights,
    DEFAULT_TIMELINE_WEIGHTS,
    timeline_swarm_plan,
)
from .adapter import timeline_swarm_plan_adapter

__all__ = [
    'Event', 'PlanetTimeline',
    'build_timelines',
    'Opportunity', 'extract_opportunities',
    'compute_budgets',
    'Bid', 'run_auction',
    'validate',
    'TimelineWeights', 'DEFAULT_TIMELINE_WEIGHTS',
    'timeline_swarm_plan',
    'timeline_swarm_plan_adapter',
]
