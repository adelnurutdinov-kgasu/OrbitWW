"""
adapter.py — точная совместимость с интерфейсом swarm.swarm_plan().

Используется для monkey-patch:

    from agent_bundle_timeline_swarm.adapter import timeline_swarm_plan_adapter
    our_module.swarm_plan = timeline_swarm_plan_adapter

Сигнатура и debug-словарь приведены к тому формату, который ожидает
agent_bundle_swarm/agent.py:
    attack_plans, swarm_dbg = swarm_plan(state, player, targets,
        weights=..., priority_lookup=..., max_targets=..., deadline=...)

И затем читает: swarm_dbg['n_candidates'], ['n_committed'],
['n_transfers'], ['n_unfunded'], ['stress'].
"""

from typing import Any, Dict, List, Optional

from .timeline_swarm import timeline_swarm_plan, DEFAULT_TIMELINE_WEIGHTS


def timeline_swarm_plan_adapter(state,
                                player: int,
                                targets: Optional[List[Any]] = None,
                                weights=None,           # SwarmWeights — игнорируем
                                priority_lookup=None,   # не используется
                                horizon=None,            # не используется
                                max_targets=None,        # не используется
                                deadline=None,           # не используется
                                **kwargs):
    """Drop-in замена swarm.swarm_plan().

    Игнорирует SwarmWeights (у timeline-swarm свои веса), но принимает
    их для совместимости.
    """
    plans, debug = timeline_swarm_plan(
        state, player, targets,
        weights=None,  # используем DEFAULT_TIMELINE_WEIGHTS
    )

    # Adapter: подмена ключей debug под ожидания agent.py.
    n_cand = debug.get('n_opportunities', 0)
    n_comm = debug.get('n_committed', 0)
    n_trans = sum(1 for p in plans if p.get('mode') == 'transfer')
    n_dir = sum(1 for p in plans if p.get('mode') == 'direct')
    swarm_dbg = {
        'n_candidates':  n_cand,
        'n_committed':   n_comm,
        'n_transfers':   n_trans,
        'n_direct':      n_dir,
        'n_unfunded':    0,                  # отсутствует у нас
        'stress':        {},                 # у нас нет stress-словаря
        'neighbor_stress': {},
        'remaining':     debug.get('updated_budgets', {}),
        'captured':      [p['tgt_id'] for p in plans],
        # Расширения timeline-планера для тех, кто хочет читать:
        '_timeline_planner':     True,
        '_opp_summary':          debug.get('opp_summary', {}),
        '_committed_by_type':    debug.get('committed_by_type', {}),
        '_plans_meta':           debug.get('plans_meta', []),
    }
    return plans, swarm_dbg


__all__ = ['timeline_swarm_plan_adapter']
