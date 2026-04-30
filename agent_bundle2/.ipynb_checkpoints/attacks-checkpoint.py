"""
Блок оценки атак: direct / multi_sync / pipeline.

Ключевые экспорты:
  eval_direct(state, att, tgt)           -> dict | None
  eval_multi_sync(state, sup, att, tgt)  -> dict | None
  eval_pipe(state, sup, att, tgt)        -> dict | None
  all_plans(state, tgt, ours, horizon)   -> list[dict]
  best_attacks(state, player, targets)   -> list[dict]  ← для агента
"""

import math
from itertools import product as _product

from orbit_sim import segment_hits_sun
from force import _rendezvous_eta

ATTACK_HORIZON        = 80
SUN_SAFETY_PIPE       = 1.5
PIPELINE_MAX_ANGLE_DEG = 120.0


def _eta(src, dst, ships, omega):
    e, _ = _rendezvous_eta(src, dst, max(1, ships), omega)
    return float(e)


def _seg_blocked(a, b, safety=SUN_SAFETY_PIPE):
    return segment_hits_sun(a.x, a.y, b.x, b.y, safety=safety)


def _angle_at_att(sup, att, tgt):
    """Угол ∠(sup-att-tgt) в градусах."""
    v1x, v1y = att.x - sup.x, att.y - sup.y
    v2x, v2y = tgt.x - att.x, tgt.y - att.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = (v1x * v2x + v1y * v2y) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def eval_direct(state, att, tgt):
    """Прямая атака att → tgt."""
    if _seg_blocked(att, tgt):
        return None
    eta = _eta(att, tgt, att.ships, state.omega)
    defender = tgt.ships + tgt.production * eta
    return {
        'mode':      'direct',
        'sup_id':    None, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_sa':    0.0,  'eta_at': eta,    't_total': eta,
        'x_sup':     0,    'x_att':  att.ships,  'prod_att': att.production,
        'x_tgt':     tgt.ships, 'prod_tgt': tgt.production,
        'strike':    att.ships, 'defender': defender,
        'margin':    att.ships - defender,
        'x_sup_min': max(0.0, defender - att.ships),
        'success':   att.ships > defender,
        'relay_reason': None,
    }


def eval_multi_sync(state, sup, att, tgt):
    """
    Синхронная атака: sup и att шлют в tgt напрямую, быстрый ждёт медленного.
    """
    if _seg_blocked(sup, tgt) or _seg_blocked(att, tgt):
        return None
    omega = state.omega
    eta_st = _eta(sup, tgt, sup.ships, omega)
    eta_at = _eta(att, tgt, att.ships, omega)
    t_sync = max(eta_st, eta_at)
    sup_part = sup.ships + sup.production * max(0.0, t_sync - eta_st)
    att_part = att.ships + att.production * max(0.0, t_sync - eta_at)
    strike   = sup_part + att_part
    defender = tgt.ships + tgt.production * t_sync
    return {
        'mode':     'multi_sync',
        'sup_id':   sup.id, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_st':   eta_st, 'eta_at': eta_at, 't_total': t_sync,
        'x_sup':    sup.ships, 'x_att': att.ships,
        'prod_sup': sup.production, 'prod_att': att.production,
        'x_tgt':    tgt.ships, 'prod_tgt': tgt.production,
        'strike':   strike, 'defender': defender,
        'margin':   strike - defender,
        'success':  strike > defender,
        'sup_part': sup_part, 'att_part': att_part,
        'relay_reason': None,
    }


def eval_pipe(state, sup, att, tgt):
    """
    Pipeline relay: sup → att (усиление) → tgt (удар).
    Фильтрует случаи, когда multi_sync лучше.
    """
    if _seg_blocked(sup, att) or _seg_blocked(att, tgt):
        return None

    sup_dir_blocked = _seg_blocked(sup, tgt)
    angle_deg = _angle_at_att(sup, att, tgt)
    if not sup_dir_blocked and angle_deg > PIPELINE_MAX_ANGLE_DEG:
        return None

    omega   = state.omega
    eta_sa  = _eta(sup, att, sup.ships, omega)
    boosted = sup.ships + att.ships + att.production * eta_sa
    eta_at  = _eta(att, tgt, boosted, omega)
    t_total = eta_sa + eta_at
    defender    = tgt.ships + tgt.production * t_total
    margin_relay = boosted - defender
    x_sup_min    = max(0.0, defender - att.ships - att.production * eta_sa)

    if sup_dir_blocked:
        relay_reason = f'sup→tgt blocked by sun (angle={angle_deg:.0f}°)'
    else:
        ms = eval_multi_sync(state, sup, att, tgt)
        if ms is None:
            relay_reason = f'multi_sync infeasible (angle={angle_deg:.0f}°)'
        elif margin_relay > ms['margin'] + 1e-6:
            relay_reason = (f'relay margin > multi_sync ({margin_relay:.1f} > {ms["margin"]:.1f})'
                            f', angle={angle_deg:.0f}°')
        else:
            return None

    return {
        'mode':        'pipeline',
        'sup_id':      sup.id, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_sa':      eta_sa, 'eta_at': eta_at, 't_total': t_total,
        'x_sup':       sup.ships, 'x_att': att.ships, 'prod_att': att.production,
        'x_tgt':       tgt.ships, 'prod_tgt': tgt.production,
        'strike':      boosted, 'defender': defender,
        'margin':      margin_relay,
        'x_sup_min':   x_sup_min,
        'sup_surplus': sup.ships - x_sup_min,
        'success':     boosted > defender,
        'angle_deg':   angle_deg,
        'relay_reason': relay_reason,
    }


def all_plans(state, tgt, ours, horizon=ATTACK_HORIZON):
    """Все возможные планы атаки на tgt (direct + multi_sync + pipeline)."""
    plans = []
    for att in ours:
        if att.id == tgt.id:
            continue
        d = eval_direct(state, att, tgt)
        if d and d['t_total'] <= horizon:
            plans.append(d)

    for att, sup in _product(ours, ours):
        if sup.id in (att.id, tgt.id) or att.id == tgt.id:
            continue
        ms = eval_multi_sync(state, sup, att, tgt)
        if ms and ms['t_total'] <= horizon:
            plans.append(ms)
        pp = eval_pipe(state, sup, att, tgt)
        if pp and pp['t_total'] <= horizon:
            plans.append(pp)
    return plans


def best_attacks(state, player, targets, top_n=6, horizon=ATTACK_HORIZON):
    """
    Для каждой цели из targets выбирает лучший план атаки.
    Возвращает список планов (не более top_n), отсортированных по margin убыванию.

    Каждый план — dict с полями: mode, sup_id, att_id, tgt_id, margin, success, ...
    """
    ours = [p for p in state.planets if p.owner == player]
    if not ours or not targets:
        return []

    best_per_target = []
    for tgt in targets:
        plans = all_plans(state, tgt, ours, horizon=horizon)
        if not plans:
            continue
        successful = [pl for pl in plans if pl['success']]
        best = (max(successful, key=lambda x: x['margin'])
                if successful
                else max(plans, key=lambda x: x['margin']))
        best_per_target.append(best)

    best_per_target.sort(key=lambda x: (-int(x['success']), -x['margin']))
    return best_per_target[:top_n]


__all__ = [
    'ATTACK_HORIZON', 'SUN_SAFETY_PIPE', 'PIPELINE_MAX_ANGLE_DEG',
    'eval_direct', 'eval_multi_sync', 'eval_pipe', 'all_plans', 'best_attacks',
]
