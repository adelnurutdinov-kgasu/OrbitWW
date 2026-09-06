"""
Блок зонирования и приоритизации планет.

Ключевые экспорты:
  compute_zones(df, ...)                       -> (df_with_zones, Z_scores)
  compute_zones_from_state(state, player, ...) -> (df, Z)  ← для агента
"""

import math
from types import SimpleNamespace
import pandas as pd
import numpy as np

from orbit_sim import dist
from force import (
    force_events, build_net_curve, discounted_area_inv, zero_crossings,
    HORIZON, SHIPS_REF,
)
from shooting import TOTAL_STEPS

# ── Фичи зонирования ───────────────────────────────────────────────────────
# Эмпирический анализ 270k запусков (500 реплеев, топ-игроки, 2025-05):
#
# late_aggression = ripeness × deepness
#   ripeness = production / (1 + ships)         — «спелость»/незащищённость
#   deepness = mean_dist(target, our_planets)   — глубина в тылу врага
#
# ripeness — добавлена как самостоятельная фича (эмп. вес: +0.9 neutral, +2.9 enemy)
#
# phase = step / TOTAL_STEPS применяется НА УРОВНЕ ВЕСА в compute_zones:
#   эффективный вес = phase × W['late_aggression']
#   → early-game вклад ≈ 0, late-game — полный, z-score не съедает разницу.
#
# Разделены W_TARGETS_NEUTRAL и W_TARGETS_ENEMY — ключевое различие:
#   neutral: late_aggression ПОЗИТИВНЫЙ (глубокие нейтральные = стратегические позиции)
#   enemy:   late_aggression НЕГАТИВНЫЙ (60% атак и топ-игроки предпочитают близких врагов)
ZONE_FEATURES = ['area_inv', 'wnn_close_res', 'mean_dist_all', 'prod', 'ships',
                 'n_cross', 'late_aggression', 'ripeness']

_NEUTRAL_OWNER = -1   # магический owner для нейтральных планет в движке

W_OURS = {
    'area_inv':        -0.1,
    'wnn_close_res':   -0.4,
    'mean_dist_all':   -0.3,
    'prod':            +0.4,
    'ships':           0,
    'n_cross':         +0.9,
    'late_aggression': -0.2,
    'ripeness':        +0.0,   # для своих планет не актуально
}

# ── Веса для НЕЙТРАЛЬНЫХ целей ──────────────────────────────────────────────
# late_aggression ПОЗИТИВНЫЙ: победа коррелирует с захватом глубоких нейтральных
# (стратегические позиции ценнее чем ближние).
# Корреляция rel_late_agg с victory: +0.13 early, +0.09 mid, +0.07 late
W_TARGETS_NEUTRAL = {
    'area_inv':        +0.4,
    'wnn_close_res':   +0.8,
    'mean_dist_all':   -0.3,   # нейтральные: dist менее критичен чем для врага
    'prod':            +0.5,
    'ships':           -0.5,
    'n_cross':         +0.4,
    'late_aggression': +0.4,   # ПОЗИТИВНЫЙ: глубокие нейтральные = стратегическая ценность
                               # phase-multiplier: early ≈ 0, late = +0.4
    'ripeness':        +0.5,   # NEW: эмп. вес +0.89, растёт к mid/late
}

# ── Веса для ВРАЖЕСКИХ целей ────────────────────────────────────────────────
# late_aggression НЕГАТИВНЫЙ: 60% атак предпочитают близких врагов,
# топ-игроки: медиана rel_late_agg = −1.91 (сильное предпочтение близких).
# mean_dist_all усилен: сильнейший предиктор победы (r=−0.56 в endgame).
# Точность предсказания: было 54.3% → стало 64.3% (+10pp).
W_TARGETS_ENEMY = {
    'area_inv':        +0.4,
    'wnn_close_res':   +0.8,
    'mean_dist_all':   -0.5,   # усилен: победа тесно связана с атакой близких врагов
    'prod':            +0.5,
    'ships':           -0.5,
    'n_cross':         +0.4,
    'late_aggression': -0.6,   # НЕГАТИВНЫЙ: флипнут с +0.9 (2025-05 эмп. анализ)
                               # phase-multiplier: early ≈ 0, late = −0.6
    'ripeness':        +0.9,   # NEW: эмп. вес +2.88, второй по силе после wnn
}

# backward-compat alias (используется старым кодом и agent.py)
W_TARGETS = W_TARGETS_ENEMY

THR_HI = 0.5
THR_LO = -0.5

ZONE_COLORS = {
    'frontline':       '#e05c3a',
    'contested':       '#f0b04a',
    'bastion':         '#4a90d9',
    'rear':            '#6fb6e8',
    'isolated':        '#a36fe8',
    'mid':             '#9aa7b8',
    'easy_target':     '#2eccaa',
    'priority_target': '#ffd24a',
    'hard_far':        '#9b3f6f',
    'periphery':       '#506070',
}


def _zscore(s):
    s = s.astype(float)
    sig = s.std(ddof=0)
    if sig < 1e-12:
        return s * 0.0
    return (s - s.mean()) / sig


def compute_zones(df, w_ours=W_OURS, w_targets=None,
                  w_targets_neutral=None, w_targets_enemy=None,
                  player=0, step=0):
    """
    Принимает DataFrame с колонками ZONE_FEATURES + 'owner' + 'pid'.
    Возвращает (df_extended, Z_scores).

    Веса целей:
      w_targets_neutral — для нейтральных планет (default: W_TARGETS_NEUTRAL)
      w_targets_enemy   — для вражеских планет   (default: W_TARGETS_ENEMY)
      w_targets         — backward-compat: если задан, используется для обоих

    phase-множитель применяется к late_aggression:
      эффективный вес = phase × w['late_aggression']
      neutral: late_aggression позитивный → растёт к концу (глубокие нейтральные ценнее)
      enemy:   late_aggression негативный → растёт штраф к концу (ближе = лучше)
    """
    # Разрешаем веса: backward-compat через w_targets
    if w_targets_neutral is None:
        w_targets_neutral = w_targets if w_targets is not None else W_TARGETS_NEUTRAL
    if w_targets_enemy is None:
        w_targets_enemy = w_targets if w_targets is not None else W_TARGETS_ENEMY

    out = df.copy()
    # z-score по всем планетам вместе (нормализация общая — так относительные
    # ранги между нейтральными и вражескими сохраняются корректно)
    feat_present = [m for m in ZONE_FEATURES if m in out.columns]
    Z = pd.DataFrame({m: _zscore(out[m]) for m in feat_present}, index=out.index)
    out['priority'] = 0.0

    is_ours    = out['owner'] == player
    is_neutral = (~is_ours) & (out['owner'] == _NEUTRAL_OWNER)
    is_enemy   = (~is_ours) & (out['owner'] != _NEUTRAL_OWNER)

    # phase-множитель — только для late_aggression (знак уже в весе)
    phase = max(0.0, min(1.0, float(step) / float(TOTAL_STEPS)))

    def _eff_weights(base):
        eff = dict(base)
        eff['late_aggression'] = eff.get('late_aggression', 0.0) * phase
        return eff

    eff_ours    = _eff_weights(w_ours)
    eff_neutral = _eff_weights(w_targets_neutral)
    eff_enemy   = _eff_weights(w_targets_enemy)

    def _score(mask, eff):
        if not mask.any():
            return
        out.loc[mask, 'priority'] = sum(
            eff.get(m, 0.0) * Z.loc[mask, m]
            for m in feat_present
        )

    _score(is_ours,    eff_ours)
    _score(is_neutral, eff_neutral)
    _score(is_enemy,   eff_enemy)

    def label(idx):
        z    = Z.loc[idx]
        ours = bool(is_ours.loc[idx])
        threat  = z['area_inv']      < THR_LO
        in_us   = z['wnn_close_res'] > THR_HI
        in_them = z['wnn_close_res'] < THR_LO
        rich    = (z['prod'] + z['ships']) / 2 > THR_HI * 0.6
        contest = z['n_cross']       > THR_HI
        far     = z['mean_dist_all'] > THR_HI
        weak    = z['ships']         < THR_LO

        if ours:
            if threat or in_them: return 'frontline'
            if contest:           return 'contested'
            if rich and in_us:    return 'bastion'
            if in_us:             return 'rear'
            if far:               return 'isolated'
            return 'mid'
        else:
            if in_us and weak:    return 'easy_target'
            if rich and not far:  return 'priority_target'
            if far and not weak:  return 'hard_far'
            if contest:           return 'contested'
            return 'periphery'

    out['zone'] = [label(i) for i in out.index]
    return out, Z


def _planet_zone_features(state, p, player, horizon, ships_ref, comet_ids=None, step=0):
    """Вычисляет фичи для зонирования (без полного planet_metrics).

    Кометы (`comet_ids`) полностью исключаются из расчётов: ни как источники
    давления (force_events), ни как «другие» планеты при усреднении расстояний.
    Это решение принято осознанно — кометы временные (живут считанные ходы),
    их состав/позиция нестабильны, и они искажают приоритеты zones.

    `step` — текущий ход матча. Используется для late_aggression-фичи
    (см. формулу ниже). Если 0 — фича = 0 для всех (нейтрально).
    """
    comet_ids = comet_ids if comet_ids is not None else set()
    # Вью на state без комет — для force_events / others.
    if comet_ids:
        non_comet_planets = [q for q in state.planets if q.id not in comet_ids]
        state_view = SimpleNamespace(planets=non_comet_planets, omega=state.omega)
    else:
        state_view = state

    events = force_events(state_view, p, horizon=horizon, ships_ref=ships_ref, player=player)
    xs, ys = build_net_curve(events, p, horizon=horizon, player=player)

    area_inv = discounted_area_inv(xs, ys)
    n_cross  = len(zero_crossings(xs, ys))

    others = [q for q in state_view.planets if q.id != p.id]

    def _sign(q):
        if q.owner == player:           return +1.0
        if q.owner not in (-1, player): return -1.0
        return 0.0

    wnn_close_res = 0.0
    wr_sum        = 0.0
    mean_dist_all = 0.0
    sum_d_to_ours = 0.0
    n_ours = 0
    for q in others:
        d = max(1.0, dist(p.x, p.y, q.x, q.y))
        w = (q.ships + 5.0 * q.production) / d
        wnn_close_res += _sign(q) * w
        wr_sum        += w
        mean_dist_all += dist(p.x, p.y, q.x, q.y)
        if q.owner == player:
            sum_d_to_ours += dist(p.x, p.y, q.x, q.y)
            n_ours += 1
    if wr_sum > 0:
        wnn_close_res /= wr_sum
    if others:
        mean_dist_all /= len(others)
    mean_d_to_ours = sum_d_to_ours / n_ours if n_ours > 0 else 0.0

    # ── late_aggression ────────────────────────────────────────────────
    # ripeness × deepness — БЕЗ phase (phase применяется на уровне веса в
    # compute_zones, иначе z-score нормализация съест общий множитель).
    #   ripeness — production/(1+ships): «спелая» цель имеет высокий prod
    #              и малый гарнизон (быстрая окупаемость захвата).
    #   deepness — среднее расстояние от p до НАШИХ планет: дальние тыловые
    #              цели получают больший вес.
    # Сама фича статична относительно phase, но её ВКЛАД в priority
    # умножается на phase в compute_zones — таким образом early-game вклад
    # ≈ 0 (нейтрально), late-game — полный.
    ripeness = float(p.production) / (1.0 + float(p.ships))
    late_aggression = ripeness * mean_d_to_ours

    return {
        'pid':            p.id,
        'owner':          p.owner,
        'prod':           float(p.production),
        'ships':          float(p.ships),
        'area_inv':       area_inv,
        'wnn_close_res':  wnn_close_res,
        'mean_dist_all':  mean_dist_all,
        'n_cross':        float(n_cross),
        'late_aggression': float(late_aggression),
        'ripeness':       float(ripeness),   # prod/(1+ships) — эмп. вес: +0.9 neutral, +2.9 enemy
    }


def compute_zones_from_state(state, player=0, horizon=HORIZON, ships_ref=SHIPS_REF,
                              w_ours=W_OURS, w_targets=W_TARGETS, step=None):
    """
    Вход: GameState, player id.
    Выход: (df_with_zones, Z_scores) — готово для агента без тяжёлых вычислений.

    `step` — текущий ход матча. Если None — берётся из state.step. Нужен для
    late_aggression-фичи (вес растёт линейно по ходу партии).

    Кометы (`state.comet_ids`) ОСОЗНАННО игнорируются:
      • не попадают в df (значит не выбираются как цели через TARGET_ZONES);
      • не учитываются как «соседи» при расчёте distance/force-фич остальных
        планет — иначе их кратковременное появление дрейфит priority стабильных
        планет каждые ~50 ходов.
    """
    comet_ids = set(getattr(state, 'comet_ids', set()) or set())
    if step is None:
        step = int(getattr(state, 'step', 0) or 0)
    rows = [_planet_zone_features(state, p, player, horizon, ships_ref,
                                   comet_ids=comet_ids, step=step)
            for p in state.planets if p.id not in comet_ids]
    df = pd.DataFrame(rows)
    return compute_zones(df, w_ours=w_ours, w_targets=w_targets,
                         player=player, step=step)


__all__ = [
    'ZONE_FEATURES', 'W_OURS', 'W_TARGETS',
    'W_TARGETS_NEUTRAL', 'W_TARGETS_ENEMY',
    'THR_HI', 'THR_LO', 'ZONE_COLORS',
    'compute_zones', 'compute_zones_from_state',
]
