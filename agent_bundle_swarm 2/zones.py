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
from shooting import TOTAL_STEPS, predict_planet_xy, is_orbital

# ── Орбитальное сближение: orbit-aware дистанция ──────────────────────────
# Вместо текущего евклида берём МИНИМАЛЬНУЮ дистанцию между двумя планетами
# за ближайшие APPROACH_LOOKAHEAD ходов — учитывает вращение орбит.
#
#   Сценарии:
#     target летит к нам  → min < current → цель «ближе» → выше приоритет
#     target летит от нас → min ≈ current → не хуже обычного
#     обе планеты орбитальные → позиции обеих обновляются
#     обе статичные (omega≈0) → возвращает обычный dist без оверхеда
#
# APPROACH_SAMPLE: сэмплируем каждые N ходов (компромисс точность/скорость).
APPROACH_LOOKAHEAD = 30   # горизонт (ходов)
APPROACH_SAMPLE    = 5    # шаг сэмплирования → 6 точек на 30 ходов


def _approach_dist(p, q, omega):
    """Минимальная дистанция между p и q за ближайшие APPROACH_LOOKAHEAD ходов.

    Возвращает обычный dist если оба стационарны или omega≈0.
    Иначе сэмплирует предсказанные позиции и возвращает минимум.
    """
    d_now = max(1.0, dist(p.x, p.y, q.x, q.y))
    if abs(omega) < 1e-12:
        return d_now
    p_orb = is_orbital(p.x, p.y, p.radius)
    q_orb = is_orbital(q.x, q.y, q.radius)
    if not p_orb and not q_orb:
        return d_now
    min_d = d_now
    for t in range(APPROACH_SAMPLE, APPROACH_LOOKAHEAD + 1, APPROACH_SAMPLE):
        px, py = predict_planet_xy(p.x, p.y, p.radius, omega, t) if p_orb else (p.x, p.y)
        qx, qy = predict_planet_xy(q.x, q.y, q.radius, omega, t) if q_orb else (q.x, q.y)
        d = max(1.0, dist(px, py, qx, qy))
        if d < min_d:
            min_d = d
    return min_d


# ── Фичи зонирования ───────────────────────────────────────────────────────
# `late_aggression` — регуляризатор, который САМА фича не содержит phase
# (это важно: z-score нормализация всё равно сократила бы общий множитель).
# Фича = ripeness · deepness:
#     ripeness = production / (1 + ships)            — «спелость»/незащищённость
#     deepness = mean_dist(target, our_planets)      — глубина в тылу врага
# А phase = step / TOTAL_STEPS (∈ [0, 1]) применяется НА УРОВНЕ ВЕСА в
# compute_zones: эффективный вес = w_phase * W_TARGETS['late_aggression'].
# Так в Q1 вклад ≈ 0 (фича не работает), в Q4 — полный, и z-score не убивает
# разницу между фазами.
ZONE_FEATURES = ['area_inv', 'wnn_close_res', 'mean_dist_all', 'prod', 'ships',
                 'n_cross', 'late_aggression']

W_OURS = {
    'area_inv':        -0.1,
    'wnn_close_res':   -0.4,
    'mean_dist_all':   -0.3,
    'prod':            +0.4,
    'ships':           0,
    'n_cross':         +0.5,
    'late_aggression':  0.2,   # для своих планет смысла не несёт
}
W_TARGETS = {
    'area_inv':        +0.4,
    'wnn_close_res':   +0.8,    # tournament-winner (turn 2026-04-28): был +0.8
    'mean_dist_all':   -0.2,    # tournament-winner (turn 2026-04-28): был -0.6
    'prod':            +0.5,
    'ships':           -0.4,    # tournament-winner (turn 2026-04-28): был -0.7
    'n_cross':         +0.4,
    'late_aggression': +0.9,   # БАЗОВЫЙ вес (phase-multiplier применяется внутри
                               # compute_zones). Эффективный вес ≈ phase·0.6:
                               #   step=  0  → 0.00
                               #   step=125 → 0.15  (Q1→Q2)
                               #   step=250 → 0.30  (Q3 средняя)
                               #   step=375 → 0.45  (Q3→Q4)
                               #   step=500 → 0.60  (финал)
                               # Тюнить ОДНУ цифру, а phase сама подскейлит.
}

THR_HI = 0.5
THR_LO = -0.5

# ── Production-scarcity boost ──────────────────────────────────────────────
# Ранняя игра: когда наш суммарный прод мал, высоко-продуктивные цели
# получают дополнительный приоритет. Эффект плавно гасится к EARLY_PHASE_THR.
#
#   scarcity_boost = SCARCITY_K / (1 + our_total_prod)
#   fade           = max(0, 1 - phase / EARLY_PHASE_THR)   ∈ [0, 1]
#   eff_prod_w     = base_prod_w × (1 + scarcity_boost × fade)
#
# Примеры (SCARCITY_K=3, base_prod_w=0.5):
#   step=0,   our_prod= 2  → fade=1.0, boost=1.0  → eff=0.5×(1+1.0)=1.00
#   step=0,   our_prod= 8  → fade=1.0, boost=0.33 → eff=0.5×(1+0.33)=0.67
#   step=75,  our_prod= 5  → fade=0.5, boost=0.5  → eff=0.5×(1+0.25)=0.63
#   step=150, our_prod=any → fade=0.0              → eff=0.5 (нет буста)
SCARCITY_K       = 5.0   # сила буста (тюнить от 1 до 5)
EARLY_PHASE_THR  = 0.2   # фаза после которой буст = 0 (0.3 × 500 = step 150)

# ── Priority-override для периферии ───────────────────────────────────────
# Проблема: zone label и priority score вычисляются независимо.
# Планета может иметь высокий priority (хорошая по совокупности фич),
# но попасть в 'periphery' или 'hard_far' — catch-all зоны, которые
# исключены из TARGET_ZONES в agent.py → никогда не попадёт в аукцион.
#
# Решение: post-pass после расчёта обоих. Если нецелевая планета имеет
# priority ≥ PRIO_RECLASSIFY_THR — переклассифицируем её в 'priority_target'.
# Это делает zone и priority согласованными: высокий score = попадает в торги.
#
# Порог: priority на z-score шкале, обычно ∈ [-3, +3] для целей.
# 0.8 ≈ top-20% среди всех целей на карте. Тюнить от 0.5 до 1.5.
PRIO_RECLASSIFY_THR = 0.8   # планеты выше → force-upgrading до priority_target
PRIO_RECLASSIFY_ZONES = frozenset({'periphery', 'hard_far'})  # какие зоны апгрейдим

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


def compute_zones(df, w_ours=W_OURS, w_targets=W_TARGETS, player=0, step=0,
                  our_total_prod=0.0):
    """
    Принимает DataFrame с колонками ZONE_FEATURES + 'owner' + 'pid'.
    Возвращает (df_extended, Z_scores).

    `step` — ход матча. Используется как phase-множитель для двух вещей:
      1. late_aggression: эффективный вес = phase · w_targets['late_aggression']
      2. production-scarcity boost (early-game): вес prod у целей усиливается
         обратно пропорционально our_total_prod и линейно гасится к EARLY_PHASE_THR.

    `our_total_prod` — суммарный прод НАШИХ планет на текущий ход. Используется
    для production-scarcity: чем меньше наш прод, тем сильнее буст на rich-цели.
    """
    out = df.copy()
    Z = pd.DataFrame({m: _zscore(out[m]) for m in ZONE_FEATURES}, index=out.index)
    out['priority'] = 0.0

    is_ours = out['owner'] == player
    is_tgt  = ~is_ours

    # phase-множитель ∈ [0, 1]
    phase = max(0.0, min(1.0, float(step) / float(TOTAL_STEPS)))

    # production-scarcity: плавный буст ранней игры на prod-вес у целей
    _fade = max(0.0, 1.0 - phase / EARLY_PHASE_THR)
    _scarcity_boost = (SCARCITY_K / (1.0 + float(our_total_prod))) * _fade

    def _eff_weights(base, is_targets=False):
        eff = dict(base)
        eff['late_aggression'] = eff.get('late_aggression', 0.0) * phase
        if is_targets and _scarcity_boost > 0:
            eff['prod'] = eff.get('prod', 0.0) * (1.0 + _scarcity_boost)
        return eff

    eff_ours    = _eff_weights(w_ours,    is_targets=False)
    eff_targets = _eff_weights(w_targets, is_targets=True)

    out.loc[is_ours, 'priority'] = sum(
        eff_ours[m] * Z.loc[is_ours, m] for m in ZONE_FEATURES
    )
    out.loc[is_tgt, 'priority'] = sum(
        eff_targets[m] * Z.loc[is_tgt, m] for m in ZONE_FEATURES
    )

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

    # ── Priority-override post-pass ────────────────────────────────────────
    # Переклассифицируем 'periphery'/'hard_far' с высоким priority в
    # 'priority_target', чтобы они попали в аукцион через TARGET_ZONES.
    # Применяется ТОЛЬКО к нецелевым (не наши) планетам.
    reclassify_mask = (
        is_tgt
        & out['zone'].isin(PRIO_RECLASSIFY_ZONES)
        & (out['priority'] >= PRIO_RECLASSIFY_THR)
    )
    if reclassify_mask.any():
        out.loc[reclassify_mask, 'zone'] = 'priority_target'

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

    # omega для orbit-aware дистанции: берём из state (GameState) или state_view
    # (SimpleNamespace с полем omega). Если поле отсутствует — 0.0 (статичные).
    _omega = getattr(state_view, 'omega', 0.0) or 0.0

    wnn_close_res = 0.0
    wr_sum        = 0.0
    mean_dist_all = 0.0
    sum_d_to_ours = 0.0
    n_ours = 0
    for q in others:
        # Orbit-aware: используем минимальную дистанцию за APPROACH_LOOKAHEAD ходов.
        # Для статичных пар (omega≈0 или обе не на орбите) идентично обычному dist.
        d = _approach_dist(p, q, _omega)
        w = (q.ships + 5.0 * q.production) / d
        wnn_close_res += _sign(q) * w
        wr_sum        += w
        mean_dist_all += d
        if q.owner == player:
            sum_d_to_ours += d
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
    }


def compute_zones_from_state(state, player=0, horizon=HORIZON, ships_ref=SHIPS_REF,
                              w_ours=W_OURS, w_targets=W_TARGETS, step=None,
                              our_total_prod=None):
    """
    Вход: GameState, player id.
    Выход: (df_with_zones, Z_scores) — готово для агента без тяжёлых вычислений.

    `step` — текущий ход матча. Если None — берётся из state.step. Нужен для
    late_aggression-фичи и production-scarcity boost.

    `our_total_prod` — суммарный прод наших планет. Если None — вычисляется
    автоматически из state.planets. Используется для production-scarcity boost
    в ранней игре: усиливает приоритет high-prod целей когда наш прод мал.

    Кометы (`state.comet_ids`) ОСОЗНАННО игнорируются:
      • не попадают в df (значит не выбираются как цели через TARGET_ZONES);
      • не учитываются как «соседи» при расчёте distance/force-фич остальных
        планет — иначе их кратковременное появление дрейфит priority стабильных
        планет каждые ~50 ходов.
    """
    comet_ids = set(getattr(state, 'comet_ids', set()) or set())
    if step is None:
        step = int(getattr(state, 'step', 0) or 0)
    if our_total_prod is None:
        our_total_prod = sum(
            float(p.production) for p in state.planets
            if p.id not in comet_ids and p.owner == player
        )
    rows = [_planet_zone_features(state, p, player, horizon, ships_ref,
                                   comet_ids=comet_ids, step=step)
            for p in state.planets if p.id not in comet_ids]
    df = pd.DataFrame(rows)
    return compute_zones(df, w_ours=w_ours, w_targets=w_targets,
                         player=player, step=step, our_total_prod=our_total_prod)


__all__ = [
    'ZONE_FEATURES', 'W_OURS', 'W_TARGETS', 'THR_HI', 'THR_LO', 'ZONE_COLORS',
    'SCARCITY_K', 'EARLY_PHASE_THR',
    'PRIO_RECLASSIFY_THR', 'PRIO_RECLASSIFY_ZONES',
    'APPROACH_LOOKAHEAD', 'APPROACH_SAMPLE', '_approach_dist',
    'compute_zones', 'compute_zones_from_state',
]
