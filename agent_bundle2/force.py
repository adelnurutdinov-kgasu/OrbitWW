"""
Базовый блок: кривые влияния / силы.

Ключевые экспорты:
  _rendezvous_eta(src, dst, ships, omega) -> (turns, (px, py))
  force_events(state, target, ...)        -> список событий
  build_net_curve(events, target, ...)    -> (xs, ys)
  build_area_curve(events, target, ...)   -> (xs, area_ys)
"""

from orbit_sim import (
    predict_planet_xy, segment_hits_sun, dist, _fleet_speed,
)

HORIZON          = 120
SHIPS_REF        = 50
RENDEZVOUS_ITERS = 3
SUN_BLOCK_PENALTY = None  # None = пропускать источник; число = штраф ходов
SUN_SAFETY        = 1.5

import math

# вес area_inv (power-law, 1/(1+t)^2)
_POWER = 2.0


def _rendezvous_eta(src, dst, ships, omega, n_iter=RENDEZVOUS_ITERS):
    """Итеративная оценка времени встречи с возможно вращающейся целью."""
    speed = _fleet_speed(max(1, ships))
    if speed < 1e-6:
        return 1e9, (dst.x, dst.y)
    px, py = dst.x, dst.y
    d = max(0.5, dist(src.x, src.y, px, py) - src.radius - dst.radius)
    t = d / speed
    for _ in range(max(1, n_iter)):
        px, py = predict_planet_xy(dst.x, dst.y, dst.radius, omega, int(round(t)))
        d = max(0.5, dist(src.x, src.y, px, py) - src.radius - dst.radius)
        t_new = d / speed
        if abs(t_new - t) < 0.5:
            t = t_new
            break
        t = t_new
    return t, (px, py)


def force_events(state, target, horizon=HORIZON, ships_ref=SHIPS_REF,
                 player=0, rendezvous_iters=RENDEZVOUS_ITERS,
                 sun_penalty=SUN_BLOCK_PENALTY):
    """
    Список событий влияния на target.
    Каждое событие: (eta_turns, source_planet, sign, ships, production)
      sign = +1 если source принадлежит player, иначе -1.
    """
    events = []
    for src in state.planets:
        if src.id == target.id or src.owner == -1:
            continue
        sign = +1 if src.owner == player else -1
        eta, pred_xy = _rendezvous_eta(src, target, ships_ref, state.omega,
                                       n_iter=rendezvous_iters)
        blocked = segment_hits_sun(src.x, src.y, pred_xy[0], pred_xy[1],
                                   safety=SUN_SAFETY)
        if blocked:
            if sun_penalty is None:
                continue
            eta += float(sun_penalty)
        if eta > horizon:
            continue
        events.append((eta, src, sign, max(0, src.ships), src.production))
    events.sort(key=lambda e: e[0])
    return events


def build_net_curve(events, target, horizon=HORIZON, player=0):
    """Кривая net(t): y > 0 — мы доминируем, y < 0 — враг."""
    if target.owner == player:
        cum, slope = float(target.ships), float(target.production)
    elif target.owner == -1:
        cum, slope = 0.0, 0.0
    else:
        cum, slope = -float(target.ships), -float(target.production)

    xs, ys = [0.0], [cum]
    prev_t = 0.0
    for eta, src, sign, ships, prod in events:
        if eta > prev_t:
            xs.append(eta)
            ys.append(cum + slope * (eta - prev_t))
        cum += slope * (eta - prev_t)
        cum += sign * ships
        slope += sign * prod
        xs.append(eta)
        ys.append(cum)
        prev_t = eta
    return xs, ys


def build_area_curve(events, target, player=0):
    """Кумулятивная площадь под net(t) (трапеции)."""
    xs, net_ys = build_net_curve(events, target, player=player)
    if len(xs) < 2:
        return xs, [0.0]
    area_ys = [0.0]
    for i in range(1, len(xs)):
        dt = xs[i] - xs[i - 1]
        seg = (net_ys[i - 1] + net_ys[i]) * dt / 2.0
        area_ys.append(area_ys[-1] + seg)
    return xs, area_ys


def discounted_area_inv(xs, ys, power=_POWER):
    """Взвешенная площадь с весом 1/(1+t)^power (area_inv)."""
    s = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if dx <= 0:
            continue
        w0 = 1.0 / (1.0 + xs[i - 1]) ** power
        w1 = 1.0 / (1.0 + xs[i]) ** power
        s += (ys[i - 1] + ys[i]) * (w0 + w1) / 4.0 * dx
    return s


def zero_crossings(xs, ys):
    """Список (t, 'up'|'down') переходов через 0."""
    out = []
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if y0 < 0 <= y1:
            t = xs[i - 1] + (-y0) / (y1 - y0 + 1e-12) * (xs[i] - xs[i - 1])
            out.append((t, 'up'))
        elif y0 >= 0 > y1:
            t = xs[i - 1] + y0 / (y0 - y1 + 1e-12) * (xs[i] - xs[i - 1])
            out.append((t, 'down'))
    return out


__all__ = [
    'HORIZON', 'SHIPS_REF', 'RENDEZVOUS_ITERS', 'SUN_BLOCK_PENALTY', 'SUN_SAFETY',
    '_rendezvous_eta', 'force_events', 'build_net_curve', 'build_area_curve',
    'discounted_area_inv', 'zero_crossings',
]
