"""
shooting.py — блок наведения Orbit Wars.

Две части:
  A. Базовые примитивы (типы, геометрия, физика) — нужны orbit_sim.py.
  B. Точная геометрия пуска — адаптировано из pipeline_bot_v1.py.

Публичный интерфейс для агента:
  aim_angle(src, tgt, ships, state)  → float (радианы)
  refine_launch(src, tgt, ships, state) → (angle, travel_time, (aim_x, aim_y)) | None
"""

import math
from collections import namedtuple

# ═══════════════════════════════════════════════════════════════════════════
# A. БАЗОВЫЕ ПРИМИТИВЫ
# ═══════════════════════════════════════════════════════════════════════════

# Kaggle obs: [id, owner, x, y, radius, ships, production]
Planet = namedtuple('Planet', ['id', 'owner', 'x', 'y', 'radius', 'ships', 'production'])
# Kaggle obs: [id, owner, x, y, angle, from_planet_id, ships]
Fleet  = namedtuple('Fleet',  ['id', 'owner', 'x', 'y', 'angle', 'from_planet_id', 'ships'])

# Константы игры (совпадают с конфигурацией по умолчанию)
CX, CY         = 50.0, 50.0   # центр солнца
SUN_R          = 10.0          # радиус солнца
BOARD          = 100.0         # размер поля
ROTATION_LIMIT = 50.0          # orbital_radius + planet_radius < 50 → вращается
LAUNCH_CLEARANCE = 1.0         # буфер запуска сверх радиуса исходной планеты
SUN_SAFETY     = 1.5           # дополнительный запас при проверке солнца
MAX_SPEED      = 6.0           # максимальная скорость флота
TOTAL_STEPS    = 500           # длина партии


# ── Геометрия ──────────────────────────────────────────────────────────────

def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def point_to_segment_dist(px: float, py: float,
                           ax: float, ay: float,
                           bx: float, by: float) -> float:
    """Расстояние от точки (px, py) до отрезка (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float,
                     safety: float = SUN_SAFETY) -> bool:
    return point_to_segment_dist(CX, CY, x1, y1, x2, y2) < SUN_R + safety


# ── Физика ─────────────────────────────────────────────────────────────────

def is_orbital(x: float, y: float, radius: float) -> bool:
    """Вращается ли планета с центром (x,y) и радиусом radius вокруг солнца."""
    return math.hypot(x - CX, y - CY) + radius < ROTATION_LIMIT


def fleet_speed_correct(ships: int) -> float:
    """Скорость флота по формуле из README (maxSpeed = 6.0)."""
    if ships <= 1:
        return 1.0
    v = math.log(max(ships, 1)) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * (v ** 1.5)


def predict_planet_xy(x: float, y: float, radius: float,
                      omega: float, turns: int) -> tuple:
    """
    Позиция планеты через `turns` ходов от её текущей позиции (x, y).
    Для не-орбитальных планет возвращает (x, y) без изменений.
    Работает и с абсолютными ходами (передавай initial.x/y + total_turns).
    """
    if not is_orbital(x, y, radius) or abs(omega) < 1e-9:
        return x, y
    r = math.hypot(x - CX, y - CY)
    angle = math.atan2(y - CY, x - CX) + omega * turns
    return CX + r * math.cos(angle), CY + r * math.sin(angle)


# ═══════════════════════════════════════════════════════════════════════════
# B. ТОЧНАЯ ГЕОМЕТРИЯ ПУСКА
#    Адаптировано из pipeline_bot_v1.py (ПАТЧ 7 + ПАТЧ 10).
#    Входной параметр — GameState из orbit_sim (вместо world из submission).
# ═══════════════════════════════════════════════════════════════════════════

_PLANET_BLOCK_LOG = []  # диагностика: [{step, src, tgt, blocker, frac, K, eta}]


def _predict_pos(planet, state, t_from_now: float) -> tuple:
    """Позиция планеты через t_from_now ходов от текущего момента."""
    return predict_planet_xy(planet.x, planet.y, planet.radius,
                             state.omega, t_from_now)


def _segment_hits_any_planet(state, src_id: int, tgt_id: int,
                              start_xy: tuple, end_xy: tuple,
                              eta: float, extra: float = 0.3) -> tuple:
    """
    4-точечный sampling вдоль траектории флота.
    Возвращает (True, blocker_id, frac) если путь пересекает другую планету,
    иначе (False, None, None).
    """
    sx, sy = start_xy
    ex, ey = end_xy
    for p in state.planets:
        if p.id == src_id or p.id == tgt_id:
            continue
        r_eff = float(p.radius) + LAUNCH_CLEARANCE + extra
        for frac in (0.15, 0.40, 0.65, 0.90):
            t = frac * eta
            fx = sx + frac * (ex - sx)
            fy = sy + frac * (ey - sy)
            px, py = _predict_pos(p, state, t)
            if math.hypot(fx - px, fy - py) < r_eff:
                return True, p.id, frac
    return False, None, None


def refine_launch(src, tgt, ships: int, state,
                  max_iter: int = 20, tol: float = 0.005):
    """
    Точный расчёт угла пуска флота src → tgt с учётом:
      - вращения орбитальных планет (итеративная сходимость);
      - блокировки солнцем;
      - блокировки другими планетами (ПАТЧ 10).

    Аргументы:
      src, tgt  — Planet (namedtuple)
      ships     — количество кораблей в флоте
      state     — GameState (из orbit_sim)

    Возвращает (angle, travel_time, (aim_x, aim_y)) или None если путь заблокирован.
    """
    v = fleet_speed_correct(int(ships))
    if v <= 0:
        return None

    sx, sy = src.x, src.y
    sr     = float(src.radius)
    tr     = float(tgt.radius)
    clear  = sr + LAUNCH_CLEARANCE

    tx0, ty0 = tgt.x, tgt.y
    base     = max(0.0, dist(sx, sy, tx0, ty0) - clear - tr)
    t_est    = base / v

    converged = False
    for _ in range(max_iter):
        tx, ty = _predict_pos(tgt, state, t_est)
        hit    = max(0.0, dist(sx, sy, tx, ty) - clear - tr)
        t_new  = hit / v
        if abs(t_new - t_est) < tol:
            t_est     = t_new
            converged = True
            break
        t_est = t_new
    if not converged:
        return None

    tx, ty   = _predict_pos(tgt, state, t_est)
    ang      = math.atan2(ty - sy, tx - sx)
    start_x  = sx + math.cos(ang) * clear
    start_y  = sy + math.sin(ang) * clear
    hit_dist = max(0.0, dist(sx, sy, tx, ty) - clear - tr)
    end_x    = start_x + math.cos(ang) * hit_dist
    end_y    = start_y + math.sin(ang) * hit_dist

    if segment_hits_sun(start_x, start_y, end_x, end_y, safety=SUN_SAFETY):
        return None

    blocked, blocker_id, frac = _segment_hits_any_planet(
        state, src.id, tgt.id, (start_x, start_y), (end_x, end_y), t_est
    )
    if blocked:
        _PLANET_BLOCK_LOG.append({
            'step': state.step, 'src': src.id, 'tgt': tgt.id,
            'blocker': blocker_id, 'frac': frac,
            'K': int(ships), 'eta': float(t_est),
        })
        return None

    return ang, t_est, (tx, ty)


def aim_angle(src, tgt, ships: int, state) -> float:
    """
    Интерфейс для agent.py: возвращает угол пуска src → tgt.
    При блокировке (солнцем или планетой) возвращает прямой угол на текущую позицию цели.
    """
    result = refine_launch(src, tgt, ships, state)
    if result is None:
        return math.atan2(tgt.y - src.y, tgt.x - src.x)
    angle, _, _ = result
    return angle


# ═══════════════════════════════════════════════════════════════════════════
__all__ = [
    # типы
    'Planet', 'Fleet',
    # константы
    'CX', 'CY', 'SUN_R', 'BOARD', 'ROTATION_LIMIT',
    'LAUNCH_CLEARANCE', 'SUN_SAFETY', 'MAX_SPEED', 'TOTAL_STEPS',
    # геометрия
    'dist', 'point_to_segment_dist', 'segment_hits_sun',
    # физика
    'is_orbital', 'fleet_speed_correct', 'predict_planet_xy',
    # наведение
    'refine_launch', 'aim_angle', '_PLANET_BLOCK_LOG',
]
