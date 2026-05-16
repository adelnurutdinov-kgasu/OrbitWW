# Auto-generated submission — 2026-05-16T09:30:16
# Source: agent_bundle_swarm 2
# Build: tuning/scripts/build_submission.py


# ╔══════════════════════════════════════════════════╗
# ║  shooting.py                                   ║
# ╚══════════════════════════════════════════════════╝

"""
shooting.py — блок наведения Orbit Wars (гибридная версия).

Главный алгоритм — `aim_hybrid` (D_hybrid):
  1. Fixpoint-итерация по будущей позиции цели.
  2. Проверка блокировки солнцем и **движущимися** планетами (sampling вдоль траектории).
  3. Fallback-sweep: перебор кандидатов launch_turn ∈ 1..48 для согласованного ETA.
  4. Safe-aim: деформация угла вокруг солнца как последний шанс.

Также включены варианты A/B/C для бенчмарка, предсказатели вражеских флотов
и `simulate_launch` — покадровый ground-truth симулятор.

Публичный интерфейс для agent.py:
  aim_angle(src, tgt, ships, state) -> float (радианы)
"""

import math
from collections import namedtuple

# ════════════════════════════════════════════════════════════════════
# Константы (совпадают с движком игры)
# ════════════════════════════════════════════════════════════════════

BOARD            = 100.0
CX, CY           = 50.0, 50.0
SUN_R            = 10.0
MAX_SPEED        = 6.0
ROTATION_LIMIT   = 50.0
LAUNCH_CLEARANCE = 0.1
SUN_SAFETY       = 1.5
BLOCKER_SAFETY   = 0.35
INTERCEPT_TOL    = 1
DEFAULT_HORIZON  = 80
FALLBACK_SWEEP   = 48
TOTAL_STEPS      = 500

Planet = namedtuple("Planet", "id owner x y radius ships production")
Fleet  = namedtuple("Fleet",  "id owner x y angle from_planet_id ships")


# ════════════════════════════════════════════════════════════════════
# Геометрия
# ════════════════════════════════════════════════════════════════════

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_dist(CX, CY, x1, y1, x2, y2) < SUN_R + safety


# ════════════════════════════════════════════════════════════════════
# Физика
# ════════════════════════════════════════════════════════════════════

def fleet_speed_correct(ships):
    """Официальная формула из README: 1 + 5 * (log(ships) / log(1000))^1.5"""
    if ships <= 1:
        return 1.0
    r = math.log(ships) / math.log(1000.0)
    r = max(0.0, min(1.0, r))
    return 1.0 + (MAX_SPEED - 1.0) * (r ** 1.5)


def fleet_speed_buggy(ships):
    """Из target-score-2000-4 — линейная (для бенчмарка)."""
    return min(1.0 + ships // 20, 6.0)


def is_orbital(x, y, radius):
    return math.hypot(x - CX, y - CY) + radius < ROTATION_LIMIT


def predict_planet_xy(x, y, radius, omega, turns):
    """Позиция планеты через `turns` ходов от точки (x, y)."""
    if turns <= 0 or abs(omega) < 1e-12 or not is_orbital(x, y, radius):
        return x, y
    r = math.hypot(x - CX, y - CY)
    ang = math.atan2(y - CY, x - CX) + omega * turns
    return CX + r * math.cos(ang), CY + r * math.sin(ang)


# ════════════════════════════════════════════════════════════════════
# Геометрия пути
# ════════════════════════════════════════════════════════════════════

def launch_point(sx, sy, sr, angle):
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    lx, ly = launch_point(sx, sy, sr, angle)
    hit_d = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    ex = lx + math.cos(angle) * hit_d
    ey = ly + math.sin(angle) * hit_d
    return angle, lx, ly, ex, ey, hit_d


# ════════════════════════════════════════════════════════════════════
# Проверка блокировки планетами
# ════════════════════════════════════════════════════════════════════

def line_blocked_static(x1, y1, x2, y2, blockers, safety=BLOCKER_SAFETY):
    for b in blockers:
        if point_to_segment_dist(b.x, b.y, x1, y1, x2, y2) <= b.radius + safety:
            return True, b.id
    return False, None


def line_blocked_moving(x1, y1, x2, y2, eta, blockers, omega,
                        samples=(0.10, 0.25, 0.40, 0.55, 0.70, 0.85),
                        safety=BLOCKER_SAFETY):
    """Старая sample-based проверка (оставлена для совместимости)."""
    for b in blockers:
        if not is_orbital(b.x, b.y, b.radius) or abs(omega) < 1e-12:
            if point_to_segment_dist(b.x, b.y, x1, y1, x2, y2) <= b.radius + safety:
                return True, b.id
        else:
            if point_to_segment_dist(b.x, b.y, x1, y1, x2, y2) <= b.radius + safety:
                return True, b.id
            for frac in samples:
                t = frac * eta
                bx, by = predict_planet_xy(b.x, b.y, b.radius, omega, t)
                fx = x1 + frac * (x2 - x1)
                fy = y1 + frac * (y2 - y1)
                if math.hypot(fx - bx, fy - by) < b.radius + safety:
                    return True, b.id
    return False, None


def line_blocked_precise(launch_xy, angle, ships, hit_d, blockers, omega,
                          safety=0.0):
    """
    Симулятор-точная проверка: walk флота по integer-ходам, проверяя
    point_to_segment_dist от каждой планеты-блокера к сегменту полёта на этом ходу.
    Возвращает (blocked, blocker_id) — 1-в-1 с simulate_launch.
    """
    speed = fleet_speed_correct(max(1, int(ships)))
    if speed < 1e-6:
        return False, None
    dx, dy = math.cos(angle), math.sin(angle)
    fx, fy = launch_xy
    max_turns = int(math.ceil(hit_d / speed)) + 1
    for turn in range(1, max_turns + 1):
        nx, ny = fx + speed * dx, fy + speed * dy
        # солнце
        if point_to_segment_dist(CX, CY, fx, fy, nx, ny) < SUN_R + safety:
            return True, -1
        for b in blockers:
            if is_orbital(b.x, b.y, b.radius) and abs(omega) > 1e-12:
                bx, by = predict_planet_xy(b.x, b.y, b.radius, omega, turn)
            else:
                bx, by = b.x, b.y
            d = point_to_segment_dist(bx, by, fx, fy, nx, ny)
            if d < b.radius + safety:
                return True, b.id
        fx, fy = nx, ny
    return False, None


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr, blockers=None,
                            sun_safety=SUN_SAFETY):
    angle, lx, ly, ex, ey, hit_d = actual_path_geometry(sx, sy, sr, tx, ty, tr)
    if segment_hits_sun(lx, ly, ex, ey, safety=sun_safety):
        return None
    if blockers:
        blocked, _ = line_blocked_static(lx, ly, ex, ey, blockers)
        if blocked:
            return None
    return angle, hit_d


# ════════════════════════════════════════════════════════════════════
# Variant A — baseline sweep по ETA + статичные blockers
# ════════════════════════════════════════════════════════════════════

def aim_baseline(src, target, ships, planets, initial_by_id, omega,
                 sweep_back=5, sweep_fwd=10, speed_fn=fleet_speed_correct):
    speed = speed_fn(ships)
    if speed <= 0:
        return None
    blockers = [p for p in planets if p.id not in (src.id, target.id)]
    cur_d = max(0.5, dist(src.x, src.y, target.x, target.y) - src.radius - target.radius)
    rough_eta = max(1, int(math.ceil(cur_d / speed)))
    is_stat = not is_orbital(target.x, target.y, target.radius)
    tol = 0 if is_stat else 1
    best, best_metric = None, None
    for g_eta in range(max(1, rough_eta - sweep_back), rough_eta + sweep_fwd + 1):
        px, py = predict_planet_xy(target.x, target.y, target.radius, omega, g_eta)
        safe = safe_angle_and_distance(src.x, src.y, src.radius, px, py, target.radius, blockers)
        if safe is None:
            continue
        angle, hit_d = safe
        actual_eta = max(1, int(math.ceil(hit_d / speed - 1e-9)))
        if abs(actual_eta - g_eta) > tol:
            continue
        rx, ry = predict_planet_xy(target.x, target.y, target.radius, omega, actual_eta)
        safe2 = safe_angle_and_distance(src.x, src.y, src.radius, rx, ry, target.radius, blockers)
        if safe2 is None:
            continue
        angle2, hit_d2 = safe2
        final_eta = max(1, int(math.ceil(hit_d2 / speed - 1e-9)))
        metric = (abs(final_eta - g_eta), final_eta, hit_d2)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best = (angle2, final_eta, rx, ry)
            if metric[0] == 0 and final_eta <= rough_eta + 2:
                break
    return best


# ════════════════════════════════════════════════════════════════════
# Variant B — fixpoint + search_safe_intercept fallback (без blockers)
# ════════════════════════════════════════════════════════════════════

def aim_fixpoint(src, target, ships, planets, initial_by_id, omega,
                 max_iter=5, sweep_horizon=FALLBACK_SWEEP,
                 speed_fn=fleet_speed_correct):
    speed = speed_fn(ships)
    if speed <= 0:
        return None

    def est_arrival(tx, ty):
        safe = safe_angle_and_distance(src.x, src.y, src.radius, tx, ty, target.radius, None)
        if safe is None:
            return None
        a, d = safe
        return a, max(1, int(math.ceil(d / speed)))

    tgt_can_move = is_orbital(target.x, target.y, target.radius)

    est = est_arrival(target.x, target.y)
    if est is not None:
        tx, ty = target.x, target.y
        for _ in range(max_iter):
            _, turns = est
            ntx, nty = predict_planet_xy(target.x, target.y, target.radius, omega, turns)
            next_est = est_arrival(ntx, nty)
            if next_est is None:
                break
            if (abs(ntx - tx) < 0.3 and abs(nty - ty) < 0.3
                    and abs(next_est[1] - turns) <= INTERCEPT_TOL):
                return next_est[0], next_est[1], ntx, nty
            tx, ty = ntx, nty
            est = next_est
        if est is not None:
            return est[0], est[1], tx, ty

    if not tgt_can_move:
        return None
    best, best_score = None, None
    max_turns = min(DEFAULT_HORIZON, sweep_horizon)
    for ct in range(1, max_turns + 1):
        px, py = predict_planet_xy(target.x, target.y, target.radius, omega, ct)
        est = est_arrival(px, py)
        if est is None:
            continue
        _, turns = est
        if abs(turns - ct) > INTERCEPT_TOL:
            continue
        actual_turns = max(turns, ct)
        apx, apy = predict_planet_xy(target.x, target.y, target.radius, omega, actual_turns)
        confirm = est_arrival(apx, apy)
        if confirm is None:
            continue
        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOL:
            continue
        score = (delta, confirm[1], ct)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], apx, apy)
    return best


# ════════════════════════════════════════════════════════════════════
# Variant C — багованная формула скорости (для бенчмарка)
# ════════════════════════════════════════════════════════════════════

def aim_tscore(src, target, ships, planets, initial_by_id, omega, iters=6):
    spd = fleet_speed_buggy(ships)
    tx, ty = target.x, target.y
    for _ in range(iters):
        d = math.hypot(tx - src.x, ty - src.y)
        turns = max(1, int(d / spd))
        tx, ty = predict_planet_xy(target.x, target.y, target.radius, omega, turns)
    angle = math.atan2(ty - src.y, tx - src.x)
    lx, ly = launch_point(src.x, src.y, src.radius, angle)
    ex = lx + math.cos(angle) * dist(src.x, src.y, tx, ty)
    ey = ly + math.sin(angle) * dist(src.x, src.y, tx, ty)
    if segment_hits_sun(lx, ly, ex, ey):
        for delta in (0.08, -0.08, 0.16, -0.16, 0.28, -0.28, 0.45, -0.45):
            a2 = angle + delta
            lx2, ly2 = launch_point(src.x, src.y, src.radius, a2)
            ex2 = lx2 + math.cos(a2) * dist(src.x, src.y, tx, ty)
            ey2 = ly2 + math.sin(a2) * dist(src.x, src.y, tx, ty)
            if not segment_hits_sun(lx2, ly2, ex2, ey2):
                angle = a2
                break
        else:
            return None
    return angle, max(1, int(dist(src.x, src.y, tx, ty) / spd)), tx, ty


# ════════════════════════════════════════════════════════════════════
# Variant D — HYBRID (главный): fixpoint + moving blockers + fallback + safe-aim
#
# Принципы наведения (закрепляем явно):
#   • src в момент пуска ведёт себя как СТАЦИОНАРНАЯ. Движок именно так и
#     работает: фаза `Fleet launch` (step 3) идёт ДО `Planet rotation`
#     (step 6). Координаты пуска берём как (src.x, src.y) — без какой-либо
#     компенсации орбитального движения источника. Поэтому когда мы
#     стреляем С движущейся (орбитальной) планеты, никакой коррекции
#     «уноса» не вводим.
#   • цель прогнозируется в зависимости от типа:
#       - СТАЦИОНАРНАЯ: aim прямо на (target.x, target.y), tol по eta = 0;
#       - ОРБИТАЛЬНАЯ:  fixpoint по predict_planet_xy(omega·turns).
#     predict_planet_xy сам определяет тип через is_orbital и для
#     стационарной возвращает текущие координаты — то есть «в стационарные
#     стреляем как стационарные», а в орбитальные — с расчётом
#     криволинейного полёта во вращающихся координатах.
# ════════════════════════════════════════════════════════════════════

def aim_hybrid(src, target, ships, planets, initial_by_id, omega,
               max_iter=5, sweep_horizon=FALLBACK_SWEEP,
               speed_fn=fleet_speed_correct):
    speed = speed_fn(ships)
    if speed <= 0:
        return None
    blockers = [p for p in planets if p.id not in (src.id, target.id)]

    # Источник всегда трактуется как стационарный (в момент пуска):
    # пуск производится из (src.x, src.y), угол строится на эту базу.
    def _eta_from_distance(d):
        """eta = ceil(d/speed), но если d/speed точно целое — добавляем 1
        ход чтобы избежать «касательного» промаха (когда последний сегмент
        флота заканчивается ровно на ближней поверхности цели; движок
        проверяет `d < radius` строго — на касательной не срабатывает)."""
        if d <= 0:
            return 1
        n_float = d / speed
        n_int = int(math.ceil(n_float))
        # касание / почти касание — добавим один ход, чтобы сегмент
        # гарантированно прошёл через центр цели
        if abs(n_int * speed - d) < 1e-6:
            n_int += 1
        return max(1, n_int)

    def est_arrival(tx, ty):
        safe = safe_angle_and_distance(src.x, src.y, src.radius, tx, ty, target.radius, blockers)
        if safe is None:
            return None
        a, d = safe
        return a, _eta_from_distance(d)

    def moving_ok(tx, ty, eta):
        # Используем точную поход-в-ход проверку (line_blocked_precise) —
        # она 1-в-1 с simulate_launch и движком: walk фон по integer-ходам,
        # проверяя позицию каждого блокера в каждый ход. Sample-проверка
        # (line_blocked_moving с 6 точками) пропускает короткие транзиты
        # орбитальных планет через траекторию.
        angle = math.atan2(ty - src.y, tx - src.x)
        lx, ly = launch_point(src.x, src.y, src.radius, angle)
        d = dist(src.x, src.y, tx, ty) - (src.radius + LAUNCH_CLEARANCE) - target.radius
        d = max(0.0, d)
        blocked, _ = line_blocked_precise((lx, ly), angle, ships, d, blockers, omega)
        return not blocked

    target_is_orbital = is_orbital(target.x, target.y, target.radius) and abs(omega) > 1e-12

    # ── Ветка для СТАЦИОНАРНОЙ цели: stationary→stationary прямой aim.
    # Без fixpoint-итераций, без sample-перебора во времени по цели —
    # цель не двигается, поэтому достаточно одной попытки + проверки
    # блокеров (включая орбитальных, движущихся вдоль траектории флота).
    if not target_is_orbital:
        est = est_arrival(target.x, target.y)
        if est is None:
            # цель закрыта солнцем/планетой по прямой — пробуем safe-aim ниже
            pass
        else:
            angle, eta = est
            if moving_ok(target.x, target.y, eta):
                return angle, eta, target.x, target.y
            # прямая линия упёрлась в орбитального блокера — идём в fallback

    # ── Ветка для ОРБИТАЛЬНОЙ цели: fixpoint по предсказанной позиции
    # цели с учётом криволинейного движения во вращающихся координатах.
    if target_is_orbital:
        est = est_arrival(target.x, target.y)
        if est is not None:
            tx, ty = target.x, target.y
            fixpoint_result = None
            for _ in range(max_iter):
                _, turns = est
                ntx, nty = predict_planet_xy(target.x, target.y, target.radius, omega, turns)
                next_est = est_arrival(ntx, nty)
                if next_est is None:
                    break
                if (abs(ntx - tx) < 0.3 and abs(nty - ty) < 0.3
                        and abs(next_est[1] - turns) <= INTERCEPT_TOL):
                    if moving_ok(ntx, nty, next_est[1]):
                        fixpoint_result = (next_est[0], next_est[1], ntx, nty)
                    break
                tx, ty = ntx, nty
                est = next_est
            if fixpoint_result is not None:
                return fixpoint_result

    # Стадия 2: fallback-sweep по eta (актуально только для орбитальной цели,
    # для статической цикл выродится в одну итерацию — predict вернёт ту же
    # точку — и tol_eta=0 даст однозначный ответ).
    tol_eta   = 0 if not target_is_orbital else INTERCEPT_TOL
    best, best_score = None, None
    max_turns = 1 if not target_is_orbital else min(DEFAULT_HORIZON, sweep_horizon)
    for ct in range(1, max_turns + 1):
        px, py = predict_planet_xy(target.x, target.y, target.radius, omega, ct)
        est = est_arrival(px, py)
        if est is None:
            continue
        _, turns = est
        if abs(turns - ct) > tol_eta:
            continue
        actual_turns = max(turns, ct)
        apx, apy = predict_planet_xy(target.x, target.y, target.radius, omega, actual_turns)
        confirm = est_arrival(apx, apy)
        if confirm is None:
            continue
        delta = abs(confirm[1] - actual_turns)
        if delta > tol_eta:
            continue
        if not moving_ok(apx, apy, confirm[1]):
            continue
        score = (delta, confirm[1], ct)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], apx, apy)
    if best is not None:
        return best

    # Стадия 3: safe-aim деформация угла (на precise-проверке)
    direct_angle = math.atan2(target.y - src.y, target.x - src.x)
    full_d = max(1.0, dist(src.x, src.y, target.x, target.y) - src.radius - target.radius)
    for delta in (0.05, -0.05, 0.10, -0.10, 0.18, -0.18, 0.30, -0.30):
        a = direct_angle + delta
        lx, ly = launch_point(src.x, src.y, src.radius, a)
        blocked, _ = line_blocked_precise((lx, ly), a, ships, full_d, blockers, omega)
        if blocked:
            continue
        t_est = _eta_from_distance(full_d)
        ex = lx + math.cos(a) * (speed * t_est)
        ey = ly + math.sin(a) * (speed * t_est)
        return a, t_est, ex, ey
    return None


# ════════════════════════════════════════════════════════════════════
# Интерфейс для agent.py
# ════════════════════════════════════════════════════════════════════

def aim_angle(src, tgt, ships, state) -> float:
    """
    Возвращает угол пуска src → tgt с учётом вращения, солнца и планет-блокеров.
    Внутри использует aim_hybrid. При полной неудаче — прямой угол на текущую цель.
    """
    initial_by_id = state.initial_by_id() if callable(state.initial_by_id) else state.initial_by_id
    result = aim_hybrid(src, tgt, ships, state.planets, initial_by_id, state.omega)
    if result is None:
        return math.atan2(tgt.y - src.y, tgt.x - src.x)
    angle, _, _, _ = result
    return angle


def refine_launch(src, tgt, ships, state):
    """
    Backward-compat: возвращает (angle, eta, (tx, ty)) или None.
    """
    initial_by_id = state.initial_by_id() if callable(state.initial_by_id) else state.initial_by_id
    result = aim_hybrid(src, tgt, ships, state.planets, initial_by_id, state.omega)
    if result is None:
        return None
    angle, eta, tx, ty = result
    return angle, eta, (tx, ty)


# ════════════════════════════════════════════════════════════════════
# Предсказание цели вражеского флота
# ════════════════════════════════════════════════════════════════════

def fleet_target_ray(fleet, planets, horizon=DEFAULT_HORIZON, speed_fn=fleet_speed_correct):
    """Ray-sphere геометрия, без учёта орбитального движения цели."""
    best, best_t = None, 1e9
    dx, dy = math.cos(fleet.angle), math.sin(fleet.angle)
    spd = speed_fn(fleet.ships)
    for p in planets:
        ddx = p.x - fleet.x
        ddy = p.y - fleet.y
        proj = ddx * dx + ddy * dy
        if proj < 0:
            continue
        perp2 = ddx * ddx + ddy * ddy - proj * proj
        r2 = p.radius * p.radius
        if perp2 >= r2:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r2 - perp2)))
        t = hit_d / spd
        if t <= horizon and t < best_t:
            best_t = t
            best = p
    if best is None:
        return None, None
    return best, int(math.ceil(best_t))


def fleet_target_angular(fleet, planets, tol_rad=0.28, speed_fn=fleet_speed_correct):
    """Угловая толеранса — версия target-score-2000-4."""
    best, bd = None, 9999.0
    for p in planets:
        a = math.atan2(p.y - fleet.y, p.x - fleet.x)
        diff = abs((a - fleet.angle + math.pi) % (2 * math.pi) - math.pi)
        if diff < tol_rad:
            d = math.hypot(p.x - fleet.x, p.y - fleet.y)
            if d < bd:
                bd, best = d, p
    if best is None:
        return None, None
    spd = speed_fn(fleet.ships)
    return best, max(1, int(bd / spd))


def fleet_target_time_aware(fleet, planets, omega,
                            horizon=DEFAULT_HORIZON, speed_fn=fleet_speed_correct):
    """Ray-sphere с итеративным учётом орбитального движения цели."""
    best, best_t = None, 1e9
    dx, dy = math.cos(fleet.angle), math.sin(fleet.angle)
    spd = speed_fn(fleet.ships)
    for p in planets:
        t_est = math.hypot(p.x - fleet.x, p.y - fleet.y) / max(spd, 1e-6)
        hit_t = None
        for _ in range(6):
            px, py = predict_planet_xy(p.x, p.y, p.radius, omega, t_est)
            ddx, ddy = px - fleet.x, py - fleet.y
            proj = ddx * dx + ddy * dy
            if proj < 0:
                break
            perp2 = max(0.0, ddx * ddx + ddy * ddy - proj * proj)
            r2 = p.radius * p.radius
            if perp2 >= r2:
                break
            hit_d = max(0.0, proj - math.sqrt(r2 - perp2))
            t_new = hit_d / spd
            if abs(t_new - t_est) < 0.05:
                hit_t = t_new
                break
            t_est = t_new
        if hit_t is not None and hit_t <= horizon and hit_t < best_t:
            best_t = hit_t
            best = p
    if best is None:
        return None, None
    return best, int(math.ceil(best_t))


# ════════════════════════════════════════════════════════════════════
# Ground-truth симулятор (для бенчмаркинга вариантов)
# ════════════════════════════════════════════════════════════════════

def simulate_launch(src, angle, ships, planets, initial_by_id, omega,
                    max_turns=200, speed_fn=fleet_speed_correct):
    """Покадровая симуляция выстрела по правилам движка.
    Возвращает dict: outcome ('HIT'|'OOB'|'SUN'|'TIMEOUT'), turn, planet_id, end."""
    spd = speed_fn(ships)
    lx, ly = launch_point(src.x, src.y, src.radius, angle)
    dx, dy = math.cos(angle), math.sin(angle)
    fx, fy = lx, ly
    for turn in range(1, max_turns + 1):
        nx, ny = fx + spd * dx, fy + spd * dy
        if nx < 0 or nx > BOARD or ny < 0 or ny > BOARD:
            return dict(outcome="OOB", turn=turn, planet_id=None, end=(nx, ny))
        if point_to_segment_dist(CX, CY, fx, fy, nx, ny) < SUN_R:
            return dict(outcome="SUN", turn=turn, planet_id=None, end=(nx, ny))
        best_pid, best_d = None, float("inf")
        for p in planets:
            if p.id == src.id:
                continue
            px, py = predict_planet_xy(p.x, p.y, p.radius, omega, turn)
            d = point_to_segment_dist(px, py, fx, fy, nx, ny)
            if d < p.radius and d < best_d:
                best_d, best_pid = d, p.id
        if best_pid is not None:
            return dict(outcome="HIT", turn=turn, planet_id=best_pid, end=(nx, ny))
        fx, fy = nx, ny
    return dict(outcome="TIMEOUT", turn=max_turns, planet_id=None, end=(fx, fy))


# ════════════════════════════════════════════════════════════════════
# Реестры (для бенчмарков)
# ════════════════════════════════════════════════════════════════════

AIM_VARIANTS = {
    "A_baseline":   aim_baseline,
    "B_fixpoint":   aim_fixpoint,
    "C_tscore":     aim_tscore,
    "D_hybrid":     aim_hybrid,
}

FLEET_TARGET_VARIANTS = {
    "ray":         lambda f, pl, ibid, om: fleet_target_ray(f, pl),
    "angular":     lambda f, pl, ibid, om: fleet_target_angular(f, pl),
    "time_aware":  lambda f, pl, ibid, om: fleet_target_time_aware(f, pl, om),
}


__all__ = [
    # типы
    'Planet', 'Fleet',
    # константы
    'CX', 'CY', 'SUN_R', 'BOARD', 'ROTATION_LIMIT',
    'LAUNCH_CLEARANCE', 'SUN_SAFETY', 'BLOCKER_SAFETY', 'MAX_SPEED',
    'INTERCEPT_TOL', 'DEFAULT_HORIZON', 'FALLBACK_SWEEP', 'TOTAL_STEPS',
    # геометрия
    'dist', 'point_to_segment_dist', 'segment_hits_sun',
    'launch_point', 'actual_path_geometry',
    # физика
    'is_orbital', 'fleet_speed_correct', 'fleet_speed_buggy', 'predict_planet_xy',
    # блокеры
    'line_blocked_static', 'line_blocked_moving', 'safe_angle_and_distance',
    # варианты наведения
    'aim_baseline', 'aim_fixpoint', 'aim_tscore', 'aim_hybrid',
    # интерфейс агента
    'aim_angle', 'refine_launch',
    # предсказатели чужих флотов
    'fleet_target_ray', 'fleet_target_angular', 'fleet_target_time_aware',
    # симулятор
    'simulate_launch',
    # реестры
    'AIM_VARIANTS', 'FLEET_TARGET_VARIANTS',
]

# ╔══════════════════════════════════════════════════╗
# ║  orbit_sim.py                                  ║
# ╚══════════════════════════════════════════════════╝

"""
orbit_sim.py — автономный симулятор Orbit Wars + граф-метрика связности

Не требует kaggle-environments. Физика взята из README + shooting.py.

Основные блоки:
  1. GameState         — снимок состояния игры
  2. generate_map()    — генерация карты по сиду (4-кратная симметрия)
  3. simulate_step()   — один ход движка (production → move → rotate → combat)
  4. GraphConnectivity — 4 варианта весов рёбер + скоры по планетам
  5. JSON import/export — совместимо с форматом kaggle-environments obs
"""

import math
import random
import json
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ── переиспользуем физику из shooting.py ──────────────────────────────────
import sys, os


# ══════════════════════════════════════════════════════════════════════════
# 1. GameState
# ══════════════════════════════════════════════════════════════════════════

class GameState:
    """Полный снимок состояния игры."""

    def __init__(self, planets: List[Planet], fleets: List[Fleet],
                 omega: float, step: int = 0,
                 initial_planets: Optional[List[Planet]] = None,
                 n_players: int = 2,
                 comet_ids: Optional[set] = None,
                 comets: Optional[list] = None):
        self.planets = list(planets)
        self.fleets  = list(fleets)
        self.omega   = omega
        self.step    = step
        self.n_players = n_players
        # initial_planets нужны для predict_planet_xy (угол от t=0)
        self._initial_planets: Dict[int, Planet] = (
            {p.id: p for p in initial_planets}
            if initial_planets
            else {p.id: p for p in planets}
        )
        self._next_fleet_id = max((f.id for f in fleets), default=-1) + 1
        # Кометы: appearing every ~50 turns, временные планеты с собственными
        # путями. Мы их осознанно игнорируем в zones/priorities, а в projection
        # обрабатываем как «трамплин» — корабли с захваченной кометы (минус 1)
        # уходят на ближайшую нашу планету.
        self.comet_ids: set = set(comet_ids) if comet_ids else set()
        self.comets: list   = list(comets) if comets else []

    # ── удобные срезы ──────────────────────────────────────────────────────

    def planets_of(self, player: int) -> List[Planet]:
        return [p for p in self.planets if p.owner == player]

    def neutral_planets(self) -> List[Planet]:
        return [p for p in self.planets if p.owner == -1]

    def enemy_planets(self, player: int) -> List[Planet]:
        return [p for p in self.planets if p.owner not in (-1, player)]

    def planet_by_id(self, pid: int) -> Optional[Planet]:
        for p in self.planets:
            if p.id == pid:
                return p
        return None

    def initial_by_id(self) -> Dict[int, Planet]:
        return self._initial_planets

    # ── предсказание позиции планеты через turns ходов ────────────────────

    def predict_pos(self, planet: Planet, turns: int) -> Tuple[float, float]:
        init = self._initial_planets.get(planet.id, planet)
        return predict_planet_xy(init.x, init.y, init.radius, self.omega, turns)

    # ── сериализация ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "omega": self.omega,
            "n_players": self.n_players,
            "planets": [list(p) for p in self.planets],
            "fleets":  [list(f) for f in self.fleets],
            "initial_planets": [list(p) for p in self._initial_planets.values()],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        planets = [Planet(*p) for p in d["planets"]]
        fleets  = [Fleet(*f)  for f in d["fleets"]]
        init    = [Planet(*p) for p in d.get("initial_planets", d["planets"])]
        state = cls(
            planets=planets, fleets=fleets,
            omega=d["omega"], step=d.get("step", 0),
            initial_planets=init,
            n_players=d.get("n_players", 2),
        )
        state.game_idx = d.get("game_idx", None)
        return state

    @classmethod
    def from_json(cls, path_or_str: str) -> "GameState":
        if path_or_str.strip().startswith("{"):
            d = json.loads(path_or_str)
        else:
            with open(path_or_str) as fh:
                d = json.load(fh)
        return cls.from_dict(d)

    # ── импорт из kaggle obs ───────────────────────────────────────────────

    @classmethod
    def from_kaggle_obs(cls, obs: dict, step: int = 0) -> "GameState":
        """obs — словарь observation из kaggle_environments.
        step берётся из obs (если есть), иначе из аргумента — нужен для
        late-game-фич zones (late_aggression растёт с phase = step/TOTAL_STEPS).
        """
        planets = [Planet(*p) for p in obs.get("planets", [])]
        fleets  = [Fleet(*f)  for f in obs.get("fleets", [])]
        init    = [Planet(*p) for p in obs.get("initial_planets", obs.get("planets", []))]
        omega   = obs.get("angular_velocity", 0.0) or 0.0
        comet_ids = set(obs.get("comet_planet_ids", []) or [])
        comets    = obs.get("comets", []) or []
        # step может быть в obs как 'step' или 'stepNumber'; иначе используем аргумент
        obs_step = obs.get("step")
        if not isinstance(obs_step, int):
            obs_step = obs.get("stepNumber")
        if isinstance(obs_step, int):
            step = obs_step
        return cls(planets=planets, fleets=fleets, omega=omega,
                   step=step, initial_planets=init,
                   comet_ids=comet_ids, comets=comets)

    def __repr__(self) -> str:
        return (f"GameState(step={self.step}, planets={len(self.planets)}, "
                f"fleets={len(self.fleets)}, omega={self.omega:.4f})")


# ══════════════════════════════════════════════════════════════════════════
# 2. Генерация карты
# ══════════════════════════════════════════════════════════════════════════

def _prod_to_radius(prod: int) -> float:
    return 1.0 + math.log(max(1, prod))


def _place_group(rng: random.Random, pid_counter: list,
                 orbital: bool, home: bool = False,
                 owner_ids: Optional[List[int]] = None) -> List[Planet]:
    """Генерирует группу из 4 планет с 4-кратной симметрией."""
    prod = rng.randint(1, 5) if not home else 3
    radius = _prod_to_radius(prod)

    if orbital:
        # Планета должна быть внутри ROTATION_LIMIT - radius
        max_r = ROTATION_LIMIT - radius - 2.0
        min_r = SUN_R + radius + 3.0
        orb_r = rng.uniform(min_r, max_r)
        angle = rng.uniform(0, math.pi / 2)  # Q1, потом симметрия
        bx = CX + orb_r * math.cos(angle)
        by = CY + orb_r * math.sin(angle)
    else:
        # Статичная: дальше ROTATION_LIMIT
        min_r = ROTATION_LIMIT + radius + 2.0
        max_r = 48.0 - radius
        if min_r >= max_r:
            min_r = ROTATION_LIMIT + radius + 1.0
            max_r = min_r + 4.0
        orb_r = rng.uniform(min_r, max_r)
        angle = rng.uniform(0, math.pi / 2)
        bx = CX + orb_r * math.cos(angle)
        by = CY + orb_r * math.sin(angle)

    # 4-кратная симметрия: (x,y), (100-x,y), (x,100-y), (100-x,100-y)
    corners = [
        (bx,        by),
        (BOARD - bx, by),
        (bx,        BOARD - by),
        (BOARD - bx, BOARD - by),
    ]

    ships_base = rng.randint(5, 25) if not home else 10
    planets = []
    for i, (cx, cy) in enumerate(corners):
        owner = -1
        ships = ships_base
        if home and owner_ids is not None:
            owner = owner_ids[i] if i < len(owner_ids) else -1
            ships = 10
        pid = pid_counter[0]
        pid_counter[0] += 1
        planets.append(Planet(pid, owner, cx, cy, radius, ships, prod))

    return planets


def generate_map(seed: int = 42, n_groups: int = 6,
                 omega: Optional[float] = None,
                 n_players: int = 2) -> GameState:
    """
    Генерирует карту по сиду. Правила:
    - 4-кратная симметрия
    - ≥ 1 орбитальная группа, ≥ 3 статичных
    - 1 домашняя группа (стартовые планеты игроков)
    - omega в диапазоне [0.025, 0.05] если не задано
    """
    rng = random.Random(seed)
    if omega is None:
        omega = rng.uniform(0.025, 0.05)

    pid_counter = [0]
    all_planets: List[Planet] = []

    # Домашняя группа — всегда статичная (для простоты, как часто бывает)
    owner_ids = list(range(n_players)) + [-1] * (4 - n_players)
    home_group = _place_group(rng, pid_counter, orbital=False,
                               home=True, owner_ids=owner_ids)
    all_planets.extend(home_group)

    # Остальные группы
    remaining = n_groups - 1
    n_orbital = max(1, rng.randint(1, max(1, remaining - 2)))
    n_static  = remaining - n_orbital

    for _ in range(n_orbital):
        all_planets.extend(_place_group(rng, pid_counter, orbital=True))
    for _ in range(n_static):
        all_planets.extend(_place_group(rng, pid_counter, orbital=False))

    initial = list(all_planets)
    return GameState(planets=all_planets, fleets=[], omega=omega,
                     initial_planets=initial, n_players=n_players)


# ══════════════════════════════════════════════════════════════════════════
# 3. Симуляция одного хода
# ══════════════════════════════════════════════════════════════════════════

def _fleet_speed(ships: int) -> float:
    return fleet_speed_correct(ships)


def _resolve_combat(garrison_owner: int, garrison_ships: int,
                    arrivals: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    arrivals: list of (owner, ships)
    Возвращает (new_owner, new_ships).
    """
    # Сгруппировать по владельцу
    by_owner: Dict[int, int] = defaultdict(int)
    by_owner[garrison_owner] += garrison_ships
    for owner, ships in arrivals:
        by_owner[owner] += ships

    # Итерации боя: самый сильный vs второй по силе
    while len(by_owner) > 1:
        sorted_owners = sorted(by_owner.items(), key=lambda x: -x[1])
        top_owner, top_ships = sorted_owners[0]
        sec_owner, sec_ships = sorted_owners[1]
        diff = top_ships - sec_ships
        del by_owner[sec_owner]
        if diff == 0:
            del by_owner[top_owner]
        else:
            by_owner[top_owner] = diff

    if not by_owner:
        # Все уничтожены — нейтральная планета остаётся
        return garrison_owner, 0

    winner_owner, winner_ships = next(iter(by_owner.items()))
    return winner_owner, winner_ships


def simulate_step(state: GameState,
                  moves_per_player: Optional[Dict[int, List]] = None) -> GameState:
    """
    Симулирует один ход. moves_per_player:
      {player_id: [[from_planet_id, angle, ships], ...]}

    Порядок: launch → production → fleet move → rotate → combat
    """
    if moves_per_player is None:
        moves_per_player = {}

    planets = {p.id: list(p) for p in state.planets}  # mutable
    fleets: List[list] = [list(f) for f in state.fleets]
    next_fid = state._next_fleet_id

    # ── 1. Fleet launch ────────────────────────────────────────────────────
    for player_id, moves in moves_per_player.items():
        for move in moves:
            src_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            if src_id not in planets:
                continue
            src = planets[src_id]
            if src[1] != player_id:   # не наша
                continue
            ships = min(ships, src[5])
            if ships < 1:
                continue
            # Spawn point
            lx = src[2] + math.cos(angle) * (src[4] + LAUNCH_CLEARANCE)
            ly = src[3] + math.sin(angle) * (src[4] + LAUNCH_CLEARANCE)
            fleets.append([next_fid, player_id, lx, ly, angle, src_id, ships])
            next_fid += 1
            planets[src_id][5] -= ships  # списываем корабли

    # ── 2. Production ──────────────────────────────────────────────────────
    for p in planets.values():
        if p[1] >= 0:  # owned
            p[5] += p[6]  # ships += production

    # ── 3. Fleet movement + collision detection ────────────────────────────
    combat_queue: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    surviving_fleets = []

    for f in fleets:
        fid, fowner, fx, fy, fangle, from_pid, fships = f
        speed = _fleet_speed(fships)
        nx = fx + math.cos(fangle) * speed
        ny = fy + math.sin(fangle) * speed

        # Out of bounds
        if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
            continue  # флот уничтожен

        # Sun collision
        if point_to_segment_dist(CX, CY, fx, fy, nx, ny) < SUN_R:
            continue

        # Planet collision
        hit_pid = None
        hit_d   = float("inf")
        for p in planets.values():
            d = point_to_segment_dist(p[2], p[3], fx, fy, nx, ny)
            if d < p[4] and d < hit_d:
                hit_d   = d
                hit_pid = p[0]

        if hit_pid is not None:
            combat_queue[hit_pid].append((fowner, fships))
        else:
            f[2], f[3] = nx, ny  # обновляем позицию
            surviving_fleets.append(f)

    # ── 4. Planet rotation ─────────────────────────────────────────────────
    if abs(state.omega) > 1e-9:
        for pid, p in planets.items():
            if is_orbital(p[2], p[3], p[4]):
                # rotate by omega
                rx = p[2] - CX
                ry = p[3] - CY
                r  = math.hypot(rx, ry)
                a  = math.atan2(ry, rx) + state.omega
                p[2] = CX + r * math.cos(a)
                p[3] = CY + r * math.sin(a)

    # Sweep: флоты которые попали в орбитальную планету после её поворота
    still_surviving = []
    for f in surviving_fleets:
        fid, fowner, fx, fy, fangle, from_pid, fships = f
        swept = False
        for p in planets.values():
            if math.hypot(fx - p[2], fy - p[3]) < p[4]:
                combat_queue[p[0]].append((fowner, fships))
                swept = True
                break
        if not swept:
            still_surviving.append(f)

    # ── 5. Combat resolution ───────────────────────────────────────────────
    for planet_id, arrivals in combat_queue.items():
        if planet_id not in planets:
            continue
        p = planets[planet_id]
        new_owner, new_ships = _resolve_combat(p[1], p[5], arrivals)
        p[1] = new_owner
        p[5] = new_ships

    # ── Собираем новое состояние ───────────────────────────────────────────
    new_planets = [Planet(*p) for p in planets.values()]
    new_fleets  = [Fleet(*f)  for f in still_surviving]

    new_state = GameState(
        planets=new_planets,
        fleets=new_fleets,
        omega=state.omega,
        step=state.step + 1,
        initial_planets=list(state._initial_planets.values()),
        n_players=state.n_players,
    )
    new_state._next_fleet_id = next_fid
    return new_state


def run_simulation(state: GameState, n_steps: int,
                   policy=None) -> List[GameState]:
    """
    Прогоняет n_steps ходов.
    policy(state, player_id) -> [[from_id, angle, ships], ...]
    """
    history = [state]
    for _ in range(n_steps):
        moves = {}
        if policy:
            for pid in range(state.n_players):
                moves[pid] = policy(history[-1], pid)
        history.append(simulate_step(history[-1], moves))
    return history


# ══════════════════════════════════════════════════════════════════════════
# 4. Граф связности (Graph Connectivity)
# ══════════════════════════════════════════════════════════════════════════

def _travel_time(src: Planet, dst: Planet, ships: int, omega: float,
                 max_iter: int = 4) -> float:
    """
    Оценка времени полёта от src до dst с учётом орбитального движения dst.
    Fixpoint по ETA, как в shooting.py.
    """
    speed = _fleet_speed(max(1, ships))
    if speed < 1e-6:
        return 1e9

    d0 = max(0.5, dist(src.x, src.y, dst.x, dst.y) - src.radius - dst.radius)
    t  = d0 / speed

    for _ in range(max_iter):
        px, py = predict_planet_xy(dst.x, dst.y, dst.radius, omega, int(t))
        d = max(0.5, dist(src.x, src.y, px, py) - src.radius - dst.radius)
        t_new = d / speed
        if abs(t_new - t) < 0.5:
            return t_new
        t = t_new
    return t


# ── Четыре варианта весов рёбер ───────────────────────────────────────────

def _w1_inv_dist(src: Planet, dst: Planet, ships: int,
                 omega: float, **_) -> float:
    """W1: пропорционально 1 / travel_time."""
    t = _travel_time(src, dst, ships, omega)
    return 1.0 / max(t, 0.1)


def _w2_relative_centrality(src: Planet, dst: Planet, ships: int,
                             omega: float, all_planets: List[Planet],
                             **_) -> float:
    """W2: обратно пропорционально среднему времени src до всех планет."""
    avg_t = sum(_travel_time(src, p, ships, omega)
                for p in all_planets if p.id != src.id) / max(1, len(all_planets) - 1)
    t = _travel_time(src, dst, ships, omega)
    return avg_t / max(t, 0.1)  # если src центральная — avg_t мало, вес мал


def _w3_strength_weighted(src: Planet, dst: Planet, ships: int,
                           omega: float, **_) -> float:
    """W3: ships / travel_time — сколько кораблей в секунду можно доставить."""
    t = _travel_time(src, dst, max(1, src.ships), omega)
    return src.ships / max(t, 0.1)


def _w4_neighbor_potential(src: Planet, dst: Planet, ships: int,
                            omega: float, all_planets: List[Planet],
                            **_) -> float:
    """
    W4: ships / t * (1 + λ * Σ_k S_k/t(dst,k))
    — поправка на связность соседей dst.
    """
    t_src_dst = _travel_time(src, dst, max(1, src.ships), omega)
    base = src.ships / max(t_src_dst, 0.1)

    # Потенциал dst: сумма S_k/t(dst→k) по соседям
    neighbor_potential = sum(
        p.ships / max(_travel_time(dst, p, max(1, p.ships), omega), 0.1)
        for p in all_planets if p.id != dst.id and p.id != src.id
    )
    lam = 0.1  # регулятор — можно тюнить
    return base * (1.0 + lam * neighbor_potential)


WEIGHT_VARIANTS = {
    "W1_inv_dist":          _w1_inv_dist,
    "W2_centrality":        _w2_relative_centrality,
    "W3_strength":          _w3_strength_weighted,
    "W4_neighbor_potential": _w4_neighbor_potential,
}


class GraphConnectivity:
    """
    Граф связности для одного состояния игры.

    Каждое ребро i→j имеет вес w(i,j) и знак:
      +1  если src — наш
      -1  если src — враг
       0  если src — нейтрал

    conn(j) = Σ_i sign(i) · w(i,j)   — "давление" на планету j
    delta(t) = Σ_{i: mine} [w(i,t) + w(t,i)]  — прирост связности при захвате t
    """

    def __init__(self, state: GameState, player: int,
                 weight_fn=None, ships_ref: int = 50):
        self.state   = state
        self.player  = player
        self.ships_ref = ships_ref
        self._wfn    = weight_fn or _w3_strength_weighted
        self._cache: Dict[Tuple[int, int], float] = {}

    def edge_weight(self, src: Planet, dst: Planet) -> float:
        key = (src.id, dst.id)
        if key not in self._cache:
            self._cache[key] = self._wfn(
                src, dst, self.ships_ref, self.state.omega,
                all_planets=self.state.planets
            )
        return self._cache[key]

    def _sign(self, p: Planet) -> int:
        if p.owner == self.player:
            return +1
        if p.owner == -1:
            return 0
        return -1

    def signed_pressure(self, target: Planet) -> float:
        """Σ_i sign(i) * w(i→target) — давление на target."""
        return sum(
            self._sign(src) * self.edge_weight(src, target)
            for src in self.state.planets
            if src.id != target.id
        )

    def all_pressures(self) -> Dict[int, float]:
        """signed_pressure для всех планет."""
        return {p.id: self.signed_pressure(p) for p in self.state.planets}

    def team_connectivity(self) -> float:
        """Внутренняя связность нашей сети: Σ_{i,j: mine} w(i→j)."""
        mine = self.state.planets_of(self.player)
        return sum(
            self.edge_weight(i, j)
            for i in mine for j in mine
            if i.id != j.id
        )

    def marginal_capture_value(self, target: Planet) -> float:
        """
        Прирост team_connectivity при захвате target.
        = Σ_{i: mine} [w(i→target) + w(target→i)]
        """
        mine = self.state.planets_of(self.player)
        return sum(
            self.edge_weight(m, target) + self.edge_weight(target, m)
            for m in mine
        )

    def capture_priority(self) -> List[Tuple[Planet, float]]:
        """
        Сортировка не-наших планет по marginal_capture_value (убывание).
        """
        non_mine = [p for p in self.state.planets if p.owner != self.player]
        scored = [(p, self.marginal_capture_value(p)) for p in non_mine]
        scored.sort(key=lambda x: -x[1])
        return scored

    def shortcut_planets(self) -> List[Tuple[Planet, float]]:
        """
        Планеты на кратчайших путях (low travel time от наших до вражеских).
        Возвращает не-наши планеты, которые являются 'шорткатами':
        их захват наиболее сокращает путь до вражеской сети.
        """
        mine   = self.state.planets_of(self.player)
        enemy  = self.state.enemy_planets(self.player)
        others = [p for p in self.state.planets
                  if p.owner != self.player and p.id not in {e.id for e in enemy}]
        # Для каждой нейтральной/враж планеты: насколько она ближе к врагу?
        result = []
        for candidate in others:
            # shortest time from candidate to nearest enemy
            t_to_enemy = min(
                (_travel_time(candidate, e, self.ships_ref, self.state.omega)
                 for e in enemy), default=1e9
            )
            # shortest time from our planets to candidate
            t_from_mine = min(
                (_travel_time(m, candidate, self.ships_ref, self.state.omega)
                 for m in mine), default=1e9
            )
            # composite: быстро достать + быстро от кандидата к врагу
            score = 1.0 / max(t_from_mine + t_to_enemy, 0.1)
            result.append((candidate, score))
        result.sort(key=lambda x: -x[1])
        return result

    def summary(self) -> dict:
        pressures = self.all_pressures()
        prio      = self.capture_priority()
        shortcuts = self.shortcut_planets()
        return {
            "player": self.player,
            "team_connectivity": round(self.team_connectivity(), 3),
            "pressures": {pid: round(v, 3) for pid, v in pressures.items()},
            "capture_priority": [
                {"planet_id": p.id, "owner": p.owner,
                 "ships": p.ships, "prod": p.production,
                 "marginal_value": round(v, 3)}
                for p, v in prio[:8]
            ],
            "shortcut_candidates": [
                {"planet_id": p.id, "owner": p.owner, "score": round(v, 4)}
                for p, v in shortcuts[:5]
            ],
        }


# ══════════════════════════════════════════════════════════════════════════
# 5. Утилиты для экспорта/сравнения вариантов
# ══════════════════════════════════════════════════════════════════════════

def compare_weight_variants(state: GameState, player: int,
                             ships_ref: int = 50) -> Dict[str, dict]:
    """
    Считает capture_priority для каждого из 4 вариантов весов.
    Возвращает dict {variant_name: ranked_list}.
    """
    result = {}
    for name, wfn in WEIGHT_VARIANTS.items():
        gc = GraphConnectivity(state, player, weight_fn=wfn, ships_ref=ships_ref)
        prio = gc.capture_priority()
        result[name] = [
            {"planet_id": p.id, "owner": p.owner,
             "ships": p.ships, "prod": p.production,
             "value": round(v, 4)}
            for p, v in prio[:6]
        ]
    return result


def export_history(history: List[GameState], path: str):
    """Сохраняет последовательность состояний в JSON."""
    data = [s.to_dict() for s in history]
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Сохранено {len(history)} состояний → {path}")


def load_history(path: str) -> List[GameState]:
    """Загружает историю из JSON (совместимо с export_history и colab_export)."""
    with open(path) as fh:
        data = json.load(fh)
    states = []
    skipped = 0
    for i, d in enumerate(data):
        # Пропускаем пустые состояния (kaggle step=0 бывает без планет)
        planets_raw = d.get("planets", [])
        if not planets_raw:
            skipped += 1
            continue
        try:
            if "omega" in d:
                s = GameState.from_dict(d)
            else:
                s = GameState.from_kaggle_obs(d, step=i)
            states.append(s)
        except Exception as e:
            skipped += 1
    if skipped:
        print(f"  (пропущено {skipped} пустых/невалидных состояний)")
    return states


# ══════════════════════════════════════════════════════════════════════════
# Быстрый smoke-test при запуске как скрипт
# ══════════════════════════════════════════════════════════════════════════

if False:  # disabled in submission (kaggle exec)
    import pprint

    print("=== Генерация карты seed=42 ===")
    state = generate_map(seed=42, n_groups=6)
    print(state)
    print(f"Планет: {len(state.planets)}, орбитальных: "
          f"{sum(1 for p in state.planets if is_orbital(p.x, p.y, p.radius))}")

    for p in state.planets:
        tag = "orb" if is_orbital(p.x, p.y, p.radius) else "sta"
        print(f"  [{tag}] id={p.id} owner={p.owner} ships={p.ships} "
              f"prod={p.production} pos=({p.x:.1f},{p.y:.1f})")

    print("\n=== Graph Connectivity (W3_strength, player=0) ===")
    gc = GraphConnectivity(state, player=0, weight_fn=_w3_strength_weighted)
    s  = gc.summary()
    pprint.pprint(s)

    print("\n=== Сравнение вариантов весов ===")
    comp = compare_weight_variants(state, player=0)
    for name, ranked in comp.items():
        ids = [f"p{r['planet_id']}(o{r['owner']})" for r in ranked[:4]]
        print(f"  {name:30s}: {ids}")

    print("\n=== 10 шагов симуляции ===")
    history = run_simulation(state, n_steps=10)
    print(f"Шаги: {[s.step for s in history]}")
    final = history[-1]
    for pid in range(state.n_players):
        print(f"  Игрок {pid}: {len(final.planets_of(pid))} планет, "
              f"{sum(p.ships for p in final.planets_of(pid))} кораблей")

    print("\nOK")

# ╔══════════════════════════════════════════════════╗
# ║  force.py                                      ║
# ╚══════════════════════════════════════════════════╝

"""
Базовый блок: кривые влияния / силы.

Ключевые экспорты:
  _rendezvous_eta(src, dst, ships, omega) -> (turns, (px, py))
  force_events(state, target, ...)        -> список событий
  build_net_curve(events, target, ...)    -> (xs, ys)
  build_area_curve(events, target, ...)   -> (xs, area_ys)
"""


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

# ╔══════════════════════════════════════════════════╗
# ║  agent_debug.py                                ║
# ╚══════════════════════════════════════════════════╝

"""
Пер-ходовой отладочный лог агента.

Активируется переменной окружения `ORBIT_AGENT_LOG` (путь к лог-файлу).
Если переменная не задана — все функции no-op (накладные расходы ≈ 0).
Формат — текст, удобный для grep/чтения глазами:

  ===== TURN <step>  player=<p>  ts=<...>  =====
  [ZONES]
     pid owner zone              priority   prod ships proj_at  area_inv  wnn_close   ...
     ...
  [TARGETS]  N=...
     pid (zone, priority)
     ...
  [PLANS]   N=...
     idx mode       sup→att→tgt   eta  x_sup  x_att   defender  margin  success
     ...
  [DECISIONS]
     plan#X mode=...  -> FIRE | SKIP: <reason>
     ...
  [MOVES]   N=...
     [src, angle_deg, ships]
     ...
  [FLEETS]  flying fleets in obs (own + enemy)
     ...

Управление:
  ORBIT_AGENT_LOG=/path/to/file.log     — путь к лог-файлу (append-mode)
  ORBIT_AGENT_LOG_FLEETS=1              — также логировать список летящих флотов
  ORBIT_AGENT_LOG_PLANS_ALL=1           — логировать ВСЕ кандидатные планы
                                          (по умолчанию только финальный
                                          best_attacks top-N)
"""

import os
import time
import math

# ── ленивый init ──────────────────────────────────────────────────────────
_LOG_FH       = None
_RESOLVED     = False
_DISABLED     = True
_LOG_FLEETS   = False
_LOG_PLANS_ALL = False


def _resolve():
    global _LOG_FH, _RESOLVED, _DISABLED, _LOG_FLEETS, _LOG_PLANS_ALL
    if _RESOLVED:
        return
    _RESOLVED = True
    path = os.environ.get('ORBIT_AGENT_LOG', '').strip()
    if not path:
        return
    try:
        _LOG_FH = open(path, 'a', buffering=1, encoding='utf-8')
        _DISABLED = False
        _LOG_FLEETS    = bool(int(os.environ.get('ORBIT_AGENT_LOG_FLEETS', '0') or '0'))
        _LOG_PLANS_ALL = bool(int(os.environ.get('ORBIT_AGENT_LOG_PLANS_ALL', '0') or '0'))
    except Exception:
        _DISABLED = True


def enabled():
    _resolve()
    return not _DISABLED


def reset():
    """Сбросить кеш и закрыть текущий лог-файл. Полезно в ноутбуке когда
    меняем ORBIT_AGENT_LOG между запусками — без reset() модуль помнит
    первое решение и игнорирует новые env-переменные."""
    global _LOG_FH, _RESOLVED, _DISABLED, _LOG_FLEETS, _LOG_PLANS_ALL
    try:
        if _LOG_FH is not None:
            _LOG_FH.flush()
            _LOG_FH.close()
    except Exception:
        pass
    _LOG_FH = None
    _RESOLVED = False
    _DISABLED = True
    _LOG_FLEETS = False
    _LOG_PLANS_ALL = False
    if hasattr(log_decision, '_started'):
        delattr(log_decision, '_started')


def _w(line=''):
    if not enabled():
        return
    try:
        _LOG_FH.write(line + '\n')
    except Exception:
        pass


def _safe(v, fmt='{:.2f}'):
    try:
        if v is None:
            return '-'
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return str(v)
            return fmt.format(v)
        return str(v)
    except Exception:
        return str(v)


# ── секции ────────────────────────────────────────────────────────────────

def begin_turn(step, player, n_planets, n_fleets):
    if not enabled():
        return
    # При каждом новом матче (step==0) пишем разделитель с таймстампом — сразу
    # видно где начинается новая партия и какой версией агента записан лог.
    if step == 0:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        _w()
        _w('=' * 72)
        _w(f'  SESSION  {ts}')
        _w('=' * 72)
    _w()
    _w(f'===== TURN {step}  player={player}  ts={time.strftime("%Y-%m-%d %H:%M:%S")}  '
       f'planets={n_planets}  fleets={n_fleets}  =====')


def log_zones(df, projected_at=None):
    """df — output of compute_zones_from_state. Печатаем компактную таблицу."""
    if not enabled() or df is None:
        return
    _w('[ZONES]')
    cols = ['pid', 'owner', 'zone', 'priority', 'prod', 'ships',
            'area_inv', 'wnn_close_res', 'mean_dist_all', 'n_cross']
    header = '   pid own zone              priority   prod ships  proj_at  area_inv  wnn_close mean_d   n_cr'
    _w(header)
    # сортировка: сначала наши по priority desc, потом цели по priority desc
    try:
        rows = df.sort_values(['owner', 'priority'], ascending=[True, False]).itertuples(index=False)
    except Exception:
        rows = df.itertuples(index=False)
    for r in rows:
        d = r._asdict() if hasattr(r, '_asdict') else dict(zip(df.columns, r))
        pid   = int(d.get('pid', -1))
        owner = int(d.get('owner', -1))
        zone  = str(d.get('zone', '?'))
        prio  = float(d.get('priority', 0.0))
        prod  = int(d.get('prod', 0))
        ships = int(d.get('ships', 0))
        ai    = float(d.get('area_inv', 0.0))
        wn    = float(d.get('wnn_close_res', 0.0))
        md    = float(d.get('mean_dist_all', 0.0))
        nc    = int(d.get('n_cross', 0))
        proj  = (projected_at or {}).get(pid, 0)
        _w(f'   {pid:>3} {owner:>3} {zone:<16}  {prio:>+7.2f}  {prod:>4} {ships:>5}  {proj:>6}   '
           f'{ai:>+7.2f}  {wn:>+7.2f}  {md:>5.1f}  {nc:>4}')


def log_targets(target_ids, zones_df=None):
    if not enabled():
        return
    _w(f'[TARGETS]  N={len(target_ids)}')
    if not target_ids:
        return
    by_pid = {}
    if zones_df is not None:
        try:
            for _, r in zones_df.iterrows():
                by_pid[int(r['pid'])] = (str(r.get('zone', '?')), float(r.get('priority', 0.0)))
        except Exception:
            pass
    for pid in target_ids:
        zone, prio = by_pid.get(int(pid), ('?', 0.0))
        _w(f'   pid={pid:<3} zone={zone:<16}  priority={prio:+.2f}')


def log_plans(plans, label='PLANS'):
    if not enabled() or not plans:
        if enabled():
            _w(f'[{label}]  N=0')
        return
    _w(f'[{label}]  N={len(plans)}')
    _w('   #  mode        sup→att→tgt           eta_sa eta_at  x_sup  x_att   def  margin  succ  reason')
    for i, p in enumerate(plans):
        mode  = p.get('mode', '?')
        sup   = p.get('sup_id', '-')
        att   = p.get('att_id', '-')
        tgt   = p.get('tgt_id', '-')
        eta_sa = p.get('eta_sa', 0)
        eta_at = p.get('eta_at', p.get('t_total', 0))
        x_sup  = int(p.get('x_sup', 0))
        x_att  = int(p.get('x_att', 0))
        defd   = p.get('defender', 0)
        marg   = p.get('margin', 0)
        succ   = bool(p.get('success', False))
        reason = p.get('relay_reason', None) or ''
        triple = f'{sup}→{att}→{tgt}' if sup not in (None, '-') else f'   {att}→{tgt}'
        _w(f'   {i:>2}  {mode:<10}  {triple:<20}  '
           f'{_safe(eta_sa, "{:.1f}"):>5}  {_safe(eta_at, "{:.1f}"):>5}  '
           f'{x_sup:>5}  {x_att:>5}  {_safe(defd, "{:.1f}"):>5}  {_safe(marg, "{:+.1f}"):>6}  '
           f'{"Y" if succ else "n":>4}  {reason}')


def log_decision(plan_idx, plan, status, reason=''):
    """status ∈ {'FIRE', 'SKIP'}"""
    if not enabled():
        return
    if not hasattr(log_decision, '_started'):
        _w('[DECISIONS]')
        log_decision._started = True
    mode = plan.get('mode', '?')
    sup  = plan.get('sup_id', '-')
    att  = plan.get('att_id', '-')
    tgt  = plan.get('tgt_id', '-')
    triple = f'{sup}→{att}→{tgt}' if sup not in (None, '-') else f'{att}→{tgt}'
    _w(f'   plan#{plan_idx} {mode:<10} {triple:<14} -> {status}: {reason}')


def reset_decisions():
    """Сбросить «уже напечатано» — между ходами."""
    if hasattr(log_decision, '_started'):
        delattr(log_decision, '_started')


def log_moves(moves):
    if not enabled():
        return
    _w(f'[MOVES]  N={len(moves)}')
    for m in moves:
        try:
            src, angle, ships = m
            _w(f'   src={int(src):>3}  angle={math.degrees(float(angle)):+7.2f}°  ships={int(ships)}')
        except Exception:
            _w(f'   {m}')


def log_fleets(fleets, player):
    if not enabled() or not _LOG_FLEETS:
        return
    if not fleets:
        _w('[FLEETS]  none flying')
        return
    _w(f'[FLEETS]  N={len(fleets)}')
    _w('   fid own from_pid    pos              angle    ships')
    for f in fleets:
        own = '\u2606' if f.owner == player else '\u00d7'
        _w(f'   {f.id:>3}  {own}   {f.from_pid:>3}     '
           f'({f.x:>5.1f},{f.y:>5.1f})   {math.degrees(f.angle):+7.2f}°   {f.ships}')


def log_remaining(remaining, zone_lookup=None):
    """Остаток кораблей на каждой нашей планете после аукциона.

    Помогает найти idle-деньги: isolated/rear планеты с сотнями кораблей
    которые не перебрасываются на фронт из-за высокого transfer_floor.
    """
    if not enabled() or not remaining:
        return
    # сортируем по убыванию остатка
    items = sorted(remaining.items(), key=lambda x: -x[1])
    parts = []
    for pid, ships in items:
        if ships > 0:
            zone = (zone_lookup or {}).get(int(pid), '?')
            parts.append(f'p{pid}={int(ships)}({zone})')
    if parts:
        _w('[REMAINING]  ' + '  '.join(parts))


def log_transfers(plans, zone_lookup=None):
    """Детальный лог transfer-планов: откуда, куда, сколько, какой градиент.

    Без этого лога видно только 'transfers=3' в [SWARM] — непонятно
    кто кому что отправил и почему.
    """
    if not enabled():
        return
    transfers = [p for p in (plans or []) if p.get('is_transfer')]
    if not transfers:
        return
    _w(f'[TRANSFERS]  N={len(transfers)}')
    for p in transfers:
        src  = p.get('att_id', '?')
        dst  = p.get('tgt_id', '?')
        ships = int(p.get('x_att', 0))
        eta  = p.get('eta_at', 0)
        score = p.get('transfer_score', 0)
        grad  = p.get('transfer_gradient', 0)
        z_src = (zone_lookup or {}).get(int(src), '?') if src != '?' else '?'
        z_dst = (zone_lookup or {}).get(int(dst), '?') if dst != '?' else '?'
        _w(f'   p{src}({z_src}) → p{dst}({z_dst})'
           f'  ships={ships}  eta={eta:.1f}  score={score:.2f}  grad={grad:+.2f}')


def log_zone_flips(prev_zones, curr_zones):
    """Логирует планеты сменившие зону с прошлого хода.

    Помогает диагностировать zone thrashing — когда планеты скачут
    между зонами каждые 2-3 хода (мешает стабильному таргетингу).
    """
    if not enabled() or not prev_zones or not curr_zones:
        return
    flips = []
    for pid, new_zone in curr_zones.items():
        old_zone = prev_zones.get(pid) or prev_zones.get(str(pid))
        if old_zone is not None and old_zone != new_zone:
            flips.append((int(pid), old_zone, new_zone))
    if flips:
        parts = '  '.join(f'p{pid}:{old}→{new}' for pid, old, new in sorted(flips))
        _w(f'[ZONE_FLIP]  N={len(flips)}  {parts}')


def log_bayes_confusion(entropy, max_entropy=1.6094):
    """Метрика непонимания противника: confusion% = entropy / max_entropy * 100.

    max_entropy = ln(5) ≈ 1.6094 для 5 пресетов — полная неопределённость.
    confusion=100% → модель ничего не знает (равномерное распределение).
    confusion=0%   → модель уверена в одном пресете.

    Логируется отдельной строкой рядом с BAYES/predict для быстрого grep.
    """
    if not enabled():
        return
    confusion = min(100.0, 100.0 * float(entropy) / max_entropy)
    bar_len = 20
    filled = int(confusion / 100.0 * bar_len)
    bar = '█' * filled + '░' * (bar_len - filled)
    _w(f'[BAYES/confusion]  {confusion:5.1f}%  [{bar}]  entropy={entropy:.3f}')


def log_error(where, exc):
    if not enabled():
        return
    _w(f'[ERROR]  in {where}: {type(exc).__name__}: {exc}')


def end_turn():
    if not enabled():
        return
    reset_decisions()
    try:
        _LOG_FH.flush()
    except Exception:
        pass


def plans_all_enabled():
    _resolve()
    return _LOG_PLANS_ALL


__all__ = [
    'enabled', 'reset', 'begin_turn', 'log_zones', 'log_targets', 'log_plans',
    'log_decision', 'log_moves', 'log_fleets', 'log_error', 'end_turn',
    'plans_all_enabled',
    'log_remaining', 'log_transfers', 'log_zone_flips', 'log_bayes_confusion',
]

# ╔══════════════════════════════════════════════════╗
# ║  projection.py                                 ║
# ╚══════════════════════════════════════════════════╝

"""
projection.py — проекция состояния планет вперёд во времени.

Идея: zones, force-кривые и attacks должны видеть планеты не в момент «сейчас»,
а в момент последнего известного события про них (прилёт уже летящего флота,
своего или чужого).

Этап:
  1. Для каждого флота в state.fleets — определяем куда летит и когда прилетит
     (через покадровую симуляцию траектории).
  2. По каждой планете в хронологическом порядке резолвим прибытия:
     - своё подкрепление: ships += fleet.ships
     - чужой удар: бой с учётом производства между событиями
  3. Возвращаем ProjectedState с .planets (спроецированные) и .raw_planets (исходные).

Главный экспорт:
  project_state(state, horizon=200) -> ProjectedState
  simulate_fleet_target(fleet, planets, omega) -> (planet_id|None, turns)
"""

import math
from collections import defaultdict



PROJECTION_HORIZON = 200
NEUTRAL_OWNER      = -1

# Сколько кораблей оставить НА КОМЕТЕ после захвата нашим флотом.
# Остаток (ships - COMET_KEEP_ON_BOARD) проецируется как «отскок» на
# ближайшую нашу планету (комета временна — держать там флот бессмысленно).
COMET_KEEP_ON_BOARD = 1


# ── 1. Симуляция траектории флота ──────────────────────────────────────

def simulate_fleet_target(fleet, planets, omega, max_turns=PROJECTION_HORIZON):
    """
    Прогоняет флот по правилам движка. Возвращает (planet_id, turns_to_hit)
    или (None, turns) если флот ушёл за поле / в солнце / не успел.

    Игнорируем коллизии с from_planet_id, пока флот ещё рядом с ним —
    иначе симуляция отчитается о столкновении с собственным источником.
    """
    spd = fleet_speed_correct(max(1, int(fleet.ships)))
    dx, dy = math.cos(fleet.angle), math.sin(fleet.angle)
    fx, fy = float(fleet.x), float(fleet.y)
    src_id = getattr(fleet, 'from_planet_id', None)

    # Узнаём радиус источника — нужно чтобы понять «когда мы вышли из его зоны»
    src_radius = 0.0
    if src_id is not None:
        for p in planets:
            if p.id == src_id:
                src_radius = float(p.radius) + 1.5  # буфер
                break

    for turn in range(1, max_turns + 1):
        nx, ny = fx + spd * dx, fy + spd * dy
        if nx < 0 or nx > BOARD or ny < 0 or ny > BOARD:
            return None, turn
        if point_to_segment_dist(CX, CY, fx, fy, nx, ny) < SUN_R:
            return None, turn
        best_pid, best_d = None, float("inf")
        for p in planets:
            # Не считаем столкновение с источником пока флот к нему ближе чем его радиус+буфер
            if p.id == src_id:
                px0, py0 = predict_planet_xy(p.x, p.y, p.radius, omega, turn)
                if math.hypot(fx - px0, fy - py0) < src_radius:
                    continue
            px, py = predict_planet_xy(p.x, p.y, p.radius, omega, turn)
            d = point_to_segment_dist(px, py, fx, fy, nx, ny)
            if d < p.radius and d < best_d:
                best_d, best_pid = d, p.id
        if best_pid is not None:
            return best_pid, turn
        fx, fy = nx, ny
    return None, max_turns


# ── 2. Резолюция боя на одной планете ──────────────────────────────────

def _resolve_arrival(cur_owner, cur_ships, cur_prod, dt, ev_owner, ev_ships):
    """
    Применяет производство за `dt` ходов и прибытие флота.
    Семантика — ровно как в движке (orbit_sim._resolve_combat):
      • тот же владелец — суммируем корабли;
      • разный — strict-inequality бой:
          ev > cur  → захват с ev - cur;
          ev < cur  → защитник остаётся с cur - ev;
          ev = cur  → НИЧЬЯ: оба обнуляются. Нейтральная цель остаётся
                      нейтральной с 0 кораблей (НЕ переходит атакёру).
    Возвращает (new_owner, new_ships).
    """
    if cur_owner != NEUTRAL_OWNER:
        cur_ships += cur_prod * dt

    if ev_owner == cur_owner:
        cur_ships += ev_ships
    elif ev_ships > cur_ships:
        cur_owner = ev_owner
        cur_ships = ev_ships - cur_ships
    elif ev_ships < cur_ships:
        cur_ships -= ev_ships
    else:
        # ничья: обе стороны уничтожены, владелец цели не меняется
        # (для нейтральной — остаётся нейтральной с 0; для вражеской — её
        # с 0; для своей — этой ветки не достигаем, т.к. ev_owner==cur_owner
        # обработана выше)
        cur_ships = 0
    return cur_owner, max(0.0, cur_ships)


# ── 3. ProjectedState — обёртка вокруг GameState ───────────────────────

class ProjectedState:
    """
    Имеет тот же интерфейс, что и GameState (для zones / force / attacks).
      .planets        — спроецированные планеты (на момент projected_at[id])
      .raw_planets    — исходные (для запусков из агента)
      .projected_at   — dict pid → turn (0 если событий нет)
      .events_per     — dict pid → [(turn, owner, ships)]
      .fleets         — пустой список (флоты резолвены)
      .raw_fleets     — исходные летящие флоты
      .omega, .step   — passthrough
      .comet_ids      — passthrough (нужен zones для фильтрации, projection
                        для bounce-логики). КРИТИЧНО: без этого пробрасывания
                        фильтр комет в zones становится no-op (ProjectedState
                        попадает на вход zones, а у него нет атрибута →
                        getattr возвращает пустое множество → кометы всё равно
                        попадают в df и выбираются как target).
      .initial_by_id() — passthrough (нужен shooting/force)
    """

    def __init__(self, planets, projected_at, events_per, base_state):
        self.planets        = planets
        self.raw_planets    = list(base_state.planets)
        self.projected_at   = projected_at
        self.events_per     = events_per
        self.raw_fleets     = list(base_state.fleets)
        self.fleets         = []
        self.omega          = base_state.omega
        self.step           = base_state.step
        self.n_players      = getattr(base_state, 'n_players', 2)
        self._initial_planets = base_state._initial_planets
        # КОМЕТЫ — обязательно пробрасываем; иначе zones не отфильтрует, а
        # bounce-логика проекции потеряет ориентир «эта планета — комета».
        self.comet_ids      = set(getattr(base_state, 'comet_ids', set()) or set())
        self.comets         = list(getattr(base_state, 'comets', []) or [])
        self._base = base_state

    # GameState-совместимые методы
    def initial_by_id(self):
        return self._initial_planets

    def planets_of(self, player):
        return [p for p in self.planets if p.owner == player]

    def neutral_planets(self):
        return [p for p in self.planets if p.owner == NEUTRAL_OWNER]

    def enemy_planets(self, player):
        return [p for p in self.planets if p.owner not in (NEUTRAL_OWNER, player)]

    def planet_by_id(self, pid):
        for p in self.planets:
            if p.id == pid:
                return p
        return None

    def __repr__(self):
        n_proj = sum(1 for v in self.projected_at.values() if v > 0)
        return (f"ProjectedState(step={self.step}, planets={len(self.planets)}, "
                f"projected={n_proj}, fleets_in={len(self.raw_fleets)})")


# ── 4. project_state — главная функция ────────────────────────────────

def _nearest_own_planet(comet, planets, comet_ids, player):
    """Ближайшая (по straight-line distance) НЕ-кометная планета игрока player.
    None если у нас нет ни одной не-кометной планеты (теоретический edge-case)."""
    own = [q for q in planets
           if q.owner == player and q.id != comet.id and q.id not in comet_ids]
    if not own:
        return None
    return min(own, key=lambda q: math.hypot(comet.x - q.x, comet.y - q.y))


def _bounce_eta(comet, dst, ships):
    """Грубая оценка ETA «отскока» с кометы на нашу планету. Используем
    straight-line/speed — это проекция, точная физика не нужна."""
    spd = fleet_speed_correct(max(1, int(ships)))
    d = max(1.0, math.hypot(comet.x - dst.x, comet.y - dst.y)
                 - comet.radius - dst.radius)
    return int(math.ceil(d / max(spd, 1e-6)))


def _resolve_chronological(p, events):
    """Прокручивает события по планете p в хронологическом порядке.
    Возвращает (owner, ships, last_turn). Если событий нет —
    (p.owner, p.ships, 0)."""
    if not events:
        return p.owner, float(p.ships), 0
    events = sorted(events, key=lambda e: e[0])
    owner = p.owner
    ships = float(p.ships)
    last_turn = 0
    for ev_turn, ev_owner, ev_ships in events:
        dt = ev_turn - last_turn
        owner, ships = _resolve_arrival(
            owner, ships, p.production, dt, ev_owner, ev_ships
        )
        last_turn = ev_turn
    return owner, ships, last_turn


def project_state(state, horizon=PROJECTION_HORIZON, player=0):
    """
    Возвращает ProjectedState — каждая планета на момент последнего события.

    КОМЕТЫ (`state.comet_ids`):
      Если наш флот «зацепил» комету (после боя owner == player и ships > 1),
      то все корабли кроме COMET_KEEP_ON_BOARD проецируются как отскок —
      виртуальное событие «прилёт нашего флота» на ближайшую нашу планету.
      Это отражает стратегию «комета — трамплин»: держать флот на временной
      планете бессмысленно, он должен возвращаться домой.
    """
    planets = list(state.planets)
    omega   = state.omega
    comet_ids = set(getattr(state, 'comet_ids', set()) or set())

    # Шаг 1: определяем цель и ETA каждого флота
    events_per = defaultdict(list)
    for f in state.fleets:
        pid, turn = simulate_fleet_target(f, planets, omega, max_turns=horizon)
        if pid is None:
            continue
        events_per[pid].append((int(turn), int(f.owner), int(f.ships)))

    # Шаг 2: для комет резолвим первыми и эмитим bounce-события на наши планеты
    bounce_events_per = defaultdict(list)  # pid → [(turn, owner, ships)]
    if comet_ids:
        for p in planets:
            if p.id not in comet_ids:
                continue
            owner, ships, last_turn = _resolve_chronological(
                p, events_per.get(p.id, [])
            )
            # Только если МЫ владеем кометой и есть что отправлять домой
            if owner == player and ships > COMET_KEEP_ON_BOARD:
                dst = _nearest_own_planet(p, planets, comet_ids, player)
                if dst is not None:
                    bounce_ships = int(ships) - COMET_KEEP_ON_BOARD
                    eta = _bounce_eta(p, dst, bounce_ships)
                    arrival = last_turn + eta
                    if arrival <= horizon:
                        bounce_events_per[dst.id].append(
                            (int(arrival), int(player), int(bounce_ships))
                        )

    # Подмешиваем bounce-события к реальным
    for pid, evs in bounce_events_per.items():
        events_per[pid].extend(evs)

    # Шаг 3: по каждой планете прокручиваем события в порядке прилёта.
    # Для комет, у которых сработал отскок, корабли (ships - KEEP) уже
    # унесены отдельным bounce-событием на ближайшую нашу планету; на
    # самой комете оставляем только COMET_KEEP_ON_BOARD.
    projected_planets = []
    projected_at = {}
    for p in planets:
        events = events_per.get(p.id, [])
        if not events and p.id not in comet_ids:
            projected_planets.append(p)
            projected_at[p.id] = 0
            continue

        owner, ships, last_turn = _resolve_chronological(p, events)

        # Для кометы: если мы захватили — оставляем COMET_KEEP_ON_BOARD,
        # остальное уже улетело bounce-событием на нашу планету.
        if p.id in comet_ids and owner == player and ships > COMET_KEEP_ON_BOARD:
            ships = float(COMET_KEEP_ON_BOARD)

        if not events and p.id in comet_ids:
            # комета без событий — оставляем как есть
            projected_planets.append(p)
            projected_at[p.id] = 0
            continue

        projected_planets.append(Planet(
            id=p.id,
            owner=owner,
            x=p.x, y=p.y,
            radius=p.radius,
            ships=int(round(ships)),
            production=p.production,
        ))
        projected_at[p.id] = last_turn

    return ProjectedState(projected_planets, dict(projected_at),
                          dict(events_per), state)


__all__ = [
    'PROJECTION_HORIZON', 'NEUTRAL_OWNER',
    'simulate_fleet_target', 'project_state', 'ProjectedState',
]

# ╔══════════════════════════════════════════════════╗
# ║  attacks.py                                    ║
# ╚══════════════════════════════════════════════════╝

"""
Блок оценки атак: direct / multi_sync / pipeline.

Главная идея: считаем МИНИМУМ кораблей нужный для победы (с учётом
производства защитника и уже летящих к цели дружественных флотов).
НЕ отправляем всё подряд.

Ключевые экспорты:
  eval_direct(state, att, tgt, incoming=0)            -> dict | None
  eval_multi_sync(state, sup, att, tgt, incoming=0)   -> dict | None
  eval_pipe(state, sup, att, tgt, incoming=0)         -> dict | None
  friendly_incoming(state, tgt, t_max, player)        -> (ships, etas)
  all_plans(state, tgt, ours, horizon, player)        -> list[dict]
  best_attacks(state, player, targets)                -> list[dict]
"""

import math
from itertools import product as _product


ATTACK_HORIZON         = 80
SUN_SAFETY_PIPE        = 1.5
PIPELINE_MAX_ANGLE_DEG = 120.0
NEUTRAL_OWNER          = -1     # движок не производит у нейтралов


def _effective_prod(tgt):
    """Производство, КОТОРОЕ РЕАЛЬНО ПРИБАВЛЯЕТСЯ во время полёта.
    Нейтральные планеты в движке (orbit_sim.simulate_step, шаг 2) НЕ
    производят корабли — production стартует только когда планета захвачена.
    Без этой проверки `defender` для нейтралов завышается на prod·eta и
    `best_attacks` ошибочно помечает план как проигрышный, хотя реальная
    оборона = tgt.ships."""
    if tgt.owner == NEUTRAL_OWNER:
        return 0.0
    return float(tgt.production)


def _overkill(tgt, risk=0):
    """Размер буфера сверх defender. Для нейтралов производства нет ⇒ дрейф
    eta не страшен, хватает +1 (от ничьей). Для owned — +2.

    risk (0..2) — снижает буфер. ВАЖНО: минимум 1, потому что движок
    использует strict-< при бое (атака побеждает только если ships > defender,
    ничья = поражение). С overkill=0 атомный исполнитель отбрасывает план
    как «wont_win: ships == defender_at_eta».

    risk=1 убирает «дрейф eta» запас (только у owned: 2 → 1, у нейтрала
    остаётся 1). risk=2 = синоним risk=1, ниже опуститься нельзя.
    """
    base = SAFETY_OVERKILL_NEUTRAL if tgt.owner == NEUTRAL_OWNER else SAFETY_OVERKILL
    return max(1, base - int(risk))

SAFETY_OVERKILL        = 2     # сколько кораблей сверх defender (для OWNED цели).
                               # =2 покрывает: 1 на ничью (движок strict-<)
                               # + 1 на дрейф eta между планированием
                               # (_rendezvous_eta) и реальной стрельбой
                               # (aim_hybrid).
SAFETY_OVERKILL_NEUTRAL = 1    # для НЕЙТРАЛОВ хватает +1: нет производства,
                               # значит нет дрейфа defender'а — нужен только
                               # буфер на ничью.
MIN_USEFUL_STRIKE      = 3     # ниже — атака не имеет смысла, пропускаем
ETA_REFINE_ITERS       = 2     # сколько раз пересчитать eta при изменении кораблей
INCOMING_ANGLE_TOL     = 0.18  # рад (~10°): флот целится в цель если так
RESERVE_ON_ATT         = 0     # минимум кораблей оставить на атакере.
                               # ОБЯЗАН совпадать с agent.RESERVE_ON_ATT.
                               # Тай на нейтрале предотвращается через
                               # SAFETY_OVERKILL_NEUTRAL=1 (нужно строго больше
                               # defender), а не через резервирование корабля.

# ── Порог "одновременности" для multi_sync ────────────────────────────────
# `agent._plan_parts` эмитит ОБА флота СРАЗУ в один ход → быстрый прилетит
# раньше медленного на |Δeta| ходов. Раз мы вообще полезли в multi_sync,
# значит att одного НЕ хватает (см. ранний return) — следовательно быстрый
# флот, прилетев один, ПРОИГРАЕТ defender'у. Defender за это время
# дорегенерирует prod·Δeta кораблей, и медленному тоже не хватит.
#
# При маленьком Δeta (<= MAX_MULTI_SYNC_DELTA_ETA) defender не успевает
# восстановиться до критичного значения, и пара "почти одновременно"
# работает как один комбинированный удар — НЕ копить два независимых
# действия в очередь, а сразу собрать их в одно.
#
# При большом Δeta выгоднее НЕ делать multi_sync вовсе: пусть планировщик
# выберет direct (от ближайшего) сейчас, а второй источник либо переразметит
# приоритет в следующем ходу, либо станет supply'ем для другой цели.
MAX_MULTI_SYNC_DELTA_ETA = 5   # ходов; одновременность с допуском.
                               # Калибровка: даже equidistant att/sup дают
                               # Δeta до 5-9 ходов из-за орбитального lead-angle
                               # (rendezvous-eta зависит от стороны подлёта к
                               # вращающейся цели). 5 ходов — компромисс между
                               # "ловить только реально несинхронные пары" и
                               # "не резать симметричные геометрии".


# ── Утилиты ────────────────────────────────────────────────────────────

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


def _plan_total_ships(plan):
    return int(plan.get('x_att', 0)) + int(plan.get('x_sup', 0))


def _mean_d_to_non_ours(p, all_planets, player):
    """Среднее расстояние от планеты `p` до всех ЧУЖИХ/НЕЙТРАЛЬНЫХ планет.
    Используется как простая мера «насколько планета на фронте»: чем меньше
    среднее расстояние — тем ближе к врагу/нейтралам.

    Если других планет нет — возвращаем +inf (планета формально «дальше всех»)."""
    others = [q for q in all_planets if q.id != p.id and q.owner != player]
    if not others:
        return float('inf')
    return sum(math.hypot(p.x - q.x, p.y - q.y) for q in others) / len(others)


def _is_more_frontline(att, sup, all_planets, player):
    """True если att ближе к не-нашим планетам чем sup → имеет смысл
    перебросить ВСЁ что есть с sup на att (att — лучшая стартовая площадка
    для следующих атак)."""
    return _mean_d_to_non_ours(att, all_planets, player) \
         < _mean_d_to_non_ours(sup, all_planets, player)


def _t_solo_capture(state, att, tgt, incoming=0.0, risk=0):
    """
    Оценка: за СКОЛЬКО ходов att захватит tgt самостоятельно (без supply).
    Считает: сколько ходов копить производство → fly direct.

    Для owned-tgt (растёт оборона) и att одновременно копится — берём
    сходящуюся оценку через несколько итераций. Для нейтрала defender
    статичен, поэтому проще.

    Возвращает (t_total, n_needed). Если att вообще не сможет (например
    маршрут заблокирован солнцем) → (math.inf, ∞).
    """
    if _seg_blocked(att, tgt):
        return math.inf, math.inf

    omega = state.omega
    proj_at = _projected_at(state, tgt.id)
    eff_prod_def = _effective_prod(tgt)
    overkill = _overkill(tgt, risk=risk)
    avail_now = max(0.0, att.ships - RESERVE_ON_ATT)

    # Итеративно сходимся: на ход t_wait att имеет (avail_now + prod*t_wait),
    # пуляет, летит eta(att, tgt, ships, omega) → arrive at t_wait+eta.
    # defender = tgt.ships + eff_prod_def * max(0, t_arrive - proj_at).
    # Нужно: ships ≥ defender + overkill.
    # Решаем перебором t_wait от 0 до horizon.
    best_total = math.inf
    best_needed = math.inf
    for t_wait in range(0, ATTACK_HORIZON + 1):
        ships = avail_now + att.production * t_wait
        if ships < 1:
            continue
        eta = _eta(att, tgt, max(1, int(ships)), omega)
        t_arrive = t_wait + eta
        # ранний перехват по raw, иначе projected
        defender = _defender_at_eta(state, tgt, t_arrive)
        needed = math.ceil(defender - incoming) + overkill
        if ships >= needed:
            best_total = t_arrive
            best_needed = needed
            break
    return best_total, best_needed


# ── Учёт уже летящих дружественных флотов к цели ───────────────────────

def friendly_incoming(state, tgt, t_max, player):
    """
    Сумма кораблей дружественных флотов которые ЦЕЛЯТСЯ В tgt и долетят
    в пределах t_max. Использует угловой критерий — быстро и достаточно.
    Возвращает (total_ships, [eta_per_fleet]).
    """
    total = 0.0
    etas  = []
    for f in state.fleets:
        if f.owner != player:
            continue
        # вектор от флота к цели и к его текущему направлению
        ax = math.atan2(tgt.y - f.y, tgt.x - f.x)
        diff = abs(((ax - f.angle + math.pi) % (2 * math.pi)) - math.pi)
        if diff > INCOMING_ANGLE_TOL:
            continue
        d = math.hypot(tgt.x - f.x, tgt.y - f.y)
        spd = fleet_speed_correct(max(1, int(f.ships)))
        eta = d / max(spd, 1e-6)
        if eta > t_max:
            continue
        total += f.ships
        etas.append(eta)
    return total, etas


# ── Расчёт минимума необходимых кораблей с уточнением eta ──────────────

def _projected_at(state, pid):
    """Возвращает projected_at[pid] из ProjectedState, или 0 если state не проецирован."""
    return getattr(state, 'projected_at', {}).get(pid, 0)


def _raw_planet(state, pid):
    """Поиск raw-планеты по id (если state — ProjectedState).
    Если state без проекции — возвращаем None (caller'у нужно использовать tgt напрямую)."""
    raw = getattr(state, 'raw_planets', None)
    if raw is None:
        return None
    for p in raw:
        if p.id == pid:
            return p
    return None


def _defender_at_eta(state, tgt, eta):
    """Defender, которого МЫ ВСТРЕТИМ при прилёте через `eta` ходов.

    Ключевая тонкость для нейтралов под чужой атакой:
      - tgt = ProjectedState.planets[id]  ⇒  это планета *после флипа* (уже
        враг по проекции).
      - projected_at[id]                  ⇒  ход когда происходит флип.

    Если eta < projected_at[id] — мы прилетим **до того как противник захватил**.
    В этот момент планета ещё RAW (нейтрал/наш предыдущий владелец) и
    защитник = raw.ships без накопления (нейтрал не производит).
    Нет смысла планировать огромный strike против будущего врага, если
    можно прилететь раньше за дёшево.

    Если eta ≥ projected_at — стандартный путь: `tgt.ships` уже учитывает
    результат боя (post-flip), производство добавляется только за время
    после флипа.
    """
    proj_at = _projected_at(state, tgt.id)

    if proj_at > 0 and eta < proj_at:
        raw = _raw_planet(state, tgt.id)
        if raw is not None:
            if raw.owner == NEUTRAL_OWNER:
                # нейтрал не производит — defender = raw.ships, статичен
                return float(raw.ships)
            # owned (наш или вражеский): production до момента нашего прилёта
            return float(raw.ships) + float(raw.production) * eta

    # стандарт: tgt — спроецированный, prod считаем от proj_at
    return tgt.ships + _effective_prod(tgt) * max(0.0, eta - proj_at)


def _required_strike(state, att, tgt, omega, init_eta_ships, incoming, risk=0):
    """
    Считает (eta, defender, needed) для атаки att→tgt.
    Если state — ProjectedState, tgt.ships уже учитывает прибытия до projected_at,
    поэтому defender = tgt.ships + production * max(0, eta - projected_at).
    `incoming` оставлен как параметр совместимости — обычно 0 при проекции.
    """
    eta = _eta(att, tgt, init_eta_ships, omega)
    needed = 0
    overkill = _overkill(tgt, risk=risk)
    for _ in range(ETA_REFINE_ITERS + 1):
        # `_defender_at_eta` сам решает: ранний перехват по raw (eta<proj_at)
        # или стандартный путь по projected с производством после флипа.
        defender = _defender_at_eta(state, tgt, eta)
        effective = max(0.0, defender - incoming)
        needed = int(math.ceil(effective)) + overkill
        new_eta = _eta(att, tgt, max(1, needed), omega)
        if abs(new_eta - eta) < 0.5:
            eta = new_eta
            break
        eta = new_eta
    defender = _defender_at_eta(state, tgt, eta)
    return eta, defender, needed


# ── eval_direct ────────────────────────────────────────────────────────

def eval_direct(state, att, tgt, incoming=0.0, risk=0, neutral_garrison=0):
    """Прямая атака att → tgt.

    x_att = минимум для победы (с учётом incoming).
    Для НЕЙТРАЛОВ: если affordable — добавляем neutral_garrison кораблей
    сверх минимума. Флот прилетит быстрее (Orbit Wars: скорость растёт с
    числом кораблей) и планета сразу получит гарнизон без отдельного трансфера.
    success определяется по минимуму (без garrison) — захватываем всегда если
    можем, garrison = приятный бонус сверху.
    """
    if _seg_blocked(att, tgt):
        return None

    eta, defender, needed = _required_strike(state, att, tgt, state.omega, att.ships, incoming, risk=risk)
    available = att.ships - RESERVE_ON_ATT
    success   = needed <= available

    if success:
        if tgt.owner == NEUTRAL_OWNER and neutral_garrison > 0:
            # Добавляем гарнизон сверх минимума, но не больше доступных кораблей
            x_att = min(available, needed + neutral_garrison)
        else:
            x_att = needed
    else:
        x_att = max(1, available)

    return {
        'mode':      'direct',
        'sup_id':    None, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_sa':    0.0,  'eta_at': eta,    't_total': eta,
        'x_sup':     0,    'x_att':  int(x_att),  'prod_att': att.production,
        'x_tgt':     tgt.ships, 'prod_tgt': tgt.production,
        'strike':    float(x_att) + incoming,
        'defender':  defender,
        'incoming':  incoming,
        'needed':    needed,
        'available': available,
        'slack':     available - needed if success else 0,
        'margin':    float(x_att) + incoming - defender,
        'success':   success,
        'relay_reason': None,
    }


# ── eval_multi_sync ────────────────────────────────────────────────────

def eval_multi_sync(state, sup, att, tgt, incoming=0.0, risk=0):
    """
    Синхронная атака с двух источников. Используется когда att одного НЕ ХВАТАЕТ.
    att шлёт всё → sup добивает минимум.
    """
    if _seg_blocked(sup, tgt) or _seg_blocked(att, tgt):
        return None
    omega = state.omega

    # eta (с полными кораблями для скорости — потом уточним)
    eta_st = _eta(sup, tgt, sup.ships, omega)
    eta_at = _eta(att, tgt, att.ships, omega)
    t_sync = max(eta_st, eta_at)

    # ── Порог "одновременности" ─────────────────────────────────────────────
    # Если sup и att прилетают с большим разрывом, multi_sync теряет смысл:
    # быстрый флот прибывает в одиночку, проигрывает defender'у (att одного
    # по определению не хватает — иначе мы бы выбрали direct), defender
    # регенерирует prod·Δeta кораблей за время полёта медленного, и второй
    # удар тоже сливается. См. MAX_MULTI_SYNC_DELTA_ETA выше.
    if abs(eta_st - eta_at) > MAX_MULTI_SYNC_DELTA_ETA:
        return None

    # ВНИМАНИЕ: agent._plan_parts эмитит оба флота СРАЗУ в этом ходу — никакого
    # «подождать на источнике, пока другой флот долетит, и пока копится
    # производство» в один ход не реализовано. Поэтому в реальный strike
    # попадает только то, что РЕАЛЬНО отправляется сейчас (current ships - reserve).
    # Фантомное `production * (t_sync - eta_X)` убрано — оно искажало success
    # на множители 5-15 ходов производства.
    att_part_max = max(0.0, att.ships - RESERVE_ON_ATT)
    sup_part_max = max(0.0, sup.ships - RESERVE_ON_ATT)
    if att_part_max <= 0 or sup_part_max <= 0:
        return None

    # defender — учитывает ранний перехват (eta<projected_at → raw)
    defender  = _defender_at_eta(state, tgt, t_sync)
    effective = max(0.0, defender - incoming)
    needed    = int(math.ceil(effective)) + _overkill(tgt, risk=risk)

    # если att одного хватает → это direct, тут не место
    if att_part_max >= needed:
        return None

    sup_part_needed = needed - att_part_max
    success         = sup_part_needed <= sup_part_max

    if success:
        sup_part = sup_part_needed
    else:
        sup_part = sup_part_max
    strike = att_part_max + sup_part + incoming

    # x_sup_send = ровно столько, сколько надо отправить (без фейкового
    # «вычета производства» — мы стартуем сейчас, ничего не накапливаем).
    x_sup_send = max(1.0, sup_part)
    x_att_send = max(1, int(att_part_max))

    # ── ПРОВЕРКА: supply должен УСКОРЯТЬ захват ─────────────────────────────
    # Если att без всякого supply справится сам не позже t_sync — supply
    # ничего не даёт (а только тратит корабли с sup). Отвергаем такой план.
    # Допуск 1 ход — учитывает округления и тики.
    t_solo, _ = _t_solo_capture(state, att, tgt, incoming=incoming)
    if t_solo <= t_sync + 1:
        return None

    return {
        'mode':     'multi_sync',
        'sup_id':   sup.id, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_st':   eta_st, 'eta_at': eta_at, 't_total': t_sync,
        'x_sup':    int(math.ceil(x_sup_send)),
        'x_att':    int(x_att_send),
        'prod_sup': sup.production, 'prod_att': att.production,
        'x_tgt':    tgt.ships, 'prod_tgt': tgt.production,
        'strike':   strike,
        'defender': defender,
        'incoming': incoming,
        'needed':   needed,
        'margin':   strike - defender,
        'success':  success,
        'sup_part': sup_part, 'att_part': att_part_max,
        'relay_reason': None,
    }


# ── eval_pipe ──────────────────────────────────────────────────────────

def eval_pipe(state, sup, att, tgt, incoming=0.0, risk=0):
    """Pipeline: sup → att → tgt. sup отправляет минимум для добивки."""
    if _seg_blocked(sup, att) or _seg_blocked(att, tgt):
        return None

    sup_dir_blocked = _seg_blocked(sup, tgt)
    angle_deg = _angle_at_att(sup, att, tgt)
    if not sup_dir_blocked and angle_deg > PIPELINE_MAX_ANGLE_DEG:
        return None

    # Forward-rebase: att «фронтовее» sup (ближе к не-нашим в среднем) ⇒
    # имеет смысл перебрасывать ВСЁ что есть с sup на att (att — лучшая
    # стартовая площадка для следующих атак).
    forward_rebase = _is_more_frontline(att, sup, state.planets, att.owner)

    # Pipeline валиден только если supply несёт самостоятельную ценность:
    #   1) sup→tgt заблокирован солнцем (relay через att — единственный путь), ИЛИ
    #   2) att «фронтовее» sup (forward-rebase оправдан стратегически).
    # Иначе direct sup→tgt доминирует, pipeline = пустая трата кораблей.
    if not sup_dir_blocked and not forward_rebase:
        return None

    omega   = state.omega
    eta_sa  = _eta(sup, att, sup.ships, omega)
    boosted_max = sup.ships + att.ships + att.production * eta_sa
    eta_at  = _eta(att, tgt, boosted_max, omega)
    t_total = eta_sa + eta_at

    # defender — учитывает ранний перехват (eta<projected_at → raw)
    defender    = _defender_at_eta(state, tgt, t_total)
    effective   = max(0.0, defender - incoming)
    needed      = int(math.ceil(effective)) + _overkill(tgt)

    # минимум кораблей с sup, чтобы att.ships + production_eta + x_sup ≥ needed
    att_after_eta = att.ships + att.production * eta_sa
    x_sup_min     = max(1.0, needed - att_after_eta)
    success_relay = x_sup_min <= (sup.ships - RESERVE_ON_ATT)

    if success_relay:
        x_sup_send = x_sup_min
        boosted    = att_after_eta + x_sup_send
    else:
        x_sup_send = max(1, sup.ships - RESERVE_ON_ATT)
        boosted    = att_after_eta + x_sup_send

    # ── Forward-rebase: если att «фронтовее» sup (ближе к не-нашим),
    # имеет смысл перебросить ВСЁ что есть с sup на att. Тогда att
    # становится новой стартовой площадкой для следующих атак, а sup
    # больше не тащит лишние корабли в тылу.
    # `forward_rebase` уже вычислен выше в проверке валидности pipeline.
    sup_avail = max(1, sup.ships - RESERVE_ON_ATT)
    if forward_rebase and sup_avail > x_sup_send:
        x_sup_send = sup_avail

    # x_att = минимум для победы (а не всё что есть)
    x_att_send = min(att.ships - RESERVE_ON_ATT, needed)
    x_att_send = max(1, int(x_att_send))

    # ── Реальный исход ─────────────────────────────────────────────────────
    # ВНИМАНИЕ: pipeline в текущей реализации (см. agent._plan_parts) —
    # это «att стреляет СЕЙЧАС со своими x_att, sup посылает x_sup на att для
    # БУДУЩЕГО». Никакого «att накопит eta_sa ходов и потом выстрелит» в
    # одном ходу не происходит. Поэтому реальная атака на tgt — это только
    # x_att кораблей (sup прилетит на att, а не на tgt).
    #
    # `boosted` (включает att.production·eta_sa и x_sup) — это ИДЕАЛИЗИРОВАННЫЙ
    # сценарий, который НЕ реализуется в один ход. Он используется только
    # для эвристики «имеет ли смысл когда-нибудь так делать», но критерий
    # success ОБЯЗАН опираться на реально летящие x_att vs defender.
    real_strike = x_att_send + incoming
    real_margin = real_strike - defender
    success_real = real_strike >= needed   # x_att сам по себе должен брать
    success     = success_real and success_relay
    margin_real = real_margin
    # старый «boosted» оставляем для отладки
    margin_boosted = boosted + incoming - defender

    # Pipeline без реальной победы att→tgt — это просто потерянные x_att кораблей.
    # Не возвращаем такой план: пусть планировщик выберет direct/multi_sync.
    if not success_real:
        return None

    if sup_dir_blocked:
        relay_reason = f'sup→tgt blocked by sun (angle={angle_deg:.0f}°)'
    else:
        ms = eval_multi_sync(state, sup, att, tgt, incoming=incoming, risk=risk)
        if ms is None:
            relay_reason = f'multi_sync infeasible (angle={angle_deg:.0f}°)'
        elif ms['success'] and not success_relay:
            return None
        elif success_relay and ms['success']:
            ms_total    = _plan_total_ships(ms)
            relay_total = int(math.ceil(x_sup_send)) + x_att_send
            if ms_total <= relay_total:
                return None
            relay_reason = f'relay uses fewer ships ({relay_total} vs {ms_total})'
        elif margin_boosted > 0:
            relay_reason = f'relay succeeds where multi_sync fails (angle={angle_deg:.0f}°)'
        else:
            return None

    return {
        'mode':        'pipeline',
        'sup_id':      sup.id, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_sa':      eta_sa, 'eta_at': eta_at, 't_total': t_total,
        'x_sup':       int(math.ceil(x_sup_send)),
        'x_att':       x_att_send,
        'prod_att':    att.production,
        'x_tgt':       tgt.ships, 'prod_tgt': tgt.production,
        'strike':      real_strike,
        'defender':    defender,
        'incoming':    incoming,
        'needed':      needed,
        'x_sup_min':   x_sup_min,
        'sup_surplus': sup.ships - x_sup_min,
        'margin':      margin_real,         # реалистичная маржа: x_att vs defender
        'margin_boost': margin_boosted,     # сценарий с накоплением (для отладки)
        'success':     success,
        'angle_deg':   angle_deg,
        'relay_reason': relay_reason,
        'forward_rebase': forward_rebase,   # True ⇒ x_sup = всё что есть у sup
    }


# ── all_plans ──────────────────────────────────────────────────────────

def all_plans(state, tgt, ours, horizon=ATTACK_HORIZON, player=0, risk=0,
              neutral_garrison=0):
    """Все возможные планы атаки на tgt.

    Если state — ProjectedState, incoming уже учтён в tgt.ships, поэтому
    дополнительно его не считаем. Иначе используем friendly_incoming как раньше.
    neutral_garrison — дополнительные корабли сверх минимума при захвате нейтрала
    (передаётся в eval_direct; multi_sync и pipeline не меняем — там арифметика сложнее).
    """
    if hasattr(state, 'projected_at'):
        incoming = 0.0
    else:
        incoming, _ = friendly_incoming(state, tgt, horizon, player)

    plans = []
    for att in ours:
        if att.id == tgt.id:
            continue
        d = eval_direct(state, att, tgt, incoming=incoming, risk=risk,
                        neutral_garrison=neutral_garrison)
        if d and d['t_total'] <= horizon:
            plans.append(d)

    for att, sup in _product(ours, ours):
        if sup.id in (att.id, tgt.id) or att.id == tgt.id:
            continue
        ms = eval_multi_sync(state, sup, att, tgt, incoming=incoming, risk=risk)
        if ms and ms['t_total'] <= horizon:
            plans.append(ms)
        pp = eval_pipe(state, sup, att, tgt, incoming=incoming, risk=risk)
        if pp and pp['t_total'] <= horizon:
            plans.append(pp)
    return plans


# ── best_attacks (главный интерфейс для агента) ────────────────────────

def best_attacks(state, player, targets, top_n=6, horizon=ATTACK_HORIZON):
    """
    Для каждой цели — план с МИНИМУМОМ затраченных кораблей среди успешных.
    Если успешного нет — лучший по margin.

    Атакёры берутся из RAW-состояния (текущие наши планеты), цели — из targets
    (обычно из ProjectedState). Если ProjectedState показывает что цель уже
    станет нашей — её всё равно пропускаем выше (зоны не выберут как цель).
    """
    # Атакёры — реальные текущие наши планеты (не проекция)
    raw_planets = getattr(state, 'raw_planets', state.planets)
    ours = [p for p in raw_planets if p.owner == player]
    if not ours or not targets:
        return []

    is_projected = hasattr(state, 'projected_at')

    best_per_target = []
    for tgt in targets:
        # Если без проекции — старая проверка через friendly_incoming
        if not is_projected:
            incoming, _ = friendly_incoming(state, tgt, horizon, player)
            defender_min = tgt.ships + _effective_prod(tgt) * 1
            if incoming >= defender_min + _overkill(tgt):
                continue

        plans = all_plans(state, tgt, ours, horizon=horizon, player=player)
        if not plans:
            continue

        successful = [pl for pl in plans if pl['success']]
        if successful:
            best = min(successful, key=lambda p: (
                _plan_total_ships(p),
                p['t_total'],
                -p.get('slack', 0),
            ))
        else:
            best = max(plans, key=lambda x: x['margin'])

        if best['success'] and _plan_total_ships(best) < MIN_USEFUL_STRIKE:
            continue
        best_per_target.append(best)

    best_per_target.sort(key=lambda x: (
        -int(x['success']),
        _plan_total_ships(x),
        x['t_total'],
    ))
    return best_per_target[:top_n]


__all__ = [
    'ATTACK_HORIZON', 'SUN_SAFETY_PIPE', 'PIPELINE_MAX_ANGLE_DEG',
    'SAFETY_OVERKILL', 'MIN_USEFUL_STRIKE', 'INCOMING_ANGLE_TOL', 'RESERVE_ON_ATT',
    'MAX_MULTI_SYNC_DELTA_ETA',
    'eval_direct', 'eval_multi_sync', 'eval_pipe',
    'friendly_incoming', 'all_plans', 'best_attacks',
]

# ╔══════════════════════════════════════════════════╗
# ║  zones.py                                      ║
# ╚══════════════════════════════════════════════════╝

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
                  our_total_prod=0.0, prio_reclassify_thr=None):
    """
    Принимает DataFrame с колонками ZONE_FEATURES + 'owner' + 'pid'.
    Возвращает (df_extended, Z_scores).

    `step` — ход матча. Используется как phase-множитель для двух вещей:
      1. late_aggression: эффективный вес = phase · w_targets['late_aggression']
      2. production-scarcity boost (early-game): вес prod у целей усиливается
         обратно пропорционально our_total_prod и линейно гасится к EARLY_PHASE_THR.

    `our_total_prod` — суммарный прод НАШИХ планет на текущий ход. Используется
    для production-scarcity: чем меньше наш прод, тем сильнее буст на rich-цели.

    `prio_reclassify_thr` — порог priority-override post-pass: планеты из
    PRIO_RECLASSIFY_ZONES с priority ≥ thr → 'priority_target'. Если None —
    используется модульная константа PRIO_RECLASSIFY_THR.
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
    # Порог: prio_reclassify_thr (параметр) > PRIO_RECLASSIFY_THR (модульный дефолт).
    _thr = PRIO_RECLASSIFY_THR if prio_reclassify_thr is None else float(prio_reclassify_thr)
    reclassify_mask = (
        is_tgt
        & out['zone'].isin(PRIO_RECLASSIFY_ZONES)
        & (out['priority'] >= _thr)
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
                              our_total_prod=None, prio_reclassify_thr=None):
    """
    Вход: GameState, player id.
    Выход: (df_with_zones, Z_scores) — готово для агента без тяжёлых вычислений.

    `step` — текущий ход матча. Если None — берётся из state.step. Нужен для
    late_aggression-фичи и production-scarcity boost.

    `our_total_prod` — суммарный прод наших планет. Если None — вычисляется
    автоматически из state.planets. Используется для production-scarcity boost
    в ранней игре: усиливает приоритет high-prod целей когда наш прод мал.

    `prio_reclassify_thr` — порог priority-override (periphery/hard_far → priority_target).
    Если None — используется PRIO_RECLASSIFY_THR. agent.py передаёт сюда stage-aware
    значение интерполированное между prio_reclassify_thr и prio_reclassify_thr_late.

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
                         player=player, step=step, our_total_prod=our_total_prod,
                         prio_reclassify_thr=prio_reclassify_thr)


__all__ = [
    'ZONE_FEATURES', 'W_OURS', 'W_TARGETS', 'THR_HI', 'THR_LO', 'ZONE_COLORS',
    'SCARCITY_K', 'EARLY_PHASE_THR',
    'PRIO_RECLASSIFY_THR', 'PRIO_RECLASSIFY_ZONES',
    'APPROACH_LOOKAHEAD', 'APPROACH_SAMPLE', '_approach_dist',
    'compute_zones', 'compute_zones_from_state',
]

# ╔══════════════════════════════════════════════════╗
# ║  swarm.py                                      ║
# ╚══════════════════════════════════════════════════╝

"""
AgentSwarm — пер-планетный планировщик с аукционом ships-бюджета.

Концепция (отличие от best_attacks):

  best_attacks:        для каждой цели → один лучший план. Глобальный priority.
                        Передачи между своими (transfer) не существует как
                        первоклассного действия — supplier'ы существуют только
                        как часть pipeline.

  AgentSwarm:          КАЖДАЯ наша планета имеет локальный ranked-список
                        действий: ATTACK_T (direct/multi/pipe), TRANSFER_Q,
                        HOLD. Действия конкурируют в едином аукционе,
                        ограниченном per-planet ship budget. Сплит возникает
                        естественно: top-action использует cost ships, остаток
                        идёт в auction для другого действия.

Алгоритм по фазам:

  0. Generate candidates:
     all_plans(state, T) для каждого target T → пул direct/multi/pipe планов.
     У каждого плана уже есть `x_att`, `x_sup` = МИНИМУМ ships для победы.

  1. Stress per planet:
     stress[P] = Σ_i top_actions[P][i].margin · γ^i,  i ∈ [0, K)
     где top_actions[P] — отсортированные по margin планы где P участвует.

  2. Auction (жадный single-pass):
     sort candidates по value desc.
     для каждого: проверить что у всех акторов хватит ships в бюджете И
     target не захвачен другим планом → commit, вычесть cost из бюджетов.
     Иначе — отбросить (потенциально неудовлетворённый actor: фиксируем).

  3. Redistribute leftovers:
     для каждой P с remaining > FLOOR:
       найти ally Q с высоким stress, путь чист от солнца, ETA в окне
       → TRANSFER(P → Q, ships=min(remaining, deficit_Q + buffer))

  4. Output:
     список планов (тот же формат что и best_attacks), плюс план типа
     'transfer' для внутренних передач.

План 'transfer' имеет shape:
    {'mode': 'transfer', 'sup_id': None, 'att_id': sender, 'tgt_id': recipient,
     'x_att': ships, 'x_sup': 0, 'success': True, ...}
agent._plan_parts → одна часть (sender → recipient, ships).
agent._execute_plan_atomically → не делает defender check (mode != 'direct'),
просто отправляет флот; _aim_and_verify обработает HIT в нашу планету.

Параметры — в SwarmWeights, чтобы крутить через grid search.
"""

import math
import random as _random
import time as _t
from dataclasses import dataclass, field
from itertools import product as _product


# ── Зональная срочность для распределения подкреплений ───────────────────
# Чем выше urgency у планеты-получателя — тем приоритетнее перебросить
# туда корабли. Frontline/contested нуждаются в подкреплениях сильнее
# тыловых bastion/rear, независимо от того есть ли у них attack-планы.
ZONE_URGENCY: dict = {
    'frontline':  3.0,
    'contested':  2.0,
    'isolated':   1.5,
    'mid':        1.0,
    'rear':       0.5,
    'bastion':    0.3,
}
_ZONE_URG_DEFAULT = 1.0   # для неизвестных меток


# ── Параметризация (для тюнинга) ────────────────────────────────────────
@dataclass
class SwarmWeights:
    """Все параметры AgentSwarm одним пакетом — для grid search."""
    # Stress
    stress_top_k:     int   = 3       # сколько top-actions берём в stress
    stress_gamma:     float = 0.6     # дисконт по rank: m_0 + γ·m_1 + γ²·m_2…
    # Action value (для сортировки в аукционе)
    eta_bonus:        float = 30.0    # tiebreaker: ближе/быстрее → выше value
    priority_bonus:   float = 1.0     # вес priority (из zones) в value
    ships_weight:     float = 4.0     # «мяч на её стороне»: log1p(ships актора) даёт буст

    # Активность: чем выше — тем сильнее «застоявшиеся» планеты тащат свои
    # действия вверх в аукционе. Idle = ships − idle_floor. Линейный буст,
    # не log: чтобы 100 ship'овая планета чувствовалась сильно мощнее 30-ti.
    # 0 = выкл (используется только log-вариант ships_weight).
    activity_weight:  float = 0.5     # вес idle-bonus (на 1 idle-корабль)
    idle_floor:       int   = 25      # ships ниже считаются «активным гарнизоном»

    # Дистанция: насколько штрафуем дальние планы. eta_term = eta_bonus / eta^(1-comfort).
    # 0 = штраф 1/eta (текущий), 1 = плоско (eta вообще не влияет).
    # 0.5 = 1/sqrt(eta) — компромисс: дальние ещё штрафуются, но не катастрофично.
    distance_comfort: float = 0.0

    # Толерантность к риску: уменьшаем SAFETY_OVERKILL у атак (буфер сверх defender).
    # Чем выше — тем больше «тонко-проходных» планов становятся success'ными:
    # на planета сейчас может быть на 2 ship'а избыточно, не атакует, мы переждали ход
    # и потеряли темп. С risk=1 owned tgt требует +1 (вместо +2), neutral +0 (вместо +1).
    # 0 = consertative current, 1 = aggressive. Дальше ставить опасно — strict-< в движке.
    risk_tolerance:  int   = 0
    # Transfer (standalone TRANSFER P→Q вне duplet'a). По умолчанию выключен:
    # supply для атак идёт только через pipeline/multi_sync (осознанный duplet),
    # defense — через agent._build_defense_plans (raw=ours, projected=enemy).
    # Если хочешь экспериментировать с диффузией — включи.
    enable_redistribute: bool = True
    transfer_horizon: float = 40.0    # макс ETA для TRANSFER (если включено)
    transfer_floor:   float = 20.0    # GARRISON_FLOOR — не отправляем ниже
    transfer_thresh:  float = 5.0     # минимум score для коммита transfer
    transfer_eta_pen: float = 0.05    # штраф за ETA в transfer-score
    transfer_buffer:  float = 1.0     # ships сверх deficit получателя
    transfer_min_ships: int = 15      # минимум ships в одной TRANSFER-партии (анти-«капельница»)
    max_transfers_per_turn: int = 2  # максимум transfer-планов за ход (антидрейн)

    # ── MCTS аукцион ──────────────────────────────────────────────────────
    # Заменяет жадный single-pass на UCT-поиск по пространству комбинаций
    # планов. Находит лучший набор когда планы конкурируют за один актор.
    # use_mcts_auction=False → старый жадный (по умолчанию, нулевой overhead).
    # Включать после профилирования: занимает часть time budget до deadline.
    use_mcts_auction:    bool  = False
    mcts_c_uct:          float = 1.414   # UCB1 exploration constant (√2)

    # ── Зональный градиент в redistribute ────────────────────────────────
    # Добавляет (urgency[Q] − urgency[P]) × zone_urgency_weight к градиенту
    # transfer-score. Направляет корабли rear/bastion → frontline/contested
    # вне зависимости от attack-stress. 0.0 = только стресс (старое поведение).
    zone_urgency_weight: float = 1.5

    # ── Фильтр мелких direct-атак ─────────────────────────────────────────
    # direct-план с x_att < min_direct_att отбрасывается ЕСЛИ цель тоже
    # крупнее порога. Это отсекает 6-кор. флоты против 50-кор. планет,
    # но оставляет легальные атаки на маленьких нейтралов (x_tgt < порога).
    min_direct_att: int = 8

    # ── Гарнизон при захвате нейтрала ────────────────────────────────────
    # Сколько кораблей сверх минимума отправлять при атаке нейтрала.
    # Пример: нейтрал = 12 кораблей → без garrison отправляем 13, прилетаем
    # с 1 кораблём → redistribute сразу планирует трансфер (долго летит,
    # блокирует проекцию). С neutral_garrison=8 отправляем 21, прилетаем
    # с 9 кораблями — гарнизон встроен. Плюс: больше кораблей = быстрее летим
    # (fleet_speed_correct зависит от кол-ва), захват приходит раньше.
    # 0 = старое поведение (минимум). Хорошее стартовое значение: 5-10.
    neutral_garrison: int = 0

    # ── Проактивный гарнизон frontline/contested (для redistribute) ───────
    # Когда transfer не привязан к конкретному unfunded-плану (нет «события»),
    # redistribute всё равно должен уметь укрепить слабые frontline/contested
    # планеты до целевого уровня.
    #
    # Цель гарнизона = production × garrison_per_prod:
    #   production=3, garrison_per_prod=8 → цель=24 кораблей
    # Дефицит = max(0, цель − current_ships). Если у планеты уже ≥ цели —
    # дополнительного трансфера нет (deficit=0, кроме unfunded-дефицита).
    #
    # 0.0 = старое поведение (только unfunded-дефицит).
    # Хорошее стартовое значение: 5–10.
    # Применяется ТОЛЬКО к зонам frontline и contested; rear/bastion не трогаем.
    garrison_per_prod: float = 0.0

    # ── Приоритет атаки по силе оппонента ────────────────────────────────
    # В FFA (и иногда в 1v1) выгодно атаковать слабого прежде сильного:
    # слабый — лёгкие планеты + устранение → меньше фронтов.
    #
    # opp_strength_weight (W):
    #   0.0 = выключено (дефолт, нет изменений)
    #   > 0 = бонус за атаку слабых / штраф за атаку сильных.
    #
    # Механика: для каждой вражеской цели вычисляем relative_strength её хозяина
    # (сила / средняя сила по всем врагам). В _action_value добавляем:
    #   opp_bonus = W * (1 - rel_strength) * eta_bonus
    # Примеры при W=0.5, eta_bonus=30:
    #   rel=0.5 (вдвое слабее): +7.5  (агрессивнее атакуем слабого)
    #   rel=1.0 (средний):        0.0  (нет изменений)
    #   rel=2.0 (вдвое сильнее): −15.0 (избегаем лезть на сильного)
    #
    # opp_prod_factor: вес производства в оценке силы.
    #   сила_i = ships_i + prod_factor * prod_i
    #   production важнее в долгосрочной перспективе, но не известен наперёд.
    #   5.0 ≈ "1 прод = 5 кораблей" (конвертируется за ~5 ходов).
    opp_strength_weight: float = 0.0
    opp_prod_factor:     float = 5.0

    # ── Priority-reclassify порог (для zones.py post-pass) ────────────────
    # prio_reclassify_thr      — порог в начале матча (step=0).
    # prio_reclassify_thr_late — порог в конце матча (step=TOTAL_STEPS).
    # agent.py линейно интерполирует между ними по фазе → stage-aware тюнинг.
    #
    # Если оба одинаковые (дефолт) — статичный порог, нет интерполяции.
    # Пример stage-aware: thr=0.4 (агрессивная ранняя экспансия) →
    #                      thr_late=1.5 (осторожно в поздней игре).
    #
    # 99.0 = фактически выключить override (никакая periphery не апгрейдится).
    # Тюнинговый диапазон: 0.4 … 1.8.
    prio_reclassify_thr:      float = 0.8
    prio_reclassify_thr_late: float = 0.8   # = thr → нет интерполяции по дефолту


DEFAULT_WEIGHTS = SwarmWeights()


# ── Helpers ─────────────────────────────────────────────────────────────

def _seg_blocked(a, b, safety=SUN_SAFETY_PIPE):
    return segment_hits_sun(a.x, a.y, b.x, b.y, safety=safety)


def _planet_by_id(state, pid):
    raw = getattr(state, 'raw_planets', state.planets)
    for p in raw:
        if p.id == pid:
            return p
    return None


def _action_actors(plan):
    """Список planet_id, чьи ships тратит этот план."""
    actors = []
    if plan.get('att_id') is not None and plan.get('x_att', 0) > 0:
        actors.append((plan['att_id'], int(plan['x_att'])))
    if plan.get('sup_id') is not None and plan.get('x_sup', 0) > 0:
        actors.append((plan['sup_id'], int(plan['x_sup'])))
    return actors


def _action_value(plan, weights, priority_lookup=None, ships_lookup=None,
                  opp_strength_lookup=None):
    """Скор плана для сортировки в аукционе.

    margin            — успешные планы положительный, fail отрицательный
    + tiebreaker по ETA (чем быстрее тем лучше)
    + бонус по priority цели (если задан priority_lookup)
    + ships_term      — «мяч на её стороне»: log1p(ships у самого нагруженного
                        актора) множится на ships_weight. Идея: если у актора
                        накопилось много, его действия должны идти первыми в
                        аукционе — иначе она годами сидит в роли supplier'a и
                        никогда не стреляет.
    + opp_bonus       — бонус за атаку слабого оппонента / штраф за сильного.
                        opp_strength_lookup: {tgt_id → relative_strength}
                        rel < 1 = слабее среднего → положительный бонус
                        rel > 1 = сильнее → отрицательный (штраф)
    """
    margin = float(plan.get('margin', 0.0))
    eta    = float(plan.get('t_total', 0.0)) + 1.0

    # eta-штраф с поправкой distance_comfort:
    #   comfort=0 → 1/eta (классика)
    #   comfort=1 → константа (eta перестаёт штрафоваться)
    #   comfort=0.5 → 1/sqrt(eta) (мягко)
    comfort = max(0.0, min(1.0, getattr(weights, 'distance_comfort', 0.0)))
    eta_term = weights.eta_bonus / (eta ** (1.0 - comfort))

    prio = 0.0
    if priority_lookup is not None:
        prio = priority_lookup.get(plan.get('tgt_id'), 0.0)

    ships_term    = 0.0
    activity_term = 0.0
    if ships_lookup is not None:
        actor_ships_list = [
            ships_lookup.get(aid, 0) for aid, _c in _action_actors(plan)
        ]
        actor_max_ships = max(actor_ships_list, default=0)
        ships_term = weights.ships_weight * math.log1p(max(0, actor_max_ships))

        # активность: насколько актор «застоялся». Берём максимум по акторам
        # (любой загруженный actor → план поднимается), линейно.
        idle_max = max(
            (max(0, s - weights.idle_floor) for s in actor_ships_list),
            default=0,
        )
        activity_term = weights.activity_weight * idle_max

    # Бонус/штраф по силе оппонента-владельца цели.
    # Масштабируется через eta_bonus — чтобы быть в той же размерности что
    # остальные слагаемые (eta_term при eta=10 → ~3.0 при eta_bonus=30).
    opp_bonus = 0.0
    _opp_w = getattr(weights, 'opp_strength_weight', 0.0)
    if _opp_w != 0.0 and opp_strength_lookup is not None:
        rel = opp_strength_lookup.get(plan.get('tgt_id'), 1.0)
        opp_bonus = _opp_w * (1.0 - rel) * weights.eta_bonus

    return (margin + eta_term + weights.priority_bonus * prio
            + ships_term + activity_term + opp_bonus)


# ── Stress ──────────────────────────────────────────────────────────────

def compute_stress(candidates, ours, weights):
    """Per-planet stress: Σ top-K marginов с дисконтом γ^i.

    Стресс — индикатор «насколько у этой планеты много полезных действий
    прямо сейчас». Высокий stress → планета нужна для атак, не отвлекаем
    её на transfer. Низкий stress → планета бесполезна сейчас, годится
    как supplier для соседей.
    """
    by_actor = {p.id: [] for p in ours}
    for plan in candidates:
        m = float(plan.get('margin', 0.0))
        for actor_id, _cost in _action_actors(plan):
            if actor_id in by_actor:
                by_actor[actor_id].append(m)

    stress = {}
    K, gamma = weights.stress_top_k, weights.stress_gamma
    for pid, ms in by_actor.items():
        ms = sorted(ms, reverse=True)[:K]
        s = 0.0
        for i, m in enumerate(ms):
            s += m * (gamma ** i)
        stress[pid] = s
    return stress


def neighbor_stress(stress, ours):
    """WNN-weighted stress соседей: Σ (1/dist) · stress[Q] / Σ (1/dist)."""
    out = {}
    for p in ours:
        wsum = 0.0
        ssum = 0.0
        for q in ours:
            if q.id == p.id:
                continue
            d = math.hypot(p.x - q.x, p.y - q.y)
            if d < 1e-6:
                continue
            w = 1.0 / d
            wsum += w
            ssum += w * stress.get(q.id, 0.0)
        out[p.id] = ssum / wsum if wsum > 0 else 0.0
    return out


# ── Auction ─────────────────────────────────────────────────────────────

def auction(candidates, ours, weights, priority_lookup=None, opp_strength_lookup=None):
    """
    Single-pass greedy: сортируем все действия по value, идём сверху,
    коммитим если у всех акторов хватит ships и target не захвачен.

    Возвращает:
      committed   — список выбранных планов
      remaining   — {pid: ships_left} после вычета cost
      captured    — set targets которые уже атакованы
      unfunded    — список планов которые не прошли из-за нехватки ships
                    (используется в redistribute как сигнал «у этого actor'a
                    был дефицит»)
    """
    # стартовый бюджет — текущие ships каждой нашей планеты (минус reserve)
    remaining = {p.id: max(0, int(p.ships) - RESERVE_ON_ATT) for p in ours}
    ships_lookup = {p.id: int(p.ships) for p in ours}

    # сортировка по value desc; стабильно — успешные планы выше
    scored = [
        (
            (1 if plan.get('success') else 0),
            _action_value(plan, weights, priority_lookup, ships_lookup,
                          opp_strength_lookup),
            i,
            plan,
        )
        for i, plan in enumerate(candidates)
    ]
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

    committed = []
    captured = set()
    unfunded = []

    for _ok, _v, _i, plan in scored:
        if not plan.get('success'):
            continue
        tgt_id = plan.get('tgt_id')
        if tgt_id in captured:
            continue

        actors = _action_actors(plan)
        if any(remaining.get(aid, 0) < cost for aid, cost in actors):
            unfunded.append(plan)
            continue
        if _plan_total_ships(plan) < MIN_USEFUL_STRIKE:
            continue

        # commit
        for aid, cost in actors:
            remaining[aid] -= cost
        captured.add(tgt_id)
        committed.append(plan)

    return committed, remaining, captured, unfunded


# ── MCTS аукцион ────────────────────────────────────────────────────────

class _ANode:
    """UCT-узел дерева аукциона.

    На глубине i дерева стоит решение по candidates[i]: включить (True)
    или пропустить (False). Каждый путь корень→лист — одна комбинация планов.
    """
    __slots__ = ('n', 'v', 'ch')
    def __init__(self):
        self.n  = 0      # число посещений
        self.v  = 0.0    # суммарная ценность backprop
        self.ch = {}     # bool → _ANode


def auction_mcts(candidates, ours, weights, priority_lookup=None,
                 opp_strength_lookup=None, time_budget=0.05, c_uct=1.414):
    """UCT-аукцион: ищет лучшую комбинацию планов вместо жадного прохода.

    Зачем: жадный single-pass проигрывает когда два плана делят один актор.
    Пример: план A (ценный, актор P) выбирается первым и блокирует планы
    B+C (меньше каждый, но сумма > A), которые оба могут пройти без A.
    MCTS исследует пространство include/skip и находит B+C.

    Алгоритм:
      1. Кандидаты сортируются по value (как в greedy).
      2. UCT-дерево: на глубине i — решение include/skip для candidates[i].
      3. Rollout из текущего узла: жадный проход до конца списка.
      4. Обновляем best-solution если rollout дал лучший суммарный value.
      5. Backprop: обновляем n/v всех узлов пути.

    Возвращает тот же интерфейс что auction().
    """
    budget0  = {p.id: max(0, int(p.ships) - RESERVE_ON_ATT) for p in ours}
    ships_lk = {p.id: int(p.ships) for p in ours}

    valid = sorted(
        [p for p in candidates
         if p.get('success') and _plan_total_ships(p) >= MIN_USEFUL_STRIKE],
        key=lambda p: -_action_value(p, weights, priority_lookup, ships_lk,
                                     opp_strength_lookup),
    )
    n = len(valid)
    if not n:
        return [], dict(budget0), set(), []

    pvals = [_action_value(p, weights, priority_lookup, ships_lk,
                           opp_strength_lookup) for p in valid]

    # Лучшее решение среди всех rollout'ов (инициализируется greedy baseline)
    best = {'v': -1.0, 'comm': [], 'rem': dict(budget0), 'cap': set()}

    def _rollout(idx, rem, cap, v0, comm0):
        """Жадный rollout с позиции idx; обновляет best если нашли лучше."""
        r = dict(rem); c = set(cap); comm = list(comm0); v = v0
        for i in range(idx, n):
            p = valid[i]; tgt = p.get('tgt_id')
            if tgt in c:
                continue
            acts = _action_actors(p)
            if any(r.get(a, 0) < cost for a, cost in acts):
                continue
            for a, cost in acts:
                r[a] -= cost
            c.add(tgt); comm.append(p); v += pvals[i]
        if v > best['v']:
            best.update(v=v, comm=comm[:], rem=r, cap=set(c))
        return v

    # Seed: чисто жадный baseline (гарантирует не хуже старого поведения)
    _rollout(0, budget0, set(), 0.0, [])

    root  = _ANode()
    t_end = _t.perf_counter() + time_budget

    while _t.perf_counter() < t_end:
        # ── Selection + Expansion ────────────────────────────────────
        node     = root
        rem      = dict(budget0)
        cap      = set()
        acc_v    = 0.0
        acc_comm = []
        path     = [root]   # узлы для backprop
        idx      = 0

        while idx < n:
            p    = valid[idx]
            tgt  = p.get('tgt_id')
            acts = _action_actors(p)
            can_inc = (tgt not in cap
                       and all(rem.get(a, 0) >= cost for a, cost in acts))
            avail = [False] + ([True] if can_inc else [])

            # Нераскрытые дети → expansion
            unexp = [a for a in avail if a not in node.ch]
            if unexp:
                action = _random.choice(unexp)
                child  = _ANode()
                node.ch[action] = child
                if action:   # include
                    for a, cost in acts:
                        rem[a] -= cost
                    cap.add(tgt); acc_v += pvals[idx]; acc_comm.append(p)
                path.append(child)
                node = child
                idx += 1
                break        # один expansion → rollout

            # UCT-выбор среди уже открытых детей
            log_n = math.log(max(1, node.n))
            best_a, best_s = None, -1e18
            for a in avail:
                ch = node.ch[a]
                s  = (ch.v / ch.n + c_uct * math.sqrt(log_n / ch.n)
                      if ch.n > 0 else 1e18)
                if s > best_s:
                    best_s = s; best_a = a

            if best_a and can_inc:
                for a, cost in acts:
                    rem[a] -= cost
                cap.add(tgt); acc_v += pvals[idx]; acc_comm.append(p)

            node = node.ch[best_a]
            path.append(node)
            idx += 1

        # ── Rollout & backprop ───────────────────────────────────────
        total = _rollout(idx, rem, cap, acc_v, acc_comm)
        for nd in path:
            nd.n += 1
            nd.v += total

    comm     = best['comm']
    unfunded = [p for p in valid if p not in comm]
    return comm, best['rem'], best['cap'], unfunded


# ── Redistribute (TRANSFER) ─────────────────────────────────────────────

def redistribute(state, remaining, stress, neigh_stress, unfunded, ours, weights,
                 zone_lookup=None):
    """
    Для каждой P с остатком ships > floor — оценить TRANSFER в соседние Q
    и выпустить план если score выше порога.

    Score для P → Q:
        gradient = stress_grad + zone_urgency_weight · (urgency[Q] - urgency[P])
        score    = gradient · ships / (1 + eta · eta_pen)
    где ships ≤ remaining[P] - floor, ограниченное deficit_Q + buffer.

    zone_lookup: dict pid → zone-метка (из zones.py). Если задан — градиент
    включает разницу зональной срочности (ZONE_URGENCY): корабли rear/bastion
    автоматически тянутся к frontline/contested даже при нулевом attack-stress.
    Если не задан — поведение идентично старому (только stress-градиент).

    deficit_Q собирается из unfunded actions: если Q был supplier'ом или
    attacker'ом в плане который не прошёл из-за нехватки ships у Q —
    значит ему недостаёт. Если Q не имеет unfunded — buffer = 0, лимит —
    только базовый buffer из weights.
    """
    floor = weights.transfer_floor
    horizon = weights.transfer_horizon

    # сколько каждому Q не хватило в auction (по cost кораблей у Q)
    deficit = {p.id: 0.0 for p in ours}
    for plan in unfunded:
        for aid, cost in _action_actors(plan):
            d = max(0.0, cost - remaining.get(aid, 0))
            deficit[aid] = max(deficit.get(aid, 0.0), d)

    # ── Проактивный гарнизонный дефицит ──────────────────────────────────
    # Для frontline/contested-планет без unfunded-события: задаём целевой
    # минимальный гарнизон = production × garrison_per_prod. Если планета
    # ниже этого порога — считаем разницу «дефицитом» и направляем трансфер.
    # Это закрывает случай «нет конкретного события, но планета слабая».
    _gprod = float(getattr(weights, 'garrison_per_prod', 0.0))
    if _gprod > 0 and zone_lookup is not None:
        for Q in ours:
            zone = zone_lookup.get(Q.id, 'mid')
            if zone in ('frontline', 'contested'):
                target_garrison = _gprod * float(Q.production or 1)
                garrison_gap    = max(0.0, target_garrison - float(Q.ships))
                if garrison_gap > deficit.get(Q.id, 0.0):
                    deficit[Q.id] = garrison_gap

    transfer_plans = []
    by_id = {p.id: p for p in ours}
    _max_tr = int(getattr(weights, 'max_transfers_per_turn', 2))

    for P in ours:
        if len(transfer_plans) >= _max_tr:
            break
        free = remaining.get(P.id, 0) - floor
        if free <= 0:
            continue
        if int(free) < MIN_USEFUL_STRIKE:
            continue

        candidates = []
        for Q in ours:
            if Q.id == P.id:
                continue
            if _seg_blocked(P, Q):
                continue
            eta_pq, _ = _rendezvous_eta(P, Q, max(1, int(free)), state.omega)
            if eta_pq > horizon:
                continue

            stress_grad = stress.get(Q.id, 0.0) - stress.get(P.id, 0.0)
            if zone_lookup is not None:
                urg_q    = ZONE_URGENCY.get(zone_lookup.get(Q.id, 'mid'), _ZONE_URG_DEFAULT)
                urg_p    = ZONE_URGENCY.get(zone_lookup.get(P.id, 'mid'), _ZONE_URG_DEFAULT)
                gradient = stress_grad + weights.zone_urgency_weight * (urg_q - urg_p)
            else:
                gradient = stress_grad
            if gradient <= 0:
                continue
            # сколько имеет смысл отправить
            max_useful = deficit.get(Q.id, 0.0) + weights.transfer_buffer
            ships = int(min(free, max_useful))
            if ships < MIN_USEFUL_STRIKE:
                continue

            score = gradient * ships / (1.0 + eta_pq * weights.transfer_eta_pen)
            candidates.append((score, ships, eta_pq, Q))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])
        best_score, ships, eta_pq, Q = candidates[0]
        if best_score < weights.transfer_thresh:
            continue
        # анти-«капельница»: маленькие партии не имеют смысла, планета должна
        # либо отправить значимый кусок, либо копить дальше
        if ships < int(getattr(weights, 'transfer_min_ships', MIN_USEFUL_STRIKE)):
            continue

        # эмитим план в формате совместимом с agent._execute_plan_atomically
        transfer_plans.append({
            'mode':       'transfer',
            'sup_id':     None,
            'att_id':     P.id,
            'tgt_id':     Q.id,
            'eta_sa':     0.0,
            'eta_at':     float(eta_pq),
            't_total':    float(eta_pq),
            'x_sup':      0,
            'x_att':      ships,
            'prod_att':   float(P.production),
            'x_tgt':      0,
            'prod_tgt':   0.0,
            'strike':     float(ships),
            'defender':   0.0,
            'incoming':   0.0,
            'needed':     0,
            'margin':     float(ships),
            'success':    True,
            'relay_reason': None,
            'transfer_score':   best_score,
            'transfer_gradient': stress.get(Q.id, 0.0) - stress.get(P.id, 0.0),
            'is_transfer':      True,
        })
        remaining[P.id] -= ships

    return transfer_plans


# ── Главная точка входа ────────────────────────────────────────────────

def swarm_plan(state, player, targets, weights=None, priority_lookup=None,
               zone_lookup=None, horizon=ATTACK_HORIZON, max_targets=None, deadline=None):
    """
    Полный цикл AgentSwarm: candidates → stress → auction → redistribute.

    max_targets: жёсткий cap на число целей для перебора пар (None = все).
                 Если cap, берём топ по priority_lookup.
    deadline:    time.perf_counter() граница в секундах. После неё прекращаем
                 наращивать candidates (что собрали — то собрали).

    Возвращает (plans, debug):
      plans  — финальный список планов
      debug  — словарь метаинформации (stress, остатки, разбивка)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = getattr(state, 'raw_planets', state.planets)
    ours = [p for p in raw if p.owner == player]
    if not ours or not targets:
        return [], {'reason': 'no ours or no targets'}

    # ── Opponent strength lookup (для opp_strength_weight) ────────────────
    # Считаем силу каждого противника: ships + prod_factor * production.
    # Нормализуем к среднему среди врагов → rel_strength = 1.0 означает средний враг.
    # opp_bonus в _action_value = W * (1.0 - rel_strength) * eta_bonus:
    #   rel < 1 (слабый) → bonus > 0 (атакуем охотнее)
    #   rel > 1 (сильный) → bonus < 0 (осторожнее)
    opp_strength_lookup: dict = {}
    _opp_w = getattr(weights, 'opp_strength_weight', 0.0)
    if _opp_w != 0.0:
        prod_factor = getattr(weights, 'opp_prod_factor', 5.0)
        # Суммируем корабли и производство по owner (планеты + флоты)
        opp_ships: dict = {}
        opp_prod:  dict = {}
        for p in raw:
            own = p.owner
            if own == player or own < 0:
                continue
            opp_ships[own] = opp_ships.get(own, 0.0) + float(getattr(p, 'ships', 0) or 0)
            opp_prod[own]  = opp_prod.get(own,  0.0) + float(getattr(p, 'production', 0) or 0)
        # Флоты тоже учитываем
        for f in getattr(state, 'fleets', []):
            own = getattr(f, 'owner', -1)
            if own == player or own < 0:
                continue
            opp_ships[own] = opp_ships.get(own, 0.0) + float(getattr(f, 'ships', 0) or 0)
        # Суммарная сила каждого врага
        opp_ids = set(opp_ships) | set(opp_prod)
        if opp_ids:
            strength = {oid: opp_ships.get(oid, 0.0) + prod_factor * opp_prod.get(oid, 0.0)
                        for oid in opp_ids}
            mean_s = sum(strength.values()) / len(strength)
            if mean_s > 0:
                rel = {oid: s / mean_s for oid, s in strength.items()}
            else:
                rel = {oid: 1.0 for oid in opp_ids}
            # Строим lookup: tgt_id (planet id) → rel_strength его owner'а
            for p in raw:
                own = p.owner
                if own in rel:
                    opp_strength_lookup[p.id] = rel[own]

    # cap по числу целей (на ход с 30+ нейтралами это спасает от per-step timeout)
    if max_targets is not None and len(targets) > max_targets:
        if priority_lookup:
            targets = sorted(targets, key=lambda t: -priority_lookup.get(t.id, 0))[:max_targets]
        else:
            targets = list(targets)[:max_targets]

    # 0. Все боевые candidates от существующего движка attacks.py
    risk = int(getattr(weights, 'risk_tolerance', 0))
    min_dir_att = int(getattr(weights, 'min_direct_att', 8))

    # Кеш заблокированных солнцем пар (src_id, tgt_id) — строится один раз
    # на весь ход. Устраняет >1000 бесполезных aim_verify_failed за матч:
    # _seg_blocked использует SUN_SAFETY_PIPE (консервативный радиус), поэтому
    # маршруты из кеша никогда не пройдут _aim_and_verify в agent.py.
    blocked_pairs: set = set()
    for _src in ours:
        for _tgt in targets:
            if _seg_blocked(_src, _tgt):
                blocked_pairs.add((_src.id, _tgt.id))

    candidates = []
    for tgt in targets:
        if deadline is not None and _t.perf_counter() > deadline:
            break  # бюджет исчерпан, играем что собрали
        # Только планеты с незаблокированным прямым маршрутом до цели
        reachable = [p for p in ours if (p.id, tgt.id) not in blocked_pairs]
        if not reachable:
            continue
        plans = all_plans(state, tgt, reachable, horizon=horizon, player=player, risk=risk,
                          neutral_garrison=int(getattr(weights, 'neutral_garrison', 0)))
        for pl in plans:
            if not (pl.get('success') and _plan_total_ships(pl) >= MIN_USEFUL_STRIKE):
                continue
            # Фильтр мелких direct-атак: x_att < порога против крупной цели —
            # флот всё равно не победит и только теряется. Нейтралов с малым
            # гарнизоном (x_tgt < порога) не трогаем — там 6 кор. нормально.
            if pl.get('mode') == 'direct':
                x_att = int(pl.get('x_att', 0))
                x_tgt = float(pl.get('x_tgt', 0))
                if x_att < min_dir_att and x_tgt >= min_dir_att:
                    continue
            candidates.append(pl)

    # 1. Stress / neighbor_stress (для отладки и transfer-scoring)
    stress = compute_stress(candidates, ours, weights)
    neigh  = neighbor_stress(stress, ours)

    # 2. Аукцион: жадный или UCT-MCTS
    if getattr(weights, 'use_mcts_auction', False):
        # Выделяем до 30% оставшегося бюджета на MCTS
        mcts_budget = 0.05
        if deadline is not None:
            rem_time    = deadline - _t.perf_counter()
            mcts_budget = max(0.02, rem_time * 0.30)
        committed, remaining, captured, unfunded = auction_mcts(
            candidates, ours, weights, priority_lookup=priority_lookup,
            opp_strength_lookup=opp_strength_lookup,
            time_budget=mcts_budget,
            c_uct=getattr(weights, 'mcts_c_uct', 1.414),
        )
    else:
        committed, remaining, captured, unfunded = auction(
            candidates, ours, weights, priority_lookup=priority_lookup,
            opp_strength_lookup=opp_strength_lookup,
        )

    # 3. Redistribute остатки в TRANSFER.
    #    По умолчанию ВЫКЛЮЧЕН — supply для атак идёт через pipeline/multi
    #    (осознанный duplet), defense через agent._build_defense_plans.
    #    Standalone-передачи легко превращают планету в «вечного supplier'a»,
    #    она копит-капает и никогда не атакует. Включай только осознанно.
    if getattr(weights, 'enable_redistribute', False):
        transfers = redistribute(
            state, remaining, stress, neigh, unfunded, ours, weights,
            zone_lookup=zone_lookup,
        )
    else:
        transfers = []

    plans = committed + transfers

    debug = {
        'n_candidates':   len(candidates),
        'n_committed':    len(committed),
        'n_transfers':    len(transfers),
        'n_unfunded':     len(unfunded),
        'stress':         stress,
        'neighbor_stress': neigh,
        'remaining':      dict(remaining),
        'captured':       sorted(captured),
    }
    return plans, debug


__all__ = [
    'SwarmWeights', 'DEFAULT_WEIGHTS',
    'ZONE_URGENCY',
    'compute_stress', 'neighbor_stress',
    'auction', 'auction_mcts', 'redistribute', 'swarm_plan',
]

# ╔══════════════════════════════════════════════════╗
# ║  context.py                                    ║
# ╚══════════════════════════════════════════════════╝

"""
context.py — Game Understanding Layer (GUL).

Глобальное «понимание игры» вычисляется один раз в начале хода и
раздаётся всем cell-decisions как контекст для модулирования action-value.

4 области:
  1. phase           — стадия игры (early/mid/late/endgame).
  2. stance          — стратегическая поза (expansion/attrition/consolidation/desperate).
                       + анализ рельефа из ships (центр масс, разделение,
                       dominance balance по grid).
  3. pressure_field  — для каждой планеты значение поля Σ sign·ships/dist²
                       и роль (deep_rear/rear/frontline/forward/isolated).
  4. opponent_intent — упрощённая ToM: passive/expanding/aggressive/
                       preparing_attack по флотам противника.

Использование:
    ctx = compute_context(state, player)
    # ctx.phase, ctx.stance, ctx.field_at[pid], ctx.opponent_intent, ...
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

NEUTRAL = -1


# ── Параметры (порог-константы; легко крутить) ──────────────────────────

# Phase
PHASE_EARLY_PROGRESS = 0.30
PHASE_MID_PROGRESS   = 0.70

# Stance
STANCE_DESPERATE_RATIO    = 0.35   # ships_ratio < этого → desperate
STANCE_CONSOLIDATION_RATIO = 0.65  # ships_ratio > И нет нейтралов → consolidation
STANCE_EXPANSION_NEUTRAL_FRAC = 0.5   # > половины нейтралов осталось → expansion

# Pressure field
FIELD_SOFTEN     = 5.0    # в знаменателе чтобы избежать singularity на самой планете
FIELD_HIGH       = 0.30   # порог field-value для роли deep_rear/isolated
FIELD_MID        = 0.05   # порог для rear/forward
GRID_STEP        = 20     # для оценки dominance_balance (5×5 = 25 точек)

# Opponent intent
ENEMY_PASSIVE_SHIPS  = 30   # < этого ships у врага → passive
INTENT_ANGLE_TOL     = 0.30 # рад: насколько угол флота должен совпадать с направлением на цель


# ── Структура контекста ─────────────────────────────────────────────────

@dataclass
class GameContext:
    """Глобальное состояние игры на текущий ход. Передаётся в cell-decisions."""
    # — Phase —
    phase: str = 'mid'              # 'early' | 'mid' | 'late' | 'endgame'
    phase_progress: float = 0.0     # 0..1: захвачено vs нейтралов от начала
    total_ships: int = 0            # все ships на планетах + в флотах

    # — Stance —
    stance: str = 'attrition'       # 'expansion'|'attrition'|'consolidation'|'desperate'
    prod_ratio: float = 1.0         # our_prod / enemy_prod (1.0 = паритет)
    ships_ratio: float = 0.5        # our / (our+enemy) — без нейтралов

    # Ship terrain (из «рельефа»)
    our_com:        Tuple[float, float] = (50.0, 50.0)   # центр масс наших (взвешен по ships)
    enemy_com:      Tuple[float, float] = (50.0, 50.0)
    com_separation: float = 0.0     # расстояние между центрами
    dominance_balance: float = 0.0  # (positive_grid_cells - negative) / total — диагностика «кто контролирует пространство»

    # — Pressure field —
    field_at: Dict[int, float] = field(default_factory=dict)   # pid → field-value (>0 наше, <0 чужое)
    role_of:  Dict[int, str]   = field(default_factory=dict)   # pid → 'deep_rear'/'rear'/'frontline'/'forward'/'isolated'

    # — Opponent intent —
    opponent_intent: str = 'unknown'                           # 'passive'|'expanding'|'aggressive'|'preparing_attack'|'idle'
    enemy_internal_transfers: List[Tuple[int, int]] = field(default_factory=list)  # (from_pid, to_pid) — флоты enemy→enemy
    threatened_planets: List[int] = field(default_factory=list)                    # вражеские pid'ы куда летит supply (готовятся стрелять)


# ── Вспомогательные ─────────────────────────────────────────────────────

def _com(planets):
    """Центр масс взвешенный по ships. Если у всех 0 ships — берём геометрический."""
    w = sum(max(0, p.ships) for p in planets)
    if w <= 0:
        if not planets:
            return (50.0, 50.0)
        cx = sum(p.x for p in planets) / len(planets)
        cy = sum(p.y for p in planets) / len(planets)
        return (cx, cy)
    cx = sum(p.x * max(0, p.ships) for p in planets) / w
    cy = sum(p.y * max(0, p.ships) for p in planets) / w
    return (cx, cy)


def _field_value(x, y, raw, player, exclude_id=None):
    """Σ sign · ships / (dist² + soften²). + наши, − чужие, нейтралы игнор."""
    s = 0.0
    sof2 = FIELD_SOFTEN ** 2
    for p in raw:
        if exclude_id == p.id or p.owner == NEUTRAL:
            continue
        sign = 1.0 if p.owner == player else -1.0
        d2 = (x - p.x) ** 2 + (y - p.y) ** 2 + sof2
        s += sign * max(0, p.ships) / d2
    return s


def _classify_role(field_v):
    """field-value → роль клетки в общем рельефе."""
    if field_v >  FIELD_HIGH: return 'deep_rear'
    if field_v >  FIELD_MID:  return 'rear'
    if field_v > -FIELD_MID:  return 'frontline'
    if field_v > -FIELD_HIGH: return 'forward'
    return 'isolated'


def _fleet_target(f, raw):
    """К какой планете летит флот (по углу с допуском). None если не ясно."""
    best_pid, best_d = None, float('inf')
    for p in raw:
        ax = math.atan2(p.y - f.y, p.x - f.x)
        diff = abs(((ax - f.angle + math.pi) % (2 * math.pi)) - math.pi)
        if diff > INTENT_ANGLE_TOL:
            continue
        d = math.hypot(p.x - f.x, p.y - f.y)
        if d < best_d:
            best_d = d
            best_pid = p.id
    return best_pid


# ── Главный entry point ─────────────────────────────────────────────────

def compute_context(state, player) -> GameContext:
    """Собирает GameContext из state. Дешёво (~O(n_planets² + n_fleets·n_planets))."""
    ctx = GameContext()
    raw = list(getattr(state, 'raw_planets', state.planets))
    fleets = list(getattr(state, 'fleets', []))

    ours    = [p for p in raw if p.owner == player]
    enemy   = [p for p in raw if p.owner not in (NEUTRAL, player)]
    neutral = [p for p in raw if p.owner == NEUTRAL]

    # Initial neutrals — оценка, чтобы знать сколько было «в начале»
    initial_planets = []
    init_dict = getattr(state, '_initial_planets', None)
    if init_dict is not None:
        try:
            initial_planets = list(init_dict.values())
        except Exception:
            initial_planets = list(init_dict)
    if not initial_planets:
        initial_planets = list(getattr(state, 'initial_planets', []) or raw)
    initial_neutral = max(1, sum(1 for p in initial_planets if p.owner == NEUTRAL))

    # ─────────────────────────────────────────────────
    # 1. PHASE
    # ─────────────────────────────────────────────────
    ctx.phase_progress = round(1.0 - len(neutral) / initial_neutral, 3)

    fleet_ours  = sum(f.ships for f in fleets if f.owner == player)
    fleet_enemy = sum(f.ships for f in fleets if f.owner not in (NEUTRAL, player))
    our_ships   = sum(p.ships for p in ours) + fleet_ours
    enemy_ships = sum(p.ships for p in enemy) + fleet_enemy
    ctx.total_ships = our_ships + enemy_ships

    if   ctx.phase_progress < PHASE_EARLY_PROGRESS: ctx.phase = 'early'
    elif ctx.phase_progress < PHASE_MID_PROGRESS:   ctx.phase = 'mid'
    elif neutral:                                   ctx.phase = 'late'
    else:                                           ctx.phase = 'endgame'

    # ─────────────────────────────────────────────────
    # 2. STANCE + ship terrain
    # ─────────────────────────────────────────────────
    our_prod   = sum(p.production for p in ours)
    enemy_prod = sum(p.production for p in enemy)
    ctx.prod_ratio  = round(our_prod / max(1, enemy_prod), 3)
    ctx.ships_ratio = round(our_ships / max(1, our_ships + enemy_ships), 3)

    ctx.our_com   = _com(ours)
    ctx.enemy_com = _com(enemy)
    ctx.com_separation = round(math.hypot(
        ctx.our_com[0] - ctx.enemy_com[0],
        ctx.our_com[1] - ctx.enemy_com[1]
    ), 2)

    # Stance (приоритет: desperate > consolidation > expansion > attrition)
    if ctx.ships_ratio < STANCE_DESPERATE_RATIO:
        ctx.stance = 'desperate'
    elif ctx.ships_ratio > STANCE_CONSOLIDATION_RATIO and not neutral:
        ctx.stance = 'consolidation'
    elif len(neutral) > STANCE_EXPANSION_NEUTRAL_FRAC * initial_neutral:
        ctx.stance = 'expansion'
    else:
        ctx.stance = 'attrition'

    # ─────────────────────────────────────────────────
    # 3. PRESSURE FIELD
    # ─────────────────────────────────────────────────
    # Field-value на каждой планете (исключая саму себя из суммы)
    for p in raw:
        v = _field_value(p.x, p.y, raw, player, exclude_id=p.id)
        ctx.field_at[p.id] = round(v, 4)
        ctx.role_of[p.id]  = _classify_role(v)

    # dominance_balance: 5x5 grid через всю карту
    pos = neg = 0
    for x in range(GRID_STEP // 2, 100, GRID_STEP):
        for y in range(GRID_STEP // 2, 100, GRID_STEP):
            v = _field_value(x, y, raw, player)
            if   v >  FIELD_MID: pos += 1
            elif v < -FIELD_MID: neg += 1
    total = max(1, pos + neg)
    ctx.dominance_balance = round((pos - neg) / total, 3)

    # ─────────────────────────────────────────────────
    # 4. OPPONENT INTENT
    # ─────────────────────────────────────────────────
    enemy_pids = {p.id for p in enemy}
    our_pids   = {p.id for p in ours}

    enemy_fleets = [f for f in fleets if f.owner not in (NEUTRAL, player)]
    n_enemy_in_flight = len(enemy_fleets)

    transfers = []                  # enemy → enemy (внутренние)
    threatened = []                 # принимающие (готовятся стрелять)
    toward_us = 0                   # летит к нашим
    for f in enemy_fleets:
        tgt = _fleet_target(f, raw)
        if tgt is None:
            continue
        if tgt in enemy_pids:
            transfers.append((int(getattr(f, 'from_pid', -1)), int(tgt)))
            threatened.append(int(tgt))
        elif tgt in our_pids:
            toward_us += 1

    ctx.enemy_internal_transfers = transfers
    ctx.threatened_planets = sorted(set(threatened))

    if enemy_ships < ENEMY_PASSIVE_SHIPS and n_enemy_in_flight == 0:
        ctx.opponent_intent = 'passive'
    elif transfers:
        ctx.opponent_intent = 'preparing_attack'
    elif n_enemy_in_flight == 0:
        ctx.opponent_intent = 'idle'
    elif toward_us > n_enemy_in_flight * 0.5:
        ctx.opponent_intent = 'aggressive'
    else:
        ctx.opponent_intent = 'expanding'

    return ctx


def context_summary(ctx: GameContext) -> str:
    """Однострочная сводка для лога."""
    return (f"phase={ctx.phase}({ctx.phase_progress:.2f}) "
            f"stance={ctx.stance} ships={ctx.ships_ratio:.2f} prod={ctx.prod_ratio:.2f} "
            f"COM_sep={ctx.com_separation:.0f} dom={ctx.dominance_balance:+.2f} "
            f"opp={ctx.opponent_intent}"
            + (f" threats={ctx.threatened_planets}" if ctx.threatened_planets else "")
            )


__all__ = ['GameContext', 'compute_context', 'context_summary']

# ╔══════════════════════════════════════════════════╗
# ║  opponent_presets.py                           ║
# ╚══════════════════════════════════════════════════╝

"""
opponent_presets.py -- Opponent presets based on SwarmWeights grid-search winners.

Each preset is a named SwarmWeights configuration from eval_8winners.csv.
Action prediction uses the same scoring logic as swarm._action_value but
simplified (no zones needed) so it runs in microseconds per call.

Score(src, tgt, W) =
    margin
    + W.eta_bonus / eta^(1 - W.distance_comfort)
    + W.ships_weight * log1p(surplus)
    + W.activity_weight * max(0, surplus - W.idle_floor)
    + W.priority_bonus * tgt.production

Public interface:
  PRESETS              : dict {name: SwarmWeights}
  get_preset_actions(preset_name, state, opp_id) -> List[dict]
"""

import math
from typing import List, Dict


# ── Preset definitions (from eval_8winners_20260502_050430.csv) ────────────
# Columns: activity_weight, idle_floor, distance_comfort, risk_tolerance,
#          ships_weight, eta_bonus, priority_bonus, stress_top_k, stress_gamma

PRESETS: Dict[str, SwarmWeights] = {
    # default SwarmWeights (baseline, winrate=0.38)
    'default': DEFAULT_WEIGHTS,

    # winner_seed19  winrate=0.45
    'w_seed19': SwarmWeights(
        activity_weight=2.14, idle_floor=38,  distance_comfort=0.28,
        risk_tolerance=0,     ships_weight=0.57, eta_bonus=51.96,
        priority_bonus=5.63,  stress_top_k=5,  stress_gamma=1.27,
    ),

    # winner_seed20  winrate=0.46
    'w_seed20': SwarmWeights(
        activity_weight=1.29, idle_floor=40,  distance_comfort=0.18,
        risk_tolerance=2,     ships_weight=1.37, eta_bonus=16.66,
        priority_bonus=2.84,  stress_top_k=3,  stress_gamma=1.50,
    ),

    # winner_seed39  winrate=0.46
    'w_seed39': SwarmWeights(
        activity_weight=2.79, idle_floor=37,  distance_comfort=0.19,
        risk_tolerance=1,     ships_weight=5.76, eta_bonus=14.33,
        priority_bonus=0.37,  stress_top_k=6,  stress_gamma=1.29,
    ),

    # winner_seed22  winrate=0.37
    'w_seed22': SwarmWeights(
        activity_weight=3.18, idle_floor=16,  distance_comfort=0.21,
        risk_tolerance=0,     ships_weight=4.40, eta_bonus=39.09,
        priority_bonus=1.84,  stress_top_k=5,  stress_gamma=1.34,
    ),
}

# ── Scoring parameters ─────────────────────────────────────────────────────
RESERVE_RATIO = 1.5    # keep production * ratio on source (mirror action_space.py)
MIN_SURPLUS   = 3      # minimum surplus to consider launching
MAX_ACTIONS   = 3      # max actions per preset
SPEED_SAMPLE  = 20     # reference ship count for speed estimate


def _surplus(p) -> int:
    return max(0, int(p.ships) - int(p.production * RESERVE_RATIO))


def _eta_est(src, tgt, ships: int) -> float:
    """Rough ETA: straight-line distance / fleet speed."""
    spd = fleet_speed_correct(max(1, ships))
    d   = math.hypot(tgt.x - src.x, tgt.y - src.y)
    d   = max(d - src.radius - tgt.radius, 1.0)
    return max(1.0, d / max(spd, 1e-6))


def _overkill(tgt, risk_tolerance: int) -> int:
    """Minimum overkill buffer mirroring SAFETY_OVERKILL logic.

    risk_tolerance reduces the buffer (more aggressive, risker attacks).
    0 = conservative (+2 for owned, +1 for neutral)
    1 = slightly risky (+1 / +0)
    2+ = very aggressive (just > defender)
    """
    if tgt.owner == NEUTRAL_OWNER:
        return max(0, 1 - risk_tolerance)
    return max(0, 2 - risk_tolerance)


def _score(src, tgt, surplus: int, W: SwarmWeights) -> float:
    """Simplified action_value for (src->tgt) under SwarmWeights W."""
    ok      = _overkill(tgt, W.risk_tolerance)
    margin  = surplus - float(tgt.ships) - ok
    eta     = _eta_est(src, tgt, surplus)
    comfort = max(0.0, min(1.0, W.distance_comfort))
    eta_term      = W.eta_bonus / (eta ** (1.0 - comfort))
    ships_term    = W.ships_weight * math.log1p(max(0, surplus))
    activity_term = W.activity_weight * max(0, surplus - W.idle_floor)
    prio_term     = W.priority_bonus * float(tgt.production)
    return margin + eta_term + ships_term + activity_term + prio_term


def _action_type(tgt, opp_id: int) -> str:
    if tgt.owner == opp_id:
        return 'reinforce'
    if tgt.owner == NEUTRAL_OWNER:
        return 'capture_neutral'
    return 'attack_enemy'


# ══════════════════════════════════════════════════════════════════════════
# Core generator
# ══════════════════════════════════════════════════════════════════════════

def get_preset_actions(preset_name: str, state, opp_id: int) -> List[dict]:
    """Predict opponent actions under the given preset (SwarmWeights).

    Algorithm:
    1. Gather opponent planets with surplus ships.
    2. For each (src, tgt) pair not blocked by sun, compute score.
    3. Greedily pick top-MAX_ACTIONS pairs (each target used at most once).

    Returns list of dicts: {from_id, target_id, ships, action_type}.
    """
    W = PRESETS.get(preset_name)
    if W is None:
        return []

    try:
        raw_planets = getattr(state, 'raw_planets', state.planets)
        all_planets = state.planets

        srcs = [(p, _surplus(p)) for p in raw_planets
                if p.owner == opp_id and _surplus(p) >= MIN_SURPLUS]
        if not srcs:
            return []

        # Candidates: all scored (src, tgt, score) pairs
        scored = []
        for src, surp in srcs:
            for tgt in all_planets:
                if tgt.id == src.id:
                    continue
                if segment_hits_sun(src.x, src.y, tgt.x, tgt.y, safety=SUN_SAFETY):
                    continue
                ok      = _overkill(tgt, W.risk_tolerance)
                needed  = int(tgt.ships) + ok
                send    = surp if tgt.owner == opp_id else min(surp, max(MIN_SURPLUS, needed))
                if send < MIN_SURPLUS:
                    continue
                # For capture: need at least needed ships
                if tgt.owner != opp_id and surp < needed:
                    continue
                sc = _score(src, tgt, surp, W)
                scored.append((sc, src, tgt, send))

        if not scored:
            return []

        scored.sort(key=lambda x: -x[0])

        # Greedy pick: each (src, tgt) used at most once
        used_src = set()
        used_tgt = set()
        actions  = []
        for sc, src, tgt, send in scored:
            if len(actions) >= MAX_ACTIONS:
                break
            if src.id in used_src or tgt.id in used_tgt:
                continue
            used_src.add(src.id)
            used_tgt.add(tgt.id)
            actions.append({
                'from_id':     src.id,
                'target_id':   tgt.id,
                'ships':       max(1, int(send)),
                'action_type': _action_type(tgt, opp_id),
            })

        return actions

    except Exception:
        return []


__all__ = ['PRESETS', 'get_preset_actions', 'MAX_ACTIONS']

# ╔══════════════════════════════════════════════════╗
# ║  opponent_model_bayesian.py                    ║
# ╚══════════════════════════════════════════════════╝

"""
opponent_model_bayesian.py -- Bayesian opponent behaviour model with analytics.

ANALYTICS_MODE (module-level bool, default False):
  When False  -- zero overhead: no history stored, no log/entropy computed.
  When True   -- full metrics collected in self.analytics_history each turn.
  Set from agent.py: import opponent_model_bayesian; opponent_model_bayesian.ANALYTICS_MODE = True

Public interface:
  ANALYTICS_MODE           : bool  (set before creating model instance)
  class OpponentModelBayesian
    .update(observed_actions, state, opp_id, step=0) -> dict|None
    .get_expected_actions(state, opp_id, threshold=0.05) -> List[dict]
    .top_preset() -> str
    .summary()   -> str
    .priors      : dict {preset_name: probability}
    ._history    : List[str]          -- top preset each turn (always)
    .analytics_history : List[dict]   -- full metrics (only if ANALYTICS_MODE)
"""

import math
from typing import List, Dict, Optional


# ── Module-level analytics flag ────────────────────────────────────────────
# Set to True from agent.py when _dbg.enabled() is True.
# All analytics code is guarded by `if ANALYTICS_MODE:` → zero cost in battle.
ANALYTICS_MODE: bool = False

# ── Parameters ─────────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 1.5
MIN_PROB            = 0.02   # probability floor per preset
MAX_VIRTUAL_FLEETS  = 3      # cap on virtual fleets per turn


class OpponentModelBayesian:
    """Bayesian distribution over opponent strategy presets.

    Fast path (ANALYTICS_MODE=False):
        Likelihoods computed, priors updated, top preset appended to _history.
        No log/entropy/history dict created.

    Full path (ANALYTICS_MODE=True):
        Additionally computes surprise, entropy, match_count, stores full
        metrics dict in analytics_history for post-game analysis.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        n = len(PRESETS)
        self.priors: Dict[str, float] = {k: 1.0 / n for k in PRESETS}
        self.temperature = temperature
        self._history: List[str] = []          # top preset name each turn (always)
        self.analytics_history: List[dict] = []  # full metrics (ANALYTICS_MODE only)

    # ── Bayesian update ────────────────────────────────────────────────────

    def update(self, observed_actions: List[dict],
               state, opp_id: int,
               step: int = 0) -> Optional[dict]:
        """Update priors given observed opponent actions.

        Args:
            observed_actions: list of {from_id, target_id, ships, action_type}
            state:            GameState when opponent acted (prev turn raw state)
            opp_id:           opponent player id
            step:             current game step (for analytics)

        Returns:
            metrics dict if ANALYTICS_MODE else None.
        """
        # ── No observed actions: opponent was idle ──────────────────────────
        if not observed_actions:
            top = self.top_preset()
            self._history.append(top)
            if ANALYTICS_MODE:
                entropy = -sum(p * math.log(max(p, 1e-12))
                               for p in self.priors.values())
                rec = {
                    'step': step,
                    'surprise': 0.0,        # idle = not surprising
                    'entropy': entropy,
                    'top_preset': top,
                    'top_prob': self.priors[top],
                    'n_opponent_actions': 0,
                    'match_count': 0,
                    'priors': dict(self.priors),
                }
                self.analytics_history.append(rec)
                return rec
            return None

        # ── Compute likelihoods ────────────────────────────────────────────
        n_obs      = max(1, len(observed_actions))
        obs_pairs  = {(a['from_id'], a['target_id']) for a in observed_actions}

        likelihoods: Dict[str, float] = {}
        pred_cache:  Dict[str, list]  = {}

        for preset_name in self.priors:
            predicted = get_preset_actions(preset_name, state, opp_id)
            pred_cache[preset_name] = predicted

            if not predicted:
                likelihoods[preset_name] = 1.0   # neutral: no prediction → no update
            else:
                pred_pairs = {(p['from_id'], p['target_id']) for p in predicted}
                pred_types = {p['action_type'] for p in predicted}

                matched_exact = sum(
                    1 for a in observed_actions
                    if (a['from_id'], a['target_id']) in pred_pairs
                )
                matched_type = sum(
                    1 for a in observed_actions
                    if a['action_type'] in pred_types
                )
                # Exact match weighs 2x type match
                score = (2 * matched_exact + matched_type) / (3 * n_obs)
                likelihoods[preset_name] = math.exp(self.temperature * score)

        # ── Analytics (pre-update) ─────────────────────────────────────────
        if ANALYTICS_MODE:
            # P(obs) = marginal likelihood (evidence term in Bayes rule)
            p_obs    = sum(self.priors[k] * likelihoods[k] for k in self.priors)
            surprise = -math.log(max(p_obs, 1e-12))

        # ── Bayesian update ────────────────────────────────────────────────
        new_priors = {k: self.priors[k] * likelihoods[k] for k in self.priors}
        total = sum(new_priors.values())

        if total < 1e-12:
            n = len(PRESETS)
            self.priors = {k: 1.0 / n for k in PRESETS}
        else:
            # Apply floor and re-normalise
            floored = {k: max(MIN_PROB, v / total) for k, v in new_priors.items()}
            total2  = sum(floored.values())
            self.priors = {k: v / total2 for k, v in floored.items()}

        top = self.top_preset()
        self._history.append(top)

        # ── Analytics (post-update) ────────────────────────────────────────
        if ANALYTICS_MODE:
            entropy = -sum(p * math.log(max(p, 1e-12))
                           for p in self.priors.values())

            top_pairs = {(p['from_id'], p['target_id'])
                         for p in pred_cache.get(top, [])}
            match_count = sum(
                1 for a in observed_actions
                if (a['from_id'], a['target_id']) in top_pairs
            )

            rec = {
                'step':               step,
                'surprise':           surprise,
                'entropy':            entropy,
                'top_preset':         top,
                'top_prob':           self.priors[top],
                'n_opponent_actions': len(observed_actions),
                'match_count':        match_count,
                'priors':             dict(self.priors),
            }
            self.analytics_history.append(rec)
            return rec

        return None

    # ── Query ──────────────────────────────────────────────────────────────

    def get_expected_actions(self, state, opp_id: int,
                              threshold: float = 0.05) -> List[dict]:
        """Return expected opponent actions from presets with prob >= threshold.

        Deduplicated by (from_id, target_id); capped at MAX_VIRTUAL_FLEETS.
        """
        candidates: List[dict] = []
        for preset_name, prob in sorted(self.priors.items(),
                                        key=lambda kv: -kv[1]):
            if prob < threshold:
                continue
            for a in get_preset_actions(preset_name, state, opp_id):
                candidates.append({**a, '_prob': prob, '_preset': preset_name})

        # Keep highest-prob per (from_id, target_id)
        seen: Dict[tuple, dict] = {}
        for a in candidates:
            key = (a['from_id'], a['target_id'])
            if key not in seen or a['_prob'] > seen[key]['_prob']:
                seen[key] = a

        unique = sorted(seen.values(), key=lambda a: (-a['_prob'], -a['ships']))
        return unique[:MAX_VIRTUAL_FLEETS]

    # ── Info ───────────────────────────────────────────────────────────────

    def top_preset(self) -> str:
        return max(self.priors, key=lambda k: self.priors[k])

    def summary(self) -> str:
        parts = [f'{k}={v:.2f}' for k, v in
                 sorted(self.priors.items(), key=lambda kv: -kv[1])]
        return '  '.join(parts)

    def dump_analytics(self) -> List[dict]:
        """Return analytics_history (empty list if ANALYTICS_MODE was False)."""
        return list(self.analytics_history)


__all__ = [
    'ANALYTICS_MODE',
    'OpponentModelBayesian',
    'MAX_VIRTUAL_FLEETS',
    'DEFAULT_TEMPERATURE',
]

# ╔══════════════════════════════════════════════════╗
# ║  agent.py                                      ║
# ╚══════════════════════════════════════════════════╝

"""
Агент Orbit Wars.

Каждый ход:
  0. Проекция     — project_state резолвит уже летящие флоты (свои+чужие):
                    каждая планета видится в момент последнего прибытия.
  [РАННЯЯ ИГРА — MCTS]
  0.5. MCTS       — если step < EARLY_GAME_SWITCH и планет <= EARLY_MAX_PLANETS,
                    запускаем run_mcts в ПАР (Пространство Адекватных Решений).
                    Возвращает лучший Action напрямую, минуя зонирование/атаки.
  [ОБЫЧНАЯ ЛОГИКА]
  1. Зонирование  — compute_zones_from_state определяет зону каждой планеты
  2. Выбор целей  — easy_target / priority_target с лучшим priority
  3. Атаки        — best_attacks подбирает план (direct/multi_sync/pipeline) с
                    минимумом необходимых кораблей
  4. Наведение    — aim_angle (гибридный: fixpoint + sweep + safe-aim)
  5. Запуск       — атомарный: план исполняется ЦЕЛИКОМ либо не исполняется
                    вообще. Это гарантирует что любой запущенный флот меняет
                    стейт планеты-цели (один или совместно с sup-частью), либо
                    является передачей на нашу планету (pipeline sup→att).
"""

import math
import dataclasses as _dc
import os as _os
import sys as _sys

_PRIO_RECLASSIFY_THR_DEFAULT = PRIO_RECLASSIFY_THR
_TOTAL_STEPS = TOTAL_STEPS

# ── MCTS / ПАР (ранняя игра) ──────────────────────────────────────────────
try:
    from action_space import generate_actions, generate_opponent_actions
    from mcts import run_mcts, OpponentModel
    _MCTS_AVAILABLE = True
except ImportError as _mcts_import_err:
    _MCTS_AVAILABLE = False

# ── MCTS параметры (ранняя игра) ─────────────────────────────────────────
# MCTS включается когда: ход < EARLY_GAME_SWITCH И планет <= EARLY_MAX_PLANETS
# Чтобы отключить глобально — выставить USE_MCTS = False.
USE_MCTS             = True
EARLY_GAME_SWITCH    = 35    # первые N ходов → MCTS (затем обычная логика)
EARLY_MAX_PLANETS    = 20    # не более M планет на карте → MCTS (защита от больших карт)
MCTS_TIME_FRACTION   = 0.65  # доля оставшегося бюджета, отдаваемая MCTS

# Режим генерации ПАР: "none" | "v1_plain" | "v2_pair" | "v3_partial" | "v4_hybrid_net"
# Управляется переменной окружения ORBIT_MCTS_MODE (по умолчанию v1_plain).
MCTS_MODE = _os.environ.get("ORBIT_MCTS_MODE", "v1_plain")

# Глобальная модель противника (живёт между ходами одного матча).
# Инициализируется лениво при первом вызове _agent_impl.
_opponent_model   = None
_pending_partials = None   # для v3: список PartialAction в полёте

# ── Байесовский предсказатель поведения противника ───────────────────────
# Включается флагом USE_OPPONENT_PREDICTION.
# Добавляет виртуальные флоты противника в state_raw перед project_state,
# что заставляет swarm_plan учитывать ожидаемые атаки при планировании.
try:
    _BAYES_AVAILABLE = True
except ImportError:
    _BAYES_AVAILABLE = False
    _opp_bayes_mod   = None

USE_OPPONENT_PREDICTION = True   # выключить → False
BAYES_THRESHOLD         = 0.05   # минимальный вес пресета для добавления флота
_VIRTUAL_FLEET_ID_BASE  = -1000  # начало диапазона ID виртуальных флотов

# Состояние между ходами (для детекции новых флотов противника)
_bayes_model    = None   # OpponentModelBayesian
_prev_fleet_ids = None   # set[int] — fleet IDs на конец предыдущего хода
_prev_state_raw = None   # GameState — raw state предыдущего хода (для update)

# ── AgentSwarm switch ─────────────────────────────────────────────────
# True  → планы строим через swarm_plan (per-planet auction + redistribute)
# False → старая глобальная логика best_attacks (fallback / A-B сравнение)
USE_SWARM        = True
SWARM_WEIGHTS    = DEFAULT_WEIGHTS   # веса для 2-player

# Веса для 4-player FFA — оптимизированы отдельно (tune_4p_sweep1/2).
# Отличия от 2-player:
#   zone_urgency_weight=3.0  — сильнее гнать корабли rear→frontline (3 фронта)
#   neutral_garrison=0       — ng_3 помогал vs 3x sub2 но не vs смешанного состава
#
# match_runner4.py переопределяет эти веса через task['weights4'].
SWARM_WEIGHTS_4P = _dc.replace(DEFAULT_WEIGHTS,
    zone_urgency_weight  = 4.0,   # sweep1-2: urg4 > urg3 в FFA
    opp_strength_weight  = 0.5,   # sweep3: pure_osw0.5 лучший (rank=1.600 win=0.400)
    # garrison=0, neutral_garrison=0 — дефолты DEFAULT_WEIGHTS (оптимальны и в FFA
    # при наличии opp_strength: opp_bonus компенсирует нужду в гарнизоне)
)

TARGET_ZONES   = ('easy_target', 'priority_target')

# Кеш зон прошлого хода для [ZONE_FLIP] лога (per-match, сбрасывается на TURN 0)
_prev_zones: dict = {}  # pid → zone
RESERVE_ON_ATT = 0     # минимум кораблей оставить на атакере.
                       # ОБЯЗАН совпадать с attacks.RESERVE_ON_ATT — иначе
                       # планировщик считает available=att.ships, а исполнитель
                       # отправляет att.ships-1 → ничья на нейтрале (0 кораблей).
MIN_FIRE       = 3     # ниже — пуск не имеет смысла

# ── Адаптивная ширина рассмотрения целей ─────────────────────────────
# Логика: чем больше доля наших кораблей в общем пуле — тем больше
# параллельных атак можем себе позволить, и тем менее жёстко режем по
# priority. Когда отстаём — фокусируемся только на jackpot-целях.
#
#   ratio          = our_ships / sum_all_ships (по raw_planets, до проекции)
#   top_n          — сколько финальных планов попадает в FIRE-loop
#   priority_floor — отсечка по zones priority; цели ниже floor не идут
#                    в attacks (экономит работу планировщика)
#
# ВНИМАНИЕ: значения подобраны от руки и подлежат тюнингу по матчам.
# Меняй ТОЛЬКО этот массив — больше ничего трогать не надо.
ADAPTIVE_TIERS = [
    # (ratio_lo, top_n, priority_floor, label)
    (0.00,   3,  +0.5, 'behind'),     # сильно отстаём — только jackpot
    (0.25,   6,  -0.5, 'parity'),     # около паритета — текущее поведение
    (0.45,  10,  -1.5, 'ahead'),      # уверенно впереди — расширяемся
    (0.65,  16,  -3.0, 'dominate'),   # доминируем — почти всё в работу
]
TOP_ATTACKS_FALLBACK = 6
PRIO_FLOOR_FALLBACK  = -1e9


def _resource_ratio(state, player):
    """Доля наших кораблей в общем пуле — нормализованный индикатор силы.
    Берём raw_planets (фактическое сейчас), не projection — projection
    «потратит» какие-то наши корабли на атаках, что даст заниженный ratio.
    Кометы (если есть) сюда учитываются — у них в общем мало кораблей,
    погрешность пренебрежимая."""
    raw = getattr(state, 'raw_planets', state.planets)
    our   = sum(max(0, int(p.ships)) for p in raw if p.owner == player)
    total = sum(max(0, int(p.ships)) for p in raw)
    if total <= 0:
        return 0.0
    return our / total


def _adapt_swarm_weights(base, ctx, ratio):
    """Адаптирует SwarmWeights под текущий контекст (возвращает копию).

    Два механизма:

    1. Динамический transfer_floor / transfer_min_ships (исправление #9):
       Фиксированный floor=20 блокирует все трансферы когда у планет <30 кораблей
       (free = ships − 20 ≤ 10 < transfer_min_ships=15 → всегда нет).
       Масштабируем пороги с ratio = our_ships / total:
         ratio=0.10 → floor≈5,  min_ships≈5
         ratio=0.25 → floor≈10, min_ships≈8
         ratio=0.50 → floor≈17, min_ships≈13
         ratio≥0.80 → floor=20, min_ships=15  (полные пороги)

    2. Desperate mode (исправление #8):
       Когда stance=desperate (ships_ratio < 0.35) агент терял темп из-за
       тех же высоких порогов и нулевого risk_tolerance. В desperate:
         - Гарнизон минимален (floor=5): нечего беречь если проигрываем
         - risk_tolerance=1: принимаем атаки с margin ≥ −1 (было margin ≥ 0)
         - transfer_thresh снижен: отправляем даже небольшие подкрепления
         - transfer_min_ships=5: маленькие партии тоже идут в ход
    """
    stance = getattr(ctx, 'stance', 'attrition') if ctx else 'attrition'

    # ── Динамические transfer-пороги (масштаб с ratio) ────────────────────
    # clamp ratio в [0, 0.4]: выше 40% пороги уже максимальные
    t = min(1.0, ratio / 0.4)
    dyn_floor     = max(5.0,  base.transfer_floor      * t)
    dyn_min_ships = max(5,    int(base.transfer_min_ships * t))
    dyn_thresh    = max(2.0,  base.transfer_thresh      * t)

    w = _dc.replace(base,
                    transfer_floor=dyn_floor,
                    transfer_min_ships=dyn_min_ships,
                    transfer_thresh=dyn_thresh)

    # ── Desperate override (перекрывает динамику) ─────────────────────────
    if stance == 'desperate':
        w = _dc.replace(w,
                        transfer_floor=5.0,         # минимальный гарнизон
                        transfer_min_ships=5,        # любое ненулевое подкрепление
                        transfer_thresh=1.5,         # нижний порог score
                        risk_tolerance=1,            # margin ≥ -1 считаем победным
                        max_transfers_per_turn=3,    # чуть быстрее консолидация
                        )

    return w


def _adaptive_attack_params(ratio):
    """По ratio выбирает (top_n, priority_floor, label) из ADAPTIVE_TIERS."""
    chosen = (TOP_ATTACKS_FALLBACK, PRIO_FLOOR_FALLBACK, 'fallback')
    for ratio_lo, top_n, floor, label in ADAPTIVE_TIERS:
        if ratio >= ratio_lo:
            chosen = (top_n, floor, label)
    return chosen


def _raw(state, pid):
    """Текущая (raw) планета по id."""
    raw_planets = getattr(state, 'raw_planets', state.planets)
    for p in raw_planets:
        if p.id == pid:
            return p
    return None


# ── Классификация наших планет: raw × projection ──────────────────────────

def _classify_our_planets(state, player):
    """Возвращает (stable, doomed, incoming_friendly).

    stable             — raw=ours И projected=ours. Безопасный атакёр /
                         базовая планета для будущих ходов.
    doomed             — raw=ours, но projected=enemy. Нас захватывают.
                         НЕЛЬЗЯ выбирать как target (мы своих не атакуем).
                         НУЖНО посылать туда подкрепление (defense plan).
    incoming_friendly  — raw=enemy/neutral, но projected=ours. Наш флот её
                         уже берёт — не дублируем атаку.
    """
    raw_planets  = getattr(state, 'raw_planets', state.planets)
    proj_owner   = {p.id: p.owner for p in state.planets}

    stable, doomed, incoming = [], [], []
    for p in raw_planets:
        future = proj_owner.get(p.id, p.owner)
        if p.owner == player and future == player:
            stable.append(p)
        elif p.owner == player and future != player:
            doomed.append(p)
        elif p.owner != player and future == player:
            incoming.append(p)
    return stable, doomed, incoming


def _build_defense_plans(state, doomed_planets, stable_planets, player):
    """Для каждой обречённой нашей планеты ищем ближайшего стабильного
    отправителя с непрегораживаемым солнцем путём, формируем план direct
    sup→tgt в формате совместимом с _execute_plan_atomically.

    Цель плана = projected (enemy now) планета по id обречённой.
    `_defender_at` корректно посчитает её через projected.ships +
    projected.production * delta — это и есть число которое надо превзойти
    чтобы перезахватить (или сохранить, если флот успеет до врага).
    """
    if not doomed_planets:
        return []

    proj_by_id = {p.id: p for p in state.planets}
    plans = []

    for doomed in doomed_planets:
        proj = proj_by_id.get(doomed.id)
        if proj is None:
            continue
        # Нужно ships > defender. defender = ships_proj + prod * (eta - proj_at).
        # Прикинем по eta=рабочему фронту, потом уточним.
        proj_at_doomed = getattr(state, 'projected_at', {}).get(doomed.id, 0)
        # Базовая оценка дефицита — проекция уже хранит «прибыль врага» в
        # proj.ships (после боя). +overkill чтобы не оказаться на ничьей.
        base_def = float(proj.ships) + 2.0

        candidates = []
        for src in stable_planets:
            if src.id == doomed.id:
                continue
            # Путь должен быть чист от солнца
            if segment_hits_sun(src.x, src.y, doomed.x, doomed.y):
                continue
            avail = max(0, src.ships - 1)  # минимум 1 в гарнизоне
            if avail < 3:                  # MIN_FIRE
                continue

            # Уточняем eta + дефицит (production за время полёта)
            eta_init, _ = _rendezvous_eta(src, doomed,
                                           max(1, avail), state.omega)
            # production только для owned (не нейтрала)
            eff_prod = float(proj.production) if proj.owner != NEUTRAL_OWNER else 0.0
            def_eta = base_def + eff_prod * max(0.0, eta_init - proj_at_doomed)
            need = int(math.ceil(def_eta)) + 1   # +1 на ничью
            if avail < need:
                continue
            send = min(avail, need + 2)  # небольшой запас, не больше доступного
            eta, _ = _rendezvous_eta(src, doomed, max(1, send), state.omega)
            candidates.append((eta, src, send, def_eta))

        if not candidates:
            continue
        # Приоритет — самый быстрый сэйв (минимизировать время «беззащитности»)
        candidates.sort(key=lambda x: x[0])
        eta, src, send, def_eta = candidates[0]

        plans.append({
            'mode':       'direct',
            'sup_id':     None,
            'att_id':     src.id,
            'tgt_id':     doomed.id,
            'eta_sa':     0.0,
            'eta_at':     eta,
            't_total':    eta,
            'x_sup':      0,
            'x_att':      int(send),
            'prod_att':   float(src.production),
            'x_tgt':      int(proj.ships),
            'prod_tgt':   float(proj.production),
            'strike':     float(send),
            'defender':   float(def_eta),
            'incoming':   0.0,
            'needed':     int(math.ceil(def_eta)) + 1,
            'available':  src.ships - 1,
            'slack':      int(src.ships - 1 - send),
            'margin':     float(send) - float(def_eta),
            'success':    True,
            'relay_reason': None,
            'is_defense': True,    # маркер для отладки
        })

    return plans


def _projected(state, pid):
    """Проецированная планета (для геометрии используем тот же x,y что и raw)."""
    for p in state.planets:
        if p.id == pid:
            return p
    return None


def _reserve_part(state, planet_id, n_ships, tentative):
    """
    Резервирует n_ships на planet_id с учётом уже зарезервированного.
    Возвращает фактически выделенное число кораблей либо None если part невозможен.
    Не модифицирует tentative (только проверяет).

    Важно: НЕ бампим x до MIN_FIRE. План (eval_direct/eval_pipe/eval_multi_sync)
    считает РОВНО нужное количество кораблей. Если план говорит x_sup=1 — значит
    1 корабль это калиброванный реинфорс (а не «мусор»). Бампинг до 3 раздувал
    каждый pipeline/multi_sync supply до MIN_FIRE и просто терял корабли.

    MIN_FIRE используется только как порог «маленькие планеты не запускают
    флоты» (avail < MIN_FIRE → None).
    """
    src = _raw(state, planet_id)
    if src is None:
        return None
    already = tentative.get(planet_id, 0)
    avail   = src.ships - already - RESERVE_ON_ATT
    if avail < MIN_FIRE:
        return None
    n_req = max(1, int(n_ships))
    return min(n_req, int(avail))


def _plan_parts(plan):
    """
    Раскладывает план в упорядоченный список частей (planet_id, aim_target_id, x).
    Для multi_sync: sup → tgt, att → tgt   (оба должны лететь одновременно).
    Для pipeline:  sup → att, att → tgt    (sup передаёт нашей планете att).
    Для direct:    att → tgt.
    """
    parts = []
    if plan.get('sup_id') is not None and plan.get('x_sup', 0) > 0:
        # sup-цель: для pipeline это наша промежуточная планета att,
        # для multi_sync — конечная цель tgt.
        sup_target = plan['att_id'] if plan['mode'] == 'pipeline' else plan['tgt_id']
        parts.append((plan['sup_id'], sup_target, int(plan['x_sup'])))
    if plan.get('x_att', 0) > 0:
        parts.append((plan['att_id'], plan['tgt_id'], int(plan['x_att'])))
    return parts


def _aim_and_verify(src, tgt, ships, state):
    """
    Считает угол через aim_hybrid и проверяет покадровой симуляцией
    (simulate_launch — 1-в-1 с движком), что флот реально попадает
    в нужную планету. Возвращает (angle, sim_eta) или None если
    ни одного годного угла нет.

    Проверяем основной угол aim_hybrid и до 6 деформаций ±epsilon, чтобы
    компенсировать редкие касательные промахи / дискретные эффекты. Это
    напрямую закрывает «иногда промахиваемся между удалёнными
    стационарными» — мы не запускаем угол, который simulate говорит, что
    промах/OOB/sun/чужая планета.
    """
    initial_by_id = state.initial_by_id() if callable(state.initial_by_id) else state.initial_by_id
    res = aim_hybrid(src, tgt, ships, state.planets, initial_by_id, state.omega)
    if res is None:
        return None
    base_angle = res[0]

    candidates = [base_angle]
    for d in (0.0025, -0.0025, 0.006, -0.006, 0.012, -0.012):
        candidates.append(base_angle + d)

    for a in candidates:
        sim = simulate_launch(src, a, ships, state.planets, initial_by_id, state.omega)
        if sim['outcome'] == 'HIT' and sim['planet_id'] == tgt.id:
            return a, int(sim['turn'])
    return None


def _defender_at(state, tgt, eta):
    """Оценка кораблей защитника в момент прибытия флота (turn = eta).
    Учитывает projected_at (если состояние спроецировано) и факт, что
    нейтральные планеты НЕ производят корабли."""
    proj_at = getattr(state, 'projected_at', {}).get(tgt.id, 0)
    if tgt.owner == NEUTRAL_OWNER:
        return float(tgt.ships)
    return float(tgt.ships) + float(tgt.production) * max(0.0, eta - proj_at)


def _execute_plan_atomically(state, plan, committed, risk_tolerance=0):
    """
    Атомарный запуск плана: либо все части плана успешно зарезервированы,
    углы посчитаны и подтверждены simulate_launch, либо план полностью
    отбрасывается (ничего не запускается).

    `risk_tolerance` — должен совпадать со значением переданным в swarm_plan.
    При risk_tolerance >= 1 снижаем eta_drift_buffer для enemy-планет с 1 до 0:
    план сгенерирован с margin ≥ −1 (overkill=1), поэтому execute проверяет
    ту же планку — иначе marginal-план всегда блокируется здесь же.

    Это гарантирует требование: любой запуск либо меняет стейт цели сам
    (direct), либо совместно с другим утверждённым запуском (multi_sync),
    либо является передачей на нашу же планету для будущего изменения
    (pipeline). Не запускаем «огрызки» планов, которые ничего не меняют.

    Дополнительно: для att-части (которая по плану должна зайти в цель
    и победить) проверяем что ships > defender_at(eta_actual). Если
    реальная eta стрельбы дрейфует и наш план уже не выигрывает — план
    отбрасываем целиком, а не сливаем мелочь, которую враг съест.

    Возвращает (moves, reason). При успехе reason='ok'. При отказе
    moves=[] и reason — короткое объяснение (для отладочного лога).
    """
    parts = _plan_parts(plan)
    if not parts:
        return [], 'empty_plan'

    # Стадия 1: пробное резервирование. Не трогаем committed пока не
    # убедились что ВСЕ части плана влезают.
    tentative = dict(committed)
    confirmed = []
    for planet_id, aim_target_id, n_req in parts:
        n_actual = _reserve_part(state, planet_id, n_req, tentative)
        if n_actual is None:
            src = _raw(state, planet_id)
            avail = (src.ships - tentative.get(planet_id, 0) - RESERVE_ON_ATT) if src else None
            return [], (f'reserve_failed: pid={planet_id} need={n_req} '
                        f'avail={avail} (after committed={tentative.get(planet_id, 0)})')
        tentative[planet_id] = tentative.get(planet_id, 0) + n_actual
        confirmed.append((planet_id, aim_target_id, n_actual))

    # Стадия 2: aim + simulate-verify для каждой части.
    moves = []
    tgt_id = plan.get('tgt_id')
    is_pipeline_sup_part = (plan.get('mode') == 'pipeline'
                            and plan.get('sup_id') is not None)
    for planet_id, aim_target_id, n_actual in confirmed:
        src = _raw(state, planet_id)
        tgt = _projected(state, aim_target_id)
        if src is None or tgt is None:
            return [], f'planet_not_found: src={planet_id} tgt={aim_target_id}'
        verified = _aim_and_verify(src, tgt, n_actual, state)
        if verified is None:
            return [], (f'aim_verify_failed: src={planet_id} aim_tgt={aim_target_id} '
                        f'ships={n_actual} (no angle simulates as HIT)')
        angle, sim_eta = verified

        is_att_to_tgt = (aim_target_id == tgt_id and planet_id == plan['att_id']
                         and not is_pipeline_sup_part
                         and plan.get('mode') == 'direct')
        if is_att_to_tgt:
            defender_actual = _defender_at(state, tgt, sim_eta)
            # Нейтралы: производства нет, defender детерминирован — строгий >
            # достаточен, лишний буфер только выбрасывал бы корабли впустую.
            # Enemy-owned: sim_eta чуть длиннее расчётного eta_at плана, за это
            # время defender успевает подрасти на +prod. Буфер +1 закрывает
            # off-by-one когда gap = 0 (77 случаев в логе).
            # risk_tolerance>=1: plan was generated with overkill=1 for enemy planets,
            # so execute uses the same buffer (0) to stay consistent.
            eta_drift_buffer = 0 if (tgt.owner == NEUTRAL_OWNER or risk_tolerance >= 1) else 1
            if n_actual <= defender_actual + eta_drift_buffer:
                return [], (f'wont_win: ships={n_actual} <= '
                            f'defender_at_eta={defender_actual:.1f}+buf={eta_drift_buffer} '
                            f'(sim_eta={sim_eta}, tgt={tgt_id})')

        moves.append([planet_id, angle, n_actual])

    committed.update(tentative)
    return moves, 'ok'


def _build_virtual_fleets(expected_actions, state_raw, opp_id: int) -> list:
    """Создать список виртуальных Fleet-объектов из предсказанных действий.

    Угол вычисляется как atan2(tgt - src) — простое приближение, достаточное
    для project_state (точный aim_hybrid здесь избыточен и дорог).
    Виртуальные флоты получают отрицательные ID чтобы не конфликтовать
    с реальными.
    """
    import math as _m
    _Fleet = Fleet
    planets_by_id = {p.id: p for p in state_raw.planets}
    virtual = []
    vid = _VIRTUAL_FLEET_ID_BASE
    for a in expected_actions:
        src = planets_by_id.get(a['from_id'])
        tgt = planets_by_id.get(a['target_id'])
        if src is None or tgt is None:
            continue
        angle = _m.atan2(tgt.y - src.y, tgt.x - src.x)
        virtual.append(_Fleet(
            id=vid,
            owner=opp_id,
            x=float(src.x),
            y=float(src.y),
            angle=angle,
            from_planet_id=src.id,
            ships=max(1, int(a['ships'])),
        ))
        vid -= 1
    return virtual


def _extract_new_opp_fleets(state_raw, prev_fleet_ids: set,
                             opp_id: int,
                             planets_by_id: dict) -> list:
    """Найти флоты противника запущенные в прошлый ход.

    Новый флот = fleet.owner == opp_id AND fleet.id не был в prev_fleet_ids.
    Для каждого нового флота вычисляем (from_id, target_id) через
    simulate_fleet_target, затем определяем action_type.
    """
    _sft = simulate_fleet_target
    _NO = NEUTRAL_OWNER
    new_fleets = [
        f for f in state_raw.fleets
        if f.owner == opp_id and f.id not in prev_fleet_ids
    ]
    actions = []
    for f in new_fleets:
        tgt_id, _ = _sft(f, state_raw.planets, state_raw.omega)
        if tgt_id is None:
            continue
        tgt = planets_by_id.get(tgt_id)
        if tgt is None:
            continue
        if tgt.owner == opp_id:
            atype = 'reinforce'
        elif tgt.owner == _NO:
            atype = 'capture_neutral'
        else:
            atype = 'attack_enemy'
        actions.append({
            'from_id':     getattr(f, 'from_planet_id', -1),
            'target_id':   tgt_id,
            'ships':       int(f.ships),
            'action_type': atype,
        })
    return actions


def _count_players(state_raw):
    """Считает число активных игроков из планет и флотов."""
    owners = set()
    for p in getattr(state_raw, 'planets', []):
        if getattr(p, 'owner', -1) != -1:
            owners.add(p.owner)
    for f in getattr(state_raw, 'fleets', []):
        owners.add(getattr(f, 'owner', -1))
    owners.discard(-1)
    return max(2, len(owners))


def _agent_impl(obs, deadline=None):
    player    = obs.get('player', 0)
    state_raw = GameState.from_kaggle_obs(obs)

    # Выбираем веса в зависимости от режима (2-player vs FFA)
    _n_players      = _count_players(state_raw)
    _is_ffa         = _n_players >= 4
    _active_weights = SWARM_WEIGHTS_4P if _is_ffa else SWARM_WEIGHTS

    # _step нужен и байесу (step=) и MCTS-логу, поэтому извлекаем сразу.
    _step = (obs.get('step') if isinstance(obs.get('step'), int) else
             obs.get('stepNumber') if isinstance(obs.get('stepNumber'), int) else
             getattr(state_raw, 'step', -1))

    # ── Байесовский предсказатель (обновление + добавление виртуальных флотов) ──
    global _bayes_model, _prev_fleet_ids, _prev_state_raw
    state_for_projection = state_raw   # может быть заменено расширенным

    if USE_OPPONENT_PREDICTION and _BAYES_AVAILABLE:
        try:
            opp_id = (player + 1) % max(2, getattr(state_raw, 'n_players', 2))
            planets_by_id = {p.id: p for p in state_raw.planets}

            # Включаем аналитику если включён debug-лог (zero-cost в бою)
            if _opp_bayes_mod is not None:
                _opp_bayes_mod.ANALYTICS_MODE = _dbg.enabled()

            # Ленивая инициализация
            if _bayes_model is None:
                _bayes_model = OpponentModelBayesian()

            # Шаг 1: обновить модель по реально наблюдённым флотам противника
            if _prev_fleet_ids is not None and _prev_state_raw is not None:
                observed = _extract_new_opp_fleets(
                    state_raw, _prev_fleet_ids, opp_id, planets_by_id
                )
                metrics = _bayes_model.update(
                    observed, _prev_state_raw, opp_id, step=_step
                )
                # Логируем аналитику если включена (metrics != None только при ANALYTICS_MODE)
                if metrics is not None and _dbg.enabled():
                    try:
                        _dbg._w(
                            f'[BAYES/update]  step={metrics["step"]}'
                            f'  surprise={metrics["surprise"]:.3f}'
                            f'  entropy={metrics["entropy"]:.3f}'
                            f'  top={metrics["top_preset"]}({metrics["top_prob"]:.2f})'
                            f'  obs={metrics["n_opponent_actions"]}'
                            f'  matched={metrics["match_count"]}'
                        )
                        # Метрика непонимания: confusion% = entropy/ln(5)*100
                        # 100% = полная неопределённость, 0% = уверен в пресете
                        _dbg.log_bayes_confusion(metrics['entropy'])
                    except Exception:
                        pass

            # Шаг 2: получить предсказанные действия и создать виртуальные флоты
            expected = _bayes_model.get_expected_actions(
                state_raw, opp_id, threshold=BAYES_THRESHOLD
            )
            if expected:
                virtual = _build_virtual_fleets(expected, state_raw, opp_id)
                if virtual:
                    augmented_fleets = list(state_raw.fleets) + virtual
                    _GS = GameState
                    state_for_projection = _GS(
                        planets=list(state_raw.planets),
                        fleets=augmented_fleets,
                        omega=state_raw.omega,
                        step=state_raw.step,
                        initial_planets=list(state_raw._initial_planets.values()),
                        n_players=getattr(state_raw, 'n_players', 2),
                        comet_ids=set(getattr(state_raw, 'comet_ids', set()) or set()),
                    )
            if _dbg.enabled():
                try:
                    _dbg._w(
                        f'[BAYES/predict]  top={_bayes_model.top_preset()}'
                        f'  expected={len(expected)}'
                        f'  virtual={len(virtual) if expected else 0}'
                        f'  dist={_bayes_model.summary()}'
                    )
                except Exception:
                    pass

        except Exception as _bayes_err:
            try: _dbg.log_error('bayes', _bayes_err)
            except Exception: pass

    # 0. Проекция: резолвим все летящие флоты (свои и чужие + виртуальные).
    #    После этого state.planets — состояние на момент последнего события.
    #    player передаётся, чтобы projection знал чьи кометы «отскакивают»
    #    обратно на ближайшую нашу планету (см. project_state docstring).
    _raw_n_fleets = len(getattr(state_for_projection, 'fleets', []))
    state = project_state(state_for_projection, horizon=PROJECTION_HORIZON, player=player)

    _dbg.begin_turn(_step, player, len(state.planets), _raw_n_fleets)
    _dbg.log_fleets(state.fleets, player)
    if _step == 0:
        global _prev_zones
        _prev_zones = {}

    # ── 0.5. MCTS (ранняя игра) ───────────────────────────────────────────────
    # Условие включения: USE_MCTS, модуль доступен, ранняя фаза, карта небольшая.
    # При успехе — возвращаем ходы прямо из MCTS, минуя весь основной пайплайн.
    # При любой ошибке — молча падаем сквозь в обычную логику (fail-safe).
    global _opponent_model, _pending_partials
    if (USE_MCTS and _MCTS_AVAILABLE and MCTS_MODE != "none"
            and _step >= 0 and _step < EARLY_GAME_SWITCH
            and len(state.planets) <= EARLY_MAX_PLANETS):
        try:
            import time as _time_mcts
            # Ленивая инициализация модели противника (сбрасывается при старте матча)
            if _opponent_model is None:
                _opponent_model = OpponentModel()

            # Генерируем ПАР согласно MCTS_MODE
            opp_id = (player + 1) % max(2, getattr(state, 'n_players', 2))

            if MCTS_MODE == "v2_pair":
                try:
                    from action_space_v2 import generate_actions_v2
                    my_actions = generate_actions_v2(state, player)
                except ImportError:
                    my_actions = generate_actions(state, player)

            elif MCTS_MODE == "v3_partial":
                try:
                    from action_space_v3 import generate_actions_v3, PartialAction
                    my_actions = generate_actions_v3(
                        state, player,
                        pending_partials=_pending_partials or [],
                    )
                except ImportError:
                    my_actions = generate_actions(state, player)

            elif MCTS_MODE == "v4_hybrid_net":
                try:
                    from action_space_v4 import generate_actions_v4
                    my_actions = generate_actions_v4(state, player)
                except ImportError:
                    my_actions = generate_actions(state, player)

            else:  # v1_plain (default)
                my_actions = generate_actions(state, player)

            opp_actions = generate_opponent_actions(state, opp_id)

            # Бюджет времени: доля от остатка до дедлайна
            if deadline is not None:
                remaining   = deadline - _time_mcts.perf_counter()
                mcts_budget = max(0.05, remaining * MCTS_TIME_FRACTION)
            else:
                mcts_budget = 0.35

            if _dbg.enabled():
                try:
                    _d._w(f'[MCTS/{MCTS_MODE}]  step={_step}'
                          f'  my_actions={len(my_actions)}'
                          f'  opp_actions={len(opp_actions)}'
                          f'  budget={mcts_budget:.3f}s')
                except Exception:
                    pass

            best_action = run_mcts(
                state, my_actions, opp_actions,
                _opponent_model, player,
                time_budget=mcts_budget,
            )

            if best_action is not None:
                if _dbg.enabled():
                    try:
                        _d._w(f'[MCTS/{MCTS_MODE}]  best={best_action}')
                    except Exception:
                        pass
                # v3: track PartialAction for second wave next turn
                if MCTS_MODE == "v3_partial":
                    try:
                        from action_space_v3 import PartialAction as _PA
                        if isinstance(best_action, _PA):
                            if _pending_partials is None:
                                _pending_partials = []
                            _pending_partials.append(best_action)
                    except Exception:
                        pass
                _dbg.end_turn()
                return best_action.to_moves()
            # Если MCTS не нашёл хода — продолжаем в обычный пайплайн
            if _dbg.enabled():
                try:
                    _d._w(f'[MCTS/{MCTS_MODE}]  no action found, falling back')
                except Exception:
                    pass
        except Exception as _mcts_err:
            try: _dbg.log_error('mcts', _mcts_err)
            except Exception: pass
            # Любая ошибка → продолжаем в обычный пайплайн

    # Game Understanding Layer — глобальное «понимание» текущего хода.
    # Пока просто логируем для проверки; интегрировать в scoring — следующий шаг.
    try:
        ctx = compute_context(state, player)
        if _dbg.enabled():
            try:
                _d._w(f'[GUL]  {context_summary(ctx)}')
            except Exception:
                pass
    except Exception as _e:
        ctx = None
        try: _dbg.log_error('compute_context', _e)
        except Exception: pass

    # 0.5. Адаптивная ширина: по ratio = our_ships/total выбираем top_n
    #      финальных планов и priority_floor (отсечку по приоритету целей).
    ratio = _resource_ratio(state, player)
    top_n, prio_floor, tier_label = _adaptive_attack_params(ratio)
    if _dbg.enabled():
        try:
            _d._w(f'[ADAPTIVE]  ratio={ratio:.3f}  tier={tier_label}  '
                  f'top_n={top_n}  prio_floor={prio_floor:+.2f}')
        except Exception:
            pass

    # 1. Зонирование (на спроецированном состоянии)
    # our_total_prod — для production-scarcity boost (ранняя игра).
    # Считаем по raw_planets (до проекции), чтобы отражать реальный прод сейчас.
    _raw_pl_early = getattr(state, 'raw_planets', state.planets)
    our_total_prod = sum(
        float(p.production) for p in _raw_pl_early if p.owner == player
    )
    # Вычисляем stage-aware порог priority-reclassify: линейная интерполяция
    # от prio_reclassify_thr (step=0) до prio_reclassify_thr_late (step=TOTAL).
    # Если оба одинаковые (дефолт) — статичный порог, интерполяции нет.
    # Читаем из активных весов (_active_weights = 2p или 4p в зависимости от режима).
    _prio_thr_early = float(getattr(_active_weights, 'prio_reclassify_thr',
                                    _PRIO_RECLASSIFY_THR_DEFAULT))
    _prio_thr_late  = float(getattr(_active_weights, 'prio_reclassify_thr_late',
                                    _prio_thr_early))
    _phase_prio     = min(1.0, max(0.0, float(_step) / float(_TOTAL_STEPS))) if _step >= 0 else 0.0
    _eff_prio_thr   = _prio_thr_early + (_prio_thr_late - _prio_thr_early) * _phase_prio

    try:
        df, _ = compute_zones_from_state(state, player=player,
                                         our_total_prod=our_total_prod,
                                         prio_reclassify_thr=_eff_prio_thr)
    except Exception as e:
        _dbg.log_error('compute_zones_from_state', e)
        _dbg.end_turn()
        return []

    _dbg.log_zones(df, projected_at=getattr(state, 'projected_at', {}))
    zone_lookup = {int(r['pid']): r['zone'] for _, r in df.iterrows()}
    _dbg.log_zone_flips(_prev_zones, zone_lookup)
    _prev_zones = dict(zone_lookup)   # обновляем кеш для следующего хода

    if _dbg.enabled():
        _TS = TOTAL_STEPS
        _phase = min(1.0, float(getattr(state, 'step', 0) or 0) / float(_TS))
        _fade  = max(0.0, 1.0 - _phase / EARLY_PHASE_THR)
        _boost = (SCARCITY_K / (1.0 + our_total_prod)) * _fade
        if _boost > 0.01:
            _dbg._w(f'[SCARCITY]  our_prod={our_total_prod:.0f}  phase={_phase:.2f}'
                    f'  fade={_fade:.2f}  prod_boost=×{1+_boost:.2f}'
                    f'  eff_prod_w={0.5*(1+_boost):.3f}')

    # 1.5. Классификация наших планет по комбинации raw × projection:
    #      stable / doomed / incoming_friendly.
    #      doomed   — наши сейчас, но проекция говорит что потеряем →
    #                 фильтруем из targets, отправляем туда defense.
    #      incoming — не наши сейчас, но станут наши по проекции →
    #                 фильтруем из targets (наш флот уже летит).
    raw_planets   = getattr(state, 'raw_planets', state.planets)
    raw_owner     = {p.id: p.owner for p in raw_planets}
    stable_ours, doomed_ours, incoming_ours = _classify_our_planets(state, player)
    doomed_ids    = {p.id for p in doomed_ours}
    incoming_ids  = {p.id for p in incoming_ours}

    if _dbg.enabled():
        try:
            _d._w(f'[CLASSIFY]  stable={len(stable_ours)}  '
                  f'doomed={[p.id for p in doomed_ours]}  '
                  f'incoming={[p.id for p in incoming_ours]}')
        except Exception:
            pass

    # 2. Выбор целей по зоне и priority + адаптивная отсечка по floor.
    #    floor применяется ДО best_attacks → меньше работы планировщику.
    #    КРОМЕ ТОГО: фильтруем планеты которые УЖЕ наши в raw (doomed)
    #    или СТАНУТ нашими через проекцию (incoming) — это не цели атаки.
    def _is_real_target(pid):
        # raw=ours → точно не атакуем (даже если projection=enemy = doomed)
        if raw_owner.get(pid, -1) == player:
            return False
        # incoming friendly → атаковать дублирующе нет смысла
        if pid in incoming_ids:
            return False
        return True

    tgt_df = (df[df['zone'].isin(TARGET_ZONES) & (df['priority'] >= prio_floor)]
                .sort_values('priority', ascending=False))
    target_ids = [pid for pid in tgt_df['pid'].tolist() if _is_real_target(pid)]
    targets    = [p for p in state.planets if p.id in target_ids]
    target_source = f'zones (floor={prio_floor:+.2f})'

    if not targets:
        # Fallback: если по floor вообще никого — пробуем без floor (только зоны).
        # Это страхует от ситуации «все priority < floor» в раннем/позднем игре.
        tgt_df = (df[df['zone'].isin(TARGET_ZONES)]
                    .sort_values('priority', ascending=False))
        target_ids = [pid for pid in tgt_df['pid'].tolist() if _is_real_target(pid)]
        targets    = [p for p in state.planets if p.id in target_ids]
        target_source = 'zones (floor relaxed)'

    if not targets:
        targets = [p for p in state.planets
                   if p.owner != player and _is_real_target(p.id)]
        target_ids = [p.id for p in targets]
        target_source = 'fallback_all_non_ours'

    _dbg.log_targets(target_ids, zones_df=df)
    if _dbg.enabled():
        try:
            _d._w(f'   (target_source={target_source})')
        except Exception:
            pass

    # 3a. (опционально) — все кандидатные планы по каждой цели для ручной отладки
    if _dbg.plans_all_enabled():
        try:
            raw_planets = getattr(state, 'raw_planets', state.planets)
            ours = [p for p in raw_planets if p.owner == player]
            for tgt in targets:
                cand = all_plans(state, tgt, ours, horizon=ATTACK_HORIZON, player=player)
                if cand:
                    _dbg.log_plans(cand, label=f'PLANS_ALL tgt={tgt.id}')
        except Exception as e:
            _dbg.log_error('all_plans', e)

    # 2.5. Defense plans: для doomed-наших планет ищем стабильного отправителя.
    #      Эти планы выполняются ПЕРЕД атакой (приоритет самосохранению).
    defense_plans = _build_defense_plans(state, doomed_ours, stable_ours, player)
    if _dbg.enabled() and defense_plans:
        _dbg.log_plans(defense_plans, label='DEFENSE_PLANS')

    # 3. Атаки.
    #    USE_SWARM=True → swarm_plan: per-planet auction + redistribute.
    #    Иначе fallback на best_attacks (глобальный priority).
    if USE_SWARM:
        # Адаптируем веса под текущий контекст (stance + ratio).
        # _adapt_swarm_weights возвращает копию SWARM_WEIGHTS с динамичными
        # transfer-порогами и desperate-режимом (см. подробный комментарий).
        eff_weights = _adapt_swarm_weights(_active_weights, ctx, ratio)
        if _dbg.enabled():
            try:
                _stance = getattr(ctx, 'stance', '?') if ctx else '?'
                _fl     = eff_weights.transfer_floor
                _ms     = eff_weights.transfer_min_ships
                _rt     = eff_weights.risk_tolerance
                _mx_tr  = eff_weights.max_transfers_per_turn
                _dbg._w(f'[WEIGHTS]  stance={_stance}  ratio={ratio:.3f}'
                        f'  floor={_fl:.1f}  min_ships={_ms}'
                        f'  risk_tol={_rt}  max_tr={_mx_tr}')
            except Exception:
                pass

        # priority_lookup для tiebreaker'а в аукционе (из zones)
        prio_lookup = {int(r.pid): float(r.priority)
                       for _, r in df.iterrows()
                       if 'priority' in df.columns}
        attack_plans, swarm_dbg = swarm_plan(
            state, player, targets,
            weights=eff_weights, priority_lookup=prio_lookup,
            zone_lookup=zone_lookup,
            max_targets=12,         # cap: на больших картах не успеваем за бюджет kaggle
            deadline=deadline,      # передаём из обёртки agent()
        )
        if _dbg.enabled():
            try:
                _dbg._w(f'[SWARM]  cand={swarm_dbg["n_candidates"]} '
                        f'committed={swarm_dbg["n_committed"]} '
                        f'transfers={swarm_dbg["n_transfers"]} '
                        f'unfunded={swarm_dbg["n_unfunded"]}')
                top_stress = sorted(swarm_dbg['stress'].items(),
                                    key=lambda x: -x[1])[:5]
                _dbg._w(f'[SWARM]  top stress: '
                        + ' '.join(f'p{pid}={s:.1f}' for pid, s in top_stress))
                # Idle-деньги: сколько кораблей осталось на каждой планете
                _dbg.log_remaining(swarm_dbg.get('remaining', {}), zone_lookup)
                # Детали transfers: откуда, куда, градиент
                _dbg.log_transfers(attack_plans, zone_lookup)
            except Exception:
                pass
    else:
        attack_plans = best_attacks(state, player, targets, top_n=top_n)
    _dbg.log_plans(attack_plans, label='PLANS_TOP')

    # 3.5. Композиция: defense_plans → attack_plans. Сначала спасаемся,
    #      потом атакуем. Резервирование кораблей (committed dict)
    #      гарантирует что defense получает приоритет на источниках.
    plans = defense_plans + attack_plans

    # 4. Сборка ходов: каждый план атомарно (success-only).
    moves     = []
    committed = {}

    for idx, plan in enumerate(plans):
        if not plan.get('success'):
            _dbg.log_decision(idx, plan, 'SKIP', 'plan.success=false (won\'t change tgt state)')
            continue
        sub, reason = _execute_plan_atomically(state, plan, committed,
                                                risk_tolerance=eff_weights.risk_tolerance
                                                if USE_SWARM else 0)
        if sub:
            _dbg.log_decision(idx, plan, 'FIRE',
                              f'parts={len(sub)} '
                              + ' '.join(f'[src={m[0]} ships={m[2]}]' for m in sub))
            moves.extend(sub)
        else:
            _dbg.log_decision(idx, plan, 'SKIP', reason)

    _dbg.log_moves(moves)
    _dbg.end_turn()

    # ── Сохраняем state для байесовского обновления на следующем ходу ─────
    if USE_OPPONENT_PREDICTION and _BAYES_AVAILABLE:
        try:
            _prev_fleet_ids = {f.id for f in state_raw.fleets}
            _prev_state_raw = state_raw
        except Exception:
            pass

    # ── Дамп аналитики байеса в конце матча ──────────────────────────────
    # Только в debug-режиме. Kaggle-бои: _dbg.enabled() == False → пропуск.
    if (_dbg.enabled() and _BAYES_AVAILABLE
            and _bayes_model is not None
            and _bayes_model.analytics_history):
        try:
            is_last = (_step >= 498)   # kaggle матч обычно 500 ходов
            if is_last:
                import json as _json
                history = _bayes_model.dump_analytics()
                _dbg._w(f'[BAYES/dump]  turns={len(history)}')
                for rec in history:
                    _dbg._w('[BAYES/H] ' + _json.dumps(rec, separators=(',', ':')))
        except Exception:
            pass

    return moves


# ── kaggle-safe entrypoint ──────────────────────────────────────────────
# Никогда не падаем. Любая ошибка → пустой список ходов (агент просто стоит,
# но матч продолжается, не дисквал). Также soft deadline по wall-clock,
# чтобы swarm_plan корректно прервался при cap превышен.
def agent(obs, config=None):
    import time as _time
    # kaggle обычно даёт 1 секунду на step; берём 0.85 как soft бюджет
    act_timeout = 1.0
    if config is not None:
        try:
            act_timeout = float(config.get('actTimeout', 1.0))
        except Exception:
            pass
    deadline = _time.perf_counter() + max(0.4, min(act_timeout * 0.85, 5.0))
    try:
        return _agent_impl(obs, deadline=deadline)
    except Exception as _e:
        try:
            _dbg.log_error('agent', _e)
        except Exception:
            pass
        return []


# ── _dbg namespace (agent_debug инлайнен, восстанавливаем ссылку) ──
import types as _types_dbg
_dbg = _types_dbg.SimpleNamespace(
    enabled=enabled,
    reset=reset,
    begin_turn=begin_turn,
    log_zones=log_zones,
    log_targets=log_targets,
    log_plans=log_plans,
    log_decision=log_decision,
    log_moves=log_moves,
    log_fleets=log_fleets,
    log_error=log_error,
    end_turn=end_turn,
    plans_all_enabled=plans_all_enabled,
    log_remaining=log_remaining,
    log_transfers=log_transfers,
    log_zone_flips=log_zone_flips,
    log_bayes_confusion=log_bayes_confusion,
    reset_decisions=reset_decisions,
    _w=_w,
    _safe=_safe,
)
