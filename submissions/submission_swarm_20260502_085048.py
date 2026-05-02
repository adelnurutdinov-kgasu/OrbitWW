# Auto-generated submission — 2026-05-02T08:50:48
# Source: agent_bundle_swarm/ + winner_seed20 веса
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

if __name__ == "__main__":
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
                               # ОБЯЗАН совпадать с agent.RESERVE_ON_ATT — иначе
                               # планировщик считает доступными att.ships, а
                               # исполнитель отправляет att.ships-1 → план с
                               # margin=+1 в реальности проигрывает defender'у
                               # на 1 корабль и планета остаётся нейтральной
                               # с 1 ship на борту (см. бандл turn=16 на
                               # multi_sync 4→0→26).

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

def eval_direct(state, att, tgt, incoming=0.0, risk=0):
    """Прямая атака att → tgt. x_att = минимум для победы (с учётом incoming)."""
    if _seg_blocked(att, tgt):
        return None

    eta, defender, needed = _required_strike(state, att, tgt, state.omega, att.ships, incoming, risk=risk)
    available = att.ships - RESERVE_ON_ATT
    success   = needed <= available

    x_att = needed if success else max(1, available)

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

def all_plans(state, tgt, ours, horizon=ATTACK_HORIZON, player=0, risk=0):
    """Все возможные планы атаки на tgt.

    Если state — ProjectedState, incoming уже учтён в tgt.ships, поэтому
    дополнительно его не считаем. Иначе используем friendly_incoming как раньше.
    """
    if hasattr(state, 'projected_at'):
        incoming = 0.0
    else:
        incoming, _ = friendly_incoming(state, tgt, horizon, player)

    plans = []
    for att in ours:
        if att.id == tgt.id:
            continue
        d = eval_direct(state, att, tgt, incoming=incoming, risk=risk)
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
    'n_cross':         +0.9,
    'late_aggression':  -0.2,   # для своих планет смысла не несёт
}
W_TARGETS = {
    'area_inv':        +0.4,
    'wnn_close_res':   +0.8,    # tournament-winner (turn 2026-04-28): был +0.8
    'mean_dist_all':   -0.2,    # tournament-winner (turn 2026-04-28): был -0.6
    'prod':            +0.5,
    'ships':           -0.5,    # tournament-winner (turn 2026-04-28): был -0.7
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


def compute_zones(df, w_ours=W_OURS, w_targets=W_TARGETS, player=0, step=0):
    """
    Принимает DataFrame с колонками ZONE_FEATURES + 'owner' + 'pid'.
    Возвращает (df_extended, Z_scores).

    `step` — ход матча. Используется как phase-множитель ТОЛЬКО для
    late_aggression-фичи: эффективный вес = phase · w_targets['late_aggression'].
    Применяется на уровне веса (а не самой фичи), чтобы z-score не сокращал
    общий phase-множитель — иначе разница между early и late игрой стиралась.
    """
    out = df.copy()
    Z = pd.DataFrame({m: _zscore(out[m]) for m in ZONE_FEATURES}, index=out.index)
    out['priority'] = 0.0

    is_ours = out['owner'] == player
    is_tgt  = ~is_ours

    # phase-множитель — для late_aggression домножим вес.
    phase = max(0.0, min(1.0, float(step) / float(TOTAL_STEPS)))

    def _eff_weights(base):
        eff = dict(base)
        eff['late_aggression'] = eff.get('late_aggression', 0.0) * phase
        return eff

    eff_ours    = _eff_weights(w_ours)
    eff_targets = _eff_weights(w_targets)

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
    'ZONE_FEATURES', 'W_OURS', 'W_TARGETS', 'THR_HI', 'THR_LO', 'ZONE_COLORS',
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
from dataclasses import dataclass, field
from itertools import product as _product



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
    enable_redistribute: bool = False
    transfer_horizon: float = 40.0    # макс ETA для TRANSFER (если включено)
    transfer_floor:   float = 20.0    # GARRISON_FLOOR — не отправляем ниже
    transfer_thresh:  float = 5.0     # минимум score для коммита transfer
    transfer_eta_pen: float = 0.05    # штраф за ETA в transfer-score
    transfer_buffer:  float = 1.0     # ships сверх deficit получателя
    transfer_min_ships: int = 15      # минимум ships в одной TRANSFER-партии (анти-«капельница»)


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


def _action_value(plan, weights, priority_lookup=None, ships_lookup=None):
    """Скор плана для сортировки в аукционе.

    margin            — успешные планы положительный, fail отрицательный
    + tiebreaker по ETA (чем быстрее тем лучше)
    + бонус по priority цели (если задан priority_lookup)
    + ships_term      — «мяч на её стороне»: log1p(ships у самого нагруженного
                        актора) множится на ships_weight. Идея: если у актора
                        накопилось много, его действия должны идти первыми в
                        аукционе — иначе она годами сидит в роли supplier'a и
                        никогда не стреляет.
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

    ships_term   = 0.0
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

    return (margin + eta_term + weights.priority_bonus * prio
            + ships_term + activity_term)


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

def auction(candidates, ours, weights, priority_lookup=None):
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
            _action_value(plan, weights, priority_lookup, ships_lookup),
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


# ── Redistribute (TRANSFER) ─────────────────────────────────────────────

def redistribute(state, remaining, stress, neigh_stress, unfunded, ours, weights):
    """
    Для каждой P с остатком ships > floor — оценить TRANSFER в соседние Q
    и выпустить план если score выше порога.

    Score для P → Q:
        score = max(0, stress[Q] - stress[P]) · ships / (1 + eta · eta_pen)
    где ships ≤ remaining[P] - floor, ограниченное deficit_Q + buffer.

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

    transfer_plans = []
    by_id = {p.id: p for p in ours}

    for P in ours:
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

            gradient = stress.get(Q.id, 0.0) - stress.get(P.id, 0.0)
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
               horizon=ATTACK_HORIZON, max_targets=None, deadline=None):
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
    import time as _t
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = getattr(state, 'raw_planets', state.planets)
    ours = [p for p in raw if p.owner == player]
    if not ours or not targets:
        return [], {'reason': 'no ours or no targets'}

    # cap по числу целей (на ход с 30+ нейтралами это спасает от per-step timeout)
    if max_targets is not None and len(targets) > max_targets:
        if priority_lookup:
            targets = sorted(targets, key=lambda t: -priority_lookup.get(t.id, 0))[:max_targets]
        else:
            targets = list(targets)[:max_targets]

    # 0. Все боевые candidates от существующего движка attacks.py
    risk = int(getattr(weights, 'risk_tolerance', 0))
    candidates = []
    for tgt in targets:
        if deadline is not None and _t.perf_counter() > deadline:
            break  # бюджет исчерпан, играем что собрали
        plans = all_plans(state, tgt, ours, horizon=horizon, player=player, risk=risk)
        for pl in plans:
            if pl.get('success') and _plan_total_ships(pl) >= MIN_USEFUL_STRIKE:
                candidates.append(pl)

    # 1. Stress / neighbor_stress (для отладки и transfer-scoring)
    stress = compute_stress(candidates, ours, weights)
    neigh  = neighbor_stress(stress, ours)

    # 2. Аукцион
    committed, remaining, captured, unfunded = auction(
        candidates, ours, weights, priority_lookup=priority_lookup,
    )

    # 3. Redistribute остатки в TRANSFER.
    #    По умолчанию ВЫКЛЮЧЕН — supply для атак идёт через pipeline/multi
    #    (осознанный duplet), defense через agent._build_defense_plans.
    #    Standalone-передачи легко превращают планету в «вечного supplier'a»,
    #    она копит-капает и никогда не атакует. Включай только осознанно.
    if getattr(weights, 'enable_redistribute', False):
        transfers = redistribute(
            state, remaining, stress, neigh, unfunded, ours, weights,
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
    'compute_stress', 'neighbor_stress',
    'auction', 'redistribute', 'swarm_plan',
]

# ╔══════════════════════════════════════════════════╗
# ║  agent.py                                      ║
# ╚══════════════════════════════════════════════════╝

"""
Агент Orbit Wars.

Каждый ход:
  0. Проекция     — project_state резолвит уже летящие флоты (свои+чужие):
                    каждая планета видится в момент последнего прибытия.
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
import os as _os
import sys as _sys

import math


# ── AgentSwarm switch ─────────────────────────────────────────────────
# True  → планы строим через swarm_plan (per-planet auction + redistribute)
# False → старая глобальная логика best_attacks (fallback / A-B сравнение)
USE_SWARM      = True
SWARM_WEIGHTS  = DEFAULT_WEIGHTS

TARGET_ZONES   = ('easy_target', 'priority_target')
RESERVE_ON_ATT = 0     # минимум кораблей оставить на атакере
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


def _execute_plan_atomically(state, plan, committed):
    """
    Атомарный запуск плана: либо все части плана успешно зарезервированы,
    углы посчитаны и подтверждены simulate_launch, либо план полностью
    отбрасывается (ничего не запускается).

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
            if n_actual <= defender_actual:
                return [], (f'wont_win: ships={n_actual} <= defender_at_eta={defender_actual:.1f} '
                            f'(sim_eta={sim_eta}, tgt={tgt_id})')

        moves.append([planet_id, angle, n_actual])

    committed.update(tentative)
    return moves, 'ok'


def _agent_impl(obs, deadline=None):
    player    = obs.get('player', 0)
    state_raw = GameState.from_kaggle_obs(obs)

    # 0. Проекция: резолвим все летящие флоты (свои и чужие).
    #    После этого state.planets — состояние на момент последнего события.
    #    player передаётся, чтобы projection знал чьи кометы «отскакивают»
    #    обратно на ближайшую нашу планету (см. project_state docstring).
    state = project_state(state_raw, horizon=PROJECTION_HORIZON, player=player)

    _step = (obs.get('step') if isinstance(obs.get('step'), int) else
             obs.get('stepNumber') if isinstance(obs.get('stepNumber'), int) else
             getattr(state_raw, 'step', -1))
    _dbg.begin_turn(_step, player, len(state.planets), len(state.fleets))
    _dbg.log_fleets(state.fleets, player)

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
    try:
        df, _ = compute_zones_from_state(state, player=player)
    except Exception as e:
        _dbg.log_error('compute_zones_from_state', e)
        _dbg.end_turn()
        return []

    _dbg.log_zones(df, projected_at=getattr(state, 'projected_at', {}))

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
        # priority_lookup для tiebreaker'а в аукционе (из zones)
        prio_lookup = {int(r.pid): float(r.priority)
                       for _, r in df.iterrows()
                       if 'priority' in df.columns}
        attack_plans, swarm_dbg = swarm_plan(
            state, player, targets,
            weights=SWARM_WEIGHTS, priority_lookup=prio_lookup,
            max_targets=12,         # cap: на больших картах не успеваем за бюджет kaggle
            deadline=deadline,      # передаём из обёртки agent()
        )
        if _dbg.enabled():
            try:
                _d._w(f'[SWARM]  cand={swarm_dbg["n_candidates"]} '
                      f'committed={swarm_dbg["n_committed"]} '
                      f'transfers={swarm_dbg["n_transfers"]} '
                      f'unfunded={swarm_dbg["n_unfunded"]}')
                top_stress = sorted(swarm_dbg['stress'].items(),
                                    key=lambda x: -x[1])[:5]
                _d._w(f'[SWARM]  top stress: '
                      + ' '.join(f'p{pid}={s:.1f}' for pid, s in top_stress))
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
        sub, reason = _execute_plan_atomically(state, plan, committed)
        if sub:
            _dbg.log_decision(idx, plan, 'FIRE',
                              f'parts={len(sub)} '
                              + ' '.join(f'[src={m[0]} ships={m[2]}]' for m in sub))
            moves.extend(sub)
        else:
            _dbg.log_decision(idx, plan, 'SKIP', reason)

    _dbg.log_moves(moves)
    _dbg.end_turn()
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


# ── BEST WEIGHTS из reverse_tournament ──────────────
# winner_seed20: WR=0.46 на 100 сидах vs default 0.38
SWARM_WEIGHTS = SwarmWeights(
    activity_weight=1.29,
    idle_floor=40,
    distance_comfort=0.18,
    risk_tolerance=2,
    ships_weight=1.37,
    eta_bonus=16.66,
    priority_bonus=2.84,
    stress_top_k=3,
    stress_gamma=1.5,
)

# ── _dbg namespace для совместимости с agent.py ──
import types as _types_for_dbg
_dbg = _types_for_dbg.SimpleNamespace()
if '_resolve' in globals(): setattr(_dbg, '_resolve', globals()['_resolve'])
if 'enabled' in globals(): setattr(_dbg, 'enabled', globals()['enabled'])
if 'reset' in globals(): setattr(_dbg, 'reset', globals()['reset'])
if '_w' in globals(): setattr(_dbg, '_w', globals()['_w'])
if '_safe' in globals(): setattr(_dbg, '_safe', globals()['_safe'])
if 'begin_turn' in globals(): setattr(_dbg, 'begin_turn', globals()['begin_turn'])
if 'log_zones' in globals(): setattr(_dbg, 'log_zones', globals()['log_zones'])
if 'log_targets' in globals(): setattr(_dbg, 'log_targets', globals()['log_targets'])
if 'log_plans' in globals(): setattr(_dbg, 'log_plans', globals()['log_plans'])
if 'log_decision' in globals(): setattr(_dbg, 'log_decision', globals()['log_decision'])
if 'reset_decisions' in globals(): setattr(_dbg, 'reset_decisions', globals()['reset_decisions'])
if 'log_moves' in globals(): setattr(_dbg, 'log_moves', globals()['log_moves'])
if 'log_fleets' in globals(): setattr(_dbg, 'log_fleets', globals()['log_fleets'])
if 'log_error' in globals(): setattr(_dbg, 'log_error', globals()['log_error'])
if 'end_turn' in globals(): setattr(_dbg, 'end_turn', globals()['end_turn'])
if 'plans_all_enabled' in globals(): setattr(_dbg, 'plans_all_enabled', globals()['plans_all_enabled'])
