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
