"""
turtle.py — режим «абсолютная защита» для agent_bundle2.

КОГДА АКТИВИРУЕТСЯ:
  Когда на карте не осталось нейтральных планет — фронт зафиксирован, дальше
  только обмен. Если у нас по итогам захвата нейтралов P_us > P_them
  (или хотя бы =), глухая оборона выигрывает на дистанции:
      ships(t+T) - ships_them(t+T) = (P_us - P_them) * T + const
  Линейный рост разрыва. Главное — не терять планет.

КЛАССИФИКАЦИЯ НАШИХ ПЛАНЕТ:
  core   — защищаем абсолютно. Это:
    • все статичные (всегда видны и достижимы для подкрепления)
    • орбитальные, до которых есть НЕЗАСОЛНЕЧНЕНЫЙ путь хотя бы от одной
      нашей статичной (значит можем перебросить корабли в любой момент)
  fringe — осознанно отпускаем (но не обязательно сами сливаем):
    • орбитальные, до которых ВСЕ наши пути блокируются солнцем
      (планета в затмении из нашей сети, не сможем оборонять)

ОПЦИОНАЛЬНО — оппортунистические захваты:
  Орбитальные ВРАЖЕСКИЕ планеты, которые сейчас в затмении из их сети
  (но достижимы из нашей) — слабая оборона, легко взять. Это симметричный
  обмен: они забирают наши fringe, мы — их eclipsed.

КОНСТАНТЫ — РЕДАКТИРУЙ В НАЧАЛЕ ФАЙЛА.
"""

import math
from shooting import (
    fleet_speed_correct, segment_hits_sun, is_orbital,
    SUN_SAFETY, BLOCKER_SAFETY, BOARD, CX, CY, SUN_R,
)
from force import _rendezvous_eta

# ── Константы ─────────────────────────────────────────────────────────────

# Сколько кораблей оставлять в гарнизоне на любой нашей планете, когда мы
# отправляем подкрепление (нельзя слить core-планету ради другой).
TURTLE_MIN_GARRISON = 1

# Минимальный запас при прибытии: нужно прибыть с ≥ defender + threat + buffer.
TURTLE_OVERKILL = 2

# Горизонт расчёта угроз — сколько ходов вперёд считаем потенциальные удары.
TURTLE_THREAT_HORIZON = 100

# Минимальный размер пуска (мелкие — не запускаем).
TURTLE_MIN_FIRE = 3

# Орбитальные считаем «защитимыми», если есть хотя бы один незасолнечненный
# путь от наших статичных. Если статичных нет — все наши орбитальные → core.
TURTLE_REQUIRE_STATIC_LOS = True

# Включить оппортунистические захваты (опасно: тратит корабли с core).
TURTLE_OPPORTUNISTIC = True


# ── Триггер ───────────────────────────────────────────────────────────────

def is_turtle_phase(state):
    """True когда все нейтралы захвачены — фронт зафиксирован."""
    return all(p.owner != -1 for p in state.planets)


# ── Классификация ────────────────────────────────────────────────────────

def _has_clear_path(src, dst, safety=SUN_SAFETY):
    """Есть ли прямой путь от src до dst, не задевающий солнце."""
    return not segment_hits_sun(src.x, src.y, dst.x, dst.y, safety=safety)


def classify_planets(state, player):
    """Возвращает (core, fringe) — наши планеты, разбитые по защитимости."""
    our = [p for p in state.planets if p.owner == player]
    static_ours = [p for p in our if not is_orbital(p.x, p.y, p.radius)]

    core, fringe = [], []
    for p in our:
        if not is_orbital(p.x, p.y, p.radius):
            core.append(p)
            continue
        # Орбитальная: проверяем есть ли LOS от наших статичных
        if not TURTLE_REQUIRE_STATIC_LOS or not static_ours:
            core.append(p)
            continue
        if any(_has_clear_path(s, p) for s in static_ours):
            core.append(p)
        else:
            fringe.append(p)
    return core, fringe


def is_enemy_eclipsed(enemy, state, player):
    """True если все ВРАЖЕСКИЕ статичные не видят этот enemy-orbital
    (он в их затмении). Удобно для оппортунистического захвата."""
    if not is_orbital(enemy.x, enemy.y, enemy.radius):
        return False
    enemy_static = [p for p in state.planets
                    if p.owner not in (-1, player)
                    and not is_orbital(p.x, p.y, p.radius)]
    if not enemy_static:
        return False  # нет статичных у врага — нечего проверять
    return not any(_has_clear_path(s, enemy) for s in enemy_static)


# ── Оценка угроз ─────────────────────────────────────────────────────────

def compute_incoming_threat(target, state, player, horizon=TURTLE_THREAT_HORIZON):
    """Сколько ВРАЖЕСКИХ кораблей может прибыть в target за horizon ходов.
    Учитываем:
      • уже летящие вражеские флоты, целящиеся в target
      • вражеские планеты с _rendezvous_eta(enemy, target) ≤ horizon
    Возвращает суммарное количество кораблей."""
    threat = 0.0

    # 1. Уже летящие вражеские флоты (грубая угловая фильтрация)
    for f in state.fleets:
        if f.owner == player:
            continue
        # Угол от флота к цели
        dx = target.x - f.x
        dy = target.y - f.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        ax = math.atan2(dy, dx)
        diff = abs(((ax - f.angle + math.pi) % (2 * math.pi)) - math.pi)
        if diff > 0.3:  # отклонение > ~17°
            continue
        spd = fleet_speed_correct(max(1, int(f.ships)))
        eta = d / max(spd, 1e-6)
        if eta <= horizon:
            threat += float(f.ships)

    # 2. Потенциальные удары с вражеских планет
    for enemy in state.planets:
        if enemy.owner in (-1, player):
            continue
        if enemy.id == target.id:
            continue
        eta, _ = _rendezvous_eta(enemy, target, max(1, enemy.ships), state.omega)
        if eta <= horizon:
            # Считаем что враг может послать всё что есть + production за время полёта
            threat += float(enemy.ships) + float(enemy.production) * eta

    return threat


# ── Подкрепление ─────────────────────────────────────────────────────────

def find_reinforcer(target, deficit, core_planets, state, player):
    """Ищет нашу core-планету, которая может послать deficit кораблей.
    Возвращает (sender, eta, available_ships) или None.
    Критерий выбора: ближайшая (по eta) с достаточным запасом."""
    candidates = []
    for src in core_planets:
        if src.id == target.id:
            continue
        avail = max(0, src.ships - TURTLE_MIN_GARRISON)
        if avail < TURTLE_MIN_FIRE:
            continue
        # Проверяем что путь чист от солнца
        if not _has_clear_path(src, target):
            continue
        eta, _ = _rendezvous_eta(src, target, max(1, int(avail)), state.omega)
        send_ships = min(avail, int(math.ceil(deficit + TURTLE_OVERKILL)))
        candidates.append((eta, src, send_ships, avail))

    if not candidates:
        return None
    # Сортируем по eta — быстрее доставить лучше
    candidates.sort(key=lambda x: x[0])
    eta, src, send, avail = candidates[0]
    return src, eta, send


# ── Оппортунистический захват вражеских eclipsed orbital ─────────────────

def find_eclipse_targets(state, player, core_planets):
    """Находит вражеские орбитальные в их собственном затмении, до которых
    у нас есть LOS. Возвращает список (enemy, attacker, ships_needed)."""
    targets = []
    if not TURTLE_OPPORTUNISTIC:
        return targets

    for enemy in state.planets:
        if enemy.owner in (-1, player):
            continue
        if not is_enemy_eclipsed(enemy, state, player):
            continue
        # Ищем нашу core с LOS
        for src in core_planets:
            if not _has_clear_path(src, enemy):
                continue
            eta, _ = _rendezvous_eta(src, enemy, max(1, src.ships), state.omega)
            defender = enemy.ships + enemy.production * eta
            needed = int(math.ceil(defender)) + TURTLE_OVERKILL
            avail = max(0, src.ships - TURTLE_MIN_GARRISON)
            if avail >= needed and avail >= TURTLE_MIN_FIRE:
                targets.append((enemy, src, needed, eta))
                break  # один атакёр на цель
    return targets


# ── Главный экспорт ──────────────────────────────────────────────────────

def turtle_moves(state, player, aim_fn):
    """
    Главная функция режима абсолютной защиты.

    Args:
      state: GameState (raw, не projection — нам нужны актуальные ships)
      player: int
      aim_fn: callable(src, dst, ships, state) -> angle | None
              функция наведения (использует aim_hybrid из shooting)

    Returns:
      (moves, debug_info) где moves = [[src_id, angle, ships], ...]
      и debug_info — dict для логирования.
    """
    core, fringe = classify_planets(state, player)
    moves = []
    committed = {}  # src_id -> уже выделено кораблей

    debug = {
        'core_ids':   [p.id for p in core],
        'fringe_ids': [p.id for p in fringe],
        'reinforcements': [],
        'opportunistic': [],
    }

    # Сортируем core по уязвимости — самые угрожаемые первыми
    threats_per = {p.id: compute_incoming_threat(p, state, player) for p in core}
    core_sorted = sorted(core, key=lambda p: -(threats_per[p.id] - p.ships))

    # ── Подкрепления ────────────────────────────────────────────────────
    for our in core_sorted:
        threat = threats_per[our.id]
        # Дефицит = ожидаемая угроза минус наш гарнизон + production за время
        # типичной поддержки. Грубо: half-horizon production уже идёт.
        defense_capacity = our.ships + our.production * (TURTLE_THREAT_HORIZON / 2)
        deficit = threat - defense_capacity
        if deficit <= 0:
            continue

        # Ищем подкрепление, исключаем уже зарезервированные ресурсы
        free_core = []
        for p in core:
            if p.id == our.id:
                continue
            avail = p.ships - committed.get(p.id, 0) - TURTLE_MIN_GARRISON
            if avail >= TURTLE_MIN_FIRE:
                # Создаём «виртуальную» планету с уменьшенным ships для recv
                # вместо мутации (Planet — namedtuple, неизменяема).
                free_core.append(p._replace(ships=int(avail) + TURTLE_MIN_GARRISON))

        result = find_reinforcer(our, deficit, free_core, state, player)
        if result is None:
            continue
        sender, eta, send_ships = result

        # Реальный sender (не виртуальный) для x,y/etc — есть в core
        real_sender = next((p for p in core if p.id == sender.id), None)
        if real_sender is None:
            continue

        angle = aim_fn(real_sender, our, send_ships, state)
        if angle is None:
            continue

        moves.append([real_sender.id, angle, send_ships])
        committed[real_sender.id] = committed.get(real_sender.id, 0) + send_ships
        debug['reinforcements'].append({
            'from': real_sender.id, 'to': our.id,
            'ships': send_ships, 'eta': round(eta, 1),
            'threat': round(threat, 1), 'deficit': round(deficit, 1),
        })

    # ── Оппортунистический захват ──────────────────────────────────────
    if TURTLE_OPPORTUNISTIC:
        eclipse_tgts = find_eclipse_targets(state, player, core)
        for enemy, attacker, needed, eta in eclipse_tgts:
            avail = attacker.ships - committed.get(attacker.id, 0) - TURTLE_MIN_GARRISON
            if avail < needed:
                continue
            angle = aim_fn(attacker, enemy, needed, state)
            if angle is None:
                continue
            moves.append([attacker.id, angle, needed])
            committed[attacker.id] = committed.get(attacker.id, 0) + needed
            debug['opportunistic'].append({
                'from': attacker.id, 'to': enemy.id,
                'ships': needed, 'eta': round(eta, 1),
            })

    return moves, debug


__all__ = [
    'is_turtle_phase', 'classify_planets', 'is_enemy_eclipsed',
    'compute_incoming_threat', 'find_reinforcer', 'find_eclipse_targets',
    'turtle_moves',
    'TURTLE_MIN_GARRISON', 'TURTLE_OVERKILL', 'TURTLE_THREAT_HORIZON',
    'TURTLE_MIN_FIRE', 'TURTLE_OPPORTUNISTIC',
]
