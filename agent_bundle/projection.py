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

from shooting import (
    fleet_speed_correct, predict_planet_xy, point_to_segment_dist,
    BOARD, CX, CY, SUN_R, Planet,
)


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
