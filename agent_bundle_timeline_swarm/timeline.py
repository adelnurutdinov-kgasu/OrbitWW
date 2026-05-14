"""
timeline.py — таймлайн владения для одной планеты.

Главная идея: вместо одного «спроецированного» состояния храним
последовательность событий (turn, owner, ships), которая полностью
описывает будущее планеты при заданном множестве летящих к ней флотов.

Это позволяет легко отвечать на вопросы:
  • кому принадлежит планета в момент t?
  • когда впервые произойдёт переход владения к конкретному игроку?
  • что будет, если убрать/добавить конкретный флот? (hypothetical)
  • есть ли "load-bearing" флоты, без которых проекция переворачивается?

Модель резолюции боя — та же, что в проекции/движке: strict-inequality
бой, ничья оставляет владельца с 0 кораблями.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

NEUTRAL = -1


# ── Event ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FleetArrival:
    """Описание прибытия флота к планете.

    arrival_turn — абсолютный (game step + delta).
    fleet_id     — идентификатор источника; None для гипотетических
                   событий (например, тех, что мы только планируем послать).
    """
    arrival_turn: int
    owner: int
    ships: float
    fleet_id: Optional[int] = None
    source_planet_id: Optional[int] = None
    purpose: str = "unknown"  # capture/defend/accelerate/reinforce/hypothetical


@dataclass(frozen=True)
class Event:
    """Состояние планеты сразу ПОСЛЕ резолюции арривала или производства."""
    turn: int
    owner: int
    ships: float
    cause: str  # "initial" | "arrival" | "horizon"
    fleet_id: Optional[int] = None


# ── Бой / резолюция ────────────────────────────────────────────────────

def _resolve(cur_owner: int, cur_ships: float, production: float,
             dt: int, ev_owner: int, ev_ships: float) -> Tuple[int, float]:
    """Тождественная семантика с projection._resolve_arrival.

    Применяем производство за dt ходов (только если владелец не нейтрал),
    затем бой по strict-inequality.
    """
    if cur_owner != NEUTRAL:
        cur_ships = cur_ships + production * dt

    if ev_owner == cur_owner:
        return cur_owner, cur_ships + ev_ships
    if ev_ships > cur_ships:
        return ev_owner, ev_ships - cur_ships
    if ev_ships < cur_ships:
        return cur_owner, cur_ships - ev_ships
    # ничья — обе стороны обнуляются, владелец цели не меняется
    return cur_owner, 0.0


# ── PlanetTimeline ────────────────────────────────────────────────────

class PlanetTimeline:
    """
    Таймлайн владения для одной планеты.

    Хранит список арривалов (отсортированный по turn) и initial-состояние.
    События `events` пересчитываются лениво по запросу.
    """

    __slots__ = (
        'planet_id', 'initial_owner', 'initial_ships', 'production',
        'current_turn', 'arrivals', '_cached_events', '_cache_key',
    )

    def __init__(self,
                 planet_id: int,
                 initial_owner: int,
                 initial_ships: float,
                 production: float,
                 current_turn: int = 0,
                 arrivals: Optional[List[FleetArrival]] = None):
        self.planet_id = planet_id
        self.initial_owner = initial_owner
        self.initial_ships = float(initial_ships)
        self.production = float(production)
        self.current_turn = int(current_turn)
        self.arrivals: List[FleetArrival] = list(arrivals or [])
        self._cached_events: Optional[List[Event]] = None
        self._cache_key: int = 0

    # ── Мутации (возвращают новый таймлайн, не модифицируют текущий) ──

    def with_added_arrival(self, arrival: FleetArrival) -> 'PlanetTimeline':
        return PlanetTimeline(
            self.planet_id, self.initial_owner, self.initial_ships,
            self.production, self.current_turn,
            arrivals=self.arrivals + [arrival],
        )

    def without_fleet(self, fleet_id: int) -> 'PlanetTimeline':
        return PlanetTimeline(
            self.planet_id, self.initial_owner, self.initial_ships,
            self.production, self.current_turn,
            arrivals=[a for a in self.arrivals if a.fleet_id != fleet_id],
        )

    def without_owner_fleets(self, owner: int) -> 'PlanetTimeline':
        """Убрать все арривалы конкретного владельца (для расчёта `naked`)."""
        return PlanetTimeline(
            self.planet_id, self.initial_owner, self.initial_ships,
            self.production, self.current_turn,
            arrivals=[a for a in self.arrivals if a.owner != owner],
        )

    # ── Запросы ────────────────────────────────────────────────────────

    def events(self) -> List[Event]:
        """Полная цепочка состояний: initial + по событию на каждый арривал."""
        key = (len(self.arrivals),
               sum(hash((a.arrival_turn, a.owner, a.ships, a.fleet_id))
                   for a in self.arrivals))
        if self._cached_events is not None and self._cache_key == key:
            return self._cached_events

        evs: List[Event] = [Event(
            turn=self.current_turn,
            owner=self.initial_owner,
            ships=self.initial_ships,
            cause="initial",
        )]
        owner, ships = self.initial_owner, self.initial_ships
        last = self.current_turn
        for a in sorted(self.arrivals, key=lambda x: x.arrival_turn):
            dt = max(0, a.arrival_turn - last)
            owner, ships = _resolve(owner, ships, self.production,
                                    dt, a.owner, a.ships)
            evs.append(Event(
                turn=a.arrival_turn,
                owner=owner,
                ships=ships,
                cause="arrival",
                fleet_id=a.fleet_id,
            ))
            last = a.arrival_turn
        self._cached_events = evs
        self._cache_key = key
        return evs

    def owner_at(self, turn: int) -> int:
        """Владелец на момент `turn` (по последнему событию ≤ turn)."""
        evs = self.events()
        owner = evs[0].owner
        for e in evs:
            if e.turn > turn:
                break
            owner = e.owner
        return owner

    def ships_at(self, turn: int) -> float:
        """Корабли на момент `turn` с учётом производства от последнего события."""
        evs = self.events()
        last = evs[0]
        for e in evs:
            if e.turn > turn:
                break
            last = e
        dt = max(0, turn - last.turn)
        ships = last.ships
        if last.owner != NEUTRAL:
            ships = ships + self.production * dt
        return ships

    def first_transition_to(self, target_owner: int,
                            after_turn: Optional[int] = None) -> Optional[int]:
        """Первый turn ≥ after_turn, когда планета принадлежит target_owner."""
        if after_turn is None:
            after_turn = self.current_turn
        for e in self.events():
            if e.turn < after_turn:
                continue
            if e.owner == target_owner:
                return e.turn
        return None

    def first_transition_from(self, owner: int,
                              after_turn: Optional[int] = None) -> Optional[int]:
        """Первый turn ≥ after_turn, когда `owner` теряет планету."""
        if after_turn is None:
            after_turn = self.current_turn
        evs = self.events()
        was_owner = False
        for e in evs:
            if e.turn < after_turn:
                was_owner = (e.owner == owner)
                continue
            if was_owner and e.owner != owner:
                return e.turn
            was_owner = (e.owner == owner)
        return None

    def safety_margin(self, target_owner: int) -> Optional[float]:
        """Сколько кораблей в момент первого перехода к target_owner.

        Это «запас прочности» захвата: если ≈0 — захват хрупкий,
        и любая случайная контр-атака его сломает.
        Возвращает None если планета не переходит к target_owner.
        """
        t = self.first_transition_to(target_owner)
        if t is None:
            return None
        return self.ships_at(t)

    def final_state(self, horizon: int) -> Tuple[int, float]:
        """Состояние в момент horizon (с производством после последнего события)."""
        return self.owner_at(horizon), self.ships_at(horizon)

    # ── Удобные предикаты ──────────────────────────────────────────────

    def is_static(self) -> bool:
        """Никаких прибытий — планета будет в исходном состоянии."""
        return len(self.arrivals) == 0

    def becomes_ours_at(self, player: int) -> Optional[int]:
        """Когда планета впервые становится нашей (None если никогда)."""
        return self.first_transition_to(player)

    def lost_at(self, player: int) -> Optional[int]:
        """Когда мы её теряем (None если не теряем)."""
        return self.first_transition_from(player)

    def __repr__(self) -> str:
        return (f"Timeline(p={self.planet_id}, "
                f"init={self.initial_owner}/{self.initial_ships:.0f}, "
                f"arrivals={len(self.arrivals)})")


__all__ = ['NEUTRAL', 'FleetArrival', 'Event', 'PlanetTimeline']
