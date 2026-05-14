"""
simulator.py — L1 пайплайна.

Строит PlanetTimeline для каждой планеты, проходясь по всем флотам
в state.fleets и определяя их цель + время прилёта.

Физика берётся из существующего движка:
  • simulate_fleet_target — реальная симуляция траектории по правилам
    игры (учитывает солнце, орбиты, источник).
  • fleet_speed_correct — скорость флота от числа кораблей (используется
    для ETA планируемых отправок в auction).

Чтобы модуль был чистым относительно остальной bundle, мы импортируем
только эти конкретные функции.
"""

import math
import sys
import os
from typing import Dict, List, Optional

# Добавляем sibling-bundle в path, чтобы переиспользовать движок без
# полного зависимостного клонирования.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING = os.path.join(os.path.dirname(_HERE), 'agent_bundle_swarm')
if _SIBLING not in sys.path:
    sys.path.insert(0, _SIBLING)

try:
    # Реальная физика из движка
    from projection import simulate_fleet_target
    from shooting import fleet_speed_correct, predict_planet_xy
except Exception:  # pragma: no cover
    # Fallback на случай изолированного тестирования без bundle —
    # будет работать на синтетических сценариях.
    simulate_fleet_target = None
    fleet_speed_correct = None
    predict_planet_xy = None

from .timeline import PlanetTimeline, FleetArrival, NEUTRAL


# ── ETA для гипотетических отправок ────────────────────────────────────

def estimate_eta(src_x: float, src_y: float, src_r: float,
                 dst_x: float, dst_y: float, dst_r: float,
                 ships: int) -> int:
    """Грубая ETA для планируемого флота. Используется в опперируемости
    и аукционе ДО реального запуска (когда у флота ещё нет траектории).

    Формула: чистое евклидово расстояние / скорость(ships). На прямой
    линии это нижняя граница; реальный полёт с учётом орбит немного
    длиннее, но для аукциона грубой оценки достаточно — финальный
    запуск всё равно идёт через `_aim_and_verify` агента.
    """
    if fleet_speed_correct is None:
        # fallback для unit-тестов: фиксированная скорость 6.0
        speed = 6.0
    else:
        speed = float(fleet_speed_correct(max(1, int(ships))))
    d = max(1.0, math.hypot(dst_x - src_x, dst_y - src_y) - src_r - dst_r)
    return int(math.ceil(d / max(speed, 1e-6)))


def distance(p, q) -> float:
    """Геометрическое расстояние центров планет."""
    return math.hypot(p.x - q.x, p.y - q.y)


# ── Главная функция: build_timelines ───────────────────────────────────

def build_timelines(state, current_turn: int, player: int,
                    horizon: int = 200) -> Dict[int, PlanetTimeline]:
    """
    Для каждой планеты state.planets создаёт PlanetTimeline.

    Все флоты в state.fleets резолвятся к своим целям через
    `simulate_fleet_target` и добавляются как FleetArrival на
    соответствующих планетах.

    horizon — горизонт симуляции для траекторий. События после
    horizon отбрасываются (флот «не успел»).
    """
    raw = getattr(state, 'raw_planets', None) or state.planets
    omega = state.omega

    timelines: Dict[int, PlanetTimeline] = {
        p.id: PlanetTimeline(
            planet_id=p.id,
            initial_owner=int(p.owner),
            initial_ships=float(p.ships),
            production=float(p.production),
            current_turn=current_turn,
        ) for p in raw
    }

    if simulate_fleet_target is None:
        # Без движка строим пустые таймлайны (полезно в чистом юнит-тесте).
        return timelines

    fleets = getattr(state, 'raw_fleets', None)
    if fleets is None:
        fleets = getattr(state, 'fleets', []) or []

    for f in fleets:
        pid, dt = simulate_fleet_target(f, raw, omega, max_turns=horizon)
        if pid is None:
            continue
        arrival_turn = current_turn + int(dt)
        if pid not in timelines:
            continue
        timelines[pid] = timelines[pid].with_added_arrival(FleetArrival(
            arrival_turn=arrival_turn,
            owner=int(f.owner),
            ships=float(f.ships),
            fleet_id=int(getattr(f, 'id', id(f))),
            source_planet_id=getattr(f, 'from_planet_id', None),
            purpose="in_flight",
        ))

    return timelines


def add_planned_arrival(timelines: Dict[int, PlanetTimeline],
                        target_id: int,
                        arrival_turn: int,
                        owner: int,
                        ships: float,
                        source_id: Optional[int] = None,
                        purpose: str = "planned") -> Dict[int, PlanetTimeline]:
    """Возвращает копию словаря таймлайнов с добавленным виртуальным
    арривалом — используется в аукционе при тентативной фиксации плана."""
    if target_id not in timelines:
        return timelines
    new_timelines = dict(timelines)
    new_timelines[target_id] = timelines[target_id].with_added_arrival(
        FleetArrival(
            arrival_turn=int(arrival_turn),
            owner=int(owner),
            ships=float(ships),
            fleet_id=None,
            source_planet_id=source_id,
            purpose=purpose,
        )
    )
    return new_timelines


__all__ = ['build_timelines', 'add_planned_arrival',
           'estimate_eta', 'distance']
