"""
action_space.py — Пространство Адекватных Решений (ПАР) для MCTS.

Генерирует конечное множество осмысленных ходов для заданного игрока.
Каждый ход: одиночный запуск флота с одной планеты на одну цель.

Типы ходов:
  capture  — захват вражеской или нейтральной планеты
  reinforce — усиление своей планеты (посылаем только излишек)

Фильтры адекватности:
  • src принадлежит player_id и имеет >= MIN_SEND кораблей
  • прямой путь не проходит через солнце (быстрая проверка)
  • ETA (по rendezvous-итерации) <= max_travel_time
  • aim_hybrid находит валидный угол (блокеры, орбиты)

Публичный интерфейс:
  class Action
  generate_actions(state, player_id, max_travel_time) -> List[Action]
  generate_opponent_actions(state, opponent_id, max_travel_time) -> List[Action]
"""

import math
from dataclasses import dataclass
from typing import List

from shooting import (
    aim_hybrid,
    segment_hits_sun,
    SUN_SAFETY,
)
from force import _rendezvous_eta
from projection import NEUTRAL_OWNER

# ── Параметры ──────────────────────────────────────────────────────────────
MAX_TRAVEL_TIME   = 25    # горизонт ранней игры (ходов)
RESERVE_FACTOR    = 1.5   # держим production * factor на источнике в резерве
MIN_SEND          = 3     # минимально осмысленный пуск
OVERKILL_NEUTRAL  = 1     # буфер для нейтрала (нет производства → нет дрейфа)
OVERKILL_OWNED    = 2     # буфер для owned (производство + дрейф eta)


# ══════════════════════════════════════════════════════════════════════════
# Класс Action
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Action:
    """Одиночный ход: пуск ships кораблей с from_id на target_id.

    angle и travel_time предрассчитаны aim_hybrid — их достаточно для
    передачи в simulate_step без повторного вычисления угла.
    """
    from_id:     int
    target_id:   int
    ships:       int
    angle:       float   # радианы (откалиброванный aim_hybrid)
    travel_time: int     # ходов до прибытия
    mode:        str     # 'capture' | 'reinforce'

    def to_moves(self) -> list:
        """Формат [[from_id, angle, ships]] для agent.py / simulate_step."""
        return [[self.from_id, self.angle, self.ships]]

    def __repr__(self) -> str:
        return (f"Action({self.mode} {self.from_id}→{self.target_id} "
                f"ships={self.ships} eta={self.travel_time})")


# ══════════════════════════════════════════════════════════════════════════
# Вспомогательные расчёты
# ══════════════════════════════════════════════════════════════════════════

def _defender_at_arrival(state, tgt, travel_time: float) -> float:
    """Оценка кораблей защитника на момент прибытия нашего флота.

    Учитывает projected_at (если состояние спроецировано) — ровно как
    в _defender_at из agent.py, чтобы логика совпадала.
    """
    proj_at = getattr(state, 'projected_at', {}).get(tgt.id, 0)
    if tgt.owner == NEUTRAL_OWNER:
        return float(tgt.ships)
    return float(tgt.ships) + float(tgt.production) * max(0.0, travel_time - proj_at)


def _ship_quantities(src, tgt, eta: float, state, player_id: int) -> List[int]:
    """Кандидатные количества кораблей для пары (src → tgt).

    Возвращает отсортированный список уникальных значений:
    - capture: min_capture, gradient (излишек), max_available
    - reinforce: gradient, max_available
    Все значения >= MIN_SEND и <= max_available (src.ships - 1).
    """
    max_avail = max(0, src.ships - 1)   # минимум 1 в гарнизоне
    if max_avail < MIN_SEND:
        return []

    candidates: set = set()

    if tgt.owner == player_id:
        # Реинфорс: посылаем только излишек, чтобы не тормозить рост источника
        grad = max(0, int(src.ships - src.production * RESERVE_FACTOR))
        if MIN_SEND <= grad <= max_avail:
            candidates.add(grad)
        candidates.add(max_avail)
    else:
        # Захват: три точки покрывают «минимальный», «экономный», «агрессивный»
        defender = _defender_at_arrival(state, tgt, eta)
        overkill = OVERKILL_NEUTRAL if tgt.owner == NEUTRAL_OWNER else OVERKILL_OWNED
        min_cap = int(math.ceil(defender)) + overkill
        if MIN_SEND <= min_cap <= max_avail:
            candidates.add(min_cap)

        grad = max(0, int(src.ships - src.production * RESERVE_FACTOR))
        if MIN_SEND <= grad <= max_avail:
            candidates.add(grad)

        candidates.add(max_avail)

    return sorted(s for s in candidates if MIN_SEND <= s <= max_avail)


def _initial_by_id(state):
    """Возвращает dict initial_planets из GameState или ProjectedState."""
    fn = getattr(state, 'initial_by_id', None)
    if callable(fn):
        return fn()
    return getattr(state, '_initial_planets', {})


# ══════════════════════════════════════════════════════════════════════════
# Основной генератор ПАР
# ══════════════════════════════════════════════════════════════════════════

def generate_actions(state, player_id: int,
                     max_travel_time: int = MAX_TRAVEL_TIME) -> List[Action]:
    """Генерирует ПАР — список адекватных ходов для player_id.

    Источник (src): raw_planets (фактическое состояние).
    Цели (tgt):    state.planets (спроецированное — правильная защита).

    Args:
        state:           ProjectedState или GameState.
        player_id:       наш player_id.
        max_travel_time: отсечка ETA.

    Returns:
        Список Action с предрассчитанными углами и ETA.
    """
    raw_planets = getattr(state, 'raw_planets', state.planets)
    src_planets = [p for p in raw_planets
                   if p.owner == player_id and p.ships >= MIN_SEND]
    all_planets = state.planets
    omega = state.omega
    ibd = _initial_by_id(state)

    actions: List[Action] = []

    for src in src_planets:
        for tgt in all_planets:
            if tgt.id == src.id:
                continue

            # ── Быстрый фильтр 1: солнце на прямой src→tgt ────────────
            if segment_hits_sun(src.x, src.y, tgt.x, tgt.y, safety=SUN_SAFETY):
                continue

            # ── Быстрый фильтр 2: примерный ETA ───────────────────────
            ref_ships = max(MIN_SEND, src.ships // 2)
            eta_est, _ = _rendezvous_eta(src, tgt, ref_ships, omega)
            if eta_est > max_travel_time:
                continue

            mode = 'reinforce' if tgt.owner == player_id else 'capture'
            quantities = _ship_quantities(src, tgt, eta_est, state, player_id)

            for ships in quantities:
                # ── Точный угол + проверка блокеров ───────────────────
                result = aim_hybrid(src, tgt, ships, all_planets, ibd, omega)
                if result is None:
                    continue
                angle, eta_actual, _, _ = result
                if int(eta_actual) > max_travel_time:
                    continue

                actions.append(Action(
                    from_id=src.id,
                    target_id=tgt.id,
                    ships=ships,
                    angle=angle,
                    travel_time=int(eta_actual),
                    mode=mode,
                ))

    return actions


def generate_opponent_actions(state, opponent_id: int,
                               max_travel_time: int = MAX_TRAVEL_TIME) -> List[Action]:
    """Симметричная генерация ПАР для противника.

    Использует те же правила что и generate_actions — игра с полной
    информацией (state один и тот же). Результат помечен mode с точки
    зрения противника.
    """
    return generate_actions(state, opponent_id, max_travel_time)


__all__ = [
    'Action',
    'MAX_TRAVEL_TIME', 'MIN_SEND', 'RESERVE_FACTOR',
    'OVERKILL_NEUTRAL', 'OVERKILL_OWNED',
    'generate_actions', 'generate_opponent_actions',
    '_defender_at_arrival',
]
