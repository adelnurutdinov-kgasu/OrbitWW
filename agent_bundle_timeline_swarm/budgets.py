"""
budgets.py — L3 пайплайна.

Для каждой нашей планеты считает сколько кораблей она может отдать
прямо сейчас, не подвергая себя опасности по её таймлайну.

free_budget[P] = min over future turns t of:
    ships_at(P, t) - required_defense(P, t)

где required_defense(P, t) — сколько кораблей нужно держать в момент t,
чтобы пережить летящие на P вражеские флоты в окне [t, t + react_horizon].

Это намного честнее, чем «ships - RESERVE_ON_ATT» из классического swarm,
потому что учитывает не только текущее число кораблей, но и угрозы,
прибывающие позже.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .timeline import PlanetTimeline, NEUTRAL


@dataclass
class BudgetParams:
    """Параметры расчёта бюджетов."""
    horizon: int = 200                # анализируемый горизонт
    react_horizon: int = 30           # окно, в котором ещё возможна реакция
    safety_floor: int = 2             # минимальный гарнизон, который не отдаём
    react_safety: float = 1.5         # коэффициент над минимумом обороны
    sample_step: int = 1              # шаг по времени для min-поиска


def required_defense_at(tl: PlanetTimeline, player: int,
                        t: int, params: BudgetParams) -> float:
    """Минимальный гарнизон, который нужен на планете в момент t.

    Эвристика: смотрим на максимальный размер вражеского арривала в окне
    [t, t + react_horizon] и требуем держать столько же × react_safety.
    Это грубо, но даёт правильный качественный сигнал: «если на нас летит
    большой флот скоро — резерв не отдавать».
    """
    if tl.owner_at(t) != player:
        return 0.0  # планета не наша в этот момент — нам не нужно её защищать

    max_threat = 0.0
    for a in tl.arrivals:
        if a.owner == player:
            continue
        if a.arrival_turn < t:
            continue
        if a.arrival_turn > t + params.react_horizon:
            continue
        if a.ships > max_threat:
            max_threat = a.ships
    return max(float(params.safety_floor), max_threat * params.react_safety)


def free_budget_for(tl: PlanetTimeline, player: int,
                    current_turn: int, params: BudgetParams) -> int:
    """Сколько кораблей планета может отдать сейчас.

    Это min over future turns: ships_at(t) - required_defense(t).
    Берём минимум, потому что если в одной точке горизонта дефицит —
    значит мы не можем отправить столько-то ships сейчас даже если
    «прямо сейчас» гарнизона хватает.

    Возвращаемое — целое ≥ 0.
    """
    if tl.owner_at(current_turn) != player:
        return 0

    horizon_abs = current_turn + params.horizon
    samples = range(current_turn, horizon_abs + 1, max(1, params.sample_step))
    min_slack = float('inf')
    for t in samples:
        if tl.owner_at(t) != player:
            # планета перестала быть нашей — не учитываем (defend разберётся)
            continue
        ships = tl.ships_at(t)
        need = required_defense_at(tl, player, t, params)
        slack = ships - need
        if slack < min_slack:
            min_slack = slack
    if min_slack == float('inf'):
        # планета теряется до того, как мы успеем что-то послать
        return 0
    return max(0, int(min_slack))


def compute_budgets(timelines: Dict[int, PlanetTimeline],
                    our_planet_ids: List[int],
                    current_turn: int,
                    params: Optional[BudgetParams] = None
                    ) -> Dict[int, int]:
    """Главная функция: словарь pid → free_budget."""
    if params is None:
        params = BudgetParams()
    player = None
    # Определяем player как initial_owner первой нашей планеты
    for pid in our_planet_ids:
        tl = timelines.get(pid)
        if tl is not None:
            player = tl.initial_owner
            break
    if player is None:
        return {pid: 0 for pid in our_planet_ids}

    return {
        pid: free_budget_for(timelines[pid], player, current_turn, params)
        for pid in our_planet_ids
        if pid in timelines
    }


__all__ = ['BudgetParams', 'compute_budgets',
           'free_budget_for', 'required_defense_at']
