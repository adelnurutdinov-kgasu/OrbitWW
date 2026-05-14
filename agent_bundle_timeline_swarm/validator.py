"""
validator.py — L5 пайплайна.

Принимает финальный список Bid и набор обновлённых таймлайнов из аукциона,
проверяет инварианты и в случае нарушения откатывает наименее ценные коммиты.

Проверки:
  • Ни одна наша планета не должна теряться из-за того, что у неё забрали
    слишком много кораблей (free budget неправильно учёл какую-то угрозу).
  • Ни одна возможность типа `capture/accelerate` не должна оказаться
    нерабочей: после применения всех коммитов планета должна стать нашей
    к обещанному moment.

В прототипе делаем простую проверку «не потеряли ли мы планет-источников»,
а на расхождения — откатываем коммит с наименьшим ROI.
"""

from typing import Dict, List, Optional, Tuple

from .timeline import PlanetTimeline, FleetArrival, NEUTRAL
from .auction import Bid


def _source_safe_after_send(tl: PlanetTimeline, player: int,
                            ships_sent: int, current_turn: int,
                            horizon: int) -> bool:
    """После того как мы убрали `ships_sent` с источника, теряется ли он?

    Грубая проверка: симулируем «уход» как добавление виртуального арривала
    того же владельца с отрицательным числом кораблей… нет, лучше построить
    новый таймлайн с пониженным initial_ships.
    """
    fake = PlanetTimeline(
        planet_id=tl.planet_id,
        initial_owner=tl.initial_owner,
        initial_ships=max(0.0, tl.initial_ships - ships_sent),
        production=tl.production,
        current_turn=tl.current_turn,
        arrivals=list(tl.arrivals),
    )
    lost = fake.first_transition_from(player, current_turn)
    return lost is None or lost > current_turn + horizon


def validate(bids: List[Bid],
             timelines: Dict[int, PlanetTimeline],
             our_planets_by_id: Dict[int, object],
             current_turn: int,
             player: int,
             horizon: int = 200) -> Tuple[List[Bid], List[Bid]]:
    """Возвращает (kept_bids, rolled_back_bids).

    Жадно откатываем самый низко-ROI коммит, пока остаются нарушения.
    """
    bids = list(bids)
    rolled_back: List[Bid] = []

    while True:
        # Считаем, сколько каждый источник суммарно отдал
        sent_per_source: Dict[int, int] = {}
        for b in bids:
            sent_per_source[b.source_id] = sent_per_source.get(b.source_id, 0) + b.ships

        problems: List[int] = []  # ids источников, у которых проблема
        for sid, total_sent in sent_per_source.items():
            tl = timelines.get(sid)
            if tl is None:
                continue
            if not _source_safe_after_send(tl, player, total_sent,
                                           current_turn, horizon):
                problems.append(sid)
        if not problems:
            break

        # Откатываем коммит с наименьшим ROI среди затронутых источников
        problem_set = set(problems)
        candidates = [b for b in bids if b.source_id in problem_set]
        if not candidates:
            break
        worst = min(candidates, key=lambda b: b.roi)
        bids.remove(worst)
        rolled_back.append(worst)

    return bids, rolled_back


__all__ = ['validate']
