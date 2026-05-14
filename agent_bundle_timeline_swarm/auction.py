"""
auction.py — L4 пайплайна.

Жадный многотиерный аукцион: матчит возможности (Opportunity) и
бюджеты источников в коммиты (Bid). Идёт по приоритетным тиерам
сверху вниз, внутри тиера — по маржинальному ROI на корабль.

Ключевая особенность относительно классического swarm:
  • value сравнивается между типами возможностей (capture, accelerate,
    defend, snipe) в одной валюте — production×turns.
  • После каждого коммита таймлайн целевой планеты обновляется
    (add_planned_arrival), и зависящие от неё возможности пересчитываются.
  • Бюджет источника обновляется одновременно — если источник опустел,
    его опции исчезают.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .timeline import PlanetTimeline, FleetArrival, NEUTRAL
from .opportunities import Opportunity, OppParams
from .simulator import estimate_eta, distance, add_planned_arrival


@dataclass
class AuctionParams:
    horizon: int = 200
    min_useful_strike: int = 3        # минимум кораблей в отправке
    min_roi: float = 0.05             # минимум value/ships для коммита
    tier_caps: Optional[Dict[int, float]] = None  # доля бюджета на тиер (None = без cap)


@dataclass
class Bid:
    """Совершённый коммит: source → target, ships, тип возможности."""
    source_id: int
    target_id: int
    ships: int
    arrival_turn: int
    value: float
    roi: float
    opp_type: str
    opp_tier: int
    purpose: str  # совпадает с opp_type обычно

    def to_plan_dict(self, src_planet, dst_planet) -> dict:
        """Совместимый со swarm-форматом план (для agent._execute_plan_atomically)."""
        return {
            'mode':       'transfer' if self.opp_type == 'reinforce_safety' else 'direct',
            'sup_id':     None,
            'att_id':     self.source_id,
            'tgt_id':     self.target_id,
            'eta_sa':     0.0,
            'eta_at':     float(self.arrival_turn),
            't_total':    float(self.arrival_turn),
            'x_sup':      0,
            'x_att':      int(self.ships),
            'prod_att':   float(getattr(src_planet, 'production', 0.0)),
            'x_tgt':      0,
            'prod_tgt':   float(getattr(dst_planet, 'production', 0.0)),
            'strike':     float(self.ships),
            'defender':   0.0,
            'incoming':   0.0,
            'needed':     0,
            'margin':     float(self.value),
            'success':    True,
            'relay_reason': None,
            'opp_type':   self.opp_type,
            'opp_tier':   self.opp_tier,
            'roi':        self.roi,
            'is_timeline_swarm': True,
        }


def _ship_levels(opp: Opportunity, budget: int,
                 params: AuctionParams) -> List[int]:
    """Несколько разумных уровней ships для отправки.

    Покрываем диапазон геометрически от max(min_useful_strike, min_ships+2)
    до budget — фиксированное число шагов даёт хорошее покрытие любого
    масштаба бюджета. Дополнительно вставляем «впритык» (min_ships+2)
    как стартовую точку.

    Аукцион выберет уровень с лучшим маржинальным ROI.
    """
    floor = max(params.min_useful_strike,
                int(math.ceil(opp.min_ships + 2)))
    if budget < floor:
        return []
    # Геометрическая прогрессия в 6 шагов от floor до budget
    raw: set = {floor, budget}
    if budget > floor:
        ratio = (budget / floor) ** (1.0 / 5.0)
        v = float(floor)
        for _ in range(5):
            v *= ratio
            raw.add(int(round(v)))
    out: List[int] = []
    for lv in sorted(raw):
        if lv < params.min_useful_strike or lv > budget:
            continue
        if lv not in out:
            out.append(lv)
    return out


def _eligible_sources(opp: Opportunity, our_planets_by_id: Dict[int, object],
                      budgets: Dict[int, int],
                      target_planet, current_turn: int,
                      params: AuctionParams) -> List[Tuple[int, int, int]]:
    """Список (source_id, eta, max_ships) — источники, которые могут
    физически довезти достаточно кораблей к deadline."""
    out: List[Tuple[int, int, int]] = []
    for sid, src in our_planets_by_id.items():
        budget = budgets.get(sid, 0)
        if budget < params.min_useful_strike:
            continue
        # Грубая ETA при отправке "большого" флота (используем budget)
        eta = estimate_eta(src.x, src.y, src.radius,
                           target_planet.x, target_planet.y, target_planet.radius,
                           ships=budget)
        arrival = current_turn + eta
        if arrival > opp.deadline:
            continue
        out.append((sid, eta, budget))
    # Сортируем по ETA — близкие источники сначала, чтобы ускорения и
    # снайпы вовремя.
    out.sort(key=lambda x: x[1])
    return out


def _opportunity_marginal_roi(opp: Opportunity,
                              source_id: int,
                              ships_to_send: int,
                              eta: int,
                              timelines: Dict[int, PlanetTimeline],
                              current_turn: int,
                              params: AuctionParams,
                              opp_params: OppParams) -> Tuple[float, float, int]:
    """ROI на 1 корабль при отправке `ships_to_send` из source_id в opp.target_id.

    Маржинальный: текущая ценность возможности уже может быть частично
    закрыта предыдущими коммитами в timelines[target]. Возвращаем
    (value, roi, arrival_turn).
    """
    arrival_turn = current_turn + eta
    if arrival_turn > opp.deadline:
        return 0.0, 0.0, arrival_turn

    tl = timelines.get(opp.target_id)
    if tl is None:
        return 0.0, 0.0, arrival_turn

    player = tl.owner_at(current_turn)  # для defend это и есть наш игрок

    # Симулируем добавление гипотетического флота и смотрим на сдвиг
    # ключевой метрики возможности.
    if opp.type == "capture" or opp.type == "snipe" or opp.type == "recapture":
        # Что мы получаем: переход к нам в момент arrival_turn (если ships
        # хватает). Симулируем: добавляем флот, смотрим owner_at(arrival_turn).
        hypothetical = tl.with_added_arrival(FleetArrival(
            arrival_turn=arrival_turn, owner=_inferred_player(tl, opp),
            ships=float(ships_to_send),
        ))
        # Без нашей помощи владелец в момент arrival
        baseline_owner = tl.owner_at(arrival_turn)
        # Игрок — это тот, кто посылает (см. инференс ниже)
        our_player = _inferred_player(tl, opp)
        new_owner = hypothetical.owner_at(arrival_turn)
        if new_owner == our_player and baseline_owner != our_player:
            value = opp.value(ships_to_send, arrival_turn, opp_params)
            return value, value / max(1, ships_to_send), arrival_turn
        return 0.0, 0.0, arrival_turn

    if opp.type == "defend":
        # value: предотвращаем потерю. Сравниваем с ТЕКУЩИМ состоянием
        # таймлайна — после возможных предыдущих коммитов планета может
        # уже не теряться, тогда новых отправок не делаем.
        our_player = _inferred_player(tl, opp)
        current_lost = tl.first_transition_from(our_player, current_turn)
        if current_lost is None:
            return 0.0, 0.0, arrival_turn  # уже защищена предыдущими коммитами
        hypothetical = tl.with_added_arrival(FleetArrival(
            arrival_turn=arrival_turn, owner=our_player,
            ships=float(ships_to_send),
        ))
        new_lost = hypothetical.first_transition_from(our_player, current_turn)
        if new_lost is None:
            # Полностью спасли — value = production × оставшееся окно
            delta = max(0, opp.horizon_end_abs - current_lost)
            value = opp.production * delta
            return value, value / max(1, ships_to_send), arrival_turn
        if new_lost <= current_lost:
            # Не отсрочили
            return 0.0, 0.0, arrival_turn
        delta = new_lost - current_lost
        value = opp.production * delta
        return value, value / max(1, ships_to_send), arrival_turn

    if opp.type == "accelerate":
        # Сравниваем с ТЕКУЩИМ eta в таймлайне (учитывает прошлые коммиты).
        our_player = _inferred_player(tl, opp)
        current_eta_in_tl = tl.first_transition_to(our_player, current_turn)
        if current_eta_in_tl is None or current_eta_in_tl <= current_turn:
            return 0.0, 0.0, arrival_turn
        hypothetical = tl.with_added_arrival(FleetArrival(
            arrival_turn=arrival_turn, owner=our_player,
            ships=float(ships_to_send),
        ))
        new_eta = hypothetical.first_transition_to(our_player, current_turn)
        if new_eta is None or new_eta >= current_eta_in_tl:
            return 0.0, 0.0, arrival_turn
        delta_t = current_eta_in_tl - new_eta
        value = opp_params.accelerate_value_discount * opp.production * delta_t
        return value, value / max(1, ships_to_send), arrival_turn

    if opp.type == "reinforce_safety":
        # Стандартный буферный бонус — value константа
        value = opp.value(ships_to_send, arrival_turn, opp_params)
        return value, value / max(1, ships_to_send), arrival_turn

    return 0.0, 0.0, arrival_turn


def _inferred_player(tl: PlanetTimeline, opp: Opportunity) -> int:
    """Вычисляем 'нашего игрока'. Для accelerate / defend это владелец
    в момент перехода / сейчас. Для capture / snipe — единственный
    разумный кандидат: тот, кто посылает (передаём через opp.extra).
    """
    # Для defend / reinforce / accelerate он восстанавливается из таймлайна.
    if opp.type == "defend":
        # тот, кому планета принадлежит сейчас
        return tl.owner_at(tl.current_turn)
    if opp.type == "reinforce_safety":
        return tl.owner_at(tl.current_turn)
    if opp.type == "accelerate":
        eta = opp.current_eta
        if eta is not None:
            return tl.owner_at(eta)
        return tl.owner_at(tl.current_turn + 1)
    # capture / snipe / recapture: используем opp.extra['player'] если есть,
    # иначе берём владельца, отличного от текущего и не нейтрала, который
    # должен возникнуть после нашей подмоги
    return opp.extra.get('player', tl.owner_at(tl.current_turn))


# ── Главная функция ───────────────────────────────────────────────────

def run_auction(opportunities: List[Opportunity],
                timelines: Dict[int, PlanetTimeline],
                budgets: Dict[int, int],
                our_planets_by_id: Dict[int, object],
                target_planets_by_id: Dict[int, object],
                current_turn: int,
                player: int,
                auction_params: Optional[AuctionParams] = None,
                opp_params: Optional[OppParams] = None
                ) -> Tuple[List[Bid], Dict[int, PlanetTimeline], Dict[int, int], Dict[str, int]]:
    """
    Многотиерный жадный аукцион.

    Возвращает (committed_bids, updated_timelines, updated_budgets, stats).
    """
    if auction_params is None:
        auction_params = AuctionParams()
    if opp_params is None:
        opp_params = OppParams(horizon=auction_params.horizon)

    # Кладём player в opp.extra для capture/snipe — там это используется
    # как «целевой owner» при симуляции
    for opp in opportunities:
        opp.extra.setdefault('player', player)

    # Группируем по тиерам
    by_tier: Dict[int, List[Opportunity]] = {}
    for opp in opportunities:
        by_tier.setdefault(opp.tier, []).append(opp)

    committed: List[Bid] = []
    timelines = dict(timelines)
    budgets = dict(budgets)
    stats = {
        'considered': 0,
        'committed': 0,
        'rejected_no_source': 0,
        'rejected_low_roi': 0,
        'tiers_used': set(),
    }

    for tier in sorted(by_tier.keys()):
        # Один проход внутри тиера
        progress = True
        while progress:
            progress = False

            # Сгенерировать все ROI для всех (opp, source) пар
            offers: List[Tuple[float, Opportunity, int, int, int, float]] = []
            for opp in by_tier[tier]:
                tgt = target_planets_by_id.get(opp.target_id)
                if tgt is None:
                    continue
                eligible = _eligible_sources(
                    opp, our_planets_by_id, budgets, tgt,
                    current_turn, auction_params,
                )
                for sid, eta, budget in eligible:
                    # Несколько уровней ships для одной пары — аукцион выберет
                    # вариант с лучшим маржинальным ROI.
                    levels = _ship_levels(opp, budget, auction_params)
                    for ships_to_send in levels:
                        value, roi, arrival = _opportunity_marginal_roi(
                            opp, sid, ships_to_send, eta,
                            timelines, current_turn,
                            auction_params, opp_params,
                        )
                        stats['considered'] += 1
                        if roi < auction_params.min_roi:
                            stats['rejected_low_roi'] += 1
                            continue
                        offers.append((roi, opp, sid, ships_to_send,
                                       arrival, value))

            if not offers:
                break

            # Берём максимальный ROI
            offers.sort(key=lambda x: -x[0])
            best_roi, best_opp, best_sid, best_ships, best_arrival, best_value = offers[0]

            # Фиксируем
            committed.append(Bid(
                source_id=best_sid,
                target_id=best_opp.target_id,
                ships=best_ships,
                arrival_turn=best_arrival,
                value=best_value,
                roi=best_roi,
                opp_type=best_opp.type,
                opp_tier=best_opp.tier,
                purpose=best_opp.type,
            ))
            stats['committed'] += 1
            stats['tiers_used'].add(tier)

            # Обновить бюджеты и таймлайны
            budgets[best_sid] = max(0, budgets[best_sid] - best_ships)
            tl = timelines[best_opp.target_id]
            timelines[best_opp.target_id] = tl.with_added_arrival(FleetArrival(
                arrival_turn=best_arrival,
                owner=_inferred_player(tl, best_opp),
                ships=float(best_ships),
                fleet_id=None,
                source_planet_id=best_sid,
                purpose=best_opp.type,
            ))

            # Удаляем эту opp из тиера если она исчерпана (наш ROI после
            # коммита упадёт — это естественно). Но проще просто перезапустить
            # цикл и пересчитать.
            progress = True

    stats['tiers_used'] = sorted(stats['tiers_used'])
    return committed, timelines, budgets, stats


__all__ = ['AuctionParams', 'Bid', 'run_auction']
