"""
opportunities.py — L2 пайплайна.

Сканирует таймлайны и выделяет «возможности» — атомарные единицы
спроса, которые планировщик может закрывать кораблями.

Типы возможностей:
  • capture     — планета не наша, физически достижима, мы можем взять её.
  • defend      — наша планета теряется в момент t; нужно прислать ships.
  • accelerate  — становится нашей в момент t_old > now + THRESHOLD;
                  можем ускорить захват.
  • snipe       — нейтральная переходит к противнику в момент t;
                  мы могли бы успеть раньше.
  • recapture   — наша планета теряется и в горизонте противник её удерживает;
                  можем вернуть.

Каждая возможность несёт:
  • value_function(ships) → ships_equivalent ценности
  • deadline             — поздняя граница полезного прибытия
  • min_ships            — нижний порог, ниже которого value=0
  • tier                 — приоритетный тиер (см. ниже)

Тиеры:
  0 — defend critical (мы теряем планету)
  1 — capture (расширение)
  2 — snipe / recapture (перехват/возврат)
  3 — accelerate (ускорение того, что и так будет нашим)
  4 — reinforce safety (буфер для тонко выигранных захватов)
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

from .timeline import PlanetTimeline, NEUTRAL


# ── Параметризация ────────────────────────────────────────────────────

@dataclass
class OppParams:
    horizon: int = 200
    accelerate_min_delay: int = 8        # ускоряем если флип > now + это
    safety_threshold: float = 5.0         # хрупкий захват если margin < это
    accelerate_value_discount: float = 0.8  # дисконт ценности «ускорений»


# ── Opportunity ────────────────────────────────────────────────────────

@dataclass
class Opportunity:
    type: str
    target_id: int
    tier: int                # 0..4
    deadline: int            # последний полезный turn прибытия (АБСОЛЮТНЫЙ)
    horizon_end_abs: int     # абсолютный turn конца окна планирования
    min_ships: float         # ниже — value=0
    naked_owner_at_deadline: int  # кто был бы без нашего вмешательства
    current_eta: Optional[int] = None  # для accelerate — спроецированный turn (АБСОЛЮТНЫЙ)
    production: float = 0.0
    distance_to_player: float = 0.0  # средняя дистанция до наших планет (для приоритезации)
    extra: Dict[str, Any] = field(default_factory=dict)

    def value(self, ships: float, arrival_turn: int, params: OppParams) -> float:
        """Сколько производственных юнитов мы получим, исполнив эту возможность
        с данным числом кораблей, прибывающих в `arrival_turn` (АБСОЛЮТНЫЙ).

        Возвращаем в «эквивалентных кораблях»: production × сколько-ходов-владения.
        Это та же валюта, что и кост в кораблях, и поэтому ROI = value / ships
        напрямую сопоставим между типами возможностей.
        """
        if arrival_turn > self.deadline:
            return 0.0
        if ships < self.min_ships:
            return 0.0

        # horizon_left — сколько ходов производства мы получим с момента arrival
        # до конца окна планирования. ВСЁ В АБСОЛЮТНЫХ ТУРНАХ.
        horizon_left = max(0, self.horizon_end_abs - arrival_turn)

        if self.type == "capture":
            return self.production * horizon_left

        if self.type == "snipe":
            # Снайп: захват + штраф противнику (тот не получает свою долю)
            return 1.5 * self.production * horizon_left

        if self.type == "defend":
            # Оборона: предотвращаем потерю → производство от lost до horizon
            lost_turn = self.extra.get('lost_turn', self.deadline)
            return self.production * max(0, self.horizon_end_abs - lost_turn)

        if self.type == "accelerate":
            # Ускорение: gain = production × (old_eta - new_arrival)
            old_eta = self.current_eta if self.current_eta is not None else self.deadline
            delta_t = max(0, old_eta - arrival_turn)
            if delta_t <= 0:
                return 0.0
            return params.accelerate_value_discount * self.production * delta_t

        if self.type == "recapture":
            return self.production * horizon_left

        if self.type == "reinforce_safety":
            return 0.3 * self.production * horizon_left

        return 0.0


# ── Извлечение ─────────────────────────────────────────────────────────

def _avg_distance_to_player(p, our_planets) -> float:
    if not our_planets:
        return 1.0
    return sum(math.hypot(p.x - q.x, p.y - q.y) for q in our_planets) / len(our_planets)


def extract_opportunities(timelines: Dict[int, PlanetTimeline],
                          state,
                          current_turn: int,
                          player: int,
                          params: Optional[OppParams] = None
                          ) -> List[Opportunity]:
    """
    Главный экстрактор.

    Идём по каждой планете и решаем, какие возможности она порождает:
      • Если ours_at(now) и lost_at(player) < horizon → defend (или recapture).
      • Если not_ours_at(now) и naked показывает enemy → snipe.
      • Если not_ours_at(now) и becomes_ours_at(player) есть → accelerate.
      • Если not_ours_at(now) и не становится нашей → capture.
    """
    if params is None:
        params = OppParams()

    raw = getattr(state, 'raw_planets', None) or state.planets
    by_id = {p.id: p for p in raw}
    our_planets = [p for p in raw if p.owner == player]

    opps: List[Opportunity] = []
    horizon_abs = current_turn + params.horizon

    for pid, tl in timelines.items():
        p = by_id.get(pid)
        if p is None:
            continue

        # naked = что было бы, если бы мы вообще ничего не отправили
        naked = tl.without_owner_fleets(player)
        naked_owner_end = naked.owner_at(horizon_abs)
        naked_owner_now = naked.owner_at(current_turn)

        owner_now = tl.owner_at(current_turn)
        becomes_ours = tl.becomes_ours_at(player)
        lost = tl.lost_at(player)
        dist_to_us = _avg_distance_to_player(p, our_planets)

        # Production: 0 для нейтрала, иначе — нормальная.
        prod = float(p.production) if naked_owner_now != NEUTRAL else float(p.production)

        # === DEFEND ===
        if owner_now == player and lost is not None and lost <= horizon_abs:
            # Оценим, сколько кораблей нужно: сравним нас в lost-1 с врагом.
            #   our_ships_at(lost-1) — наш гарнизон ровно перед боем
            #   "врага ships" — это размер прибывающего арривала
            # Найдём максимальный вражеский арривал в окрестности lost
            enemy_arr = 0.0
            for a in tl.arrivals:
                if a.owner != player and abs(a.arrival_turn - lost) <= 1:
                    if a.ships > enemy_arr:
                        enemy_arr = a.ships
            our_at_lost = tl.ships_at(lost - 1) if lost > current_turn else float(p.ships)
            need = max(1.0, enemy_arr - our_at_lost + 1.0)
            opps.append(Opportunity(
                type="defend",
                target_id=pid,
                tier=0,
                deadline=lost - 1,
                horizon_end_abs=horizon_abs,
                min_ships=need,
                naked_owner_at_deadline=naked.owner_at(lost),
                production=prod,
                distance_to_player=dist_to_us,
                extra={'lost_turn': lost,
                       'incoming_enemy_ships': enemy_arr,
                       'our_at_lost': our_at_lost},
            ))
            continue

        # === Если планета сейчас уже наша ===
        if owner_now == player:
            margin = tl.safety_margin(player)
            if margin is not None and margin < params.safety_threshold:
                opps.append(Opportunity(
                    type="reinforce_safety",
                    target_id=pid,
                    tier=4,
                    deadline=horizon_abs,
                    horizon_end_abs=horizon_abs,
                    min_ships=1.0,
                    naked_owner_at_deadline=naked_owner_end,
                    production=prod,
                    distance_to_player=dist_to_us,
                    extra={'margin': margin},
                ))
            continue

        # === Планета сейчас НЕ наша ===

        if becomes_ours is not None:
            # ACCELERATE: если становится нашей, но не скоро
            delay = becomes_ours - current_turn
            if delay >= params.accelerate_min_delay:
                opps.append(Opportunity(
                    type="accelerate",
                    target_id=pid,
                    tier=3,
                    deadline=becomes_ours - 1,
                    horizon_end_abs=horizon_abs,
                    min_ships=1.0,
                    naked_owner_at_deadline=naked.owner_at(becomes_ours),
                    current_eta=becomes_ours,
                    production=prod,
                    distance_to_player=dist_to_us,
                    extra={'delay': delay},
                ))
            continue

        # Не становится нашей сама по себе
        if naked_owner_end == player:
            continue

        # SNIPE: противник захватывает планету в горизонте
        if naked_owner_end != naked_owner_now and naked_owner_end != NEUTRAL \
                and naked_owner_end != player:
            takeover_turn = naked.first_transition_to(naked_owner_end, current_turn)
            if takeover_turn is not None:
                opps.append(Opportunity(
                    type="snipe",
                    target_id=pid,
                    tier=2,
                    deadline=takeover_turn - 1,
                    horizon_end_abs=horizon_abs,
                    min_ships=max(1.0, naked.ships_at(takeover_turn) + 1.0),
                    naked_owner_at_deadline=naked_owner_end,
                    production=prod,
                    distance_to_player=dist_to_us,
                    extra={'enemy_takeover': takeover_turn,
                           'enemy_owner': naked_owner_end},
                ))
                continue

        # CAPTURE: классический захват
        # min_ships оцениваем как ships_at(horizon_end_abs) у naked + 1.
        # Если naked owner — наш, значит ships_at вернёт нашу проекцию: тогда
        # opp всё равно прошла бы как accelerate выше; сюда попадаем когда
        # планета остаётся не-нашей. Производство нейтрала по правилам не
        # копится; для вражеской ситуация сложнее, но min_ships как
        # naked.ships_at(deadline) — корректная нижняя граница.
        min_ships = max(1.0, naked.ships_at(horizon_abs) + 1.0)
        opps.append(Opportunity(
            type="capture",
            target_id=pid,
            tier=1,
            deadline=horizon_abs,
            horizon_end_abs=horizon_abs,
            min_ships=min_ships,
            naked_owner_at_deadline=naked_owner_end,
            production=prod,
            distance_to_player=dist_to_us,
        ))

    return opps


def opportunities_summary(opps: List[Opportunity]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for o in opps:
        out[o.type] = out.get(o.type, 0) + 1
    return out


__all__ = ['OppParams', 'Opportunity',
           'extract_opportunities', 'opportunities_summary']
