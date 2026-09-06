"""
AgentSwarm — пер-планетный планировщик с аукционом ships-бюджета.

Концепция (отличие от best_attacks):

  best_attacks:        для каждой цели → один лучший план. Глобальный priority.
                        Передачи между своими (transfer) не существует как
                        первоклассного действия — supplier'ы существуют только
                        как часть pipeline.

  AgentSwarm:          КАЖДАЯ наша планета имеет локальный ranked-список
                        действий: ATTACK_T (direct/multi/pipe), TRANSFER_Q,
                        HOLD. Действия конкурируют в едином аукционе,
                        ограниченном per-planet ship budget. Сплит возникает
                        естественно: top-action использует cost ships, остаток
                        идёт в auction для другого действия.

Алгоритм по фазам:

  0. Generate candidates:
     all_plans(state, T) для каждого target T → пул direct/multi/pipe планов.
     У каждого плана уже есть `x_att`, `x_sup` = МИНИМУМ ships для победы.

  1. Stress per planet:
     stress[P] = Σ_i top_actions[P][i].margin · γ^i,  i ∈ [0, K)
     где top_actions[P] — отсортированные по margin планы где P участвует.

  2. Auction (жадный single-pass):
     sort candidates по value desc.
     для каждого: проверить что у всех акторов хватит ships в бюджете И
     target не захвачен другим планом → commit, вычесть cost из бюджетов.
     Иначе — отбросить (потенциально неудовлетворённый actor: фиксируем).

  3. Redistribute leftovers:
     для каждой P с remaining > FLOOR:
       найти ally Q с высоким stress, путь чист от солнца, ETA в окне
       → TRANSFER(P → Q, ships=min(remaining, deficit_Q + buffer))

  4. Output:
     список планов (тот же формат что и best_attacks), плюс план типа
     'transfer' для внутренних передач.

План 'transfer' имеет shape:
    {'mode': 'transfer', 'sup_id': None, 'att_id': sender, 'tgt_id': recipient,
     'x_att': ships, 'x_sup': 0, 'success': True, ...}
agent._plan_parts → одна часть (sender → recipient, ships).
agent._execute_plan_atomically → не делает defender check (mode != 'direct'),
просто отправляет флот; _aim_and_verify обработает HIT в нашу планету.

Параметры — в SwarmWeights, чтобы крутить через grid search.
"""

import math
import random as _random
import time as _t
from dataclasses import dataclass, field
from itertools import product as _product

from orbit_sim import segment_hits_sun
from force import _rendezvous_eta
from attacks import (
    all_plans, ATTACK_HORIZON, MIN_USEFUL_STRIKE, RESERVE_ON_ATT,
    SUN_SAFETY_PIPE, _plan_total_ships,
)

# ── Зональная срочность для распределения подкреплений ───────────────────
# Чем выше urgency у планеты-получателя — тем приоритетнее перебросить
# туда корабли. Frontline/contested нуждаются в подкреплениях сильнее
# тыловых bastion/rear, независимо от того есть ли у них attack-планы.
ZONE_URGENCY: dict = {
    'frontline':  3.0,
    'contested':  2.0,
    'isolated':   1.5,
    'mid':        1.0,
    'rear':       0.5,
    'bastion':    0.3,
}
_ZONE_URG_DEFAULT = 1.0   # для неизвестных меток


# ── Параметризация (для тюнинга) ────────────────────────────────────────
@dataclass
class SwarmWeights:
    """Все параметры AgentSwarm одним пакетом — для grid search."""
    # Stress
    stress_top_k:     int   = 3       # сколько top-actions берём в stress
    stress_gamma:     float = 0.6     # дисконт по rank: m_0 + γ·m_1 + γ²·m_2…
    # Action value (для сортировки в аукционе)
    eta_bonus:        float = 30.0    # tiebreaker: ближе/быстрее → выше value
    priority_bonus:   float = 1.0     # вес priority (из zones) в value
    ships_weight:     float = 4.0     # «мяч на её стороне»: log1p(ships актора) даёт буст

    # Активность: чем выше — тем сильнее «застоявшиеся» планеты тащат свои
    # действия вверх в аукционе. Idle = ships − idle_floor. Линейный буст,
    # не log: чтобы 100 ship'овая планета чувствовалась сильно мощнее 30-ti.
    # 0 = выкл (используется только log-вариант ships_weight).
    activity_weight:  float = 0.5     # вес idle-bonus (на 1 idle-корабль)
    idle_floor:       int   = 25      # ships ниже считаются «активным гарнизоном»

    # Дистанция: насколько штрафуем дальние планы. eta_term = eta_bonus / eta^(1-comfort).
    # 0 = штраф 1/eta (текущий), 1 = плоско (eta вообще не влияет).
    # 0.5 = 1/sqrt(eta) — компромисс: дальние ещё штрафуются, но не катастрофично.
    distance_comfort: float = 0.0

    # Толерантность к риску: уменьшаем SAFETY_OVERKILL у атак (буфер сверх defender).
    # Чем выше — тем больше «тонко-проходных» планов становятся success'ными:
    # на planета сейчас может быть на 2 ship'а избыточно, не атакует, мы переждали ход
    # и потеряли темп. С risk=1 owned tgt требует +1 (вместо +2), neutral +0 (вместо +1).
    # 0 = consertative current, 1 = aggressive. Дальше ставить опасно — strict-< в движке.
    risk_tolerance:  int   = 0
    # Transfer (standalone TRANSFER P→Q вне duplet'a). По умолчанию выключен:
    # supply для атак идёт только через pipeline/multi_sync (осознанный duplet),
    # defense — через agent._build_defense_plans (raw=ours, projected=enemy).
    # Если хочешь экспериментировать с диффузией — включи.
    enable_redistribute: bool = True
    transfer_horizon: float = 40.0    # макс ETA для TRANSFER (если включено)
    transfer_floor:   float = 20.0    # GARRISON_FLOOR — не отправляем ниже
    transfer_thresh:  float = 5.0     # минимум score для коммита transfer
    transfer_eta_pen: float = 0.05    # штраф за ETA в transfer-score
    transfer_buffer:  float = 1.0     # ships сверх deficit получателя
    transfer_min_ships: int = 15      # минимум ships в одной TRANSFER-партии (анти-«капельница»)
    max_transfers_per_turn: int = 2  # максимум transfer-планов за ход (антидрейн)

    # ── MCTS аукцион ──────────────────────────────────────────────────────
    # Заменяет жадный single-pass на UCT-поиск по пространству комбинаций
    # планов. Находит лучший набор когда планы конкурируют за один актор.
    # use_mcts_auction=False → старый жадный (по умолчанию, нулевой overhead).
    # Включать после профилирования: занимает часть time budget до deadline.
    use_mcts_auction:    bool  = False
    mcts_c_uct:          float = 1.414   # UCB1 exploration constant (√2)

    # ── Зональный градиент в redistribute ────────────────────────────────
    # Добавляет (urgency[Q] − urgency[P]) × zone_urgency_weight к градиенту
    # transfer-score. Направляет корабли rear/bastion → frontline/contested
    # вне зависимости от attack-stress. 0.0 = только стресс (старое поведение).
    zone_urgency_weight: float = 1.5

    # ── Фильтр мелких direct-атак ─────────────────────────────────────────
    # direct-план с x_att < min_direct_att отбрасывается ЕСЛИ цель тоже
    # крупнее порога. Это отсекает 6-кор. флоты против 50-кор. планет,
    # но оставляет легальные атаки на маленьких нейтралов (x_tgt < порога).
    min_direct_att: int = 8

    # ── Гарнизон при захвате нейтрала ────────────────────────────────────
    # Сколько кораблей сверх минимума отправлять при атаке нейтрала.
    # Пример: нейтрал = 12 кораблей → без garrison отправляем 13, прилетаем
    # с 1 кораблём → redistribute сразу планирует трансфер (долго летит,
    # блокирует проекцию). С neutral_garrison=8 отправляем 21, прилетаем
    # с 9 кораблями — гарнизон встроен. Плюс: больше кораблей = быстрее летим
    # (fleet_speed_correct зависит от кол-ва), захват приходит раньше.
    # 0 = старое поведение (минимум). Хорошее стартовое значение: 5-10.
    neutral_garrison: int = 0

    # ── Проактивный гарнизон frontline/contested (для redistribute) ───────
    # Когда transfer не привязан к конкретному unfunded-плану (нет «события»),
    # redistribute всё равно должен уметь укрепить слабые frontline/contested
    # планеты до целевого уровня.
    #
    # Цель гарнизона = production × garrison_per_prod:
    #   production=3, garrison_per_prod=8 → цель=24 кораблей
    # Дефицит = max(0, цель − current_ships). Если у планеты уже ≥ цели —
    # дополнительного трансфера нет (deficit=0, кроме unfunded-дефицита).
    #
    # 0.0 = старое поведение (только unfunded-дефицит).
    # Хорошее стартовое значение: 5–10.
    # Применяется ТОЛЬКО к зонам frontline и contested; rear/bastion не трогаем.
    garrison_per_prod: float = 0.0

    # ── Приоритет атаки по силе оппонента ────────────────────────────────
    # В FFA (и иногда в 1v1) выгодно атаковать слабого прежде сильного:
    # слабый — лёгкие планеты + устранение → меньше фронтов.
    #
    # opp_strength_weight (W):
    #   0.0 = выключено (дефолт, нет изменений)
    #   > 0 = бонус за атаку слабых / штраф за атаку сильных.
    #
    # Механика: для каждой вражеской цели вычисляем relative_strength её хозяина
    # (сила / средняя сила по всем врагам). В _action_value добавляем:
    #   opp_bonus = W * (1 - rel_strength) * eta_bonus
    # Примеры при W=0.5, eta_bonus=30:
    #   rel=0.5 (вдвое слабее): +7.5  (агрессивнее атакуем слабого)
    #   rel=1.0 (средний):        0.0  (нет изменений)
    #   rel=2.0 (вдвое сильнее): −15.0 (избегаем лезть на сильного)
    #
    # opp_prod_factor: вес производства в оценке силы.
    #   сила_i = ships_i + prod_factor * prod_i
    #   production важнее в долгосрочной перспективе, но не известен наперёд.
    #   5.0 ≈ "1 прод = 5 кораблей" (конвертируется за ~5 ходов).
    opp_strength_weight: float = 0.0
    opp_prod_factor:     float = 5.0

    # ── Priority-reclassify порог (для zones.py post-pass) ────────────────
    # prio_reclassify_thr      — порог в начале матча (step=0).
    # prio_reclassify_thr_late — порог в конце матча (step=TOTAL_STEPS).
    # agent.py линейно интерполирует между ними по фазе → stage-aware тюнинг.
    #
    # Если оба одинаковые (дефолт) — статичный порог, нет интерполяции.
    # Пример stage-aware: thr=0.4 (агрессивная ранняя экспансия) →
    #                      thr_late=1.5 (осторожно в поздней игре).
    #
    # 99.0 = фактически выключить override (никакая periphery не апгрейдится).
    # Тюнинговый диапазон: 0.4 … 1.8.
    prio_reclassify_thr:      float = 0.8
    prio_reclassify_thr_late: float = 0.8   # = thr → нет интерполяции по дефолту


DEFAULT_WEIGHTS = SwarmWeights()


# ── Helpers ─────────────────────────────────────────────────────────────

def _swarm_seg_blocked(a, b, safety=SUN_SAFETY_PIPE):
    return segment_hits_sun(a.x, a.y, b.x, b.y, safety=safety)


def _planet_by_id(state, pid):
    raw = getattr(state, 'raw_planets', state.planets)
    for p in raw:
        if p.id == pid:
            return p
    return None


def _action_actors(plan):
    """Список planet_id, чьи ships тратит этот план."""
    actors = []
    if plan.get('att_id') is not None and plan.get('x_att', 0) > 0:
        actors.append((plan['att_id'], int(plan['x_att'])))
    if plan.get('sup_id') is not None and plan.get('x_sup', 0) > 0:
        actors.append((plan['sup_id'], int(plan['x_sup'])))
    return actors


def _action_value(plan, weights, priority_lookup=None, ships_lookup=None,
                  opp_strength_lookup=None):
    """Скор плана для сортировки в аукционе.

    margin            — успешные планы положительный, fail отрицательный
    + tiebreaker по ETA (чем быстрее тем лучше)
    + бонус по priority цели (если задан priority_lookup)
    + ships_term      — «мяч на её стороне»: log1p(ships у самого нагруженного
                        актора) множится на ships_weight. Идея: если у актора
                        накопилось много, его действия должны идти первыми в
                        аукционе — иначе она годами сидит в роли supplier'a и
                        никогда не стреляет.
    + opp_bonus       — бонус за атаку слабого оппонента / штраф за сильного.
                        opp_strength_lookup: {tgt_id → relative_strength}
                        rel < 1 = слабее среднего → положительный бонус
                        rel > 1 = сильнее → отрицательный (штраф)
    """
    margin = float(plan.get('margin', 0.0))
    eta    = float(plan.get('t_total', 0.0)) + 1.0

    # eta-штраф с поправкой distance_comfort:
    #   comfort=0 → 1/eta (классика)
    #   comfort=1 → константа (eta перестаёт штрафоваться)
    #   comfort=0.5 → 1/sqrt(eta) (мягко)
    comfort = max(0.0, min(1.0, getattr(weights, 'distance_comfort', 0.0)))
    eta_term = weights.eta_bonus / (eta ** (1.0 - comfort))

    prio = 0.0
    if priority_lookup is not None:
        prio = priority_lookup.get(plan.get('tgt_id'), 0.0)

    ships_term    = 0.0
    activity_term = 0.0
    if ships_lookup is not None:
        actor_ships_list = [
            ships_lookup.get(aid, 0) for aid, _c in _action_actors(plan)
        ]
        actor_max_ships = max(actor_ships_list, default=0)
        ships_term = weights.ships_weight * math.log1p(max(0, actor_max_ships))

        # активность: насколько актор «застоялся». Берём максимум по акторам
        # (любой загруженный actor → план поднимается), линейно.
        idle_max = max(
            (max(0, s - weights.idle_floor) for s in actor_ships_list),
            default=0,
        )
        activity_term = weights.activity_weight * idle_max

    # Бонус/штраф по силе оппонента-владельца цели.
    # Масштабируется через eta_bonus — чтобы быть в той же размерности что
    # остальные слагаемые (eta_term при eta=10 → ~3.0 при eta_bonus=30).
    opp_bonus = 0.0
    _opp_w = getattr(weights, 'opp_strength_weight', 0.0)
    if _opp_w != 0.0 and opp_strength_lookup is not None:
        rel = opp_strength_lookup.get(plan.get('tgt_id'), 1.0)
        opp_bonus = _opp_w * (1.0 - rel) * weights.eta_bonus

    return (margin + eta_term + weights.priority_bonus * prio
            + ships_term + activity_term + opp_bonus)


# ── Stress ──────────────────────────────────────────────────────────────

def compute_stress(candidates, ours, weights):
    """Per-planet stress: Σ top-K marginов с дисконтом γ^i.

    Стресс — индикатор «насколько у этой планеты много полезных действий
    прямо сейчас». Высокий stress → планета нужна для атак, не отвлекаем
    её на transfer. Низкий stress → планета бесполезна сейчас, годится
    как supplier для соседей.
    """
    by_actor = {p.id: [] for p in ours}
    for plan in candidates:
        m = float(plan.get('margin', 0.0))
        for actor_id, _cost in _action_actors(plan):
            if actor_id in by_actor:
                by_actor[actor_id].append(m)

    stress = {}
    K, gamma = weights.stress_top_k, weights.stress_gamma
    for pid, ms in by_actor.items():
        ms = sorted(ms, reverse=True)[:K]
        s = 0.0
        for i, m in enumerate(ms):
            s += m * (gamma ** i)
        stress[pid] = s
    return stress


def neighbor_stress(stress, ours):
    """WNN-weighted stress соседей: Σ (1/dist) · stress[Q] / Σ (1/dist)."""
    out = {}
    for p in ours:
        wsum = 0.0
        ssum = 0.0
        for q in ours:
            if q.id == p.id:
                continue
            d = math.hypot(p.x - q.x, p.y - q.y)
            if d < 1e-6:
                continue
            w = 1.0 / d
            wsum += w
            ssum += w * stress.get(q.id, 0.0)
        out[p.id] = ssum / wsum if wsum > 0 else 0.0
    return out


# ── Auction ─────────────────────────────────────────────────────────────

def auction(candidates, ours, weights, priority_lookup=None, opp_strength_lookup=None):
    """
    Single-pass greedy: сортируем все действия по value, идём сверху,
    коммитим если у всех акторов хватит ships и target не захвачен.

    Возвращает:
      committed   — список выбранных планов
      remaining   — {pid: ships_left} после вычета cost
      captured    — set targets которые уже атакованы
      unfunded    — список планов которые не прошли из-за нехватки ships
                    (используется в redistribute как сигнал «у этого actor'a
                    был дефицит»)
    """
    # стартовый бюджет — текущие ships каждой нашей планеты (минус reserve)
    remaining = {p.id: max(0, int(p.ships) - RESERVE_ON_ATT) for p in ours}
    ships_lookup = {p.id: int(p.ships) for p in ours}

    # сортировка по value desc; стабильно — успешные планы выше
    scored = [
        (
            (1 if plan.get('success') else 0),
            _action_value(plan, weights, priority_lookup, ships_lookup,
                          opp_strength_lookup),
            i,
            plan,
        )
        for i, plan in enumerate(candidates)
    ]
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

    committed = []
    captured = set()
    unfunded = []

    for _ok, _v, _i, plan in scored:
        if not plan.get('success'):
            continue
        tgt_id = plan.get('tgt_id')
        if tgt_id in captured:
            continue

        actors = _action_actors(plan)
        if any(remaining.get(aid, 0) < cost for aid, cost in actors):
            unfunded.append(plan)
            continue
        if _plan_total_ships(plan) < MIN_USEFUL_STRIKE:
            continue

        # commit
        for aid, cost in actors:
            remaining[aid] -= cost
        captured.add(tgt_id)
        committed.append(plan)

    return committed, remaining, captured, unfunded


# ── MCTS аукцион ────────────────────────────────────────────────────────

class _ANode:
    """UCT-узел дерева аукциона.

    На глубине i дерева стоит решение по candidates[i]: включить (True)
    или пропустить (False). Каждый путь корень→лист — одна комбинация планов.
    """
    __slots__ = ('n', 'v', 'ch')
    def __init__(self):
        self.n  = 0      # число посещений
        self.v  = 0.0    # суммарная ценность backprop
        self.ch = {}     # bool → _ANode


def auction_mcts(candidates, ours, weights, priority_lookup=None,
                 opp_strength_lookup=None, time_budget=0.05, c_uct=1.414):
    """UCT-аукцион: ищет лучшую комбинацию планов вместо жадного прохода.

    Зачем: жадный single-pass проигрывает когда два плана делят один актор.
    Пример: план A (ценный, актор P) выбирается первым и блокирует планы
    B+C (меньше каждый, но сумма > A), которые оба могут пройти без A.
    MCTS исследует пространство include/skip и находит B+C.

    Алгоритм:
      1. Кандидаты сортируются по value (как в greedy).
      2. UCT-дерево: на глубине i — решение include/skip для candidates[i].
      3. Rollout из текущего узла: жадный проход до конца списка.
      4. Обновляем best-solution если rollout дал лучший суммарный value.
      5. Backprop: обновляем n/v всех узлов пути.

    Возвращает тот же интерфейс что auction().
    """
    budget0  = {p.id: max(0, int(p.ships) - RESERVE_ON_ATT) for p in ours}
    ships_lk = {p.id: int(p.ships) for p in ours}

    valid = sorted(
        [p for p in candidates
         if p.get('success') and _plan_total_ships(p) >= MIN_USEFUL_STRIKE],
        key=lambda p: -_action_value(p, weights, priority_lookup, ships_lk,
                                     opp_strength_lookup),
    )
    n = len(valid)
    if not n:
        return [], dict(budget0), set(), []

    pvals = [_action_value(p, weights, priority_lookup, ships_lk,
                           opp_strength_lookup) for p in valid]

    # Лучшее решение среди всех rollout'ов (инициализируется greedy baseline)
    best = {'v': -1.0, 'comm': [], 'rem': dict(budget0), 'cap': set()}

    def _rollout(idx, rem, cap, v0, comm0):
        """Жадный rollout с позиции idx; обновляет best если нашли лучше."""
        r = dict(rem); c = set(cap); comm = list(comm0); v = v0
        for i in range(idx, n):
            p = valid[i]; tgt = p.get('tgt_id')
            if tgt in c:
                continue
            acts = _action_actors(p)
            if any(r.get(a, 0) < cost for a, cost in acts):
                continue
            for a, cost in acts:
                r[a] -= cost
            c.add(tgt); comm.append(p); v += pvals[i]
        if v > best['v']:
            best.update(v=v, comm=comm[:], rem=r, cap=set(c))
        return v

    # Seed: чисто жадный baseline (гарантирует не хуже старого поведения)
    _rollout(0, budget0, set(), 0.0, [])

    root  = _ANode()
    t_end = _t.perf_counter() + time_budget

    while _t.perf_counter() < t_end:
        # ── Selection + Expansion ────────────────────────────────────
        node     = root
        rem      = dict(budget0)
        cap      = set()
        acc_v    = 0.0
        acc_comm = []
        path     = [root]   # узлы для backprop
        idx      = 0

        while idx < n:
            p    = valid[idx]
            tgt  = p.get('tgt_id')
            acts = _action_actors(p)
            can_inc = (tgt not in cap
                       and all(rem.get(a, 0) >= cost for a, cost in acts))
            avail = [False] + ([True] if can_inc else [])

            # Нераскрытые дети → expansion
            unexp = [a for a in avail if a not in node.ch]
            if unexp:
                action = _random.choice(unexp)
                child  = _ANode()
                node.ch[action] = child
                if action:   # include
                    for a, cost in acts:
                        rem[a] -= cost
                    cap.add(tgt); acc_v += pvals[idx]; acc_comm.append(p)
                path.append(child)
                node = child
                idx += 1
                break        # один expansion → rollout

            # UCT-выбор среди уже открытых детей
            log_n = math.log(max(1, node.n))
            best_a, best_s = None, -1e18
            for a in avail:
                ch = node.ch[a]
                s  = (ch.v / ch.n + c_uct * math.sqrt(log_n / ch.n)
                      if ch.n > 0 else 1e18)
                if s > best_s:
                    best_s = s; best_a = a

            if best_a and can_inc:
                for a, cost in acts:
                    rem[a] -= cost
                cap.add(tgt); acc_v += pvals[idx]; acc_comm.append(p)

            node = node.ch[best_a]
            path.append(node)
            idx += 1

        # ── Rollout & backprop ───────────────────────────────────────
        total = _rollout(idx, rem, cap, acc_v, acc_comm)
        for nd in path:
            nd.n += 1
            nd.v += total

    comm     = best['comm']
    unfunded = [p for p in valid if p not in comm]
    return comm, best['rem'], best['cap'], unfunded


# ── Redistribute (TRANSFER) ─────────────────────────────────────────────

def redistribute(state, remaining, stress, neigh_stress, unfunded, ours, weights,
                 zone_lookup=None):
    """
    Для каждой P с остатком ships > floor — оценить TRANSFER в соседние Q
    и выпустить план если score выше порога.

    Score для P → Q:
        gradient = stress_grad + zone_urgency_weight · (urgency[Q] - urgency[P])
        score    = gradient · ships / (1 + eta · eta_pen)
    где ships ≤ remaining[P] - floor, ограниченное deficit_Q + buffer.

    zone_lookup: dict pid → zone-метка (из zones.py). Если задан — градиент
    включает разницу зональной срочности (ZONE_URGENCY): корабли rear/bastion
    автоматически тянутся к frontline/contested даже при нулевом attack-stress.
    Если не задан — поведение идентично старому (только stress-градиент).

    deficit_Q собирается из unfunded actions: если Q был supplier'ом или
    attacker'ом в плане который не прошёл из-за нехватки ships у Q —
    значит ему недостаёт. Если Q не имеет unfunded — buffer = 0, лимит —
    только базовый buffer из weights.
    """
    floor = weights.transfer_floor
    horizon = weights.transfer_horizon

    # сколько каждому Q не хватило в auction (по cost кораблей у Q)
    deficit = {p.id: 0.0 for p in ours}
    for plan in unfunded:
        for aid, cost in _action_actors(plan):
            d = max(0.0, cost - remaining.get(aid, 0))
            deficit[aid] = max(deficit.get(aid, 0.0), d)

    # ── Проактивный гарнизонный дефицит ──────────────────────────────────
    # Для frontline/contested-планет без unfunded-события: задаём целевой
    # минимальный гарнизон = production × garrison_per_prod. Если планета
    # ниже этого порога — считаем разницу «дефицитом» и направляем трансфер.
    # Это закрывает случай «нет конкретного события, но планета слабая».
    _gprod = float(getattr(weights, 'garrison_per_prod', 0.0))
    if _gprod > 0 and zone_lookup is not None:
        for Q in ours:
            zone = zone_lookup.get(Q.id, 'mid')
            if zone in ('frontline', 'contested'):
                target_garrison = _gprod * float(Q.production or 1)
                garrison_gap    = max(0.0, target_garrison - float(Q.ships))
                if garrison_gap > deficit.get(Q.id, 0.0):
                    deficit[Q.id] = garrison_gap

    transfer_plans = []
    by_id = {p.id: p for p in ours}
    _max_tr = int(getattr(weights, 'max_transfers_per_turn', 2))

    for P in ours:
        if len(transfer_plans) >= _max_tr:
            break
        free = remaining.get(P.id, 0) - floor
        if free <= 0:
            continue
        if int(free) < MIN_USEFUL_STRIKE:
            continue

        candidates = []
        for Q in ours:
            if Q.id == P.id:
                continue
            if _swarm_seg_blocked(P, Q):
                continue
            eta_pq, _ = _rendezvous_eta(P, Q, max(1, int(free)), state.omega)
            if eta_pq > horizon:
                continue

            stress_grad = stress.get(Q.id, 0.0) - stress.get(P.id, 0.0)
            if zone_lookup is not None:
                urg_q    = ZONE_URGENCY.get(zone_lookup.get(Q.id, 'mid'), _ZONE_URG_DEFAULT)
                urg_p    = ZONE_URGENCY.get(zone_lookup.get(P.id, 'mid'), _ZONE_URG_DEFAULT)
                gradient = stress_grad + weights.zone_urgency_weight * (urg_q - urg_p)
            else:
                gradient = stress_grad
            if gradient <= 0:
                continue
            # сколько имеет смысл отправить
            max_useful = deficit.get(Q.id, 0.0) + weights.transfer_buffer
            ships = int(min(free, max_useful))
            if ships < MIN_USEFUL_STRIKE:
                continue

            score = gradient * ships / (1.0 + eta_pq * weights.transfer_eta_pen)
            candidates.append((score, ships, eta_pq, Q))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])
        best_score, ships, eta_pq, Q = candidates[0]
        if best_score < weights.transfer_thresh:
            continue
        # анти-«капельница»: маленькие партии не имеют смысла, планета должна
        # либо отправить значимый кусок, либо копить дальше
        if ships < int(getattr(weights, 'transfer_min_ships', MIN_USEFUL_STRIKE)):
            continue

        # эмитим план в формате совместимом с agent._execute_plan_atomically
        transfer_plans.append({
            'mode':       'transfer',
            'sup_id':     None,
            'att_id':     P.id,
            'tgt_id':     Q.id,
            'eta_sa':     0.0,
            'eta_at':     float(eta_pq),
            't_total':    float(eta_pq),
            'x_sup':      0,
            'x_att':      ships,
            'prod_att':   float(P.production),
            'x_tgt':      0,
            'prod_tgt':   0.0,
            'strike':     float(ships),
            'defender':   0.0,
            'incoming':   0.0,
            'needed':     0,
            'margin':     float(ships),
            'success':    True,
            'relay_reason': None,
            'transfer_score':   best_score,
            'transfer_gradient': stress.get(Q.id, 0.0) - stress.get(P.id, 0.0),
            'is_transfer':      True,
        })
        remaining[P.id] -= ships

    return transfer_plans


# ── Главная точка входа ────────────────────────────────────────────────

def swarm_plan(state, player, targets, weights=None, priority_lookup=None,
               zone_lookup=None, horizon=ATTACK_HORIZON, max_targets=None, deadline=None):
    """
    Полный цикл AgentSwarm: candidates → stress → auction → redistribute.

    max_targets: жёсткий cap на число целей для перебора пар (None = все).
                 Если cap, берём топ по priority_lookup.
    deadline:    time.perf_counter() граница в секундах. После неё прекращаем
                 наращивать candidates (что собрали — то собрали).

    Возвращает (plans, debug):
      plans  — финальный список планов
      debug  — словарь метаинформации (stress, остатки, разбивка)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = getattr(state, 'raw_planets', state.planets)
    ours = [p for p in raw if p.owner == player]
    if not ours or not targets:
        return [], {'reason': 'no ours or no targets'}

    # ── Opponent strength lookup (для opp_strength_weight) ────────────────
    # Считаем силу каждого противника: ships + prod_factor * production.
    # Нормализуем к среднему среди врагов → rel_strength = 1.0 означает средний враг.
    # opp_bonus в _action_value = W * (1.0 - rel_strength) * eta_bonus:
    #   rel < 1 (слабый) → bonus > 0 (атакуем охотнее)
    #   rel > 1 (сильный) → bonus < 0 (осторожнее)
    opp_strength_lookup: dict = {}
    _opp_w = getattr(weights, 'opp_strength_weight', 0.0)
    if _opp_w != 0.0:
        prod_factor = getattr(weights, 'opp_prod_factor', 5.0)
        # Суммируем корабли и производство по owner (планеты + флоты)
        opp_ships: dict = {}
        opp_prod:  dict = {}
        for p in raw:
            own = p.owner
            if own == player or own < 0:
                continue
            opp_ships[own] = opp_ships.get(own, 0.0) + float(getattr(p, 'ships', 0) or 0)
            opp_prod[own]  = opp_prod.get(own,  0.0) + float(getattr(p, 'production', 0) or 0)
        # Флоты тоже учитываем
        for f in getattr(state, 'fleets', []):
            own = getattr(f, 'owner', -1)
            if own == player or own < 0:
                continue
            opp_ships[own] = opp_ships.get(own, 0.0) + float(getattr(f, 'ships', 0) or 0)
        # Суммарная сила каждого врага
        opp_ids = set(opp_ships) | set(opp_prod)
        if opp_ids:
            strength = {oid: opp_ships.get(oid, 0.0) + prod_factor * opp_prod.get(oid, 0.0)
                        for oid in opp_ids}
            mean_s = sum(strength.values()) / len(strength)
            if mean_s > 0:
                rel = {oid: s / mean_s for oid, s in strength.items()}
            else:
                rel = {oid: 1.0 for oid in opp_ids}
            # Строим lookup: tgt_id (planet id) → rel_strength его owner'а
            for p in raw:
                own = p.owner
                if own in rel:
                    opp_strength_lookup[p.id] = rel[own]

    # cap по числу целей (на ход с 30+ нейтралами это спасает от per-step timeout)
    if max_targets is not None and len(targets) > max_targets:
        if priority_lookup:
            targets = sorted(targets, key=lambda t: -priority_lookup.get(t.id, 0))[:max_targets]
        else:
            targets = list(targets)[:max_targets]

    # 0. Все боевые candidates от существующего движка attacks.py
    risk = int(getattr(weights, 'risk_tolerance', 0))
    min_dir_att = int(getattr(weights, 'min_direct_att', 8))

    # Кеш заблокированных солнцем пар (src_id, tgt_id) — строится один раз
    # на весь ход. Устраняет >1000 бесполезных aim_verify_failed за матч:
    # _seg_blocked использует SUN_SAFETY_PIPE (консервативный радиус), поэтому
    # маршруты из кеша никогда не пройдут _aim_and_verify в agent.py.
    blocked_pairs: set = set()
    for _src in ours:
        for _tgt in targets:
            if _swarm_seg_blocked(_src, _tgt):
                blocked_pairs.add((_src.id, _tgt.id))

    candidates = []
    for tgt in targets:
        if deadline is not None and _t.perf_counter() > deadline:
            break  # бюджет исчерпан, играем что собрали
        # Только планеты с незаблокированным прямым маршрутом до цели
        reachable = [p for p in ours if (p.id, tgt.id) not in blocked_pairs]
        if not reachable:
            continue
        plans = all_plans(state, tgt, reachable, horizon=horizon, player=player, risk=risk,
                          neutral_garrison=int(getattr(weights, 'neutral_garrison', 0)))
        for pl in plans:
            if not (pl.get('success') and _plan_total_ships(pl) >= MIN_USEFUL_STRIKE):
                continue
            # Фильтр мелких direct-атак: x_att < порога против крупной цели —
            # флот всё равно не победит и только теряется. Нейтралов с малым
            # гарнизоном (x_tgt < порога) не трогаем — там 6 кор. нормально.
            if pl.get('mode') == 'direct':
                x_att = int(pl.get('x_att', 0))
                x_tgt = float(pl.get('x_tgt', 0))
                if x_att < min_dir_att and x_tgt >= min_dir_att:
                    continue
            candidates.append(pl)

    # 1. Stress / neighbor_stress (для отладки и transfer-scoring)
    stress = compute_stress(candidates, ours, weights)
    neigh  = neighbor_stress(stress, ours)

    # 2. Аукцион: жадный или UCT-MCTS
    if getattr(weights, 'use_mcts_auction', False):
        # Выделяем до 30% оставшегося бюджета на MCTS
        mcts_budget = 0.05
        if deadline is not None:
            rem_time    = deadline - _t.perf_counter()
            mcts_budget = max(0.02, rem_time * 0.30)
        committed, remaining, captured, unfunded = auction_mcts(
            candidates, ours, weights, priority_lookup=priority_lookup,
            opp_strength_lookup=opp_strength_lookup,
            time_budget=mcts_budget,
            c_uct=getattr(weights, 'mcts_c_uct', 1.414),
        )
    else:
        committed, remaining, captured, unfunded = auction(
            candidates, ours, weights, priority_lookup=priority_lookup,
            opp_strength_lookup=opp_strength_lookup,
        )

    # 3. Redistribute остатки в TRANSFER.
    #    По умолчанию ВЫКЛЮЧЕН — supply для атак идёт через pipeline/multi
    #    (осознанный duplet), defense через agent._build_defense_plans.
    #    Standalone-передачи легко превращают планету в «вечного supplier'a»,
    #    она копит-капает и никогда не атакует. Включай только осознанно.
    if getattr(weights, 'enable_redistribute', False):
        transfers = redistribute(
            state, remaining, stress, neigh, unfunded, ours, weights,
            zone_lookup=zone_lookup,
        )
    else:
        transfers = []

    plans = committed + transfers

    debug = {
        'n_candidates':   len(candidates),
        'n_committed':    len(committed),
        'n_transfers':    len(transfers),
        'n_unfunded':     len(unfunded),
        'stress':         stress,
        'neighbor_stress': neigh,
        'remaining':      dict(remaining),
        'captured':       sorted(captured),
    }
    return plans, debug


__all__ = [
    'SwarmWeights', 'DEFAULT_WEIGHTS',
    'ZONE_URGENCY',
    'compute_stress', 'neighbor_stress',
    'auction', 'auction_mcts', 'redistribute', 'swarm_plan',
]
