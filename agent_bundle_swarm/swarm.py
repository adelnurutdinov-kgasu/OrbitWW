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
from dataclasses import dataclass, field
from itertools import product as _product

from orbit_sim import segment_hits_sun
from force import _rendezvous_eta
from attacks import (
    all_plans, ATTACK_HORIZON, MIN_USEFUL_STRIKE, RESERVE_ON_ATT,
    SUN_SAFETY_PIPE, _plan_total_ships,
)


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
    enable_redistribute: bool = False
    transfer_horizon: float = 40.0    # макс ETA для TRANSFER (если включено)
    transfer_floor:   float = 20.0    # GARRISON_FLOOR — не отправляем ниже
    transfer_thresh:  float = 5.0     # минимум score для коммита transfer
    transfer_eta_pen: float = 0.05    # штраф за ETA в transfer-score
    transfer_buffer:  float = 1.0     # ships сверх deficit получателя
    transfer_min_ships: int = 15      # минимум ships в одной TRANSFER-партии (анти-«капельница»)


DEFAULT_WEIGHTS = SwarmWeights()


# ── Helpers ─────────────────────────────────────────────────────────────

def _seg_blocked(a, b, safety=SUN_SAFETY_PIPE):
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


def _action_value(plan, weights, priority_lookup=None, ships_lookup=None):
    """Скор плана для сортировки в аукционе.

    margin            — успешные планы положительный, fail отрицательный
    + tiebreaker по ETA (чем быстрее тем лучше)
    + бонус по priority цели (если задан priority_lookup)
    + ships_term      — «мяч на её стороне»: log1p(ships у самого нагруженного
                        актора) множится на ships_weight. Идея: если у актора
                        накопилось много, его действия должны идти первыми в
                        аукционе — иначе она годами сидит в роли supplier'a и
                        никогда не стреляет.
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

    ships_term   = 0.0
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

    return (margin + eta_term + weights.priority_bonus * prio
            + ships_term + activity_term)


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

def auction(candidates, ours, weights, priority_lookup=None):
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
            _action_value(plan, weights, priority_lookup, ships_lookup),
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


# ── Redistribute (TRANSFER) ─────────────────────────────────────────────

def redistribute(state, remaining, stress, neigh_stress, unfunded, ours, weights):
    """
    Для каждой P с остатком ships > floor — оценить TRANSFER в соседние Q
    и выпустить план если score выше порога.

    Score для P → Q:
        score = max(0, stress[Q] - stress[P]) · ships / (1 + eta · eta_pen)
    где ships ≤ remaining[P] - floor, ограниченное deficit_Q + buffer.

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

    transfer_plans = []
    by_id = {p.id: p for p in ours}

    for P in ours:
        free = remaining.get(P.id, 0) - floor
        if free <= 0:
            continue
        if int(free) < MIN_USEFUL_STRIKE:
            continue

        candidates = []
        for Q in ours:
            if Q.id == P.id:
                continue
            if _seg_blocked(P, Q):
                continue
            eta_pq, _ = _rendezvous_eta(P, Q, max(1, int(free)), state.omega)
            if eta_pq > horizon:
                continue

            gradient = stress.get(Q.id, 0.0) - stress.get(P.id, 0.0)
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
               horizon=ATTACK_HORIZON, max_targets=None, deadline=None):
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
    import time as _t
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = getattr(state, 'raw_planets', state.planets)
    ours = [p for p in raw if p.owner == player]
    if not ours or not targets:
        return [], {'reason': 'no ours or no targets'}

    # cap по числу целей (на ход с 30+ нейтралами это спасает от per-step timeout)
    if max_targets is not None and len(targets) > max_targets:
        if priority_lookup:
            targets = sorted(targets, key=lambda t: -priority_lookup.get(t.id, 0))[:max_targets]
        else:
            targets = list(targets)[:max_targets]

    # 0. Все боевые candidates от существующего движка attacks.py
    risk = int(getattr(weights, 'risk_tolerance', 0))
    candidates = []
    for tgt in targets:
        if deadline is not None and _t.perf_counter() > deadline:
            break  # бюджет исчерпан, играем что собрали
        plans = all_plans(state, tgt, ours, horizon=horizon, player=player, risk=risk)
        for pl in plans:
            if pl.get('success') and _plan_total_ships(pl) >= MIN_USEFUL_STRIKE:
                candidates.append(pl)

    # 1. Stress / neighbor_stress (для отладки и transfer-scoring)
    stress = compute_stress(candidates, ours, weights)
    neigh  = neighbor_stress(stress, ours)

    # 2. Аукцион
    committed, remaining, captured, unfunded = auction(
        candidates, ours, weights, priority_lookup=priority_lookup,
    )

    # 3. Redistribute остатки в TRANSFER.
    #    По умолчанию ВЫКЛЮЧЕН — supply для атак идёт через pipeline/multi
    #    (осознанный duplet), defense через agent._build_defense_plans.
    #    Standalone-передачи легко превращают планету в «вечного supplier'a»,
    #    она копит-капает и никогда не атакует. Включай только осознанно.
    if getattr(weights, 'enable_redistribute', False):
        transfers = redistribute(
            state, remaining, stress, neigh, unfunded, ours, weights,
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
    'compute_stress', 'neighbor_stress',
    'auction', 'redistribute', 'swarm_plan',
]
