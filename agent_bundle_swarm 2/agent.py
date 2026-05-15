"""
Агент Orbit Wars.

Каждый ход:
  0. Проекция     — project_state резолвит уже летящие флоты (свои+чужие):
                    каждая планета видится в момент последнего прибытия.
  [РАННЯЯ ИГРА — MCTS]
  0.5. MCTS       — если step < EARLY_GAME_SWITCH и планет <= EARLY_MAX_PLANETS,
                    запускаем run_mcts в ПАР (Пространство Адекватных Решений).
                    Возвращает лучший Action напрямую, минуя зонирование/атаки.
  [ОБЫЧНАЯ ЛОГИКА]
  1. Зонирование  — compute_zones_from_state определяет зону каждой планеты
  2. Выбор целей  — easy_target / priority_target с лучшим priority
  3. Атаки        — best_attacks подбирает план (direct/multi_sync/pipeline) с
                    минимумом необходимых кораблей
  4. Наведение    — aim_angle (гибридный: fixpoint + sweep + safe-aim)
  5. Запуск       — атомарный: план исполняется ЦЕЛИКОМ либо не исполняется
                    вообще. Это гарантирует что любой запущенный флот меняет
                    стейт планеты-цели (один или совместно с sup-частью), либо
                    является передачей на нашу планету (pipeline sup→att).
"""

import math
import dataclasses as _dc
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from orbit_sim import GameState
from projection import project_state, PROJECTION_HORIZON, NEUTRAL_OWNER
from zones import compute_zones_from_state, PRIO_RECLASSIFY_THR as _PRIO_RECLASSIFY_THR_DEFAULT
from attacks import best_attacks, all_plans, ATTACK_HORIZON
from shooting import aim_hybrid, simulate_launch, segment_hits_sun, TOTAL_STEPS as _TOTAL_STEPS
from force import _rendezvous_eta
from swarm import swarm_plan, DEFAULT_WEIGHTS
from context import compute_context, context_summary
import agent_debug as _dbg

# ── MCTS / ПАР (ранняя игра) ──────────────────────────────────────────────
try:
    from action_space import generate_actions, generate_opponent_actions
    from mcts import run_mcts, OpponentModel
    _MCTS_AVAILABLE = True
except ImportError as _mcts_import_err:
    _MCTS_AVAILABLE = False

# ── MCTS параметры (ранняя игра) ─────────────────────────────────────────
# MCTS включается когда: ход < EARLY_GAME_SWITCH И планет <= EARLY_MAX_PLANETS
# Чтобы отключить глобально — выставить USE_MCTS = False.
USE_MCTS             = True
EARLY_GAME_SWITCH    = 35    # первые N ходов → MCTS (затем обычная логика)
EARLY_MAX_PLANETS    = 20    # не более M планет на карте → MCTS (защита от больших карт)
MCTS_TIME_FRACTION   = 0.65  # доля оставшегося бюджета, отдаваемая MCTS

# Режим генерации ПАР: "none" | "v1_plain" | "v2_pair" | "v3_partial" | "v4_hybrid_net"
# Управляется переменной окружения ORBIT_MCTS_MODE (по умолчанию v1_plain).
MCTS_MODE = _os.environ.get("ORBIT_MCTS_MODE", "v1_plain")

# Глобальная модель противника (живёт между ходами одного матча).
# Инициализируется лениво при первом вызове _agent_impl.
_opponent_model   = None
_pending_partials = None   # для v3: список PartialAction в полёте

# ── Байесовский предсказатель поведения противника ───────────────────────
# Включается флагом USE_OPPONENT_PREDICTION.
# Добавляет виртуальные флоты противника в state_raw перед project_state,
# что заставляет swarm_plan учитывать ожидаемые атаки при планировании.
try:
    import opponent_model_bayesian as _opp_bayes_mod
    from opponent_model_bayesian import OpponentModelBayesian
    from opponent_presets import PRESETS
    _BAYES_AVAILABLE = True
except ImportError:
    _BAYES_AVAILABLE = False
    _opp_bayes_mod   = None

USE_OPPONENT_PREDICTION = True   # выключить → False
BAYES_THRESHOLD         = 0.05   # минимальный вес пресета для добавления флота
_VIRTUAL_FLEET_ID_BASE  = -1000  # начало диапазона ID виртуальных флотов

# Состояние между ходами (для детекции новых флотов противника)
_bayes_model    = None   # OpponentModelBayesian
_prev_fleet_ids = None   # set[int] — fleet IDs на конец предыдущего хода
_prev_state_raw = None   # GameState — raw state предыдущего хода (для update)

# ── AgentSwarm switch ─────────────────────────────────────────────────
# True  → планы строим через swarm_plan (per-planet auction + redistribute)
# False → старая глобальная логика best_attacks (fallback / A-B сравнение)
USE_SWARM      = True
SWARM_WEIGHTS  = DEFAULT_WEIGHTS

TARGET_ZONES   = ('easy_target', 'priority_target')

# Кеш зон прошлого хода для [ZONE_FLIP] лога (per-match, сбрасывается на TURN 0)
_prev_zones: dict = {}  # pid → zone
RESERVE_ON_ATT = 0     # минимум кораблей оставить на атакере.
                       # ОБЯЗАН совпадать с attacks.RESERVE_ON_ATT — иначе
                       # планировщик считает available=att.ships, а исполнитель
                       # отправляет att.ships-1 → ничья на нейтрале (0 кораблей).
MIN_FIRE       = 3     # ниже — пуск не имеет смысла

# ── Адаптивная ширина рассмотрения целей ─────────────────────────────
# Логика: чем больше доля наших кораблей в общем пуле — тем больше
# параллельных атак можем себе позволить, и тем менее жёстко режем по
# priority. Когда отстаём — фокусируемся только на jackpot-целях.
#
#   ratio          = our_ships / sum_all_ships (по raw_planets, до проекции)
#   top_n          — сколько финальных планов попадает в FIRE-loop
#   priority_floor — отсечка по zones priority; цели ниже floor не идут
#                    в attacks (экономит работу планировщика)
#
# ВНИМАНИЕ: значения подобраны от руки и подлежат тюнингу по матчам.
# Меняй ТОЛЬКО этот массив — больше ничего трогать не надо.
ADAPTIVE_TIERS = [
    # (ratio_lo, top_n, priority_floor, label)
    (0.00,   3,  +0.5, 'behind'),     # сильно отстаём — только jackpot
    (0.25,   6,  -0.5, 'parity'),     # около паритета — текущее поведение
    (0.45,  10,  -1.5, 'ahead'),      # уверенно впереди — расширяемся
    (0.65,  16,  -3.0, 'dominate'),   # доминируем — почти всё в работу
]
TOP_ATTACKS_FALLBACK = 6
PRIO_FLOOR_FALLBACK  = -1e9


def _resource_ratio(state, player):
    """Доля наших кораблей в общем пуле — нормализованный индикатор силы.
    Берём raw_planets (фактическое сейчас), не projection — projection
    «потратит» какие-то наши корабли на атаках, что даст заниженный ratio.
    Кометы (если есть) сюда учитываются — у них в общем мало кораблей,
    погрешность пренебрежимая."""
    raw = getattr(state, 'raw_planets', state.planets)
    our   = sum(max(0, int(p.ships)) for p in raw if p.owner == player)
    total = sum(max(0, int(p.ships)) for p in raw)
    if total <= 0:
        return 0.0
    return our / total


def _adapt_swarm_weights(base, ctx, ratio):
    """Адаптирует SwarmWeights под текущий контекст (возвращает копию).

    Два механизма:

    1. Динамический transfer_floor / transfer_min_ships (исправление #9):
       Фиксированный floor=20 блокирует все трансферы когда у планет <30 кораблей
       (free = ships − 20 ≤ 10 < transfer_min_ships=15 → всегда нет).
       Масштабируем пороги с ratio = our_ships / total:
         ratio=0.10 → floor≈5,  min_ships≈5
         ratio=0.25 → floor≈10, min_ships≈8
         ratio=0.50 → floor≈17, min_ships≈13
         ratio≥0.80 → floor=20, min_ships=15  (полные пороги)

    2. Desperate mode (исправление #8):
       Когда stance=desperate (ships_ratio < 0.35) агент терял темп из-за
       тех же высоких порогов и нулевого risk_tolerance. В desperate:
         - Гарнизон минимален (floor=5): нечего беречь если проигрываем
         - risk_tolerance=1: принимаем атаки с margin ≥ −1 (было margin ≥ 0)
         - transfer_thresh снижен: отправляем даже небольшие подкрепления
         - transfer_min_ships=5: маленькие партии тоже идут в ход
    """
    stance = getattr(ctx, 'stance', 'attrition') if ctx else 'attrition'

    # ── Динамические transfer-пороги (масштаб с ratio) ────────────────────
    # clamp ratio в [0, 0.4]: выше 40% пороги уже максимальные
    t = min(1.0, ratio / 0.4)
    dyn_floor     = max(5.0,  base.transfer_floor      * t)
    dyn_min_ships = max(5,    int(base.transfer_min_ships * t))
    dyn_thresh    = max(2.0,  base.transfer_thresh      * t)

    w = _dc.replace(base,
                    transfer_floor=dyn_floor,
                    transfer_min_ships=dyn_min_ships,
                    transfer_thresh=dyn_thresh)

    # ── Desperate override (перекрывает динамику) ─────────────────────────
    if stance == 'desperate':
        w = _dc.replace(w,
                        transfer_floor=5.0,         # минимальный гарнизон
                        transfer_min_ships=5,        # любое ненулевое подкрепление
                        transfer_thresh=1.5,         # нижний порог score
                        risk_tolerance=1,            # margin ≥ -1 считаем победным
                        max_transfers_per_turn=3,    # чуть быстрее консолидация
                        )

    return w


def _adaptive_attack_params(ratio):
    """По ratio выбирает (top_n, priority_floor, label) из ADAPTIVE_TIERS."""
    chosen = (TOP_ATTACKS_FALLBACK, PRIO_FLOOR_FALLBACK, 'fallback')
    for ratio_lo, top_n, floor, label in ADAPTIVE_TIERS:
        if ratio >= ratio_lo:
            chosen = (top_n, floor, label)
    return chosen


def _raw(state, pid):
    """Текущая (raw) планета по id."""
    raw_planets = getattr(state, 'raw_planets', state.planets)
    for p in raw_planets:
        if p.id == pid:
            return p
    return None


# ── Классификация наших планет: raw × projection ──────────────────────────

def _classify_our_planets(state, player):
    """Возвращает (stable, doomed, incoming_friendly).

    stable             — raw=ours И projected=ours. Безопасный атакёр /
                         базовая планета для будущих ходов.
    doomed             — raw=ours, но projected=enemy. Нас захватывают.
                         НЕЛЬЗЯ выбирать как target (мы своих не атакуем).
                         НУЖНО посылать туда подкрепление (defense plan).
    incoming_friendly  — raw=enemy/neutral, но projected=ours. Наш флот её
                         уже берёт — не дублируем атаку.
    """
    raw_planets  = getattr(state, 'raw_planets', state.planets)
    proj_owner   = {p.id: p.owner for p in state.planets}

    stable, doomed, incoming = [], [], []
    for p in raw_planets:
        future = proj_owner.get(p.id, p.owner)
        if p.owner == player and future == player:
            stable.append(p)
        elif p.owner == player and future != player:
            doomed.append(p)
        elif p.owner != player and future == player:
            incoming.append(p)
    return stable, doomed, incoming


def _build_defense_plans(state, doomed_planets, stable_planets, player):
    """Для каждой обречённой нашей планеты ищем ближайшего стабильного
    отправителя с непрегораживаемым солнцем путём, формируем план direct
    sup→tgt в формате совместимом с _execute_plan_atomically.

    Цель плана = projected (enemy now) планета по id обречённой.
    `_defender_at` корректно посчитает её через projected.ships +
    projected.production * delta — это и есть число которое надо превзойти
    чтобы перезахватить (или сохранить, если флот успеет до врага).
    """
    if not doomed_planets:
        return []

    proj_by_id = {p.id: p for p in state.planets}
    plans = []

    for doomed in doomed_planets:
        proj = proj_by_id.get(doomed.id)
        if proj is None:
            continue
        # Нужно ships > defender. defender = ships_proj + prod * (eta - proj_at).
        # Прикинем по eta=рабочему фронту, потом уточним.
        proj_at_doomed = getattr(state, 'projected_at', {}).get(doomed.id, 0)
        # Базовая оценка дефицита — проекция уже хранит «прибыль врага» в
        # proj.ships (после боя). +overkill чтобы не оказаться на ничьей.
        base_def = float(proj.ships) + 2.0

        candidates = []
        for src in stable_planets:
            if src.id == doomed.id:
                continue
            # Путь должен быть чист от солнца
            if segment_hits_sun(src.x, src.y, doomed.x, doomed.y):
                continue
            avail = max(0, src.ships - 1)  # минимум 1 в гарнизоне
            if avail < 3:                  # MIN_FIRE
                continue

            # Уточняем eta + дефицит (production за время полёта)
            eta_init, _ = _rendezvous_eta(src, doomed,
                                           max(1, avail), state.omega)
            # production только для owned (не нейтрала)
            eff_prod = float(proj.production) if proj.owner != NEUTRAL_OWNER else 0.0
            def_eta = base_def + eff_prod * max(0.0, eta_init - proj_at_doomed)
            need = int(math.ceil(def_eta)) + 1   # +1 на ничью
            if avail < need:
                continue
            send = min(avail, need + 2)  # небольшой запас, не больше доступного
            eta, _ = _rendezvous_eta(src, doomed, max(1, send), state.omega)
            candidates.append((eta, src, send, def_eta))

        if not candidates:
            continue
        # Приоритет — самый быстрый сэйв (минимизировать время «беззащитности»)
        candidates.sort(key=lambda x: x[0])
        eta, src, send, def_eta = candidates[0]

        plans.append({
            'mode':       'direct',
            'sup_id':     None,
            'att_id':     src.id,
            'tgt_id':     doomed.id,
            'eta_sa':     0.0,
            'eta_at':     eta,
            't_total':    eta,
            'x_sup':      0,
            'x_att':      int(send),
            'prod_att':   float(src.production),
            'x_tgt':      int(proj.ships),
            'prod_tgt':   float(proj.production),
            'strike':     float(send),
            'defender':   float(def_eta),
            'incoming':   0.0,
            'needed':     int(math.ceil(def_eta)) + 1,
            'available':  src.ships - 1,
            'slack':      int(src.ships - 1 - send),
            'margin':     float(send) - float(def_eta),
            'success':    True,
            'relay_reason': None,
            'is_defense': True,    # маркер для отладки
        })

    return plans


def _projected(state, pid):
    """Проецированная планета (для геометрии используем тот же x,y что и raw)."""
    for p in state.planets:
        if p.id == pid:
            return p
    return None


def _reserve_part(state, planet_id, n_ships, tentative):
    """
    Резервирует n_ships на planet_id с учётом уже зарезервированного.
    Возвращает фактически выделенное число кораблей либо None если part невозможен.
    Не модифицирует tentative (только проверяет).

    Важно: НЕ бампим x до MIN_FIRE. План (eval_direct/eval_pipe/eval_multi_sync)
    считает РОВНО нужное количество кораблей. Если план говорит x_sup=1 — значит
    1 корабль это калиброванный реинфорс (а не «мусор»). Бампинг до 3 раздувал
    каждый pipeline/multi_sync supply до MIN_FIRE и просто терял корабли.

    MIN_FIRE используется только как порог «маленькие планеты не запускают
    флоты» (avail < MIN_FIRE → None).
    """
    src = _raw(state, planet_id)
    if src is None:
        return None
    already = tentative.get(planet_id, 0)
    avail   = src.ships - already - RESERVE_ON_ATT
    if avail < MIN_FIRE:
        return None
    n_req = max(1, int(n_ships))
    return min(n_req, int(avail))


def _plan_parts(plan):
    """
    Раскладывает план в упорядоченный список частей (planet_id, aim_target_id, x).
    Для multi_sync: sup → tgt, att → tgt   (оба должны лететь одновременно).
    Для pipeline:  sup → att, att → tgt    (sup передаёт нашей планете att).
    Для direct:    att → tgt.
    """
    parts = []
    if plan.get('sup_id') is not None and plan.get('x_sup', 0) > 0:
        # sup-цель: для pipeline это наша промежуточная планета att,
        # для multi_sync — конечная цель tgt.
        sup_target = plan['att_id'] if plan['mode'] == 'pipeline' else plan['tgt_id']
        parts.append((plan['sup_id'], sup_target, int(plan['x_sup'])))
    if plan.get('x_att', 0) > 0:
        parts.append((plan['att_id'], plan['tgt_id'], int(plan['x_att'])))
    return parts


def _aim_and_verify(src, tgt, ships, state):
    """
    Считает угол через aim_hybrid и проверяет покадровой симуляцией
    (simulate_launch — 1-в-1 с движком), что флот реально попадает
    в нужную планету. Возвращает (angle, sim_eta) или None если
    ни одного годного угла нет.

    Проверяем основной угол aim_hybrid и до 6 деформаций ±epsilon, чтобы
    компенсировать редкие касательные промахи / дискретные эффекты. Это
    напрямую закрывает «иногда промахиваемся между удалёнными
    стационарными» — мы не запускаем угол, который simulate говорит, что
    промах/OOB/sun/чужая планета.
    """
    initial_by_id = state.initial_by_id() if callable(state.initial_by_id) else state.initial_by_id
    res = aim_hybrid(src, tgt, ships, state.planets, initial_by_id, state.omega)
    if res is None:
        return None
    base_angle = res[0]

    candidates = [base_angle]
    for d in (0.0025, -0.0025, 0.006, -0.006, 0.012, -0.012):
        candidates.append(base_angle + d)

    for a in candidates:
        sim = simulate_launch(src, a, ships, state.planets, initial_by_id, state.omega)
        if sim['outcome'] == 'HIT' and sim['planet_id'] == tgt.id:
            return a, int(sim['turn'])
    return None


def _defender_at(state, tgt, eta):
    """Оценка кораблей защитника в момент прибытия флота (turn = eta).
    Учитывает projected_at (если состояние спроецировано) и факт, что
    нейтральные планеты НЕ производят корабли."""
    proj_at = getattr(state, 'projected_at', {}).get(tgt.id, 0)
    if tgt.owner == NEUTRAL_OWNER:
        return float(tgt.ships)
    return float(tgt.ships) + float(tgt.production) * max(0.0, eta - proj_at)


def _execute_plan_atomically(state, plan, committed, risk_tolerance=0):
    """
    Атомарный запуск плана: либо все части плана успешно зарезервированы,
    углы посчитаны и подтверждены simulate_launch, либо план полностью
    отбрасывается (ничего не запускается).

    `risk_tolerance` — должен совпадать со значением переданным в swarm_plan.
    При risk_tolerance >= 1 снижаем eta_drift_buffer для enemy-планет с 1 до 0:
    план сгенерирован с margin ≥ −1 (overkill=1), поэтому execute проверяет
    ту же планку — иначе marginal-план всегда блокируется здесь же.

    Это гарантирует требование: любой запуск либо меняет стейт цели сам
    (direct), либо совместно с другим утверждённым запуском (multi_sync),
    либо является передачей на нашу же планету для будущего изменения
    (pipeline). Не запускаем «огрызки» планов, которые ничего не меняют.

    Дополнительно: для att-части (которая по плану должна зайти в цель
    и победить) проверяем что ships > defender_at(eta_actual). Если
    реальная eta стрельбы дрейфует и наш план уже не выигрывает — план
    отбрасываем целиком, а не сливаем мелочь, которую враг съест.

    Возвращает (moves, reason). При успехе reason='ok'. При отказе
    moves=[] и reason — короткое объяснение (для отладочного лога).
    """
    parts = _plan_parts(plan)
    if not parts:
        return [], 'empty_plan'

    # Стадия 1: пробное резервирование. Не трогаем committed пока не
    # убедились что ВСЕ части плана влезают.
    tentative = dict(committed)
    confirmed = []
    for planet_id, aim_target_id, n_req in parts:
        n_actual = _reserve_part(state, planet_id, n_req, tentative)
        if n_actual is None:
            src = _raw(state, planet_id)
            avail = (src.ships - tentative.get(planet_id, 0) - RESERVE_ON_ATT) if src else None
            return [], (f'reserve_failed: pid={planet_id} need={n_req} '
                        f'avail={avail} (after committed={tentative.get(planet_id, 0)})')
        tentative[planet_id] = tentative.get(planet_id, 0) + n_actual
        confirmed.append((planet_id, aim_target_id, n_actual))

    # Стадия 2: aim + simulate-verify для каждой части.
    moves = []
    tgt_id = plan.get('tgt_id')
    is_pipeline_sup_part = (plan.get('mode') == 'pipeline'
                            and plan.get('sup_id') is not None)
    for planet_id, aim_target_id, n_actual in confirmed:
        src = _raw(state, planet_id)
        tgt = _projected(state, aim_target_id)
        if src is None or tgt is None:
            return [], f'planet_not_found: src={planet_id} tgt={aim_target_id}'
        verified = _aim_and_verify(src, tgt, n_actual, state)
        if verified is None:
            return [], (f'aim_verify_failed: src={planet_id} aim_tgt={aim_target_id} '
                        f'ships={n_actual} (no angle simulates as HIT)')
        angle, sim_eta = verified

        is_att_to_tgt = (aim_target_id == tgt_id and planet_id == plan['att_id']
                         and not is_pipeline_sup_part
                         and plan.get('mode') == 'direct')
        if is_att_to_tgt:
            defender_actual = _defender_at(state, tgt, sim_eta)
            # Нейтралы: производства нет, defender детерминирован — строгий >
            # достаточен, лишний буфер только выбрасывал бы корабли впустую.
            # Enemy-owned: sim_eta чуть длиннее расчётного eta_at плана, за это
            # время defender успевает подрасти на +prod. Буфер +1 закрывает
            # off-by-one когда gap = 0 (77 случаев в логе).
            # risk_tolerance>=1: plan was generated with overkill=1 for enemy planets,
            # so execute uses the same buffer (0) to stay consistent.
            eta_drift_buffer = 0 if (tgt.owner == NEUTRAL_OWNER or risk_tolerance >= 1) else 1
            if n_actual <= defender_actual + eta_drift_buffer:
                return [], (f'wont_win: ships={n_actual} <= '
                            f'defender_at_eta={defender_actual:.1f}+buf={eta_drift_buffer} '
                            f'(sim_eta={sim_eta}, tgt={tgt_id})')

        moves.append([planet_id, angle, n_actual])

    committed.update(tentative)
    return moves, 'ok'


def _build_virtual_fleets(expected_actions, state_raw, opp_id: int) -> list:
    """Создать список виртуальных Fleet-объектов из предсказанных действий.

    Угол вычисляется как atan2(tgt - src) — простое приближение, достаточное
    для project_state (точный aim_hybrid здесь избыточен и дорог).
    Виртуальные флоты получают отрицательные ID чтобы не конфликтовать
    с реальными.
    """
    import math as _m
    from shooting import Fleet as _Fleet
    planets_by_id = {p.id: p for p in state_raw.planets}
    virtual = []
    vid = _VIRTUAL_FLEET_ID_BASE
    for a in expected_actions:
        src = planets_by_id.get(a['from_id'])
        tgt = planets_by_id.get(a['target_id'])
        if src is None or tgt is None:
            continue
        angle = _m.atan2(tgt.y - src.y, tgt.x - src.x)
        virtual.append(_Fleet(
            id=vid,
            owner=opp_id,
            x=float(src.x),
            y=float(src.y),
            angle=angle,
            from_planet_id=src.id,
            ships=max(1, int(a['ships'])),
        ))
        vid -= 1
    return virtual


def _extract_new_opp_fleets(state_raw, prev_fleet_ids: set,
                             opp_id: int,
                             planets_by_id: dict) -> list:
    """Найти флоты противника запущенные в прошлый ход.

    Новый флот = fleet.owner == opp_id AND fleet.id не был в prev_fleet_ids.
    Для каждого нового флота вычисляем (from_id, target_id) через
    simulate_fleet_target, затем определяем action_type.
    """
    from projection import simulate_fleet_target as _sft, NEUTRAL_OWNER as _NO
    new_fleets = [
        f for f in state_raw.fleets
        if f.owner == opp_id and f.id not in prev_fleet_ids
    ]
    actions = []
    for f in new_fleets:
        tgt_id, _ = _sft(f, state_raw.planets, state_raw.omega)
        if tgt_id is None:
            continue
        tgt = planets_by_id.get(tgt_id)
        if tgt is None:
            continue
        if tgt.owner == opp_id:
            atype = 'reinforce'
        elif tgt.owner == _NO:
            atype = 'capture_neutral'
        else:
            atype = 'attack_enemy'
        actions.append({
            'from_id':     getattr(f, 'from_planet_id', -1),
            'target_id':   tgt_id,
            'ships':       int(f.ships),
            'action_type': atype,
        })
    return actions


def _agent_impl(obs, deadline=None):
    player    = obs.get('player', 0)
    state_raw = GameState.from_kaggle_obs(obs)

    # _step нужен и байесу (step=) и MCTS-логу, поэтому извлекаем сразу.
    _step = (obs.get('step') if isinstance(obs.get('step'), int) else
             obs.get('stepNumber') if isinstance(obs.get('stepNumber'), int) else
             getattr(state_raw, 'step', -1))

    # ── Байесовский предсказатель (обновление + добавление виртуальных флотов) ──
    global _bayes_model, _prev_fleet_ids, _prev_state_raw
    state_for_projection = state_raw   # может быть заменено расширенным

    if USE_OPPONENT_PREDICTION and _BAYES_AVAILABLE:
        try:
            opp_id = (player + 1) % max(2, getattr(state_raw, 'n_players', 2))
            planets_by_id = {p.id: p for p in state_raw.planets}

            # Включаем аналитику если включён debug-лог (zero-cost в бою)
            if _opp_bayes_mod is not None:
                _opp_bayes_mod.ANALYTICS_MODE = _dbg.enabled()

            # Ленивая инициализация
            if _bayes_model is None:
                _bayes_model = OpponentModelBayesian()

            # Шаг 1: обновить модель по реально наблюдённым флотам противника
            if _prev_fleet_ids is not None and _prev_state_raw is not None:
                observed = _extract_new_opp_fleets(
                    state_raw, _prev_fleet_ids, opp_id, planets_by_id
                )
                metrics = _bayes_model.update(
                    observed, _prev_state_raw, opp_id, step=_step
                )
                # Логируем аналитику если включена (metrics != None только при ANALYTICS_MODE)
                if metrics is not None and _dbg.enabled():
                    try:
                        _dbg._w(
                            f'[BAYES/update]  step={metrics["step"]}'
                            f'  surprise={metrics["surprise"]:.3f}'
                            f'  entropy={metrics["entropy"]:.3f}'
                            f'  top={metrics["top_preset"]}({metrics["top_prob"]:.2f})'
                            f'  obs={metrics["n_opponent_actions"]}'
                            f'  matched={metrics["match_count"]}'
                        )
                        # Метрика непонимания: confusion% = entropy/ln(5)*100
                        # 100% = полная неопределённость, 0% = уверен в пресете
                        _dbg.log_bayes_confusion(metrics['entropy'])
                    except Exception:
                        pass

            # Шаг 2: получить предсказанные действия и создать виртуальные флоты
            expected = _bayes_model.get_expected_actions(
                state_raw, opp_id, threshold=BAYES_THRESHOLD
            )
            if expected:
                virtual = _build_virtual_fleets(expected, state_raw, opp_id)
                if virtual:
                    augmented_fleets = list(state_raw.fleets) + virtual
                    from orbit_sim import GameState as _GS
                    state_for_projection = _GS(
                        planets=list(state_raw.planets),
                        fleets=augmented_fleets,
                        omega=state_raw.omega,
                        step=state_raw.step,
                        initial_planets=list(state_raw._initial_planets.values()),
                        n_players=getattr(state_raw, 'n_players', 2),
                        comet_ids=set(getattr(state_raw, 'comet_ids', set()) or set()),
                    )
            if _dbg.enabled():
                try:
                    _dbg._w(
                        f'[BAYES/predict]  top={_bayes_model.top_preset()}'
                        f'  expected={len(expected)}'
                        f'  virtual={len(virtual) if expected else 0}'
                        f'  dist={_bayes_model.summary()}'
                    )
                except Exception:
                    pass

        except Exception as _bayes_err:
            try: _dbg.log_error('bayes', _bayes_err)
            except Exception: pass

    # 0. Проекция: резолвим все летящие флоты (свои и чужие + виртуальные).
    #    После этого state.planets — состояние на момент последнего события.
    #    player передаётся, чтобы projection знал чьи кометы «отскакивают»
    #    обратно на ближайшую нашу планету (см. project_state docstring).
    _raw_n_fleets = len(getattr(state_for_projection, 'fleets', []))
    state = project_state(state_for_projection, horizon=PROJECTION_HORIZON, player=player)

    _dbg.begin_turn(_step, player, len(state.planets), _raw_n_fleets)
    _dbg.log_fleets(state.fleets, player)
    if _step == 0:
        global _prev_zones
        _prev_zones = {}

    # ── 0.5. MCTS (ранняя игра) ───────────────────────────────────────────────
    # Условие включения: USE_MCTS, модуль доступен, ранняя фаза, карта небольшая.
    # При успехе — возвращаем ходы прямо из MCTS, минуя весь основной пайплайн.
    # При любой ошибке — молча падаем сквозь в обычную логику (fail-safe).
    global _opponent_model, _pending_partials
    if (USE_MCTS and _MCTS_AVAILABLE and MCTS_MODE != "none"
            and _step >= 0 and _step < EARLY_GAME_SWITCH
            and len(state.planets) <= EARLY_MAX_PLANETS):
        try:
            import time as _time_mcts
            # Ленивая инициализация модели противника (сбрасывается при старте матча)
            if _opponent_model is None:
                _opponent_model = OpponentModel()

            # Генерируем ПАР согласно MCTS_MODE
            opp_id = (player + 1) % max(2, getattr(state, 'n_players', 2))

            if MCTS_MODE == "v2_pair":
                try:
                    from action_space_v2 import generate_actions_v2
                    my_actions = generate_actions_v2(state, player)
                except ImportError:
                    my_actions = generate_actions(state, player)

            elif MCTS_MODE == "v3_partial":
                try:
                    from action_space_v3 import generate_actions_v3, PartialAction
                    my_actions = generate_actions_v3(
                        state, player,
                        pending_partials=_pending_partials or [],
                    )
                except ImportError:
                    my_actions = generate_actions(state, player)

            elif MCTS_MODE == "v4_hybrid_net":
                try:
                    from action_space_v4 import generate_actions_v4
                    my_actions = generate_actions_v4(state, player)
                except ImportError:
                    my_actions = generate_actions(state, player)

            else:  # v1_plain (default)
                my_actions = generate_actions(state, player)

            opp_actions = generate_opponent_actions(state, opp_id)

            # Бюджет времени: доля от остатка до дедлайна
            if deadline is not None:
                remaining   = deadline - _time_mcts.perf_counter()
                mcts_budget = max(0.05, remaining * MCTS_TIME_FRACTION)
            else:
                mcts_budget = 0.35

            if _dbg.enabled():
                try:
                    import agent_debug as _d
                    _d._w(f'[MCTS/{MCTS_MODE}]  step={_step}'
                          f'  my_actions={len(my_actions)}'
                          f'  opp_actions={len(opp_actions)}'
                          f'  budget={mcts_budget:.3f}s')
                except Exception:
                    pass

            best_action = run_mcts(
                state, my_actions, opp_actions,
                _opponent_model, player,
                time_budget=mcts_budget,
            )

            if best_action is not None:
                if _dbg.enabled():
                    try:
                        import agent_debug as _d
                        _d._w(f'[MCTS/{MCTS_MODE}]  best={best_action}')
                    except Exception:
                        pass
                # v3: track PartialAction for second wave next turn
                if MCTS_MODE == "v3_partial":
                    try:
                        from action_space_v3 import PartialAction as _PA
                        if isinstance(best_action, _PA):
                            if _pending_partials is None:
                                _pending_partials = []
                            _pending_partials.append(best_action)
                    except Exception:
                        pass
                _dbg.end_turn()
                return best_action.to_moves()
            # Если MCTS не нашёл хода — продолжаем в обычный пайплайн
            if _dbg.enabled():
                try:
                    import agent_debug as _d
                    _d._w(f'[MCTS/{MCTS_MODE}]  no action found, falling back')
                except Exception:
                    pass
        except Exception as _mcts_err:
            try: _dbg.log_error('mcts', _mcts_err)
            except Exception: pass
            # Любая ошибка → продолжаем в обычный пайплайн

    # Game Understanding Layer — глобальное «понимание» текущего хода.
    # Пока просто логируем для проверки; интегрировать в scoring — следующий шаг.
    try:
        ctx = compute_context(state, player)
        if _dbg.enabled():
            try:
                import agent_debug as _d
                _d._w(f'[GUL]  {context_summary(ctx)}')
            except Exception:
                pass
    except Exception as _e:
        ctx = None
        try: _dbg.log_error('compute_context', _e)
        except Exception: pass

    # 0.5. Адаптивная ширина: по ratio = our_ships/total выбираем top_n
    #      финальных планов и priority_floor (отсечку по приоритету целей).
    ratio = _resource_ratio(state, player)
    top_n, prio_floor, tier_label = _adaptive_attack_params(ratio)
    if _dbg.enabled():
        try:
            import agent_debug as _d
            _d._w(f'[ADAPTIVE]  ratio={ratio:.3f}  tier={tier_label}  '
                  f'top_n={top_n}  prio_floor={prio_floor:+.2f}')
        except Exception:
            pass

    # 1. Зонирование (на спроецированном состоянии)
    # our_total_prod — для production-scarcity boost (ранняя игра).
    # Считаем по raw_planets (до проекции), чтобы отражать реальный прод сейчас.
    _raw_pl_early = getattr(state, 'raw_planets', state.planets)
    our_total_prod = sum(
        float(p.production) for p in _raw_pl_early if p.owner == player
    )
    # Вычисляем stage-aware порог priority-reclassify: линейная интерполяция
    # от prio_reclassify_thr (step=0) до prio_reclassify_thr_late (step=TOTAL).
    # Если оба одинаковые (дефолт) — статичный порог, интерполяции нет.
    # Читаем из SWARM_WEIGHTS напрямую: _adapt_swarm_weights не трогает эти поля.
    _prio_thr_early = float(getattr(SWARM_WEIGHTS, 'prio_reclassify_thr',
                                    _PRIO_RECLASSIFY_THR_DEFAULT))
    _prio_thr_late  = float(getattr(SWARM_WEIGHTS, 'prio_reclassify_thr_late',
                                    _prio_thr_early))
    _phase_prio     = min(1.0, max(0.0, float(_step) / float(_TOTAL_STEPS))) if _step >= 0 else 0.0
    _eff_prio_thr   = _prio_thr_early + (_prio_thr_late - _prio_thr_early) * _phase_prio

    try:
        df, _ = compute_zones_from_state(state, player=player,
                                         our_total_prod=our_total_prod,
                                         prio_reclassify_thr=_eff_prio_thr)
    except Exception as e:
        _dbg.log_error('compute_zones_from_state', e)
        _dbg.end_turn()
        return []

    _dbg.log_zones(df, projected_at=getattr(state, 'projected_at', {}))
    zone_lookup = {int(r['pid']): r['zone'] for _, r in df.iterrows()}
    _dbg.log_zone_flips(_prev_zones, zone_lookup)
    _prev_zones = dict(zone_lookup)   # обновляем кеш для следующего хода

    if _dbg.enabled():
        from zones import SCARCITY_K, EARLY_PHASE_THR
        from shooting import TOTAL_STEPS as _TS
        _phase = min(1.0, float(getattr(state, 'step', 0) or 0) / float(_TS))
        _fade  = max(0.0, 1.0 - _phase / EARLY_PHASE_THR)
        _boost = (SCARCITY_K / (1.0 + our_total_prod)) * _fade
        if _boost > 0.01:
            _dbg._w(f'[SCARCITY]  our_prod={our_total_prod:.0f}  phase={_phase:.2f}'
                    f'  fade={_fade:.2f}  prod_boost=×{1+_boost:.2f}'
                    f'  eff_prod_w={0.5*(1+_boost):.3f}')

    # 1.5. Классификация наших планет по комбинации raw × projection:
    #      stable / doomed / incoming_friendly.
    #      doomed   — наши сейчас, но проекция говорит что потеряем →
    #                 фильтруем из targets, отправляем туда defense.
    #      incoming — не наши сейчас, но станут наши по проекции →
    #                 фильтруем из targets (наш флот уже летит).
    raw_planets   = getattr(state, 'raw_planets', state.planets)
    raw_owner     = {p.id: p.owner for p in raw_planets}
    stable_ours, doomed_ours, incoming_ours = _classify_our_planets(state, player)
    doomed_ids    = {p.id for p in doomed_ours}
    incoming_ids  = {p.id for p in incoming_ours}

    if _dbg.enabled():
        try:
            import agent_debug as _d
            _d._w(f'[CLASSIFY]  stable={len(stable_ours)}  '
                  f'doomed={[p.id for p in doomed_ours]}  '
                  f'incoming={[p.id for p in incoming_ours]}')
        except Exception:
            pass

    # 2. Выбор целей по зоне и priority + адаптивная отсечка по floor.
    #    floor применяется ДО best_attacks → меньше работы планировщику.
    #    КРОМЕ ТОГО: фильтруем планеты которые УЖЕ наши в raw (doomed)
    #    или СТАНУТ нашими через проекцию (incoming) — это не цели атаки.
    def _is_real_target(pid):
        # raw=ours → точно не атакуем (даже если projection=enemy = doomed)
        if raw_owner.get(pid, -1) == player:
            return False
        # incoming friendly → атаковать дублирующе нет смысла
        if pid in incoming_ids:
            return False
        return True

    tgt_df = (df[df['zone'].isin(TARGET_ZONES) & (df['priority'] >= prio_floor)]
                .sort_values('priority', ascending=False))
    target_ids = [pid for pid in tgt_df['pid'].tolist() if _is_real_target(pid)]
    targets    = [p for p in state.planets if p.id in target_ids]
    target_source = f'zones (floor={prio_floor:+.2f})'

    if not targets:
        # Fallback: если по floor вообще никого — пробуем без floor (только зоны).
        # Это страхует от ситуации «все priority < floor» в раннем/позднем игре.
        tgt_df = (df[df['zone'].isin(TARGET_ZONES)]
                    .sort_values('priority', ascending=False))
        target_ids = [pid for pid in tgt_df['pid'].tolist() if _is_real_target(pid)]
        targets    = [p for p in state.planets if p.id in target_ids]
        target_source = 'zones (floor relaxed)'

    if not targets:
        targets = [p for p in state.planets
                   if p.owner != player and _is_real_target(p.id)]
        target_ids = [p.id for p in targets]
        target_source = 'fallback_all_non_ours'

    _dbg.log_targets(target_ids, zones_df=df)
    if _dbg.enabled():
        try:
            import agent_debug as _d
            _d._w(f'   (target_source={target_source})')
        except Exception:
            pass

    # 3a. (опционально) — все кандидатные планы по каждой цели для ручной отладки
    if _dbg.plans_all_enabled():
        try:
            raw_planets = getattr(state, 'raw_planets', state.planets)
            ours = [p for p in raw_planets if p.owner == player]
            for tgt in targets:
                cand = all_plans(state, tgt, ours, horizon=ATTACK_HORIZON, player=player)
                if cand:
                    _dbg.log_plans(cand, label=f'PLANS_ALL tgt={tgt.id}')
        except Exception as e:
            _dbg.log_error('all_plans', e)

    # 2.5. Defense plans: для doomed-наших планет ищем стабильного отправителя.
    #      Эти планы выполняются ПЕРЕД атакой (приоритет самосохранению).
    defense_plans = _build_defense_plans(state, doomed_ours, stable_ours, player)
    if _dbg.enabled() and defense_plans:
        _dbg.log_plans(defense_plans, label='DEFENSE_PLANS')

    # 3. Атаки.
    #    USE_SWARM=True → swarm_plan: per-planet auction + redistribute.
    #    Иначе fallback на best_attacks (глобальный priority).
    if USE_SWARM:
        # Адаптируем веса под текущий контекст (stance + ratio).
        # _adapt_swarm_weights возвращает копию SWARM_WEIGHTS с динамичными
        # transfer-порогами и desperate-режимом (см. подробный комментарий).
        eff_weights = _adapt_swarm_weights(SWARM_WEIGHTS, ctx, ratio)
        if _dbg.enabled():
            try:
                _stance = getattr(ctx, 'stance', '?') if ctx else '?'
                _fl     = eff_weights.transfer_floor
                _ms     = eff_weights.transfer_min_ships
                _rt     = eff_weights.risk_tolerance
                _mx_tr  = eff_weights.max_transfers_per_turn
                _dbg._w(f'[WEIGHTS]  stance={_stance}  ratio={ratio:.3f}'
                        f'  floor={_fl:.1f}  min_ships={_ms}'
                        f'  risk_tol={_rt}  max_tr={_mx_tr}')
            except Exception:
                pass

        # priority_lookup для tiebreaker'а в аукционе (из zones)
        prio_lookup = {int(r.pid): float(r.priority)
                       for _, r in df.iterrows()
                       if 'priority' in df.columns}
        attack_plans, swarm_dbg = swarm_plan(
            state, player, targets,
            weights=eff_weights, priority_lookup=prio_lookup,
            zone_lookup=zone_lookup,
            max_targets=12,         # cap: на больших картах не успеваем за бюджет kaggle
            deadline=deadline,      # передаём из обёртки agent()
        )
        if _dbg.enabled():
            try:
                _dbg._w(f'[SWARM]  cand={swarm_dbg["n_candidates"]} '
                        f'committed={swarm_dbg["n_committed"]} '
                        f'transfers={swarm_dbg["n_transfers"]} '
                        f'unfunded={swarm_dbg["n_unfunded"]}')
                top_stress = sorted(swarm_dbg['stress'].items(),
                                    key=lambda x: -x[1])[:5]
                _dbg._w(f'[SWARM]  top stress: '
                        + ' '.join(f'p{pid}={s:.1f}' for pid, s in top_stress))
                # Idle-деньги: сколько кораблей осталось на каждой планете
                _dbg.log_remaining(swarm_dbg.get('remaining', {}), zone_lookup)
                # Детали transfers: откуда, куда, градиент
                _dbg.log_transfers(attack_plans, zone_lookup)
            except Exception:
                pass
    else:
        attack_plans = best_attacks(state, player, targets, top_n=top_n)
    _dbg.log_plans(attack_plans, label='PLANS_TOP')

    # 3.5. Композиция: defense_plans → attack_plans. Сначала спасаемся,
    #      потом атакуем. Резервирование кораблей (committed dict)
    #      гарантирует что defense получает приоритет на источниках.
    plans = defense_plans + attack_plans

    # 4. Сборка ходов: каждый план атомарно (success-only).
    moves     = []
    committed = {}

    for idx, plan in enumerate(plans):
        if not plan.get('success'):
            _dbg.log_decision(idx, plan, 'SKIP', 'plan.success=false (won\'t change tgt state)')
            continue
        sub, reason = _execute_plan_atomically(state, plan, committed,
                                                risk_tolerance=eff_weights.risk_tolerance
                                                if USE_SWARM else 0)
        if sub:
            _dbg.log_decision(idx, plan, 'FIRE',
                              f'parts={len(sub)} '
                              + ' '.join(f'[src={m[0]} ships={m[2]}]' for m in sub))
            moves.extend(sub)
        else:
            _dbg.log_decision(idx, plan, 'SKIP', reason)

    _dbg.log_moves(moves)
    _dbg.end_turn()

    # ── Сохраняем state для байесовского обновления на следующем ходу ─────
    if USE_OPPONENT_PREDICTION and _BAYES_AVAILABLE:
        try:
            _prev_fleet_ids = {f.id for f in state_raw.fleets}
            _prev_state_raw = state_raw
        except Exception:
            pass

    # ── Дамп аналитики байеса в конце матча ──────────────────────────────
    # Только в debug-режиме. Kaggle-бои: _dbg.enabled() == False → пропуск.
    if (_dbg.enabled() and _BAYES_AVAILABLE
            and _bayes_model is not None
            and _bayes_model.analytics_history):
        try:
            is_last = (_step >= 498)   # kaggle матч обычно 500 ходов
            if is_last:
                import json as _json
                history = _bayes_model.dump_analytics()
                _dbg._w(f'[BAYES/dump]  turns={len(history)}')
                for rec in history:
                    _dbg._w('[BAYES/H] ' + _json.dumps(rec, separators=(',', ':')))
        except Exception:
            pass

    return moves


# ── kaggle-safe entrypoint ──────────────────────────────────────────────
# Никогда не падаем. Любая ошибка → пустой список ходов (агент просто стоит,
# но матч продолжается, не дисквал). Также soft deadline по wall-clock,
# чтобы swarm_plan корректно прервался при cap превышен.
def agent(obs, config=None):
    import time as _time
    # kaggle обычно даёт 1 секунду на step; берём 0.85 как soft бюджет
    act_timeout = 1.0
    if config is not None:
        try:
            act_timeout = float(config.get('actTimeout', 1.0))
        except Exception:
            pass
    deadline = _time.perf_counter() + max(0.4, min(act_timeout * 0.85, 5.0))
    try:
        return _agent_impl(obs, deadline=deadline)
    except Exception as _e:
        try:
            _dbg.log_error('agent', _e)
        except Exception:
            pass
        return []
