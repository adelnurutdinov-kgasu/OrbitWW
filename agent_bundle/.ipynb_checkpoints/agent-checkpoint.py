"""
Агент Orbit Wars.

Каждый ход:
  0. Проекция     — project_state резолвит уже летящие флоты (свои+чужие):
                    каждая планета видится в момент последнего прибытия.
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
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from orbit_sim import GameState
from projection import project_state, PROJECTION_HORIZON, NEUTRAL_OWNER
from zones import compute_zones_from_state
from attacks import best_attacks, SAFETY_OVERKILL, all_plans, ATTACK_HORIZON
from shooting import aim_hybrid, simulate_launch
import agent_debug as _dbg

TARGET_ZONES   = ('easy_target', 'priority_target')
TOP_ATTACKS    = 6     # сколько целей рассматривать за ход
RESERVE_ON_ATT = 1     # минимум кораблей оставить на атакере
MIN_FIRE       = 3     # ниже — пуск не имеет смысла


def _raw(state, pid):
    """Текущая (raw) планета по id."""
    raw_planets = getattr(state, 'raw_planets', state.planets)
    for p in raw_planets:
        if p.id == pid:
            return p
    return None


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


def _execute_plan_atomically(state, plan, committed):
    """
    Атомарный запуск плана: либо все части плана успешно зарезервированы,
    углы посчитаны и подтверждены simulate_launch, либо план полностью
    отбрасывается (ничего не запускается).

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
            if n_actual <= defender_actual:
                return [], (f'wont_win: ships={n_actual} <= defender_at_eta={defender_actual:.1f} '
                            f'(sim_eta={sim_eta}, tgt={tgt_id})')

        moves.append([planet_id, angle, n_actual])

    committed.update(tentative)
    return moves, 'ok'


def agent(obs):
    player    = obs.get('player', 0)
    state_raw = GameState.from_kaggle_obs(obs)

    # 0. Проекция: резолвим все летящие флоты (свои и чужие).
    #    После этого state.planets — состояние на момент последнего события.
    #    player передаётся, чтобы projection знал чьи кометы «отскакивают»
    #    обратно на ближайшую нашу планету (см. project_state docstring).
    state = project_state(state_raw, horizon=PROJECTION_HORIZON, player=player)

    _step = (obs.get('step') if isinstance(obs.get('step'), int) else
             obs.get('stepNumber') if isinstance(obs.get('stepNumber'), int) else
             getattr(state_raw, 'step', -1))
    _dbg.begin_turn(_step, player, len(state.planets), len(state.fleets))
    _dbg.log_fleets(state.fleets, player)

    # 1. Зонирование (на спроецированном состоянии)
    try:
        df, _ = compute_zones_from_state(state, player=player)
    except Exception as e:
        _dbg.log_error('compute_zones_from_state', e)
        _dbg.end_turn()
        return []

    _dbg.log_zones(df, projected_at=getattr(state, 'projected_at', {}))

    # 2. Выбор целей по зоне и priority
    tgt_df = (df[df['zone'].isin(TARGET_ZONES)]
                .sort_values('priority', ascending=False))
    target_ids = tgt_df['pid'].tolist()
    targets    = [p for p in state.planets if p.id in target_ids]
    target_source = 'zones'

    if not targets:
        targets = [p for p in state.planets if p.owner != player]
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

    # 3. Атаки (best_attacks внутри учитывает уже летящие флоты)
    plans = best_attacks(state, player, targets, top_n=TOP_ATTACKS)
    _dbg.log_plans(plans, label='PLANS_TOP')

    # 4. Сборка ходов: каждый план атомарно (success-only).
    moves     = []
    committed = {}

    for idx, plan in enumerate(plans):
        if not plan.get('success'):
            _dbg.log_decision(idx, plan, 'SKIP', 'plan.success=false (won\'t change tgt state)')
            continue
        sub, reason = _execute_plan_atomically(state, plan, committed)
        if sub:
            _dbg.log_decision(idx, plan, 'FIRE',
                              f'parts={len(sub)} '
                              + ' '.join(f'[src={m[0]} ships={m[2]}]' for m in sub))
            moves.extend(sub)
        else:
            _dbg.log_decision(idx, plan, 'SKIP', reason)

    _dbg.log_moves(moves)
    _dbg.end_turn()
    return moves
