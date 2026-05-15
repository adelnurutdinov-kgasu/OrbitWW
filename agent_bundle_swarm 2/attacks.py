"""
Блок оценки атак: direct / multi_sync / pipeline.

Главная идея: считаем МИНИМУМ кораблей нужный для победы (с учётом
производства защитника и уже летящих к цели дружественных флотов).
НЕ отправляем всё подряд.

Ключевые экспорты:
  eval_direct(state, att, tgt, incoming=0)            -> dict | None
  eval_multi_sync(state, sup, att, tgt, incoming=0)   -> dict | None
  eval_pipe(state, sup, att, tgt, incoming=0)         -> dict | None
  friendly_incoming(state, tgt, t_max, player)        -> (ships, etas)
  all_plans(state, tgt, ours, horizon, player)        -> list[dict]
  best_attacks(state, player, targets)                -> list[dict]
"""

import math
from itertools import product as _product

from orbit_sim import segment_hits_sun
from force import _rendezvous_eta
from shooting import fleet_speed_correct

ATTACK_HORIZON         = 80
SUN_SAFETY_PIPE        = 1.5
PIPELINE_MAX_ANGLE_DEG = 120.0
NEUTRAL_OWNER          = -1     # движок не производит у нейтралов


def _effective_prod(tgt):
    """Производство, КОТОРОЕ РЕАЛЬНО ПРИБАВЛЯЕТСЯ во время полёта.
    Нейтральные планеты в движке (orbit_sim.simulate_step, шаг 2) НЕ
    производят корабли — production стартует только когда планета захвачена.
    Без этой проверки `defender` для нейтралов завышается на prod·eta и
    `best_attacks` ошибочно помечает план как проигрышный, хотя реальная
    оборона = tgt.ships."""
    if tgt.owner == NEUTRAL_OWNER:
        return 0.0
    return float(tgt.production)


def _overkill(tgt, risk=0):
    """Размер буфера сверх defender. Для нейтралов производства нет ⇒ дрейф
    eta не страшен, хватает +1 (от ничьей). Для owned — +2.

    risk (0..2) — снижает буфер. ВАЖНО: минимум 1, потому что движок
    использует strict-< при бое (атака побеждает только если ships > defender,
    ничья = поражение). С overkill=0 атомный исполнитель отбрасывает план
    как «wont_win: ships == defender_at_eta».

    risk=1 убирает «дрейф eta» запас (только у owned: 2 → 1, у нейтрала
    остаётся 1). risk=2 = синоним risk=1, ниже опуститься нельзя.
    """
    base = SAFETY_OVERKILL_NEUTRAL if tgt.owner == NEUTRAL_OWNER else SAFETY_OVERKILL
    return max(1, base - int(risk))

SAFETY_OVERKILL        = 2     # сколько кораблей сверх defender (для OWNED цели).
                               # =2 покрывает: 1 на ничью (движок strict-<)
                               # + 1 на дрейф eta между планированием
                               # (_rendezvous_eta) и реальной стрельбой
                               # (aim_hybrid).
SAFETY_OVERKILL_NEUTRAL = 1    # для НЕЙТРАЛОВ хватает +1: нет производства,
                               # значит нет дрейфа defender'а — нужен только
                               # буфер на ничью.
MIN_USEFUL_STRIKE      = 3     # ниже — атака не имеет смысла, пропускаем
ETA_REFINE_ITERS       = 2     # сколько раз пересчитать eta при изменении кораблей
INCOMING_ANGLE_TOL     = 0.18  # рад (~10°): флот целится в цель если так
RESERVE_ON_ATT         = 0     # минимум кораблей оставить на атакере.
                               # ОБЯЗАН совпадать с agent.RESERVE_ON_ATT.
                               # Тай на нейтрале предотвращается через
                               # SAFETY_OVERKILL_NEUTRAL=1 (нужно строго больше
                               # defender), а не через резервирование корабля.

# ── Порог "одновременности" для multi_sync ────────────────────────────────
# `agent._plan_parts` эмитит ОБА флота СРАЗУ в один ход → быстрый прилетит
# раньше медленного на |Δeta| ходов. Раз мы вообще полезли в multi_sync,
# значит att одного НЕ хватает (см. ранний return) — следовательно быстрый
# флот, прилетев один, ПРОИГРАЕТ defender'у. Defender за это время
# дорегенерирует prod·Δeta кораблей, и медленному тоже не хватит.
#
# При маленьком Δeta (<= MAX_MULTI_SYNC_DELTA_ETA) defender не успевает
# восстановиться до критичного значения, и пара "почти одновременно"
# работает как один комбинированный удар — НЕ копить два независимых
# действия в очередь, а сразу собрать их в одно.
#
# При большом Δeta выгоднее НЕ делать multi_sync вовсе: пусть планировщик
# выберет direct (от ближайшего) сейчас, а второй источник либо переразметит
# приоритет в следующем ходу, либо станет supply'ем для другой цели.
MAX_MULTI_SYNC_DELTA_ETA = 5   # ходов; одновременность с допуском.
                               # Калибровка: даже equidistant att/sup дают
                               # Δeta до 5-9 ходов из-за орбитального lead-angle
                               # (rendezvous-eta зависит от стороны подлёта к
                               # вращающейся цели). 5 ходов — компромисс между
                               # "ловить только реально несинхронные пары" и
                               # "не резать симметричные геометрии".


# ── Утилиты ────────────────────────────────────────────────────────────

def _eta(src, dst, ships, omega):
    e, _ = _rendezvous_eta(src, dst, max(1, ships), omega)
    return float(e)


def _seg_blocked(a, b, safety=SUN_SAFETY_PIPE):
    return segment_hits_sun(a.x, a.y, b.x, b.y, safety=safety)


def _angle_at_att(sup, att, tgt):
    """Угол ∠(sup-att-tgt) в градусах."""
    v1x, v1y = att.x - sup.x, att.y - sup.y
    v2x, v2y = tgt.x - att.x, tgt.y - att.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = (v1x * v2x + v1y * v2y) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def _plan_total_ships(plan):
    return int(plan.get('x_att', 0)) + int(plan.get('x_sup', 0))


def _mean_d_to_non_ours(p, all_planets, player):
    """Среднее расстояние от планеты `p` до всех ЧУЖИХ/НЕЙТРАЛЬНЫХ планет.
    Используется как простая мера «насколько планета на фронте»: чем меньше
    среднее расстояние — тем ближе к врагу/нейтралам.

    Если других планет нет — возвращаем +inf (планета формально «дальше всех»)."""
    others = [q for q in all_planets if q.id != p.id and q.owner != player]
    if not others:
        return float('inf')
    return sum(math.hypot(p.x - q.x, p.y - q.y) for q in others) / len(others)


def _is_more_frontline(att, sup, all_planets, player):
    """True если att ближе к не-нашим планетам чем sup → имеет смысл
    перебросить ВСЁ что есть с sup на att (att — лучшая стартовая площадка
    для следующих атак)."""
    return _mean_d_to_non_ours(att, all_planets, player) \
         < _mean_d_to_non_ours(sup, all_planets, player)


def _t_solo_capture(state, att, tgt, incoming=0.0, risk=0):
    """
    Оценка: за СКОЛЬКО ходов att захватит tgt самостоятельно (без supply).
    Считает: сколько ходов копить производство → fly direct.

    Для owned-tgt (растёт оборона) и att одновременно копится — берём
    сходящуюся оценку через несколько итераций. Для нейтрала defender
    статичен, поэтому проще.

    Возвращает (t_total, n_needed). Если att вообще не сможет (например
    маршрут заблокирован солнцем) → (math.inf, ∞).
    """
    if _seg_blocked(att, tgt):
        return math.inf, math.inf

    omega = state.omega
    proj_at = _projected_at(state, tgt.id)
    eff_prod_def = _effective_prod(tgt)
    overkill = _overkill(tgt, risk=risk)
    avail_now = max(0.0, att.ships - RESERVE_ON_ATT)

    # Итеративно сходимся: на ход t_wait att имеет (avail_now + prod*t_wait),
    # пуляет, летит eta(att, tgt, ships, omega) → arrive at t_wait+eta.
    # defender = tgt.ships + eff_prod_def * max(0, t_arrive - proj_at).
    # Нужно: ships ≥ defender + overkill.
    # Решаем перебором t_wait от 0 до horizon.
    best_total = math.inf
    best_needed = math.inf
    for t_wait in range(0, ATTACK_HORIZON + 1):
        ships = avail_now + att.production * t_wait
        if ships < 1:
            continue
        eta = _eta(att, tgt, max(1, int(ships)), omega)
        t_arrive = t_wait + eta
        # ранний перехват по raw, иначе projected
        defender = _defender_at_eta(state, tgt, t_arrive)
        needed = math.ceil(defender - incoming) + overkill
        if ships >= needed:
            best_total = t_arrive
            best_needed = needed
            break
    return best_total, best_needed


# ── Учёт уже летящих дружественных флотов к цели ───────────────────────

def friendly_incoming(state, tgt, t_max, player):
    """
    Сумма кораблей дружественных флотов которые ЦЕЛЯТСЯ В tgt и долетят
    в пределах t_max. Использует угловой критерий — быстро и достаточно.
    Возвращает (total_ships, [eta_per_fleet]).
    """
    total = 0.0
    etas  = []
    for f in state.fleets:
        if f.owner != player:
            continue
        # вектор от флота к цели и к его текущему направлению
        ax = math.atan2(tgt.y - f.y, tgt.x - f.x)
        diff = abs(((ax - f.angle + math.pi) % (2 * math.pi)) - math.pi)
        if diff > INCOMING_ANGLE_TOL:
            continue
        d = math.hypot(tgt.x - f.x, tgt.y - f.y)
        spd = fleet_speed_correct(max(1, int(f.ships)))
        eta = d / max(spd, 1e-6)
        if eta > t_max:
            continue
        total += f.ships
        etas.append(eta)
    return total, etas


# ── Расчёт минимума необходимых кораблей с уточнением eta ──────────────

def _projected_at(state, pid):
    """Возвращает projected_at[pid] из ProjectedState, или 0 если state не проецирован."""
    return getattr(state, 'projected_at', {}).get(pid, 0)


def _raw_planet(state, pid):
    """Поиск raw-планеты по id (если state — ProjectedState).
    Если state без проекции — возвращаем None (caller'у нужно использовать tgt напрямую)."""
    raw = getattr(state, 'raw_planets', None)
    if raw is None:
        return None
    for p in raw:
        if p.id == pid:
            return p
    return None


def _defender_at_eta(state, tgt, eta):
    """Defender, которого МЫ ВСТРЕТИМ при прилёте через `eta` ходов.

    Ключевая тонкость для нейтралов под чужой атакой:
      - tgt = ProjectedState.planets[id]  ⇒  это планета *после флипа* (уже
        враг по проекции).
      - projected_at[id]                  ⇒  ход когда происходит флип.

    Если eta < projected_at[id] — мы прилетим **до того как противник захватил**.
    В этот момент планета ещё RAW (нейтрал/наш предыдущий владелец) и
    защитник = raw.ships без накопления (нейтрал не производит).
    Нет смысла планировать огромный strike против будущего врага, если
    можно прилететь раньше за дёшево.

    Если eta ≥ projected_at — стандартный путь: `tgt.ships` уже учитывает
    результат боя (post-flip), производство добавляется только за время
    после флипа.
    """
    proj_at = _projected_at(state, tgt.id)

    if proj_at > 0 and eta < proj_at:
        raw = _raw_planet(state, tgt.id)
        if raw is not None:
            if raw.owner == NEUTRAL_OWNER:
                # нейтрал не производит — defender = raw.ships, статичен
                return float(raw.ships)
            # owned (наш или вражеский): production до момента нашего прилёта
            return float(raw.ships) + float(raw.production) * eta

    # стандарт: tgt — спроецированный, prod считаем от proj_at
    return tgt.ships + _effective_prod(tgt) * max(0.0, eta - proj_at)


def _required_strike(state, att, tgt, omega, init_eta_ships, incoming, risk=0):
    """
    Считает (eta, defender, needed) для атаки att→tgt.
    Если state — ProjectedState, tgt.ships уже учитывает прибытия до projected_at,
    поэтому defender = tgt.ships + production * max(0, eta - projected_at).
    `incoming` оставлен как параметр совместимости — обычно 0 при проекции.
    """
    eta = _eta(att, tgt, init_eta_ships, omega)
    needed = 0
    overkill = _overkill(tgt, risk=risk)
    for _ in range(ETA_REFINE_ITERS + 1):
        # `_defender_at_eta` сам решает: ранний перехват по raw (eta<proj_at)
        # или стандартный путь по projected с производством после флипа.
        defender = _defender_at_eta(state, tgt, eta)
        effective = max(0.0, defender - incoming)
        needed = int(math.ceil(effective)) + overkill
        new_eta = _eta(att, tgt, max(1, needed), omega)
        if abs(new_eta - eta) < 0.5:
            eta = new_eta
            break
        eta = new_eta
    defender = _defender_at_eta(state, tgt, eta)
    return eta, defender, needed


# ── eval_direct ────────────────────────────────────────────────────────

def eval_direct(state, att, tgt, incoming=0.0, risk=0, neutral_garrison=0):
    """Прямая атака att → tgt.

    x_att = минимум для победы (с учётом incoming).
    Для НЕЙТРАЛОВ: если affordable — добавляем neutral_garrison кораблей
    сверх минимума. Флот прилетит быстрее (Orbit Wars: скорость растёт с
    числом кораблей) и планета сразу получит гарнизон без отдельного трансфера.
    success определяется по минимуму (без garrison) — захватываем всегда если
    можем, garrison = приятный бонус сверху.
    """
    if _seg_blocked(att, tgt):
        return None

    eta, defender, needed = _required_strike(state, att, tgt, state.omega, att.ships, incoming, risk=risk)
    available = att.ships - RESERVE_ON_ATT
    success   = needed <= available

    if success:
        if tgt.owner == NEUTRAL_OWNER and neutral_garrison > 0:
            # Добавляем гарнизон сверх минимума, но не больше доступных кораблей
            x_att = min(available, needed + neutral_garrison)
        else:
            x_att = needed
    else:
        x_att = max(1, available)

    return {
        'mode':      'direct',
        'sup_id':    None, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_sa':    0.0,  'eta_at': eta,    't_total': eta,
        'x_sup':     0,    'x_att':  int(x_att),  'prod_att': att.production,
        'x_tgt':     tgt.ships, 'prod_tgt': tgt.production,
        'strike':    float(x_att) + incoming,
        'defender':  defender,
        'incoming':  incoming,
        'needed':    needed,
        'available': available,
        'slack':     available - needed if success else 0,
        'margin':    float(x_att) + incoming - defender,
        'success':   success,
        'relay_reason': None,
    }


# ── eval_multi_sync ────────────────────────────────────────────────────

def eval_multi_sync(state, sup, att, tgt, incoming=0.0, risk=0):
    """
    Синхронная атака с двух источников. Используется когда att одного НЕ ХВАТАЕТ.
    att шлёт всё → sup добивает минимум.
    """
    if _seg_blocked(sup, tgt) or _seg_blocked(att, tgt):
        return None
    omega = state.omega

    # eta (с полными кораблями для скорости — потом уточним)
    eta_st = _eta(sup, tgt, sup.ships, omega)
    eta_at = _eta(att, tgt, att.ships, omega)
    t_sync = max(eta_st, eta_at)

    # ── Порог "одновременности" ─────────────────────────────────────────────
    # Если sup и att прилетают с большим разрывом, multi_sync теряет смысл:
    # быстрый флот прибывает в одиночку, проигрывает defender'у (att одного
    # по определению не хватает — иначе мы бы выбрали direct), defender
    # регенерирует prod·Δeta кораблей за время полёта медленного, и второй
    # удар тоже сливается. См. MAX_MULTI_SYNC_DELTA_ETA выше.
    if abs(eta_st - eta_at) > MAX_MULTI_SYNC_DELTA_ETA:
        return None

    # ВНИМАНИЕ: agent._plan_parts эмитит оба флота СРАЗУ в этом ходу — никакого
    # «подождать на источнике, пока другой флот долетит, и пока копится
    # производство» в один ход не реализовано. Поэтому в реальный strike
    # попадает только то, что РЕАЛЬНО отправляется сейчас (current ships - reserve).
    # Фантомное `production * (t_sync - eta_X)` убрано — оно искажало success
    # на множители 5-15 ходов производства.
    att_part_max = max(0.0, att.ships - RESERVE_ON_ATT)
    sup_part_max = max(0.0, sup.ships - RESERVE_ON_ATT)
    if att_part_max <= 0 or sup_part_max <= 0:
        return None

    # defender — учитывает ранний перехват (eta<projected_at → raw)
    defender  = _defender_at_eta(state, tgt, t_sync)
    effective = max(0.0, defender - incoming)
    needed    = int(math.ceil(effective)) + _overkill(tgt, risk=risk)

    # если att одного хватает → это direct, тут не место
    if att_part_max >= needed:
        return None

    sup_part_needed = needed - att_part_max
    success         = sup_part_needed <= sup_part_max

    if success:
        sup_part = sup_part_needed
    else:
        sup_part = sup_part_max
    strike = att_part_max + sup_part + incoming

    # x_sup_send = ровно столько, сколько надо отправить (без фейкового
    # «вычета производства» — мы стартуем сейчас, ничего не накапливаем).
    x_sup_send = max(1.0, sup_part)
    x_att_send = max(1, int(att_part_max))

    # ── ПРОВЕРКА: supply должен УСКОРЯТЬ захват ─────────────────────────────
    # Если att без всякого supply справится сам не позже t_sync — supply
    # ничего не даёт (а только тратит корабли с sup). Отвергаем такой план.
    # Допуск 1 ход — учитывает округления и тики.
    t_solo, _ = _t_solo_capture(state, att, tgt, incoming=incoming)
    if t_solo <= t_sync + 1:
        return None

    return {
        'mode':     'multi_sync',
        'sup_id':   sup.id, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_st':   eta_st, 'eta_at': eta_at, 't_total': t_sync,
        'x_sup':    int(math.ceil(x_sup_send)),
        'x_att':    int(x_att_send),
        'prod_sup': sup.production, 'prod_att': att.production,
        'x_tgt':    tgt.ships, 'prod_tgt': tgt.production,
        'strike':   strike,
        'defender': defender,
        'incoming': incoming,
        'needed':   needed,
        'margin':   strike - defender,
        'success':  success,
        'sup_part': sup_part, 'att_part': att_part_max,
        'relay_reason': None,
    }


# ── eval_pipe ──────────────────────────────────────────────────────────

def eval_pipe(state, sup, att, tgt, incoming=0.0, risk=0):
    """Pipeline: sup → att → tgt. sup отправляет минимум для добивки."""
    if _seg_blocked(sup, att) or _seg_blocked(att, tgt):
        return None

    sup_dir_blocked = _seg_blocked(sup, tgt)
    angle_deg = _angle_at_att(sup, att, tgt)
    if not sup_dir_blocked and angle_deg > PIPELINE_MAX_ANGLE_DEG:
        return None

    # Forward-rebase: att «фронтовее» sup (ближе к не-нашим в среднем) ⇒
    # имеет смысл перебрасывать ВСЁ что есть с sup на att (att — лучшая
    # стартовая площадка для следующих атак).
    forward_rebase = _is_more_frontline(att, sup, state.planets, att.owner)

    # Pipeline валиден только если supply несёт самостоятельную ценность:
    #   1) sup→tgt заблокирован солнцем (relay через att — единственный путь), ИЛИ
    #   2) att «фронтовее» sup (forward-rebase оправдан стратегически).
    # Иначе direct sup→tgt доминирует, pipeline = пустая трата кораблей.
    if not sup_dir_blocked and not forward_rebase:
        return None

    omega   = state.omega
    eta_sa  = _eta(sup, att, sup.ships, omega)
    boosted_max = sup.ships + att.ships + att.production * eta_sa
    eta_at  = _eta(att, tgt, boosted_max, omega)
    t_total = eta_sa + eta_at

    # defender — учитывает ранний перехват (eta<projected_at → raw)
    defender    = _defender_at_eta(state, tgt, t_total)
    effective   = max(0.0, defender - incoming)
    needed      = int(math.ceil(effective)) + _overkill(tgt)

    # минимум кораблей с sup, чтобы att.ships + production_eta + x_sup ≥ needed
    att_after_eta = att.ships + att.production * eta_sa
    x_sup_min     = max(1.0, needed - att_after_eta)
    success_relay = x_sup_min <= (sup.ships - RESERVE_ON_ATT)

    if success_relay:
        x_sup_send = x_sup_min
        boosted    = att_after_eta + x_sup_send
    else:
        x_sup_send = max(1, sup.ships - RESERVE_ON_ATT)
        boosted    = att_after_eta + x_sup_send

    # ── Forward-rebase: если att «фронтовее» sup (ближе к не-нашим),
    # имеет смысл перебросить ВСЁ что есть с sup на att. Тогда att
    # становится новой стартовой площадкой для следующих атак, а sup
    # больше не тащит лишние корабли в тылу.
    # `forward_rebase` уже вычислен выше в проверке валидности pipeline.
    sup_avail = max(1, sup.ships - RESERVE_ON_ATT)
    if forward_rebase and sup_avail > x_sup_send:
        x_sup_send = sup_avail

    # x_att = минимум для победы (а не всё что есть)
    x_att_send = min(att.ships - RESERVE_ON_ATT, needed)
    x_att_send = max(1, int(x_att_send))

    # ── Реальный исход ─────────────────────────────────────────────────────
    # ВНИМАНИЕ: pipeline в текущей реализации (см. agent._plan_parts) —
    # это «att стреляет СЕЙЧАС со своими x_att, sup посылает x_sup на att для
    # БУДУЩЕГО». Никакого «att накопит eta_sa ходов и потом выстрелит» в
    # одном ходу не происходит. Поэтому реальная атака на tgt — это только
    # x_att кораблей (sup прилетит на att, а не на tgt).
    #
    # `boosted` (включает att.production·eta_sa и x_sup) — это ИДЕАЛИЗИРОВАННЫЙ
    # сценарий, который НЕ реализуется в один ход. Он используется только
    # для эвристики «имеет ли смысл когда-нибудь так делать», но критерий
    # success ОБЯЗАН опираться на реально летящие x_att vs defender.
    real_strike = x_att_send + incoming
    real_margin = real_strike - defender
    success_real = real_strike >= needed   # x_att сам по себе должен брать
    success     = success_real and success_relay
    margin_real = real_margin
    # старый «boosted» оставляем для отладки
    margin_boosted = boosted + incoming - defender

    # Pipeline без реальной победы att→tgt — это просто потерянные x_att кораблей.
    # Не возвращаем такой план: пусть планировщик выберет direct/multi_sync.
    if not success_real:
        return None

    if sup_dir_blocked:
        relay_reason = f'sup→tgt blocked by sun (angle={angle_deg:.0f}°)'
    else:
        ms = eval_multi_sync(state, sup, att, tgt, incoming=incoming, risk=risk)
        if ms is None:
            relay_reason = f'multi_sync infeasible (angle={angle_deg:.0f}°)'
        elif ms['success'] and not success_relay:
            return None
        elif success_relay and ms['success']:
            ms_total    = _plan_total_ships(ms)
            relay_total = int(math.ceil(x_sup_send)) + x_att_send
            if ms_total <= relay_total:
                return None
            relay_reason = f'relay uses fewer ships ({relay_total} vs {ms_total})'
        elif margin_boosted > 0:
            relay_reason = f'relay succeeds where multi_sync fails (angle={angle_deg:.0f}°)'
        else:
            return None

    return {
        'mode':        'pipeline',
        'sup_id':      sup.id, 'att_id': att.id, 'tgt_id': tgt.id,
        'eta_sa':      eta_sa, 'eta_at': eta_at, 't_total': t_total,
        'x_sup':       int(math.ceil(x_sup_send)),
        'x_att':       x_att_send,
        'prod_att':    att.production,
        'x_tgt':       tgt.ships, 'prod_tgt': tgt.production,
        'strike':      real_strike,
        'defender':    defender,
        'incoming':    incoming,
        'needed':      needed,
        'x_sup_min':   x_sup_min,
        'sup_surplus': sup.ships - x_sup_min,
        'margin':      margin_real,         # реалистичная маржа: x_att vs defender
        'margin_boost': margin_boosted,     # сценарий с накоплением (для отладки)
        'success':     success,
        'angle_deg':   angle_deg,
        'relay_reason': relay_reason,
        'forward_rebase': forward_rebase,   # True ⇒ x_sup = всё что есть у sup
    }


# ── all_plans ──────────────────────────────────────────────────────────

def all_plans(state, tgt, ours, horizon=ATTACK_HORIZON, player=0, risk=0,
              neutral_garrison=0):
    """Все возможные планы атаки на tgt.

    Если state — ProjectedState, incoming уже учтён в tgt.ships, поэтому
    дополнительно его не считаем. Иначе используем friendly_incoming как раньше.
    neutral_garrison — дополнительные корабли сверх минимума при захвате нейтрала
    (передаётся в eval_direct; multi_sync и pipeline не меняем — там арифметика сложнее).
    """
    if hasattr(state, 'projected_at'):
        incoming = 0.0
    else:
        incoming, _ = friendly_incoming(state, tgt, horizon, player)

    plans = []
    for att in ours:
        if att.id == tgt.id:
            continue
        d = eval_direct(state, att, tgt, incoming=incoming, risk=risk,
                        neutral_garrison=neutral_garrison)
        if d and d['t_total'] <= horizon:
            plans.append(d)

    for att, sup in _product(ours, ours):
        if sup.id in (att.id, tgt.id) or att.id == tgt.id:
            continue
        ms = eval_multi_sync(state, sup, att, tgt, incoming=incoming, risk=risk)
        if ms and ms['t_total'] <= horizon:
            plans.append(ms)
        pp = eval_pipe(state, sup, att, tgt, incoming=incoming, risk=risk)
        if pp and pp['t_total'] <= horizon:
            plans.append(pp)
    return plans


# ── best_attacks (главный интерфейс для агента) ────────────────────────

def best_attacks(state, player, targets, top_n=6, horizon=ATTACK_HORIZON):
    """
    Для каждой цели — план с МИНИМУМОМ затраченных кораблей среди успешных.
    Если успешного нет — лучший по margin.

    Атакёры берутся из RAW-состояния (текущие наши планеты), цели — из targets
    (обычно из ProjectedState). Если ProjectedState показывает что цель уже
    станет нашей — её всё равно пропускаем выше (зоны не выберут как цель).
    """
    # Атакёры — реальные текущие наши планеты (не проекция)
    raw_planets = getattr(state, 'raw_planets', state.planets)
    ours = [p for p in raw_planets if p.owner == player]
    if not ours or not targets:
        return []

    is_projected = hasattr(state, 'projected_at')

    best_per_target = []
    for tgt in targets:
        # Если без проекции — старая проверка через friendly_incoming
        if not is_projected:
            incoming, _ = friendly_incoming(state, tgt, horizon, player)
            defender_min = tgt.ships + _effective_prod(tgt) * 1
            if incoming >= defender_min + _overkill(tgt):
                continue

        plans = all_plans(state, tgt, ours, horizon=horizon, player=player)
        if not plans:
            continue

        successful = [pl for pl in plans if pl['success']]
        if successful:
            best = min(successful, key=lambda p: (
                _plan_total_ships(p),
                p['t_total'],
                -p.get('slack', 0),
            ))
        else:
            best = max(plans, key=lambda x: x['margin'])

        if best['success'] and _plan_total_ships(best) < MIN_USEFUL_STRIKE:
            continue
        best_per_target.append(best)

    best_per_target.sort(key=lambda x: (
        -int(x['success']),
        _plan_total_ships(x),
        x['t_total'],
    ))
    return best_per_target[:top_n]


__all__ = [
    'ATTACK_HORIZON', 'SUN_SAFETY_PIPE', 'PIPELINE_MAX_ANGLE_DEG',
    'SAFETY_OVERKILL', 'MIN_USEFUL_STRIKE', 'INCOMING_ANGLE_TOL', 'RESERVE_ON_ATT',
    'MAX_MULTI_SYNC_DELTA_ETA',
    'eval_direct', 'eval_multi_sync', 'eval_pipe',
    'friendly_incoming', 'all_plans', 'best_attacks',
]
