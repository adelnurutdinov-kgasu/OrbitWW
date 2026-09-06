"""
context.py — Game Understanding Layer (GUL).

Глобальное «понимание игры» вычисляется один раз в начале хода и
раздаётся всем cell-decisions как контекст для модулирования action-value.

4 области:
  1. phase           — стадия игры (early/mid/late/endgame).
  2. stance          — стратегическая поза (expansion/attrition/consolidation/desperate).
                       + анализ рельефа из ships (центр масс, разделение,
                       dominance balance по grid).
  3. pressure_field  — для каждой планеты значение поля Σ sign·ships/dist²
                       и роль (deep_rear/rear/frontline/forward/isolated).
  4. opponent_intent — упрощённая ToM: passive/expanding/aggressive/
                       preparing_attack по флотам противника.

Использование:
    from context import compute_context
    ctx = compute_context(state, player)
    # ctx.phase, ctx.stance, ctx.field_at[pid], ctx.opponent_intent, ...
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

NEUTRAL = -1


# ── Параметры (порог-константы; легко крутить) ──────────────────────────

# Phase
PHASE_EARLY_PROGRESS = 0.30
PHASE_MID_PROGRESS   = 0.70

# Stance
STANCE_DESPERATE_RATIO    = 0.35   # ships_ratio < этого → desperate
STANCE_CONSOLIDATION_RATIO = 0.65  # ships_ratio > И нет нейтралов → consolidation
STANCE_EXPANSION_NEUTRAL_FRAC = 0.5   # > половины нейтралов осталось → expansion

# Pressure field
FIELD_SOFTEN     = 5.0    # в знаменателе чтобы избежать singularity на самой планете
FIELD_HIGH       = 0.30   # порог field-value для роли deep_rear/isolated
FIELD_MID        = 0.05   # порог для rear/forward
GRID_STEP        = 20     # для оценки dominance_balance (5×5 = 25 точек)

# Opponent intent
ENEMY_PASSIVE_SHIPS  = 30   # < этого ships у врага → passive
INTENT_ANGLE_TOL     = 0.30 # рад: насколько угол флота должен совпадать с направлением на цель


# ── Структура контекста ─────────────────────────────────────────────────

@dataclass
class GameContext:
    """Глобальное состояние игры на текущий ход. Передаётся в cell-decisions."""
    # — Phase —
    phase: str = 'mid'              # 'early' | 'mid' | 'late' | 'endgame'
    phase_progress: float = 0.0     # 0..1: захвачено vs нейтралов от начала
    total_ships: int = 0            # все ships на планетах + в флотах

    # — Stance —
    stance: str = 'attrition'       # 'expansion'|'attrition'|'consolidation'|'desperate'
    prod_ratio: float = 1.0         # our_prod / enemy_prod (1.0 = паритет)
    ships_ratio: float = 0.5        # our / (our+enemy) — без нейтралов

    # Ship terrain (из «рельефа»)
    our_com:        Tuple[float, float] = (50.0, 50.0)   # центр масс наших (взвешен по ships)
    enemy_com:      Tuple[float, float] = (50.0, 50.0)
    com_separation: float = 0.0     # расстояние между центрами
    dominance_balance: float = 0.0  # (positive_grid_cells - negative) / total — диагностика «кто контролирует пространство»

    # — Pressure field —
    field_at: Dict[int, float] = field(default_factory=dict)   # pid → field-value (>0 наше, <0 чужое)
    role_of:  Dict[int, str]   = field(default_factory=dict)   # pid → 'deep_rear'/'rear'/'frontline'/'forward'/'isolated'

    # — Opponent intent —
    opponent_intent: str = 'unknown'                           # 'passive'|'expanding'|'aggressive'|'preparing_attack'|'idle'
    enemy_internal_transfers: List[Tuple[int, int]] = field(default_factory=list)  # (from_pid, to_pid) — флоты enemy→enemy
    threatened_planets: List[int] = field(default_factory=list)                    # вражеские pid'ы куда летит supply (готовятся стрелять)


# ── Вспомогательные ─────────────────────────────────────────────────────

def _com(planets):
    """Центр масс взвешенный по ships. Если у всех 0 ships — берём геометрический."""
    w = sum(max(0, p.ships) for p in planets)
    if w <= 0:
        if not planets:
            return (50.0, 50.0)
        cx = sum(p.x for p in planets) / len(planets)
        cy = sum(p.y for p in planets) / len(planets)
        return (cx, cy)
    cx = sum(p.x * max(0, p.ships) for p in planets) / w
    cy = sum(p.y * max(0, p.ships) for p in planets) / w
    return (cx, cy)


def _field_value(x, y, raw, player, exclude_id=None):
    """Σ sign · ships / (dist² + soften²). + наши, − чужие, нейтралы игнор."""
    s = 0.0
    sof2 = FIELD_SOFTEN ** 2
    for p in raw:
        if exclude_id == p.id or p.owner == NEUTRAL:
            continue
        sign = 1.0 if p.owner == player else -1.0
        d2 = (x - p.x) ** 2 + (y - p.y) ** 2 + sof2
        s += sign * max(0, p.ships) / d2
    return s


def _classify_role(field_v):
    """field-value → роль клетки в общем рельефе."""
    if field_v >  FIELD_HIGH: return 'deep_rear'
    if field_v >  FIELD_MID:  return 'rear'
    if field_v > -FIELD_MID:  return 'frontline'
    if field_v > -FIELD_HIGH: return 'forward'
    return 'isolated'


def _fleet_target(f, raw):
    """К какой планете летит флот (по углу с допуском). None если не ясно."""
    best_pid, best_d = None, float('inf')
    for p in raw:
        ax = math.atan2(p.y - f.y, p.x - f.x)
        diff = abs(((ax - f.angle + math.pi) % (2 * math.pi)) - math.pi)
        if diff > INTENT_ANGLE_TOL:
            continue
        d = math.hypot(p.x - f.x, p.y - f.y)
        if d < best_d:
            best_d = d
            best_pid = p.id
    return best_pid


# ── Главный entry point ─────────────────────────────────────────────────

def compute_context(state, player) -> GameContext:
    """Собирает GameContext из state. Дешёво (~O(n_planets² + n_fleets·n_planets))."""
    ctx = GameContext()
    raw = list(getattr(state, 'raw_planets', state.planets))
    fleets = list(getattr(state, 'fleets', []))

    ours    = [p for p in raw if p.owner == player]
    enemy   = [p for p in raw if p.owner not in (NEUTRAL, player)]
    neutral = [p for p in raw if p.owner == NEUTRAL]

    # Initial neutrals — оценка, чтобы знать сколько было «в начале»
    initial_planets = []
    init_dict = getattr(state, '_initial_planets', None)
    if init_dict is not None:
        try:
            initial_planets = list(init_dict.values())
        except Exception:
            initial_planets = list(init_dict)
    if not initial_planets:
        initial_planets = list(getattr(state, 'initial_planets', []) or raw)
    initial_neutral = max(1, sum(1 for p in initial_planets if p.owner == NEUTRAL))

    # ─────────────────────────────────────────────────
    # 1. PHASE
    # ─────────────────────────────────────────────────
    ctx.phase_progress = round(1.0 - len(neutral) / initial_neutral, 3)

    fleet_ours  = sum(f.ships for f in fleets if f.owner == player)
    fleet_enemy = sum(f.ships for f in fleets if f.owner not in (NEUTRAL, player))
    our_ships   = sum(p.ships for p in ours) + fleet_ours
    enemy_ships = sum(p.ships for p in enemy) + fleet_enemy
    ctx.total_ships = our_ships + enemy_ships

    if   ctx.phase_progress < PHASE_EARLY_PROGRESS: ctx.phase = 'early'
    elif ctx.phase_progress < PHASE_MID_PROGRESS:   ctx.phase = 'mid'
    elif neutral:                                   ctx.phase = 'late'
    else:                                           ctx.phase = 'endgame'

    # ─────────────────────────────────────────────────
    # 2. STANCE + ship terrain
    # ─────────────────────────────────────────────────
    our_prod   = sum(p.production for p in ours)
    enemy_prod = sum(p.production for p in enemy)
    ctx.prod_ratio  = round(our_prod / max(1, enemy_prod), 3)
    ctx.ships_ratio = round(our_ships / max(1, our_ships + enemy_ships), 3)

    ctx.our_com   = _com(ours)
    ctx.enemy_com = _com(enemy)
    ctx.com_separation = round(math.hypot(
        ctx.our_com[0] - ctx.enemy_com[0],
        ctx.our_com[1] - ctx.enemy_com[1]
    ), 2)

    # Stance (приоритет: desperate > consolidation > expansion > attrition)
    if ctx.ships_ratio < STANCE_DESPERATE_RATIO:
        ctx.stance = 'desperate'
    elif ctx.ships_ratio > STANCE_CONSOLIDATION_RATIO and not neutral:
        ctx.stance = 'consolidation'
    elif len(neutral) > STANCE_EXPANSION_NEUTRAL_FRAC * initial_neutral:
        ctx.stance = 'expansion'
    else:
        ctx.stance = 'attrition'

    # ─────────────────────────────────────────────────
    # 3. PRESSURE FIELD
    # ─────────────────────────────────────────────────
    # Field-value на каждой планете (исключая саму себя из суммы)
    for p in raw:
        v = _field_value(p.x, p.y, raw, player, exclude_id=p.id)
        ctx.field_at[p.id] = round(v, 4)
        ctx.role_of[p.id]  = _classify_role(v)

    # dominance_balance: 5x5 grid через всю карту
    pos = neg = 0
    for x in range(GRID_STEP // 2, 100, GRID_STEP):
        for y in range(GRID_STEP // 2, 100, GRID_STEP):
            v = _field_value(x, y, raw, player)
            if   v >  FIELD_MID: pos += 1
            elif v < -FIELD_MID: neg += 1
    total = max(1, pos + neg)
    ctx.dominance_balance = round((pos - neg) / total, 3)

    # ─────────────────────────────────────────────────
    # 4. OPPONENT INTENT
    # ─────────────────────────────────────────────────
    enemy_pids = {p.id for p in enemy}
    our_pids   = {p.id for p in ours}

    enemy_fleets = [f for f in fleets if f.owner not in (NEUTRAL, player)]
    n_enemy_in_flight = len(enemy_fleets)

    transfers = []                  # enemy → enemy (внутренние)
    threatened = []                 # принимающие (готовятся стрелять)
    toward_us = 0                   # летит к нашим
    for f in enemy_fleets:
        tgt = _fleet_target(f, raw)
        if tgt is None:
            continue
        if tgt in enemy_pids:
            transfers.append((int(getattr(f, 'from_pid', -1)), int(tgt)))
            threatened.append(int(tgt))
        elif tgt in our_pids:
            toward_us += 1

    ctx.enemy_internal_transfers = transfers
    ctx.threatened_planets = sorted(set(threatened))

    if enemy_ships < ENEMY_PASSIVE_SHIPS and n_enemy_in_flight == 0:
        ctx.opponent_intent = 'passive'
    elif transfers:
        ctx.opponent_intent = 'preparing_attack'
    elif n_enemy_in_flight == 0:
        ctx.opponent_intent = 'idle'
    elif toward_us > n_enemy_in_flight * 0.5:
        ctx.opponent_intent = 'aggressive'
    else:
        ctx.opponent_intent = 'expanding'

    return ctx


def context_summary(ctx: GameContext) -> str:
    """Однострочная сводка для лога."""
    return (f"phase={ctx.phase}({ctx.phase_progress:.2f}) "
            f"stance={ctx.stance} ships={ctx.ships_ratio:.2f} prod={ctx.prod_ratio:.2f} "
            f"COM_sep={ctx.com_separation:.0f} dom={ctx.dominance_balance:+.2f} "
            f"opp={ctx.opponent_intent}"
            + (f" threats={ctx.threatened_planets}" if ctx.threatened_planets else "")
            )


__all__ = ['GameContext', 'compute_context', 'context_summary']
