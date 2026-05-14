"""
mcts.py — MCTS поиск в ПАР (Пространстве Адекватных Решений) для ранней игры.

Архитектура: одноуровневое дерево — каждый узел соответствует НАШЕМУ ходу.
На каждой итерации:
  1. Selection  — UCT с prior-смещением от модели противника.
  2. Expansion  — добавляем один дочерний узел (наш ход + сэмпл хода противника
                  из его ПАР → simulate_step → новое состояние).
  3. Simulation — быстрый роллаут до MAX_ROLLOUT_DEPTH ходов.
  4. Backprop   — обновляем visits и total_score вверх по дереву.

Модель противника (OpponentModel):
  • Хранит EMA-веса трёх типов его ходов: захват нейтрала, захват нашей планеты, реинфорс.
  • update() вызывается из agent.py после каждого хода (при наличии наблюдений).
  • sample_action() возвращает взвешенно-случайный ход для diversification.
  • prior_weight() используется в UCB как смещение (exploration bias).

Публичный интерфейс:
  class OpponentModel
  run_mcts(root_state, my_actions, opp_actions, opponent_model,
           player, time_budget) -> Optional[Action]
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from action_space import (
    Action, generate_actions, generate_opponent_actions,
    MIN_SEND, MAX_TRAVEL_TIME,
)
from orbit_sim import GameState, simulate_step
from projection import NEUTRAL_OWNER

# ── Параметры MCTS ─────────────────────────────────────────────────────────
C_UCT              = 1.41421   # sqrt(2) — баланс exploration/exploitation
MAX_ROLLOUT_DEPTH  = 25         # максимальная глубина роллаута (игровых ходов)
PRIOR_WEIGHT       = 0.25      # вес prior в UCB: ucb += PRIOR_WEIGHT * prior(action)
MAX_EXPAND_ACTIONS = 10        # максимум действий в узле (обрезаем по приоритету)
ROLLOUT_MIN_DEPTH  = 1         # минимум ходов роллаута даже при нехватке времени


# ══════════════════════════════════════════════════════════════════════════
# Модель противника
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class OpponentModel:
    """Адаптивная вероятностная модель поведения противника.

    Три типа действий:
      capture_neutral — захват нейтральных планет
      capture_enemy   — атака наших планет
      reinforce       — усиление своих

    После каждого хода update() обновляет веса через EMA.
    В дереве sample_action() сэмплирует пропорционально весам.
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        'capture_neutral': 1.0,
        'capture_enemy':   1.0,
        'reinforce':       0.5,
    })
    alpha: float = 0.3   # коэффициент EMA (0 = никогда не меняется, 1 = мгновенно)

    def _classify(self, action: Action, planets_by_id: dict, our_player: int) -> str:
        tgt = planets_by_id.get(action.target_id)
        if tgt is None:
            return 'capture_neutral'
        if tgt.owner == our_player:
            return 'capture_enemy'
        if tgt.owner == NEUTRAL_OWNER or tgt.owner == -1:
            return 'capture_neutral'
        return 'reinforce'

    def update(self, observed_actions: List[Action],
               planets_by_id: dict, our_player: int):
        """Обновить модель по наблюдаемым ходам противника за прошлый ход."""
        if not observed_actions:
            return
        counts: Dict[str, int] = {k: 0 for k in self.weights}
        for a in observed_actions:
            k = self._classify(a, planets_by_id, our_player)
            counts[k] = counts.get(k, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return
        for k in self.weights:
            obs_frac = counts.get(k, 0) / total
            self.weights[k] = (1 - self.alpha) * self.weights[k] + self.alpha * obs_frac

    def action_weight(self, action: Action, planets_by_id: dict, our_player: int) -> float:
        """Вес (prior) отдельного действия противника."""
        k = self._classify(action, planets_by_id, our_player)
        return max(0.01, self.weights.get(k, 1.0))

    def sample_action(self, actions: List[Action],
                      planets_by_id: dict, our_player: int) -> Optional[Action]:
        """Выбрать ход противника взвешенно-случайно по модели."""
        if not actions:
            return None
        weights = [self.action_weight(a, planets_by_id, our_player) for a in actions]
        total = sum(weights)
        if total < 1e-9:
            return random.choice(actions)
        r = random.uniform(0.0, total)
        cumsum = 0.0
        for a, w in zip(actions, weights):
            cumsum += w
            if r <= cumsum:
                return a
        return actions[-1]

    def normalized_prior(self, action: Action,
                         planets_by_id: dict, our_player: int,
                         total_weight: float) -> float:
        """Prior [0,1]: weight / total_weight (для UCB-смещения)."""
        if total_weight < 1e-9:
            return 1.0 / max(1, len(self.weights))
        return self.action_weight(action, planets_by_id, our_player) / total_weight


# ══════════════════════════════════════════════════════════════════════════
# Узел дерева
# ══════════════════════════════════════════════════════════════════════════

class MCTSNode:
    """Узел MCTS. Каждый узел соответствует одному НАШЕМУ выбору действия."""

    __slots__ = ('state', 'action', 'parent', 'children',
                 'visits', 'total_score', '_untried')

    def __init__(self, state: GameState,
                 action: Optional[Action] = None,
                 parent: 'MCTSNode' = None):
        self.state:       GameState          = state
        self.action:      Optional[Action]   = action   # None для корня
        self.parent:      Optional[MCTSNode] = parent
        self.children:    List[MCTSNode]     = []
        self.visits:      int                = 0
        self.total_score: float              = 0.0
        self._untried:    Optional[List[Action]] = None  # ленивая init

    # ── Статистика ──────────────────────────────────────────────────────

    def avg_score(self) -> float:
        return self.total_score / self.visits if self.visits > 0 else 0.0

    def ucb(self, parent_visits: int, prior: float = 0.0) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.total_score / self.visits
        explore = C_UCT * math.sqrt(math.log(max(1, parent_visits)) / self.visits)
        return exploit + explore + PRIOR_WEIGHT * prior

    # ── Дерево ─────────────────────────────────────────────────────────

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child_ucb(self, prior_fn=None) -> 'MCTSNode':
        """UCT выбор ребёнка. prior_fn(action) -> float ∈ [0,1]."""
        best, best_s = None, float('-inf')
        for ch in self.children:
            prior = prior_fn(ch.action) if (prior_fn and ch.action) else 0.0
            s = ch.ucb(self.visits, prior)
            if s > best_s:
                best_s, best = s, ch
        return best

    def expand(self, action: Action, new_state: GameState) -> 'MCTSNode':
        child = MCTSNode(new_state, action=action, parent=self)
        self.children.append(child)
        return child

    def backpropagate(self, score: float):
        node = self
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent


# ══════════════════════════════════════════════════════════════════════════
# Конвертация состояний
# ══════════════════════════════════════════════════════════════════════════

def _to_game_state(state) -> GameState:
    """Конвертировать ProjectedState → GameState для simulate_step.

    При работе с ProjectedState используем raw_planets + raw_fleets, чтобы
    simulate_step работал с реальным текущим состоянием (флоты в полёте
    учитываются, а не «съедены» проекцией).
    """
    if isinstance(state, GameState):
        return state

    raw_p  = list(getattr(state, 'raw_planets', state.planets))
    raw_fl = list(getattr(state, 'raw_fleets', []))

    init_dict = {}
    fn = getattr(state, 'initial_by_id', None)
    if callable(fn):
        init_dict = fn()
    elif hasattr(state, '_initial_planets'):
        init_dict = state._initial_planets

    init_list = list(init_dict.values()) if init_dict else raw_p

    gs = GameState(
        planets=raw_p,
        fleets=raw_fl,
        omega=state.omega,
        step=state.step,
        initial_planets=init_list,
        n_players=getattr(state, 'n_players', 2),
        comet_ids=set(getattr(state, 'comet_ids', set()) or set()),
    )
    return gs


# ══════════════════════════════════════════════════════════════════════════
# Оценочная функция и роллаут
# ══════════════════════════════════════════════════════════════════════════

def _evaluate(state: GameState, player: int) -> float:
    """Быстрая эвристика позиции [0, 1] с точки зрения player.

    Взвешенная комбинация:
      40% — доля кораблей
      35% — доля производства (важнее кораблей в перспективе)
      25% — доля планет
    """
    our_ships = our_prod = our_cnt = 0
    opp_ships = opp_prod = opp_cnt = 0

    for p in state.planets:
        if p.owner == player:
            our_ships += p.ships
            our_prod  += p.production
            our_cnt   += 1
        elif p.owner not in (NEUTRAL_OWNER, -1):
            opp_ships += p.ships
            opp_prod  += p.production
            opp_cnt   += 1

    # Концы игры
    if opp_cnt == 0 and opp_ships == 0:
        return 1.0
    if our_cnt == 0 and our_ships == 0:
        return 0.0

    total_ships = our_ships + opp_ships
    total_prod  = our_prod  + opp_prod
    total_cnt   = our_cnt   + opp_cnt

    s_ship   = our_ships / max(1, total_ships)
    s_prod   = our_prod  / max(1, total_prod)
    s_planet = our_cnt   / max(1, total_cnt)

    return 0.1 * s_ship + 0.6 * s_prod + 0.3 * s_planet


def _rollout_move(state: GameState, player_id: int) -> Optional[list]:
    """Быстрый эвристический ход для роллаута (без aim_hybrid).

    Использует math.atan2 как приближение угла — значительно быстрее
    aim_hybrid, достаточно для оценки листьев.
    Логика: источник с наибольшим излишком → ближайшая нейтральная/вражеская цель.
    """
    my_pl = [p for p in state.planets if p.owner == player_id and p.ships >= MIN_SEND]
    if not my_pl:
        return None
    targets = [p for p in state.planets if p.owner != player_id]
    if not targets:
        return None

    # Источник: максимальный излишек над резервом
    src = max(my_pl, key=lambda p: max(0, p.ships - int(p.production * 1.5)))
    avail = src.ships - 1
    if avail < MIN_SEND:
        return None

    # Цель: нейтральные предпочтительны (дешевле захватить), иначе ближайшая
    neutrals = [t for t in targets if t.owner in (NEUTRAL_OWNER, -1)]
    pool = neutrals if neutrals else targets
    import math as _m
    tgt = min(pool, key=lambda t: _m.hypot(t.x - src.x, t.y - src.y))

    angle = _m.atan2(tgt.y - src.y, tgt.x - src.x)
    ships = min(avail, max(MIN_SEND, int(tgt.ships) + 2))
    return [[src.id, angle, ships]]


def _rollout(state: GameState, player: int, opponent: int,
             depth: int) -> float:
    """Быстрый роллаут до `depth` ходов. Обе стороны — жадная эвристика."""
    cur = state
    for _ in range(depth):
        moves: dict = {}
        our = _rollout_move(cur, player)
        opp = _rollout_move(cur, opponent)
        if our:
            moves[player] = our
        if opp:
            moves[opponent] = opp
        if not moves:
            break
        try:
            cur = simulate_step(cur, moves)
        except Exception:
            break
    return _evaluate(cur, player)


# ══════════════════════════════════════════════════════════════════════════
# Главная функция MCTS
# ══════════════════════════════════════════════════════════════════════════

def _sort_actions(actions: List[Action]) -> List[Action]:
    """Приоритизировать действия для expansion: сначала захваты с min кораблей."""
    return sorted(
        actions,
        key=lambda a: (0 if a.mode == 'capture' else 1, a.ships)
    )


def run_mcts(root_state,
             my_actions:      List[Action],
             opp_actions:     List[Action],
             opponent_model:  OpponentModel,
             player:          int,
             time_budget:     float = 0.40) -> Optional[Action]:
    """Запустить MCTS и вернуть лучший Action для player.

    Args:
        root_state:     Текущее состояние (GameState или ProjectedState).
        my_actions:     ПАР для нас (из generate_actions).
        opp_actions:    ПАР для противника (из generate_opponent_actions).
        opponent_model: Адаптивная модель противника.
        player:         Наш player_id.
        time_budget:    Бюджет времени (секунды).

    Returns:
        Лучший Action или None если ПАР пуст.

    Гарантия: никогда не падает — любые ошибки симуляции пропускаются.
    """
    if not my_actions:
        return None

    opponent = (player + 1) % 2      # для 2-player игры
    gs_root  = _to_game_state(root_state)
    planets_by_id = {p.id: p for p in gs_root.planets}

    # Общий вес ПАР противника (для нормализации prior)
    total_opp_weight = sum(
        opponent_model.action_weight(a, planets_by_id, player)
        for a in opp_actions
    ) if opp_actions else 1.0

    def _prior_fn(action: Action) -> float:
        """Prior [0,1] для action противника (для UCB-смещения)."""
        return opponent_model.normalized_prior(
            action, planets_by_id, player, total_opp_weight
        )

    # Ограничиваем и сортируем наши действия для expansion
    sorted_my = _sort_actions(my_actions)[:MAX_EXPAND_ACTIONS]

    root = MCTSNode(gs_root)
    root._untried = list(sorted_my)   # копируем — pop() изменяет список

    deadline   = time.perf_counter() + time_budget
    iterations = 0

    while time.perf_counter() < deadline:
        # ── Фаза 1: Selection ──────────────────────────────────────────
        node = root
        # Спускаемся пока узел полностью раскрыт (untried пуст) И имеет детей
        while (node._untried is not None
               and len(node._untried) == 0
               and len(node.children) > 0):
            node = node.best_child_ucb(_prior_fn)

        # ── Фаза 2: Expansion ──────────────────────────────────────────
        # Инициализируем untried для нового листа (ленивая генерация)
        if node._untried is None:
            child_acts = _sort_actions(
                generate_actions(node.state, player, MAX_TRAVEL_TIME)
            )[:MAX_EXPAND_ACTIONS]
            node._untried = list(child_acts)

        expanded_node = node
        if node._untried:
            action = node._untried.pop()

            # Сэмплируем ход противника по модели (может быть None)
            opp_action = opponent_model.sample_action(
                opp_actions, planets_by_id, player
            )

            # Применяем оба хода → получаем новое состояние
            moves: dict = {player: action.to_moves()}
            if opp_action is not None:
                moves[opponent] = opp_action.to_moves()

            try:
                new_state = simulate_step(node.state, moves)
            except Exception:
                # Если симуляция падает — пропускаем этот ход
                continue

            expanded_node = node.expand(action, new_state)

        # ── Фаза 3: Simulation (rollout) ───────────────────────────────
        remaining = deadline - time.perf_counter()
        # Адаптируем глубину роллаута по остатку времени
        roll_depth = max(
            ROLLOUT_MIN_DEPTH,
            min(MAX_ROLLOUT_DEPTH, int(remaining / 0.004))
        )
        score = _rollout(expanded_node.state, player, opponent, roll_depth)

        # ── Фаза 4: Backpropagation ────────────────────────────────────
        expanded_node.backpropagate(score)
        iterations += 1

    # ── Выбор финального хода ─────────────────────────────────────────
    if not root.children:
        # Ни одной итерации не удалось завершить → fallback
        return sorted_my[0] if sorted_my else None

    # Robustness: выбираем действие с наибольшим числом посещений
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action


__all__ = [
    'OpponentModel',
    'MCTSNode',
    'run_mcts',
    '_evaluate',
    '_to_game_state',
]
