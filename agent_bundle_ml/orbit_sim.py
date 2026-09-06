"""
orbit_sim.py — автономный симулятор Orbit Wars + граф-метрика связности

Не требует kaggle-environments. Физика взята из README + shooting.py.

Основные блоки:
  1. GameState         — снимок состояния игры
  2. generate_map()    — генерация карты по сиду (4-кратная симметрия)
  3. simulate_step()   — один ход движка (production → move → rotate → combat)
  4. GraphConnectivity — 4 варианта весов рёбер + скоры по планетам
  5. JSON import/export — совместимо с форматом kaggle-environments obs
"""

import math
import random
import json
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ── переиспользуем физику из shooting.py ──────────────────────────────────
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from shooting import (
    Planet, Fleet,
    predict_planet_xy, is_orbital, fleet_speed_correct,
    dist, point_to_segment_dist, segment_hits_sun,
    CX, CY, SUN_R, BOARD, ROTATION_LIMIT, LAUNCH_CLEARANCE,
)

# ══════════════════════════════════════════════════════════════════════════
# 1. GameState
# ══════════════════════════════════════════════════════════════════════════

class GameState:
    """Полный снимок состояния игры."""

    def __init__(self, planets: List[Planet], fleets: List[Fleet],
                 omega: float, step: int = 0,
                 initial_planets: Optional[List[Planet]] = None,
                 n_players: int = 2,
                 comet_ids: Optional[set] = None,
                 comets: Optional[list] = None):
        self.planets = list(planets)
        self.fleets  = list(fleets)
        self.omega   = omega
        self.step    = step
        self.n_players = n_players
        # initial_planets нужны для predict_planet_xy (угол от t=0)
        self._initial_planets: Dict[int, Planet] = (
            {p.id: p for p in initial_planets}
            if initial_planets
            else {p.id: p for p in planets}
        )
        self._next_fleet_id = max((f.id for f in fleets), default=-1) + 1
        # Кометы: appearing every ~50 turns, временные планеты с собственными
        # путями. Мы их осознанно игнорируем в zones/priorities, а в projection
        # обрабатываем как «трамплин» — корабли с захваченной кометы (минус 1)
        # уходят на ближайшую нашу планету.
        self.comet_ids: set = set(comet_ids) if comet_ids else set()
        self.comets: list   = list(comets) if comets else []

    # ── удобные срезы ──────────────────────────────────────────────────────

    def planets_of(self, player: int) -> List[Planet]:
        return [p for p in self.planets if p.owner == player]

    def neutral_planets(self) -> List[Planet]:
        return [p for p in self.planets if p.owner == -1]

    def enemy_planets(self, player: int) -> List[Planet]:
        return [p for p in self.planets if p.owner not in (-1, player)]

    def planet_by_id(self, pid: int) -> Optional[Planet]:
        for p in self.planets:
            if p.id == pid:
                return p
        return None

    def initial_by_id(self) -> Dict[int, Planet]:
        return self._initial_planets

    # ── предсказание позиции планеты через turns ходов ────────────────────

    def predict_pos(self, planet: Planet, turns: int) -> Tuple[float, float]:
        init = self._initial_planets.get(planet.id, planet)
        return predict_planet_xy(init.x, init.y, init.radius, self.omega, turns)

    # ── сериализация ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "omega": self.omega,
            "n_players": self.n_players,
            "planets": [list(p) for p in self.planets],
            "fleets":  [list(f) for f in self.fleets],
            "initial_planets": [list(p) for p in self._initial_planets.values()],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        planets = [Planet(*p) for p in d["planets"]]
        fleets  = [Fleet(*f)  for f in d["fleets"]]
        init    = [Planet(*p) for p in d.get("initial_planets", d["planets"])]
        state = cls(
            planets=planets, fleets=fleets,
            omega=d["omega"], step=d.get("step", 0),
            initial_planets=init,
            n_players=d.get("n_players", 2),
        )
        state.game_idx = d.get("game_idx", None)
        return state

    @classmethod
    def from_json(cls, path_or_str: str) -> "GameState":
        if path_or_str.strip().startswith("{"):
            d = json.loads(path_or_str)
        else:
            with open(path_or_str) as fh:
                d = json.load(fh)
        return cls.from_dict(d)

    # ── импорт из kaggle obs ───────────────────────────────────────────────

    @classmethod
    def from_kaggle_obs(cls, obs: dict, step: int = 0) -> "GameState":
        """obs — словарь observation из kaggle_environments.
        step берётся из obs (если есть), иначе из аргумента — нужен для
        late-game-фич zones (late_aggression растёт с phase = step/TOTAL_STEPS).
        """
        planets = [Planet(*p) for p in obs.get("planets", [])]
        fleets  = [Fleet(*f)  for f in obs.get("fleets", [])]
        init    = [Planet(*p) for p in obs.get("initial_planets", obs.get("planets", []))]
        omega   = obs.get("angular_velocity", 0.0) or 0.0
        comet_ids = set(obs.get("comet_planet_ids", []) or [])
        comets    = obs.get("comets", []) or []
        # step может быть в obs как 'step' или 'stepNumber'; иначе используем аргумент
        obs_step = obs.get("step")
        if not isinstance(obs_step, int):
            obs_step = obs.get("stepNumber")
        if isinstance(obs_step, int):
            step = obs_step
        return cls(planets=planets, fleets=fleets, omega=omega,
                   step=step, initial_planets=init,
                   comet_ids=comet_ids, comets=comets)

    def __repr__(self) -> str:
        return (f"GameState(step={self.step}, planets={len(self.planets)}, "
                f"fleets={len(self.fleets)}, omega={self.omega:.4f})")


# ══════════════════════════════════════════════════════════════════════════
# 2. Генерация карты
# ══════════════════════════════════════════════════════════════════════════

def _prod_to_radius(prod: int) -> float:
    return 1.0 + math.log(max(1, prod))


def _place_group(rng: random.Random, pid_counter: list,
                 orbital: bool, home: bool = False,
                 owner_ids: Optional[List[int]] = None) -> List[Planet]:
    """Генерирует группу из 4 планет с 4-кратной симметрией."""
    prod = rng.randint(1, 5) if not home else 3
    radius = _prod_to_radius(prod)

    if orbital:
        # Планета должна быть внутри ROTATION_LIMIT - radius
        max_r = ROTATION_LIMIT - radius - 2.0
        min_r = SUN_R + radius + 3.0
        orb_r = rng.uniform(min_r, max_r)
        angle = rng.uniform(0, math.pi / 2)  # Q1, потом симметрия
        bx = CX + orb_r * math.cos(angle)
        by = CY + orb_r * math.sin(angle)
    else:
        # Статичная: дальше ROTATION_LIMIT
        min_r = ROTATION_LIMIT + radius + 2.0
        max_r = 48.0 - radius
        if min_r >= max_r:
            min_r = ROTATION_LIMIT + radius + 1.0
            max_r = min_r + 4.0
        orb_r = rng.uniform(min_r, max_r)
        angle = rng.uniform(0, math.pi / 2)
        bx = CX + orb_r * math.cos(angle)
        by = CY + orb_r * math.sin(angle)

    # 4-кратная симметрия: (x,y), (100-x,y), (x,100-y), (100-x,100-y)
    corners = [
        (bx,        by),
        (BOARD - bx, by),
        (bx,        BOARD - by),
        (BOARD - bx, BOARD - by),
    ]

    ships_base = rng.randint(5, 25) if not home else 10
    planets = []
    for i, (cx, cy) in enumerate(corners):
        owner = -1
        ships = ships_base
        if home and owner_ids is not None:
            owner = owner_ids[i] if i < len(owner_ids) else -1
            ships = 10
        pid = pid_counter[0]
        pid_counter[0] += 1
        planets.append(Planet(pid, owner, cx, cy, radius, ships, prod))

    return planets


def generate_map(seed: int = 42, n_groups: int = 6,
                 omega: Optional[float] = None,
                 n_players: int = 2) -> GameState:
    """
    Генерирует карту по сиду. Правила:
    - 4-кратная симметрия
    - ≥ 1 орбитальная группа, ≥ 3 статичных
    - 1 домашняя группа (стартовые планеты игроков)
    - omega в диапазоне [0.025, 0.05] если не задано
    """
    rng = random.Random(seed)
    if omega is None:
        omega = rng.uniform(0.025, 0.05)

    pid_counter = [0]
    all_planets: List[Planet] = []

    # Домашняя группа — всегда статичная (для простоты, как часто бывает)
    owner_ids = list(range(n_players)) + [-1] * (4 - n_players)
    home_group = _place_group(rng, pid_counter, orbital=False,
                               home=True, owner_ids=owner_ids)
    all_planets.extend(home_group)

    # Остальные группы
    remaining = n_groups - 1
    n_orbital = max(1, rng.randint(1, max(1, remaining - 2)))
    n_static  = remaining - n_orbital

    for _ in range(n_orbital):
        all_planets.extend(_place_group(rng, pid_counter, orbital=True))
    for _ in range(n_static):
        all_planets.extend(_place_group(rng, pid_counter, orbital=False))

    initial = list(all_planets)
    return GameState(planets=all_planets, fleets=[], omega=omega,
                     initial_planets=initial, n_players=n_players)


# ══════════════════════════════════════════════════════════════════════════
# 3. Симуляция одного хода
# ══════════════════════════════════════════════════════════════════════════

def _fleet_speed(ships: int) -> float:
    return fleet_speed_correct(ships)


def _resolve_combat(garrison_owner: int, garrison_ships: int,
                    arrivals: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    arrivals: list of (owner, ships)
    Возвращает (new_owner, new_ships).
    """
    # Сгруппировать по владельцу
    by_owner: Dict[int, int] = defaultdict(int)
    by_owner[garrison_owner] += garrison_ships
    for owner, ships in arrivals:
        by_owner[owner] += ships

    # Итерации боя: самый сильный vs второй по силе
    while len(by_owner) > 1:
        sorted_owners = sorted(by_owner.items(), key=lambda x: -x[1])
        top_owner, top_ships = sorted_owners[0]
        sec_owner, sec_ships = sorted_owners[1]
        diff = top_ships - sec_ships
        del by_owner[sec_owner]
        if diff == 0:
            del by_owner[top_owner]
        else:
            by_owner[top_owner] = diff

    if not by_owner:
        # Все уничтожены — нейтральная планета остаётся
        return garrison_owner, 0

    winner_owner, winner_ships = next(iter(by_owner.items()))
    return winner_owner, winner_ships


def simulate_step(state: GameState,
                  moves_per_player: Optional[Dict[int, List]] = None) -> GameState:
    """
    Симулирует один ход. moves_per_player:
      {player_id: [[from_planet_id, angle, ships], ...]}

    Порядок: launch → production → fleet move → rotate → combat
    """
    if moves_per_player is None:
        moves_per_player = {}

    planets = {p.id: list(p) for p in state.planets}  # mutable
    fleets: List[list] = [list(f) for f in state.fleets]
    next_fid = state._next_fleet_id

    # ── 1. Fleet launch ────────────────────────────────────────────────────
    for player_id, moves in moves_per_player.items():
        for move in moves:
            src_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            if src_id not in planets:
                continue
            src = planets[src_id]
            if src[1] != player_id:   # не наша
                continue
            ships = min(ships, src[5])
            if ships < 1:
                continue
            # Spawn point
            lx = src[2] + math.cos(angle) * (src[4] + LAUNCH_CLEARANCE)
            ly = src[3] + math.sin(angle) * (src[4] + LAUNCH_CLEARANCE)
            fleets.append([next_fid, player_id, lx, ly, angle, src_id, ships])
            next_fid += 1
            planets[src_id][5] -= ships  # списываем корабли

    # ── 2. Production ──────────────────────────────────────────────────────
    for p in planets.values():
        if p[1] >= 0:  # owned
            p[5] += p[6]  # ships += production

    # ── 3. Fleet movement + collision detection ────────────────────────────
    combat_queue: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    surviving_fleets = []

    for f in fleets:
        fid, fowner, fx, fy, fangle, from_pid, fships = f
        speed = _fleet_speed(fships)
        nx = fx + math.cos(fangle) * speed
        ny = fy + math.sin(fangle) * speed

        # Out of bounds
        if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
            continue  # флот уничтожен

        # Sun collision
        if point_to_segment_dist(CX, CY, fx, fy, nx, ny) < SUN_R:
            continue

        # Planet collision
        hit_pid = None
        hit_d   = float("inf")
        for p in planets.values():
            d = point_to_segment_dist(p[2], p[3], fx, fy, nx, ny)
            if d < p[4] and d < hit_d:
                hit_d   = d
                hit_pid = p[0]

        if hit_pid is not None:
            combat_queue[hit_pid].append((fowner, fships))
        else:
            f[2], f[3] = nx, ny  # обновляем позицию
            surviving_fleets.append(f)

    # ── 4. Planet rotation ─────────────────────────────────────────────────
    if abs(state.omega) > 1e-9:
        for pid, p in planets.items():
            if is_orbital(p[2], p[3], p[4]):
                # rotate by omega
                rx = p[2] - CX
                ry = p[3] - CY
                r  = math.hypot(rx, ry)
                a  = math.atan2(ry, rx) + state.omega
                p[2] = CX + r * math.cos(a)
                p[3] = CY + r * math.sin(a)

    # Sweep: флоты которые попали в орбитальную планету после её поворота
    still_surviving = []
    for f in surviving_fleets:
        fid, fowner, fx, fy, fangle, from_pid, fships = f
        swept = False
        for p in planets.values():
            if math.hypot(fx - p[2], fy - p[3]) < p[4]:
                combat_queue[p[0]].append((fowner, fships))
                swept = True
                break
        if not swept:
            still_surviving.append(f)

    # ── 5. Combat resolution ───────────────────────────────────────────────
    for planet_id, arrivals in combat_queue.items():
        if planet_id not in planets:
            continue
        p = planets[planet_id]
        new_owner, new_ships = _resolve_combat(p[1], p[5], arrivals)
        p[1] = new_owner
        p[5] = new_ships

    # ── Собираем новое состояние ───────────────────────────────────────────
    new_planets = [Planet(*p) for p in planets.values()]
    new_fleets  = [Fleet(*f)  for f in still_surviving]

    new_state = GameState(
        planets=new_planets,
        fleets=new_fleets,
        omega=state.omega,
        step=state.step + 1,
        initial_planets=list(state._initial_planets.values()),
        n_players=state.n_players,
    )
    new_state._next_fleet_id = next_fid
    return new_state


def run_simulation(state: GameState, n_steps: int,
                   policy=None) -> List[GameState]:
    """
    Прогоняет n_steps ходов.
    policy(state, player_id) -> [[from_id, angle, ships], ...]
    """
    history = [state]
    for _ in range(n_steps):
        moves = {}
        if policy:
            for pid in range(state.n_players):
                moves[pid] = policy(history[-1], pid)
        history.append(simulate_step(history[-1], moves))
    return history


# ══════════════════════════════════════════════════════════════════════════
# 4. Граф связности (Graph Connectivity)
# ══════════════════════════════════════════════════════════════════════════

def _travel_time(src: Planet, dst: Planet, ships: int, omega: float,
                 max_iter: int = 4) -> float:
    """
    Оценка времени полёта от src до dst с учётом орбитального движения dst.
    Fixpoint по ETA, как в shooting.py.
    """
    speed = _fleet_speed(max(1, ships))
    if speed < 1e-6:
        return 1e9

    d0 = max(0.5, dist(src.x, src.y, dst.x, dst.y) - src.radius - dst.radius)
    t  = d0 / speed

    for _ in range(max_iter):
        px, py = predict_planet_xy(dst.x, dst.y, dst.radius, omega, int(t))
        d = max(0.5, dist(src.x, src.y, px, py) - src.radius - dst.radius)
        t_new = d / speed
        if abs(t_new - t) < 0.5:
            return t_new
        t = t_new
    return t


# ── Четыре варианта весов рёбер ───────────────────────────────────────────

def _w1_inv_dist(src: Planet, dst: Planet, ships: int,
                 omega: float, **_) -> float:
    """W1: пропорционально 1 / travel_time."""
    t = _travel_time(src, dst, ships, omega)
    return 1.0 / max(t, 0.1)


def _w2_relative_centrality(src: Planet, dst: Planet, ships: int,
                             omega: float, all_planets: List[Planet],
                             **_) -> float:
    """W2: обратно пропорционально среднему времени src до всех планет."""
    avg_t = sum(_travel_time(src, p, ships, omega)
                for p in all_planets if p.id != src.id) / max(1, len(all_planets) - 1)
    t = _travel_time(src, dst, ships, omega)
    return avg_t / max(t, 0.1)  # если src центральная — avg_t мало, вес мал


def _w3_strength_weighted(src: Planet, dst: Planet, ships: int,
                           omega: float, **_) -> float:
    """W3: ships / travel_time — сколько кораблей в секунду можно доставить."""
    t = _travel_time(src, dst, max(1, src.ships), omega)
    return src.ships / max(t, 0.1)


def _w4_neighbor_potential(src: Planet, dst: Planet, ships: int,
                            omega: float, all_planets: List[Planet],
                            **_) -> float:
    """
    W4: ships / t * (1 + λ * Σ_k S_k/t(dst,k))
    — поправка на связность соседей dst.
    """
    t_src_dst = _travel_time(src, dst, max(1, src.ships), omega)
    base = src.ships / max(t_src_dst, 0.1)

    # Потенциал dst: сумма S_k/t(dst→k) по соседям
    neighbor_potential = sum(
        p.ships / max(_travel_time(dst, p, max(1, p.ships), omega), 0.1)
        for p in all_planets if p.id != dst.id and p.id != src.id
    )
    lam = 0.1  # регулятор — можно тюнить
    return base * (1.0 + lam * neighbor_potential)


WEIGHT_VARIANTS = {
    "W1_inv_dist":          _w1_inv_dist,
    "W2_centrality":        _w2_relative_centrality,
    "W3_strength":          _w3_strength_weighted,
    "W4_neighbor_potential": _w4_neighbor_potential,
}


class GraphConnectivity:
    """
    Граф связности для одного состояния игры.

    Каждое ребро i→j имеет вес w(i,j) и знак:
      +1  если src — наш
      -1  если src — враг
       0  если src — нейтрал

    conn(j) = Σ_i sign(i) · w(i,j)   — "давление" на планету j
    delta(t) = Σ_{i: mine} [w(i,t) + w(t,i)]  — прирост связности при захвате t
    """

    def __init__(self, state: GameState, player: int,
                 weight_fn=None, ships_ref: int = 50):
        self.state   = state
        self.player  = player
        self.ships_ref = ships_ref
        self._wfn    = weight_fn or _w3_strength_weighted
        self._cache: Dict[Tuple[int, int], float] = {}

    def edge_weight(self, src: Planet, dst: Planet) -> float:
        key = (src.id, dst.id)
        if key not in self._cache:
            self._cache[key] = self._wfn(
                src, dst, self.ships_ref, self.state.omega,
                all_planets=self.state.planets
            )
        return self._cache[key]

    def _sign(self, p: Planet) -> int:
        if p.owner == self.player:
            return +1
        if p.owner == -1:
            return 0
        return -1

    def signed_pressure(self, target: Planet) -> float:
        """Σ_i sign(i) * w(i→target) — давление на target."""
        return sum(
            self._sign(src) * self.edge_weight(src, target)
            for src in self.state.planets
            if src.id != target.id
        )

    def all_pressures(self) -> Dict[int, float]:
        """signed_pressure для всех планет."""
        return {p.id: self.signed_pressure(p) for p in self.state.planets}

    def team_connectivity(self) -> float:
        """Внутренняя связность нашей сети: Σ_{i,j: mine} w(i→j)."""
        mine = self.state.planets_of(self.player)
        return sum(
            self.edge_weight(i, j)
            for i in mine for j in mine
            if i.id != j.id
        )

    def marginal_capture_value(self, target: Planet) -> float:
        """
        Прирост team_connectivity при захвате target.
        = Σ_{i: mine} [w(i→target) + w(target→i)]
        """
        mine = self.state.planets_of(self.player)
        return sum(
            self.edge_weight(m, target) + self.edge_weight(target, m)
            for m in mine
        )

    def capture_priority(self) -> List[Tuple[Planet, float]]:
        """
        Сортировка не-наших планет по marginal_capture_value (убывание).
        """
        non_mine = [p for p in self.state.planets if p.owner != self.player]
        scored = [(p, self.marginal_capture_value(p)) for p in non_mine]
        scored.sort(key=lambda x: -x[1])
        return scored

    def shortcut_planets(self) -> List[Tuple[Planet, float]]:
        """
        Планеты на кратчайших путях (low travel time от наших до вражеских).
        Возвращает не-наши планеты, которые являются 'шорткатами':
        их захват наиболее сокращает путь до вражеской сети.
        """
        mine   = self.state.planets_of(self.player)
        enemy  = self.state.enemy_planets(self.player)
        others = [p for p in self.state.planets
                  if p.owner != self.player and p.id not in {e.id for e in enemy}]
        # Для каждой нейтральной/враж планеты: насколько она ближе к врагу?
        result = []
        for candidate in others:
            # shortest time from candidate to nearest enemy
            t_to_enemy = min(
                (_travel_time(candidate, e, self.ships_ref, self.state.omega)
                 for e in enemy), default=1e9
            )
            # shortest time from our planets to candidate
            t_from_mine = min(
                (_travel_time(m, candidate, self.ships_ref, self.state.omega)
                 for m in mine), default=1e9
            )
            # composite: быстро достать + быстро от кандидата к врагу
            score = 1.0 / max(t_from_mine + t_to_enemy, 0.1)
            result.append((candidate, score))
        result.sort(key=lambda x: -x[1])
        return result

    def summary(self) -> dict:
        pressures = self.all_pressures()
        prio      = self.capture_priority()
        shortcuts = self.shortcut_planets()
        return {
            "player": self.player,
            "team_connectivity": round(self.team_connectivity(), 3),
            "pressures": {pid: round(v, 3) for pid, v in pressures.items()},
            "capture_priority": [
                {"planet_id": p.id, "owner": p.owner,
                 "ships": p.ships, "prod": p.production,
                 "marginal_value": round(v, 3)}
                for p, v in prio[:8]
            ],
            "shortcut_candidates": [
                {"planet_id": p.id, "owner": p.owner, "score": round(v, 4)}
                for p, v in shortcuts[:5]
            ],
        }


# ══════════════════════════════════════════════════════════════════════════
# 5. Утилиты для экспорта/сравнения вариантов
# ══════════════════════════════════════════════════════════════════════════

def compare_weight_variants(state: GameState, player: int,
                             ships_ref: int = 50) -> Dict[str, dict]:
    """
    Считает capture_priority для каждого из 4 вариантов весов.
    Возвращает dict {variant_name: ranked_list}.
    """
    result = {}
    for name, wfn in WEIGHT_VARIANTS.items():
        gc = GraphConnectivity(state, player, weight_fn=wfn, ships_ref=ships_ref)
        prio = gc.capture_priority()
        result[name] = [
            {"planet_id": p.id, "owner": p.owner,
             "ships": p.ships, "prod": p.production,
             "value": round(v, 4)}
            for p, v in prio[:6]
        ]
    return result


def export_history(history: List[GameState], path: str):
    """Сохраняет последовательность состояний в JSON."""
    data = [s.to_dict() for s in history]
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Сохранено {len(history)} состояний → {path}")


def load_history(path: str) -> List[GameState]:
    """Загружает историю из JSON (совместимо с export_history и colab_export)."""
    with open(path) as fh:
        data = json.load(fh)
    states = []
    skipped = 0
    for i, d in enumerate(data):
        # Пропускаем пустые состояния (kaggle step=0 бывает без планет)
        planets_raw = d.get("planets", [])
        if not planets_raw:
            skipped += 1
            continue
        try:
            if "omega" in d:
                s = GameState.from_dict(d)
            else:
                s = GameState.from_kaggle_obs(d, step=i)
            states.append(s)
        except Exception as e:
            skipped += 1
    if skipped:
        print(f"  (пропущено {skipped} пустых/невалидных состояний)")
    return states


# ══════════════════════════════════════════════════════════════════════════
# Быстрый smoke-test при запуске как скрипт
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    print("=== Генерация карты seed=42 ===")
    state = generate_map(seed=42, n_groups=6)
    print(state)
    print(f"Планет: {len(state.planets)}, орбитальных: "
          f"{sum(1 for p in state.planets if is_orbital(p.x, p.y, p.radius))}")

    for p in state.planets:
        tag = "orb" if is_orbital(p.x, p.y, p.radius) else "sta"
        print(f"  [{tag}] id={p.id} owner={p.owner} ships={p.ships} "
              f"prod={p.production} pos=({p.x:.1f},{p.y:.1f})")

    print("\n=== Graph Connectivity (W3_strength, player=0) ===")
    gc = GraphConnectivity(state, player=0, weight_fn=_w3_strength_weighted)
    s  = gc.summary()
    pprint.pprint(s)

    print("\n=== Сравнение вариантов весов ===")
    comp = compare_weight_variants(state, player=0)
    for name, ranked in comp.items():
        ids = [f"p{r['planet_id']}(o{r['owner']})" for r in ranked[:4]]
        print(f"  {name:30s}: {ids}")

    print("\n=== 10 шагов симуляции ===")
    history = run_simulation(state, n_steps=10)
    print(f"Шаги: {[s.step for s in history]}")
    final = history[-1]
    for pid in range(state.n_players):
        print(f"  Игрок {pid}: {len(final.planets_of(pid))} планет, "
              f"{sum(p.ships for p in final.planets_of(pid))} кораблей")

    print("\nOK")
