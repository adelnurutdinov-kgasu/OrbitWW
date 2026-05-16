"""
gradient_likelihood.py — inference через градиентное поле намерений.

Каждое наблюдаемое действие оппонента представляется как вектор намерения Δs.
Likelihood архетипа k = exp(cosine(Δs_obs, g_k) / τ).

Интерфейс:
    from opponent_learning.gradient_likelihood import GradientLikelihood
    gl = GradientLikelihood()          # грузит gradient_fields.json
    likelihoods = gl.get(observed_actions, state, opp_id)
    # → dict {cluster_id: likelihood_weight}
"""

from __future__ import annotations
import json
import math
import numpy as np
from pathlib import Path
from typing import List

_HERE    = Path(__file__).parent
_GF_PATH = _HERE / "results" / "gradient_fields.json"

TEMPERATURE = 8.0   # τ: выше → мягче различия, ниже → резче
MIN_WEIGHT  = 0.02  # floor чтобы не было нулей


class GradientLikelihood:
    """Байесовская likelihood через cosine-сходство градиентных полей."""

    def __init__(self, gf_path: str | None = None, temperature: float = TEMPERATURE):
        path = Path(gf_path) if gf_path else _GF_PATH
        if not path.exists():
            self._ready = False
            return

        data = json.loads(path.read_text())
        self._dims   = data['delta_dims']
        self._tau    = temperature
        self._ready  = True

        self._clusters = {}   # int → {g, sigma, prior}
        for k_str, v in data['clusters'].items():
            k = int(k_str)
            self._clusters[k] = {
                'g':     np.array(v['gradient'], dtype=np.float32),
                'sigma': np.array(v['std'],      dtype=np.float32),
                'prior': float(v['prior']),
                'n':     int(v['n_samples']),
            }

        self._cluster_ids = sorted(self._clusters.keys())

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def cluster_ids(self) -> list:
        return list(self._cluster_ids)

    # ── Feature extraction from live game ─────────────────────────────────

    def extract_delta(self,
                      observed_actions: List[dict],
                      state,
                      opp_id: int) -> np.ndarray | None:
        """Вычисляет delta-вектор из реально наблюдённых действий оппонента.

        Args:
            observed_actions: [{from_id, target_id, ships, action_type}, ...]
                action_type: 'reinforce' | 'capture_neutral' | 'attack_enemy'
            state: GameState (orbit_sim.GameState)
            opp_id: id оппонента

        Returns:
            np.ndarray shape (10,) или None если действий нет.
        """
        if not observed_actions:
            return None

        planets_by_id = {p.id: p for p in state.planets}
        n_pl  = max(1, len(state.planets))
        n_own = max(1, sum(1 for p in state.planets if p.owner == opp_id))
        n_neut = sum(1 for p in state.planets if p.owner == -1)

        own_ships_total  = sum(p.ships for p in state.planets if p.owner == opp_id)
        all_ships_total  = sum(max(0, p.ships) for p in state.planets)
        own_ship_ratio   = own_ships_total / max(1, all_ships_total)

        # Флоты: под угрозой — вражеские флоты летят на наши планеты
        own_planet_ids = {p.id for p in state.planets if p.owner == opp_id}
        under_threat   = sum(
            1 for f in state.fleets
            if f.owner != opp_id
            for p in state.planets
            if p.owner == opp_id  # упрощённо: считаем вражеских флотов в воздухе
        )
        # точнее: уникальные наши планеты куда летят вражеские флоты
        # (используем упрощённый подсчёт без destination inference)
        enemy_fleet_count = sum(1 for f in state.fleets if f.owner != opp_id and f.owner != -1)

        # Разбиваем действия по типам
        n_neutral = sum(1 for a in observed_actions if a['action_type'] == 'capture_neutral')
        n_enemy   = sum(1 for a in observed_actions if a['action_type'] == 'attack_enemy')
        n_reinf   = sum(1 for a in observed_actions if a['action_type'] == 'reinforce')
        n_total   = max(1, len(observed_actions))

        total_ships = max(1, sum(a['ships'] for a in observed_actions))

        # Характеристики целей
        prod_vals, dist_vals, overkill_vals = [], [], []
        for a in observed_actions:
            tgt = planets_by_id.get(a['target_id'])
            if tgt is None:
                continue
            prod_vals.append(float(tgt.production))
            # дистанция: нормализуем по max_dist (оценка)
            src = planets_by_id.get(a.get('from_id', -1))
            if src:
                d = math.hypot(tgt.x - src.x, tgt.y - src.y)
                # грубая нормализация: max ≈ 2 * BOARD_HALF ≈ 400
                dist_vals.append(min(1.0, d / 400.0))
            tgt_ships = max(1.0, tgt.ships)
            overkill_vals.append(a['ships'] / tgt_ships)

        avg_prod     = sum(prod_vals)     / len(prod_vals)     if prod_vals     else 0.0
        avg_dist     = sum(dist_vals)     / len(dist_vals)     if dist_vals     else 0.3
        avg_overkill = sum(overkill_vals) / len(overkill_vals) if overkill_vals else 1.0
        avg_overkill = min(avg_overkill, 10.0)

        # own ships approximate fraction sent
        sf = total_ships / max(1, own_ships_total)
        sf = min(sf, 1.5)

        delta = np.array([
            n_neutral / n_total,                          # d_neutral_grab
            n_enemy   / n_total,                          # d_enemy_press
            n_reinf   / n_total,                          # d_reinforce
            min(1.0, sf / 1.5),                           # d_commitment
            1.0 / (1.0 + avg_overkill),                   # d_efficiency
            avg_dist,                                     # d_reach
            min(1.0, avg_prod / 5.0),                     # d_prod_hunger
            min(1.0, enemy_fleet_count / max(1, n_own)),  # d_threat_resp (прокси)
            n_neut / n_pl,                                # d_exp_opportunity
            own_ship_ratio,                               # d_position
        ], dtype=np.float32)

        return delta

    # ── Likelihood ─────────────────────────────────────────────────────────

    def get(self,
            observed_actions: List[dict],
            state,
            opp_id: int,
            smoothing: float = MIN_WEIGHT) -> dict:
        """Возвращает dict {cluster_id: likelihood_weight}.

        Вес нормализован в сумму 1.0.
        """
        delta = self.extract_delta(observed_actions, state, opp_id)

        if delta is None or not self._ready:
            # Нет действий → равномерное распределение
            n = len(self._cluster_ids)
            return {k: 1.0 / n for k in self._cluster_ids}

        weights = {}
        for k in self._cluster_ids:
            g  = self._clusters[k]['g']
            na = np.linalg.norm(delta)
            ng = np.linalg.norm(g)
            if na < 1e-9 or ng < 1e-9:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(delta, g) / (na * ng))
            weights[k] = math.exp(cos_sim / self._tau)

        # Сглаживание + нормализация
        total = sum(weights.values())
        return {k: max(smoothing, v / total) for k, v in weights.items()}

    def cluster_names(self) -> dict:
        return {k: f"cluster_{k}" for k in self._cluster_ids}


# Singleton
_instance: GradientLikelihood | None = None


def get_gradient_likelihood() -> GradientLikelihood:
    global _instance
    if _instance is None:
        _instance = GradientLikelihood()
    return _instance
