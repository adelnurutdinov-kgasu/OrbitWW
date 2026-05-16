#!/usr/bin/env python3
"""
07_gradient_field.py — строит «рельеф намерений» оппонентов.

Идея:
  Каждый запуск флота — это попытка игрока сдвинуть состояние мира
  в определённом направлении. Мы представляем это как вектор Δs
  в N-мерном пространстве признаков.

  Каждый архетип (кластер) имеет характерный «градиент» g_k:
  среднее направление в котором игроки этого кластера толкают мир.

  Likelihood для байесовской модели:
    P(action | archetype_k) ∝ exp( cosine(Δs_obs, g_k) / τ )

Delta-вектор (10 измерений, все в [0, 1]):
  d_neutral_grab   — доля атак на нейтралов (экспансия)
  d_enemy_press    — доля атак на врага (прямая агрессия)
  d_reinforce      — доля подкреплений (оборона/консолидация)
  d_commitment     — масштаб: сколько кораблей потрачено (норм.)
  d_efficiency     — 1/(1+overkill): точность против расточительности
  d_reach          — дальность целей (далеко vs близко)
  d_prod_hunger    — производство захватываемых планет (ценность целей)
  d_threat_resp    — ответ на угрозу: under_threat / own_planets
  d_exp_opportunity— сколько нейтралов доступно: n_neutral / n_planets
  d_position       — позиция: own_ship_ratio (лидирует или проигрывает)

Запуск:
  python3 opponent_learning/scripts/07_gradient_field.py
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

HERE     = Path(__file__).parent
PROC_DIR = HERE.parent / "data" / "processed"
RES_DIR  = HERE.parent / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

DELTA_DIMS = [
    'd_neutral_grab',    # экспансия в нейтралов
    'd_enemy_press',     # давление на врага
    'd_reinforce',       # оборонительная консолидация
    'd_commitment',      # масштаб вложений
    'd_efficiency',      # точность (1 - расточительность)
    'd_reach',           # дальность атак
    'd_prod_hunger',     # ценность захватываемых ресурсов
    'd_threat_resp',     # ответ на угрозу
    'd_exp_opportunity', # доступность нейтралов
    'd_position',        # текущая позиция (winning?)
]


# ── Delta-вектор из одного шага ──────────────────────────────────────────────

def compute_delta(row: pd.Series) -> np.ndarray:
    """Вычисляет intended-delta для одного активного шага.

    Все измерения в [0, 1] — нормализованы чтобы косинус работал корректно.
    """
    # Клипаем выбросы
    sf   = float(np.clip(row['ships_fraction_sent'], 0.0, 1.5))
    ok   = float(np.clip(row['avg_overkill'],        0.0, 10.0))
    prod = float(row.get('avg_target_prod', 0.0))
    dist = float(np.clip(row.get('avg_target_dist_frac', 0.0), 0.0, 1.0))

    n_pl   = max(1, int(row['n_planets']))
    n_own  = max(1, int(row.get('n_own_planets', 1)))
    n_neut = max(0, int(row['n_neutral_planets']))

    under_threat = float(row.get('own_under_threat_count', 0.0))
    own_ratio    = float(np.clip(row['own_ship_ratio'], 0.0, 1.0))

    d = np.array([
        float(row['attack_neutral_ratio']),             # d_neutral_grab
        float(row['attack_enemy_ratio']),               # d_enemy_press
        float(row['reinforce_ratio']),                  # d_reinforce
        float(np.clip(sf / 1.5, 0.0, 1.0)),            # d_commitment  (норм. к 1.5)
        1.0 / (1.0 + ok),                              # d_efficiency  (высокий ok → низкая эф.)
        dist,                                          # d_reach
        float(np.clip(prod / 5.0, 0.0, 1.0)),          # d_prod_hunger (max prod = 5)
        float(np.clip(under_threat / max(1, n_own), 0.0, 1.0)),  # d_threat_resp
        float(n_neut / n_pl),                          # d_exp_opportunity
        own_ratio,                                     # d_position
    ], dtype=np.float32)

    return d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Загрузка ─────────────────────────────────────────────────────────────
    print("Загружаем steps.csv …")
    df = pd.read_csv(PROC_DIR / 'steps.csv')
    cl = pd.read_csv(PROC_DIR / 'features_clustered.csv')[
        ['episode_id', 'player_idx', 'cluster']
    ]
    df = df.merge(cl, on=['episode_id', 'player_idx'], how='left')
    df = df[df['cluster'].notna()].copy()
    df['cluster'] = df['cluster'].astype(int)

    # Только активные ходы
    active = df[df['did_act'] == 1].copy()
    print(f"  Всего строк: {len(df):,}  |  активных: {len(active):,}")

    # ── Вычисляем delta-вектор для каждого шага ──────────────────────────────
    print("\nВычисляем delta-векторы …")
    deltas = np.stack([compute_delta(row) for _, row in active.iterrows()])
    print(f"  Матрица delta: {deltas.shape}  (шагов × {len(DELTA_DIMS)} измерений)")

    # Добавляем в датафрейм
    delta_df = pd.DataFrame(deltas, columns=DELTA_DIMS, index=active.index)
    active = pd.concat([active[['episode_id','player_idx','step','cluster','reward']], delta_df], axis=1)

    # ── Градиент g_k и ковариация per cluster ────────────────────────────────
    print("\n" + "═"*70)
    print("Градиентные поля по кластерам  g_k = mean(Δs | cluster=k)")
    print("═"*70)

    clusters = sorted(active['cluster'].unique())
    gradients = {}   # cluster_id → g_k  (mean delta)
    stds      = {}   # cluster_id → sigma_k (std per dim)
    counts    = {}

    for k in clusters:
        mask = active['cluster'] == k
        dk   = deltas[mask.values]
        g    = dk.mean(axis=0)
        s    = dk.std(axis=0)
        gradients[k] = g
        stds[k]      = s
        counts[k]    = int(mask.sum())

    # Печать по кластерам
    print(f"\n{'Dim':<22}", end='')
    for k in clusters:
        print(f"  cl{k}({counts[k]//1000}k)", end='')
    print()
    print('─' * (22 + 12 * len(clusters)))

    for i, dim in enumerate(DELTA_DIMS):
        print(f"  {dim:<20}", end='')
        for k in clusters:
            print(f"  {gradients[k][i]:>8.3f}", end='')
        print()

    # ── Косинусная матрица между кластерами ─────────────────────────────────
    print("\n── Косинусное сходство между g_k (насколько архетипы различаются) ──")
    print(f"{'':>6}", end='')
    for k in clusters:
        print(f"  cl{k}", end='')
    print()
    for ki in clusters:
        print(f"  cl{ki}", end='')
        for kj in clusters:
            c = cosine(gradients[ki], gradients[kj])
            print(f"  {c:.3f}", end='')
        print()

    # ── Анализ «чистоты» кластеров ──────────────────────────────────────────
    print("\n── Дисперсия внутри кластеров (меньше = чище) ──")
    print(f"{'Dim':<22}", end='')
    for k in clusters:
        print(f"  cl{k}", end='')
    print()
    for i, dim in enumerate(DELTA_DIMS):
        print(f"  {dim:<20}", end='')
        for k in clusters:
            print(f"  {stds[k][i]:>8.3f}", end='')
        print()

    # ── Самые различающиеся измерения (discriminative power) ────────────────
    print("\n── Discriminative power (std между g_k / mean внутри σ) ──")
    between_std = np.array([gradients[k] for k in clusters]).std(axis=0)
    within_std  = np.array([stds[k]      for k in clusters]).mean(axis=0)
    disc = between_std / np.maximum(within_std, 1e-6)
    for i in np.argsort(-disc):
        print(f"  {DELTA_DIMS[i]:<22}  between={between_std[i]:.4f}  "
              f"within={within_std[i]:.4f}  disc={disc[i]:.3f}")

    # ── Сохраняем ────────────────────────────────────────────────────────────
    out = {
        'delta_dims': DELTA_DIMS,
        'clusters': {},
    }
    for k in clusters:
        out['clusters'][str(k)] = {
            'gradient':  gradients[k].tolist(),
            'std':       stds[k].tolist(),
            'n_samples': counts[k],
            'prior':     counts[k] / sum(counts.values()),
        }

    out_path = RES_DIR / 'gradient_fields.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ gradient_fields.json → {out_path}")

    # ── Пишем inference-модуль ────────────────────────────────────────────────
    _write_inference_module()
    print(f"✓ gradient_likelihood.py → {HERE.parent / 'gradient_likelihood.py'}")

    print("\nСледующий шаг:")
    print("  python3 opponent_learning/scripts/08_integrate_gradient.py")


# ── Inference-модуль ─────────────────────────────────────────────────────────

INFERENCE_CODE = '''"""
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
        self._dims   = data[\'delta_dims\']
        self._tau    = temperature
        self._ready  = True

        self._clusters = {}   # int → {g, sigma, prior}
        for k_str, v in data[\'clusters\'].items():
            k = int(k_str)
            self._clusters[k] = {
                \'g\':     np.array(v[\'gradient\'], dtype=np.float32),
                \'sigma\': np.array(v[\'std\'],      dtype=np.float32),
                \'prior\': float(v[\'prior\']),
                \'n\':     int(v[\'n_samples\']),
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
                action_type: \'reinforce\' | \'capture_neutral\' | \'attack_enemy\'
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
        n_neutral = sum(1 for a in observed_actions if a[\'action_type\'] == \'capture_neutral\')
        n_enemy   = sum(1 for a in observed_actions if a[\'action_type\'] == \'attack_enemy\')
        n_reinf   = sum(1 for a in observed_actions if a[\'action_type\'] == \'reinforce\')
        n_total   = max(1, len(observed_actions))

        total_ships = max(1, sum(a[\'ships\'] for a in observed_actions))

        # Характеристики целей
        prod_vals, dist_vals, overkill_vals = [], [], []
        for a in observed_actions:
            tgt = planets_by_id.get(a[\'target_id\'])
            if tgt is None:
                continue
            prod_vals.append(float(tgt.production))
            # дистанция: нормализуем по max_dist (оценка)
            src = planets_by_id.get(a.get(\'from_id\', -1))
            if src:
                d = math.hypot(tgt.x - src.x, tgt.y - src.y)
                # грубая нормализация: max ≈ 2 * BOARD_HALF ≈ 400
                dist_vals.append(min(1.0, d / 400.0))
            tgt_ships = max(1.0, tgt.ships)
            overkill_vals.append(a[\'ships\'] / tgt_ships)

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
            g  = self._clusters[k][\'g\']
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
'''


def _write_inference_module():
    out = HERE.parent / "gradient_likelihood.py"
    out.write_text(INFERENCE_CODE)


if __name__ == '__main__':
    main()
