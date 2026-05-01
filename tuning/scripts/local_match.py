"""
local_match.py — детерминированный runner матчей без kaggle_environments.

Используем напрямую orbit_sim:
  1. generate_map(seed) — стартовое состояние (deterministic).
  2. Loop simulate_step с двумя агентами; каждому передаём obs в формате
     совместимом с agent.GameState.from_kaggle_obs.
  3. Победитель = тот, у кого после конца игры больше общая масса
     (ships на планетах + ships в флотах). Игра завершается когда одна
     сторона потеряла все планеты ИЛИ достигнут MAX_STEPS.

ПОЧЕМУ НЕ kaggle_environments
=============================
В реальных прогонах с одинаковым `seed` kaggle_environments выдаёт
разные карты/исходы. Локальный orbit_sim полностью детерминирован
(random.Random(seed) только в generate_map, simulate_step без рандома).
Для тюнинга это критично: одна и та же конфигурация всегда даёт один
и тот же результат → можно честно сравнивать.

ЦЕНА
====
Карта генерится по правилам orbit_sim, а не точно по kaggle. Распределение
похожее (см. match_analyzer features), но не побитово совпадает. Для
ранжирования весов это не важно. Для финальной валидации лучших
конфигов — гонять через test.ipynb / kaggle env вручную.
"""

import os, sys, time, math
import inspect
from typing import Callable, Dict, List, Optional


def _make_caller(agent_fn):
    """Адаптер под разные сигнатуры agent: agent(obs) / agent(obs, config).
    Определяется один раз по signature, чтобы не делать try/except на каждом ходу
    (это глотало TypeError → moves=[] → агент столбом, исход независим от весов)."""
    try:
        sig = inspect.signature(agent_fn)
        n_params = len(sig.parameters)
    except (TypeError, ValueError):
        n_params = 2  # safe fallback
    if n_params <= 1:
        return lambda obs, config: agent_fn(obs)
    return agent_fn


# Ленивый импорт orbit_sim — caller должен добавить bundle в sys.path
def _import_sim(bundle_dir):
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    import orbit_sim
    return orbit_sim


# ── obs builder ─────────────────────────────────────────────────────────────

def _build_obs(state, player: int) -> dict:
    """
    Формирует obs словарь который видит агент. Формат — как у kaggle_env:
      planets: [[id, owner, x, y, radius, ships, production], ...]
      fleets:  [[id, owner, x, y, angle, from_pid, ships], ...]
      angular_velocity: omega
      step: int
      player: int
      initial_planets: исходные (для projection)
      comet_planet_ids: []
    """
    return {
        'step': state.step,
        'player': player,
        'angular_velocity': state.omega,
        'omega': state.omega,
        'n_players': state.n_players,
        'planets':  [list(p) for p in state.planets],
        'fleets':   [list(f) for f in state.fleets],
        'initial_planets': [list(p) for p in state._initial_planets.values()],
        'comet_planet_ids': list(getattr(state, 'comet_ids', set()) or []),
        'comets': [],
    }


# ── проверки конца ──────────────────────────────────────────────────────────

def _player_total_mass(state, player: int) -> int:
    """ships на планетах + ships во флотах (потенциальный потенциал)."""
    s = sum(p.ships for p in state.planets if p.owner == player)
    s += sum(f.ships for f in state.fleets if f.owner == player)
    return s


def _player_has_anything(state, player: int) -> bool:
    if any(p.owner == player for p in state.planets):
        return True
    if any(f.owner == player for f in state.fleets):
        return True
    return False


# ── главный runner ──────────────────────────────────────────────────────────

def run_match(
    seed: int,
    agent_a: Callable,
    agent_b: Callable,
    bundle_dir: str,
    max_steps: int = 300,
    n_groups: int = 6,
    omega: Optional[float] = None,
    return_history: bool = False,
) -> dict:
    """
    Прогоняет один матч A vs B с фиксированным `seed`. Полностью детерминирован.

    agent_a / agent_b — функции `agent(obs, config=None) -> moves`.
    bundle_dir       — путь к папке с orbit_sim.py (и нашим бандлом).
    Возвращает dict: seed, win_a, win_b, draw, ships_diff, steps, time_sec, history?.
    """
    sim = _import_sim(bundle_dir)

    state = sim.generate_map(seed=seed, n_groups=n_groups, omega=omega, n_players=2)

    history = [state.to_dict()] if return_history else None
    t0 = time.time()
    config = {'actTimeout': 1}   # на всякий случай — обходим внутренние таймеры sub'ов

    # адаптируем под сигнатуры (наш agent(obs) vs sub(obs, config))
    call_a = _make_caller(agent_a)
    call_b = _make_caller(agent_b)

    final_step = 0
    n_errors_a = 0
    n_errors_b = 0
    for step_idx in range(max_steps):
        # win condition: один из игроков потерял всё
        a_alive = _player_has_anything(state, 0)
        b_alive = _player_has_anything(state, 1)
        if not (a_alive and b_alive):
            break

        # обе стороны делают ход (синхронно, как в kaggle env)
        obs_a = _build_obs(state, 0)
        obs_b = _build_obs(state, 1)
        try:
            moves_a = call_a(obs_a, config) or []
        except Exception as e:
            n_errors_a += 1
            if n_errors_a <= 3:   # логируем первые ошибки чтобы не молчали
                print(f"[local_match] agent A error step={step_idx}: {type(e).__name__}: {e}",
                      file=sys.stderr)
            moves_a = []
        try:
            moves_b = call_b(obs_b, config) or []
        except Exception as e:
            n_errors_b += 1
            if n_errors_b <= 3:
                print(f"[local_match] agent B error step={step_idx}: {type(e).__name__}: {e}",
                      file=sys.stderr)
            moves_b = []

        state = sim.simulate_step(state, {0: moves_a, 1: moves_b})
        if return_history:
            history.append(state.to_dict())
        final_step = step_idx + 1

    a_mass = _player_total_mass(state, 0)
    b_mass = _player_total_mass(state, 1)

    # Итог:
    # - если только один с непустой массой → он победил
    # - иначе — у кого больше total mass (ships в планетах+флотах)
    if a_mass > 0 and b_mass == 0:
        win_a, win_b, draw = 1, 0, 0
    elif b_mass > 0 and a_mass == 0:
        win_a, win_b, draw = 0, 1, 0
    elif a_mass > b_mass:
        win_a, win_b, draw = 1, 0, 0
    elif b_mass > a_mass:
        win_a, win_b, draw = 0, 1, 0
    else:
        win_a, win_b, draw = 0, 0, 1

    result = {
        'seed': seed,
        'win_a': win_a, 'win_b': win_b, 'draw': draw,
        'ships_a_final': a_mass,
        'ships_b_final': b_mass,
        'ships_diff': a_mass - b_mass,
        'steps': final_step,
        'time_sec': round(time.time() - t0, 2),
    }
    if return_history:
        result['history'] = history
    return result
