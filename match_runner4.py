"""
match_runner4.py — воркер для параллельного запуска 4-player матчей.

Отличия от match_runner.py:
  - env.run([our, opp1, opp2, opp3]) — четыре агента
  - Результаты: reward всех 4 игроков, наш ранг (1–4), timeseries по каждому
  - task['opps'] — список из 3 имён оппонентов (может повторяться)
"""

import os, sys, time, importlib.util

HERE       = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(HERE, "agent_bundle_swarm 2")
SUB_PATH   = os.path.join(HERE, "sub2.py")

if BUNDLE_DIR not in sys.path:
    sys.path.insert(0, BUNDLE_DIR)


def _silence():
    for k in ("ORBIT_AGENT_LOG", "ORBIT_AGENT_LOG_PLANS_ALL",
              "ORBIT_AGENT_LOG_FLEETS",
              "SUB_AGENT_LOG", "SUB_AGENT_LOG_MISSIONS_ALL",
              "SUB_AGENT_LOG_FLEETS"):
        os.environ.pop(k, None)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_opp(opp_name, uid):
    if opp_name == 'noop':
        return lambda obs, cfg=None: []
    if opp_name == 'sub2':
        return _load(SUB_PATH, f"_opp_{uid}").agent
    opp_path = os.path.join(HERE, f"{opp_name}.py")
    if not os.path.exists(opp_path):
        raise ValueError(f"unknown opp '{opp_name}': не найден {opp_path}")
    return _load(opp_path, f"_opp_{uid}").agent


def run_match4(task):
    """
    task = {
        'seed':     int,
        'our_path': str,
        'opps':     [str, str, str],   # три оппонента (могут повторяться)
        'weights':  dict | None,
        'label':    str,
    }
    Возвращает dict — всё сериализуемо.
    """
    _silence()

    seed     = task['seed']
    our_path = task.get('our_path', os.path.join(BUNDLE_DIR, 'agent.py'))
    opps     = task.get('opps', ['sub2', 'sub2', 'sub2'])
    weights  = task.get('weights') or {}
    label    = task.get('label', '+'.join(opps))

    uid = f"{seed}_{label}_{abs(hash(str(task)))%10**6}"

    # ── наш агент ──────────────────────────────────────────────────────
    our_mod = _load(our_path, f"_our_{uid}")
    if weights:
        from dataclasses import asdict, fields as dc_fields
        from swarm import SwarmWeights
        base = asdict(SwarmWeights())
        base.update(weights)
        for f in dc_fields(SwarmWeights):
            if f.type is int and f.name in base:
                base[f.name] = int(base[f.name])
        our_mod.SWARM_WEIGHTS = SwarmWeights(**base)

    # ── три оппонента ──────────────────────────────────────────────────
    opp_fns = [_load_opp(name, f"{uid}_{i}") for i, name in enumerate(opps)]

    # ── матч ───────────────────────────────────────────────────────────
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0  = time.time()
    env.run([our_mod.agent] + opp_fns)
    elapsed = time.time() - t0

    # ── извлечение результатов ─────────────────────────────────────────
    final   = env.steps[-1]
    rewards = [float(final[i].get('reward') or 0) for i in range(4)]
    r0      = rewards[0]

    # Ранг: сколько игроков получили reward > нашего → наш ранг = N+1
    rank = 1 + sum(1 for r in rewards[1:] if r > r0)

    # timeseries: корабли + производство для каждого из 4 игроков
    ts_ships = [[] for _ in range(4)]
    ts_prod  = [[] for _ in range(4)]

    for step in env.steps:
        obs     = step[0].get('observation') or {}
        planets = obs.get('planets') or []
        fleets  = obs.get('fleets')  or []
        ships = [0.0] * 4
        prod  = [0.0] * 4
        for p in planets:
            own  = p[1] if isinstance(p, (list, tuple)) else p.get('owner', -1)
            shps = float(p[5] if isinstance(p, (list, tuple)) else p.get('ships', 0) or 0)
            prdn = float(p[6] if isinstance(p, (list, tuple)) else p.get('production', 0) or 0)
            if 0 <= own < 4:
                ships[own] += shps
                prod[own]  += prdn
        for f in fleets:
            own  = f[1] if isinstance(f, (list, tuple)) else f.get('owner', -1)
            shps = float(f[6] if isinstance(f, (list, tuple)) else f.get('ships', 0) or 0)
            if 0 <= own < 4:
                ships[own] += shps
        for i in range(4):
            ts_ships[i].append(ships[i])
            ts_prod[i].append(prod[i])

    # финальные планеты
    last_obs     = (env.steps[-1][0].get('observation') or {})
    last_planets = last_obs.get('planets') or []
    n_planets = [0] * 4
    for p in last_planets:
        own = p[1] if isinstance(p, (list, tuple)) else p.get('owner', -1)
        if 0 <= own < 4:
            n_planets[own] += 1

    result = {
        'seed':    seed,
        'label':   label,
        'opps':    '+'.join(opps),
        'rank':    rank,
        'win':     int(rank == 1),
        'top2':    int(rank <= 2),
        'steps':   len(env.steps),
        'time_sec': round(elapsed, 1),
    }
    for i in range(4):
        result[f'reward_{i}'] = rewards[i]
        result[f'ships_final_{i}'] = ts_ships[i][-1] if ts_ships[i] else 0
        result[f'prod_final_{i}']  = ts_prod[i][-1]  if ts_prod[i]  else 0
        result[f'planets_final_{i}'] = n_planets[i]
        result[f'ts_ships_{i}'] = ts_ships[i]
        result[f'ts_prod_{i}']  = ts_prod[i]

    return result
