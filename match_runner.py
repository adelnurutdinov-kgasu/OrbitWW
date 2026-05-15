"""
match_runner.py — воркер для параллельного запуска матчей из Jupyter.

Должен лежать рядом с test.ipynb и быть импортируемым модулем
(не определяться inline в ноутбуке) — требование macOS spawn-режима.
"""

import os, sys, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
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
    # каждый процесс грузит свою копию — нет конфликта кешей
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_match(task):
    """
    task = {
        'seed':       int,
        'our_path':   str,          # путь к agent.py
        'opp':        'sub2'|'noop',
        'weights':    dict | None,  # overrides для SwarmWeights (опционально)
        'label':      str,          # произвольная метка (для группировки)
    }
    Возвращает dict с результатами — всё сериализуемо (no pandas, no env objects).
    """
    _silence()

    seed      = task['seed']
    our_path  = task.get('our_path', os.path.join(BUNDLE_DIR, 'agent.py'))
    opp_name  = task.get('opp', 'sub2')
    weights   = task.get('weights') or {}
    label     = task.get('label', opp_name)

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

    # ── оппонент ───────────────────────────────────────────────────────
    if opp_name == 'sub2':
        opp_mod = _load(SUB_PATH, f"_opp_{uid}")
        opp_fn  = opp_mod.agent
    elif opp_name == 'noop':
        opp_fn  = lambda obs, cfg=None: []
    else:
        # произвольный файл — ищем HERE/<opp_name>.py
        opp_path = os.path.join(HERE, f"{opp_name}.py")
        if not os.path.exists(opp_path):
            raise ValueError(f"unknown opp '{opp_name}': файл не найден: {opp_path}")
        opp_mod = _load(opp_path, f"_opp_{uid}")
        opp_fn  = opp_mod.agent

    # ── матч ───────────────────────────────────────────────────────────
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0  = time.time()
    env.run([our_mod.agent, opp_fn])
    elapsed = time.time() - t0

    # ── извлечение данных ──────────────────────────────────────────────
    final   = env.steps[-1]
    r0 = float(final[0].get('reward') or 0)
    r1 = float(final[1].get('reward') or 0)

    # временной ряд: корабли + производство по ходам
    ts_ships0, ts_ships1, ts_prod0, ts_prod1 = [], [], [], []
    for step in env.steps:
        obs     = step[0].get('observation') or {}
        planets = obs.get('planets') or []
        fleets  = obs.get('fleets')  or []
        s0 = s1 = p0 = p1 = 0.0
        for p in planets:
            own  = p[1] if isinstance(p, (list, tuple)) else p.get('owner', -1)
            shps = float(p[5] if isinstance(p, (list, tuple)) else p.get('ships', 0) or 0)
            prod = float(p[6] if isinstance(p, (list, tuple)) else p.get('production', 0) or 0)
            if own == 0:   s0 += shps; p0 += prod
            elif own == 1: s1 += shps; p1 += prod
        for f in fleets:
            own  = f[1] if isinstance(f, (list, tuple)) else f.get('owner', -1)
            shps = float(f[6] if isinstance(f, (list, tuple)) else f.get('ships', 0) or 0)
            if own == 0:   s0 += shps
            elif own == 1: s1 += shps
        ts_ships0.append(s0); ts_ships1.append(s1)
        ts_prod0.append(p0);  ts_prod1.append(p1)

    last_obs     = (env.steps[-1][0].get('observation') or {})
    last_planets = last_obs.get('planets') or []
    n0 = sum(1 for p in last_planets if (p[1] if isinstance(p,(list,tuple)) else p.get('owner'))==0)
    n1 = sum(1 for p in last_planets if (p[1] if isinstance(p,(list,tuple)) else p.get('owner'))==1)

    return {
        'seed':      seed,
        'label':     label,
        'opp':       opp_name,
        'win':       int(r0 > r1),
        'draw':      int(r0 == r1),
        'r0':        r0, 'r1': r1,
        'steps':     len(env.steps),
        'n0_final':  n0, 'n1_final': n1,
        'ships0_final': ts_ships0[-1] if ts_ships0 else 0,
        'ships1_final': ts_ships1[-1] if ts_ships1 else 0,
        'prod0_final':  ts_prod0[-1]  if ts_prod0  else 0,
        'prod1_final':  ts_prod1[-1]  if ts_prod1  else 0,
        'ships_diff':   (ts_ships0[-1] - ts_ships1[-1]) if ts_ships0 else 0,
        'time_sec':  round(elapsed, 1),
        # временные ряды для графиков
        'ts_ships0': ts_ships0,
        'ts_ships1': ts_ships1,
        'ts_prod0':  ts_prod0,
        'ts_prod1':  ts_prod1,
    }
