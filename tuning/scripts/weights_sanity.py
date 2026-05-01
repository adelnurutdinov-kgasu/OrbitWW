#!/usr/bin/env python3
"""
weights_sanity.py — быстрая проверка: реально ли веса влияют на исход матча.

Прогоняет ОДИН seed с радикально разными конфигурациями SwarmWeights и
печатает (steps, ships_diff). Если все строки идентичны — либо инжекция
сломана, либо на этом сиде агент ведёт себя по одному и тому же сценарию
независимо от весов (нет конкуренции за ships, плюс/минус никаких
дополнительных действий).

Запуск:  python3 tuning/scripts/weights_sanity.py [SEED]
"""

import os, sys, importlib.util, time
from dataclasses import asdict

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
BUNDLE_DIR   = os.path.join(PROJECT_ROOT, "agent_bundle_swarm")
OUR_PATH     = os.path.join(BUNDLE_DIR, "agent.py")
SUB_PATH     = os.path.join(PROJECT_ROOT, "sub2.py")

sys.path.insert(0, HERE); sys.path.insert(0, BUNDLE_DIR)
from local_match import run_match
from swarm import SwarmWeights


def _load_module(path, name):
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_with(seed, label, weights, use_swarm=True):
    """Загружает свежий agent_mod, инжектит веса, прогоняет матч."""
    suf = f"{seed}_{label.replace(' ', '_')}_{int(time.time()*1000)%1000000}"
    our = _load_module(OUR_PATH, f"_san_our_{suf}")
    sub = _load_module(SUB_PATH, f"_san_sub_{suf}")
    our.SWARM_WEIGHTS = weights
    our.USE_SWARM = use_swarm
    res = run_match(seed, our.agent, sub.agent, bundle_dir=BUNDLE_DIR, max_steps=300)
    return res


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    cfgs = [
        ("baseline default",        SwarmWeights(),                                       True),
        ("USE_SWARM=False",         SwarmWeights(),                                       False),
        ("redistribute=ON",         SwarmWeights(enable_redistribute=True,
                                                 transfer_floor=5, transfer_min_ships=5,
                                                 transfer_thresh=0.0),                    True),
        ("risk_tolerance=2",        SwarmWeights(risk_tolerance=2),                       True),
        ("activity HUGE",           SwarmWeights(activity_weight=20.0, idle_floor=5),     True),
        ("distance comfort=1",      SwarmWeights(distance_comfort=0.95),                  True),
        ("ships_weight=0",          SwarmWeights(ships_weight=0.0),                       True),
        ("eta_bonus=0",             SwarmWeights(eta_bonus=0.0),                          True),
        ("priority_bonus=0",        SwarmWeights(priority_bonus=0.0),                     True),
        ("ALL aggressive",          SwarmWeights(activity_weight=8.0, idle_floor=10,
                                                 risk_tolerance=2, distance_comfort=0.7,
                                                 ships_weight=8.0, enable_redistribute=True,
                                                 transfer_floor=5, transfer_min_ships=5,
                                                 transfer_thresh=0.0),                    True),
    ]

    print(f"=== seed={seed} sanity sweep ===\n")
    print(f"{'config':<26} {'use_swarm':<10} {'steps':>5} {'a_ships':>8} {'b_ships':>8} {'diff':>8}")
    print("-" * 80)
    base_diff = base_steps = None
    for label, w, use_sw in cfgs:
        try:
            r = run_with(seed, label, w, use_swarm=use_sw)
        except Exception as e:
            print(f"{label:<26}  ERROR: {e}")
            continue
        marker = ''
        if base_diff is None:
            base_diff, base_steps = r['ships_diff'], r['steps']
        else:
            if (r['ships_diff'], r['steps']) != (base_diff, base_steps):
                marker = '  ← DIFFERENT (веса влияют)'
        print(f"{label:<26} {str(use_sw):<10} {r['steps']:>5} {r['ships_a_final']:>8} "
              f"{r['ships_b_final']:>8} {r['ships_diff']:>+8}{marker}")

    print()
    print("Если ВСЕ строки одинаковые: либо инжекция SWARM_WEIGHTS не работает,")
    print("либо на этом сиде у агента нет планов которые веса могут переупорядочить.")
    print("Попробуй другой seed или проверь USE_SWARM/SWARM_WEIGHTS в agent.py.")


if __name__ == "__main__":
    main()
