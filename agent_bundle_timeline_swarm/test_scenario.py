"""
test_scenario.py — синтетический сценарий с измеримым выигрышем от
ускорения захвата.

Запуск:
    cd /Users/adel/Documents/GitHub/OrbitWW/agent_bundle_timeline_swarm
    python test_scenario.py

Что тестируем:
  • Планировщик НЕ ломается на простых случаях (smoke).
  • На сценарии, специально построенном под концепцию ускорения,
    timeline_swarm выдаёт отправку из «свободной» планеты, а классический
    swarm — нет.

Сценарий «accelerate»:
  Planet 0  — наша, рядом с целью, ships=100, отправили флот 60 ships → P2.
              Флот летит ~40 ходов, прибывает с гарнизоном P2 = 15 (нейтрал) →
              захват с margin=45 через ~40 ходов.
  Planet 1  — наша, в тылу, ships=120, бездействует. Дистанция до P2
              позволяет прилететь за 35 ходов с любым флотом.
  Planet 2  — НЕЙТРАЛЬНАЯ ЦЕЛЬ, ships=15, prod=2.0. По проекции станет
              нашей через 40 ходов.
  Planet 3  — наша далеко, ships=50, prod=1.0 (для шумовой нагрузки).

Ожидание:
  • Классический swarm: P2 уже наша в проекции → никаких новых отправок.
  • Timeline swarm: видит accelerate-opportunity (delay=40), есть бюджет
    на P1 → должен послать ~50 ships из P1 в P2 для сокращения времени
    захвата. Выигрыш = production(P2) × Δt = 2.0 × ~5 = 10 ships.

Если так — концепция работает. Если timeline_swarm НЕ посылает —
посмотрим в debug и поправим параметры.
"""

import sys
import os
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_BUNDLE = os.path.join(_PARENT, 'agent_bundle_swarm')
sys.path.insert(0, _PARENT)   # чтобы работало `import agent_bundle_timeline_swarm`
sys.path.insert(0, _BUNDLE)   # сразу даём доступ к движку (shooting, swarm)

# Импорты движка для совместимости форматов
from shooting import Planet, Fleet
from agent_bundle_timeline_swarm.timeline_swarm import (
    timeline_swarm_plan, TimelineWeights,
)


# Минимальный shim для state, который ожидает timeline_swarm
class StubState:
    def __init__(self, planets, fleets, omega=0.01, step=0):
        self.planets = list(planets)
        self.raw_planets = list(planets)
        self.fleets = list(fleets)
        self.raw_fleets = list(fleets)
        self.omega = omega
        self.step = step
        self.n_players = 2
        self._initial_planets = {p.id: p for p in planets}
        self.comet_ids = set()
        self.comets = []

    def initial_by_id(self):
        return self._initial_planets


# BOARD=100, sun at (50,50) r=10 — все сценарии в правильном масштабе движка.

def make_scenario_accelerate():
    """Гипотеза концепции ускорения.

    Геометрия:
      P0 (10,15)  фронт, ships=100, prod=1.5
      P1 (50,30)  тыл,   ships=200, prod=1.5  ← БЛИЖЕ к P2, может ускорить
      P2 (90,15)  ЦЕЛЬ нейтрал, ships=15, prod=2.0
      P3 (10,90)  шум
      P4 (90,90)  противник

    Уже летящий флот P0→P2 (80 ships) — путь по y=15, не пересекает P1.
    distance(P0,P2)=80, speed(80)≈3.5 → ETA≈23 → P2 наша в t=33.
    distance(P1,P2)≈43, speed=lighter → ETA≈12 → можно прислать раньше.
    Ожидаем: opp accelerate, и P1 коммитит ships для ускорения.
    """
    planets = [
        Planet(0, 0,  10.0, 15.0, 3.0, 100.0, 1.5),
        Planet(1, 0,  50.0, 30.0, 3.0, 200.0, 1.5),
        Planet(2, -1, 90.0, 15.0, 2.0,  15.0, 2.0),
        Planet(3, 0,  10.0, 90.0, 3.0,  50.0, 1.0),
        Planet(4, 1,  90.0, 90.0, 3.0,  80.0, 1.5),
    ]
    import math as _m
    dx = planets[2].x - planets[0].x
    dy = planets[2].y - planets[0].y
    angle = _m.atan2(dy, dx)
    fx = planets[0].x + _m.cos(angle) * (planets[0].radius + 0.5)
    fy = planets[0].y + _m.sin(angle) * (planets[0].radius + 0.5)
    fleets = [Fleet(id=1000, owner=0, x=fx, y=fy, angle=angle,
                    from_planet_id=0, ships=80)]
    return StubState(planets, fleets, omega=0.0, step=10)


def make_scenario_pure_capture():
    """Простой захват: один фронтальный наш + два нейтрала."""
    planets = [
        Planet(0, 0,  10.0, 20.0, 3.0, 200.0, 1.5),
        Planet(1, -1, 30.0, 20.0, 2.0,  30.0, 1.5),
        Planet(2, -1, 10.0, 80.0, 2.0,  30.0, 1.0),
        Planet(3, 1,  90.0, 90.0, 3.0,  80.0, 1.5),
    ]
    return StubState(planets, [], omega=0.0, step=10)


def make_scenario_defend():
    """Defend: слабая фронтовая планета под ударом.

    P0 (20,20) наш, ships=10, prod=0.5 — слабый
    P1 (10,20) тыл, ships=200 — может спасти
    P2 (90,20) враг, ships=200, шлёт 100 ships → P0.

    Distance P2→P0 = 70, speed(100)≈3.7 → ETA≈19. К прибытию P0 имеет
    10+19*0.5=19 — враг побеждает (100 vs 19).
    """
    planets = [
        Planet(0, 0,  20.0, 20.0, 3.0,  10.0, 0.5),
        Planet(1, 0,  10.0, 20.0, 3.0, 200.0, 1.5),
        Planet(2, 1,  90.0, 20.0, 3.0, 200.0, 1.5),
    ]
    import math as _m
    dx = planets[0].x - planets[2].x
    dy = planets[0].y - planets[2].y
    angle = _m.atan2(dy, dx)
    fx = planets[2].x + _m.cos(angle) * (planets[2].radius + 0.5)
    fy = planets[2].y + _m.sin(angle) * (planets[2].radius + 0.5)
    fleets = [Fleet(id=2000, owner=1, x=fx, y=fy, angle=angle,
                    from_planet_id=2, ships=100)]
    return StubState(planets, fleets, omega=0.0, step=10)


def fmt_plans(plans):
    if not plans:
        return "  (нет планов)"
    lines = []
    for p in plans:
        lines.append(
            f"  {p['att_id']}→{p['tgt_id']}  ships={p['x_att']:>4}  "
            f"eta={p['t_total']:>5.1f}  {p.get('opp_type','?'):>12}  "
            f"tier={p.get('opp_tier','?')}  roi={p.get('roi', 0):.2f}"
        )
    return "\n".join(lines)


def print_debug(debug):
    print(f"  opps_total = {debug.get('n_opportunities', '?')}")
    print(f"  opps_by_type = {debug.get('opp_summary', {})}")
    print(f"  committed_by_type = {debug.get('committed_by_type', {})}")
    print(f"  budgets = {debug.get('budgets', {})}")
    auct = debug.get('auction_stats', {})
    print(f"  auction: considered={auct.get('considered')}, "
          f"committed={auct.get('committed')}, "
          f"rejected_low_roi={auct.get('rejected_low_roi')}, "
          f"tiers={auct.get('tiers_used')}")
    if debug.get('n_rolled_back'):
        print(f"  rolled_back = {debug['n_rolled_back']}")


def try_classic_swarm(state, player=0):
    """Запускаем классический swarm для сравнения. Если bundle не
    импортируется (битые зависимости в test-env) — возвращаем None."""
    try:
        from swarm import swarm_plan
        # Передаём только non-our и non-self в качестве "targets"
        targets = [p for p in state.raw_planets if p.owner != player]
        plans, dbg = swarm_plan(state, player, targets)
        return plans, dbg
    except Exception as e:
        return None, {'error': repr(e)}


def run_one(name, state):
    print(f"\n{'═' * 72}")
    print(f"СЦЕНАРИЙ: {name}")
    print('═' * 72)
    print(f"Планеты:")
    for p in state.raw_planets:
        print(f"  P{p.id} owner={p.owner:>2} ships={p.ships:>4.0f} "
              f"prod={p.production:.1f} ({p.x:.0f}, {p.y:.0f})")
    if state.raw_fleets:
        print(f"Летящие флоты:")
        for f in state.raw_fleets:
            print(f"  fleet{f.id} owner={f.owner} ships={f.ships} "
                  f"from=P{f.from_planet_id}")

    print("\n— Timeline Swarm —")
    plans, debug = timeline_swarm_plan(state, player=0,
                                        current_turn=state.step)
    print(fmt_plans(plans))
    print_debug(debug)

    print("\n— Classic Swarm (для сравнения) —")
    cplans, cdebug = try_classic_swarm(state, player=0)
    if cplans is None:
        print(f"  не запустился: {cdebug.get('error')}")
    else:
        if not cplans:
            print("  (нет планов)")
        else:
            for p in cplans:
                print(f"  {p.get('att_id')}→{p.get('tgt_id')}  "
                      f"ships={p.get('x_att','?')}  "
                      f"eta={p.get('t_total','?')}  "
                      f"mode={p.get('mode','?')}  "
                      f"margin={p.get('margin', 0):.1f}")


if __name__ == '__main__':
    run_one("PURE_CAPTURE — простой захват без флотов в полёте",
            make_scenario_pure_capture())
    run_one("ACCELERATE — гипотеза концепции",
            make_scenario_accelerate())
    run_one("DEFEND — нашу планету атакуют",
            make_scenario_defend())
