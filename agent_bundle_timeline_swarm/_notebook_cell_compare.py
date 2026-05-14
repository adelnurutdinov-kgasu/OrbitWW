# ═══════════════════════════════════════════════════════════════════════
# Сравнение Classic vs Timeline Swarm против sub2 (с рендерами)
# ═══════════════════════════════════════════════════════════════════════
# Эта ячейка:
#   1) грузит свежий экземпляр нашего агента (agent_bundle_swarm/agent.py),
#      прогоняет матч против sub2 — это baseline (Classic swarm);
#   2) загружает ВТОРОЙ свежий экземпляр того же агента, monkey-patch'ит
#      swarm_plan на timeline_swarm_plan_adapter, прогоняет матч — это Timeline;
#   3) показывает HTML-render обоих матчей для визуального сравнения действий.
#
# Семантика monkey-patch'a:
#   agent.py делает `from swarm import swarm_plan` — это копирует имя в
#   module-scope в момент импорта. После загрузки модуля переменная
#   our_mod.swarm_plan указывает на оригинальную функцию. Мы её перезаписываем
#   на свой adapter ПЕРЕД env.run(...), и agent.py начинает использовать наш.

import sys, os, importlib, importlib.util
from IPython.display import display, HTML, Markdown

HERE     = os.getcwd()
OUR_PATH = os.path.join(HERE, 'agent_bundle_swarm', 'agent.py')
SUB_PATH = os.path.join(HERE, 'sub2.py')

# Гарантируем что папка с timeline-bundle доступна
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TS_BUNDLE = os.path.join(HERE, 'agent_bundle_timeline_swarm')
if TS_BUNDLE not in sys.path:
    sys.path.insert(0, TS_BUNDLE)

# Сид для обоих матчей (одинаковый — sub2 даёт детерминированное поведение)
COMPARE_SEED = 0


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent, mod


def _run_match(use_timeline, seed):
    """Прогон одного матча. Если use_timeline=True — подменяет swarm_plan."""
    tag = 'tl' if use_timeline else 'cl'
    our_fn, our_mod = _load(OUR_PATH, f'cmp_{tag}_{seed}')
    sub_fn, _       = _load(SUB_PATH,  f'cmp_{tag}_sub_{seed}')

    try:
        our_mod._dbg.reset()
    except Exception:
        pass

    our_mod.USE_MCTS                = False
    our_mod.USE_OPPONENT_PREDICTION = False
    our_mod._opponent_model         = None
    our_mod._pending_partials       = None
    our_mod._bayes_model            = None
    our_mod._prev_fleet_ids         = None
    our_mod._prev_state_raw         = None

    if use_timeline:
        # Перезагружаем bundle на свежо (если внутри Jupyter уже подгружен)
        if 'agent_bundle_timeline_swarm' in sys.modules:
            importlib.reload(sys.modules['agent_bundle_timeline_swarm'])
        import agent_bundle_timeline_swarm as _ts
        importlib.reload(_ts)
        our_mod.swarm_plan = _ts.timeline_swarm_plan_adapter
        print(f'  [timeline] swarm_plan -> {our_mod.swarm_plan.__name__}')

    from kaggle_environments import make as _make
    env = _make('orbit_wars', debug=False, configuration={'seed': seed})
    env.run([our_fn, sub_fn])
    return env, our_mod


def _result_str(env):
    try:
        r0 = env.steps[-1][0]['reward']
        r1 = env.steps[-1][1]['reward']
        if r0 is None or r1 is None:
            return f'(reward None — игра прервана?  r0={r0} r1={r1})'
        if r0 > r1: return f'P0 (наш) WIN  — reward {r0} vs {r1}'
        if r0 < r1: return f'P0 (наш) LOSS — reward {r0} vs {r1}'
        return f'DRAW — reward {r0} vs {r1}'
    except Exception as e:
        return f'(не удалось прочесть исход: {e})'


def _planets_summary(env, label):
    """Грубая статистика по финальному состоянию."""
    try:
        last = env.steps[-1]
        obs0 = last[0].get('observation', {})
        planets = obs0.get('planets')
        if planets:
            cnt0 = sum(1 for p in planets if p.get('owner', -1) == 0)
            cnt1 = sum(1 for p in planets if p.get('owner', -1) == 1)
            cntN = sum(1 for p in planets if p.get('owner', -1) == -1)
            return (f'  {label:<18} planets:  ours={cnt0}  '
                    f'enemy={cnt1}  neutral={cntN}')
    except Exception:
        pass
    return f'  {label:<18} (нет деталей)'


# ── Прогон 1: Classic Swarm ───────────────────────────────────────────
display(Markdown(f'### Матч 1: **Classic Swarm** vs sub2 (seed={COMPARE_SEED})'))
env_classic, mod_classic = _run_match(use_timeline=False, seed=COMPARE_SEED)
print(_result_str(env_classic))
print(f'Шагов сыграно: {len(env_classic.steps)}')
display(HTML(env_classic.render(mode='html', width=600, height=600)))


# ── Прогон 2: Timeline Swarm ──────────────────────────────────────────
display(Markdown(f'### Матч 2: **Timeline Swarm** vs sub2 (seed={COMPARE_SEED})'))
env_timeline, mod_timeline = _run_match(use_timeline=True, seed=COMPARE_SEED)
print(_result_str(env_timeline))
print(f'Шагов сыграно: {len(env_timeline.steps)}')
display(HTML(env_timeline.render(mode='html', width=600, height=600)))


# ── Side-by-side summary ──────────────────────────────────────────────
display(Markdown('### Итоговая статистика'))
print(_planets_summary(env_classic,  'Classic Swarm:'))
print(_planets_summary(env_timeline, 'Timeline Swarm:'))
