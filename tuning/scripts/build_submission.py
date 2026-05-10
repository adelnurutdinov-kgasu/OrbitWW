#!/usr/bin/env python3
"""
build_submission.py — собирает single-file submission.py из agent_bundle_swarm.

Что делает:
  1. Читает все .py из agent_bundle_swarm в правильном топологическом порядке.
  2. Убирает sibling-импорты (`from <module> import …`) — они уже инлайнены.
  3. Конкатенирует в один файл.
  4. В конце инжектит SWARM_WEIGHTS = winner_seed20 (зашитые best-веса).
  5. Выходной файл submission_swarm_<ts>.py пригоден для kaggle submission.

Запуск:  python3 tuning/scripts/build_submission.py
"""

import os, sys, re
from datetime import datetime

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
BUNDLE_DIR   = os.path.join(PROJECT_ROOT, "agent_bundle_swarm")
SUBM_DIR     = os.path.join(PROJECT_ROOT, "submissions")
os.makedirs(SUBM_DIR, exist_ok=True)

# Топологический порядок: листья → корень.
ORDER = [
    'shooting.py',     # ничего не зависит от sibling'ов
    'orbit_sim.py',    # depends on shooting
    'force.py',        # depends on orbit_sim
    'agent_debug.py',  # leaf
    'projection.py',   # depends on shooting
    'attacks.py',      # depends on orbit_sim, force, shooting
    'zones.py',        # depends on orbit_sim, force, shooting
    'swarm.py',        # depends on orbit_sim, force, attacks
    'context.py',      # GUL — без sibling-deps (только math, dataclasses)
    'agent.py',        # depends on все выше + context
]
# уникализируем сохраняя порядок
seen = set()
ORDER = [m for m in ORDER if not (m in seen or seen.add(m))]

# Sibling-modules чьи импорты надо вырезать
SIBLINGS = {'orbit_sim', 'force', 'projection', 'shooting', 'zones',
            'attacks', 'swarm', 'agent_debug', 'context'}


def _strip_sibling_imports(src):
    """Убирает `from <sibling> import …`, `import <sibling>`, и любые
    строки использующие `__file__` (kaggle exec'ит submission без __file__).
    """
    lines = src.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # многострочный from X import (a, b, c)
        m = re.match(r'^from\s+(\w+)\s+import\s+\(', stripped)
        if m and m.group(1) in SIBLINGS:
            while i < len(lines) and ')' not in lines[i]:
                i += 1
            i += 1
            continue
        # однострочный from X import …
        m = re.match(r'^from\s+(\w+)\s+import\s+', stripped)
        if m and m.group(1) in SIBLINGS:
            i += 1
            continue
        # import X / import X as Y
        m = re.match(r'^import\s+(\w+)(\s+as\s+\w+)?\s*$', stripped)
        if m and m.group(1) in SIBLINGS:
            i += 1
            continue
        # __file__-зависимые строки (kaggle exec без __file__)
        if '__file__' in line:
            i += 1
            continue
        # _HERE / sys.path.insert вокруг __file__ — соседние строки тоже мусор
        if re.match(r'^_HERE\s*=', stripped):
            i += 1
            continue
        if re.match(r'^if\s+_HERE\s+not\s+in\s+sys\.path', stripped):
            # пропускаем if + следующую строку (sys.path.insert)
            i += 1
            if i < len(lines) and 'sys.path.insert' in lines[i]:
                i += 1
            continue
        if 'sys.path.insert(0, _HERE)' in line or 'sys.path.insert(0, _os.path' in line:
            i += 1
            continue
        out.append(line)
        i += 1
    return ''.join(out)


# Best-веса — вписываются в финальную SWARM_WEIGHTS
WINNER_SEED20 = dict(
    activity_weight=1.29, idle_floor=40, distance_comfort=0.18,
    risk_tolerance=2, ships_weight=1.37, eta_bonus=16.66,
    priority_bonus=2.84, stress_top_k=3, stress_gamma=1.50,
)


def main():
    parts = []
    parts.append(f"# Auto-generated submission — {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"# Source: agent_bundle_swarm/ + winner_seed20 веса\n")
    parts.append(f"# Build: tuning/scripts/build_submission.py\n\n")

    for fname in ORDER:
        path = os.path.join(BUNDLE_DIR, fname)
        if not os.path.exists(path):
            print(f"⚠ skip missing: {fname}")
            continue
        with open(path) as f:
            src = f.read()
        stripped = _strip_sibling_imports(src)
        parts.append(f"\n# ╔══════════════════════════════════════════════════╗\n")
        parts.append(f"# ║  {fname:<46}║\n")
        parts.append(f"# ╚══════════════════════════════════════════════════╝\n\n")
        parts.append(stripped)

    # В самом конце — переопределение SWARM_WEIGHTS на winner_seed20
    parts.append(f"\n\n# ── BEST WEIGHTS из reverse_tournament ──────────────\n")
    parts.append(f"# winner_seed20: WR=0.46 на 100 сидах vs default 0.38\n")
    parts.append("SWARM_WEIGHTS = SwarmWeights(\n")
    for k, v in WINNER_SEED20.items():
        parts.append(f"    {k}={v!r},\n")
    parts.append(")\n")

    full = ''.join(parts)

    # `if __name__ == "__main__":` блоки в модулях запустятся при kaggle-exec,
    # потратят время на init и засорят stdout (что ломает kaggle pipe).
    full = re.sub(
        r'^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:',
        'if False:  # disabled: kaggle execs the whole submission',
        full, flags=re.MULTILINE,
    )

    # Kaggle вызывает agent(obs, config=None). Если в исходнике agent принимает
    # только obs — добавим второй параметр через автозамену сигнатуры.
    if 'def agent(obs):' in full:
        full = full.replace('def agent(obs):', 'def agent(obs, config=None):')
        print("  ✓ patched: def agent(obs) → def agent(obs, config=None)")

    # `import agent_debug as _dbg` был выкошен, а `_dbg.foo()` остались.
    # Соберём _dbg как namespace со всеми top-level def / переменными
    # из inlined agent_debug.py.
    debug_path = os.path.join(BUNDLE_DIR, 'agent_debug.py')
    if os.path.exists(debug_path):
        with open(debug_path) as f:
            dbg_src = f.read()
        # имена top-level def
        dbg_names = re.findall(r'^def\s+(\w+)\s*\(', dbg_src, flags=re.MULTILINE)
        # плюс переменные верхнего уровня которые могут понадобиться
        full += "\n# ── _dbg namespace для совместимости с agent.py ──\n"
        full += "import types as _types_for_dbg\n"
        full += "_dbg = _types_for_dbg.SimpleNamespace()\n"
        for name in dbg_names:
            full += f"if '{name}' in globals(): setattr(_dbg, '{name}', globals()['{name}'])\n"
        print(f"  ✓ injected _dbg namespace ({len(dbg_names)} symbols)")

    # Sanity: компилируется ли
    try:
        compile(full, '<submission>', 'exec')
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR в собранном submission: {e}")
        # сохраним для дебага
        bad_path = os.path.join(SUBM_DIR, "_BROKEN_submission.py")
        open(bad_path, 'w').write(full)
        print(f"saved broken для разбора: {bad_path}")
        sys.exit(1)

    # Сохраняем
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(SUBM_DIR, f"submission_swarm_{ts}.py")
    with open(out_path, 'w') as f:
        f.write(full)

    n_lines = full.count('\n')
    print(f"✓ собрано {len(ORDER)} модулей → {out_path}")
    print(f"  размер: {n_lines} строк, {len(full)} байт")
    print(f"  weights: winner_seed20 (WR=0.46)")

    # Проверим что есть def agent(obs, …)
    if 'def agent(obs' in full:
        print(f"  ✓ entrypoint def agent(obs, …) присутствует")
    else:
        print(f"  ⚠ entrypoint def agent(...) НЕ НАЙДЕН — kaggle не запустит")


if __name__ == "__main__":
    main()
