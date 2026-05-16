#!/usr/bin/env python3
"""
build_submission.py — собирает single-file submission.py из agent_bundle_swarm 2.

Что делает:
  1. Читает все .py из "agent_bundle_swarm 2" в топологическом порядке.
  2. Убирает sibling-импорты (`from <module> import …`) — уже инлайнены.
  3. Убирает sys.path.insert/__file__ строки (kaggle exec без __file__).
  4. Инжектит _dbg namespace (agent.py использует `import agent_debug as _dbg`,
     который при инлайнинге убирается — восстанавливаем через SimpleNamespace).
  5. Sanity-check: compile() перед сохранением.

Запуск:  python3 tuning/scripts/build_submission.py
Вывод:   submissions/submission_swarm_<ts>.py
"""

import os, sys, re
from datetime import datetime

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
BUNDLE_DIR   = os.path.join(PROJECT_ROOT, "agent_bundle_swarm 2")
SUBM_DIR     = os.path.join(PROJECT_ROOT, "submissions")
os.makedirs(SUBM_DIR, exist_ok=True)

# ── Топологический порядок: листья → корень ───────────────────────────────
# Правило: модуль идёт ПОСЛЕ всех своих sibling-зависимостей.
#
#   shooting          — нет sibling-зависимостей
#   orbit_sim         — shooting
#   force             — orbit_sim, shooting
#   agent_debug       — нет sibling-зависимостей
#   projection        — shooting, orbit_sim
#   attacks           — orbit_sim, force, shooting
#   zones             — orbit_sim, force, shooting
#   swarm             — orbit_sim, force, attacks
#   context           — (только stdlib, dataclasses)
#   opponent_presets  — orbit_sim, force, shooting
#   opponent_model_bayesian — opponent_presets, swarm, shooting, projection
#   agent             — все выше
ORDER = [
    'shooting.py',
    'orbit_sim.py',
    'force.py',
    'agent_debug.py',
    'projection.py',
    'attacks.py',
    'zones.py',
    'swarm.py',
    'context.py',
    'opponent_presets.py',
    'opponent_model_bayesian.py',
    'agent.py',
]
# уникализируем сохраняя порядок
seen = set()
ORDER = [m for m in ORDER if not (m in seen or seen.add(m))]

# ── Sibling-модули — их импорты вырезаем (уже инлайнены) ─────────────────
SIBLINGS = {
    'orbit_sim', 'force', 'projection', 'shooting', 'zones',
    'attacks', 'swarm', 'agent_debug', 'context',
    'opponent_model_bayesian', 'opponent_presets',
}

# ── Функции из agent_debug.__all__ — нужны в _dbg namespace ──────────────
DBG_EXPORTS = [
    'enabled', 'reset', 'begin_turn',
    'log_zones', 'log_targets', 'log_plans', 'log_decision',
    'log_moves', 'log_fleets', 'log_error', 'end_turn',
    'plans_all_enabled', 'log_remaining', 'log_transfers',
    'log_zone_flips', 'log_bayes_confusion', 'reset_decisions',
    '_w', '_safe',
]


def _strip_sibling_imports(src: str) -> str:
    """Убирает sibling-импорты и __file__-зависимые строки."""
    lines = src.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # многострочный: from X import (\n    a, b\n)
        m = re.match(r'^from\s+(\w+)\s+import\s+\(', stripped)
        if m and m.group(1) in SIBLINGS:
            while i < len(lines) and ')' not in lines[i]:
                i += 1
            i += 1   # строка с ')'
            continue

        # однострочный: from X import A, B as C, D as E, …
        # Вместо удаления — эмитируем только alias-присвоения (C = B, E = D).
        # Не-aliased имена уже в globals из инлайненного модуля.
        m = re.match(r'^from\s+(\w+)\s+import\s+(.+)', stripped)
        if m and m.group(1) in SIBLINGS:
            indent = line[: len(line) - len(line.lstrip())]
            items_str = m.group(2).split('#')[0].strip()  # убираем inline-комментарий
            alias_lines = []
            for item in items_str.split(','):
                item = item.strip()
                # X as Y → нужно Y = X
                am = re.match(r'^(\w+)\s+as\s+(\w+)$', item)
                if am:
                    alias_lines.append(f"{indent}{am.group(2)} = {am.group(1)}\n")
            if alias_lines:
                out.extend(alias_lines)
            i += 1
            continue

        # import X  /  import X as Y  (top-level module import)
        m = re.match(r'^import\s+(\w+)(\s+as\s+(\w+))?\s*$', stripped)
        if m and m.group(1) in SIBLINGS:
            # `import agent_debug as _dbg` — _dbg восстанавливается namespace'ом в конце
            # `import sibling as alias` внутри try → пусть падает с ImportError (сafely caught)
            # На top-level просто пропускаем
            i += 1
            continue

        # __file__ использование — небезопасно в kaggle exec
        if '__file__' in line:
            i += 1
            continue

        # _HERE = ... (уже убрано выше через __file__)
        # if _HERE not in sys.path: → убираем if + тело целиком
        if re.match(r'^if\s+\w*_?HERE\w*\s+not\s+in\s+sys\.path', stripped):
            i += 1
            # пропускаем indented тело (sys.path.insert)
            while i < len(lines) and lines[i].startswith((' ', '\t')):
                i += 1
            continue

        # _sys.path.insert(0, _os.path...) — на верхнем уровне (без if)
        if re.search(r'_?sys\.path\.insert\s*\(', line) and not line.startswith((' ', '\t')):
            i += 1
            continue

        out.append(line)
        i += 1
    return ''.join(out)


def main():
    ts_iso = datetime.now().isoformat(timespec='seconds')
    parts = [
        f"# Auto-generated submission — {ts_iso}\n",
        f"# Source: agent_bundle_swarm 2\n",
        f"# Build: tuning/scripts/build_submission.py\n\n",
    ]

    included = []
    for fname in ORDER:
        path = os.path.join(BUNDLE_DIR, fname)
        if not os.path.exists(path):
            print(f"  ⚠  skip missing: {fname}")
            continue
        with open(path, encoding='utf-8') as f:
            src = f.read()
        stripped = _strip_sibling_imports(src)
        parts += [
            f"\n# ╔══════════════════════════════════════════════════╗\n",
            f"# ║  {fname:<46}║\n",
            f"# ╚══════════════════════════════════════════════════╝\n\n",
            stripped,
        ]
        included.append(fname)

    full = ''.join(parts)

    # ── if __name__ == "__main__" блоки → заглушка ────────────────────────
    # Kaggle exec'ит весь файл как скрипт — без заглушки они запустятся.
    full = re.sub(
        r'^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:',
        'if False:  # disabled in submission (kaggle exec)',
        full, flags=re.MULTILINE,
    )

    # ── agent(obs) → agent(obs, config=None) ─────────────────────────────
    if 'def agent(obs):' in full:
        full = full.replace('def agent(obs):', 'def agent(obs, config=None):')
        print("  ✓ patched: def agent(obs) → def agent(obs, config=None)")

    # ── _dbg namespace ────────────────────────────────────────────────────
    # `import agent_debug as _dbg` убирается стриппером, но agent.py вызывает
    # _dbg.enabled(), _dbg.log_error() и т.д.
    # agent_debug.py уже инлайнен → все функции в globals().
    # Добавляем _dbg = SimpleNamespace(...) в конец файла.
    # Python ищет _dbg в globals() lazily при вызове agent() — порядок ОК.
    #
    # Проверяем что строчка убрана (иначе дублирование не нужно)
    if 'import agent_debug as _dbg' not in full:
        dbg_assignments = '\n'.join(
            f"    {name}={name},"
            for name in DBG_EXPORTS
            if name not in ('_w', '_safe')   # приватные — проверяем наличие
        )
        dbg_private = '\n'.join(
            f"    **({{'_{n}': _{n}}} if '_{n}' in dir() else {{}})"
            for n in ('w', 'safe')
        )
        # Простой вариант: просто перечисляем все имена которые точно есть
        dbg_lines = []
        for name in DBG_EXPORTS:
            dbg_lines.append(f"    {name}={name},")

        full += "\n\n# ── _dbg namespace (agent_debug инлайнен, восстанавливаем ссылку) ──\n"
        full += "import types as _types_dbg\n"
        full += "_dbg = _types_dbg.SimpleNamespace(\n"
        full += '\n'.join(dbg_lines) + '\n'
        full += ")\n"
        print(f"  ✓ injected _dbg namespace ({len(DBG_EXPORTS)} symbols)")

    # ── Sanity: compile ───────────────────────────────────────────────────
    try:
        compile(full, '<submission>', 'exec')
        print("  ✓ compile() passed")
    except SyntaxError as e:
        print(f"  ❌ SYNTAX ERROR: {e}")
        bad_path = os.path.join(SUBM_DIR, "_BROKEN_submission.py")
        with open(bad_path, 'w', encoding='utf-8') as f:
            f.write(full)
        print(f"     сохранено для разбора: {bad_path}")
        sys.exit(1)

    # ── Сохраняем ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(SUBM_DIR, f"submission_swarm_{ts}.py")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(full)

    n_lines = full.count('\n')
    print(f"\n✓ собрано {len(included)}/{len(ORDER)} модулей → {out_path}")
    print(f"  размер: {n_lines} строк  ({len(full):,} байт)")
    print(f"  модули: {', '.join(included)}")

    if 'def agent(obs' in full:
        print("  ✓ entrypoint def agent(obs, …) присутствует")
    else:
        print("  ⚠ entrypoint def agent(...) НЕ НАЙДЕН — kaggle не запустит!")

    # Выводим активные веса (что будет в submission)
    print("\n  Активные веса в submission:")
    for line in full.split('\n'):
        if re.match(r'^SWARM_WEIGHTS\s*=|^SWARM_WEIGHTS_4P\s*=', line):
            print(f"    {line.strip()}")


if __name__ == "__main__":
    main()
