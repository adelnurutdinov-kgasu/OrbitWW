---
tags: [orbit-wars, features, index]
---

# Orbit Wars — feature catalog

Структурированный каталог фич, связей и моделей принятия решений нашего
swarm-агента. Источник правды — код в `agent_bundle_swarm/`.

## Навигация

- [[10_atoms]] — атомарные сущности: Planet, Fleet, Sun, Time, Player.
- [[20_intrinsic]] — фичи планеты прямо из state (ships, prod, owner, …).
- [[30_pairs]] — фичи пары: distance, ETA, sun-block, угол.
- [[40_action_plans]] — типы действий: `direct`, `multi_sync`, `pipe`,
  `transfer`. Уравнения, cost, success-условия.
- [[50_gul_context]] — глобальный контекст (Game Understanding Layer):
  phase, stance, pressure field, opponent intent.
- [[60_zones_priority]] — зонирование планет и формула приоритета.
- [[70_swarm_decisions]] — auction, redistribute, action_value формула,
  SwarmWeights.
- [[99_dependencies]] — Mermaid-граф связей между фичами.

## Соглашения

- `[[ИмяФайла#Раздел]]` — кросс-ссылка (Obsidian).
- `#tag:планета` `#tag:пара` `#tag:глобал` `#tag:временное` — категории фич.
- Формулы дают **только семантику**. Точную реализацию см. в коде по ссылке
  `code:модуль.py:строка`.
- Если фича упоминается без обратной ссылки на `code:` — её **нет в коде**,
  это **идея/гэп** (TODO).

## Текущий состав агента

Стек:
1. Проекция (resolve летящих флотов) → `code:projection.py`
2. Зонирование → `code:zones.py` (см. [[60_zones_priority]])
3. Game Understanding Layer → `code:context.py` (см. [[50_gul_context]])
4. Defense plans → `code:agent.py:_build_defense_plans`
5. Swarm planning (auction) → `code:swarm.py` (см. [[70_swarm_decisions]])
6. Атомарное исполнение → `code:agent.py:_execute_plan_atomically`

## Что в каталоге **не** покрыто (намеренно)

- Геометрия движения, sun-collision, orbital lead (см. `code:shooting.py`).
- Реализация прицеливания `aim_hybrid` (см. `code:shooting.py`).
- Симулятор `simulate_step` (см. `code:orbit_sim.py`).

Эти слои — «физика», они стабильны и не участвуют в стратегических решениях.
