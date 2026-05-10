---
tags: [orbit-wars, features, atoms]
---

# Атомарные сущности

Базовые объекты игры. Из них собирается всё.

## Planet

`code:orbit_sim.py:Planet`

Поля (порядок в кортеже obs):
- `id` — int, уникальный
- `owner` — int: `0`=мы, `1`=враг, `-1`=нейтрал
- `x, y` — координаты центра в момент `t`
- `radius` — размер диска (`_prod_to_radius(prod)`)
- `ships` — текущий гарнизон
- `production` — прирост ships/ход (только если `owner ≥ 0`!)

Свойства:
- Орбитальные ↔ статичные: `is_orbital(x, y, radius)` (`code:shooting.py`).
- Орбитальные крутятся со скоростью `state.omega` вокруг центра поля.
- Нейтралы НЕ производят корабли (`_effective_prod` возвращает 0).

См. [[20_intrinsic]] для derived-фич.

## Fleet

`code:orbit_sim.py:Fleet`

Поля: `id, owner, x, y, angle, from_pid, ships`.

- Движется по прямой со скоростью `_fleet_speed(ships)` (быстрее → меньше ships).
- Прибывает на планету когда отрезок шага задевает её диск.
- Может погибнуть: за границей поля или через солнце (`SUN_R = 10`).

Связь с атаками: каждый запуск порождает Fleet, который потом резолвится
в [[40_action_plans#combat-resolution]].

## Sun

Один объект в центре поля (`CX=50, CY=50, SUN_R=10`).

Влияет на:
- **Блок траектории**: `segment_hits_sun(a.x, a.y, b.x, b.y, safety=SUN_SAFETY)`.
  `SUN_SAFETY=1.5` — буфер вокруг диска.
- Не учитывается как «гравитационное» поле — только preventive geometry.

См. [[30_pairs#sun-block]].

## Time

- `state.step` — текущий ход (0-индекс).
- `TOTAL_STEPS = 500` (kaggle default).
- `state.omega ∈ [0.025, 0.05]` — угловая скорость орбит (фиксирована per-game).

См. [[50_gul_context#phase]].

## Player

- `player_id ∈ {0, 1}` — наш и противник.
- `NEUTRAL = -1` — особое значение для нейтралов.

#tag:планета #tag:флот #tag:время
