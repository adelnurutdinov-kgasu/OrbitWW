---
tags: [orbit-wars, features, planet, intrinsic]
---

# Per-planet фичи (intrinsic + derived)

Считаются для каждой [[10_atoms#Planet]] в `code:zones.py:_planet_zone_features`
+ `code:swarm.py` + `code:context.py`.

## Прямо из state

- `ships(t)` — гарнизон. Обновл: **tick**.
- `prod` — производительность. Обновл: **const**.
- `owner(t)` — принадлежность. Обновл: **event** (захват).
- `pos(t) = (x, y)` — текущая позиция (учитывает поворот для орбитальных).
- `radius` — размер диска. Используется в combat geometry.

## Net-force curve (см. `code:force.py`)

Для каждой планеты-цели строим график «как меняется наш net forces vs времени»:

- `force_events(state, target, horizon, ships_ref, player)`
  → список событий `(eta, src, sign, ships, prod)`.
  Каждое событие = «корабли из src долетают до target в момент eta».
  `sign = +1` если src наш, `-1` если чужой.
- `build_net_curve(events, target)` → `(xs, ys)` накопительная кривая.
  В t=0: `±target.ships`, далее каждое событие добавляет `sign·ships`.

Из кривой считаются:

- **`area_pos`, `area_neg`, `area`** — площадь под кривой (положительная/
  отрицательная/итого).
- **`area_inv` (`discounted_area_inv`)** — time-discounted: события у `t≈0`
  весят сильнее, поздние даунскейлятся. Семантика: «срочность ближнего
  будущего». См. формулу в `code:force.py:_POWER`.
- **`n_cross`** (`zero_crossings`) — сколько раз кривая пересекает 0.
  Высокая `n_cross` = горячая граница, фронт.

## Соседство / расстояния

`code:zones.py:_planet_zone_features` строки 165+.

- `mean_dist_all` — среднее расстояние до всех других планет (без комет).
- `mean_d_to_ours` — среднее до своих (диагностически).
- `wnn_close_res` — weighted nearest neighbor: `Σ sign(q) · w(q) / Σ w(q)`,
  где `w(q) = (ships + 5·prod) / dist`. Знак: +1=ours, −1=enemy, 0=neutral.
  Семантика: окружение взвешенное по близости и силе соседей.
  - `+1.0` — все близкие сильные планеты наши (deep rear).
  - `−1.0` — окружены богатыми чужими.
  - `~0` — нейтральный регион / фронт.

## Inferred-фичи (Game Understanding Layer)

См. [[50_gul_context]].

- `field_at[pid]` — Σ sign·ships/(dist²+softener²). Скаляр.
- `role_of[pid]` ∈ `{deep_rear, rear, frontline, forward, isolated}`.

## Late-aggression модификатор

`code:zones.py:_planet_zone_features` через `step`.

- `late_aggression = step / TOTAL_STEPS`. От 0 до 1.
- Используется как множитель веса `W_TARGETS['late_aggression']` в [[60_zones_priority]].

## Гэпы (TODO)

- **`time_to_lose(p)`** — через сколько ходов потеряем нашу `p` если не действовать.
  Сейчас есть `t_first_threat`, но это «когда появится угроза», не «когда сдадим».
- **`marginal_prod_gain(p)`** — насколько вырастет наш `Σ prod` если захватим `p`,
  с учётом likelihood удержания на оставшемся горизонте.
- **`reach_k(p)`** — сколько целей в ETA ≤ k ходов с этой планеты (connectivity).
  Связано с [[30_pairs#eta]].

#tag:планета #tag:derived
