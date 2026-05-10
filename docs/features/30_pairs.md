---
tags: [orbit-wars, features, pair]
---

# Фичи пары (planet × planet, planet × fleet)

Базовые геометрические/временные характеристики между двумя сущностями.

## Distance

- `dist(i, j) = √(Δx² + Δy²)` — текущее расстояние.
- Без учёта движения орбит / упреждения — это снимок «сейчас».

## ETA (rendezvous time)

`code:force.py:_rendezvous_eta(src, dst, ships, omega, n_iter=3)`

Время от запуска до прибытия с учётом **орбитального упреждения** —
fixpoint итерация:
1. t₀ = dist(src.now, dst.now) / fleet_speed(ships).
2. dst_predicted = predict_planet_xy(dst, omega, t₀).
3. t₁ = dist(src, dst_predicted) / fleet_speed(ships).
4. Повторить 3 раза.

Параметры:
- `ships` — влияет на `_fleet_speed(ships)` (больше ships → медленнее).
- `omega` — угловая скорость поля.
- `n_iter=3` — обычно сходится за 2.

Используется ВЕЗДЕ где нужно «когда что куда долетит». См. [[40_action_plans]].

## Sun-block

`code:attacks.py:_seg_blocked(a, b, safety=1.5)` → `segment_hits_sun(a.x, a.y, b.x, b.y, 1.5)`.

True если **прямой отрезок** между центрами пересекает солнце с буфером 1.5.

Применяется:
- `eval_direct` — путь att→tgt должен быть чист.
- `eval_pipe` — оба сегмента (sup→att, att→tgt) должны быть чисты.
- `eval_multi_sync` — оба пути (sup→tgt, att→tgt).
- В [[40_action_plans#pipeline]] — exception: если sup→tgt блокирован, relay
  оставляем как единственный путь.

## Pipeline angle (триплетная)

`code:attacks.py:_angle_at_att(sup, att, tgt)`

Угол между приходящей траекторией `(att − sup)` и исходящей `(tgt − att)`.
- 0° = supplier строго «за спиной» атакера, идеальная прямая.
- 90° = L-образный крюк.
- 180° = U-turn (supplier и target по одну сторону).

`PIPELINE_MAX_ANGLE_DEG = 120` — выше → relay делает крюк, supplier лучше
стреляет независимо. См. [[40_action_plans#pipeline]].

## Incoming fleets (planet × fleet)

`code:attacks.py:friendly_incoming(state, tgt, t_max, player)`

Для каждого летящего флота `f`:
- угол `(tgt − f) ≈ f.angle` с допуском `INCOMING_ANGLE_TOL = 0.18` рад.
- если совпадает → флот «целится в tgt».
- сумма `f.ships` и список ETA каждого.

Используется для учёта **уже летящих наших** при расчёте `needed` для атаки.

## Гэпы (TODO)

- **`reach_k_neighbors(p, k)`** — список планет в ETA ≤ k. Для оценки [[20_intrinsic#connectivity]].
- **`time_from_to(i, j, ships)`** — кешированный ETA на тике (сейчас пересчитывается O(n²)).
- **`opponent_fleet_eta_to(p)`** — когда враг прилетит к нашей `p`.
  Часть [[50_gul_context#opponent-intent]].

#tag:пара #tag:геометрия #tag:время
