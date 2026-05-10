---
tags: [orbit-wars, features, global, gul]
---

# GUL — Game Understanding Layer

`code:context.py:compute_context(state, player)` → `GameContext`.

Глобальное «понимание игры», вычисляется один раз в начале хода. Сейчас
**только логируется** (`agent.py` строки после `compute_context`), в scoring
ещё не интегрирован. См. [[70_swarm_decisions]] для следующего шага.

## Поля GameContext

### Phase

```
phase_progress = 1 − n_neutrals_remaining / n_neutrals_initial   # 0..1
total_ships    = Σ ships (planets + fleets), оба игрока
```

`phase`:
- `early`  — phase_progress < 0.30
- `mid`    — 0.30 ≤ < 0.70
- `late`   — ≥ 0.70 и ещё есть нейтралы
- `endgame` — нейтралов нет

### Stance

```
prod_ratio  = our_prod / max(1, enemy_prod)
ships_ratio = our / (our + enemy)        # без нейтралов
our_com     = взвешенный центр масс наших (по ships)
enemy_com   = взвешенный центр масс чужих
com_separation = |our_com − enemy_com|
```

`stance` (приоритет: desperate > consolidation > expansion > attrition):
- `desperate`     — ships_ratio < 0.35
- `consolidation` — ships_ratio > 0.65 и нет нейтралов
- `expansion`     — нейтралов > 50% от initial
- `attrition`     — всё остальное

### Pressure field

```
field_at[pid] = Σ sign · ships / (dist² + SOFTEN²)      # SOFTEN=5
```

Знак `+` для своих, `−` для чужих, нейтралы исключены. Сила = ships
поделённое на квадрат расстояния (плюс смягчающий буфер от singularity).

`role_of[pid]`:
- `deep_rear`  (>+0.30)
- `rear`       (>+0.05)
- `frontline`  (между ±0.05)
- `forward`    (<−0.05)
- `isolated`   (<−0.30)

```
dominance_balance ∈ [−1.0, +1.0]
```

Сэмплируется 5×5 grid через всю карту, считаем `pos - neg`/`pos+neg`.
Семантика: **пространственное** доминирование.

### Opponent intent

```
enemy_internal_transfers = [(from, to)]     # флоты enemy→enemy
threatened_planets       = [pid, …]         # принимающие, готовятся стрелять
```

`opponent_intent` (определяется по флотам противника по углу с tol=0.3 рад):
- `passive`          — enemy_ships < 30 и нет флотов в полёте
- `preparing_attack` — есть `internal_transfers`
- `idle`             — нет флотов
- `aggressive`       — >50% флотов летят к нашим
- `expanding`        — флоты к нейтралам

## Используется (планируется)

В `_action_value` ([[70_swarm_decisions#action-value]]):
- `phase = early` → activity_weight ↓
- `stance = desperate` → risk_tolerance auto-2
- `field_at[att] < −0.30` → defensive penalty для атак из глубины врага
- `opponent_intent = preparing_attack` → defense priority для `threatened`

## Гэпы (TODO)

- **time_to_clash** — когда первое столкновение
- **goal_set** — топ-5 «решающих» планет
- **agent_health** — внутренние KPI планировщика (unfunded count, avg margin)
- **production_horizon** — оценка оставшегося времени для prod-капитализации
- **connectivity_pressure** — насколько новые захваты улучшают reach

#tag:глобал #tag:gul
