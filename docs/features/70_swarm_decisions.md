---
tags: [orbit-wars, features, swarm, decisions]
---

# Swarm decisions — аукцион и веса

`code:swarm.py:swarm_plan` — главная точка входа.

## Pipeline

```
1. candidates = all_plans(state, target, ours, risk)  для каждого priority target
2. stress[p]  = compute_stress(candidates, weights)
3. neigh[p]   = neighbor_stress(stress, ours)
4. auction(candidates) → committed + remaining + unfunded
5. (optional) redistribute → transfer plans
6. return committed + transfers
```

## Auction

`code:swarm.py:auction`

Single-pass greedy:
1. Сортируем `candidates` по `(success desc, action_value desc, stable_idx)`.
2. Walk: для каждого — у всех акторов хватит ships? target ещё не захвачен?
3. Yes → commit, вычитаем cost из бюджета каждого. No → в `unfunded`.

```
remaining[p.id] = ships(p) − RESERVE_ON_ATT     # стартовый бюджет
ships_lookup[p.id] = ships(p)                   # для action_value
```

## Action value

`code:swarm.py:_action_value(plan, weights, priority_lookup, ships_lookup)`

```
value = margin
      + eta_bonus / eta^(1 − distance_comfort)         # tiebreaker по дистанции
      + priority_bonus · priority_lookup[tgt]           # из zones
      + ships_weight · log1p(max actor_ships)           # «мяч на её стороне»
      + activity_weight · max(0, max_actor − idle_floor)  # idle-bonus
```

## SwarmWeights (параметры)

`code:swarm.py:SwarmWeights` (dataclass — точка тюнинга):

```python
stress_top_k:        int   = 3       # top-K действий в stress
stress_gamma:        float = 0.6     # дисконт по rank: γ^i
eta_bonus:           float = 30.0    # tiebreaker
priority_bonus:      float = 1.0     # вес priority (zones) в value
ships_weight:        float = 4.0     # log1p(ships actor)
activity_weight:     float = 0.5     # × idle_max
idle_floor:          int   = 25      # порог idle
distance_comfort:    float = 0.0     # 0=1/eta, 1=плоско
risk_tolerance:      int   = 0       # снижает overkill (0..1)

# Transfer (по умолчанию OFF)
enable_redistribute: bool  = False
transfer_horizon:    float = 40.0
transfer_floor:      float = 20.0
transfer_thresh:     float = 5.0
transfer_eta_pen:    float = 0.05
transfer_buffer:     float = 1.0
transfer_min_ships:  int   = 15
```

Лучший single найден через reverse_tournament: **winner_seed20** —
`activity=1.29 idle=40 comfort=0.18 risk=2 ships_w=1.37 eta_b=16.66 prio_b=2.84 K=3 γ=1.5`.

## Stress

`code:swarm.py:compute_stress`

```
top_actions[p] = sorted([action.margin for action in candidates if p in actors], desc)
stress[p] = Σ_i top_actions[p][i] · γ^i      # i ∈ [0, K)
```

Интерпретация: сколько боевой ценности доступно с этой планеты прямо сейчас.

`neighbor_stress(stress)` = WNN-weighted: `Σ (1/dist) · stress[Q] / Σ (1/dist)`.

Сейчас используется только для:
1. Скоринга TRANSFER в `redistribute` (если `enable_redistribute=True`).
2. Логирования.

## Target selection (агент уровень)

`code:agent.py` строки 425-455:

```python
ratio = our_ships / total          # из raw state (не projection)
top_n, prio_floor = _adaptive_attack_params(ratio)      # ADAPTIVE_TIERS
targets = df[df.zone ∈ ('easy_target', 'priority_target') AND priority >= prio_floor]
        .sort('priority', desc)
```

`ADAPTIVE_TIERS`:

| ratio_lo | top_n | prio_floor | label    |
|---------:|------:|-----------:|----------|
|     0.00 |     3 |       +0.5 | behind   |
|     0.25 |     6 |       −0.5 | parity   |
|     0.45 |    10 |       −1.5 | ahead    |
|     0.65 |    16 |       −3.0 | dominate |

## Atomic execution

`code:agent.py:_execute_plan_atomically`

Каждый plan исполняется ЦЕЛИКОМ или отбрасывается:
1. Резервирование ships у всех акторов (без бамп до MIN_FIRE).
2. `aim_hybrid` + до 6 углов-кандидатов `± epsilon`.
3. `simulate_launch` — должен дать `outcome='HIT'` и `planet_id==tgt`.
4. Для direct: `defender_actual` проверяется через `simulate_eta` → если
   `n_actual <= defender` — план SKIP.

Если ВСЕ части прошли → коммит в `committed` (общий бюджет всего хода).

## Гэпы (TODO)

- **Top-K MoE / ensemble** — top-2 в нашем eval даёт +12 п.п. (см. router_v2).
- **Look-ahead** — симуляция первых 10 ходов через `local_match` для выбора
  weights/stance в начале матча.
- **GUL-integrated value** — пока [[50_gul_context]] только логируется, не модифицирует `_action_value`.

#tag:аукцион #tag:swarm #tag:параметры
