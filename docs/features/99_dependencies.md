---
tags: [orbit-wars, features, dependencies, diagram]
---

# Граф зависимостей фич

Кликабельные ссылки на разделы (Obsidian: hover → preview).

## Поток данных в одном ходе

```mermaid
graph TD
  obs[obs from kaggle]
  state_raw[state_raw = GameState.from_kaggle_obs]
  state_proj[state = project_state]
  
  obs --> state_raw
  state_raw --> state_proj

  state_proj --> zones[compute_zones_from_state]
  state_proj --> gul[compute_context GUL]
  state_proj --> classify[_classify_our_planets]
  state_proj --> defense[_build_defense_plans]

  classify --> defense
  zones --> targets[target selection by zone+priority+floor]
  zones --> priority_lookup
  
  targets --> swarm[swarm_plan]
  priority_lookup --> swarm
  state_raw --> swarm

  swarm --> auction[auction]
  swarm --> redistribute
  auction --> committed_plans
  redistribute --> transfer_plans
  
  defense --> all_plans_final[defense + attacks + transfers]
  committed_plans --> all_plans_final
  transfer_plans --> all_plans_final

  all_plans_final --> exec[_execute_plan_atomically]
  exec --> moves[final moves]
```

## Феатурные цепочки

```mermaid
graph LR
  ships --> field_at
  prod --> field_at
  owner --> field_at
  
  pos_t --> dist_t
  dist_t --> eta
  omega --> eta
  
  eta --> margin
  ships --> margin
  prod_target --> margin
  
  margin --> action_value
  eta --> action_value
  ships --> action_value
  priority_zones --> action_value
  
  field_at --> role
  field_at --> dominance_balance
  role --> action_modifier_TODO[action_modifier — TODO]
  
  ships_ratio --> stance
  prod_ratio --> stance
  com_separation --> stance
  
  phase_progress --> phase
  total_ships --> phase

  area_inv --> priority_zones
  wnn_close_res --> priority_zones
  n_cross --> priority_zones
  prod --> priority_zones
  ships --> priority_zones
  mean_dist_all --> priority_zones
  late_aggression --> priority_zones
  
  enemy_fleets --> opponent_intent
  internal_transfers --> opponent_intent
  threatened_planets --> opponent_intent
```

## Ключевые точки расширения (где интегрировать новое)

- `_action_value` ([[70_swarm_decisions#action-value]]) — самая частая точка
  расширения. Любая новая фича → может стать модификатором.
- `compute_zones_from_state` ([[60_zones_priority]]) — meta-priority, влияет
  на target selection.
- `compute_context` ([[50_gul_context]]) — глобальная сводка, потенциал для
  условных весов.

## Логика обновления каталога

Когда меняем код:
- Новая константа в `swarm.py` → [[70_swarm_decisions#swarmweights]].
- Новая метрика в `zones.py` → [[60_zones_priority#что-считаем]].
- Новое поле в `GameContext` → [[50_gul_context]].
- Новый тип action → [[40_action_plans]].
- Новая физическая фича (sun, angle, ETA) → [[30_pairs]].
