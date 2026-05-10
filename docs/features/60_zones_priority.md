---
tags: [orbit-wars, features, zones, priority]
---

# Zones и Priority

`code:zones.py:compute_zones_from_state(state, player, step)` →
DataFrame `(pid, owner, zone, priority, …)`.

## Что считаем

Для каждой планеты собираем feature-vector из [[20_intrinsic]]:

```
ZONE_FEATURES = [
  'area_inv',         # срочность ближнего будущего
  'wnn_close_res',    # окружение (взвеш. соседи по ships+5·prod / dist)
  'mean_dist_all',    # геогр. изоляция
  'prod',             # производительность
  'ships',            # текущая масса
  'n_cross',          # волатильность фронта
  'late_aggression',  # = step/TOTAL_STEPS, для эскалации в конце
]
```

Все 7 фич **z-нормируются** внутри игры (mean=0, std=1).

## Priority

```
priority(p) = Σ_i  W[i] · z(feature_i)
```

Два разных набора весов (для своих vs целей), потому что семантика обратная:

### W_OURS — для НАШИХ планет (защита)

```python
W_OURS = {
    'area_inv':       −0.1,
    'wnn_close_res':  −0.4,    # высокий = тыл (priority ниже)
    'mean_dist_all':  −0.3,
    'prod':           +0.4,
    'ships':           0.0,
    'n_cross':        +0.9,    # горячая зона → priority выше
    'late_aggression':−0.2,    # для своих не нужен late mode
}
```

Интерпретация: «высокий priority» = «нужно следить, может быть атакована».

### W_TARGETS — для НЕ-наших (захват)

```python
W_TARGETS = {
    'area_inv':       +0.4,    # наше преимущество там → priority выше
    'wnn_close_res':  +0.8,    # окружена нашими → дешёво взять
    'mean_dist_all':  −0.2,
    'prod':           +0.5,    # ценная цель
    'ships':          −0.5,    # много гарнизон → штраф
    'n_cross':        +0.4,
    'late_aggression':+0.9,    # эффективный вес ≈ phase·0.6
}
```

В позднюю фазу `late_aggression` множитель `≈ step/TOTAL_STEPS · 0.9` поднимает
priority **всех** не-наших → больше атак ближе к концу.

## Zones (категориальная метка)

`code:zones.py:_label_planet`

На z-нормированных порогах `THR_HI=0.5, THR_LO=−0.5`:

**Для своих** (приоритет в порядке):
- `frontline` — `threat=True` или `in_them=True`
- `contested` — `n_cross` высокий
- `bastion`   — богатая (rich) и в `wnn_close_res > 0.5`
- `rear`      — `wnn_close_res > 0.5`
- `isolated`  — `mean_dist_all > 0.5`
- `mid`       — всё остальное

**Для не-наших** (целей):
- `easy_target`     — `in_us=True` и `weak=True` (ships низкий)
- `priority_target` — rich и не far
- `hard_far`        — far и не weak
- `contested`       — `n_cross` высокий
- `periphery`       — всё остальное

В [[70_swarm_decisions#target-selection]] фильтруется по zones и priority_floor.

## Где это используется

- `agent.py` строки 425-455: фильтрация targets для swarm_plan по `TARGET_ZONES = ('easy_target', 'priority_target')` и `priority >= prio_floor`.
- `priority_lookup` передаётся в `swarm_plan` для tiebreaker'а в [[70_swarm_decisions#action-value]].

#tag:зонирование #tag:priority
