# Value Function V2 — отчёт
## Изменения vs V1
1. Убраны action features (9 шт.) из STATE — устраняет endogeneity.
2. Убрана `phase` — устраняет leak через концовку.
3. Добавлена стратификация по `phase` quartiles и `n_players`.
4. Добавлены тривиальные бейзлайны для измерения lift.

## Метрики overall
| Model | AUC | Brier | Acc | n |
|-------|-----|-------|-----|---|
| constant winrate | — | 0.213 | 0.6923 | 54184 |
| linear (5 macro) | 0.9039 | 0.1111 | 0.8368 | 54184 |
| xgb (5 macro)    | 0.9054 | 0.1096 | 0.8428 | 54184 |
| **value_v2**     | **0.9063** | **0.1091** | **0.8436** | 54184 |

**Lift over xgb_5macro: +0.0009 AUC**

## По фазам игры
| Phase quartile | n | AUC | Acc | pos_rate |
|---|---|---|---|---|
| q1_early | 13,616 | 0.6931 | 0.7168 | 0.3078 |
| q2_mid | 13,530 | 0.85 | 0.7933 | 0.3075 |
| q3_late_mid | 13,460 | 0.9561 | 0.8982 | 0.3075 |
| q4_late | 13,578 | 0.9919 | 0.9667 | 0.3079 |

## По n_players
| n_players | n | AUC | Acc | pos_rate |
|---|---|---|---|---|
| 2 | 12,504 | 0.9128 | 0.8083 | 0.5 |
| 4 | 41,680 | 0.8987 | 0.8542 | 0.25 |

## Топ фич
```
prod_gap                 0.4602
own_prod_ratio           0.1852
ship_gap                 0.0368
own_prod_total           0.0285
own_planet_ratio         0.0264
own_ship_ratio           0.0175
min_own_ships            0.0169
enemy_fleets_count       0.0142
proj_own_ship_ratio      0.0139
closest_threat_dist      0.0132
proj_own_planet_ratio    0.0127
incoming_enemy_to_own    0.0115
own_ships_total          0.0114
n_own_planets            0.0109
enemy_fleets_ships       0.0106
```

## Интерпретация
Эта модель честно отвечает на вопрос "из этого state кто победит?",
а не "это победитель действует или проигравший?". Сравнение с
linear/xgb_macro показывает, сколько даёт нелинейность и расширение
фич за пределы базовых ratio. Стратификация по phase покажет, где
игра ещё контестная (низкий AUC) vs где уже решена (высокий AUC).
