# Resolution Moments — эмпирический t* для 1v1 партий

## Метод

Определяем `t*` как самый ранний шаг где партия зашла в "safe zone"
по `own_prod_ratio` и оставалась там до конца. Safe zones найдены эмпирически:
диапазоны параметра где winrate ≤ 2% (точно проиграет) или ≥ 98% (точно выиграет).

## Найденные safe zones

| Параметр | safe_low (≤2% winrate) | safe_high (≥98% winrate) |
|----------|------------------------|--------------------------|
| prod_advantage | 0.16000000016000002 | 0.84000000084 |
| ship_advantage | 0.08000000008000001 | 0.9200000009200001 |
| planet_advantage | 0.26000000026000003 | 0.74000000074 |
| own_prod_ratio | None | 0.6000000006 |

## Статистика t*

- Партий-игроков с найденным t*: **462** / 462
- Без resolution в данных: 0
- Direction at t* совпадает с финальным исходом: 462 (100.0%)

### Распределение `phase_at_t_star`

```
count    462.000
mean       0.636
std        0.176
min        0.047
25%        0.565
50%        0.676
75%        0.751
max        0.906
```

### Распределение `window_size` (20% длины партии)

```
count    462.0
mean      34.4
std       14.8
min        4.0
25%       25.0
50%       31.0
75%       39.0
max      100.0
```

## Графики

- `resolution_safe_zones.png` — winrate vs own_prod_ratio с границами safe zones
- `resolution_distribution.png` — распределение phase_at_t_star и window_size

## Следующий шаг

`06_pre_resolution_geometry.py` — для каждой (episode, player) загрузить raw obs
для шагов в окне `[t_window_start, t_star]`, посчитать геометрические признаки
поля контроля (площадь, длина фронта, кривизна, race-pair flip count),
и сравнить их **изменения** между победителями и проигравшими.
