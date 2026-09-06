# Стратифицированный анализ delta-features по типам карт

**Данные**: 462 партий, 491 карт

## Размеры страт (size × orbital)

```
map_type
large_orbital      61
large_static       13
medium_orbital    151
medium_static     102
small_orbital      38
small_static      126
```

## Cohen's d общий vs максимальный внутри страты

Если max_in_strata > overall — стратификация выявляет более резкий сигнал.

| feature | overall d | max |d| в страте | усиление |
|---|---|---|---|
| delta_own_prod_total | 1.78 | 2.678 | 0.898 |
| delta_incoming_own_reinforce | 0.455 | 1.288 | 0.833 |
| delta_own_prod_ratio | 1.881 | 2.71 | 0.829 |
| delta_ship_gap | 1.328 | 2.111 | 0.783 |
| delta_n_own_planets | 1.571 | 2.3 | 0.729 |
| delta_own_planet_ratio | 1.751 | 2.419 | 0.668 |
| delta_net_incoming_own | 0.426 | 1.079 | 0.653 |
| delta_own_fleets_ships | 0.374 | 0.991 | 0.617 |
| delta_fleet_balance | 0.501 | 1.095 | 0.594 |
| delta_own_ships_total | 0.799 | 1.393 | 0.594 |
| delta_own_fleets_count | 0.393 | 0.928 | 0.535 |
| delta_own_ship_ratio | 1.37 | 1.866 | 0.496 |
| delta_n_enemy_planets | -1.533 | 1.99 | 0.457 |
| delta_enemy_fleets_ships | -0.366 | 0.801 | 0.435 |
| delta_enemy_fleets_count | -0.373 | 0.804 | 0.431 |
| delta_max_own_ships | 0.414 | 0.753 | 0.339 |
| delta_neutral_contested_count | 0.038 | 0.339 | 0.301 |
| delta_incoming_enemy_to_own | -0.177 | 0.435 | 0.258 |
| delta_std_own_ships | 0.307 | 0.562 | 0.255 |
| delta_mean_own_ships | 0.191 | 0.433 | 0.242 |
| delta_own_under_threat_ships | -0.136 | 0.322 | 0.186 |
| delta_prod_gap | 3.133 | 3.297 | 0.164 |
| delta_closest_threat_dist | -0.008 | 0.142 | 0.134 |
| delta_own_under_threat_count | -0.171 | 0.294 | 0.123 |
| delta_enemy_contested_count | 0.091 | 0.211 | 0.12 |
| delta_n_neutral_planets | -0.031 | 0.114 | 0.083 |

## Топ-10 фич × все страты

| feature | страта | n_win | n_los | Cohen's d |
|---|---|---|---|---|
| delta_prod_gap | small_static | 58 | 58 | 3.297 |
| delta_prod_gap | large_orbital | 29 | 29 | 3.277 |
| delta_prod_gap | medium_static | 43 | 43 | 3.254 |
| delta_prod_gap | medium_orbital | 69 | 69 | 3.126 |
| delta_prod_gap | small_orbital | 19 | 19 | 2.829 |
| delta_own_prod_ratio | medium_static | 43 | 43 | 2.71 |
| delta_own_prod_ratio | small_orbital | 19 | 19 | 2.686 |
| delta_own_prod_ratio | small_static | 58 | 58 | 1.835 |
| delta_own_prod_ratio | large_orbital | 29 | 29 | 1.796 |
| delta_own_prod_ratio | medium_orbital | 69 | 69 | 1.683 |
| delta_own_prod_total | small_orbital | 19 | 19 | 2.678 |
| delta_own_prod_total | medium_static | 43 | 43 | 2.514 |
| delta_own_prod_total | large_orbital | 29 | 29 | 1.721 |
| delta_own_prod_total | small_static | 58 | 58 | 1.7 |
| delta_own_prod_total | medium_orbital | 69 | 69 | 1.676 |
| delta_own_planet_ratio | small_orbital | 19 | 19 | 2.419 |
| delta_own_planet_ratio | medium_static | 43 | 43 | 2.277 |
| delta_own_planet_ratio | small_static | 58 | 58 | 1.7 |
| delta_own_planet_ratio | large_orbital | 29 | 29 | 1.666 |
| delta_own_planet_ratio | medium_orbital | 69 | 69 | 1.593 |
| delta_n_own_planets | small_orbital | 19 | 19 | 2.3 |
| delta_n_own_planets | medium_static | 43 | 43 | 1.997 |
| delta_n_own_planets | large_orbital | 29 | 29 | 1.625 |
| delta_n_own_planets | small_static | 58 | 58 | 1.538 |
| delta_n_own_planets | medium_orbital | 69 | 69 | 1.476 |
| delta_ship_gap | medium_static | 43 | 43 | 2.111 |
| delta_ship_gap | large_orbital | 29 | 29 | 1.717 |
| delta_ship_gap | small_orbital | 19 | 19 | 1.665 |
| delta_ship_gap | medium_orbital | 69 | 69 | 1.374 |
| delta_ship_gap | small_static | 58 | 58 | 0.861 |
| delta_n_enemy_planets | medium_orbital | 69 | 69 | -1.365 |
| delta_n_enemy_planets | large_orbital | 29 | 29 | -1.498 |
| delta_n_enemy_planets | small_static | 58 | 58 | -1.621 |
| delta_n_enemy_planets | medium_static | 43 | 43 | -1.979 |
| delta_n_enemy_planets | small_orbital | 19 | 19 | -1.99 |
| delta_own_ship_ratio | medium_static | 43 | 43 | 1.866 |
| delta_own_ship_ratio | large_orbital | 29 | 29 | 1.569 |
| delta_own_ship_ratio | medium_orbital | 69 | 69 | 1.4 |
| delta_own_ship_ratio | small_orbital | 19 | 19 | 1.386 |
| delta_own_ship_ratio | small_static | 58 | 58 | 1.167 |
| delta_own_ships_total | medium_static | 43 | 43 | 1.393 |
| delta_own_ships_total | medium_orbital | 69 | 69 | 0.883 |
| delta_own_ships_total | small_orbital | 19 | 19 | 0.726 |
| delta_own_ships_total | large_orbital | 29 | 29 | 0.722 |
| delta_own_ships_total | small_static | 58 | 58 | 0.681 |
| delta_incoming_own_reinforce | small_orbital | 19 | 19 | 1.288 |
| delta_incoming_own_reinforce | large_orbital | 29 | 29 | 0.643 |
| delta_incoming_own_reinforce | medium_static | 43 | 43 | 0.473 |
| delta_incoming_own_reinforce | small_static | 58 | 58 | 0.471 |
| delta_incoming_own_reinforce | medium_orbital | 69 | 69 | 0.333 |

## Графики

- `stratified_amplification.png` — усиление по фичам
- `stratified_top_features.png` — топ фич по стратам
