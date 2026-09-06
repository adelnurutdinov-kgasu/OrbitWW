# Delta-features в pre-resolution окне

**Партий-игроков**: 462 (winners=231, losers=231)

## Главная цифра

| Модель | n_features | AUC mean | AUC std | Acc mean |
|---|---|---|---|---|

Порог осмысленности: AUC ≥ 0.65 для delta_only — значит изменения в окне разделяют исходы.
Если delta+start = start_only — изменения ничего не добавляют, только текущая позиция.

## Топ-20 delta-фич по эффект-размеру

| feature | gap | d (Cohen) | log10_p |
|---|---|---|---|
| delta_prod_gap | 27.1818 | 3.133 | -300.0 |
| delta_own_prod_ratio | 0.1859 | 1.881 | -300.0 |
| delta_own_prod_total | 13.7922 | 1.78 | -300.0 |
| delta_own_planet_ratio | 0.1616 | 1.751 | -300.0 |
| delta_n_own_planets | 4.7143 | 1.571 | -300.0 |
| delta_n_enemy_planets | -4.5887 | -1.533 | -300.0 |
| delta_own_ship_ratio | 0.185 | 1.37 | -300.0 |
| delta_ship_gap | 352.3333 | 1.328 | -300.0 |
| delta_own_ships_total | 180.2468 | 0.799 | -300.0 |
| delta_fleet_balance | 136.6926 | 0.501 | -7.14 |
| delta_incoming_own_reinforce | 44.1039 | 0.455 | -5.56 |
| delta_net_incoming_own | 63.7186 | 0.426 | -5.3 |
| delta_max_own_ships | 28.342 | 0.414 | -4.96 |
| delta_own_fleets_count | 5.0823 | 0.393 | -4.36 |
| delta_own_fleets_ships | 68.0476 | 0.374 | -4.12 |
| delta_enemy_fleets_count | -5.0519 | -0.373 | -3.96 |
| delta_enemy_fleets_ships | -68.645 | -0.366 | -3.95 |
| delta_std_own_ships | 6.0834 | 0.307 | -3.01 |
| delta_mean_own_ships | 3.4297 | 0.191 | -1.39 |
| delta_incoming_enemy_to_own | -19.6147 | -0.177 | -1.22 |

## Графики

- `delta_features_top12.png` — гистограммы winners vs losers по топ-12 фич
