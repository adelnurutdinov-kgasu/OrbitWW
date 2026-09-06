# Redo Stratification (по size_bucket only)

Перезапуск 08/09/10 после корректировки 07b: ось orbital/static
была артефактом. Используем только корректную ось — `size_bucket`.

## A. Delta-features: усиление при стратификации

| feature | d_overall | small | medium | large | max_abs_d | amplification |
|---|---|---|---|---|---|---|
| delta_incoming_enemy_to_own | -0.177 | -0.069 | -0.144 | -0.48 | 0.48 | 0.303 |
| delta_neutral_contested_count | 0.038 | 0.338 | -0.077 | -0.058 | 0.338 | 0.3 |
| delta_net_incoming_own | 0.426 | 0.502 | 0.378 | 0.716 | 0.716 | 0.29 |
| delta_fleet_balance | 0.501 | 0.79 | 0.418 | 0.785 | 0.79 | 0.289 |
| delta_enemy_fleets_ships | -0.366 | -0.557 | -0.294 | -0.621 | 0.621 | 0.255 |
| delta_own_fleets_ships | 0.374 | 0.501 | 0.334 | 0.626 | 0.626 | 0.252 |
| delta_ship_gap | 1.328 | 0.922 | 1.574 | 1.517 | 1.574 | 0.246 |
| delta_enemy_contested_count | 0.091 | 0.109 | -0.004 | 0.333 | 0.333 | 0.242 |
| delta_own_ships_total | 0.799 | 0.673 | 1.015 | 0.693 | 1.015 | 0.216 |
| delta_enemy_fleets_count | -0.373 | -0.559 | -0.307 | -0.512 | 0.559 | 0.186 |
| delta_incoming_own_reinforce | 0.455 | 0.588 | 0.393 | 0.637 | 0.637 | 0.182 |
| delta_own_ship_ratio | 1.37 | 1.157 | 1.539 | 1.392 | 1.539 | 0.169 |
| delta_own_fleets_count | 0.393 | 0.551 | 0.355 | 0.516 | 0.551 | 0.158 |
| delta_own_under_threat_ships | -0.136 | 0.064 | -0.191 | -0.291 | 0.291 | 0.155 |
| delta_mean_own_ships | 0.191 | 0.035 | 0.324 | 0.277 | 0.324 | 0.133 |
| delta_own_under_threat_count | -0.171 | 0.042 | -0.287 | -0.227 | 0.287 | 0.116 |
| delta_n_enemy_planets | -1.533 | -1.625 | -1.517 | -1.578 | 1.625 | 0.092 |
| delta_n_own_planets | 1.571 | 1.569 | 1.599 | 1.661 | 1.661 | 0.09 |
| delta_own_prod_total | 1.78 | 1.729 | 1.869 | 1.715 | 1.869 | 0.089 |
| delta_max_own_ships | 0.414 | 0.422 | 0.423 | 0.472 | 0.472 | 0.058 |

## B. Value AUC по size_bucket

### Все партии (1v1+FFA)

| stratum | n | AUC | Acc | pos_rate |
|---|---|---|---|---|
| large | 8418 | 0.9512 | 0.8747 | 0.3295 |
| medium | 29322 | 0.8898 | 0.8295 | 0.3049 |
| small | 16294 | 0.9076 | 0.8521 | 0.2997 |

### 1v1 only

| stratum | n | AUC |
|---|---|---|
| large | 2678 | 0.9148 |
| medium | 6438 | 0.9031 |
| small | 3238 | 0.9249 |

### AUC vs phase quartile × size_bucket (1v1)

| stratum | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| large | 0.528 | 0.853 | 0.997 | 1.0 |
| medium | 0.597 | 0.821 | 0.964 | 1.0 |
| small | 0.642 | 0.85 | 0.992 | 1.0 |

## C. Safe zones по size_bucket (1v1)

| stratum | n | soft_low | soft_high | hard_low | hard_high |
|---|---|---|---|---|---|
| large | 14254 | 0.24000000024 | 0.7600000007600001 | 0.14000000014000002 | 0.8600000008600001 |
| medium | 40036 | 0.38000000038000004 | 0.6000000006 | 0.14000000014000002 | 0.8600000008600001 |
| small | 26272 | 0.40000000040000006 | 0.6000000006 | 0.22000000022000002 | 0.7800000007800001 |

GLOBAL: soft=(np.float64(0.38000000038000004), np.float64(0.6000000006))  hard=(np.float64(0.16000000016000002), np.float64(0.84000000084))
