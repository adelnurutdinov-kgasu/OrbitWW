# Retrained ranker — zones features (готов к интеграции)

**Dataset**: 42636 train + 5222 val choice_sets
**Features**: 7 planet + 11 context

## Метрики

| Model | top1 | top3 | top5 | top10 | median_rank |
|---|---|---|---|---|---|
| baseline (zones priority) | 0.0546 | 0.1478 | 0.2361 | 0.4642 | 11.0 |
| linear | 0.0751 | 0.1963 | 0.3043 | 0.5699 | 9.0 |

## Топ-10 весов linear (по |w|)

| kind | feature | weight |
|---|---|---|
| context | n_neutral_planets | +0.9920 |
| context | src_wnn | +0.8629 |
| context | src_prod | -0.6606 |
| context | n_enemy_planets | -0.6505 |
| context | src_ships | -0.6203 |
| context | own_ship_ratio | -0.5706 |
| context | phase | -0.4293 |
| context | src_ripeness | -0.3546 |
| context | own_prod_ratio | -0.3386 |
| planet | mean_dist_all | +0.3113 |

## Готов для интеграции

Эти веса можно подключить как residual в zones.compute_zones_from_state
через скрипт 19_tournament_eval.py.
