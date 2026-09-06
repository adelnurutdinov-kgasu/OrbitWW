# Pro vs Heuristic — comparison report

**Данные**: 590 pro-launches, валидных 296
**Эвристика**: `agent_bundle_swarm 2`, USE_MCTS=False

## Главные числа

| Metric | Value |
|---|---|
| Top-1 target match | 17.9% |
| Top-3 target match | 37.8% |
| Top-5 target match | 52.0% |
| Top-10 target match | 72.0% |
| Type match (n/e/r) | 32.4% |
| Source match | 14.2% |
| Heur idle when pro fired | 31.5% |
| Median pro_rank | 5 |

## По типу действия профи

| type | n | top1 | top3 | median_rank |
|---|---|---|---|---|
| neutral | 61 | 13.1% | 44.3% | 6 |
| enemy | 190 | 16.8% | 50.0% | 6 |
| reinforce | 45 | 28.9% | 71.1% | 3 |

## По фазе

```
           n   top1   top3  med_rank
phase_q                             
q1        44  0.182  0.295       8.0
q2        92  0.152  0.337       5.0
q3       108  0.194  0.380       5.0
q4        52  0.192  0.519       3.0
```

## По n_players

```
             n   top1   top3  med_rank
n_players                             
2          112  0.188  0.375       5.5
4          184  0.174  0.380       5.0
```

## Графики

- `pro_vs_heur_rank_hist.png` — распределение rank pro_target
- `pro_vs_heur_match_by_phase.png` — top-K match по фазам
