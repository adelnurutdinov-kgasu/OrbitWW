# BO результат: zones-веса под pro-matching

**Target metric**: top3
**Trials**: 150, sample: 400

## Метрики baseline vs best

| metric | baseline | best | Δ |
|---|---|---|---|
| top1 | 0.1327 | 0.1224 | -0.0102 |
| top3 | 0.3163 | 0.4082 | +0.0918 |
| top5 | 0.4286 | 0.5204 | +0.0918 |
| top10 | 0.6429 | 0.6531 | +0.0102 |
| median_rank | 7.0 | 5.0 | -2.0 |

## Найденные веса W_OURS

| feature | baseline | best | Δ |
|---|---|---|---|
| area_inv | -0.100 | -1.438 | -1.338 |
| wnn_close_res | -0.400 | +0.546 | +0.946 |
| mean_dist_all | -0.300 | -0.524 | -0.224 |
| prod | +0.400 | +0.035 | -0.365 |
| ships | +0.000 | +0.180 | +0.180 |
| n_cross | +0.500 | -1.146 | -1.646 |
| late_aggression | +0.200 | +0.029 | -0.171 |

## Найденные веса W_TARGETS

| feature | baseline | best | Δ |
|---|---|---|---|
| area_inv | +0.400 | +0.088 | -0.312 |
| wnn_close_res | +0.800 | +0.634 | -0.166 |
| mean_dist_all | -0.200 | +0.152 | +0.352 |
| prod | +0.500 | -0.300 | -0.800 |
| ships | -0.400 | -0.555 | -0.155 |
| n_cross | +0.400 | +1.403 | +1.003 |
| late_aggression | +0.900 | +0.473 | -0.427 |

## Готовые подставки в zones.py

```python
W_OURS = {
    'area_inv': -1.438,
    'wnn_close_res': +0.546,
    'mean_dist_all': -0.524,
    'prod': +0.035,
    'ships': +0.180,
    'n_cross': -1.146,
    'late_aggression': +0.029,
}

W_TARGETS = {
    'area_inv': +0.088,
    'wnn_close_res': +0.634,
    'mean_dist_all': +0.152,
    'prod': -0.300,
    'ships': -0.555,
    'n_cross': +1.403,
    'late_aggression': +0.473,
}
```
