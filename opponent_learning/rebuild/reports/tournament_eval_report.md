# Tournament: baseline vs ML residual

**Model**: linear residual on zones features
**Seeds**: 5 per alpha

## Результаты

| alpha | n | win_rate_ml | avg_ships_diff |
|---|---|---|---|
| 0.0 | 5 | 0.400 | +25.6 |
| 0.3 | 5 | 0.400 | -687.2 |
| 0.7 | 5 | 0.400 | -533.6 |
| 1.0 | 5 | 0.400 | -364.6 |

## Интерпретация

- win_rate_ml > 0.55 → ML residual даёт реальное улучшение
- win_rate_ml ≈ 0.5 → нейтрально
- win_rate_ml < 0.45 → ML residual ухудшает игру
