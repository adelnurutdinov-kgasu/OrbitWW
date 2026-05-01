# tuning/

Все скрипты для подбора параметров и сбора статистики, плюс CSV-результаты.

## Структура

```
tuning/
├── scripts/               # запускаемые скрипты
│   ├── zones_grid_search.py    — full grid по W_TARGETS из zones.py
│   ├── zones_tournament.py     — successive halving по W_TARGETS
│   ├── wours_tournament.py     — то же по W_OURS
│   ├── zones_bayesian.py       — TPE/optuna по всем 14 весам сразу
│   ├── match_analyzer.py       — сбор map-features + per-turn time series
│   ├── reverse_tournament.py   — RL-стиль: на каждый LOSS ищем веса для победы
│   └── local_match.py          — детерминированный runner (orbit_sim напрямую)
└── results/               # CSV / parquet / SQLite — выходы скриптов
    ├── analysis_meta_*.csv      — мета на seed (map-features + derived)
    ├── analysis_ts_*.parquet    — long-format per-turn series
    ├── analysis_summary_*.txt
    ├── zones_grid_*.csv
    ├── zones_tour_*.csv
    ├── wours_tour_*.csv
    ├── zones_bayesian_*.csv
    └── zones_bayesian_study.db
```

## Запуск

Каждый скрипт автономен, имеет блок `── КОНФИГ ──` в начале. Запуск из любой
директории — пути вычисляются относительно расположения скрипта:

```bash
python3 tuning/scripts/zones_tournament.py
python3 tuning/scripts/wours_tournament.py
python3 tuning/scripts/zones_bayesian.py     # требует pip install optuna
python3 tuning/scripts/match_analyzer.py
```

Результаты пишутся в `tuning/results/`. Существующая логика `test.ipynb` на
эти скрипты не завязана — продолжает работать как раньше.

## Что делает каждый скрипт

| Скрипт | Что ищет | Метод | Бюджет |
|---|---|---|---|
| `zones_grid_search.py` | W_TARGETS | full grid | N_seeds × N_combos |
| `zones_tournament.py` | W_TARGETS | successive halving | ~1/4 от full grid |
| `wours_tournament.py` | W_OURS | successive halving | то же |
| `zones_bayesian.py` | W_OURS + W_TARGETS (14D) | TPE | 250 trials × 5 seeds |
| `match_analyzer.py` | сбор данных | run N matches с фиксированными весами | N seeds |
| `reverse_tournament.py` | SwarmWeights (по проигрышам) | random + CEM-довод, train/holdout | ~(LOSSES × 12-20) матчей |

### `reverse_tournament.py` — детали

RL-стиль: baseline → найти все LOSS → для каждого ищем веса до победы → consensus.

* **Train/holdout split**: SEED разбивается 70/30 (детерминированно), поиск идёт
  только на TRAIN. Holdout проверяется в конце с найденным consensus — если
  винрейт там не упал, режим обобщается и не подгонялся под конкретные карты.
* **Фаза 1 — random uniform** (PHASE1_BUDGET=12 по умолчанию).
* **Фаза 2 — CEM** на топ-5 элите фазы 1 (PHASE2_BUDGET=8). Включается только
  если фаза 1 не дала победы. EARLY_STOP=True по дефолту: первая победа на
  сиде → переходим к следующему.
* **Dead-end сиды** — те, на которых ни одна попытка не выиграла. Их список
  печатается отдельно — это сигнал «либо баг, либо структурно невозможный сид».
* **Consensus weights** — медиана по победным trial'ам, проверяется на holdout.
* **Оценка времени**: 70 сидов, baseline ~3-6s/match через локальный runner
  → 5 минут. Если 50% LOSS, поиск ~50 × (12+8) × 4s ≈ 70 минут на 2 worker'а.

### `local_match.py` — почему не kaggle_environments

Эмпирически kaggle_environments выдаёт разные исходы при том же `seed`
(вероятно неполный сид, time-based jitter где-то внутри обёртки). Для
тюнинга это ломает RL-цикл: ты находишь «победные» веса на seed=N,
запускаешь снова — а там уже LOSS с теми же весами. Бесполезно сравнивать
конфиги.

Локальный runner использует `orbit_sim.generate_map(seed)` + `simulate_step`
напрямую. Никакого рандома вне seed → одинаковая конфигурация всегда даёт
одинаковый исход (проверено: побайтово). Дополнительно ~10× быстрее, так как
нет overhead'а kaggle env.

Карта генерится по правилам orbit_sim, а не точно как у kaggle. Распределение
похожее (omega ∈ [0.025, 0.05], n_groups, geometry), но не побитово совпадает.
Для ранжирования весов это не принципиально. Финальный smoke-test лучших
конфигов делать через `test.ipynb` / kaggle env вручную.
