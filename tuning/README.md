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
│   └── match_analyzer.py       — сбор map-features + per-turn time series
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
