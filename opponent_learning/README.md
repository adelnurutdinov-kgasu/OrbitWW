# Opponent Learning Pipeline

Автоматическое обучение байесовской модели оппонента на реальных replay-логах
топ-игроков Kaggle orbit_wars.

## Зачем это нужно

Текущие пресеты в `opponent_presets.py` — варианты **нашего собственного агента**
(SwarmWeights с разными параметрами). Реальные соперники используют совершенно
другую логику и байесовская модель никогда не сходится к правильному типу.

Обучение на логах даёт:
1. **Реальные пресеты** — 4–5 поведенческих архетипов реальных участников
2. **Калиброванный prior** — знаем какой тип встречается в ~30% игр, а какой в ~10%
3. **Map-agnostic likelihood** — сравнение по поведенческим фичам, не по `(from_id, target_id)`

## Что извлекаем из логов

Для каждого игрока в каждой игре вычисляем 9 поведенческих фич:

| Фича | Что измеряет | Диапазон |
|------|-------------|----------|
| `attack_rate` | доля ходов с хоть одним запуском | 0..1 |
| `expansion_bias` | доля атак на нейтральные планеты | 0=только враг, 1=только нейтралы |
| `ships_fraction` | средняя доля кораблей отправленных за ход | 0..1 |
| `avg_distance_frac` | средняя нормализованная дистанция до цели | 0=ближний, 1=дальний |
| `overkill_ratio` | среднее ships_sent / target_ships | 0.5..3+ |
| `early_rate` | attack_rate на первых 100 ходах | 0..1 |
| `late_rate` | attack_rate на последних 100 ходах | 0..1 |
| `multi_launch_rate` | доля ходов с >1 одновременным запуском | 0..1 |
| `idle_peak_frac` | пик доли кораблей «залёживающихся» без атаки | 0..1 |

## Варианты обучения

### Вариант A (реализован): Поведенческие архетипы → SwarmWeights-пресеты

```
Фичи → K-Means k=5 → 5 кластеров → линейный маппинг → SwarmWeights
```

Плюсы: простой, интерпретируемый, совместим с текущей архитектурой  
Минусы: линейный маппинг неточен (реальные оппоненты ≠ SwarmWeights-агент)

### Вариант B: Feature-based Bayesian (улучшение likelihood)

Вместо точного `(from_id, target_id)` матчинга — сравнение поведенческих
фич текущего хода (`expansion_bias`, `ships_fraction`, `n_actions`) с
предсказанием пресета. Map-agnostic, не зависит от случайной геометрии.

Реализовано в `04_update_presets.py` как `_behavioral_likelihood()`.

### Вариант C: Прямой supervised learning (будущее)

```
(game_state_features, step) → предсказываемые действия
```

Нужно: 500+ игр, нейросеть / gradient boosting.  
Даст: предсказание конкретных планет-целей, а не просто тип поведения.

### Вариант D: Self-play разнообразие

Запустить наш агент со всеми найденными настройками против себя → получить
synthetic replay-данные дешевле чем ждать Kaggle-матчей.

## Запуск

```bash
# 1. Задать Kaggle credentials
export KAGGLE_USERNAME=твой_логин
export KAGGLE_KEY=твой_ключ

# 2. Скачать ~2000 эпизодов топ-игроков (≈35 минут, ~1 req/sec)
python3 opponent_learning/scripts/01_download_episodes.py

# 3. Извлечь поведенческие фичи из replay-файлов
python3 opponent_learning/scripts/02_extract_features.py

# 4. Кластеризовать и смаппировать на SwarmWeights
python3 opponent_learning/scripts/03_cluster_and_fit.py --k 5

# 5. Обновить opponent_presets.py и opponent_model_bayesian.py
python3 opponent_learning/scripts/04_update_presets.py --dry-run  # проверить
python3 opponent_learning/scripts/04_update_presets.py             # применить

# 6. Пересобрать submission
python3 tuning/scripts/build_submission.py
```

## Структура папок

```
opponent_learning/
  data/
    raw/           ← replay JSON от Kaggle API (*.json)
    processed/
      features.csv          ← поведенческие фичи (вывод скрипта 02)
      features_clustered.csv ← с меткой кластера (вывод скрипта 03)
  results/
    clusters.json           ← описание кластеров + weights
    presets_generated.py    ← фрагмент кода для вставки вручную
  scripts/
    01_download_episodes.py
    02_extract_features.py
    03_cluster_and_fit.py
    04_update_presets.py
  README.md
```

## Ожидаемые архетипы (гипотеза)

| Архетип | attack_rate | expansion_bias | ships_fraction | Описание |
|---------|-------------|----------------|----------------|----------|
| rusher | >0.6 | >0.6 | >0.5 | Ранняя экспансия нейтралов, много запусков |
| brawler | >0.5 | <0.4 | >0.5 | Агрессивно атакует врага, расточителен |
| expander | 0.4–0.6 | >0.65 | 0.3–0.5 | Тихая экспансия, избегает конфликтов |
| turtle | <0.3 | any | <0.3 | Пассивный, накапливает и бьёт поздно |
| balanced | 0.3–0.5 | ~0.5 | ~0.4 | Средний по всем параметрам |
