# Состояние проекта на старте Phase 0

Короткий отчёт после разведки репозитория — что у нас есть, что отсутствует, и какой первый коммит делать в Phase 0.

## Что уже есть и переиспользуем

### Симулятор (`agent_bundle/orbit_sim.py`) — полная имплементация

Это серьёзный актив. Имеется:

- `GameState` — снимок состояния с сериализацией (`to_dict`, `from_dict`, `from_kaggle_obs`, `from_json`). Поля: planets, fleets, omega, step, n_players, comet_ids, initial_planets.
- `simulate_step(state, moves_per_player)` — полный шаг физики: launch → production → fleet move → rotate → combat. Один-в-один с движком Kaggle.
- `run_simulation(state, n_steps, policy)` — multi-step rollouts.
- `GraphConnectivity` с 4 вариантами весов рёбер (W1–W4) — это де-факто часть Level 3 нашего плана: pressure-метрика, marginal_capture_value, capture_priority, shortcut_planets.
- `generate_map(seed)` — синтетические карты с симметрией.

**Следствие**: Phase 5 и Phase 6 нашего плана разблокированы — симулятор есть, counterfactual rollouts возможны напрямую.

### Эвристический агент (`agent_bundle/`, `agent_bundle2/`)

Полный pipeline: проекция флотов (`projection.py`), зонирование (`zones.py`), планирование атак (`attacks.py`), нацеливание (`shooting.py`), force-вычисления (`force.py`), debug (`agent_debug.py`). Адаптивный режим (`behind/parity/ahead/dominate`) на основе ratio наших кораблей.

**Следствие**: у нас уже есть сильный non-trivial бейзлайн для Phase 0 (не только `nearest_weakest`). Это поднимает планку нетривиальности — модель должна обыграть **этого** агента, не только примитивные эвристики.

### Турнирная инфраструктура (`tuning/scripts/`)

Около 25 скриптов: tournaments (`zones_tournament`, `wours_tournament`, `reverse_tournament`), grid search и Bayesian оптимизация весов, eval против оппонентов (`eval_8_winners`, `eval_vs_opponents`, `evaluate_candidates`), match analyzer, визуализаторы, два варианта router (`train_router`, `train_router_v2`).

**Следствие**: Phase 7 (round-robin tournament, ELO) делается на существующей инфраструктуре. Не плодим параллельный код.

### Match-level статистика (`tuning/results/analysis_*.csv/.parquet`)

Уже посчитанные per-match признаки: dominance_area, mid_game_lead, lead_changes, turn_total_war, avg_ships_diff по квартилям времени, cluster. Это партии **локальных** турниров их агента (не топ-20).

**Следствие**: формат для нашего будущего feature store уже намечен. Полезно как референс.

### Документация (`docs/features/`)

Obsidian vault: atoms, intrinsic features, pairs, action_plans, gul_context, zones_priority, swarm_decisions, dependencies.

### Данные (`opponent_learning/data/raw/`)

- 500 эпизодов в JSON (2.6 GB) — формат Kaggle environments
- `episodes_index.json` — список 4704 episode ID (то есть скачано пока 500 из 4704 потенциально доступных)
- Структура одного эпизода:
  - `configuration`: seed, episodeSteps, shipSpeed, cometSpeed
  - `steps[t]`: list из 2 (по игроку), каждый = `{action, observation, reward, status}`
  - `observation`: planets, fleets, initial_planets, angular_velocity, step, player
  - `planets`/`fleets`: list of tuples — формат совпадает с `Planet`/`Fleet` из orbit_sim
  - `action`: список ходов `[[from_id, angle, ships], ...]` (пустой список = no-op)
  - `rewards[i]`: +1/-1/0 — исход для игрока i
  - `info.TeamNames`: имена игроков (для будущей фильтрации по топ-20)

**Следствие**: `from_kaggle_obs(obs)` уже умеет читать эти observation, парсер становится тонким адаптером.

## Чего нет — то, что нам нужно построить

1. **Парсер реплеев JSON → стандартизованный per-step датасет**
2. **Бейзлайны как чистые функции** `state → action_distribution` (random, nearest_weakest, reinforce_weakest, linear_macro, и существующий heuristic agent как baseline)
3. **`triviality_score`** — разметка каждого состояния
4. **Feature store на Parquet** с per-step признаками
5. **Pipeline Level 1–3 признаков** (некоторые частично уже есть в orbit_sim/zones — переиспользуем)
6. **Value function на макро-фичах**

## Расхождения с исходным планом

В плане я писал "500 реплеев → 386K ситуаций". С учётом того, что в среднем партия ~150–200 шагов × 2 игрока — получим ~150–200K (одна точка на (игрок, шаг)). Возможно, в исходных подсчётах включались доп. варианты (попадание/отказ, обе перспективы и т.д.). Уточним по факту парсинга.

В плане Phase 5/6 я закладывал день-два на симулятор — этот пункт **закрыт**, экономия 1–2 дня.

GraphConnectivity покрывает примерно 60% наших задумок Level 3 (поле контроля как граф). Дополним только: топологию (компоненты связности, длина фронта в R²), reachability volumes, race-pair матрицы.

## Предложение по первому коммиту

**Задача**: парсер эпизодов в стандартизованный per-step датасет.

**Что делаем**:
- `opponent_learning/scripts/parse_episodes.py` — функция, читающая один JSON эпизода и возвращающая итератор `(replay_id, step, player_idx, state: GameState, action: list, outcome: int, opponent_name: str)`.
- `opponent_learning/scripts/build_dataset.py` — обходит все 500 файлов, складывает в Parquet-shards (по 50 эпизодов в shard, чтобы влезало в память).
- Smoke-test: 5 случайных эпизодов прогнать через парсер, восстановить `GameState`, прогнать `simulate_step` с реальным action, сравнить с следующим состоянием в реплее. Если совпадает — парсер корректен и симулятор синхронен.
- Базовые sanity-метрики: распределение длин партий, action_type, num_planets, owner balance.

**Срок**: 1 день.

**Гейт**:
- ✅ 100% эпизодов парсятся
- ✅ Smoke-test (sim vs reality) сходится на 5/5 партий с допуском по координатам
- ✅ Сохранён Parquet с N ≥ 100K записей (state не сериализуем целиком, только references — JSON-сжатая state per record)

После этого открыта вся остальная Phase 0.
