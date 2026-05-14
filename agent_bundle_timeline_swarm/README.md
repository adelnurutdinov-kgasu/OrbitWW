# agent_bundle_timeline_swarm

Экспериментальный планировщик на основе **полных таймлайнов владения** планет.
Альтернатива классическому `agent_bundle_swarm/swarm.py` — той же функцией
сигнатуры, но с принципиально иной моделью мира.

## Зачем

Классический `swarm_plan` опирается на **бинарную проекцию**: каждая планета
после `project_state` имеет один спроецированный owner (наш / враг / нейтрал).
Это теряет три важных измерения:

1. **Время** — планета, которая станет нашей через 30 ходов, не отличима от
   той, что уже наша.
2. **Хрупкость** — захват «впритык» (margin=1) выглядит так же, как захват с
   огромным запасом (margin=50).
3. **Зависимости** — без понимания, какие конкретно флоты «держат» захват,
   невозможно осмысленно решать про reroute / acceleration.

Таймлайн владения возвращает все три. Каждая планета представляется как
последовательность событий `(turn, owner, ships)`, и любой ответ —
«кому принадлежит P в момент t?», «когда впервые становится нашей?»,
«а если убрать вот этот флот?» — получается элементарной операцией над
таймлайном.

## Архитектура: 6 слоёв

```
L0 World          текущее state + летящие fleets
   ↓
L1 Timelines      simulator.build_timelines
                  для каждой планеты — PlanetTimeline с резолюцией событий
   ↓
L2 Opportunities  opportunities.extract_opportunities
                  таймлайны → atomic спрос (capture/defend/accelerate/snipe/...)
   ↓
L3 Budgets        budgets.compute_budgets
                  для каждой нашей планеты: min over horizon of (ships - defense)
   ↓
L4 Auction        auction.run_auction
                  многотиерный жадный аукцион по marginal ROI
                  тиер 0 (defend) → 1 (capture) → 2 (snipe) → 3 (accelerate) → 4 (safety)
   ↓
L5 Validate       validator.validate
                  откат самых низко-ROI коммитов если планеты теряются
   ↓
L6 Orders         совместимый со swarm_plan формат планов
```

## Ключевая новизна — accelerate

Возможность **accelerate** возникает, когда планета по таймлайну становится
нашей, но не сразу: `becomes_ours_at > now + ACCELERATE_MIN_DELAY`.

Value такой возможности: `production × (old_eta - new_arrival) × discount`.

Конкурирует в аукционе с обычными `capture`/`defend` на равных, в одной
валюте «эквивалентных кораблей». Тиерная защита гарантирует, что
ускорение не отъест корабли у критической обороны или захватов.

## Совместимость

`timeline_swarm_plan(state, player, targets, weights)` возвращает
`(plans, debug)` в формате, который умеет потреблять
`agent._execute_plan_atomically`. Поля плана:

```python
{
  'mode': 'direct' | 'transfer',
  'sup_id': None,
  'att_id': src_id, 'tgt_id': dst_id,
  'x_att': ships, 'x_sup': 0,
  't_total': eta, 'success': True,
  'opp_type': 'capture'|'accelerate'|...,
  'opp_tier': int,
  'roi': float,
  ...
}
```

Можно подменить планировщик в `agent.py` одной строкой:

```python
# from swarm import swarm_plan
from agent_bundle_timeline_swarm import timeline_swarm_plan as swarm_plan
```

## Запуск синтетического теста

```bash
cd agent_bundle_timeline_swarm
python test_scenario.py
```

Три сценария: pure_capture, accelerate, defend. Каждый прогоняется
через timeline_swarm и (если bundle импортируется) через классический
swarm — для side-by-side сравнения.

## Текущие ограничения

* Аукцион **жадный без бэктрекинга**. Локальные оптимумы возможны.
* `_eligible_sources` использует простую евклидову ETA для планируемых
  отправок (не вызывает `_rendezvous_eta`). Это вычислительно дешевле
  и достаточно для прототипа; финальный запуск всё равно идёт через
  `_aim_and_verify` в агенте.
* Не учитываются кометы (`comet_ids` игнорируются — это TODO).
* `required_defense_at` — грубая эвристика (max threat × safety). На
  реальных игровых state, возможно, потребуется тюнинг `react_safety`.
* `validator` откатывает по одному коммиту за итерацию; на сложных
  состояниях возможен N² по числу проблем.

## Файлы

| Файл              | Слой | Назначение                                  |
|-------------------|------|---------------------------------------------|
| `timeline.py`     | —    | PlanetTimeline + Event + резолюция боя      |
| `simulator.py`    | L1   | build_timelines из state.fleets             |
| `opportunities.py`| L2   | scan timelines → list[Opportunity]          |
| `budgets.py`      | L3   | min-over-horizon free ships                 |
| `auction.py`      | L4   | многотиерный greedy marginal ROI            |
| `validator.py`    | L5   | rollback по нарушениям инвариантов          |
| `timeline_swarm.py`| L6   | главный entry point                         |
| `test_scenario.py`| —    | 3 синтетических сценария + классический swarm для сравнения |
