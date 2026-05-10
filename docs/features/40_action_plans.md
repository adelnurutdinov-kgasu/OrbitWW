---
tags: [orbit-wars, features, action]
---

# Action plans — типы действий

Каждая запись в `plans = [...]` это **action**. Все виды и их формулы.

`code:attacks.py:all_plans(state, tgt, ours, horizon, player, risk)`

## Common fields

```
mode:        'direct' | 'multi_sync' | 'pipeline' | 'transfer'
att_id:      кто стреляет (главный)
sup_id:      кто помогает (None для direct/transfer)
tgt_id:      куда летит
x_att:       сколько ships из att
x_sup:       сколько ships из sup (0 для direct)
eta_sa:      ETA sup→att (для pipeline), 0 для остальных
eta_at:      ETA att→tgt (для direct и pipeline)
eta_st:      ETA sup→tgt (для multi_sync)
t_total:     полное время до удара
defender:    расчётная защита в момент прилёта
strike:      наш удар (cost ships)
margin:      strike − defender (положительное = win)
success:     bool, выигрывает ли план
```

## Direct

`code:attacks.py:eval_direct`

```
strike   = x_att
defender = _defender_at_eta(state, tgt, eta)        # учитывает early intercept
needed   = ceil(defender − incoming) + overkill
success  = needed <= att.ships − RESERVE_ON_ATT
```

- Если path att→tgt пересекает солнце → `None` (нет плана).
- `incoming` = ships от уже летящих наших флотов к этой цели.

## Multi-sync (duplet)

`code:attacks.py:eval_multi_sync`

Два источника, один target. Прилетают почти одновременно:

```
eta_st  = ETA(sup, tgt, sup.ships)
eta_at  = ETA(att, tgt, att.ships)
t_sync  = max(eta_st, eta_at)

# одновременность с допуском
|eta_st − eta_at| ≤ MAX_MULTI_SYNC_DELTA_ETA (5)

strike   = (att.ships) + (sup.ships)       # сразу, без накопления prod
defender = _defender_at_eta(state, tgt, t_sync)
```

Используется когда `att` одного не хватает. Отсев: 
- солнце на любом из двух путей → None
- |Δeta| > 5 → None (быстрый прилетит один, проиграет)

## Pipeline (relay)

`code:attacks.py:eval_pipe`

Цепочка: sup → att (передача) → att накапливает prod → att → tgt (удар).

```
eta_sa  = ETA(sup, att, sup.ships)
boosted = sup.ships + att.ships + att.production · eta_sa
eta_at  = ETA(att, tgt, boosted)
t_total = eta_sa + eta_at

x_sup_min = max(1, needed − att.ships − att.production · eta_sa)
```

**Уравнение баланса в момент удара:**

> X_sup + X_att + PROD_att · T_sa = X_tgt + PROD_tgt · (T_sa + T_at)

Отсев:
- солнце на sup→att или att→tgt → None
- угол `_angle_at_att(sup,att,tgt) > 120°` → None (крюк)
- если multi_sync даёт меньше ships при том же success → None (multi лучше)

Forward-rebase: если `att` фронтовее `sup` (ближе к не-нашим), `x_sup_send = sup.ships`
(всё что есть), чтобы att стал новой площадкой.

## Transfer (внутри-кооператив)

`code:swarm.py:redistribute` (по умолчанию выключен через `enable_redistribute=False`).

Передача ship'ов от P к нашей Q без атаки. Когда:
- у P есть остаток после auction (> `transfer_floor`),
- у Q высокий stress (см. [[70_swarm_decisions#stress]]),
- путь чист от солнца,
- ETA ≤ `transfer_horizon`.

```
score = max(0, stress[Q] − stress[P]) · ships / (1 + eta·eta_pen)
ships ≤ min(remaining[P] − floor, deficit_Q + buffer)
```

`mode='transfer'` ловит `_execute_plan_atomically` НЕ как direct → не делает
defender check (своя планета, нет защиты).

## Defense plans (`is_defense=True`)

`code:agent.py:_build_defense_plans`

Для каждой нашей **doomed** планеты (`raw=ours, projected=enemy`) ищем
стабильного отправителя с непрегораженным солнцем путём. Формат —
direct-like, но с `defender = projected.ships + production · delta`.

Выполняются ПЕРЕД attack_plans (priority на самосохранение).

## Combat resolution

`code:orbit_sim.py:_resolve_combat`

При одновременном прибытии нескольких флотов к одной планете:
- сортируем по `ships` desc;
- top сражается с #2: `diff = top − sec`, проигравший выбывает;
- повторяем пока не останется один владелец или 0 ships.

Strict-`<`: ничья → защитник выигрывает (важно для overkill).

#tag:действие #tag:атака
