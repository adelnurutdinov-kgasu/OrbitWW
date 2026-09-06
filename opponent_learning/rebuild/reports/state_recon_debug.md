# State Reconstruction Debug

**Проверок**: 400 launches

## Главные числа

| Metric | Value |
|---|---|
| Planets perfect | 293/400 (73.2%) |
| Fleets count perfect | 297/400 (74.2%) |
| State fully perfect | 293/400 |

## По числу launches на шаге

|   n_total_launches |   n |   fleet_ok |   planets_ok |
|-------------------:|----:|-----------:|-------------:|
|                  0 |  21 |      0.905 |        0.905 |
|                  1 |  45 |      0.933 |        0.933 |
|                  2 |  56 |      0.875 |        0.875 |
|                  3 |  55 |      0.855 |        0.836 |
|                  4 |  59 |      0.763 |        0.746 |
|                  5 |  36 |      0.556 |        0.556 |
|                  6 |  22 |      0.682 |        0.682 |
|                  7 |  20 |      0.55  |        0.55  |
|                  8 |  19 |      0.684 |        0.684 |
|                  9 |   8 |      0.5   |        0.5   |
|                 10 |   7 |      0.714 |        0.571 |
|                 12 |   5 |      0.6   |        0.6   |
|                 13 |   5 |      0.2   |        0.2   |
|                 14 |   2 |      0.5   |        0.5   |
|                 15 |   4 |      0.75  |        0.75  |

## По n_players

|   n_players |   n |   fleet_ok |   planets_ok |
|------------:|----:|-----------:|-------------:|
|           2 | 132 |      0.78  |        0.78  |
|           4 | 268 |      0.724 |        0.709 |

## Примеры failure cases


### Example 1: 76276130 step=211

- n_players: 4
- launches на шаге: 6 (2 активных игроков)
- planets diff: 2/28
- fleets pred/real: 31/33

Planet diffs:
```
  p 15: pred owner=1 ships=389 | real owner=1 ships=377
  p 23 ORBITAL: pred owner=1 ships=167 | real owner=1 ships=155
```

### Example 2: 76293330 step=93

- n_players: 4
- launches на шаге: 4 (1 активных игроков)
- planets diff: 2/24
- fleets pred/real: 16/14

Planet diffs:
```
  p  2: pred owner=3 ships=31 | real owner=3 ships=100
  p 18: pred owner=3 ships=1 | real owner=3 ships=67
```

### Example 3: 75947221 step=253

- n_players: 4
- launches на шаге: 1 (1 активных игроков)
- planets diff: 2/28
- fleets pred/real: 25/27

Planet diffs:
```
  p 13: pred owner=3 ships=9 | real owner=3 ships=42
  p 23: pred owner=2 ships=10 | real owner=2 ships=14
```

### Example 4: 75925168 step=68

- n_players: 4
- launches на шаге: 2 (2 активных игроков)
- planets diff: 1/32
- fleets pred/real: 24/25

Planet diffs:
```
  p 21: pred owner=3 ships=49 | real owner=3 ships=33
```

### Example 5: 76744055 step=70

- n_players: 4
- launches на шаге: 4 (2 активных игроков)
- planets diff: 1/36
- fleets pred/real: 33/34

Planet diffs:
```
  p 20: pred owner=1 ships=23 | real owner=1 ships=3
```
