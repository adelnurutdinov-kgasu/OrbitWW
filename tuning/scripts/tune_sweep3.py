#!/usr/bin/env python3
"""
tune_sweep3.py — sweep по активности и агрессии в 2-player режиме.

═══════════════════════════════════════════════════════════════════════════════
ВЫВОДЫ ИЗ SWEEP 2
═══════════════════════════════════════════════════════════════════════════════

  new_baseline лучший: avg_rank=0.267, winrate=0.633
    neutral_garrison=0, garrison_per_prod=0, prio_reclassify_thr=0.8

  Наблюдения из Kaggle-эпизодов (3 проигранные 2p-игры):
    - Агент делает ходы (83/176, 40/119, 34/134 активных шагов) — не крашится.
    - Максимум idle ships без действия: 638, 301, 326 (огромные залежи!)
    - Соотношение launches: us=4422 vs opp=6899; us=1474 vs opp=12134 (в 8 раз!)
    - Оппонент захватывает больше планет с шагов 7-29 (ранняя экспансия).

  Диагноз: стратегическая пассивность.
    Корабли накапливаются на планетах без использования.
    Причины: idle_floor=25 (не считает корабли ниже 25 «лишними»),
    activity_weight=0.5 (слабый толчок к атаке), risk_tolerance=0 (консерватизм).

═══════════════════════════════════════════════════════════════════════════════
ГИПОТЕЗЫ SWEEP 3
═══════════════════════════════════════════════════════════════════════════════

  Группа A — activity_weight (текущий 0.5):
    Увеличение → сильнее штрафуем «застоявшиеся» планеты → больше атак.
    Тест: 0.5 (baseline), 1.0, 2.0, 3.0, 5.0

  Группа B — idle_floor (текущий 25):
    Снижение → больше кораблей считаются «лишними» → активнее используются.
    Тест: 25 (baseline), 15, 10, 5, 0

  Группа C — risk_tolerance (текущий 0):
    Повышение → атакуем «почти-проходные» планы (нужно на 1-2 меньше кораблей).
    Тест: 0 (baseline), 1, 2

  Группа D — opp_strength_weight для 2p (сейчас 0):
    В 1v1 один оппонент → rel_strength=1.0 всегда → бонус=0 постоянно.
    НО: значение может варьироваться если у него разные планеты с разной силой.
    Собственно, смысл такой же как в 4p — атаковать слабые планеты прежде
    сильных. Проверяем влияние: 0 (baseline), 0.3, 0.5, 1.0

  Группа E — комбинации (лучшие из A + B + C):
    act2+floor10: activity_weight=2.0, idle_floor=10
    act2+risk1:   activity_weight=2.0, risk_tolerance=1
    act1+floor15+risk1: умеренная комбинация
    act3+floor5:  максимальная активность

Запуск:  python3 tuning/scripts/tune_sweep3.py
Вывод:   tuning/results/tune_sweep3_<ts>.csv + консольная сводка.
"""

import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BUNDLE_DIR = os.path.join(PROJECT_ROOT, "agent_bundle_swarm 2")
OUR_PATH   = os.path.join(BUNDLE_DIR, "agent.py")

sys.path.insert(0, PROJECT_ROOT)
from match_runner import run_match

# ══════════════════════════════════════════════════════════════════════════════
EVAL_SEEDS  = list(range(0, 30))
OPPONENT    = 'sub2'
N_WORKERS   = max(2, os.cpu_count() // 2)
CHECKPOINTS = [30, 100, 200]

# Базис sweep 2: новые дефолты после sweep 1
_BASE = dict(
    neutral_garrison=0,
    garrison_per_prod=0.0,
    prio_reclassify_thr=0.8,
    prio_reclassify_thr_late=0.8,
)

# ══════════════════════════════════════════════════════════════════════════════
CONFIGS = [

    # ── Baseline (победитель sweep 2) ────────────────────────────────────
    dict(name='baseline',        **_BASE),

    # ── Группа A: activity_weight (текущий дефолт 0.5) ──────────────────
    dict(name='act_1.0',         **_BASE, activity_weight=1.0),
    dict(name='act_2.0',         **_BASE, activity_weight=2.0),
    dict(name='act_3.0',         **_BASE, activity_weight=3.0),
    dict(name='act_5.0',         **_BASE, activity_weight=5.0),

    # ── Группа B: idle_floor (текущий дефолт 25) ────────────────────────
    dict(name='floor_15',        **_BASE, idle_floor=15),
    dict(name='floor_10',        **_BASE, idle_floor=10),
    dict(name='floor_5',         **_BASE, idle_floor=5),
    dict(name='floor_0',         **_BASE, idle_floor=0),

    # ── Группа C: risk_tolerance (текущий дефолт 0) ─────────────────────
    dict(name='risk_1',          **_BASE, risk_tolerance=1),
    dict(name='risk_2',          **_BASE, risk_tolerance=2),

    # ── Группа D: opp_strength_weight для 2p (текущий 0.0) ──────────────
    dict(name='osw_0.3',         **_BASE, opp_strength_weight=0.3),
    dict(name='osw_0.5',         **_BASE, opp_strength_weight=0.5),
    dict(name='osw_1.0',         **_BASE, opp_strength_weight=1.0),

    # ── Группа E: комбинации ─────────────────────────────────────────────
    dict(name='act2+floor10',    **_BASE, activity_weight=2.0, idle_floor=10),
    dict(name='act2+risk1',      **_BASE, activity_weight=2.0, risk_tolerance=1),
    dict(name='act1+floor15+risk1', **_BASE,
         activity_weight=1.0, idle_floor=15, risk_tolerance=1),
    dict(name='act3+floor5',     **_BASE, activity_weight=3.0, idle_floor=5),
    dict(name='act2+floor10+risk1', **_BASE,
         activity_weight=2.0, idle_floor=10, risk_tolerance=1),
]


# ── Вспомогательные ──────────────────────────────────────────────────────────

def _stage_metrics(r):
    ts0 = r.get('ts_ships0') or []
    ts1 = r.get('ts_ships1') or []
    tp0 = r.get('ts_prod0')  or []
    n   = len(ts0)
    out = {}
    for t in CHECKPOINTS + ['final']:
        idx = (n - 1 if n > 0 else 0) if t == 'final' \
              else (min(int(t), n - 1) if n > 0 else 0)
        s0 = ts0[idx] if idx < len(ts0) else 0.0
        s1 = ts1[idx] if idx < len(ts1) else 0.0
        p0 = tp0[idx] if idx < len(tp0) else 0.0
        total = s0 + s1
        out[f'sr_{t}'] = s0 / max(1.0, total)
        out[f'pr_{t}'] = p0
    return out


def _print_summary(df):
    sr_cols  = [f'sr_{t}' for t in CHECKPOINTS + ['final']]
    avail_sr = [c for c in sr_cols if c in df.columns]

    agg_dict = dict(
        avg_rank  =('r0',  'mean'),
        winrate   =('win', 'mean'),
        avg_steps =('steps', 'mean'),
        n         =('win', 'count'),
    )
    for c in avail_sr:
        agg_dict[c] = (c, 'mean')

    agg = (df.groupby('label').agg(**agg_dict)
             .reset_index()
             .sort_values('avg_rank', ascending=False))

    sep = '═' * 110

    print(f'\n{sep}')
    print('AVG RANK (↑ лучше = r0 выше)  +  ships_ratio  (sr > 0.5 = опережаем)')
    print(sep)
    disp_cols = ['label', 'avg_rank', 'winrate', 'avg_steps'] + avail_sr
    disp = agg[disp_cols].copy()
    for c in disp.columns[1:]:
        disp[c] = disp[c].round(3)
    print(disp.to_string(index=False))

    print(f'\n{"─"*80}')
    print('RANK по ships_ratio на каждом checkpoint:')
    for col in avail_sr:
        ranked = agg[['label', col]].sort_values(col, ascending=False).reset_index(drop=True)
        best  = ranked.iloc[0]
        worst = ranked.iloc[-1]
        cp    = col.replace('sr_', 'T')
        print(f'  {cp:<8}  '
              f'1. {best["label"]:<26} ({best[col]:.3f})  …  '
              f'last: {worst["label"]:<26} ({worst[col]:.3f})')

    print(f'\n{"─"*80}')
    best = agg.iloc[0]
    best_wr = agg.sort_values('winrate', ascending=False).iloc[0]
    print(f'ЛУЧШИЙ по avg_rank: {best["label"]}  '
          f'(rank={best["avg_rank"]:.3f}  win={best["winrate"]:.3f})')
    if best_wr['label'] != best['label']:
        print(f'ЛУЧШИЙ по winrate:  {best_wr["label"]}  '
              f'(rank={best_wr["avg_rank"]:.3f}  win={best_wr["winrate"]:.3f})')
    print(sep)


def main():
    tasks = []
    for cfg in CONFIGS:
        name = cfg['name']
        weights = {k: v for k, v in cfg.items() if k != 'name'}
        for s in EVAL_SEEDS:
            tasks.append({
                'seed':     s,
                'our_path': OUR_PATH,
                'opp':      OPPONENT,
                'weights':  weights,
                'label':    name,
            })

    n_tasks = len(tasks)
    print(f'Конфигов: {len(CONFIGS)}  Сидов: {len(EVAL_SEEDS)}  '
          f'Задач: {n_tasks}  Воркеров: {N_WORKERS}')
    print(f'Оппонент: {OPPONENT}')
    print('\nКонфиги (overrides от baseline):')
    for cfg in CONFIGS:
        overrides = {k: v for k, v in cfg.items()
                     if k != 'name' and k not in _BASE}
        print(f'  {cfg["name"]:<26}  {overrides if overrides else "(baseline)"}')
    print()

    rows, errors = [], []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_match, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r     = fut.result()
                stage = _stage_metrics(r)
                row   = {k: v for k, v in r.items()
                         if not k.startswith('ts_')}
                row.update(stage)
                rows.append(row)
            except Exception as e:
                t = futs[fut]
                errors.append((t['label'], t['seed'], str(e)))
                print(f'  ERROR  label={t["label"]}  seed={t["seed"]}: {e}',
                      file=sys.stderr)

            if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                elapsed = time.time() - t0
                eta_min = elapsed / done * (n_tasks - done) / 60
                print(f'  [{done:>4}/{n_tasks}]  '
                      f'elapsed={elapsed:>5.0f}s  eta={eta_min:>4.0f}min  '
                      f'errors={len(errors)}')

    if not rows:
        print('Нет результатов.')
        return

    df = pd.DataFrame(rows)
    ts_str  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f'tune_sweep3_{ts_str}.csv')
    df.to_csv(out_csv, index=False)

    _print_summary(df)

    elapsed_total = time.time() - t0
    print(f'\nсохранено: {out_csv}')
    print(f'всего: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)  '
          f'ошибок: {len(errors)}')
    if errors:
        for label, seed, msg in errors[:10]:
            print(f'  label={label} seed={seed}: {msg}')


if __name__ == '__main__':
    main()
