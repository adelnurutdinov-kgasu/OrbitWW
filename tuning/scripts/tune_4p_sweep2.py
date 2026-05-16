#!/usr/bin/env python3
"""
tune_4p_sweep2.py — второй тюнинг-проход для 4-player FFA.

═══════════════════════════════════════════════════════════════════════════════
ВЫВОДЫ ИЗ SWEEP 1
═══════════════════════════════════════════════════════════════════════════════

  urg_4.0  — явный победитель: avg_rank=1.500, winrate=0.500, top2=1.000
             (никогда ниже 2-го места из 4). Сильный зональный градиент
             в FFA критичен: rear→frontline при атаках с трёх сторон.

  ng_3     — лучший neutral_garrison для FFA (vs ng_0 в 1v1).
             Спорные нейтралы захватываются тремя → маленький буфер помогает.

  gar_8    — проактивный гарнизон работает в FFA (в 1v1 не работал).
             Три фронта = больше входящих атак = garrison нужен.

  prio_1.2 — неожиданно хорош в FFA (в 1v1 плохой).
             Высокий порог = избирательность против трёх врагов.

  Комбо провалились в sweep 1 — тестировались сразу 3-4 параметра,
  интерференция сломала баланс. В sweep 2 комбинируем аккуратно,
  по одному параметру к лучшему базису (urg_4.0).

═══════════════════════════════════════════════════════════════════════════════
ИЗМЕНЕНИЕ ОППОНЕНТОВ
═══════════════════════════════════════════════════════════════════════════════

  Sweep 1 использовал 3x sub2 — слишком однородно.
  Sweep 2: ['sub2', 'sub2', 'opp_heuristic'] — два сильных + один слабый.
  Это реалистичнее: в реальном матче оппоненты разного уровня.
  Слабый оппонент создаёт «приз» — его можно быстро устранить и получить
  ресурсы, но это отвлекает от сильных противников.

═══════════════════════════════════════════════════════════════════════════════
ГИПОТЕЗЫ
═══════════════════════════════════════════════════════════════════════════════

  Группа A — подтверждение urg_4.0 с новым составом оппонентов
  Группа B — комбинации urg_4.0 с лучшими параметрами из sweep 1
  Группа C — поиск потолка zone_urgency_weight (5.0, 6.0)
  Группа D — тонкая настройка transfer + activity в FFA-контексте

Запуск:  python3 tuning/scripts/tune_4p_sweep2.py
Вывод:   tuning/results/tune_4p_sweep2_<ts>.csv + консольная сводка.
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
from match_runner4 import run_match4

# ══════════════════════════════════════════════════════════════════════════════
EVAL_SEEDS  = list(range(0, 30))
# 2 сильных (sub2) + 1 слабый (opp_heuristic) — реалистичный FFA
OPPS        = ['sub2', 'sub2', 'opp_heuristic']
N_WORKERS   = max(2, os.cpu_count() // 2)
CHECKPOINTS = [30, 100, 200]

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГИ
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS = [

    # ── Baseline (дефолты 2p — без FFA-специфики) ─────────────────────────
    dict(name='baseline',
         weights4={}),

    # ── Группа A: подтверждение победителей sweep 1 с новым составом ──────
    dict(name='urg_4.0',
         weights4=dict(zone_urgency_weight=4.0)),

    dict(name='ng_3',
         weights4=dict(neutral_garrison=3)),

    dict(name='gar_8',
         weights4=dict(garrison_per_prod=8.0)),

    dict(name='prio_1.2',
         weights4=dict(prio_reclassify_thr=1.2, prio_reclassify_thr_late=1.2)),

    # ── Группа B: комбинации на базе urg_4.0 (аккуратно, по одному) ───────
    dict(name='urg4+ng3',
         weights4=dict(zone_urgency_weight=4.0, neutral_garrison=3)),

    dict(name='urg4+gar8',
         weights4=dict(zone_urgency_weight=4.0, garrison_per_prod=8.0)),

    dict(name='urg4+prio1.2',
         weights4=dict(zone_urgency_weight=4.0,
                       prio_reclassify_thr=1.2, prio_reclassify_thr_late=1.2)),

    dict(name='urg4+ng3+gar8',
         weights4=dict(zone_urgency_weight=4.0, neutral_garrison=3,
                       garrison_per_prod=8.0)),

    dict(name='urg4+ng3+gar8+p1.2',
         weights4=dict(zone_urgency_weight=4.0, neutral_garrison=3,
                       garrison_per_prod=8.0,
                       prio_reclassify_thr=1.2, prio_reclassify_thr_late=1.2)),

    # ── Группа C: потолок zone_urgency_weight ─────────────────────────────
    # Вопрос: urg_4.0 уже потолок или можно выше?
    dict(name='urg_5.0',
         weights4=dict(zone_urgency_weight=5.0)),

    dict(name='urg_6.0',
         weights4=dict(zone_urgency_weight=6.0)),

    dict(name='urg_3.0',   # контрольная точка sweep 1
         weights4=dict(zone_urgency_weight=3.0)),

    # ── Группа D: transfer + activity в FFA ───────────────────────────────
    # В FFA мелкие трансферы (xfer_8) помогали в 1v1. Проверим в FFA + urg_4.0
    dict(name='urg4+xfer8',
         weights4=dict(zone_urgency_weight=4.0, transfer_min_ships=8)),

    # idle_floor: более агрессивный вывод «застоявшихся» тыловых кораблей в бой
    dict(name='urg4+idle15',
         weights4=dict(zone_urgency_weight=4.0, idle_floor=15)),
]


# ── Вспомогательные ──────────────────────────────────────────────────────────

def _stage_metrics(r):
    ts0    = r.get('ts_ships_0') or []
    all_ts = [r.get(f'ts_ships_{i}') or [] for i in range(4)]
    n      = len(ts0)
    out    = {}
    for t in CHECKPOINTS + ['final']:
        idx = (n - 1 if n > 0 else 0) if t == 'final' else (min(int(t), n - 1) if n > 0 else 0)
        s0    = ts0[idx] if ts0 else 0.0
        total = sum((ts[idx] if idx < len(ts) else 0.0) for ts in all_ts)
        out[f'sr_{t}'] = s0 / max(1.0, total)
    return out


def _print_summary(df):
    sr_cols  = [f'sr_{t}' for t in CHECKPOINTS + ['final']]
    avail_sr = [c for c in sr_cols if c in df.columns]

    agg_dict = dict(
        avg_rank  =('rank',  'mean'),
        winrate   =('win',   'mean'),
        top2_rate =('top2',  'mean'),
        avg_steps =('steps', 'mean'),
        n         =('win',   'count'),
    )
    for c in avail_sr:
        agg_dict[c] = (c, 'mean')

    agg = (df.groupby('label').agg(**agg_dict)
             .reset_index()
             .sort_values('avg_rank'))

    sep = '═' * 115

    print(f'\n{sep}')
    print('AVG RANK (↓ лучше)  +  ships_ratio  (sr > 0.25 = выше нормы в 4-player)')
    print(sep)
    disp_cols = ['label', 'avg_rank', 'winrate', 'top2_rate', 'avg_steps'] + avail_sr
    disp = agg[disp_cols].copy()
    for c in disp.columns[1:]:
        disp[c] = disp[c].round(3)
    print(disp.to_string(index=False))

    print(f'\n{"─"*80}')
    print('RANK по ships_ratio на каждом checkpoint\'е:')
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
          f'(rank={best["avg_rank"]:.3f}  win={best["winrate"]:.3f}  top2={best["top2_rate"]:.3f})')
    if best_wr['label'] != best['label']:
        print(f'ЛУЧШИЙ по winrate:  {best_wr["label"]}  '
              f'(rank={best_wr["avg_rank"]:.3f}  win={best_wr["winrate"]:.3f}  top2={best_wr["top2_rate"]:.3f})')
    print(sep)


def main():
    tasks = []
    for cfg in CONFIGS:
        name     = cfg['name']
        weights4 = cfg.get('weights4', {})
        weights  = cfg.get('weights',  {})
        for s in EVAL_SEEDS:
            tasks.append({
                'seed':     s,
                'our_path': OUR_PATH,
                'opps':     OPPS,
                'weights':  weights,
                'weights4': weights4,
                'label':    name,
            })

    n_tasks = len(tasks)
    print(f'Конфигов: {len(CONFIGS)}  Сидов: {len(EVAL_SEEDS)}  '
          f'Задач: {n_tasks}  Воркеров: {N_WORKERS}')
    print(f'Оппоненты: {OPPS}')
    print('\nКонфиги (weights4):')
    for cfg in CONFIGS:
        print(f'  {cfg["name"]:<26}  {cfg.get("weights4", {})}')
    print()

    rows, errors = [], []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_match4, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r     = fut.result()
                stage = _stage_metrics(r)
                row   = {k: v for k, v in r.items() if not k.startswith('ts_')}
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
    out_csv = os.path.join(RESULTS_DIR, f'tune_4p_sweep2_{ts_str}.csv')
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
