#!/usr/bin/env python3
"""
tune_4p_sweep4.py — точная настройка вокруг pure_osw0.5.

═══════════════════════════════════════════════════════════════════════════════
ВЫВОДЫ ИЗ SWEEP 3
═══════════════════════════════════════════════════════════════════════════════

  pure_osw0.5 — явный победитель: avg_rank=1.600, winrate=0.400, top2=1.000
    config: zone_urgency_weight=4.0, opp_strength_weight=0.5
    (без ng3, без gar8!)

  Ключевые инсайты:
  1. opp_strength помогает — но только БЕЗ garrison бюджетного давления.
     osw_0.5 + ng3+gar8 = 1.867 (хуже baseline!), а pure_osw0.5 = 1.600.
  2. opp_strength_weight=1.0 с базисом = 1.633 (2-е место).
     Более сильный сигнал частично пробивает garrison-интерференцию.
  3. sr-тренд у pure_osw0.5 единственный растёт весь матч:
     T30=0.29 → T100=0.31 → T200=0.38 → final=0.40.
     Значит: opp-targeting помогает в мид-лейте когда разрыв в силах растёт.
  4. prod_factor (2/3/8) при urg4+ng3+gar8 не помог — интерференция.
     В sweep 4 тестируем prod_factor с чистым базисом (urg4+osw).

═══════════════════════════════════════════════════════════════════════════════
ГИПОТЕЗЫ SWEEP 4
═══════════════════════════════════════════════════════════════════════════════

  Группа A — fine-tune opp_strength_weight (0.35, 0.5, 0.65, 0.8, 1.0)
              с чистым базисом zone_urgency_weight=4.0

  Группа B — opp_prod_factor с лучшим W из группы A (чистый базис)
              prod_factor=2.0, 3.0, 5.0, 8.0, 12.0

  Группа C — zone_urgency_weight + osw (4.0 vs 3.0 vs 5.0 с osw=0.5)
              Вопрос: urg4 всё ещё оптимальный при наличии opp_strength?

  Группа D — слабые добавки: может небольшой ng (ng_2) или маленький gar (gar_4)
              помогут поверх pure_osw0.5?

Запуск:  python3 tuning/scripts/tune_4p_sweep4.py
Вывод:   tuning/results/tune_4p_sweep4_<ts>.csv + консольная сводка.
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
OPPS        = ['sub2', 'sub2', 'opp_heuristic']
N_WORKERS   = max(2, os.cpu_count() // 2)
CHECKPOINTS = [30, 100, 200]

# Чистый базис (победитель sweep 3)
_URG4 = dict(zone_urgency_weight=4.0)

# ══════════════════════════════════════════════════════════════════════════════
CONFIGS = [

    # ── Reference: pure_osw0.5 из sweep 3 ────────────────────────────────
    dict(name='pure_osw0.5',
         weights4=dict(zone_urgency_weight=4.0, opp_strength_weight=0.5)),

    # ── Группа A: fine-tune opp_strength_weight ───────────────────────────
    dict(name='osw_0.35',
         weights4=dict(**_URG4, opp_strength_weight=0.35)),

    dict(name='osw_0.65',
         weights4=dict(**_URG4, opp_strength_weight=0.65)),

    dict(name='osw_0.80',
         weights4=dict(**_URG4, opp_strength_weight=0.80)),

    dict(name='osw_1.0',
         weights4=dict(**_URG4, opp_strength_weight=1.0)),

    dict(name='osw_1.5',
         weights4=dict(**_URG4, opp_strength_weight=1.5)),

    # ── Группа B: opp_prod_factor с чистым базисом + osw=0.5 ─────────────
    # sweep 3 тестировал pf только с urg4+ng3+gar8 — интерференция мешала.
    # Теперь чистый базис.
    dict(name='osw0.5_pf2',
         weights4=dict(**_URG4, opp_strength_weight=0.5, opp_prod_factor=2.0)),

    dict(name='osw0.5_pf3',
         weights4=dict(**_URG4, opp_strength_weight=0.5, opp_prod_factor=3.0)),

    dict(name='osw0.5_pf8',
         weights4=dict(**_URG4, opp_strength_weight=0.5, opp_prod_factor=8.0)),

    dict(name='osw0.5_pf12',
         weights4=dict(**_URG4, opp_strength_weight=0.5, opp_prod_factor=12.0)),

    # ── Группа C: zone_urgency_weight × opp_strength ─────────────────────
    dict(name='urg3+osw0.5',
         weights4=dict(zone_urgency_weight=3.0, opp_strength_weight=0.5)),

    dict(name='urg5+osw0.5',
         weights4=dict(zone_urgency_weight=5.0, opp_strength_weight=0.5)),

    dict(name='urg3+osw1.0',
         weights4=dict(zone_urgency_weight=3.0, opp_strength_weight=1.0)),

    dict(name='urg5+osw1.0',
         weights4=dict(zone_urgency_weight=5.0, opp_strength_weight=1.0)),

    # ── Группа D: минимальные добавки поверх pure_osw0.5 ─────────────────
    # Вопрос: помогает ли маленький гарнизон / нейтральный буфер?
    dict(name='osw0.5+ng2',
         weights4=dict(**_URG4, opp_strength_weight=0.5, neutral_garrison=2)),

    dict(name='osw0.5+gar4',
         weights4=dict(**_URG4, opp_strength_weight=0.5, garrison_per_prod=4.0)),

    dict(name='osw0.5+ng2+gar4',
         weights4=dict(**_URG4, opp_strength_weight=0.5,
                       neutral_garrison=2, garrison_per_prod=4.0)),
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
    out_csv = os.path.join(RESULTS_DIR, f'tune_4p_sweep4_{ts_str}.csv')
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
