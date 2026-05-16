#!/usr/bin/env python3
"""
tune_4p.py — тюнинг параметров агента для 4-player FFA.

═══════════════════════════════════════════════════════════════════════════════
ОТЛИЧИЯ FFA ОТ 1v1
═══════════════════════════════════════════════════════════════════════════════

  В 4-player режиме:
  - Три врага вместо одного → фронт со всех сторон
  - «Пусть дерутся» (LET_THEM_FIGHT): иногда выгоднее не лезть пока двое
    воюют между собой, а потом добрать ослабленного
  - Нейтралы оспариваются тремя игроками → раннее расширение критичнее
  - Гарнизон важнее: атаки с нескольких сторон одновременно
  - Пороги агрессии должны быть другими: не нужно убивать конкретного
    оппонента, достаточно набрать больше всех

  Параметры, которые скорее всего оптимальны по-другому:
  - neutral_garrison: в 1v1 выгоден 0, в FFA спорные нейтралы могут
    требовать буфера
  - prio_reclassify_thr: с тремя врагами «хорошая» периферия важнее
  - garrison_per_prod: фронты повсюду → может быть полезнее чем в 1v1
  - zone_urgency_weight: зональный градиент важнее при атаках с 3 сторон
  - distance_comfort: дальние цели менее выгодны (враг перехватит)
  - activity_weight: «застоявшиеся» планеты в тылу важнее выводить в бой

═══════════════════════════════════════════════════════════════════════════════
МЕТРИКИ
═══════════════════════════════════════════════════════════════════════════════
  rank:       средний ранг (1–4). 1 = лучший. Цель: < 2.5.
  winrate:    доля 1-х мест.
  top2_rate:  доля топ-2 мест (win + 2nd).
  sr_N:       наша доля кораблей от общего пула на ходу N (> 0.25 = лидируем).

Запуск:  python3 tuning/scripts/tune_4p.py
Вывод:   tuning/results/tune_4p_<ts>.csv + консольная сводка.
"""

import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

# ── Пути ─────────────────────────────────────────────────────────────────────
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
# Три одинаковых sub2 — стабильный FFA-бенчмарк
OPPS        = ['sub2', 'sub2', 'sub2']
N_WORKERS   = max(2, os.cpu_count() // 2)
CHECKPOINTS = [30, 100, 200]

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГИ
# weights4 → SWARM_WEIGHTS_4P (используется когда агент видит 4 игроков)
# weights  → SWARM_WEIGHTS (2-player; оставляем дефолты из sweep 1-2)
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS = [

    # ── Baseline: текущие дефолты (оптимальные для 1v1) применяются и в FFA ──
    # Используется как точка отсчёта — что будет если не настраивать отдельно.
    dict(name='baseline_2p_defaults',
         weights4={}),

    # ── Группа A: neutral_garrison в FFA ─────────────────────────────────────
    # Гипотеза: в 1v1 ng=0 лучший, но в FFA спорные нейтралы захватываются
    # тремя игроками → маленький буфер может помочь удержать
    dict(name='ng_3',   weights4=dict(neutral_garrison=3)),
    dict(name='ng_5',   weights4=dict(neutral_garrison=5)),
    dict(name='ng_8',   weights4=dict(neutral_garrison=8)),

    # ── Группа B: garrison_per_prod в FFA ────────────────────────────────────
    # В 1v1 gar_off = baseline. В FFA фронты со всех сторон →
    # проактивный гарнизон может реально помогать
    dict(name='gar_5',  weights4=dict(garrison_per_prod=5.0,  neutral_garrison=0)),
    dict(name='gar_8',  weights4=dict(garrison_per_prod=8.0,  neutral_garrison=0)),
    dict(name='gar_12', weights4=dict(garrison_per_prod=12.0, neutral_garrison=0)),

    # ── Группа C: prio_reclassify в FFA ──────────────────────────────────────
    # С тремя врагами ранняя экспансия критичнее → более агрессивный порог
    # (ниже = больше планет апгрейдируется в priority_target)
    dict(name='prio_0.4', weights4=dict(prio_reclassify_thr=0.4,
                                        prio_reclassify_thr_late=0.4)),
    dict(name='prio_0.6', weights4=dict(prio_reclassify_thr=0.6,
                                        prio_reclassify_thr_late=0.6)),
    dict(name='prio_1.2', weights4=dict(prio_reclassify_thr=1.2,
                                        prio_reclassify_thr_late=1.2)),

    # ── Группа D: zone_urgency_weight в FFA ──────────────────────────────────
    # Фронты с трёх сторон → сильный зональный градиент (rear→frontline)
    # может помочь больше чем в 1v1
    dict(name='urg_2.5', weights4=dict(zone_urgency_weight=2.5)),
    dict(name='urg_4.0', weights4=dict(zone_urgency_weight=4.0)),

    # ── Группа E: комбинации для FFA ─────────────────────────────────────────
    # Осторожная FFA: небольшой garrison + сильный зональный gradient
    dict(name='combo_defensive',
         weights4=dict(neutral_garrison=3, garrison_per_prod=5.0,
                       zone_urgency_weight=2.5, prio_reclassify_thr=0.8,
                       prio_reclassify_thr_late=0.8)),

    # Агрессивная FFA: низкий порог priority + большой garrison
    dict(name='combo_aggressive',
         weights4=dict(neutral_garrison=5, garrison_per_prod=8.0,
                       zone_urgency_weight=2.0, prio_reclassify_thr=0.4,
                       prio_reclassify_thr_late=0.8)),

    # Умеренная FFA: оптимистичный первый guess
    dict(name='combo_moderate',
         weights4=dict(neutral_garrison=3, garrison_per_prod=5.0,
                       zone_urgency_weight=2.0, prio_reclassify_thr=0.6,
                       prio_reclassify_thr_late=0.8)),
]


# ── Вспомогательные ──────────────────────────────────────────────────────────

def _stage_metrics(r):
    """Извлекает нашу долю кораблей (ships_ratio vs всего пула) на каждом чекпоинте."""
    # ts_ships_0 = наши корабли, нужно суммировать всех 4 для total
    ts0 = r.get('ts_ships_0') or []
    n   = len(ts0)
    # Суммируем все 4 игрока для total
    all_ts = [r.get(f'ts_ships_{i}') or [] for i in range(4)]

    out = {}
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
        winrate   =('win',   'mean'),
        top2_rate =('top2',  'mean'),
        avg_rank  =('rank',  'mean'),
        avg_steps =('steps', 'mean'),
        n         =('win',   'count'),
    )
    for c in avail_sr:
        agg_dict[c] = (c, 'mean')

    agg = (df.groupby('label').agg(**agg_dict)
             .reset_index()
             .sort_values('avg_rank'))   # сортировка по avg_rank (меньше = лучше)

    sep = '═' * 110

    print(f'\n{sep}')
    print('AVG RANK (↓ лучше)  +  ships_ratio на checkpoint\'ах  (sr > 0.25 = наша доля выше нормы)')
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
              f'1. {best["label"]:<28} ({best[col]:.3f})  …  '
              f'last: {worst["label"]:<28} ({worst[col]:.3f})')

    print(f'\n{"─"*80}')
    best_rank = agg.iloc[0]
    best_wr   = agg.sort_values('winrate', ascending=False).iloc[0]
    print(f'ЛУЧШИЙ по avg_rank: {best_rank["label"]}  '
          f'(rank={best_rank["avg_rank"]:.3f}  '
          f'winrate={best_rank["winrate"]:.3f}  '
          f'top2={best_rank["top2_rate"]:.3f})')
    if best_wr['label'] != best_rank['label']:
        print(f'ЛУЧШИЙ по winrate:  {best_wr["label"]}  '
              f'(rank={best_wr["avg_rank"]:.3f}  '
              f'winrate={best_wr["winrate"]:.3f}  '
              f'top2={best_wr["top2_rate"]:.3f})')
    print(sep)


# ── Главный запуск ────────────────────────────────────────────────────────────

def main():
    tasks = []
    for cfg in CONFIGS:
        name     = cfg['name']
        weights4 = {k: v for k, v in cfg.items() if k not in ('name', 'weights', 'weights4')}
        weights4.update(cfg.get('weights4', {}))
        weights  = cfg.get('weights', {})
        for s in EVAL_SEEDS:
            tasks.append({
                'seed':     s,
                'our_path': OUR_PATH,
                'opps':     OPPS,
                'weights':  weights,
                'weights4': weights4,
                'label':    name,
            })

    n_configs = len(CONFIGS)
    n_seeds   = len(EVAL_SEEDS)
    n_tasks   = len(tasks)
    print(f'Конфигов: {n_configs}  Сидов: {n_seeds}  '
          f'Задач: {n_tasks}  Воркеров: {N_WORKERS}  '
          f'Оппоненты: {OPPS}')
    print('\nКонфиги (только weights4, weights=дефолты):')
    for cfg in CONFIGS:
        w4 = cfg.get('weights4', {})
        print(f'  {cfg["name"]:<26}  {w4}')
    print()

    rows   = []
    errors = []
    t0     = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_match4, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r     = fut.result()
                stage = _stage_metrics(r)
                # Убираем timeseries (большие)
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
        print('Нет результатов — все задачи упали с ошибками.')
        return

    df = pd.DataFrame(rows)

    ts_str  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f'tune_4p_{ts_str}.csv')
    df.to_csv(out_csv, index=False)

    _print_summary(df)

    elapsed_total = time.time() - t0
    print(f'\nсохранено: {out_csv}')
    print(f'всего: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)  '
          f'ошибок: {len(errors)}')
    if errors:
        print('Ошибки:')
        for label, seed, msg in errors[:10]:
            print(f'  label={label} seed={seed}: {msg}')


if __name__ == '__main__':
    main()
