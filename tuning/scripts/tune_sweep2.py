#!/usr/bin/env python3
"""
tune_sweep2.py — второй тюнинг-проход на основе результатов tune_expansion.

═══════════════════════════════════════════════════════════════════════════════
БАЗОВЫЕ ВЫВОДЫ ИЗ SWEEP 1
═══════════════════════════════════════════════════════════════════════════════

  neutral_garrison=0   лучший (0.633 winrate vs 0.600 baseline).
                       Тратить корабли на гарнизон нейтрала = морить атаку.
                       Обновлён дефолт в SwarmWeights.

  garrison_per_prod=0  нейтрален (gar_off = baseline), gar_4/gar_12 вредят.
                       Проактивный гарнизон overrides attack budget напрасно.
                       Обновлён дефолт в SwarmWeights.

  prio_reclassify_thr  Большинство значений дают 0.600. Выключать нельзя (0.367).
                       prio_1.2 — неожиданный провал (0.300). Нужно разобраться.

═══════════════════════════════════════════════════════════════════════════════
ГИПОТЕЗЫ ЭТОГО ПРОГОНА
═══════════════════════════════════════════════════════════════════════════════

  Группа A — дотюниваем neutral_garrison вокруг 0:
    Вопрос: ng=0 — действительно минимум, или крошечный буфер (1-2) помогает
    выжить после захвата оспариваемой планеты до прихода redistribute?

  Группа B — диагностика prio_1.2:
    Вопрос: где ломается? Плавный скан 0.9→1.15 должен показать порог.
    Гипотеза: именно при 1.2 рекласcифицируется класс «средних» планет
    которые привлекательны на бумаге но реально слишком далеко/оспариваемы.

  Группа C — distance_comfort (нетронутый параметр):
    Текущий дефолт 0.0 = штраф 1/eta (сильно режет дальние цели).
    0.3-0.5 смягчает → агент видит больше дальних целей.
    Гипотеза: с ng_0 у нас больше кораблей в атаке → можем позволить дальше.

  Группа D — zone_urgency_weight (нетронутый параметр):
    Управляет силой зонального градиента в redistribute (rear → frontline).
    Вопрос: дефолт 1.5 оптимален? Слабее (0.5) = меньше «принудительного»
    перемещения. Сильнее (3.0) = агрессивнее гонит корабли на фронт.

  Группа E — transfer_min_ships (нетронутый параметр):
    Дефолт 15. Меньше (8) = гранулярнее, чаще маленькие подкрепления.
    Больше (25) = реже, но крупными пакетами.

  Группа F — комбинации (перспективные связки):
    Агрессивная переброска: urg_3.0 + xfer_8 → сильный зональный градиент
    + мелкие переброски = frontline всегда пополняется.
    Дальний захват: dist_0.5 + prio_0.4 → мягкий штраф за дистанцию
    + агрессивная reclassify = лезем за дальними приоритетными целями.

═══════════════════════════════════════════════════════════════════════════════
НОВЫЙ BASELINE
═══════════════════════════════════════════════════════════════════════════════
  neutral_garrison=0, garrison_per_prod=0, prio_reclassify_thr=0.8 (оба)
  Это текущие дефолты в SwarmWeights после обновления.

Запуск:  python3 tuning/scripts/tune_sweep2.py
Вывод:   tuning/results/tune_sweep2_<ts>.csv + консольная сводка.
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
from match_runner import run_match

# ══════════════════════════════════════════════════════════════════════════════
EVAL_SEEDS  = list(range(0, 30))
OPPONENT    = 'sub2'
N_WORKERS   = max(2, os.cpu_count() // 2)
CHECKPOINTS = [30, 100, 200]

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГИ
# Незаданные поля = новые дефолты SwarmWeights (neutral_garrison=0, garrison_per_prod=0)
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS = [

    # ── Новый baseline (дефолты после sweep 1) ────────────────────────────
    dict(name='new_baseline',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8),

    # ── Группа A: fine-grain neutral_garrison вокруг 0 ────────────────────
    # Вопрос: крошечный буфер помогает или нет?
    dict(name='ng_1',
         neutral_garrison=1,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8),

    dict(name='ng_3',
         neutral_garrison=3,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8),

    # ── Группа B: диагностика prio_1.2 ───────────────────────────────────
    dict(name='prio_0.9',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.9, prio_reclassify_thr_late=0.9),

    dict(name='prio_1.0',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=1.0, prio_reclassify_thr_late=1.0),

    dict(name='prio_1.1',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=1.1, prio_reclassify_thr_late=1.1),

    dict(name='prio_1.2',       # контрольная точка — должна повторить провал
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=1.2, prio_reclassify_thr_late=1.2),

    # ── Группа C: distance_comfort ────────────────────────────────────────
    # 0.0 = штраф 1/eta (дефолт), 0.5 = 1/sqrt(eta), 1.0 = плоско
    dict(name='dist_0.3',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8,
         distance_comfort=0.3),

    dict(name='dist_0.5',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8,
         distance_comfort=0.5),

    # ── Группа D: zone_urgency_weight ────────────────────────────────────
    dict(name='urg_0.5',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8,
         zone_urgency_weight=0.5),

    dict(name='urg_3.0',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8,
         zone_urgency_weight=3.0),

    # ── Группа E: transfer_min_ships ─────────────────────────────────────
    dict(name='xfer_8',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8,
         transfer_min_ships=8),

    dict(name='xfer_25',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8,
         transfer_min_ships=25),

    # ── Группа F: комбинации ──────────────────────────────────────────────
    # Агрессивная переброска на фронт: сильный зональный градиент + мелкие трансферы
    dict(name='combo_front',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.8, prio_reclassify_thr_late=0.8,
         zone_urgency_weight=3.0, transfer_min_ships=8),

    # Дальний захват: мягкий дистанционный штраф + агрессивная reclassify
    dict(name='combo_reach',
         neutral_garrison=0,    garrison_per_prod=0.0,
         prio_reclassify_thr=0.4, prio_reclassify_thr_late=0.8,
         distance_comfort=0.5),
]


# ── Вспомогательные (копия из tune_expansion.py) ─────────────────────────────

def _stage_metrics(r):
    ts0 = r.get('ts_ships0') or []
    ts1 = r.get('ts_ships1') or []
    tp0 = r.get('ts_prod0')  or []
    tp1 = r.get('ts_prod1')  or []
    n   = len(ts0)
    out = {}
    for t in CHECKPOINTS + ['final']:
        idx = (n - 1 if n > 0 else 0) if t == 'final' else (min(int(t), n - 1) if n > 0 else 0)
        s0 = ts0[idx] if ts0 else 0.0
        s1 = ts1[idx] if ts1 else 0.0
        p0 = tp0[idx] if tp0 else 0.0
        p1 = tp1[idx] if tp1 else 0.0
        tag = str(t)
        out[f'sr_{tag}'] = s0 / max(1.0, s0 + s1)
        out[f'pr_{tag}'] = p0 / max(1.0, p0 + p1)
    return out


def _print_summary(df):
    sr_cols   = [f'sr_{t}' for t in CHECKPOINTS + ['final']]
    pr_cols   = [f'pr_{t}' for t in CHECKPOINTS + ['final']]
    avail_sr  = [c for c in sr_cols if c in df.columns]
    avail_pr  = [c for c in pr_cols if c in df.columns]

    agg_dict = dict(
        winrate       =('win',        'mean'),
        draw_rate     =('draw',       'mean'),
        avg_steps     =('steps',      'mean'),
        avg_ships_diff=('ships_diff', 'mean'),
        n             =('win',        'count'),
    )
    for c in avail_sr + avail_pr:
        agg_dict[c] = (c, 'mean')

    agg = (df.groupby('label').agg(**agg_dict)
             .reset_index()
             .sort_values('winrate', ascending=False))

    sep = '═' * 115

    print(f'\n{sep}')
    print('WINRATE  +  ships_ratio на checkpoint\'ах  (sr > 0.5 = мы лидируем)')
    print(sep)
    disp_cols = ['label', 'winrate', 'draw_rate', 'avg_ships_diff', 'avg_steps'] + avail_sr
    disp = agg[disp_cols].copy()
    for c in disp.columns[1:]:
        disp[c] = disp[c].round(3)
    print(disp.to_string(index=False))

    if avail_pr:
        print(f'\n{"─"*80}')
        print('prod_ratio на checkpoint\'ах  (pr > 0.5 = наш прод выше)')
        print(f'{"─"*80}')
        pd_disp = agg[['label'] + avail_pr].copy()
        for c in avail_pr:
            pd_disp[c] = pd_disp[c].round(3)
        print(pd_disp.to_string(index=False))

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
    best_overall = agg.iloc[0]
    print(f'ЛУЧШИЙ по winrate:  {best_overall["label"]}  '
          f'(win={best_overall["winrate"]:.3f}  '
          f'sr_final={best_overall.get("sr_final", float("nan")):.3f})')
    print(sep)


# ── Главный запуск ────────────────────────────────────────────────────────────

def main():
    tasks = []
    for cfg in CONFIGS:
        name    = cfg['name']
        weights = {k: v for k, v in cfg.items() if k != 'name'}
        for s in EVAL_SEEDS:
            tasks.append({
                'seed':     s,
                'our_path': OUR_PATH,
                'opp':      OPPONENT,
                'weights':  weights,
                'label':    name,
            })

    n_configs = len(CONFIGS)
    n_seeds   = len(EVAL_SEEDS)
    n_tasks   = len(tasks)
    print(f'Конфигов: {n_configs}  Сидов: {n_seeds}  '
          f'Задач: {n_tasks}  Воркеров: {N_WORKERS}  Оппонент: {OPPONENT}')
    print('\nКонфиги:')
    for cfg in CONFIGS:
        w = {k: v for k, v in cfg.items() if k != 'name'}
        print(f'  {cfg["name"]:<22}  {w}')
    print()

    rows   = []
    errors = []
    t0     = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_match, t): t for t in tasks}
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
        print('Нет результатов — все задачи упали с ошибками.')
        return

    df = pd.DataFrame(rows)

    ts_str  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f'tune_sweep2_{ts_str}.csv')
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
