#!/usr/bin/env python3
"""
tune_expansion.py — тюнинг параметров экспансии на фиксированных сидах.

═══════════════════════════════════════════════════════════════════════════════
ПАРАМЕТРЫ ДЛЯ ТЮНИНГА
═══════════════════════════════════════════════════════════════════════════════

  prio_reclassify_thr       Порог priority-override: планеты 'periphery'/'hard_far'
                            с priority ≥ thr → 'priority_target' (попадают в аукцион).
                            Низкий (0.4) = агрессивная экспансия, много целей.
                            Высокий (1.5) = консервативно, только явные jackpot.
                            99.0 = полностью выключить feature.

  prio_reclassify_thr_late  Тот же порог для конца матча. agent.py линейно
                            интерполирует thr → thr_late по ходу (step/TOTAL_STEPS).
                            Пример: thr=0.4, thr_late=1.5 → ранняя игра = жадная
                            экспансия, поздняя = фокус только на top-targets.

  garrison_per_prod         Целевой гарнизон frontline/contested = production × N.
                            Redistribute отправляет туда подкрепления до этого уровня.
                            0 = выключить, 7 = дефолт, 12 = агрессивная защита.

  neutral_garrison          Бонус-корабли при захвате нейтрала (сверх минимума).
                            0 = только минимум (прилетаем с 1 кораблём), 5 = дефолт.

═══════════════════════════════════════════════════════════════════════════════
МЕТРИКИ
═══════════════════════════════════════════════════════════════════════════════

  winrate           Финальный winrate по всем сидам.
  ships_r_tN        ships_ratio = наши / (наши + враг) на ходу N.
                    > 0.5 = мы лидируем. Позволяет видеть ранний / средний /
                    поздний профиль конфига.
  prod_r_tN         Аналогично по производству.

  Checkpoint'ы: T30 (ранняя экспансия), T100 (середина), T200 (поздняя),
                final (последний ход).

═══════════════════════════════════════════════════════════════════════════════
ДЕТЕРМИНИРОВАННОСТЬ
═══════════════════════════════════════════════════════════════════════════════

  Сиды передаются явно в make("orbit_wars", configuration={"seed": N}).
  Карты детерминированы независимо от перезапуска Python.
  Все задачи идут в ОДНОМ ProcessPoolExecutor → нет регенерации между конфигами.
  Конфиг A и конфиг B видят ровно одну и ту же начальную расстановку.

Запуск:  python3 tuning/scripts/tune_expansion.py
Вывод:   tuning/results/tune_expansion_<ts>.csv + консольная сводка.
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
# КОНФИГ ЗАПУСКА — меняй здесь
# ══════════════════════════════════════════════════════════════════════════════
EVAL_SEEDS  = list(range(0, 30))       # фиксированные сиды (не менять между запусками!)
OPPONENT    = 'sub2'                   # 'sub2' или 'noop'
N_WORKERS   = max(2, os.cpu_count() // 2)
CHECKPOINTS = [30, 100, 200]           # ходы для поэтапного анализа + 'final' всегда

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГИ ДЛЯ СРАВНЕНИЯ
#
# Ключ 'name' — метка в таблице.
# Остальные ключи → override для SwarmWeights в match_runner.run_match.
# Незаданные поля = значения DEFAULT_WEIGHTS (из swarm.py).
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS = [
    # ── Базовый (текущие дефолты) ──────────────────────────────────────────
    dict(name='baseline',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=0.8,
         garrison_per_prod=7.0,    neutral_garrison=5),

    # ── Вариации prio_reclassify (статичный порог) ─────────────────────────
    dict(name='prio_0.4',
         prio_reclassify_thr=0.4,  prio_reclassify_thr_late=0.4,
         garrison_per_prod=7.0,    neutral_garrison=5),

    dict(name='prio_1.2',
         prio_reclassify_thr=1.2,  prio_reclassify_thr_late=1.2,
         garrison_per_prod=7.0,    neutral_garrison=5),

    dict(name='prio_1.8',
         prio_reclassify_thr=1.8,  prio_reclassify_thr_late=1.8,
         garrison_per_prod=7.0,    neutral_garrison=5),

    dict(name='prio_off',
         prio_reclassify_thr=99.,  prio_reclassify_thr_late=99.,
         garrison_per_prod=7.0,    neutral_garrison=5),

    # ── Stage-aware: агрессивный старт → осторожная поздняя игра ──────────
    dict(name='stage_0.4→1.5',
         prio_reclassify_thr=0.4,  prio_reclassify_thr_late=1.5,
         garrison_per_prod=7.0,    neutral_garrison=5),

    dict(name='stage_0.6→1.2',
         prio_reclassify_thr=0.6,  prio_reclassify_thr_late=1.2,
         garrison_per_prod=7.0,    neutral_garrison=5),

    dict(name='stage_0.8→1.5',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=1.5,
         garrison_per_prod=7.0,    neutral_garrison=5),

    # ── Вариации garrison_per_prod ─────────────────────────────────────────
    dict(name='gar_off',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=0.8,
         garrison_per_prod=0.0,    neutral_garrison=5),

    dict(name='gar_4',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=0.8,
         garrison_per_prod=4.0,    neutral_garrison=5),

    dict(name='gar_12',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=0.8,
         garrison_per_prod=12.0,   neutral_garrison=5),

    # ── Вариации neutral_garrison ──────────────────────────────────────────
    dict(name='ng_0',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=0.8,
         garrison_per_prod=7.0,    neutral_garrison=0),

    dict(name='ng_8',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=0.8,
         garrison_per_prod=7.0,    neutral_garrison=8),

    dict(name='ng_12',
         prio_reclassify_thr=0.8,  prio_reclassify_thr_late=0.8,
         garrison_per_prod=7.0,    neutral_garrison=12),
]


# ── Вспомогательные ──────────────────────────────────────────────────────────

def _stage_metrics(r):
    """Извлекает ships_ratio и prod_ratio на каждом checkpoint'е из timeseries."""
    ts0 = r.get('ts_ships0') or []
    ts1 = r.get('ts_ships1') or []
    tp0 = r.get('ts_prod0')  or []
    tp1 = r.get('ts_prod1')  or []
    n   = len(ts0)

    out = {}
    for t in CHECKPOINTS + ['final']:
        if t == 'final':
            idx = n - 1 if n > 0 else 0
        else:
            idx = min(int(t), n - 1) if n > 0 else 0

        s0 = ts0[idx] if ts0 else 0.0
        s1 = ts1[idx] if ts1 else 0.0
        p0 = tp0[idx] if tp0 else 0.0
        p1 = tp1[idx] if tp1 else 0.0
        tag = str(t)
        out[f'sr_{tag}'] = s0 / max(1.0, s0 + s1)   # ships_ratio
        out[f'pr_{tag}'] = p0 / max(1.0, p0 + p1)   # prod_ratio
    return out


def _print_summary(df):
    """Выводит сравнительную сводку конфигов."""
    sr_cols = [f'sr_{t}' for t in CHECKPOINTS + ['final']]
    pr_cols = [f'pr_{t}' for t in CHECKPOINTS + ['final']]

    avail_sr = [c for c in sr_cols if c in df.columns]
    avail_pr = [c for c in pr_cols if c in df.columns]

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

    sep = '═' * 110

    # ── Winrate + ships_ratio ──────────────────────────────────────────────
    print(f'\n{sep}')
    print('WINRATE  +  ships_ratio на checkpoint\'ах  (sr > 0.5 = мы лидируем)')
    print(sep)
    disp_cols = ['label', 'winrate', 'draw_rate', 'avg_ships_diff', 'avg_steps'] + avail_sr
    disp = agg[disp_cols].copy()
    for c in disp.columns[1:]:
        disp[c] = disp[c].round(3)
    # Отметить лидера в каждой колонке
    print(disp.to_string(index=False))

    # ── Prod ratio ────────────────────────────────────────────────────────
    if avail_pr:
        print(f'\n{"─"*80}')
        print('prod_ratio на checkpoint\'ах  (pr > 0.5 = наш прод выше)')
        print(f'{"─"*80}')
        pd_disp = agg[['label'] + avail_pr].copy()
        for c in avail_pr:
            pd_disp[c] = pd_disp[c].round(3)
        print(pd_disp.to_string(index=False))

    # ── Rank-сводка по каждому checkpoint'у ───────────────────────────────
    print(f'\n{"─"*80}')
    print('RANK по ships_ratio на каждом checkpoint\'е:')
    for col in avail_sr:
        ranked = agg[['label', col]].sort_values(col, ascending=False).reset_index(drop=True)
        best  = ranked.iloc[0]
        worst = ranked.iloc[-1]
        cp    = col.replace('sr_', 'T')
        print(f'  {cp:<8}  '
              f'1. {best["label"]:<22} ({best[col]:.3f})  …  '
              f'last: {worst["label"]:<22} ({worst[col]:.3f})')

    # ── Кандидат для следующего шага ──────────────────────────────────────
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
        print(f'  {cfg["name"]:<26}  {w}')
    print()

    rows   = []
    errors = []
    t0     = time.time()

    # ── ОДИН пул для ВСЕХ задач ──────────────────────────────────────────
    # ProcessPoolExecutor создаётся один раз — среда не перезапускается.
    # Сиды передаются явно → карты детерминированы независимо от процесса.
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_match, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r      = fut.result()
                stage  = _stage_metrics(r)
                # Сохраняем всё кроме timeseries (они большие, нужны только для stage)
                row    = {k: v for k, v in r.items() if not k.startswith('ts_')}
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

    # ── Сохранить CSV ────────────────────────────────────────────────────
    ts_str  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f'tune_expansion_{ts_str}.csv')
    df.to_csv(out_csv, index=False)

    # ── Консольная сводка ────────────────────────────────────────────────
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
