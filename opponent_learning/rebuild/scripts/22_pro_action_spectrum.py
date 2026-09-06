#!/usr/bin/env python3
"""
22_pro_action_spectrum.py — изучение «спектра действий» сильных игроков по сырым реплеям.

Зачем
─────
Перед редизайном генератора кандидатов нам нужно эмпирически понять, КАКИЕ
действия вообще совершают профи: куда стреляют (нейтрал/враг/своя=reinforce),
на какую дистанцию и на какой «ранг близости», сколько кораблей шлют относительно
источника и обороны цели, сколько запусков за ход, бьют ли по кометам.

Это фундамент для метрики recall кандидатов: candidate-generator обязан как
минимум содержать ВСЕ реальные действия профи. Распределения отсюда говорят,
какие отсечки (горизонт, дистанция, ранг) безопасны, а какие убивают recall.

Что делает скрипт
─────────────────
  • стримит .json реплеи из --corpus (формат Kaggle orbit_wars),
  • определяет «pro»-места (по умолчанию — победители; можно по именам команд),
  • для каждого запуска профи резолвит цель (ближайшая по направлению, с
    остатком угла как прокси шума метки),
  • пишет per-launch CSV (одна строка = один запуск) — основа для downstream,
  • печатает/сохраняет summary с перцентилями и гистограммами.

Замечания по данным (подтверждены на реальных реплеях)
─────────────────────────────────────────────────────
  • Позиции берём из ЖИВОГО obs['planets'] (он всегда синхронизирован между
    игроками). initial_planets НЕ используем (известный рассинхрон у комет).
  • player_idx берём из позиции в steps[si], а НЕ из obs['player'] (на 4p step0
    у всех player:0). step берём из индекса si.
  • Действие — это (src_id, angle, ships); цель в данных не записана, поэтому
    резолвится. Резолв вынесен в resolve_target() — позже можно заменить на
    lead-aware решатель из shooting.py агента.

Запуск
──────
  python3 22_pro_action_spectrum.py --corpus /path/to/archive0 --out ./spectrum_out
  python3 22_pro_action_spectrum.py --corpus ... --limit 100            # быстрый прогон
  python3 22_pro_action_spectrum.py --corpus ... --pros winners         # по умолчанию
  python3 22_pro_action_spectrum.py --corpus ... --pros all             # все игроки
  python3 22_pro_action_spectrum.py --corpus ... --pros "teams:Isaiah @ Tufa Labs,TonyK"
  python3 22_pro_action_spectrum.py --corpus ... --workers 8            # параллельно
"""

import os
import sys
import csv
import json
import math
import argparse
import statistics as st
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

NEUTRAL_OWNER = -1


# ═══════════════════════════════════════════════════════════════════════
# Парсинг реплея
# ═══════════════════════════════════════════════════════════════════════

def _planet_map(planets_raw):
    """obs['planets'] → {id: {...}} из ЖИВОГО состояния (не initial_planets)."""
    pm = {}
    for p in planets_raw:
        try:
            pm[int(p[0])] = {
                'id': int(p[0]), 'owner': int(p[1]),
                'x': float(p[2]), 'y': float(p[3]),
                'ships': float(p[5]), 'prod': float(p[6]),
            }
        except (TypeError, IndexError, ValueError):
            continue
    return pm


def resolve_target(src, angle, pm):
    """
    Резолв цели запуска (src, angle) → (target_dict, angle_residual_rad).

    v1: ближайшая планета по направлению (без упреждения на движение).
    angle_residual — |Δ| между направлением на цель и фактическим углом флота;
    его распределение оценивает шум метки и задаёт толеранс для candidate-recall.
    """
    sx, sy = src['x'], src['y']
    best, bres = None, math.pi
    for t in pm.values():
        if t['id'] == src['id']:
            continue
        a = math.atan2(t['y'] - sy, t['x'] - sx)
        res = abs(math.atan2(math.sin(angle - a), math.cos(angle - a)))
        if res < bres:
            bres, best = res, t
    return best, bres


def pro_seats(team_names, rewards, mode):
    """Вернёт множество индексов мест, считающихся 'pro' по выбранному правилу."""
    if mode == 'all':
        return set(range(len(rewards)))
    if mode == 'winners':
        if not rewards:
            return set()
        mx = max(rewards)
        return {i for i, r in enumerate(rewards) if r == mx}
    if mode.startswith('teams:'):
        wanted = {n.strip() for n in mode[len('teams:'):].split(',') if n.strip()}
        return {i for i, n in enumerate(team_names or []) if n in wanted}
    return set()


def process_replay(path, pros_mode):
    """Один реплей → список row-dict по запускам pro-мест. Безопасен к битым файлам."""
    rows = []
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return rows, {'parse_error': 1}

    steps = d.get('steps') or []
    rewards = d.get('rewards') or []
    info = d.get('info', {}) or {}
    team_names = info.get('TeamNames') or []
    n_players = len(rewards) if rewards else (len(steps[0]) if steps else 0)
    mode = '1v1' if n_players == 2 else ('4p' if n_players == 4 else f'{n_players}p')
    pros = pro_seats(team_names, rewards, pros_mode)
    if not pros:
        return rows, {'no_pro_seat': 1}

    epid = info.get('EpisodeId') or d.get('id') or os.path.basename(path)

    for si, step in enumerate(steps):
        for pi, seat in enumerate(step):
            if pi not in pros:
                continue
            act = seat.get('action')
            if not act:
                continue
            obs = seat.get('observation') or {}
            pm = _planet_map(obs.get('planets') or [])
            if not pm:
                continue
            comet_ids = set(obs.get('comet_planet_ids') or [])
            # предрасчёт дистанций от каждого источника считаем лениво ниже
            n_launch_this_turn = len(act)
            for launch in act:
                try:
                    srcid, angle, ships = int(launch[0]), float(launch[1]), float(launch[2])
                except (TypeError, IndexError, ValueError):
                    continue
                src = pm.get(srcid)
                if src is None:
                    continue
                tgt, res = resolve_target(src, angle, pm)
                if tgt is None:
                    continue
                dist = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
                # ранг близости цели среди всех прочих планет от источника
                rank = 1
                for q in pm.values():
                    if q['id'] == src['id'] or q['id'] == tgt['id']:
                        continue
                    if math.hypot(q['x'] - src['x'], q['y'] - src['y']) < dist:
                        rank += 1
                if tgt['owner'] == pi:
                    owner_type = 'own'           # reinforce
                elif tgt['owner'] == NEUTRAL_OWNER:
                    owner_type = 'neutral'
                else:
                    owner_type = 'enemy'
                rows.append({
                    'episode_id': epid, 'mode': mode, 'seat': pi,
                    'team': team_names[pi] if pi < len(team_names) else '',
                    'won': int(rewards[pi] == max(rewards)) if rewards else '',
                    'step': si, 'n_planets': len(pm),
                    'launches_this_turn': n_launch_this_turn,
                    'src_id': srcid, 'src_ships': src['ships'], 'src_prod': src['prod'],
                    'tgt_id': tgt['id'], 'tgt_owner_type': owner_type,
                    'tgt_ships': tgt['ships'], 'tgt_prod': tgt['prod'],
                    'tgt_is_comet': int(tgt['id'] in comet_ids),
                    'ships': ships,
                    'ships_per_src': ships / max(1.0, src['ships']),
                    'ships_per_tgt': ships / max(1.0, tgt['ships']),
                    'dist': dist, 'dist_rank': rank,
                    'angle_resid': res,
                })
    return rows, {'ok': 1}


# ═══════════════════════════════════════════════════════════════════════
# Агрегация / отчёт
# ═══════════════════════════════════════════════════════════════════════

def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def _line(name, xs):
    if not xs:
        return f"  {name:18s} (нет данных)"
    return (f"  {name:18s} min={min(xs):7.2f} p25={_pct(xs,.25):7.2f} "
            f"med={st.median(xs):7.2f} p75={_pct(xs,.75):7.2f} "
            f"p95={_pct(xs,.95):7.2f} max={max(xs):7.2f}")


def build_report(rows, meta):
    n = len(rows)
    out = []
    out.append("# Спектр действий профи — отчёт\n")
    out.append(f"- реплеев обработано: {meta['files_ok']} "
               f"(ошибок парсинга: {meta['parse_error']}, без pro-места: {meta['no_pro_seat']})")
    out.append(f"- pro-режим: `{meta['pros_mode']}`")
    out.append(f"- запусков профи всего: **{n}**")
    if n == 0:
        return "\n".join(out)
    by_mode = Counter(r['mode'] for r in rows)
    out.append(f"- по режиму: " + ", ".join(f"{k}={v}" for k, v in by_mode.most_common()))
    out.append("")

    out.append("## Тип цели")
    out.append("")
    for k, v in Counter(r['tgt_owner_type'] for r in rows).most_common():
        out.append(f"- {k}: {v} ({100*v/n:.1f}%)")
    ncomet = sum(r['tgt_is_comet'] for r in rows)
    out.append(f"- по кометам: {ncomet} ({100*ncomet/n:.1f}%)")
    out.append("")

    out.append("## Распределения\n")
    out.append("```")
    out.append(_line('dist src->tgt', [r['dist'] for r in rows]))
    out.append(_line('dist_rank', [r['dist_rank'] for r in rows]))
    out.append(_line('ships', [r['ships'] for r in rows]))
    out.append(_line('ships/src_ships', [r['ships_per_src'] for r in rows]))
    out.append(_line('ships/tgt_ships', [r['ships_per_tgt'] for r in rows]))
    out.append(_line('angle_resid(rad)', [r['angle_resid'] for r in rows]))
    out.append("```")
    out.append("")

    out.append("## Ранг близости цели (как далеко вниз по списку ближайших)\n")
    out.append("```")
    rc = Counter(min(r['dist_rank'], 11) for r in rows)
    cum = 0
    for k in range(1, 12):
        v = rc.get(k, 0)
        cum += v
        lbl = f"{k}+" if k == 11 else f"{k} "
        out.append(f"  rank {lbl}: {v:6d}  {100*v/n:5.1f}%   (cum {100*cum/n:5.1f}%)")
    out.append("```")
    out.append("")

    out.append("## Запусков за ход (структура координации)\n")
    turn = Counter((r['episode_id'], r['seat'], r['step']) for r in rows)
    out.append("```")
    out.append(_line('launches/turn', list(turn.values())))
    multi = sum(1 for v in turn.values() if v > 1)
    out.append(f"  ходов с >1 запуском: {multi} / {len(turn)} ({100*multi/max(1,len(turn)):.1f}%)")
    out.append("```")
    out.append("")

    out.append("## Кандидат-recall: что отсекут типовые прунинги (по dist_rank)\n")
    out.append("Доля pro-запусков, которые ВЫЖИВУТ при ограничении target-set по рангу близости:")
    out.append("")
    for K in (3, 5, 8, 10, 15, 20):
        kept = sum(1 for r in rows if r['dist_rank'] <= K)
        out.append(f"- top-{K} ближайших: recall = {100*kept/n:.1f}%")
    out.append("")
    out.append("> Чем агрессивнее отсечка по близости, тем больше pro-действий мы теряем "
               "ещё до планировщика. Это прямая оценка потолка генератора кандидатов.")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

CSV_FIELDS = ['episode_id', 'mode', 'seat', 'team', 'won', 'step', 'n_planets',
              'launches_this_turn', 'src_id', 'src_ships', 'src_prod',
              'tgt_id', 'tgt_owner_type', 'tgt_ships', 'tgt_prod', 'tgt_is_comet',
              'ships', 'ships_per_src', 'ships_per_tgt', 'dist', 'dist_rank', 'angle_resid']


def main():
    ap = argparse.ArgumentParser(description="Pro action spectrum study")
    ap.add_argument('--corpus', required=True, help='папка с .json реплеями')
    ap.add_argument('--out', default='./spectrum_out', help='папка вывода')
    ap.add_argument('--limit', type=int, default=0, help='макс. реплеев (0 = все)')
    ap.add_argument('--pros', default='winners',
                    help='winners | all | "teams:Name1,Name2"')
    ap.add_argument('--workers', type=int, default=1, help='процессов (параллелизм)')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.corpus) if f.endswith('.json'))
    if args.limit:
        files = files[:args.limit]
    paths = [os.path.join(args.corpus, f) for f in files]
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, 'pro_launches.csv')
    rep_path = os.path.join(args.out, 'pro_action_spectrum.md')

    print(f"Реплеев к обработке: {len(paths)}  | pros={args.pros} | workers={args.workers}")
    all_rows = []
    meta = {'files_ok': 0, 'parse_error': 0, 'no_pro_seat': 0, 'pros_mode': args.pros}

    def _account(stat):
        if stat.get('parse_error'):
            meta['parse_error'] += 1
        elif stat.get('no_pro_seat'):
            meta['no_pro_seat'] += 1
        else:
            meta['files_ok'] += 1

    done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_replay, p, args.pros): p for p in paths}
            for fut in as_completed(futs):
                rows, stat = fut.result()
                all_rows.extend(rows)
                _account(stat)
                done += 1
                if done % 200 == 0:
                    print(f"  ...{done}/{len(paths)}  (запусков: {len(all_rows)})")
    else:
        for p in paths:
            rows, stat = process_replay(p, args.pros)
            all_rows.extend(rows)
            _account(stat)
            done += 1
            if done % 200 == 0:
                print(f"  ...{done}/{len(paths)}  (запусков: {len(all_rows)})")

    # CSV
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    report = build_report(all_rows, meta)
    with open(rep_path, 'w') as f:
        f.write(report)

    print("\n" + report)
    print(f"\n[saved] {csv_path}  ({len(all_rows)} строк)")
    print(f"[saved] {rep_path}")


if __name__ == '__main__':
    main()
