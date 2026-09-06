#!/usr/bin/env python3
"""
24_field_dynamics_study.py — оценка действий через призму поля φ.

Идея
────
Доска рассматривается как скалярное поле φ(p) над планетами, которое «хочет»
прийти в состояние «мы доминируем везде» (φ ≥ 0 повсюду). Действия игрока =
перенос силы; хорошее действие СПУСКАЕТ потенциал U = Σ relu(−φ)·val.

Через эту призму получают численную оценку регруппировки (own→own) и мульти-атаки,
которые attack-центричная логика (needed/margin) оценить не могла.

Поле
────
  φ(p) = sign_p·str_p  +  Σ_{q≠p} sign_q·str_q·k(eta(p,q))
    sign  = +1 свой / −1 враг / 0 нейтрал   (с точки зрения игрока pi)
    str   = ships + 5·prod
    k(eta)= 1/(1+eta),  eta — орбит-aware (force._rendezvous_eta, ships_ref)
  demand(p) = relu(−φ(p)) · val(p),   val = 1 + prod
  U = Σ_p demand(p)            ← поле стремится к U → 0

Оценка запуска (S кораблей, src→tgt)
  ΔU ≈ Δdemand(tgt | +S) + Δdemand(src | −S)   (дёшево, без полного пересчёта)
  transport = demand(tgt) − demand(src)
  align     = φ(tgt) < φ(src)   (двигаем из тыла к фронту)

Эксперимент
  Для каждого pro-запуска считаем метрики и сравниваем с NULL (тот же src/ships,
  случайная другая цель). Гипотеза: pro даёт более отрицательный ΔU.
  Разбивка по maneuver_type (берётся из pro_maneuvers.csv, если есть).

Запуск
  python3 24_field_dynamics_study.py --corpus /path/archive0 \
      --bundle "/path/agent_bundle_swarm 2" --out ./field_out \
      --labels ./maneuver_out/pro_maneuvers.csv --limit 40 --workers 8 [--snapshot]
"""

import os
import sys
import csv
import json
import math
import random
import argparse
import statistics as st
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

NEUTRAL_OWNER = -1
SHIPS_REF = 50
PROD_W = 5.0

_B = {}


def _load(bundle):
    if _B:
        return _B
    sys.path.insert(0, bundle)
    from orbit_sim import GameState
    from force import _rendezvous_eta
    _B.update(GameState=GameState, reta=_rendezvous_eta)
    return _B


def _strength(p):
    return max(0.0, p.ships) + PROD_W * max(0.0, p.production)


def _val(p):
    return 1.0 + max(0.0, p.production)


def compute_field(planets, pi, omega, reta):
    """φ(p) для всех планет с точки зрения игрока pi. Возвращает dict id->phi и eta-кэш."""
    n = len(planets)
    sign = {}
    strs = {}
    for p in planets:
        sign[p.id] = (1.0 if p.owner == pi else (-1.0 if p.owner != NEUTRAL_OWNER else 0.0))
        strs[p.id] = _strength(p)
    # eta матрица (симметризуем грубо: считаем eta(p->q))
    phi = {}
    for p in planets:
        s = sign[p.id] * strs[p.id]            # собственный гарнизон
        for q in planets:
            if q.id == p.id:
                continue
            if sign[q.id] == 0.0:
                continue
            eta, _ = reta(p, q, SHIPS_REF, omega)
            s += sign[q.id] * strs[q.id] / (1.0 + eta)
        phi[p.id] = s
    return phi


def process_replay(path, bundle, labels):
    B = _load(bundle)
    GameState = B['GameState']
    reta = B['reta']
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return [], {'parse_error': 1}
    steps = d.get('steps') or []
    rewards = d.get('rewards') or []
    info = d.get('info', {}) or {}
    team_names = info.get('TeamNames') or []
    n_players = len(rewards) if rewards else (len(steps[0]) if steps else 0)
    mode = '1v1' if n_players == 2 else ('4p' if n_players == 4 else f'{n_players}p')
    epid = str(info.get('EpisodeId') or d.get('id') or os.path.basename(path))
    max_step = len(steps)

    rng = random.Random(hash(epid) & 0xffffffff)
    rows = []

    # кэш поля по (si): планеты одинаковы для всех мест, но φ зависит от pi
    field_cache = {}  # (si, pi) -> (phi, planets, demand, U)

    def get_field(si, pi, obs):
        key = (si, pi)
        if key in field_cache:
            return field_cache[key]
        stt = GameState.from_kaggle_obs(obs, step=si)
        planets = stt.planets
        phi = compute_field(planets, pi, stt.omega, reta)
        demand = {p.id: max(0.0, -phi[p.id]) * _val(p) for p in planets}
        U = sum(demand.values())
        pmap = {p.id: p for p in planets}
        field_cache[key] = (phi, pmap, demand, U)
        return field_cache[key]

    def dU_of(phi, pmap, demand, srcid, tgtid, ships, pi):
        """Приближённое ΔU от переноса ships с учётом боёвки на цели."""
        src, tgt = pmap[srcid], pmap[tgtid]
        # эффект на φ(tgt) зависит от владельца:
        #  own    — подкрепление: +ships
        #  enemy  — снижаем гарнизон/свинг: +ships (захват если ≥ обороны)
        #  neutral— только перехлёст сверх обороны реально захватывает: +max(0,S−def)
        if tgt.owner == pi:
            dphi_tgt = ships
        elif tgt.owner == NEUTRAL_OWNER:
            dphi_tgt = max(0.0, ships - tgt.ships)
        else:
            dphi_tgt = ships
        d_tgt_new = max(0.0, -(phi[tgtid] + dphi_tgt)) * _val(tgt)
        d_src_new = max(0.0, -(phi[srcid] - ships)) * _val(src)
        return (d_tgt_new - demand[tgtid]) + (d_src_new - demand[srcid])

    for si, step in enumerate(steps):
        for pi in range(len(step)):
            seat = step[pi]
            act = seat.get('action')
            if not act:
                continue
            obs = seat.get('observation') or {}
            if not obs.get('planets'):
                continue
            phi, pmap, demand, U = get_field(si, pi, obs)
            ids = list(pmap.keys())
            for launch in act:
                try:
                    srcid, angle, ships = int(launch[0]), float(launch[1]), float(launch[2])
                except (TypeError, IndexError, ValueError):
                    continue
                if srcid not in pmap:
                    continue
                # резолв цели здесь не делаем — берём из labels по ключу, иначе
                # приблизим как ближайшую по направлению (для null нам нужен любой tgt)
                lbl = labels.get((epid, pi, si, srcid)) if labels else None
                if lbl is None:
                    continue  # без метки цель неизвестна — пропускаем (нужен 23-CSV)
                tgtid = lbl['tgt_id']
                if tgtid not in pmap:
                    continue
                # pro метрики
                pro_dU = dU_of(phi, pmap, demand, srcid, tgtid, ships, pi)
                pro_transport = demand[tgtid] - demand[srcid]
                pro_align = int(phi[tgtid] < phi[srcid])
                # null: случайная другая цель из того же src
                alt = [i for i in ids if i != srcid and i != tgtid]
                if alt:
                    nid = rng.choice(alt)
                    null_dU = dU_of(phi, pmap, demand, srcid, nid, ships, pi)
                    null_transport = demand[nid] - demand[srcid]
                    null_align = int(phi[nid] < phi[srcid])
                else:
                    null_dU = null_transport = null_align = ''
                rows.append({
                    'episode_id': epid, 'mode': mode, 'seat': pi, 'step': si,
                    'phase': round(si / max(1, max_step), 3),
                    'maneuver_type': lbl['maneuver_type'],
                    'src_id': srcid, 'tgt_id': tgtid, 'ships': ships,
                    'phi_src': round(phi[srcid], 2), 'phi_tgt': round(phi[tgtid], 2),
                    'demand_src': round(demand[srcid], 2), 'demand_tgt': round(demand[tgtid], 2),
                    'U': round(U, 1),
                    'pro_dU': round(pro_dU, 3), 'pro_transport': round(pro_transport, 3),
                    'pro_align': pro_align,
                    'null_dU': round(null_dU, 3) if null_dU != '' else '',
                    'null_transport': round(null_transport, 3) if null_transport != '' else '',
                    'null_align': null_align,
                })
    return rows, {'ok': 1}


# ── загрузка меток из 23-CSV ──────────────────────────────────────────────

def load_labels(path):
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                key = (str(r['episode_id']), int(r['seat']), int(r['step']), int(r['src_id']))
                out[key] = {'tgt_id': int(r['tgt_id']), 'maneuver_type': r['maneuver_type']}
            except (KeyError, ValueError):
                continue
    return out


# ── отчёт ─────────────────────────────────────────────────────────────────

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float('nan')


def build_report(rows, meta):
    n = len(rows)
    out = ["# Поле φ — оценка действий (pro vs null)\n"]
    out.append(f"- реплеев ок: {meta['files_ok']} | запусков с меткой: **{n}**")
    if n == 0:
        out.append("\n(нет строк — нужен --labels с pro_maneuvers.csv от скрипта 23)")
        return "\n".join(out)
    out.append(f"- режимы: " + ", ".join(f"{k}={v}" for k, v in Counter(r['mode'] for r in rows).most_common()))
    out.append("")

    def frac_neg(key):
        xs = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return 100 * sum(1 for x in xs if x < 0) / len(xs) if xs else float('nan')

    out.append("## Главное: спускают ли действия потенциал U?\n")
    out.append("```")
    out.append(f"  pro  ΔU: mean={_mean([r['pro_dU'] for r in rows]):8.3f}   доля ΔU<0 = {frac_neg('pro_dU'):.1f}%")
    out.append(f"  null ΔU: mean={_mean([r['null_dU'] for r in rows]):8.3f}   доля ΔU<0 = {frac_neg('null_dU'):.1f}%")
    out.append(f"  pro  align (tgt ближе к фронту): {100*_mean([r['pro_align'] for r in rows]):.1f}%")
    out.append(f"  null align:                      {100*_mean([r['null_align'] for r in rows]):.1f}%")
    out.append("```")
    out.append("> ΔU<0 = действие спускает поле к равновесию. Если pro << null — поле объясняет выбор.")
    out.append("")

    out.append("## ΔU по типу манёвра (pro)\n")
    by = defaultdict(list)
    for r in rows:
        if isinstance(r['pro_dU'], (int, float)):
            by[r['maneuver_type']].append(r['pro_dU'])
    out.append("```")
    out.append(f"  {'maneuver':18s} {'n':>7s} {'mean ΔU':>9s} {'%ΔU<0':>7s}")
    for k in sorted(by, key=lambda k: _mean(by[k])):
        xs = by[k]
        out.append(f"  {k:18s} {len(xs):7d} {_mean(xs):9.3f} {100*sum(1 for x in xs if x<0)/len(xs):6.1f}%")
    out.append("```")
    out.append("")

    out.append("## Транспорт силы к спросу (demand_tgt − demand_src), pro vs null\n```")
    out.append(f"  pro  transport mean = {_mean([r['pro_transport'] for r in rows]):8.3f}")
    out.append(f"  null transport mean = {_mean([r['null_transport'] for r in rows]):8.3f}")
    out.append("```")
    out.append("> Положительный transport = ведём корабли туда, где силы не хватает.")
    return "\n".join(out)


FIELDS = ['episode_id', 'mode', 'seat', 'step', 'phase', 'maneuver_type', 'src_id',
          'tgt_id', 'ships', 'phi_src', 'phi_tgt', 'demand_src', 'demand_tgt', 'U',
          'pro_dU', 'pro_transport', 'pro_align', 'null_dU', 'null_transport', 'null_align']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default='./field_out')
    ap.add_argument('--labels', default='', help='pro_maneuvers.csv от скрипта 23 (резолв цели+метка)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.corpus) if f.endswith('.json'))
    if args.limit:
        files = files[:args.limit]
    paths = [os.path.join(args.corpus, f) for f in files]
    os.makedirs(args.out, exist_ok=True)
    labels = load_labels(args.labels)
    print(f"Реплеев: {len(paths)} | меток загружено: {len(labels)} | workers={args.workers}")

    all_rows = []
    meta = {'files_ok': 0, 'parse_error': 0}
    done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_load, initargs=(args.bundle,)) as ex:
            futs = {ex.submit(process_replay, p, args.bundle, labels): p for p in paths}
            for fut in as_completed(futs):
                rows, stat = fut.result()
                all_rows.extend(rows)
                meta['files_ok'] += stat.get('ok', 0)
                meta['parse_error'] += stat.get('parse_error', 0)
                done += 1
                if done % 50 == 0:
                    print(f"  ...{done}/{len(paths)} (строк {len(all_rows)})")
    else:
        _load(args.bundle)
        for p in paths:
            rows, stat = process_replay(p, args.bundle, labels)
            all_rows.extend(rows)
            meta['files_ok'] += stat.get('ok', 0)
            meta['parse_error'] += stat.get('parse_error', 0)
            done += 1

    csv_path = os.path.join(args.out, 'field_eval.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    report = build_report(all_rows, meta)
    with open(os.path.join(args.out, 'field_dynamics.md'), 'w') as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[saved] {csv_path} ({len(all_rows)} строк)")


if __name__ == '__main__':
    main()
