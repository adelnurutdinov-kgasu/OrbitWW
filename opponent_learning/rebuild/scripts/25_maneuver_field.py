#!/usr/bin/env python3
"""
25_maneuver_field.py — маневро-уровневый φ: «схлопывание пайплайна».

Гипотеза
────────
Per-launch потенциал U штрафует регруппировки, потому что промежуточный хоп
забирает корабли с источника (поднимает его дефицит), а ценность реализуется
на конечной планете. Если СХЛОПНУТЬ манёвр — посчитать чистый приток на каждую
планету (Σ вошло − Σ вышло внутри maneuver_id) — хабы-посредники зануляются,
остаётся «тыл потерял → фронт получил», и ΔU должен стать отрицательным.

Эксперимент
  Для каждого maneuver_id (из pro_maneuvers.csv, размер ≥2):
    • строим поле φ на СТАРТОВОМ состоянии манёвра (из raw),
    • naive_dU   = Σ по хопам независимых ΔU (как в 24, на одном φ),
    • collapsed_dU = Σ по планетам Δdemand(чистый приток) — хабы схлопнуты.
  Сравниваем по типу манёвра: «реабилитируются» ли регруппировки.

Запуск
  python3 25_maneuver_field.py --corpus /path/archive0 \
      --bundle "/path/agent_bundle_swarm 2" --labels ./maneuver_out/pro_maneuvers.csv \
      --out ./mfield_out --limit 30 --workers 8
"""

import os
import sys
import csv
import json
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
    """Возвращает (phi, Ohat):
       phi  — оборонное поле (signed influence),
       Ohat — нормированный потенциал проекции: достижимые из p возможности
              (нейтралы+враги, взвешенные ценностью и 1/(1+eta))."""
    sign = {p.id: (1.0 if p.owner == pi else (-1.0 if p.owner != NEUTRAL_OWNER else 0.0)) for p in planets}
    strs = {p.id: _strength(p) for p in planets}
    phi, O = {}, {}
    for p in planets:
        s = sign[p.id] * strs[p.id]
        o = 0.0
        for q in planets:
            if q.id == p.id:
                continue
            eta, _ = reta(p, q, SHIPS_REF, omega)
            if sign[q.id] != 0.0:
                s += sign[q.id] * strs[q.id] / (1.0 + eta)
            if q.owner != pi:                       # возможность экспансии
                o += _val(q) / (1.0 + eta)
        phi[p.id] = s
        O[p.id] = o
    mean_o = (sum(O.values()) / len(O)) if O else 0.0
    Ohat = {k: v / (mean_o + 1e-9) for k, v in O.items()}
    return phi, Ohat


def _demand(phi_p, p):
    return max(0.0, -phi_p) * _val(p)


def _dphi_effect(tgt, ships, pi):
    """Приближённый сдвиг φ(tgt) от прихода ships (с учётом боёвки)."""
    if ships >= 0:
        if tgt.owner == pi or tgt.owner != NEUTRAL_OWNER:
            return ships
        return max(0.0, ships - tgt.ships)   # нейтрал: только перехлёст захватывает
    return ships  # отрицательный приток (источник теряет силу)


def process_replay(path, bundle, man_by_ep, lam=0.0):
    B = _load(bundle)
    GameState, reta = B['GameState'], B['reta']
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return [], {'parse_error': 1}
    steps = d.get('steps') or []
    info = d.get('info', {}) or {}
    epid = str(info.get('EpisodeId') or d.get('id') or os.path.basename(path))
    mans = man_by_ep.get(epid)
    if not mans:
        return [], {'no_man': 1}
    rewards = d.get('rewards') or []
    n_players = len(rewards) if rewards else (len(steps[0]) if steps else 0)
    mode = '1v1' if n_players == 2 else ('4p' if n_players == 4 else f'{n_players}p')

    field_cache = {}
    rows = []
    for mid, members in mans.items():
        if len(members) < 2:
            continue
        seat = members[0]['seat']
        start = min(m['step'] for m in members)
        mtype = members[0]['mtype']
        key = (start, seat)
        if key in field_cache:
            phi, Ohat, pmap = field_cache[key]
        else:
            try:
                obs = steps[start][seat]['observation']
                stt = GameState.from_kaggle_obs(obs, step=start)
            except Exception:
                continue
            phi, Ohat = compute_field(stt.planets, seat, stt.omega, reta)
            pmap = {p.id: p for p in stt.planets}
            field_cache[key] = (phi, Ohat, pmap)

        # все участники должны иметь известные планеты в стартовом состоянии
        if any(m['src'] not in pmap or m['tgt'] not in pmap for m in members):
            continue

        # naive: сумма независимых ΔU по хопам (оборона) на одном φ
        naive_def = 0.0
        for m in members:
            s, t, sh = m['src'], m['tgt'], m['ships']
            d_t = _demand(phi[t] + _dphi_effect(pmap[t], sh, seat), pmap[t]) - _demand(phi[t], pmap[t])
            d_s = _demand(phi[s] - sh, pmap[s]) - _demand(phi[s], pmap[s])
            naive_def += d_t + d_s

        # collapsed: чистый приток на планету (хабы зануляются)
        net = defaultdict(float)
        for m in members:
            net[m['tgt']] += m['ships']
            net[m['src']] -= m['ships']
        collapsed_def = 0.0
        for pid, ns in net.items():
            if abs(ns) < 1e-9:
                continue
            collapsed_def += _demand(phi[pid] + _dphi_effect(pmap[pid], ns, seat), pmap[pid]) - _demand(phi[pid], pmap[pid])

        # проекционный член: ΔU_proj = −λ·Σ net_p·Ô(p)  (линеен → одинаков для naive/collapsed)
        proj = sum(ns * Ohat.get(pid, 0.0) for pid, ns in net.items())
        dU_proj = -lam * proj

        rows.append({
            'episode_id': epid, 'mode': mode, 'maneuver_type': mtype,
            'n_launches': len(members), 'n_planets_net': sum(1 for v in net.values() if abs(v) > 1e-9),
            'ships_total': sum(m['ships'] for m in members),
            'naive_dU': round(naive_def + dU_proj, 3),
            'collapsed_dU': round(collapsed_def + dU_proj, 3),
            'dU_proj': round(dU_proj, 3),
            'collapsed_def': round(collapsed_def, 3),
            'rehab': round(naive_def - collapsed_def, 3),
        })
    return rows, {'ok': 1}


def load_maneuvers(path):
    """maneuver_id (size≥2) → list of member launches, сгруппировано по episode."""
    by_ep = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ep = str(r['episode_id']); mid = r['maneuver_id']
                by_ep[ep][mid].append({
                    'seat': int(r['seat']), 'step': int(r['step']),
                    'src': int(r['src_id']), 'tgt': int(r['tgt_id']),
                    'ships': float(r['ships']), 'mtype': r['maneuver_type'],
                })
            except (KeyError, ValueError):
                continue
    return {ep: dict(m) for ep, m in by_ep.items()}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float('nan')


def build_report(rows, meta):
    n = len(rows)
    out = ["# Маневро-уровневый φ: схлопывание пайплайна\n"]
    out.append(f"- реплеев ок: {meta['files_ok']} | манёвров (size≥2): **{n}**")
    if n == 0:
        return "\n".join(out)
    out.append("")
    out.append("## naive (по хопам) → collapsed (схлопнутый) ΔU, по типу манёвра\n```")
    out.append(f"  {'maneuver':16s} {'n':>6s} {'naive ΔU':>10s} {'collapsed':>10s}  {'%collapsed<0':>12s}")
    by = defaultdict(list)
    for r in rows:
        by[r['maneuver_type']].append(r)
    for k in sorted(by, key=lambda k: _mean([r['collapsed_dU'] for r in by[k]])):
        rs = by[k]
        nav = _mean([r['naive_dU'] for r in rs])
        col = _mean([r['collapsed_dU'] for r in rs])
        pneg = 100 * sum(1 for r in rs if r['collapsed_dU'] < 0) / len(rs)
        out.append(f"  {k:16s} {len(rs):6d} {nav:10.2f} {col:10.2f}  {pneg:11.1f}%")
    out.append("```")
    out.append("")
    nav_all = _mean([r['naive_dU'] for r in rows])
    col_all = _mean([r['collapsed_dU'] for r in rows])
    out.append("## Итог\n```")
    out.append(f"  naive ΔU     mean = {nav_all:8.2f}   доля<0 = {100*sum(1 for r in rows if r['naive_dU']<0)/n:.1f}%")
    out.append(f"  collapsed ΔU mean = {col_all:8.2f}   доля<0 = {100*sum(1 for r in rows if r['collapsed_dU']<0)/n:.1f}%")
    out.append("```")
    out.append("> Если collapsed << naive (и доля<0 растёт) — схлопывание пайплайна "
               "«реабилитирует» регруппировки: их ценность видна на уровне манёвра, не хопа.")
    return "\n".join(out)


FIELDS = ['episode_id', 'mode', 'maneuver_type', 'n_launches', 'n_planets_net',
          'ships_total', 'naive_dU', 'collapsed_dU', 'rehab']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', default='./mfield_out')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.corpus) if f.endswith('.json'))
    if args.limit:
        files = files[:args.limit]
    paths = [os.path.join(args.corpus, f) for f in files]
    os.makedirs(args.out, exist_ok=True)
    man_by_ep = load_maneuvers(args.labels)
    print(f"Реплеев: {len(paths)} | эпизодов с манёврами: {len(man_by_ep)} | workers={args.workers}")

    all_rows = []
    meta = {'files_ok': 0, 'parse_error': 0}
    done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_load, initargs=(args.bundle,)) as ex:
            futs = {ex.submit(process_replay, p, args.bundle, man_by_ep): p for p in paths}
            for fut in as_completed(futs):
                rows, stat = fut.result()
                all_rows.extend(rows)
                meta['files_ok'] += stat.get('ok', 0)
                meta['parse_error'] += stat.get('parse_error', 0)
                done += 1
                if done % 50 == 0:
                    print(f"  ...{done}/{len(paths)} (манёвров {len(all_rows)})")
    else:
        _load(args.bundle)
        for p in paths:
            rows, stat = process_replay(p, args.bundle, man_by_ep)
            all_rows.extend(rows)
            meta['files_ok'] += stat.get('ok', 0)
            done += 1

    csv_path = os.path.join(args.out, 'maneuver_field.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    report = build_report(all_rows, meta)
    with open(os.path.join(args.out, 'maneuver_field.md'), 'w') as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[saved] {csv_path} ({len(all_rows)} строк)")


if __name__ == '__main__':
    main()
