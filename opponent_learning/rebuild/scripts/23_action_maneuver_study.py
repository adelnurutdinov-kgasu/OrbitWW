#!/usr/bin/env python3
"""
23_action_maneuver_study.py — глубокое изучение пространства действий + контекста.

Цель
────
Понять не «куда стрельнул», а «какой манёвр исполнял и в каком положении».
Для каждого запуска ЛЮБОГО игрока:
  • контекст совершения (phase, наш ratio, ситуация у src и tgt, гонка, incoming),
  • самодостаточность — через симулятор агента: ships ≥ needed(tgt) ?,
  • композиция — связывание запусков в манёвры:
      multi_sync   — несколько запусков в ОДИН ход на одну цель,
      staged       — запуски в РАЗНЫЕ ходы, флоты прилетают в одно окно к цели,
      relay/forward— цепочка владения: флот пришёл на P → P вскоре пускает дальше.

Резолв цели — родным lead-aware решателем агента projection.simulate_fleet_target
(симулируем флот под фактическим углом, находим планету, которую он реально
встретит с учётом вращения). needed — attacks._required_strike (тот же расчёт,
что в рантайме → нет train/serve skew).

Гигиена данных
──────────────
  • позиции из живого obs['planets']; player_idx из позиции в steps[si];
  • self-sufficiency меряем ships(actual) vs needed, а НЕ eval_direct.success
    (последний опирается на src.ships, который в obs бывает рассинхрон с действием).

Запуск
──────
  python3 23_action_maneuver_study.py --corpus /path/archive0 --out ./maneuver_out \
      --bundle "/path/agent_bundle_swarm 2" --limit 200 --workers 8
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

# окна для композиции (в ходах)
STAGED_ARRIVAL_WINDOW = 3   # флоты на одну цель, прилёт в пределах ±W → staged
RELAY_WINDOW          = 4   # P получил флот → пустил дальше в пределах K ходов
RELAY_ANGLE_TOL       = 0.6 # рад (~34°): outbound должен продолжать направление inbound
RELAY_MASS_FRAC       = 0.5 # outbound должен нести ≥ половины пришедшей массы
ENDGAME_PHASE         = 0.85 # phase ≥ → конец игры
ENDGAME_MIN_SHIPS     = 150  # крупный запуск в конце → выделяем в endgame_mass

# глобали для воркеров (импорт бандла один раз на процесс)
_BUNDLE = {}


def _load_bundle(bundle_path):
    if _BUNDLE:
        return _BUNDLE
    sys.path.insert(0, bundle_path)
    from orbit_sim import GameState, Planet, Fleet
    from projection import simulate_fleet_target
    from attacks import _required_strike, friendly_incoming, _eta, _effective_prod
    _BUNDLE.update(dict(GameState=GameState, Planet=Planet, Fleet=Fleet,
                        simulate_fleet_target=simulate_fleet_target,
                        required_strike=_required_strike,
                        friendly_incoming=friendly_incoming, eta=_eta,
                        eff_prod=_effective_prod))
    return _BUNDLE


# ═══════════════════════════════════════════════════════════════════════
# Union-Find для группировки запусков в манёвры
# ═══════════════════════════════════════════════════════════════════════

class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# ═══════════════════════════════════════════════════════════════════════
# Обработка одного реплея
# ═══════════════════════════════════════════════════════════════════════

def process_replay(path, bundle_path):
    B = _load_bundle(bundle_path)
    GameState, Fleet = B['GameState'], B['Fleet']
    resolve = B['simulate_fleet_target']
    required_strike = B['required_strike']
    friendly_incoming = B['friendly_incoming']
    eta_fn = B['eta']

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
    epid = info.get('EpisodeId') or d.get('id') or os.path.basename(path)
    max_step = len(steps)

    # собираем запуски по местам
    per_seat = defaultdict(list)   # pi -> list of launch dicts

    for si, step in enumerate(steps):
        for pi in range(len(step)):
            seat = step[pi]
            act = seat.get('action')
            if not act:
                continue
            obs = seat.get('observation') or {}
            planets_raw = obs.get('planets') or []
            if not planets_raw:
                continue
            try:
                stt = GameState.from_kaggle_obs(obs, step=si)
            except Exception:
                continue
            omega = stt.omega
            pid_map = {p.id: p for p in stt.planets}
            comet_ids = set(obs.get('comet_planet_ids') or [])

            # --- контекст хода (общий для всех запусков этого хода) ---
            ours = [p for p in stt.planets if p.owner == pi]
            enemies = [p for p in stt.planets if p.owner not in (pi, NEUTRAL_OWNER)]
            neutrals = [p for p in stt.planets if p.owner == NEUTRAL_OWNER]
            our_ships = sum(p.ships for p in ours)
            tot_ships = sum(p.ships for p in stt.planets) or 1.0
            ratio = our_ships / tot_ships
            phase = si / max(1, max_step)

            def mean_d_nonours(p):
                others = [q for q in stt.planets if q.id != p.id and q.owner != pi]
                if not others:
                    return float('inf')
                return sum(math.hypot(p.x - q.x, p.y - q.y) for q in others) / len(others)

            for launch in act:
                try:
                    srcid, angle, ships = int(launch[0]), float(launch[1]), float(launch[2])
                except (TypeError, IndexError, ValueError):
                    continue
                src = pid_map.get(srcid)
                if src is None:
                    continue
                f = Fleet(999999, pi, src.x, src.y, angle, srcid, max(1, int(ships)))
                try:
                    tpid, teta = resolve(f, stt.planets, omega)
                except Exception:
                    tpid, teta = None, None
                if tpid is None:
                    continue
                tgt = pid_map.get(tpid)
                if tgt is None:
                    continue

                if tgt.owner == pi:
                    owner_type = 'own'
                elif tgt.owner == NEUTRAL_OWNER:
                    owner_type = 'neutral'
                else:
                    owner_type = 'enemy'

                # --- needed / самодостаточность (только для не-своих) ---
                needed = None
                race_win = ''
                if owner_type != 'own':
                    try:
                        e2, defender, needed = required_strike(stt, src, tgt, omega, max(1, int(ships)), 0.0)
                    except Exception:
                        needed = None
                    # гонка осмысленна только для НЕЙТРАЛОВ (кто схватит первым);
                    # цель исключаем (для вражеской цели ближайший враг — она сама).
                    if owner_type == 'neutral':
                        try:
                            enemy_eta = min((eta_fn(e, tgt, max(1, int(e.ships)), omega)
                                             for e in enemies if e.id != tgt.id),
                                            default=float('inf'))
                            race_win = int(teta <= enemy_eta)
                        except Exception:
                            race_win = ''

                # incoming свои к цели (контест)
                try:
                    inc, _ = friendly_incoming(stt, tgt, 200, pi)
                except Exception:
                    inc = ''

                # ранг близости цели
                dist = math.hypot(tgt.x - src.x, tgt.y - src.y)
                rank = 1
                for q in stt.planets:
                    if q.id in (src.id, tgt.id):
                        continue
                    if math.hypot(q.x - src.x, q.y - src.y) < dist:
                        rank += 1

                self_suff = ''
                if owner_type != 'own' and needed is not None:
                    self_suff = int(ships >= needed)
                margin = (ships - needed) if (needed is not None) else ''

                per_seat[pi].append({
                    'episode_id': epid, 'mode': mode, 'seat': pi,
                    'team': team_names[pi] if pi < len(team_names) else '',
                    'won': int(rewards[pi] == max(rewards)) if rewards else '',
                    'step': si, 'phase': round(phase, 3),
                    'ratio': round(ratio, 3),
                    'n_own': len(ours), 'n_enemy': len(enemies), 'n_neutral': len(neutrals),
                    'src_id': srcid, 'src_ships': src.ships, 'src_prod': src.production,
                    'src_frontline_d': round(mean_d_nonours(src), 1),
                    'tgt_id': tpid, 'tgt_owner_type': owner_type,
                    'tgt_ships': tgt.ships, 'tgt_prod': tgt.production,
                    'tgt_is_comet': int(tpid in comet_ids),
                    'angle': angle,
                    'ships': ships, 'needed': needed,
                    'self_sufficient': self_suff, 'margin': margin,
                    'eta': round(teta, 2) if teta is not None else '',
                    'arrival': (si + teta) if teta is not None else None,
                    'incoming_friendly': inc, 'race_win': race_win,
                    'dist': round(dist, 1), 'dist_rank': rank,
                    # поля для композиции (заполним ниже)
                    'maneuver_id': None, 'maneuver_type': None, 'role': None,
                })

    # ── композиция: ЛОКАЛЬНЫЕ роли (без транзитивного слияния в мега-блобы) ──
    # multi_sync / staged — группировка по цели в ограниченном окне (bounded),
    # relay — pairwise 1:1 связь без union (флаги relay_in/out на запуске),
    # endgame_mass — крупные поздние запуски выделяем первой, отдельной категорией.
    def _adiff(a, b):
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    out_rows = []
    for pi, launches in per_seat.items():
        n = len(launches)
        relay_in = [False] * n     # прилёт этого запуска форвардится дальше
        relay_out = [False] * n    # этот запуск форвардит ранее пришедший поток
        sync_group = [None] * n    # id группы multi_sync/staged
        sync_kind = [None] * n     # 'multi_sync' | 'staged'

        # группы по (цель, ход) — multi_sync; по (цель, бакет прилёта) — staged
        by_target_turn = defaultdict(list)
        by_target_bucket = defaultdict(list)
        arrivals_at = defaultdict(list)   # planet -> [(arrival, idx)]
        outbound_from = defaultdict(list) # planet -> [(turn, idx)]
        for i, L in enumerate(launches):
            by_target_turn[(L['tgt_id'], L['step'])].append(i)
            if L['arrival'] is not None:
                b = int(L['arrival'] // STAGED_ARRIVAL_WINDOW)
                by_target_bucket[(L['tgt_id'], b)].append(i)
                arrivals_at[L['tgt_id']].append((L['arrival'], i))
            outbound_from[L['src_id']].append((L['step'], i))

        # multi_sync: одна цель, один ход, >1 запуска
        for (tg, si_), idxs in by_target_turn.items():
            if len(idxs) > 1:
                gid = f"{epid}-{pi}-ms-{tg}-{si_}"
                for i in idxs:
                    sync_group[i] = gid; sync_kind[i] = 'multi_sync'

        # staged: одна цель, один бакет прилёта, >1 запуска из разных ходов
        for (tg, b), idxs in by_target_bucket.items():
            steps_set = {launches[i]['step'] for i in idxs}
            if len(idxs) > 1 and len(steps_set) > 1:
                gid = f"{epid}-{pi}-st-{tg}-{b}"
                for i in idxs:
                    if sync_group[i] is None:        # multi_sync приоритетнее
                        sync_group[i] = gid; sync_kind[i] = 'staged'

        # relay: строгий 1:1 матчинг прилёт→форвард (флаги, без union)
        link_id = {}
        lc = 0
        for P, outs in outbound_from.items():
            arrs = sorted(arrivals_at.get(P, []))
            outs_sorted = sorted(outs)
            consumed = set()
            for (aP, ia) in arrs:
                in_ang = launches[ia]['angle']; in_shp = launches[ia]['ships']
                for (si_b, ib) in outs_sorted:
                    if ib in consumed or ia == ib:
                        continue
                    if not (aP <= si_b <= aP + RELAY_WINDOW):
                        continue
                    if _adiff(launches[ib]['angle'], in_ang) > RELAY_ANGLE_TOL:
                        continue
                    if launches[ib]['ships'] < RELAY_MASS_FRAC * in_shp:
                        continue
                    relay_in[ia] = True; relay_out[ib] = True
                    lid = f"{epid}-{pi}-rl-{lc}"; lc += 1
                    link_id.setdefault(ia, lid); link_id[ib] = lid
                    consumed.add(ib)
                    break

        # классификация: endgame_mass → (по типу цели) reinforce-семья / strike-семья
        def _relay_role(i):
            return 'relay_mid' if (relay_in[i] and relay_out[i]) else ('relay_in' if relay_in[i] else 'relay_out')

        for i, L in enumerate(launches):
            phase = L['phase']; ships = L['ships']
            is_relay = relay_in[i] or relay_out[i]
            in_sync = sync_group[i] is not None
            own = L['tgt_owner_type'] == 'own'

            if phase >= ENDGAME_PHASE and ships >= ENDGAME_MIN_SHIPS:
                mtype, role, mid = 'endgame_mass', 'mass', f"{epid}-{pi}-eg"
            elif own:
                # своя цель = подкрепление: forward_stage / mass_reinforce / atomic
                if is_relay:
                    mtype, role, mid = 'forward_stage', _relay_role(i), link_id.get(i, f"{epid}-{pi}-rl-solo")
                elif in_sync:
                    mtype, role, mid = 'mass_reinforce', 'member', sync_group[i]
                else:
                    mtype, role, mid = 'atomic_reinforce', 'solo', f"{epid}-{pi}-solo-{i}"
            else:
                # не-своя цель = атака: sync-strike / relay / atomic / feint
                if in_sync:
                    mtype, role, mid = sync_kind[i], 'member', sync_group[i]
                elif is_relay:
                    mtype, role, mid = 'relay', _relay_role(i), link_id.get(i, f"{epid}-{pi}-rl-solo")
                elif L['self_sufficient'] == 1:
                    mtype, role, mid = 'atomic_capture', 'solo', f"{epid}-{pi}-solo-{i}"
                elif L['self_sufficient'] == 0:
                    mtype, role, mid = 'feint_insufficient', 'solo', f"{epid}-{pi}-solo-{i}"
                else:
                    mtype, role, mid = 'atomic_other', 'solo', f"{epid}-{pi}-solo-{i}"
            L['maneuver_id'] = mid
            L['maneuver_type'] = mtype
            L['role'] = role
            out_rows.append(L)

    return out_rows, {'ok': 1}


# ═══════════════════════════════════════════════════════════════════════
# Отчёт
# ═══════════════════════════════════════════════════════════════════════

def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')


def _line(name, xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return f"  {name:20s} (нет данных)"
    return (f"  {name:20s} min={min(xs):7.2f} p25={_pct(xs,.25):7.2f} "
            f"med={st.median(xs):7.2f} p75={_pct(xs,.75):7.2f} "
            f"p95={_pct(xs,.95):7.2f} max={max(xs):7.2f}")


def build_report(rows, meta):
    n = len(rows)
    out = ["# Спектр действий + манёвры — отчёт\n"]
    out.append(f"- реплеев ок: {meta['files_ok']} (parse_err {meta['parse_error']})")
    out.append(f"- запусков всего: **{n}**")
    if n == 0:
        return "\n".join(out)
    out.append("- режимы: " + ", ".join(f"{k}={v}" for k, v in Counter(r['mode'] for r in rows).most_common()))
    out.append("")

    out.append("## Тип манёвра (атомарные vs композитные)\n")
    mt = Counter(r['maneuver_type'] for r in rows)
    for k, v in mt.most_common():
        out.append(f"- {k}: {v} ({100*v/n:.1f}%)")
    COMPOSITE = ('multi_sync', 'staged', 'forward_stage', 'relay', 'mass_reinforce')
    atomic = sum(v for k, v in mt.items() if k.startswith('atomic'))
    composite = sum(v for k, v in mt.items() if k in COMPOSITE)
    endgame = mt.get('endgame_mass', 0)
    feint = mt.get('feint_insufficient', 0)
    out.append(f"\n**атомарные: {100*atomic/n:.1f}%  |  композитные: {100*composite/n:.1f}%  "
               f"|  endgame_mass: {100*endgame/n:.1f}%  |  feint: {100*feint/n:.1f}%**")
    out.append("")

    out.append("## Тип цели\n")
    for k, v in Counter(r['tgt_owner_type'] for r in rows).most_common():
        out.append(f"- {k}: {v} ({100*v/n:.1f}%)")
    out.append("")

    out.append("## own→own (reinforce): сколько из них часть манёвра\n")
    own = [r for r in rows if r['tgt_owner_type'] == 'own']
    if own:
        omt = Counter(r['maneuver_type'] for r in own)
        for k, v in omt.most_common():
            out.append(f"- {k}: {v} ({100*v/len(own):.1f}% от own)")
    out.append("")

    out.append("## Самодостаточность по не-своим целям\n")
    nonown = [r for r in rows if r['tgt_owner_type'] != 'own' and isinstance(r['self_sufficient'], int)]
    if nonown:
        ss = sum(r['self_sufficient'] for r in nonown)
        out.append(f"- ships ≥ needed (берёт в одиночку): {100*ss/len(nonown):.1f}%")
        out.append(f"- ships < needed (вклад/феинт): {100*(len(nonown)-ss)/len(nonown):.1f}%")
        rw = [r['race_win'] for r in nonown if isinstance(r['race_win'], int)]
        if rw:
            out.append(f"- выигрывают гонку к цели (наш eta ≤ ближайшего врага): {100*sum(rw)/len(rw):.1f}%")
    out.append("")

    out.append("## Распределения\n```")
    out.append(_line('dist src->tgt', [r['dist'] for r in rows]))
    out.append(_line('dist_rank', [r['dist_rank'] for r in rows]))
    out.append(_line('ships', [r['ships'] for r in rows]))
    out.append(_line('margin (ships-needed)', [r['margin'] for r in rows]))
    out.append(_line('eta', [r['eta'] for r in rows]))
    out.append(_line('phase', [r['phase'] for r in rows]))
    out.append(_line('ratio (our share)', [r['ratio'] for r in rows]))
    out.append("```\n")

    out.append("## Манёвр × фаза игры (доля композитных растёт?)\n```")
    for lo, hi, lbl in [(0, .33, 'early'), (.33, .66, 'mid'), (.66, 1.01, 'late')]:
        seg = [r for r in rows if lo <= r['phase'] < hi]
        if not seg:
            continue
        comp = sum(1 for r in seg if r['maneuver_type'] in ('multi_sync', 'staged', 'forward_stage', 'relay', 'mass_reinforce'))
        eg = sum(1 for r in seg if r['maneuver_type'] == 'endgame_mass')
        out.append(f"  {lbl:6s} n={len(seg):6d}  композитных={100*comp/len(seg):4.1f}%  endgame_mass={100*eg/len(seg):4.1f}%")
    out.append("```")
    return "\n".join(out)


CSV_FIELDS = ['episode_id', 'mode', 'seat', 'team', 'won', 'step', 'phase', 'ratio',
              'n_own', 'n_enemy', 'n_neutral', 'src_id', 'src_ships', 'src_prod',
              'src_frontline_d', 'tgt_id', 'tgt_owner_type', 'tgt_ships', 'tgt_prod',
              'tgt_is_comet', 'ships', 'needed', 'self_sufficient', 'margin', 'eta',
              'arrival', 'incoming_friendly', 'race_win', 'dist', 'dist_rank',
              'maneuver_id', 'maneuver_type', 'role']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', default='./maneuver_out')
    ap.add_argument('--bundle', required=True, help='путь к agent bundle (для импорта симулятора)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.corpus) if f.endswith('.json'))
    if args.limit:
        files = files[:args.limit]
    paths = [os.path.join(args.corpus, f) for f in files]
    os.makedirs(args.out, exist_ok=True)

    print(f"Реплеев: {len(paths)} | bundle={args.bundle} | workers={args.workers}")
    all_rows = []
    meta = {'files_ok': 0, 'parse_error': 0}
    done = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_load_bundle, initargs=(args.bundle,)) as ex:
            futs = {ex.submit(process_replay, p, args.bundle): p for p in paths}
            for fut in as_completed(futs):
                rows, stat = fut.result()
                all_rows.extend(rows)
                meta['files_ok'] += stat.get('ok', 0)
                meta['parse_error'] += stat.get('parse_error', 0)
                done += 1
                if done % 100 == 0:
                    print(f"  ...{done}/{len(paths)} (запусков {len(all_rows)})")
    else:
        _load_bundle(args.bundle)
        for p in paths:
            rows, stat = process_replay(p, args.bundle)
            all_rows.extend(rows)
            meta['files_ok'] += stat.get('ok', 0)
            meta['parse_error'] += stat.get('parse_error', 0)
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(paths)} (запусков {len(all_rows)})")

    csv_path = os.path.join(args.out, 'pro_maneuvers.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in CSV_FIELDS})

    report = build_report(all_rows, meta)
    rep_path = os.path.join(args.out, 'maneuver_study.md')
    with open(rep_path, 'w') as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[saved] {csv_path} ({len(all_rows)} строк)")
    print(f"[saved] {rep_path}")


if __name__ == '__main__':
    main()
