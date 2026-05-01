"""
Пер-ходовой отладочный лог агента.

Активируется переменной окружения `ORBIT_AGENT_LOG` (путь к лог-файлу).
Если переменная не задана — все функции no-op (накладные расходы ≈ 0).
Формат — текст, удобный для grep/чтения глазами:

  ===== TURN <step>  player=<p>  ts=<...>  =====
  [ZONES]
     pid owner zone              priority   prod ships proj_at  area_inv  wnn_close   ...
     ...
  [TARGETS]  N=...
     pid (zone, priority)
     ...
  [PLANS]   N=...
     idx mode       sup→att→tgt   eta  x_sup  x_att   defender  margin  success
     ...
  [DECISIONS]
     plan#X mode=...  -> FIRE | SKIP: <reason>
     ...
  [MOVES]   N=...
     [src, angle_deg, ships]
     ...
  [FLEETS]  flying fleets in obs (own + enemy)
     ...

Управление:
  ORBIT_AGENT_LOG=/path/to/file.log     — путь к лог-файлу (append-mode)
  ORBIT_AGENT_LOG_FLEETS=1              — также логировать список летящих флотов
  ORBIT_AGENT_LOG_PLANS_ALL=1           — логировать ВСЕ кандидатные планы
                                          (по умолчанию только финальный
                                          best_attacks top-N)
"""

import os
import time
import math

# ── ленивый init ──────────────────────────────────────────────────────────
_LOG_FH       = None
_RESOLVED     = False
_DISABLED     = True
_LOG_FLEETS   = False
_LOG_PLANS_ALL = False


def _resolve():
    global _LOG_FH, _RESOLVED, _DISABLED, _LOG_FLEETS, _LOG_PLANS_ALL
    if _RESOLVED:
        return
    _RESOLVED = True
    path = os.environ.get('ORBIT_AGENT_LOG', '').strip()
    if not path:
        return
    try:
        _LOG_FH = open(path, 'a', buffering=1, encoding='utf-8')
        _DISABLED = False
        _LOG_FLEETS    = bool(int(os.environ.get('ORBIT_AGENT_LOG_FLEETS', '0') or '0'))
        _LOG_PLANS_ALL = bool(int(os.environ.get('ORBIT_AGENT_LOG_PLANS_ALL', '0') or '0'))
    except Exception:
        _DISABLED = True


def enabled():
    _resolve()
    return not _DISABLED


def reset():
    """Сбросить кеш и закрыть текущий лог-файл. Полезно в ноутбуке когда
    меняем ORBIT_AGENT_LOG между запусками — без reset() модуль помнит
    первое решение и игнорирует новые env-переменные."""
    global _LOG_FH, _RESOLVED, _DISABLED, _LOG_FLEETS, _LOG_PLANS_ALL
    try:
        if _LOG_FH is not None:
            _LOG_FH.flush()
            _LOG_FH.close()
    except Exception:
        pass
    _LOG_FH = None
    _RESOLVED = False
    _DISABLED = True
    _LOG_FLEETS = False
    _LOG_PLANS_ALL = False
    if hasattr(log_decision, '_started'):
        delattr(log_decision, '_started')


def _w(line=''):
    if not enabled():
        return
    try:
        _LOG_FH.write(line + '\n')
    except Exception:
        pass


def _safe(v, fmt='{:.2f}'):
    try:
        if v is None:
            return '-'
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return str(v)
            return fmt.format(v)
        return str(v)
    except Exception:
        return str(v)


# ── секции ────────────────────────────────────────────────────────────────

def begin_turn(step, player, n_planets, n_fleets):
    if not enabled():
        return
    _w()
    _w(f'===== TURN {step}  player={player}  ts={time.strftime("%Y-%m-%d %H:%M:%S")}  '
       f'planets={n_planets}  fleets={n_fleets}  =====')


def log_zones(df, projected_at=None):
    """df — output of compute_zones_from_state. Печатаем компактную таблицу."""
    if not enabled() or df is None:
        return
    _w('[ZONES]')
    cols = ['pid', 'owner', 'zone', 'priority', 'prod', 'ships',
            'area_inv', 'wnn_close_res', 'mean_dist_all', 'n_cross']
    header = '   pid own zone              priority   prod ships  proj_at  area_inv  wnn_close mean_d   n_cr'
    _w(header)
    # сортировка: сначала наши по priority desc, потом цели по priority desc
    try:
        rows = df.sort_values(['owner', 'priority'], ascending=[True, False]).itertuples(index=False)
    except Exception:
        rows = df.itertuples(index=False)
    for r in rows:
        d = r._asdict() if hasattr(r, '_asdict') else dict(zip(df.columns, r))
        pid   = int(d.get('pid', -1))
        owner = int(d.get('owner', -1))
        zone  = str(d.get('zone', '?'))
        prio  = float(d.get('priority', 0.0))
        prod  = int(d.get('prod', 0))
        ships = int(d.get('ships', 0))
        ai    = float(d.get('area_inv', 0.0))
        wn    = float(d.get('wnn_close_res', 0.0))
        md    = float(d.get('mean_dist_all', 0.0))
        nc    = int(d.get('n_cross', 0))
        proj  = (projected_at or {}).get(pid, 0)
        _w(f'   {pid:>3} {owner:>3} {zone:<16}  {prio:>+7.2f}  {prod:>4} {ships:>5}  {proj:>6}   '
           f'{ai:>+7.2f}  {wn:>+7.2f}  {md:>5.1f}  {nc:>4}')


def log_targets(target_ids, zones_df=None):
    if not enabled():
        return
    _w(f'[TARGETS]  N={len(target_ids)}')
    if not target_ids:
        return
    by_pid = {}
    if zones_df is not None:
        try:
            for _, r in zones_df.iterrows():
                by_pid[int(r['pid'])] = (str(r.get('zone', '?')), float(r.get('priority', 0.0)))
        except Exception:
            pass
    for pid in target_ids:
        zone, prio = by_pid.get(int(pid), ('?', 0.0))
        _w(f'   pid={pid:<3} zone={zone:<16}  priority={prio:+.2f}')


def log_plans(plans, label='PLANS'):
    if not enabled() or not plans:
        if enabled():
            _w(f'[{label}]  N=0')
        return
    _w(f'[{label}]  N={len(plans)}')
    _w('   #  mode        sup→att→tgt           eta_sa eta_at  x_sup  x_att   def  margin  succ  reason')
    for i, p in enumerate(plans):
        mode  = p.get('mode', '?')
        sup   = p.get('sup_id', '-')
        att   = p.get('att_id', '-')
        tgt   = p.get('tgt_id', '-')
        eta_sa = p.get('eta_sa', 0)
        eta_at = p.get('eta_at', p.get('t_total', 0))
        x_sup  = int(p.get('x_sup', 0))
        x_att  = int(p.get('x_att', 0))
        defd   = p.get('defender', 0)
        marg   = p.get('margin', 0)
        succ   = bool(p.get('success', False))
        reason = p.get('relay_reason', None) or ''
        triple = f'{sup}→{att}→{tgt}' if sup not in (None, '-') else f'   {att}→{tgt}'
        _w(f'   {i:>2}  {mode:<10}  {triple:<20}  '
           f'{_safe(eta_sa, "{:.1f}"):>5}  {_safe(eta_at, "{:.1f}"):>5}  '
           f'{x_sup:>5}  {x_att:>5}  {_safe(defd, "{:.1f}"):>5}  {_safe(marg, "{:+.1f}"):>6}  '
           f'{"Y" if succ else "n":>4}  {reason}')


def log_decision(plan_idx, plan, status, reason=''):
    """status ∈ {'FIRE', 'SKIP'}"""
    if not enabled():
        return
    if not hasattr(log_decision, '_started'):
        _w('[DECISIONS]')
        log_decision._started = True
    mode = plan.get('mode', '?')
    sup  = plan.get('sup_id', '-')
    att  = plan.get('att_id', '-')
    tgt  = plan.get('tgt_id', '-')
    triple = f'{sup}→{att}→{tgt}' if sup not in (None, '-') else f'{att}→{tgt}'
    _w(f'   plan#{plan_idx} {mode:<10} {triple:<14} -> {status}: {reason}')


def reset_decisions():
    """Сбросить «уже напечатано» — между ходами."""
    if hasattr(log_decision, '_started'):
        delattr(log_decision, '_started')


def log_moves(moves):
    if not enabled():
        return
    _w(f'[MOVES]  N={len(moves)}')
    for m in moves:
        try:
            src, angle, ships = m
            _w(f'   src={int(src):>3}  angle={math.degrees(float(angle)):+7.2f}°  ships={int(ships)}')
        except Exception:
            _w(f'   {m}')


def log_fleets(fleets, player):
    if not enabled() or not _LOG_FLEETS:
        return
    if not fleets:
        _w('[FLEETS]  none flying')
        return
    _w(f'[FLEETS]  N={len(fleets)}')
    _w('   fid own from_pid    pos              angle    ships')
    for f in fleets:
        own = '\u2606' if f.owner == player else '\u00d7'
        _w(f'   {f.id:>3}  {own}   {f.from_pid:>3}     '
           f'({f.x:>5.1f},{f.y:>5.1f})   {math.degrees(f.angle):+7.2f}°   {f.ships}')


def log_error(where, exc):
    if not enabled():
        return
    _w(f'[ERROR]  in {where}: {type(exc).__name__}: {exc}')


def end_turn():
    if not enabled():
        return
    reset_decisions()
    try:
        _LOG_FH.flush()
    except Exception:
        pass


def plans_all_enabled():
    _resolve()
    return _LOG_PLANS_ALL


__all__ = [
    'enabled', 'reset', 'begin_turn', 'log_zones', 'log_targets', 'log_plans',
    'log_decision', 'log_moves', 'log_fleets', 'log_error', 'end_turn',
    'plans_all_enabled',
]
