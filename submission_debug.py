"""
Пер-ходовой отладочный лог для агента из `submission.py`.

Параллельно с нашим `agent_debug.py`, но для submission. Нужен чтобы
сравнивать «как submission принимает решения» с нашими ходами и понять,
из-за каких процессов он отыгрывает поздние стадии лучше.

Использование (notebook):

    import importlib.util, sys
    spec = importlib.util.spec_from_file_location('sub', 'submission.py')
    sub  = importlib.util.module_from_spec(spec); sys.modules['sub'] = sub
    spec.loader.exec_module(sub)

    import submission_debug
    submission_debug.reset()
    os.environ['SUB_AGENT_LOG'] = '/path/to/submission_debug.log'
    strong_agent = submission_debug.wrap(sub)   # ← оборачивает sub.agent

Или как готовый агент (если submission импортируется обычным `import submission`):

    os.environ['SUB_AGENT_LOG'] = '/path/.../submission_debug.log'
    from submission_debug import agent

Что пишется в лог:

  ===== TURN <step>  player=<p>  ts=...  =====
  [GLOBAL]   my_total/enemy_total/my_prod/enemy_prod, флаги фаз
  [MODES]    domination, is_behind/is_ahead/is_dominating/is_finishing
  [PLANETS]  pid owner ships prod  x y  static fall_turn keep_need first_enemy
  [POLICY]   attack_budget per source
  [MISSIONS] kind score src→tgt  turns  ships  (sorted desc)
  [MOVES]    src angle ships ~tgt
  [FLEETS]   опционально (SUB_AGENT_LOG_FLEETS=1)

Управление env vars:
  SUB_AGENT_LOG               — путь к лог-файлу (обязательно для активации)
  SUB_AGENT_LOG_FLEETS=1      — дамп летящих флотов
  SUB_AGENT_LOG_MISSIONS_ALL=1— все missions, не только top-30 по score
"""

import os
import math
import time
import importlib

# ── состояние ─────────────────────────────────────────────────────────────
_FH = None
_RESOLVED = False
_DISABLED = True
_LOG_FLEETS = False
_LOG_MISS_ALL = False

# Per-turn capture (заполняется патчами)
_CAPTURE = {
    'missions': [],     # все Mission(...) этого хода
    'modes': None,      # вывод build_modes
    'policy': None,     # вывод build_policy_state
}

# модули, на которых уже стоят патчи (чтобы не патчить дважды)
_PATCHED_MODS = set()


def _resolve():
    global _FH, _RESOLVED, _DISABLED, _LOG_FLEETS, _LOG_MISS_ALL
    if _RESOLVED:
        return
    _RESOLVED = True
    path = os.environ.get('SUB_AGENT_LOG', '').strip()
    if not path:
        return
    try:
        _FH = open(path, 'a', buffering=1, encoding='utf-8')
        _DISABLED = False
        _LOG_FLEETS    = bool(int(os.environ.get('SUB_AGENT_LOG_FLEETS', '0') or '0'))
        _LOG_MISS_ALL  = bool(int(os.environ.get('SUB_AGENT_LOG_MISSIONS_ALL', '0') or '0'))
    except Exception:
        _DISABLED = True


def reset():
    """Закрыть лог и сбросить кеш env. Полезно в ноутбуке между запусками,
    когда меняем `SUB_AGENT_LOG`. Также снимает все ранее наложенные патчи."""
    global _FH, _RESOLVED, _DISABLED, _LOG_FLEETS, _LOG_MISS_ALL, _PATCHED_MODS
    try:
        if _FH is not None:
            _FH.flush()
            _FH.close()
    except Exception:
        pass
    _FH = None
    _RESOLVED = False
    _DISABLED = True
    _LOG_FLEETS = False
    _LOG_MISS_ALL = False
    # снять патчи (восстановить originals, если хранятся)
    for mod in list(_PATCHED_MODS):
        try:
            origs = getattr(mod, '__sub_dbg_originals__', None)
            if origs:
                for name, fn in origs.items():
                    setattr(mod, name, fn)
                delattr(mod, '__sub_dbg_originals__')
        except Exception:
            pass
    _PATCHED_MODS.clear()


def enabled():
    _resolve()
    return not _DISABLED


def _w(line=''):
    if not enabled():
        return
    try:
        _FH.write(line + '\n')
    except Exception:
        pass


# ── monkey-patch ──────────────────────────────────────────────────────────

def _install_patches(sub):
    """Накладывает захват на: Mission, build_modes, build_policy_state.

    Внутри submission.py все обращения вида `Mission(...)` / `build_modes(world)`
    идут через имя в globals модуля → достаточно подменить атрибуты модуля.
    """
    if sub in _PATCHED_MODS:
        return
    origs = {}

    # ── Mission: ловим все создания (rescue/recapture/reinforce/snipe/single/swarm/...)
    OrigMission = sub.Mission
    origs['Mission'] = OrigMission

    def _MissionCapture(*args, **kwargs):
        m = OrigMission(*args, **kwargs)
        try:
            _CAPTURE['missions'].append(m)
        except Exception:
            pass
        return m

    sub.Mission = _MissionCapture

    # ── build_modes
    if hasattr(sub, 'build_modes'):
        orig_modes = sub.build_modes
        origs['build_modes'] = orig_modes
        def _modes_capture(world):
            r = orig_modes(world)
            _CAPTURE['modes'] = r
            return r
        sub.build_modes = _modes_capture

    # ── build_policy_state
    if hasattr(sub, 'build_policy_state'):
        orig_policy = sub.build_policy_state
        origs['build_policy_state'] = orig_policy
        def _policy_capture(world, deadline=None):
            r = orig_policy(world, deadline=deadline)
            _CAPTURE['policy'] = r
            return r
        sub.build_policy_state = _policy_capture

    sub.__sub_dbg_originals__ = origs
    _PATCHED_MODS.add(sub)


# ── target inference для логов ───────────────────────────────────────────

def _infer_target(sub, world, src_id, angle, ships, horizon=120):
    """Симулируем полёт флота шаг-за-шагом и возвращаем планету, в которую он
    реально влетит (с учётом орбитального вращения нецентрических планет).

    Это та же логика, что и у движка: каждый ход флот сдвигается на (vx, vy),
    планеты на орбите проворачиваются на ω. Цель — первая планета, у которой
    расстояние до флота < радиус планеты.

    Возвращает (tgt_id, eta, miss=0) или None если флот за `horizon` шагов
    никуда не попал (улетит в пустоту / OOB / в солнце).
    """
    try:
        src = world.planet_by_id[src_id]
        sx, sy = sub.launch_point(src.x, src.y, src.radius, angle)
    except Exception:
        return None

    speed = sub.fleet_speed(int(ships))
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    omega = float(getattr(world, 'ang_vel', 0.0) or 0.0)
    cx, cy = sub.CENTER_X, sub.CENTER_Y

    # Заранее вычислим начальные углы и радиусы орбитальных планет.
    static_pids = set()
    rel = {}
    for p in world.planets:
        if p.id == src_id:
            continue
        try:
            is_static = sub.is_static_planet(p)
        except Exception:
            is_static = True
        if is_static:
            static_pids.add(p.id)
            rel[p.id] = (p.x, p.y, p.radius)
        else:
            rx = p.x - cx
            ry = p.y - cy
            r0 = math.hypot(rx, ry)
            theta0 = math.atan2(ry, rx)
            rel[p.id] = (r0, theta0, p.radius)

    for step in range(1, horizon + 1):
        fx = sx + vx * step
        fy = sy + vy * step
        # OOB
        if fx < 0 or fx > sub.BOARD or fy < 0 or fy > sub.BOARD:
            return None
        # солнце
        if math.hypot(fx - cx, fy - cy) < sub.SUN_R:
            return None
        for p in world.planets:
            if p.id == src_id:
                continue
            if p.id in static_pids:
                px, py, pr = rel[p.id]
            else:
                r0, theta0, pr = rel[p.id]
                ang = theta0 + omega * step
                px = cx + r0 * math.cos(ang)
                py = cy + r0 * math.sin(ang)
            if math.hypot(fx - px, fy - py) < pr:
                return (p.id, step, 0.0)
    return None


# ── основной API ─────────────────────────────────────────────────────────

def wrap(submission_module, install_now=True):
    """Возвращает агент-функцию (obs, config) → moves, которая внутри
    вызывает `submission_module.agent(obs, config)` и пишет лог.

    Параметры:
      submission_module — уже загруженный модуль с функциями
                          `agent`, `build_world`, `Mission`, `build_modes`,
                          `build_policy_state` (т.е. submission.py).
      install_now       — наложить patch'и сразу (по умолчанию да).
    """
    if install_now:
        _install_patches(submission_module)

    def _agent(obs, config=None):
        _install_patches(submission_module)  # на случай reload модуля
        # обнуляем capture перед вызовом
        _CAPTURE['missions'] = []
        _CAPTURE['modes'] = None
        _CAPTURE['policy'] = None

        try:
            moves = submission_module.agent(obs, config)
        except Exception as exc:
            _w(f'[ERROR] submission.agent raised {type(exc).__name__}: {exc}')
            raise

        # Соберём world ровно на этом obs (мы НЕ знаем _agent_step,
        # который уже был инкрементирован submission'ом — берём obs.step)
        try:
            obs_step = submission_module._read(obs, 'step', 0) or 0
            world = submission_module.build_world(obs, inferred_step=obs_step)
            _log_turn(submission_module, world, moves)
        except Exception as exc:
            _w(f'[ERROR] log_turn failed: {type(exc).__name__}: {exc}')

        return moves

    return _agent


def agent(obs, config=None):
    """Готовый агент для случая, когда submission импортируется как модуль:
        import submission_debug; agent_fn = submission_debug.agent
    Использует `import submission` (и патчит её)."""
    sub = importlib.import_module('submission')
    return wrap(sub, install_now=True)(obs, config)


# ── рендер лога ──────────────────────────────────────────────────────────

def _log_turn(sub, world, moves):
    if not enabled():
        return
    step = world.step
    player = world.player
    n_planets = len(world.planets)
    n_fleets = len(world.fleets)

    _w()
    _w(f'===== TURN {step}  player={player}  ts={time.strftime("%Y-%m-%d %H:%M:%S")}  '
       f'planets={n_planets}  fleets={n_fleets}  =====')

    # ── глобальная картина
    _w('[GLOBAL]')
    _w(f'   my_total={world.my_total}  enemy_total={world.enemy_total}  '
       f'my_prod={world.my_prod}  enemy_prod={world.enemy_prod}')
    _w(f'   my_planets={len(world.my_planets)}  enemy_planets={len(world.enemy_planets)}  '
       f'neutral={len(world.neutral_planets)}  4p={world.is_four_player}')
    _w(f'   step={step}  remaining={world.remaining_steps}  '
       f'is_early={world.is_early}  is_opening={world.is_opening}  '
       f'is_late={world.is_late}  is_very_late={world.is_very_late}  '
       f'is_total_war={world.is_total_war}')

    modes = _CAPTURE['modes'] or {}
    if modes:
        _w('[MODES]')
        _w(f'   domination={modes.get("domination", 0):+.3f}  '
           f'is_behind={modes.get("is_behind")}  is_ahead={modes.get("is_ahead")}  '
           f'is_dominating={modes.get("is_dominating")}  is_finishing={modes.get("is_finishing")}  '
           f'attack_margin_mult={modes.get("attack_margin_mult", 1.0):.2f}')

    # ── планеты
    _w('[PLANETS]')
    _w('   pid own ships prod    x      y    type fall_t keep_n first_en')
    for p in sorted(world.planets, key=lambda x: x.id):
        own = p.owner
        own_s = ' me' if own == player else ('  -' if own == -1 else f' p{own}')
        try:
            stat = 'S' if sub.is_static_planet(p) else 'O'
        except Exception:
            stat = '?'
        ft = world.fall_turn_map.get(p.id)
        kn = world.keep_needed_map.get(p.id, 0)
        fe = world.first_enemy_map.get(p.id)
        ft_s = '   -' if ft is None else f'{ft:>4}'
        fe_s = '   -' if fe is None else f'{fe:>4}'
        _w(f'   {p.id:>3} {own_s:>3} {p.ships:>5} {p.production:>4}  '
           f'{p.x:>5.1f} {p.y:>5.1f}    {stat}   '
           f'{ft_s} {kn:>5}    {fe_s}')

    # ── policy: attack_budget
    pol = _CAPTURE['policy']
    if pol:
        ab = pol.get('attack_budget') or {}
        if ab:
            _w('[POLICY]  attack_budget per source')
            for src_id in sorted(ab.keys()):
                _w(f'   src={int(src_id):>3}  budget={int(ab[src_id])}')

    # ── missions
    miss = list(_CAPTURE['missions'] or [])
    miss.sort(key=lambda m: -float(getattr(m, 'score', 0.0)))
    _w(f'[MISSIONS]  N={len(miss)}  (sorted by score desc)')
    show = miss if _LOG_MISS_ALL else miss[:30]
    if not show:
        _w('   (none)')
    for i, m in enumerate(show):
        kind  = getattr(m, 'kind', '?')
        score = float(getattr(m, 'score', 0.0))
        tgt   = getattr(m, 'target_id', '?')
        turns = getattr(m, 'turns', '?')
        opts  = getattr(m, 'options', []) or []
        srcs  = ','.join(str(o.src_id) for o in opts) if opts else '-'
        ships = sum(int(getattr(o, 'send_cap', 0)) for o in opts) if opts else 0
        need  = sum(int(getattr(o, 'needed', 0))    for o in opts) if opts else 0
        _w(f'   #{i:<2} {kind:<14} score={score:>+9.2f}  '
           f'src=[{srcs}]→tgt={str(tgt):>3}  turns={str(turns):>3}  '
           f'need={need:>4}  cap={ships:>4}')
    if not _LOG_MISS_ALL and len(miss) > len(show):
        _w(f'   ... ещё {len(miss)-len(show)} missions (set SUB_AGENT_LOG_MISSIONS_ALL=1)')

    # ── moves
    _w(f'[MOVES]  N={len(moves)}')
    for m in moves:
        try:
            src = int(m[0]); angle = float(m[1]); ships = int(m[2])
            inf = _infer_target(sub, world, src, angle, ships)
            if inf is None:
                tgt_str = 'tgt=? (no hit within horizon)'
            else:
                tgt_id, eta, _ = inf
                tp = world.planet_by_id.get(tgt_id)
                if tp is None:
                    tgt_str = f'tgt={tgt_id}'
                else:
                    tp_own = ('me' if tp.owner == player
                              else ('-' if tp.owner == -1 else f'p{tp.owner}'))
                    tgt_str = (f'tgt={tgt_id} (own={tp_own}, ships={tp.ships}, '
                               f'eta={eta})')
            _w(f'   src={src:>3}  angle={math.degrees(angle):+7.2f}°  '
               f'ships={ships:>4}   {tgt_str}')
        except Exception:
            _w(f'   {m}')

    # ── fleets (опционально)
    if _LOG_FLEETS:
        if world.fleets:
            _w(f'[FLEETS]  N={len(world.fleets)}')
            _w('   fid own from   pos              ang     ships')
            for f in world.fleets:
                own = ' *' if f.owner == player else f'p{f.owner}'
                _w(f'   {f.id:>3} {own:>3}  {f.from_planet_id:>3}    '
                   f'({f.x:>5.1f},{f.y:>5.1f})  {math.degrees(f.angle):+7.2f}°  {f.ships}')
        else:
            _w('[FLEETS]  none')

    try:
        _FH.flush()
    except Exception:
        pass


__all__ = ['agent', 'wrap', 'reset', 'enabled']
