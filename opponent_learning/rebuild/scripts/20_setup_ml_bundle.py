#!/usr/bin/env python3
"""
20_setup_ml_bundle.py — создаёт изолированный ML-агент в отдельной папке.

ЧТО ДЕЛАЕТ
──────────
1. Удаляет existing agent_bundle_ml/ если есть
2. Копирует agent_bundle_swarm 2/ → agent_bundle_ml/ (без __pycache__)
3. В agent_bundle_ml/zones.py добавляет inline ML residual:
       priority += alpha * z(ML_score)
   используя обученные веса из rebuild/models/ml_zones_linear_*.

ВЫЗОВ
─────
  python3 20_setup_ml_bundle.py --alpha 0.5

ВЫХОД
─────
  agent_bundle_ml/ — готовый к использованию ML-агент

После этого 21_tournament_proper.py делает честный A/B.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
PROJECT_ROOT = OL_DIR.parent

SRC_BUNDLE = PROJECT_ROOT / "agent_bundle_swarm 2"
DST_BUNDLE = PROJECT_ROOT / "agent_bundle_ml"

MODEL_META = REBUILD / "models" / "ml_zones_linear_meta.json"


# Inline-патч для zones.py — добавляется в конец файла
ZONES_PATCH = '''

# ══════════════════════════════════════════════════════════════════════════
# ML RESIDUAL (добавлено 20_setup_ml_bundle.py)
# ══════════════════════════════════════════════════════════════════════════

import json as _json
import os as _os
import numpy as _np

_ML_META_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), 'ml_zones_meta.json'
)
_ML_ALPHA = {alpha}

try:
    with open(_ML_META_PATH) as _f:
        _ML_META = _json.load(_f)
    _ML_PLANET_FEATS = _ML_META['planet_features']
    _ML_CTX_FEATS    = _ML_META['context_features']
    _ML_FEAT_STATS   = _ML_META['feat_stats']
    _ML_W_PLANET     = _np.array([_ML_META['weights_planet'][c] for c in _ML_PLANET_FEATS])
    _ML_W_CTX        = _np.array([_ML_META['weights_ctx'][c] for c in _ML_CTX_FEATS])
    _ML_ENABLED      = True
except Exception as _e:
    _ML_ENABLED = False
    _ML_ERR = repr(_e)


def _ml_compute_context(state, player):
    """Контекст карты для ML модели."""
    planets = state.planets
    own = [p for p in planets if p.owner == player]
    enemy = [p for p in planets if p.owner not in (-1, player)]
    neutral = [p for p in planets if p.owner == -1]
    own_ships = sum(p.ships for p in own)
    enemy_ships = sum(p.ships for p in enemy)
    own_prod = sum(p.production for p in own)
    enemy_prod = sum(p.production for p in enemy)
    total_ships = max(own_ships + enemy_ships + sum(p.ships for p in neutral), 1)
    total_prod = max(own_prod + enemy_prod + sum(p.production for p in neutral), 1)
    total_planets = max(len(planets), 1)
    try:
        from shooting import TOTAL_STEPS as _TS
        phase = min(1.0, state.step / max(_TS, 1))
    except Exception:
        phase = 0.5
    return {{
        'phase':              phase,
        'own_ship_ratio':     own_ships / total_ships,
        'own_prod_ratio':     own_prod / total_prod,
        'own_planet_ratio':   len(own) / total_planets,
        'n_own_planets':      len(own),
        'n_neutral_planets':  len(neutral),
        'n_enemy_planets':    len(enemy),
        'src_ships':          float(_np.mean([p.ships for p in own])) if own else 0,
        'src_prod':           float(_np.mean([p.production for p in own])) if own else 0,
        'src_wnn':            0.0,
        'src_ripeness':       0.0,
    }}


def _ml_score(df_zones, state, player):
    """Возвращает np.array(len(df)) ML scores (нормализованные)."""
    if not _ML_ENABLED or _ML_ALPHA == 0:
        return _np.zeros(len(df_zones))
    n = len(df_zones)
    # Per-planet features
    z_planet = _np.zeros((n, len(_ML_PLANET_FEATS)), dtype=_np.float32)
    for j, c in enumerate(_ML_PLANET_FEATS):
        s = _ML_FEAT_STATS.get(c, {{'mean': 0, 'std': 1}})
        vals = df_zones[c].astype(float).fillna(0).values if c in df_zones.columns else _np.zeros(n)
        z_planet[:, j] = (vals - s['mean']) / max(s['std'], 1e-6)
    # Context
    ctx = _ml_compute_context(state, player)
    z_ctx = _np.zeros(len(_ML_CTX_FEATS), dtype=_np.float32)
    for j, c in enumerate(_ML_CTX_FEATS):
        s = _ML_FEAT_STATS.get(c, {{'mean': 0, 'std': 1}})
        v = float(ctx.get(c, 0.0))
        z_ctx[j] = (v - s['mean']) / max(s['std'], 1e-6)
    raw_scores = z_planet @ _ML_W_PLANET + (z_ctx @ _ML_W_CTX)
    # z-normalize по карте
    std = raw_scores.std()
    if std < 1e-9:
        return _np.zeros(n)
    return (raw_scores - raw_scores.mean()) / std


# Wrapping оригинальной compute_zones_from_state
_original_compute_zones_from_state = compute_zones_from_state

def compute_zones_from_state(state, player=0, **kwargs):
    df, info = _original_compute_zones_from_state(state, player=player, **kwargs)
    if _ML_ENABLED and _ML_ALPHA > 0:
        try:
            ml_scores = _ml_score(df, state, player)
            df = df.copy()
            df['priority'] = df['priority'].values + _ML_ALPHA * ml_scores
        except Exception:
            pass
    return df, info
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='Вес residual ML score (0=baseline, 1=full)')
    args = ap.parse_args()

    if not MODEL_META.exists():
        sys.exit(f'⚠ Нет {MODEL_META}. Запусти 18_retrain_on_zones_features.py')
    if not SRC_BUNDLE.exists():
        sys.exit(f'⚠ Нет {SRC_BUNDLE}')

    if DST_BUNDLE.exists():
        print(f'Удаляю {DST_BUNDLE} …')
        shutil.rmtree(DST_BUNDLE)

    print(f'Копирую {SRC_BUNDLE.name} → {DST_BUNDLE.name} …')
    # ignore_patterns не копирует __pycache__
    shutil.copytree(SRC_BUNDLE, DST_BUNDLE,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc',
                                                    '.DS_Store',
                                                    '.ipynb_checkpoints'))
    print(f'  скопировано')

    # Копируем ML модель в bundle (с переименованием)
    src_meta = MODEL_META
    dst_meta = DST_BUNDLE / 'ml_zones_meta.json'
    shutil.copy(src_meta, dst_meta)
    print(f'  ML meta → {dst_meta.name}')

    # Патчим zones.py
    zones_py = DST_BUNDLE / 'zones.py'
    src = zones_py.read_text()
    patched = src + ZONES_PATCH.format(alpha=args.alpha)
    zones_py.write_text(patched)
    print(f'  патч zones.py (alpha={args.alpha})')

    print(f'\n✓ ML-bundle готов: {DST_BUNDLE}')
    print(f'  • zones.py wrapped compute_zones_from_state с ML residual')
    print(f'  • alpha = {args.alpha}')
    print(f'  • Для другого alpha — перезапусти 20_setup_ml_bundle.py --alpha NEW')
    print(f'\nДля турнира:')
    print(f'  python3 opponent_learning/rebuild/scripts/21_tournament_proper.py')


if __name__ == '__main__':
    main()
