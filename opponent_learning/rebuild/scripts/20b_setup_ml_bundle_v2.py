#!/usr/bin/env python3
"""
20b_setup_ml_bundle_v2.py — изолированный ML-агент с уникальным zones_ml.

ПОЧЕМУ V2
─────────
В 20_* был bug: Python кэширует `sys.modules['zones']` после первого
`import zones`. Когда baseline и ML грузятся из разных папок, второй
агент получает **уже загруженный** zones из первого (baseline). Patches
в ML-zones игнорируются. Это источник асимметрии в 21_tournament_proper.

ФИКС
────
Переименовать `zones.py` → `zones_ml.py` в agent_bundle_ml/.
Заменить во ВСЕХ файлах bundle:
    `from zones import` → `from zones_ml import`
    `import zones`      → `import zones_ml as zones`
Теперь sys.modules['zones'] и sys.modules['zones_ml'] — разные namespace.

ВЫЗОВ
─────
  python3 20b_setup_ml_bundle_v2.py --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import re
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


ZONES_PATCH = '''

# ══════════════════════════════════════════════════════════════════════════
# ML RESIDUAL (добавлено 20b_setup_ml_bundle_v2.py)
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
except Exception:
    _ML_ENABLED = False


def _ml_compute_context(state, player):
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
        'phase': phase,
        'own_ship_ratio': own_ships / total_ships,
        'own_prod_ratio': own_prod / total_prod,
        'own_planet_ratio': len(own) / total_planets,
        'n_own_planets': len(own),
        'n_neutral_planets': len(neutral),
        'n_enemy_planets': len(enemy),
        'src_ships': float(_np.mean([p.ships for p in own])) if own else 0,
        'src_prod': float(_np.mean([p.production for p in own])) if own else 0,
        'src_wnn': 0.0,
        'src_ripeness': 0.0,
    }}


def _ml_score(df_zones, state, player):
    if not _ML_ENABLED or _ML_ALPHA == 0:
        return _np.zeros(len(df_zones))
    n = len(df_zones)
    z_planet = _np.zeros((n, len(_ML_PLANET_FEATS)), dtype=_np.float32)
    for j, c in enumerate(_ML_PLANET_FEATS):
        s = _ML_FEAT_STATS.get(c, {{'mean': 0, 'std': 1}})
        vals = df_zones[c].astype(float).fillna(0).values if c in df_zones.columns else _np.zeros(n)
        z_planet[:, j] = (vals - s['mean']) / max(s['std'], 1e-6)
    ctx = _ml_compute_context(state, player)
    z_ctx = _np.zeros(len(_ML_CTX_FEATS), dtype=_np.float32)
    for j, c in enumerate(_ML_CTX_FEATS):
        s = _ML_FEAT_STATS.get(c, {{'mean': 0, 'std': 1}})
        v = float(ctx.get(c, 0.0))
        z_ctx[j] = (v - s['mean']) / max(s['std'], 1e-6)
    raw_scores = z_planet @ _ML_W_PLANET + (z_ctx @ _ML_W_CTX)
    std = raw_scores.std()
    if std < 1e-9:
        return _np.zeros(n)
    return (raw_scores - raw_scores.mean()) / std


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


def _rewrite_imports_in_file(path: Path):
    """Заменяет `from zones import` → `from zones_ml import` и
    `import zones` → `import zones_ml as zones` (последнее сохраняет
    локальный alias чтобы не сломать остальной код)."""
    try:
        text = path.read_text()
    except Exception:
        return False
    original = text
    # `from zones import X` → `from zones_ml import X`
    text = re.sub(r'\bfrom\s+zones\s+import\b', 'from zones_ml import', text)
    # `import zones as zones_mod` → `import zones_ml as zones_mod`
    text = re.sub(r'\bimport\s+zones\s+as\s+(\w+)', r'import zones_ml as \1', text)
    # `import zones` (без as) → `import zones_ml as zones` (alias чтобы код работал)
    text = re.sub(r'^(\s*)import\s+zones\s*$', r'\1import zones_ml as zones',
                   text, flags=re.MULTILINE)

    if text != original:
        path.write_text(text)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.5)
    args = ap.parse_args()

    if not MODEL_META.exists():
        sys.exit(f'⚠ Нет {MODEL_META}. Запусти 18_retrain_on_zones_features.py')
    if not SRC_BUNDLE.exists():
        sys.exit(f'⚠ Нет {SRC_BUNDLE}')

    if DST_BUNDLE.exists():
        print(f'Удаляю {DST_BUNDLE} …')
        shutil.rmtree(DST_BUNDLE)

    print(f'Копирую {SRC_BUNDLE.name} → {DST_BUNDLE.name} …')
    shutil.copytree(SRC_BUNDLE, DST_BUNDLE,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc',
                                                    '.DS_Store',
                                                    '.ipynb_checkpoints'))

    # 1. Переименовать zones.py → zones_ml.py
    src_zones = DST_BUNDLE / 'zones.py'
    dst_zones = DST_BUNDLE / 'zones_ml.py'
    if src_zones.exists():
        src_zones.rename(dst_zones)
        print(f'  переименовано: zones.py → zones_ml.py')
    else:
        sys.exit(f'⚠ Нет {src_zones}')

    # 2. Применить ML патч в zones_ml.py
    src = dst_zones.read_text()
    patched = src + ZONES_PATCH.format(alpha=args.alpha)
    dst_zones.write_text(patched)
    print(f'  патч ML residual в zones_ml.py (alpha={args.alpha})')

    # 3. Скопировать model meta
    shutil.copy(MODEL_META, DST_BUNDLE / 'ml_zones_meta.json')
    print(f'  ml_zones_meta.json скопирован')

    # 4. Обновить все импорты в .py файлах bundle
    print(f'\n  Перепись `from zones` → `from zones_ml` во всех .py файлах bundle …')
    n_changed = 0
    for py_file in DST_BUNDLE.glob('*.py'):
        if py_file.name == 'zones_ml.py':
            continue  # сам zones_ml не трогаем
        if _rewrite_imports_in_file(py_file):
            n_changed += 1
            print(f'    ✓ {py_file.name}')

    print(f'\n  Изменено файлов: {n_changed}')
    print(f'\n✓ ML-bundle готов: {DST_BUNDLE}')
    print(f'  alpha={args.alpha}, zones_ml.py с ML residual')
    print(f'\nДля турнира:')
    print(f'  python3 opponent_learning/rebuild/scripts/21_tournament_proper.py --seeds 15 --swap')


if __name__ == '__main__':
    main()
