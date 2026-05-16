#!/usr/bin/env python3
"""
04_update_presets.py — обновляет opponent_presets.py новыми кластерными пресетами
и обновляет likelihood-функцию в opponent_model_bayesian.py на поведенческую
(map-agnostic вместо точного (from_id, target_id) матчинга).

Что делает:
  1. Читает results/clusters.json
  2. Заменяет блок PRESETS в opponent_presets.py на data-driven пресеты
  3. Обновляет PRESET_PRIORS в opponent_model_bayesian.py (частоты кластеров)
  4. Добавляет/заменяет функцию _behavioral_likelihood в bayesian-модели

Запуск (после 03_cluster_and_fit.py):
  python3 opponent_learning/scripts/04_update_presets.py [--dry-run]
"""

import json
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime

HERE         = Path(__file__).parent
RES_DIR      = HERE.parent / "results"
BUNDLE_DIR   = HERE.parent.parent / "agent_bundle_swarm 2"
PRESETS_FILE = BUNDLE_DIR / "opponent_presets.py"
BAYES_FILE   = BUNDLE_DIR / "opponent_model_bayesian.py"


# ── Генерация кода PRESETS ────────────────────────────────────────────────────

def build_presets_dict(clusters: list) -> str:
    """Возвращает строку Python-кода — новый словарь PRESETS."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = [
        f"# ── Data-driven presets (fitted {ts} by 03_cluster_and_fit.py) ──────",
        "PRESETS: Dict[str, SwarmWeights] = {",
        "    # default — нетронутый baseline",
        "    'default': DEFAULT_WEIGHTS,",
        "",
    ]
    for cl in sorted(clusters, key=lambda x: -x['prior']):
        name  = cl['name']
        w     = cl['weights']
        prior = cl['prior']
        desc  = cl['description']
        n     = cl['n_samples']
        lines.append(
            f"    # {name}  prior={prior:.3f}  n={n}  —  {desc}")
        lines.append(f"    '{name}': SwarmWeights(")
        for k, v in w.items():
            if isinstance(v, float):
                lines.append(f"        {k}={v},")
            else:
                lines.append(f"        {k}={v},")
        lines.append("    ),")
        lines.append("")
    lines.append("}")
    return "\n".join(lines)


def build_priors_dict(clusters: list) -> str:
    """Возвращает строку Python-кода — DEFAULT_PRESET_PRIORS."""
    lines = ["# ── Calibrated priors from cluster frequencies ───────────────────────",
             "DEFAULT_PRESET_PRIORS: Dict[str, float] = {",
             "    'default': 0.05,   # fallback, низкий prior"]
    for cl in sorted(clusters, key=lambda x: -x['prior']):
        lines.append(f"    '{cl['name']}': {cl['prior']:.4f},")
    lines.append("}")
    return "\n".join(lines)


# ── Патч opponent_model_bayesian.py ──────────────────────────────────────────

BEHAVIORAL_LIKELIHOOD_CODE = '''
# ── Поведенческое сравнение (map-agnostic) ────────────────────────────────
# Вместо точного (from_id, target_id) матчинга используем поведенческие
# фичи текущего хода — работает независимо от геометрии карты.

def _behavioral_obs_features(observed_actions: list, state, opp_id: int) -> dict:
    """Вычисляет поведенческие фичи одного хода оппонента."""
    if not observed_actions:
        return {}
    import math as _math
    try:
        raw_pl = getattr(state, 'raw_planets', state.planets)
        pl_map = {p.id: p for p in raw_pl}

        own_ships = sum(max(0, int(p.ships)) for p in raw_pl if p.owner == opp_id)
        total_sent = sum(max(1, a.get('ships', 1)) for a in observed_actions)

        n_neutral = sum(
            1 for a in observed_actions
            if pl_map.get(a.get('target_id'), None) is not None
            and getattr(pl_map[a['target_id']], 'owner', -1) != opp_id
            and getattr(pl_map[a['target_id']], 'owner', -1) in (-1, 99, None)
        )
        n_enemy = len(observed_actions) - n_neutral

        dists = []
        for a in observed_actions:
            src = pl_map.get(a.get('from_id'))
            tgt = pl_map.get(a.get('target_id'))
            if src and tgt:
                d = _math.hypot(
                    getattr(tgt, 'x', 0) - getattr(src, 'x', 0),
                    getattr(tgt, 'y', 0) - getattr(src, 'y', 0),
                )
                dists.append(d)

        return {
            'expansion_bias':  n_neutral / max(1, len(observed_actions)),
            'ships_fraction':  min(1.0, total_sent / max(1, own_ships + total_sent)),
            'avg_dist':        sum(dists) / len(dists) if dists else 0.0,
            'n_actions':       len(observed_actions),
        }
    except Exception:
        return {}


def _behavioral_likelihood(obs_feat: dict, predicted: list,
                            state, opp_id: int) -> float:
    """Score [0..1] поведенческого сходства наблюдения и предсказания."""
    if not predicted or not obs_feat:
        return 1.0   # нет данных → нейтрально

    import math as _math
    try:
        raw_pl = getattr(state, 'raw_planets', state.planets)
        pl_map = {p.id: p for p in raw_pl}

        # Expansion bias предсказания
        n_neu_pred = sum(
            1 for p in predicted
            if pl_map.get(p.get('target_id')) is not None
            and getattr(pl_map[p['target_id']], 'owner', -1) in (-1, 99, None)
        )
        pred_eb = n_neu_pred / max(1, len(predicted))

        # Ships fraction предсказания
        own_ships = sum(max(0, int(p.ships)) for p in raw_pl if p.owner == opp_id)
        pred_sent = sum(max(1, p.get('ships', 1)) for p in predicted)
        pred_sf   = min(1.0, pred_sent / max(1, own_ships + pred_sent))

        # Сходство по каждому измерению
        eb_sim = 1.0 - abs(obs_feat.get('expansion_bias', 0.5) - pred_eb)
        sf_sim = 1.0 - abs(obs_feat.get('ships_fraction', 0.5) - pred_sf)
        n_sim  = 1.0 - abs(len(predicted) - obs_feat.get('n_actions', 1)) / max(
            len(predicted), obs_feat.get('n_actions', 1))

        score = 0.4 * eb_sim + 0.4 * sf_sim + 0.2 * n_sim
        return max(0.1, score)
    except Exception:
        return 1.0
'''


def patch_bayesian_model(bayes_path: Path, clusters: list, dry_run: bool):
    """Добавляет DEFAULT_PRESET_PRIORS и поведенческий likelihood в bayesian-файл."""
    src = bayes_path.read_text(encoding='utf-8')

    # 1. Добавляем DEFAULT_PRESET_PRIORS если нет
    priors_code = build_priors_dict(clusters)
    if 'DEFAULT_PRESET_PRIORS' not in src:
        # Вставляем после строки MIN_PROB
        src = re.sub(
            r'(MIN_PROB\s*=.*\n)',
            r'\1\n' + priors_code + '\n\n',
            src
        )
        print("  ✓ добавлен DEFAULT_PRESET_PRIORS")
    else:
        # Заменяем существующий блок
        src = re.sub(
            r'# ── Calibrated priors.*?^}\n',
            priors_code + '\n',
            src, flags=re.MULTILINE | re.DOTALL
        )
        print("  ✓ обновлён DEFAULT_PRESET_PRIORS")

    # 2. Добавляем поведенческие функции если нет
    if '_behavioral_likelihood' not in src:
        # Вставляем перед классом OpponentModelBayesian
        src = src.replace(
            'class OpponentModelBayesian:',
            BEHAVIORAL_LIKELIHOOD_CODE + '\nclass OpponentModelBayesian:'
        )
        print("  ✓ добавлены _behavioral_obs_features, _behavioral_likelihood")

    # 3. Обновляем __init__ модели — используем DEFAULT_PRESET_PRIORS
    if 'DEFAULT_PRESET_PRIORS' in src and 'self.priors = {k: 1.0 / n' in src:
        src = src.replace(
            'n = len(PRESETS)\n        self.priors: Dict[str, float] = {k: 1.0 / n for k in PRESETS}',
            'self.priors: Dict[str, float] = dict(DEFAULT_PRESET_PRIORS) '
            'if DEFAULT_PRESET_PRIORS else {k: 1.0/len(PRESETS) for k in PRESETS}'
        )
        print("  ✓ __init__ использует DEFAULT_PRESET_PRIORS")

    if dry_run:
        print(f"\n[DRY RUN] bayes файл не изменён. Первые 3000 символов нового кода:")
        print(src[:3000])
    else:
        bayes_path.write_text(src, encoding='utf-8')
        print(f"  ✓ сохранено: {bayes_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Показать изменения без записи файлов')
    parser.add_argument('--clusters', default=str(RES_DIR / 'clusters.json'))
    args = parser.parse_args()

    clusters_path = Path(args.clusters)
    if not clusters_path.exists():
        print(f"✗ Не найден {clusters_path}. Сначала запусти 03_cluster_and_fit.py")
        return

    with open(clusters_path) as f:
        clusters = json.load(f)
    print(f"Загружено {len(clusters)} кластеров из {clusters_path}")

    # ── Обновляем opponent_presets.py ───────────────────────────────────────
    print(f"\n── opponent_presets.py ────────────────────────────────")
    src = PRESETS_FILE.read_text(encoding='utf-8')

    new_presets = build_presets_dict(clusters)

    # Заменяем блок PRESETS: Dict[...] = { ... }
    new_src = re.sub(
        r'# ── (?:Preset definitions|Data-driven presets).*?^}\n',
        new_presets + '\n',
        src, flags=re.MULTILINE | re.DOTALL
    )
    if new_src == src:
        # Fallback: ищем PRESETS: Dict напрямую
        new_src = re.sub(
            r'PRESETS:\s*Dict\[.*?\]\s*=\s*\{.*?^\}',
            new_presets,
            src, flags=re.MULTILINE | re.DOTALL
        )

    if args.dry_run:
        print("[DRY RUN] opponent_presets.py не изменён")
        print(new_presets[:2000])
    else:
        # Бэкап
        bak = PRESETS_FILE.with_suffix('.py.bak')
        shutil.copy2(PRESETS_FILE, bak)
        PRESETS_FILE.write_text(new_src, encoding='utf-8')
        print(f"  ✓ {PRESETS_FILE.name} обновлён (бэкап: {bak.name})")

    # ── Обновляем opponent_model_bayesian.py ────────────────────────────────
    print(f"\n── opponent_model_bayesian.py ─────────────────────────")
    patch_bayesian_model(BAYES_FILE, clusters, args.dry_run)

    if not args.dry_run:
        print("\n✓ Готово. Пересобери submission:")
        print("  python3 tuning/scripts/build_submission.py")


if __name__ == '__main__':
    main()
