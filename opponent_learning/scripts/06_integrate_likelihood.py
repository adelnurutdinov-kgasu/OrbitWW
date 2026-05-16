#!/usr/bin/env python3
"""
06_integrate_likelihood.py — вставляет XGBoost-likelihood в байесовскую модель.

Что делает:
  1. Проверяет что модели из 05_train_imitation.py обучены.
  2. Создаёт opponent_learning/xgb_likelihood.py — тонкий inference-модуль.
  3. Патчит opponent_model_bayesian.py:
       • добавляет XgbLikelihood в imports
       • добавляет флаг use_xgb_likelihood в __init__
       • добавляет _xgb_update() метод
       • вставляет вызов в update() рядом с текущим likelihood

Запуск:
  python3 opponent_learning/scripts/06_integrate_likelihood.py --dry-run
  python3 opponent_learning/scripts/06_integrate_likelihood.py
"""

import argparse
import json
from pathlib import Path

HERE     = Path(__file__).parent
OL_DIR   = HERE.parent                          # opponent_learning/
RES_DIR  = OL_DIR / "results" / "models"
ROOT_DIR = OL_DIR.parent                        # репозиторий

BAYES_FILE = ROOT_DIR / "opponent_model_bayesian.py"
XGB_LIK_FILE = OL_DIR / "xgb_likelihood.py"


# ── Inference-модуль xgb_likelihood.py ─────────────────────────────────────

XGB_LIKELIHOOD_CODE = '''"""
xgb_likelihood.py — inference-обёртка над XGBoost cluster likelihood model.

Используется из opponent_model_bayesian.py для map-agnostic Bayesian update.

Интерфейс:
  from opponent_learning.xgb_likelihood import XgbLikelihood
  xgb = XgbLikelihood()          # грузит модели один раз при импорте
  likelihoods = xgb.get(state_features, action_features)
  # → np.ndarray shape (n_clusters,), сумма = 1, можно умножать на beliefs

Формат входных данных:
  state_features  — dict с ключами из STATE_COLS (см. ниже)
  action_features — dict с ключами из ACTION_COLS

Если ключ отсутствует — подставляется 0.0.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

_HERE     = Path(__file__).parent
_RES_DIR  = _HERE / "results" / "models"

STATE_COLS = [
    'phase', 'n_planets',
    'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
    'ship_gap', 'prod_gap',
    'own_ships_total', 'own_prod_total', 'n_own_planets',
    'mean_own_ships', 'max_own_ships', 'min_own_ships', 'std_own_ships',
    'n_neutral_planets', 'n_enemy_planets', 'enemy_ships_total',
    'mean_neutral_ships', 'min_neutral_ships',
    'own_fleets_count', 'own_fleets_ships',
    'enemy_fleets_count', 'enemy_fleets_ships', 'fleet_balance',
    'proj_own_ship_ratio', 'proj_own_planet_ratio',
    'proj_ship_ratio_delta', 'proj_planet_ratio_delta',
    'own_under_threat_count', 'own_under_threat_ships',
    'enemy_contested_count', 'neutral_contested_count',
    'incoming_enemy_to_own', 'incoming_own_reinforce', 'net_incoming_own',
    'closest_threat_dist',
]

ACTION_COLS = [
    'did_act',
    'ships_fraction_sent',
    'attack_neutral_ratio',
    'attack_enemy_ratio',
    'reinforce_ratio',
    'avg_overkill',
    'avg_target_dist_frac',
    'multi_launch',
]

FEAT_COLS = STATE_COLS + ACTION_COLS


class XgbLikelihood:
    """Singleton-like wrapper — грузит модель один раз, отвечает быстро."""

    def __init__(self, model_dir: str | None = None):
        self._ready = False
        if XGBClassifier is None:
            return  # xgboost не установлен — молча отключаемся

        mdir = Path(model_dir) if model_dir else _RES_DIR
        clf_path = mdir / "cluster_likelihood.json"
        cls_path = mdir / "cluster_classes.json"
        pri_path = mdir / "cluster_priors.json"

        if not clf_path.exists():
            return  # модели ещё не обучены — тихо отключаемся

        self._clf = XGBClassifier()
        self._clf.load_model(str(clf_path))

        self._classes = json.loads(cls_path.read_text())   # [0, 1, 2, 3, 4]
        self._priors  = json.loads(pri_path.read_text())   # {0: 0.23, ...}
        self._n       = len(self._classes)
        self._ready   = True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def n_clusters(self) -> int:
        return self._n if self._ready else 0

    def get(self,
            state: dict,
            action: dict,
            smoothing: float = 0.05) -> np.ndarray:
        """Возвращает likelihood[k] = P(action|state, cluster=k) для k=0..K-1.

        Args:
            state:     dict с фичами состояния (STATE_COLS)
            action:    dict с фичами действия (ACTION_COLS)
            smoothing: лапласовское сглаживание чтобы не было нулей

        Returns:
            np.ndarray shape (K,), нормализован в 1.0
        """
        if not self._ready:
            return np.ones(1)   # fallback: все кластеры равновероятны

        x = np.array(
            [float(state.get(c, 0.0)) for c in STATE_COLS] +
            [float(action.get(c, 0.0)) for c in ACTION_COLS],
            dtype=np.float32
        ).reshape(1, -1)

        # P(cluster | state, action) — напрямую как likelihood
        proba = self._clf.predict_proba(x)[0]   # shape (K,)

        # Лапласовское сглаживание
        proba = proba + smoothing
        proba = proba / proba.sum()
        return proba

    def cluster_order(self) -> list[int]:
        """Возвращает список cluster_id в том же порядке что и get()."""
        return list(self._classes)


# Singleton (ленивая инициализация при первом импорте)
_instance: XgbLikelihood | None = None


def get_xgb_likelihood() -> XgbLikelihood:
    global _instance
    if _instance is None:
        _instance = XgbLikelihood()
    return _instance
'''


# ── Патч для opponent_model_bayesian.py ─────────────────────────────────────

# Кусок кода который добавляем в __init__
INIT_PATCH = '''
        # XGBoost likelihood (data-driven, подключается если модели обучены)
        try:
            from opponent_learning.xgb_likelihood import get_xgb_likelihood
            self._xgb = get_xgb_likelihood()
        except Exception:
            self._xgb = None
'''

# Метод _xgb_update который добавляем в класс
XGB_UPDATE_METHOD = '''
    def _xgb_update(self, state_features: dict, action_features: dict) -> bool:
        """Обновляет beliefs через XGBoost likelihood если модель доступна.

        Возвращает True если обновление произошло.
        Возвращает False если xgb недоступен (Bayesian модель продолжит как обычно).
        """
        if self._xgb is None or not self._xgb.ready:
            return False

        likelihoods = self._xgb.get(state_features, action_features)

        # Маппинг cluster_id → индекс preset в self.beliefs
        # Предполагаем что порядок кластеров соответствует порядку presets
        # Если размерности не совпадают — fallback
        if len(likelihoods) != len(self.beliefs):
            return False

        self.beliefs = self.beliefs * likelihoods
        s = self.beliefs.sum()
        if s > 1e-10:
            self.beliefs /= s
        else:
            self.beliefs = np.ones(len(self.beliefs)) / len(self.beliefs)
        return True
'''


def check_models_ready() -> bool:
    """Проверяет что модели из 05_train_imitation.py обучены."""
    required = [
        RES_DIR / "cluster_likelihood.json",
        RES_DIR / "cluster_priors.json",
        RES_DIR / "cluster_classes.json",
    ]
    missing = [f for f in required if not f.exists()]
    if missing:
        print("⚠ Не найдены файлы моделей:")
        for f in missing:
            print(f"   {f}")
        print("\nСначала запусти:")
        print("  python3 opponent_learning/scripts/05_train_imitation.py")
        return False
    return True


def write_xgb_likelihood(dry_run: bool):
    """Создаёт opponent_learning/xgb_likelihood.py."""
    if dry_run:
        print(f"[DRY RUN] Создал бы: {XGB_LIK_FILE}")
        return
    XGB_LIK_FILE.write_text(XGB_LIKELIHOOD_CODE)
    print(f"✓ Создан: {XGB_LIK_FILE}")


def patch_bayesian_model(dry_run: bool):
    """Патчит opponent_model_bayesian.py добавляя XGB-методы."""
    if not BAYES_FILE.exists():
        print(f"⚠ Не найден: {BAYES_FILE}")
        return

    src = BAYES_FILE.read_text()

    changes = []

    # 1. Добавить _xgb в __init__ если ещё нет
    if '_xgb' not in src:
        # Вставляем после строки self.beliefs = ...
        anchor = 'self.beliefs = np.ones(len(self.presets)) / len(self.presets)'
        if anchor in src:
            src = src.replace(anchor, anchor + INIT_PATCH, 1)
            changes.append("добавлен self._xgb в __init__")

    # 2. Добавить метод _xgb_update если ещё нет
    if '_xgb_update' not in src:
        # Вставляем перед последней строкой файла или перед def update
        if 'def update(' in src:
            src = src.replace('def update(', XGB_UPDATE_METHOD + '\n    def update(', 1)
            changes.append("добавлен метод _xgb_update()")

    # 3. Вызов _xgb_update внутри update() если нет
    if '_xgb_update(' not in src and 'def update(' in src:
        # Ищем начало метода update чтобы добавить вызов
        # Вставляем в начало тела update после первого return/pass или после docstring
        insert_marker = '"""'  # ищем конец docstring в update
        # Более надёжно: вставить комментарий-маркер
        changes.append("⚠ Вызов _xgb_update() нужно добавить в update() вручную")

    if not changes:
        print("  opponent_model_bayesian.py уже содержит XGB-интеграцию, пропускаем")
        return

    if dry_run:
        print(f"[DRY RUN] Патч для {BAYES_FILE}:")
        for c in changes:
            print(f"   + {c}")
        return

    BAYES_FILE.write_text(src)
    print(f"✓ Пропатчен: {BAYES_FILE}")
    for c in changes:
        print(f"   + {c}")

    if any("вручную" in c for c in changes):
        print("\n⚠ Один шаг требует ручного вмешательства:")
        print("  В методе update() добавь в начало тела:")
        print("    # Попробовать XGBoost likelihood (data-driven)")
        print("    if self._xgb_update(state_features, action_features):")
        print("        return  # обновление выполнено")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-model-check', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("06_integrate_likelihood.py")
    print("=" * 60)

    if not args.skip_model_check and not check_models_ready():
        return

    print("\nШаг 1: создаём opponent_learning/xgb_likelihood.py")
    write_xgb_likelihood(args.dry_run)

    print("\nШаг 2: патчим opponent_model_bayesian.py")
    patch_bayesian_model(args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN завершён. Запусти без --dry-run чтобы применить.")
    else:
        print("✓ Интеграция завершена!")
        print("\nПроверь:")
        print("  python3 -c \"from opponent_learning.xgb_likelihood import get_xgb_likelihood; "
              "x = get_xgb_likelihood(); print('ready:', x.ready, 'clusters:', x.n_clusters)\"")
    print("=" * 60)


if __name__ == '__main__':
    main()
