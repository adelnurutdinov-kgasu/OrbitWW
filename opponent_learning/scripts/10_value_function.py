#!/usr/bin/env python3
"""
10_value_function.py — V(state) → P(win), обучение на steps.csv.

Задача
──────
Функция ценности (value function) отвечает на вопрос:
  «Из этого состояния, насколько вероятна наша победа?»

Это graph-level предсказание: вход — вся игровая ситуация на шаге,
выход — вероятность победы (reward → 0/1).

Данные
──────
steps.csv (≈360k строк): каждая строка = (episode_id, player_idx, step, …фичи…, reward)
reward ∈ {-1, 1} → label = (reward + 1) / 2 ∈ {0, 1}

Архитектура (baseline XGBoost)
──────────────────────────────
XGBoostClassifier, predict_proba → P(win)

Почему XGBoost, а не нейросеть?
  - steps.csv уже содержит агрегированные фичи (ratio, total, mean, …)
  - Нейросеть поверх таких фич даёт ~ту же точность, но намного дольше обучается
  - XGBoost — отличный baseline; GNN (11_gnn_base.py) будет лучше,
    т.к. видит планеты по отдельности, а не только агрегаты

Выходы
──────
  models/value_function.json          — XGBoost модель
  models/value_function_meta.json     — метрики, фичи, threshold
  plots/value_calibration.png         — calibration curve (если matplotlib есть)

Использование модели
────────────────────
  from xgboost import XGBClassifier
  import json, numpy as np

  model = XGBClassifier()
  model.load_model('opponent_learning/results/models/value_function.json')
  meta  = json.load(open('opponent_learning/results/models/value_function_meta.json'))

  row = { … }   # dict с фичами из meta['features']
  X   = np.array([[row[f] for f in meta['features']]])
  p_win = model.predict_proba(X)[0, 1]

Запуск:
  python3 opponent_learning/scripts/10_value_function.py
  python3 opponent_learning/scripts/10_value_function.py --out-dir /tmp/vf_test
  python3 opponent_learning/scripts/10_value_function.py --calibrate  # + isotonic calibration
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE     = Path(__file__).parent
OL_DIR   = HERE.parent
PROC_DIR = OL_DIR / "data" / "processed"
RES_DIR  = OL_DIR / "results"
MODEL_DIR = RES_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STEPS_CSV = PROC_DIR / "steps.csv"

# Фичи состояния (не включаем target/action фичи — только state)
STATE_FEATURES = [
    # Соотношения сил
    'own_ship_ratio',
    'own_prod_ratio',
    'own_planet_ratio',
    'ship_gap',
    'prod_gap',
    # Абсолютные значения
    'own_ships_total',
    'own_prod_total',
    'n_own_planets',
    'n_neutral_planets',
    'n_enemy_planets',
    'enemy_ships_total',
    # Статистики по своим планетам
    'mean_own_ships',
    'max_own_ships',
    'min_own_ships',
    'std_own_ships',
    # Флоты
    'own_fleets_count',
    'own_fleets_ships',
    'enemy_fleets_count',
    'enemy_fleets_ships',
    'fleet_balance',
    # Проекции
    'proj_own_ship_ratio',
    'proj_own_planet_ratio',
    'proj_ship_ratio_delta',
    'proj_planet_ratio_delta',
    # Угрозы
    'own_under_threat_count',
    'own_under_threat_ships',
    'enemy_contested_count',
    'neutral_contested_count',
    'incoming_enemy_to_own',
    'net_incoming_own',
    'closest_threat_dist',
    # Нейтральные
    'mean_neutral_ships',
    'min_neutral_ships',
    # Фаза
    'phase',
    # Действия (поведенческий контекст — опционально)
    'did_act',
    'n_launches',
    'ships_fraction_sent',
    'attack_enemy_ratio',
    'attack_neutral_ratio',
    'reinforce_ratio',
    'avg_overkill',
    'avg_target_dist_frac',
    'multi_launch',
]


def load_data(steps_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    print(f"Loading {steps_path} …")
    df = pd.read_csv(steps_path)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Episodes: {df['episode_id'].nunique():,}")
    print(f"  Win rate (label=1): {(df['reward'] > 0).mean():.3f}")

    # Target: reward → 0/1
    df['label'] = (df['reward'] > 0).astype(int)

    # Отбираем доступные фичи
    feats = [f for f in STATE_FEATURES if f in df.columns]
    missing = [f for f in STATE_FEATURES if f not in df.columns]
    if missing:
        print(f"  ⚠ Missing features (will skip): {missing}")
    print(f"  Using {len(feats)} features")

    X = df[feats].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y = df['label'].values.astype(np.int32)
    return X, y, feats


def split_by_episode(df_full: pd.DataFrame, val_frac: float = 0.15):
    """Train/val split по episode_id (чтобы не было лика между шагами)."""
    eids = df_full['episode_id'].unique()
    rng  = np.random.default_rng(42)
    rng.shuffle(eids)
    n_val = max(1, int(len(eids) * val_frac))
    val_eids = set(eids[:n_val])
    return df_full['episode_id'].isin(val_eids)


def train(args):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("⚠ xgboost not installed: pip install xgboost --break-system-packages")
        return

    # ── Загрузка ──────────────────────────────────────────────────
    df = pd.read_csv(STEPS_CSV)
    df['label'] = (df['reward'] > 0).astype(int)
    feats = [f for f in STATE_FEATURES if f in df.columns]

    is_val = split_by_episode(df, val_frac=0.15)
    df_tr  = df[~is_val]
    df_val = df[is_val]

    def _X(d): return d[feats].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    def _y(d): return d['label'].values.astype(np.int32)

    X_tr, y_tr   = _X(df_tr),  _y(df_tr)
    X_val, y_val = _X(df_val), _y(df_val)

    print(f"\nTrain: {len(X_tr):,} rows ({df_tr['episode_id'].nunique():,} episodes)")
    print(f"Val:   {len(X_val):,} rows ({df_val['episode_id'].nunique():,} episodes)")

    # ── Обучение ──────────────────────────────────────────────────
    print("\nTraining XGBoost …")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,   # регуляризация (шаги коррелированы внутри эпизода)
        reg_lambda=1.5,
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=50)

    # ── Метрики ───────────────────────────────────────────────────
    from sklearn.metrics import (roc_auc_score, log_loss,
                                  brier_score_loss, accuracy_score)

    p_val = model.predict_proba(X_val)[:, 1]
    auc   = roc_auc_score(y_val, p_val)
    ll    = log_loss(y_val, p_val)
    brier = brier_score_loss(y_val, p_val)
    acc   = accuracy_score(y_val, (p_val > 0.5).astype(int))

    print(f"\n── Val metrics ───────────────────────────────")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Log-loss:    {ll:.4f}")
    print(f"  Brier score: {brier:.4f}")
    print(f"  Accuracy:    {acc:.4f}")

    # Метрики по фазе партии
    df_val2 = df_val.copy()
    df_val2['p_win'] = p_val
    df_val2['correct'] = (df_val2['p_win'] > 0.5) == (df_val2['label'] == 1)
    print("\n── Accuracy by game phase ────────────────────")
    bins = pd.cut(df_val2['phase'], bins=[0, 0.25, 0.5, 0.75, 1.01],
                  labels=['early', 'mid-early', 'mid-late', 'late'])
    print(df_val2.groupby(bins, observed=True)['correct'].mean().round(3).to_string())

    # Важность фич
    imp = pd.Series(model.feature_importances_, index=feats)
    imp = imp.sort_values(ascending=False)
    print("\n── Top-15 feature importances ────────────────")
    print(imp.head(15).round(4).to_string())

    # Опциональная калибровка
    calibrated_model = None
    if args.calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        print("\nApplying isotonic calibration …")
        # Переобучаем на val как hold-out для калибровки
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_val, y_val)
        p_cal = calibrated_model.predict_proba(X_val)[:, 1]
        print(f"  Brier after calibration: {brier_score_loss(y_val, p_cal):.4f}")

    # ── Сохранение ────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / 'value_function.json'
    model.save_model(str(model_path))
    print(f"\n✓ Model saved: {model_path}")

    meta = {
        'features':     feats,
        'n_train':      int(len(X_tr)),
        'n_val':        int(len(X_val)),
        'n_episodes_tr': int(df_tr['episode_id'].nunique()),
        'n_episodes_val': int(df_val['episode_id'].nunique()),
        'metrics': {
            'val_auc':   round(auc, 4),
            'val_logloss': round(ll, 4),
            'val_brier': round(brier, 4),
            'val_acc':   round(acc, 4),
        },
        'top_features': imp.head(20).round(4).to_dict(),
        'threshold': 0.5,
        'note': (
            'Predicts P(win) from aggregated state features. '
            'GNN (11_gnn_base.py) can improve by seeing individual planets.'
        ),
    }
    meta_path = out_dir / 'value_function_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✓ Meta saved:  {meta_path}")

    # ── Калибровочный plot ─────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        fig, ax = plt.subplots(figsize=(7, 5))
        frac_pos, mean_pred = calibration_curve(y_val, p_val, n_bins=20)
        ax.plot(mean_pred, frac_pos, 's-', label='XGBoost')
        if calibrated_model is not None:
            p_cal2 = calibrated_model.predict_proba(X_val)[:, 1]
            fp2, mp2 = calibration_curve(y_val, p_cal2, n_bins=20)
            ax.plot(mp2, fp2, 's-', label='XGBoost + isotonic')
        ax.plot([0, 1], [0, 1], 'k--', label='perfect')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Value Function — Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / 'value_calibration.png'
        plt.savefig(plot_path, dpi=120)
        print(f"✓ Plot saved:  {plot_path}")
    except ImportError:
        print("  (matplotlib not available — skipping calibration plot)")

    print(f"\n{'═'*56}")
    print("VALUE FUNCTION SUMMARY")
    print(f"{'═'*56}")
    print(f"  AUC {auc:.3f}  |  Accuracy {acc:.3f}  |  Brier {brier:.4f}")
    print(f"  Best feature: {imp.index[0]}  ({imp.iloc[0]:.4f})")
    print()
    print("Usage in agent:")
    print("  from xgboost import XGBClassifier")
    print("  vf = XGBClassifier()")
    print(f"  vf.load_model('{model_path}')")
    print("  p_win = vf.predict_proba(X)[0, 1]")
    print(f"{'═'*56}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train value function V(state) → P(win)')
    parser.add_argument('--out-dir',   default=None,
                        help='Directory for model output (default: results/models/)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Apply isotonic calibration after training')
    args = parser.parse_args()

    if not STEPS_CSV.exists():
        print(f"⚠ Not found: {STEPS_CSV}")
        print("  Run first: python3 opponent_learning/scripts/02b_extract_steps.py")
        return

    train(args)


if __name__ == '__main__':
    main()
