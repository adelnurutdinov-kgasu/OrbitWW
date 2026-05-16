#!/usr/bin/env python3
"""
05_train_imitation.py — обучает XGBoost-модели имитации на пошаговом датасете.

Что обучаем
───────────
  A) Action predictors  (state → action)
     По одной модели на каждый целевой показатель действия:
       did_act, ships_fraction_sent, attack_neutral_ratio,
       attack_enemy_ratio, reinforce_ratio, avg_overkill,
       avg_target_dist_frac, multi_launch
     Используются для анализа и для нашего собственного агента.

  B) Cluster likelihood classifier  (state + action → cluster_id)
     Единая XGBClassifier на (state_cols + action_cols) → кластер оппонента.
     P(cluster=k | state, action) используется как likelihood в байесовской модели.
     Это заменяет текущую грубую angle-matching функцию на data-driven.

Входные данные
──────────────
  data/processed/steps.csv            — пошаговый датасет (вывод 02b)
  data/processed/features_clustered.csv — метки кластеров (вывод 03)

Выходные данные
───────────────
  results/models/action_<target>.json      — XGBoost action predictors
  results/models/cluster_likelihood.json  — XGBoost cluster classifier
  results/models/feature_names.json       — списки фич для inference
  results/models/cluster_priors.json      — prior P(cluster=k)

Запуск:
  python3 opponent_learning/scripts/05_train_imitation.py
  python3 opponent_learning/scripts/05_train_imitation.py --no-action-models  # только classifier
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score,
                                 mean_absolute_error, r2_score)
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    raise ImportError("pip install xgboost scikit-learn --break-system-packages")

HERE     = Path(__file__).parent
PROC_DIR = HERE.parent / "data" / "processed"
RES_DIR  = HERE.parent / "results" / "models"
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Колонки ─────────────────────────────────────────────────────────────────

STATE_COLS = [
    # фаза
    'phase', 'n_planets',
    # баланс сил
    'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
    'ship_gap', 'prod_gap',
    # детали своих планет
    'own_ships_total', 'own_prod_total', 'n_own_planets',
    'mean_own_ships', 'max_own_ships', 'min_own_ships', 'std_own_ships',
    # враг и нейтралы
    'n_neutral_planets', 'n_enemy_planets', 'enemy_ships_total',
    'mean_neutral_ships', 'min_neutral_ships',
    # флоты в воздухе (текущее)
    'own_fleets_count', 'own_fleets_ships',
    'enemy_fleets_count', 'enemy_fleets_ships', 'fleet_balance',
    # projected state (ключевые)
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

# Типы целевых переменных: 'clf' или 'reg'
ACTION_TYPES = {
    'did_act':              'clf',
    'ships_fraction_sent':  'reg',
    'attack_neutral_ratio': 'reg',
    'attack_enemy_ratio':   'reg',
    'reinforce_ratio':      'reg',
    'avg_overkill':         'reg',
    'avg_target_dist_frac': 'reg',
    'multi_launch':         'clf',
}

# XGBoost гиперпараметры (сбалансированы под 360k строк)
XGB_PARAMS_CLF = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss',
    n_jobs=-1, random_state=42,
)
XGB_PARAMS_REG = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mae',
    n_jobs=-1, random_state=42,
)
XGB_PARAMS_LIKELIHOOD = dict(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    n_jobs=-1, random_state=42,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _fill_missing(df, cols):
    """Заполняем NaN медианой и клипаем inf."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(df[c].median())
    return df


def _clip_overkill(df):
    """overkill_ratio — long tail."""
    if 'avg_overkill' in df.columns:
        cap = df['avg_overkill'].quantile(0.99)
        df['avg_overkill'] = df['avg_overkill'].clip(upper=cap)
    return df


# ── A) Action predictors ─────────────────────────────────────────────────────

def train_action_models(df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("A) Action predictors  (state → action)")
    print("═" * 70)

    X = df[STATE_COLS].values
    X_tr, X_te = train_test_split(X, test_size=0.15, random_state=42)
    # запоминаем индексы чтобы выровнять y
    tr_idx, te_idx = train_test_split(np.arange(len(df)),
                                      test_size=0.15, random_state=42)

    results = {}
    for target, kind in ACTION_TYPES.items():
        y = df[target].values
        y_tr, y_te = y[tr_idx], y[te_idx]

        if kind == 'clf':
            model = XGBClassifier(**XGB_PARAMS_CLF)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)], verbose=False)
            pred = model.predict(X_te)
            acc  = accuracy_score(y_te, pred)
            f1   = f1_score(y_te, pred, average='macro', zero_division=0)
            print(f"  {target:<26}  acc={acc:.3f}  f1={f1:.3f}")
            results[target] = {'acc': round(acc, 4), 'f1': round(f1, 4)}
        else:
            model = XGBRegressor(**XGB_PARAMS_REG)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)], verbose=False)
            pred = model.predict(X_te)
            mae  = mean_absolute_error(y_te, pred)
            r2   = r2_score(y_te, pred)
            print(f"  {target:<26}  MAE={mae:.4f}  R²={r2:.3f}")
            results[target] = {'mae': round(mae, 5), 'r2': round(r2, 4)}

        out = RES_DIR / f"action_{target}.json"
        model.save_model(str(out))

    # Сохраняем имена фич
    (RES_DIR / "feature_names.json").write_text(
        json.dumps({'state_cols': STATE_COLS, 'action_cols': ACTION_COLS}, indent=2)
    )
    print(f"\n  ✓ Модели сохранены в {RES_DIR}")
    return results


# ── B) Cluster likelihood classifier ────────────────────────────────────────

def train_likelihood_model(df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("B) Cluster likelihood  (state + action → cluster)")
    print("═" * 70)

    if 'cluster' not in df.columns:
        raise ValueError("Колонка 'cluster' не найдена — убедись что features_clustered.csv смержен")

    FEAT_COLS = STATE_COLS + ACTION_COLS
    X = df[FEAT_COLS].values
    y = df['cluster'].values

    # Кодируем кластеры
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_.tolist()
    n_cls   = len(classes)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    model = XGBClassifier(num_class=n_cls, **XGB_PARAMS_LIKELIHOOD)
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)], verbose=False)

    pred  = model.predict(X_te)
    acc   = accuracy_score(y_te, pred)
    f1    = f1_score(y_te, pred, average='macro', zero_division=0)
    print(f"  Кластеров: {n_cls}  |  acc={acc:.3f}  f1_macro={f1:.3f}")

    # Per-class accuracy
    print("\n  Per-cluster accuracy:")
    for ci, cname in enumerate(classes):
        mask = y_te == ci
        if mask.sum() == 0:
            continue
        cacc = accuracy_score(y_te[mask], pred[mask])
        n    = mask.sum()
        print(f"    cluster {cname}: acc={cacc:.3f}  n={n:>6,}")

    # Priors
    priors = {int(c): float((y == c).mean()) for c in classes}
    print(f"\n  Priors: {priors}")

    # Сохраняем
    out = RES_DIR / "cluster_likelihood.json"
    model.save_model(str(out))
    (RES_DIR / "cluster_priors.json").write_text(json.dumps(priors, indent=2))
    (RES_DIR / "cluster_classes.json").write_text(json.dumps(classes, indent=2))

    print(f"\n  ✓ cluster_likelihood.json → {out}")
    return {'acc': round(acc, 4), 'f1': round(f1, 4), 'n_clusters': n_cls}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',    default=str(PROC_DIR / 'steps.csv'))
    parser.add_argument('--clusters', default=str(PROC_DIR / 'features_clustered.csv'))
    parser.add_argument('--no-action-models', action='store_true',
                        help='Пропустить обучение action predictors (только likelihood)')
    args = parser.parse_args()

    # ── Загрузка ─────────────────────────────────────────────────────────────
    print(f"Загружаем {args.steps} …")
    df = pd.read_csv(args.steps)
    print(f"  steps.csv: {len(df):,} строк  {len(df.columns)} колонок")

    print(f"Загружаем {args.clusters} …")
    cl = pd.read_csv(args.clusters)[['episode_id', 'player_idx', 'cluster']]
    print(f"  features_clustered.csv: {len(cl):,} строк")

    # Мержим метки кластеров
    df = df.merge(cl, on=['episode_id', 'player_idx'], how='left')
    n_labeled = df['cluster'].notna().sum()
    print(f"  Помечено кластером: {n_labeled:,} / {len(df):,} строк "
          f"({100*n_labeled/len(df):.1f}%)")

    # Предобработка
    df = _fill_missing(df, STATE_COLS + ACTION_COLS)
    df = _clip_overkill(df)

    # ── A) Action predictors ─────────────────────────────────────────────────
    if not args.no_action_models:
        train_action_models(df)

    # ── B) Likelihood classifier ─────────────────────────────────────────────
    df_labeled = df[df['cluster'].notna()].copy()
    df_labeled['cluster'] = df_labeled['cluster'].astype(int)
    print(f"\nДля likelihood модели: {len(df_labeled):,} строк с метками кластеров")
    train_likelihood_model(df_labeled)

    # ── Итог ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("✓ Все модели сохранены:")
    for f in sorted(RES_DIR.iterdir()):
        sz = f.stat().st_size / 1024
        print(f"  {f.name:<40}  {sz:>7.1f} KB")
    print("═" * 70)
    print("\nСледующий шаг:")
    print("  python3 opponent_learning/scripts/06_integrate_likelihood.py")


if __name__ == '__main__':
    main()
