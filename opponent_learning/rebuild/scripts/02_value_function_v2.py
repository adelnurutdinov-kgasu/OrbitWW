#!/usr/bin/env python3
"""
02_value_function_v2.py — переобучает value function без endogenous фич.

ЧТО МЫ ЧИНИМ (см. AUDIT.md, пункт #2)
─────────────────────────────────────
Оригинал (10_value_function.py) включал в STATE_FEATURES:
  • action features: did_act, n_launches, ships_fraction_sent,
    attack_enemy_ratio, attack_neutral_ratio, reinforce_ratio,
    avg_overkill, avg_target_dist_frac, multi_launch
  • phase (доля времени партии) — leak через концовку

В итоге AUC 0.91, но это не P(win|state), а классификатор "это победитель?"
по leaked features. Top importance: prod_gap (0.36) — endogenous follower скилла.

ИЗМЕНЕНИЯ В V2
──────────────
  1) STATE_FEATURES_CLEAN = только структурные/макро-фичи (33 шт.),
     без action и без phase
  2) Стратификация метрик по phase quartiles (early/mid-early/mid-late/late)
  3) Стратификация по n_players (1v1 vs FFA-4) через mergе с episodes_meta.csv
  4) Сравнение с тривиальными бейзлайнами:
     • constant: P(win) = train winrate (для каждой страты отдельно)
     • linear:  логистическая регрессия на 5 макро-фичах
     • xgb_macro: XGBoost на тех же 5 макро-фичах
     Это покажет реальный "lift over baseline" — главная цифра нетривиальности.
  5) Train/val split — по episode_id (как было правильно в v1)
  6) Сохранение в rebuild/models/ + rebuild/reports/

ВХОД
────
  data/processed/steps.csv         — существующий per-step датасет
  rebuild/data/episodes_meta.csv   — n_players по эпизодам (вывод 01)

ВЫХОД
─────
  rebuild/models/value_function_v2.json        — XGBoost модель
  rebuild/models/value_function_v2_meta.json   — метрики, фичи
  rebuild/reports/value_v2.md                  — отчёт с разборкой по слоям
  rebuild/reports/value_v2_metrics.json        — все числа в JSON

ТЯЖЁЛЫЙ скрипт (~3-5 минут на 360K строк, обучение XGBoost).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE         = Path(__file__).resolve().parent
REBUILD_DIR  = HERE.parent
OL_DIR       = REBUILD_DIR.parent
PROC_DIR     = OL_DIR / "data" / "processed"
META_PATH    = REBUILD_DIR / "data" / "episodes_meta.csv"
MODEL_DIR    = REBUILD_DIR / "models"
REPORT_DIR   = REBUILD_DIR / "reports"
for d in (MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

STEPS_CSV    = PROC_DIR / "steps.csv"

# ── Фичи V2 — БЕЗ action, БЕЗ phase ─────────────────────────────────────
STATE_FEATURES_CLEAN = [
    # Resource ratios
    'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
    'ship_gap', 'prod_gap',
    # Absolutes
    'own_ships_total', 'own_prod_total',
    'n_own_planets', 'n_neutral_planets', 'n_enemy_planets',
    'enemy_ships_total',
    'n_planets',
    # Own planet distributions
    'mean_own_ships', 'max_own_ships', 'min_own_ships', 'std_own_ships',
    # Fleets in transit (state, not action)
    'own_fleets_count', 'own_fleets_ships',
    'enemy_fleets_count', 'enemy_fleets_ships', 'fleet_balance',
    # Projected state (структурный forward-look, не выбор игрока)
    'proj_own_ship_ratio', 'proj_own_planet_ratio',
    'proj_ship_ratio_delta', 'proj_planet_ratio_delta',
    # Threat structural
    'own_under_threat_count', 'own_under_threat_ships',
    'enemy_contested_count', 'neutral_contested_count',
    'incoming_enemy_to_own', 'net_incoming_own',
    'closest_threat_dist',
    # Neutrals
    'mean_neutral_ships', 'min_neutral_ships',
]

# Бейзлайн: только 5 наиболее очевидных макро-фич
MACRO_BASELINE_FEATURES = [
    'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
    'ship_gap', 'prod_gap',
]


# ── helpers ──────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Заполнить NaN/inf нулями, привести к float32."""
    sub = df[cols].copy()
    return (sub.fillna(0)
              .replace([np.inf, -np.inf], 0)
              .values.astype(np.float32))


def split_by_episode(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    """Разбиваем train/val по episode_id (никакого leakage между шагами)."""
    eids = df['episode_id'].astype(str).unique()
    rng  = np.random.default_rng(seed)
    rng.shuffle(eids)
    n_val = max(1, int(len(eids) * val_frac))
    val_set = set(eids[:n_val])
    return df['episode_id'].astype(str).isin(val_set)


def metrics_for(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """AUC / LogLoss / Brier / Accuracy. Возвращает NaN-safe значения."""
    from sklearn.metrics import (roc_auc_score, log_loss,
                                  brier_score_loss, accuracy_score)
    out = {}
    # AUC требует обоих классов в y_true
    if len(np.unique(y_true)) > 1:
        out['auc']    = round(float(roc_auc_score(y_true, p_pred)), 4)
        out['logloss']= round(float(log_loss(y_true, p_pred,
                                             labels=[0, 1])), 4)
        out['brier']  = round(float(brier_score_loss(y_true, p_pred)), 4)
    else:
        out['auc'] = out['logloss'] = out['brier'] = None
    out['acc'] = round(float(accuracy_score(y_true, (p_pred > 0.5).astype(int))), 4)
    out['n']   = int(len(y_true))
    out['pos_rate'] = round(float(y_true.mean()), 4)
    return out


def stratified_metrics(y_true, p_pred, strata: pd.Series, name: str) -> dict:
    """Метрики по каждой страте."""
    out = {}
    for s in sorted(strata.dropna().unique()):
        mask = strata == s
        if mask.sum() < 50:
            continue
        out[str(s)] = metrics_for(y_true[mask.values], p_pred[mask.values])
    return out


# ── main ─────────────────────────────────────────────────────────────────

def train(args):
    try:
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print('pip install xgboost scikit-learn --break-system-packages')
        sys.exit(1)

    # ── Загрузка ──────────────────────────────────────────────────────────
    print(f'Loading {STEPS_CSV} …')
    df = pd.read_csv(STEPS_CSV)
    print(f'  steps.csv: {len(df):,} строк, {df.shape[1]} колонок')
    df['label'] = (df['reward'] > 0).astype(int)
    print(f'  win rate (label=1) overall: {df["label"].mean():.3f}')

    # Мержим n_players если файл есть
    if META_PATH.exists():
        meta = pd.read_csv(META_PATH)[['episode_id', 'n_players']]
        meta['episode_id'] = meta['episode_id'].astype(str)
        df['episode_id'] = df['episode_id'].astype(str)
        df = df.merge(meta, on='episode_id', how='left')
        n_with_meta = df['n_players'].notna().sum()
        print(f'  merged n_players: {n_with_meta}/{len(df)} строк '
              f'({100*n_with_meta/len(df):.1f}%)')
    else:
        print(f'  ⚠ {META_PATH} не найден — стратификация по n_players пропускается.')
        print(f'     Запустите сначала 01_episode_metadata.py')
        df['n_players'] = np.nan

    # ── Featuresready ─────────────────────────────────────────────────────
    feats = [f for f in STATE_FEATURES_CLEAN if f in df.columns]
    missing = [f for f in STATE_FEATURES_CLEAN if f not in df.columns]
    if missing:
        print(f'  ⚠ отсутствует {len(missing)} фич: {missing[:5]}…')
    print(f'  использовано {len(feats)} state-фич (БЕЗ action, БЕЗ phase)')

    macro_feats = [f for f in MACRO_BASELINE_FEATURES if f in df.columns]

    # ── Split by episode ──────────────────────────────────────────────────
    is_val = split_by_episode(df, val_frac=0.15, seed=42)
    df_tr  = df[~is_val].copy()
    df_val = df[is_val].copy()
    print(f'\nTrain: {len(df_tr):,} строк ({df_tr["episode_id"].nunique()} эпизодов)')
    print(f'Val:   {len(df_val):,} строк ({df_val["episode_id"].nunique()} эпизодов)')

    X_tr  = _clean(df_tr,  feats)
    y_tr  = df_tr['label'].values.astype(np.int32)
    X_val = _clean(df_val, feats)
    y_val = df_val['label'].values.astype(np.int32)

    X_tr_m  = _clean(df_tr,  macro_feats)
    X_val_m = _clean(df_val, macro_feats)

    # ── Бейзлайны ─────────────────────────────────────────────────────────
    print('\n── Baselines ─────────────────────────────────────────')
    train_winrate = float(y_tr.mean())
    p_const = np.full(len(y_val), train_winrate)
    m_const = metrics_for(y_val, p_const)
    print(f'  constant winrate ({train_winrate:.3f}):  '
          f'AUC={m_const["auc"]}  Brier={m_const["brier"]}  Acc={m_const["acc"]}')

    print('  Linear logistic on 5 macro features …')
    lin = LogisticRegression(max_iter=200)
    lin.fit(X_tr_m, y_tr)
    p_lin = lin.predict_proba(X_val_m)[:, 1]
    m_lin = metrics_for(y_val, p_lin)
    print(f'  linear macro:  AUC={m_lin["auc"]}  '
          f'Brier={m_lin["brier"]}  Acc={m_lin["acc"]}')

    print('  XGBoost on 5 macro …')
    xgb_m = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(X_tr_m, y_tr, eval_set=[(X_val_m, y_val)], verbose=False)
    p_xgbm = xgb_m.predict_proba(X_val_m)[:, 1]
    m_xgbm = metrics_for(y_val, p_xgbm)
    print(f'  xgb macro:     AUC={m_xgbm["auc"]}  '
          f'Brier={m_xgbm["brier"]}  Acc={m_xgbm["acc"]}')

    # ── V2 модель ─────────────────────────────────────────────────────────
    print('\n── Value V2 (XGBoost on cleaned state) ──────────────')
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=1.5,
        eval_metric='logloss', early_stopping_rounds=30,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    p_val = model.predict_proba(X_val)[:, 1]
    m_v2  = metrics_for(y_val, p_val)
    print(f'  V2 overall:    AUC={m_v2["auc"]}  '
          f'Brier={m_v2["brier"]}  Acc={m_v2["acc"]}')

    # ── Lift over baselines ──────────────────────────────────────────────
    lift = {
        'auc_lift_over_constant': round(m_v2['auc'] - 0.5, 4),
        'auc_lift_over_linear':   round(m_v2['auc'] - (m_lin['auc'] or 0.5), 4),
        'auc_lift_over_xgb_macro':round(m_v2['auc'] - (m_xgbm['auc'] or 0.5), 4),
        'brier_redux_over_constant':
            round((m_const['brier'] - m_v2['brier']) / m_const['brier'], 4),
    }
    print(f'\n  Lift over xgb_macro:  +{lift["auc_lift_over_xgb_macro"]:.4f} AUC')
    print(f'  Brier reduction:      {100*lift["brier_redux_over_constant"]:.1f}% vs constant')

    # ── Stratify by phase ────────────────────────────────────────────────
    print('\n── Stratification by phase quartile ──────────────────')
    if 'phase' in df_val.columns:
        phase_q = pd.cut(df_val['phase'].values,
                          bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
                          labels=['q1_early', 'q2_mid', 'q3_late_mid', 'q4_late'])
        strat_phase = stratified_metrics(y_val, p_val,
                                          pd.Series(phase_q, index=df_val.index),
                                          'phase')
        for s, m in strat_phase.items():
            print(f'  {s:<14} n={m["n"]:>6,}  AUC={m["auc"]}  '
                  f'Acc={m["acc"]}  pos_rate={m["pos_rate"]}')
    else:
        strat_phase = {}

    # ── Stratify by n_players ────────────────────────────────────────────
    print('\n── Stratification by n_players ───────────────────────')
    strat_np = {}
    if 'n_players' in df_val.columns and df_val['n_players'].notna().any():
        strat_np = stratified_metrics(y_val, p_val,
                                       df_val['n_players'].astype('Int64').astype(str),
                                       'n_players')
        for s, m in strat_np.items():
            print(f'  n_players={s:<5} n={m["n"]:>6,}  AUC={m["auc"]}  '
                  f'Acc={m["acc"]}  pos_rate={m["pos_rate"]}')
    else:
        print('  (нет данных n_players — пропускаем)')

    # ── Feature importances ──────────────────────────────────────────────
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print(f'\n── Top-10 feature importances ────────────────────────')
    print(imp.head(10).round(4).to_string())

    # ── Save model + meta ───────────────────────────────────────────────
    model_path = MODEL_DIR / 'value_function_v2.json'
    model.save_model(str(model_path))

    all_metrics = {
        'overall':        m_v2,
        'baselines': {
            'constant':   m_const,
            'linear_5m':  m_lin,
            'xgb_5m':     m_xgbm,
        },
        'lift':           lift,
        'by_phase':       strat_phase,
        'by_n_players':   strat_np,
        'features':       feats,
        'macro_features': macro_feats,
        'n_train':        int(len(X_tr)),
        'n_val':          int(len(X_val)),
        'n_train_episodes': int(df_tr['episode_id'].nunique()),
        'n_val_episodes':   int(df_val['episode_id'].nunique()),
        'top_features':   imp.head(15).round(4).to_dict(),
        'note': (
            'V2: cleaned state features only (no action, no phase). '
            'Honest baseline against constant/linear/xgb_5macro. '
            'Stratified by phase and n_players.'
        ),
    }
    meta_path = MODEL_DIR / 'value_function_v2_meta.json'
    meta_path.write_text(json.dumps(all_metrics, indent=2))

    metrics_path = REPORT_DIR / 'value_v2_metrics.json'
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f'\n✓ Сохранено:')
    print(f'  model → {model_path}')
    print(f'  meta  → {meta_path}')
    print(f'  metrics → {metrics_path}')

    # ── Markdown отчёт ───────────────────────────────────────────────────
    md = []
    md.append('# Value Function V2 — отчёт\n')
    md.append('## Изменения vs V1\n')
    md.append('1. Убраны action features (9 шт.) из STATE — устраняет endogeneity.\n')
    md.append('2. Убрана `phase` — устраняет leak через концовку.\n')
    md.append('3. Добавлена стратификация по `phase` quartiles и `n_players`.\n')
    md.append('4. Добавлены тривиальные бейзлайны для измерения lift.\n')
    md.append('\n## Метрики overall\n')
    md.append(f'| Model | AUC | Brier | Acc | n |\n')
    md.append(f'|-------|-----|-------|-----|---|\n')
    md.append(f'| constant winrate | — | {m_const["brier"]} | {m_const["acc"]} | {m_const["n"]} |\n')
    md.append(f'| linear (5 macro) | {m_lin["auc"]} | {m_lin["brier"]} | {m_lin["acc"]} | {m_lin["n"]} |\n')
    md.append(f'| xgb (5 macro)    | {m_xgbm["auc"]} | {m_xgbm["brier"]} | {m_xgbm["acc"]} | {m_xgbm["n"]} |\n')
    md.append(f'| **value_v2**     | **{m_v2["auc"]}** | **{m_v2["brier"]}** | **{m_v2["acc"]}** | {m_v2["n"]} |\n')
    md.append(f'\n**Lift over xgb_5macro: +{lift["auc_lift_over_xgb_macro"]:.4f} AUC**\n')
    md.append(f'\n## По фазам игры\n')
    if strat_phase:
        md.append('| Phase quartile | n | AUC | Acc | pos_rate |\n|---|---|---|---|---|\n')
        for s, m in strat_phase.items():
            md.append(f'| {s} | {m["n"]:,} | {m["auc"]} | {m["acc"]} | {m["pos_rate"]} |\n')
    md.append(f'\n## По n_players\n')
    if strat_np:
        md.append('| n_players | n | AUC | Acc | pos_rate |\n|---|---|---|---|---|\n')
        for s, m in strat_np.items():
            md.append(f'| {s} | {m["n"]:,} | {m["auc"]} | {m["acc"]} | {m["pos_rate"]} |\n')
    md.append(f'\n## Топ фич\n')
    md.append('```\n')
    md.append(imp.head(15).round(4).to_string())
    md.append('\n```\n')
    md.append(f'\n## Интерпретация\n')
    md.append('Эта модель честно отвечает на вопрос "из этого state кто победит?",\n')
    md.append('а не "это победитель действует или проигравший?". Сравнение с\n')
    md.append('linear/xgb_macro показывает, сколько даёт нелинейность и расширение\n')
    md.append('фич за пределы базовых ratio. Стратификация по phase покажет, где\n')
    md.append('игра ещё контестная (низкий AUC) vs где уже решена (высокий AUC).\n')
    (REPORT_DIR / 'value_v2.md').write_text(''.join(md))
    print(f'  report → {REPORT_DIR}/value_v2.md')


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    if not STEPS_CSV.exists():
        print(f'⚠ Не найден {STEPS_CSV}')
        print('  Запустите сначала opponent_learning/scripts/02b_extract_steps.py')
        sys.exit(1)
    train(args)


if __name__ == '__main__':
    main()
