#!/usr/bin/env python3
"""
train_router.py — учим роутер «map features → какой winner-конфиг включить».

Вход:
  1) eval_8winners_*.csv  — список выигранных сидов на каждого кандидата
  2) analysis_meta_*.csv  — фичи карты на каждый seed (omega, n_planets, …)

Логика:
  - Для каждого seed находим «лучшего» кандидата (любой выигрывающий, при
    нескольких — тот, у кого выше глобальный winrate).
  - Сиды где НИКТО не выигрывает помечаем 'none' и исключаем из training set
    (это структурный dead-end, роутер не поможет).
  - Тренируем DecisionTree (depth=4) на map-features.
  - Кросс-валидация 5-fold, показываем accuracy и предсказанный winrate.
  - Печатаем правила дерева в человекочитаемом виде.

Запуск:  python3 tuning/scripts/train_router.py
"""

import os, sys, glob
from datetime import datetime
import pandas as pd
import numpy as np

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")


def main():
    # ── 1. Загружаем покрытие ───────────────────────────────────────────
    eval_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_8winners_*.csv")))
    if not eval_files:
        print("Нет eval_8winners_*.csv в", RESULTS_DIR); return
    eval_csv = eval_files[-1]
    print(f"eval CSV:  {eval_csv}")

    meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "analysis_meta_*.csv")))
    if not meta_files:
        print("Нет analysis_meta_*.csv в", RESULTS_DIR);
        print("Запусти сначала match_analyzer.py чтобы собрать map-features.");
        return
    meta_csv = meta_files[-1]
    print(f"meta CSV:  {meta_csv}\n")

    edf = pd.read_csv(eval_csv)
    mdf = pd.read_csv(meta_csv)

    # ── 2. Карта seed → set of winning candidates ───────────────────────
    cand_winrate = dict(zip(edf['name'], edf['winrate']))
    seed_wins = {}   # seed → list of (cand, global_winrate)
    for _, row in edf.iterrows():
        cand = row['name']
        s = str(row.get('won_seeds') or '')
        for sid in (int(x) for x in s.split(',') if x.strip()):
            seed_wins.setdefault(sid, []).append((cand, cand_winrate.get(cand, 0)))

    # ── 3. Лейблим каждый seed его «лучшим» победителем ────────────────
    all_seeds = sorted(set(mdf['seed']) & set(range(1, int(edf['n'].max()) + 1)))
    rows = []
    for sid in all_seeds:
        wins = seed_wins.get(sid, [])
        if not wins:
            label = 'none'   # никто не выиграл
        else:
            # тай-брейк: глобально-сильный кандидат
            label = max(wins, key=lambda kv: kv[1])[0]
        rows.append({'seed': sid, 'label': label})
    labels_df = pd.DataFrame(rows)
    print("Распределение меток (best winner per seed):")
    print(labels_df['label'].value_counts())

    # ── 4. Сшиваем с фичами карты ───────────────────────────────────────
    df = labels_df.merge(mdf, on='seed', how='inner')
    print(f"\njoined: {len(df)} сидов с фичами + меткой")

    # Фильтруем 'none' для training
    train = df[df['label'] != 'none'].copy()
    print(f"для обучения (исключая 'none'): {len(train)}")

    # ── 5. Тренируем дерево с CV ────────────────────────────────────────
    FEATURES = ['omega', 'n_planets', 'n_orbital', 'n_static', 'n_neutrals',
                'prod_total', 'prod_total_neutrals', 'prod_mean', 'prod_var',
                'min_dist_home_neutral', 'mean_dist_home_neutral',
                'mean_inter_dist', 'max_inter_dist', 'sun_blockage_pct',
                'n_prod_1', 'n_prod_2', 'n_prod_3', 'n_prod_4', 'n_prod_5']
    available = [f for f in FEATURES if f in train.columns]
    print(f"\nфичей: {len(available)}")

    X = train[available].values
    y = train['label'].values

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("sklearn нет — pip install scikit-learn"); return

    print("\n── DecisionTree (depth=4) ──")
    n_classes = len(np.unique(y))
    n_splits = min(5, len(y) // n_classes)
    if n_splits < 2:
        print(f"мало сидов на класс ({n_classes} классов, {len(y)} сидов) — CV нестабилен, "
              f"делаем простой fit-without-CV")
        dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)
        train_acc = dt.score(X, y)
        print(f"  train accuracy: {train_acc:.3f}")
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        cv_scores = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')
        print(f"  CV accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
              f"({n_splits}-fold)")
        # фит на всех для итогового дерева
        dt.fit(X, y)
        print(f"  train accuracy: {dt.score(X, y):.3f}")

    # ── 6. Считаем предсказанный winrate ────────────────────────────────
    # Для каждого seed (включая 'none'):
    #   prediction = dt.predict(features)  → cand
    #   win = (cand победил seed?)
    df_full = df.copy()
    if len(available) > 0 and len(train) >= n_splits:
        # CV-style: предсказываем используя out-of-fold модели
        from sklearn.model_selection import cross_val_predict
        preds = cross_val_predict(dt, X, y, cv=min(5, len(y)//n_classes))
        # мапим обратно в df
        train_idx = train.index
        df_full['prediction'] = 'default'   # default для 'none'
        for i, idx in enumerate(train_idx):
            df_full.loc[idx, 'prediction'] = preds[i]
    else:
        df_full['prediction'] = 'default'

    # вычисляем winrate роутера
    n_correct = 0
    for _, row in df_full.iterrows():
        sid = row['seed']
        pred = row['prediction']
        # выиграет ли pred на этом seed?
        winners = [c for c, _ in seed_wins.get(sid, [])]
        if pred in winners:
            n_correct += 1
    router_winrate = n_correct / len(df_full)
    print(f"\nРоутер OOF-winrate: {router_winrate:.3f}  ({n_correct}/{len(df_full)})")
    best_single = edf['winrate'].max()
    oracle = sum(1 for s in all_seeds if seed_wins.get(s)) / len(all_seeds)
    print(f"Сравни:")
    print(f"  default:      {edf[edf['name']=='default']['winrate'].iloc[0]:.3f}")
    print(f"  best single:  {best_single:.3f}  ({edf.loc[edf['winrate'].idxmax(), 'name']})")
    print(f"  oracle:       {oracle:.3f}  (если бы всегда правильно роутили)")
    print(f"  router:       {router_winrate:.3f}  ← наш decision tree")

    # ── 7. Распечатываем дерево ─────────────────────────────────────────
    print("\n── Decision tree rules ──")
    if len(available) > 0:
        rules = export_text(dt, feature_names=available, max_depth=4)
        print(rules)

    # Feature importance
    print("\n── Feature importance (top-10) ──")
    fi = sorted(zip(available, dt.feature_importances_), key=lambda kv: -kv[1])[:10]
    for f, imp in fi:
        if imp > 0:
            print(f"  {f:<26}  {imp:.3f}")

    # ── 8. Random Forest для контроля ───────────────────────────────────
    if len(y) >= 20:
        print("\n── RandomForest (n_estimators=100) ──")
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        if n_splits >= 2:
            rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
            print(f"  CV accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
        else:
            rf.fit(X, y)
            print(f"  train acc: {rf.score(X, y):.3f}")

    # ── 9. Сохраняем ────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f"router_predictions_{ts}.csv")
    df_full[['seed','label','prediction'] + available].to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
