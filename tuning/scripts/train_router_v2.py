#!/usr/bin/env python3
"""
train_router_v2.py — multi-label роутер.

Отличие от v1:
  v1: один классификатор «лучший cand» — теряет инфу когда несколько
       cand'ов одновременно выигрывают, плюс класс «winner_seed47» с 1
       примером не учится.
  v2: для КАЖДОГО кандидата свой бинарный классификатор P(этот выиграет |
       map_features). На предикте argmax по вероятностям. Любой winner
       считается успехом, нам не нужен «лучший».

Логика:
  - Один Random Forest на cand (binary y = 1/0).
  - На предикте все RF голосуют, берём cand с max P(win).
  - Опционально умножаем на «prior» = глобальный winrate cand'a.
  - OOF accuracy: % сидов где предсказанный cand реально выиграл.

Запуск:  python3 tuning/scripts/train_router_v2.py
"""

import os, sys, glob
from datetime import datetime
import pandas as pd
import numpy as np

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")


def main():
    eval_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_8winners_*.csv")))
    meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "analysis_meta_*.csv")))
    if not eval_files or not meta_files:
        print("Нет eval_8winners или analysis_meta CSV"); return

    eval_csv = eval_files[-1]
    meta_csv = meta_files[-1]
    print(f"eval CSV: {eval_csv}")
    print(f"meta CSV: {meta_csv}\n")

    edf = pd.read_csv(eval_csv)
    mdf = pd.read_csv(meta_csv)

    # ── 1. Per-(cand, seed) win matrix ──────────────────────────────────
    cands = list(edf['name'])
    cand_winrate = dict(zip(edf['name'], edf['winrate']))
    n_test = int(edf['n'].max())

    seed_wins = {}   # seed → set(cands)
    for _, row in edf.iterrows():
        name = row['name']
        s = str(row.get('won_seeds') or '')
        for sid in (int(x) for x in s.split(',') if x.strip()):
            seed_wins.setdefault(sid, set()).add(name)

    FEATURES = ['omega', 'n_planets', 'n_orbital', 'n_static', 'n_neutrals',
                'prod_total', 'prod_total_neutrals', 'prod_mean', 'prod_var',
                'min_dist_home_neutral', 'mean_dist_home_neutral',
                'mean_inter_dist', 'max_inter_dist', 'sun_blockage_pct',
                'n_prod_1', 'n_prod_2', 'n_prod_3', 'n_prod_4', 'n_prod_5']
    FEATURES = [f for f in FEATURES if f in mdf.columns]

    # Сиды с фичами
    common_seeds = sorted(set(mdf['seed']) & set(range(1, n_test + 1)))
    feat_by_seed = mdf.set_index('seed')[FEATURES].to_dict('index')

    X = np.array([[feat_by_seed[s][f] for f in FEATURES] for s in common_seeds])
    print(f"сидов: {len(common_seeds)}, фичей: {len(FEATURES)}\n")

    # Бинарные метки на cand
    Y = {}   # cand → np.array of 0/1, длина = len(common_seeds)
    for c in cands:
        Y[c] = np.array([int(c in seed_wins.get(s, set())) for s in common_seeds])

    # Печатаем баланс
    print("Баланс по cand'ам (1 = выигрывает на сиде):")
    for c in cands:
        n_win = int(Y[c].sum())
        print(f"  {c:<16}  pos={n_win:>3}/{len(Y[c])}  ({100*n_win/len(Y[c]):.0f}%)")

    # ── 2. Бинарный RF на каждый cand с CV-OOF probas ───────────────────
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, KFold
    except ImportError:
        print("\nsklearn нет"); return

    # OOF probas: для каждого cand, для каждого сида — outOfFold P(win)
    n = len(common_seeds)
    proba_oof = np.zeros((n, len(cands)))
    for ci, c in enumerate(cands):
        y = Y[c]
        if y.sum() < 2 or y.sum() > n - 2:
            # слишком вырожденно — fallback на prior
            proba_oof[:, ci] = cand_winrate.get(c, 0.0)
            continue
        n_splits = min(5, int(y.sum()), int((y == 0).sum()))
        if n_splits < 2:
            proba_oof[:, ci] = cand_winrate.get(c, 0.0)
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X, y):
            rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                         random_state=42, n_jobs=-1).fit(X[train_idx], y[train_idx])
            # вероятность класса 1
            classes = rf.classes_
            if 1 in classes:
                p = rf.predict_proba(X[test_idx])[:, list(classes).index(1)]
            else:
                p = np.zeros(len(test_idx))
            proba_oof[test_idx, ci] = p

    # ── 3. На каждом сиде выбираем argmax (можно с prior) ───────────────
    USE_PRIOR = False   # умножать ли на cand_winrate; обычно делает хуже
    if USE_PRIOR:
        priors = np.array([cand_winrate.get(c, 0.0) for c in cands])
        scores = proba_oof * priors[None, :]
    else:
        scores = proba_oof

    pred_idx = scores.argmax(axis=1)
    pred_cand = [cands[i] for i in pred_idx]

    # ── 4. Метрики ──────────────────────────────────────────────────────
    # Top-1: предсказанный cand реально победил?
    top1_correct = sum(1 for sid, c in zip(common_seeds, pred_cand)
                       if c in seed_wins.get(sid, set()))
    # Top-2: один из 2 лучших по proba побеждает?
    top2_idx = scores.argsort(axis=1)[:, -2:]
    top2_correct = 0
    for i, sid in enumerate(common_seeds):
        winners = seed_wins.get(sid, set())
        chosen = {cands[k] for k in top2_idx[i]}
        if chosen & winners:
            top2_correct += 1

    oracle = sum(1 for s in common_seeds if seed_wins.get(s)) / len(common_seeds)
    default_wr = edf[edf['name'] == 'default']['winrate'].iloc[0] if (edf['name'] == 'default').any() else 0
    best_single_row = edf.loc[edf['winrate'].idxmax()]

    print("\n=== Сравнение стратегий ===")
    print(f"  default:      WR = {default_wr:.3f}")
    print(f"  best single:  WR = {best_single_row['winrate']:.3f}  ({best_single_row['name']})")
    print(f"  router top-1: WR = {top1_correct/len(common_seeds):.3f}  ({top1_correct}/{len(common_seeds)})")
    print(f"  router top-2: WR = {top2_correct/len(common_seeds):.3f}  ← если бы гоняли 2 лучших и брали max")
    print(f"  oracle:       WR = {oracle:.3f}  (теоретический max)")

    # ── 5. Какие cand'ы реально выбирает роутер ─────────────────────────
    print("\nЧастоты выбора cand'ом роутера:")
    from collections import Counter
    cnt = Counter(pred_cand)
    for c, n in cnt.most_common():
        # из них сколько верных
        ok = sum(1 for sid, cc in zip(common_seeds, pred_cand)
                 if cc == c and c in seed_wins.get(sid, set()))
        print(f"  {c:<16}  выбран {n:>3}× | верно {ok:>3} ({100*ok/max(n,1):.0f}%)")

    # ── 6. Per-cand binary CV accuracy ──────────────────────────────────
    print("\nПо-кандидатные binary CV accuracies (1 = выиграет):")
    from sklearn.metrics import roc_auc_score
    for ci, c in enumerate(cands):
        y = Y[c]; p = proba_oof[:, ci]
        try:
            auc = roc_auc_score(y, p) if (0 < y.sum() < n) else 0.5
        except Exception:
            auc = 0.5
        # лучший threshold
        from sklearn.metrics import accuracy_score
        best_acc, best_thr = 0, 0.5
        for thr in np.arange(0.1, 0.9, 0.05):
            acc = accuracy_score(y, (p > thr).astype(int))
            if acc > best_acc: best_acc, best_thr = acc, thr
        print(f"  {c:<16}  AUC={auc:.3f}  best_acc={best_acc:.3f} @ thr={best_thr:.2f}")

    # ── 7. Сохраняем ────────────────────────────────────────────────────
    out = pd.DataFrame({
        'seed': common_seeds,
        'predicted': pred_cand,
        'predicted_won': [int(c in seed_wins.get(s, set()))
                          for s, c in zip(common_seeds, pred_cand)],
        'actual_winners': [','.join(sorted(seed_wins.get(s, set()))) for s in common_seeds],
    })
    for ci, c in enumerate(cands):
        out[f'p_{c}'] = proba_oof[:, ci]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(RESULTS_DIR, f"router_v2_{ts}.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
