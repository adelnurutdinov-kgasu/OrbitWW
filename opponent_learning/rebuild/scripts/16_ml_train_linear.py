#!/usr/bin/env python3
"""
16_ml_train_linear.py — listwise learning-to-rank: linear baseline в PyTorch.

ИДЕЯ
────
В choice_sets.csv для каждого pro решения есть **все альтернативные планеты**
с фичами и флагом is_chosen. Это идеальный listwise ranking датасет.

Архитектура:
  Per-planet features + global state context → conv → score (scalar)
  Softmax по группе (choice_set) → distribution
  Loss: negative log-likelihood of chosen planet

Baseline линейный — это **тот же** функционал что W_TARGETS, но:
  (1) обученные под правильный target (pro-matching не winrate)
  (2) с listwise softmax loss (не z-score нормализация)
  (3) с включением global context features (phase, ratios) как conditioning

Если линейный даёт значимое улучшение vs baseline эвристики — переходим к MLP/NN.
Если нет — проблема не в фичах, а в их измерении.

ИНТЕГРАЦИЯ В ЭВРИСТИКУ
──────────────────────
После обучения сохраняем веса. В zones.py добавляем residual:
    final_priority = heuristic_priority + alpha * z(ML_score)
alpha настраиваем через top-K на val. При alpha=0 поведение = эвристика.

ВХОД
────
  opponent_learning/data/processed/choice_sets.csv  (1.7 GB — sample)

ВЫХОД
─────
  rebuild/models/ml_linear_weights.pt
  rebuild/models/ml_linear_meta.json   — feature names, stats, метрики
  rebuild/reports/ml_linear_report.md

ТЯЖЁЛЫЙ скрипт.
  Read sample (~200K rows): 30 сек
  Train (CPU, 10-30 epochs): 1-3 минуты
  Eval: 10 сек
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit('PyTorch не установлен. pip install torch')

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
PROC_DIR   = OL_DIR / "data" / "processed"
DATA_DIR   = REBUILD / "data"
MODEL_DIR  = REBUILD / "models"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHOICE_SETS = PROC_DIR / "choice_sets.csv"

# Per-planet features (характеристики кандидата)
PLANET_FEATURES = [
    'prod',           # production планеты
    'ships',          # текущие корабли
    'dist_frac',      # нормированное расстояние от источника
    'ripeness',       # production/ships — "спелость"
    'wnn',            # weighted nearest neighbour (близость к нашим)
    'deepness',       # глубина в тылу врага
    'late_aggression',# importance для late game
    'priority_approx',# эвристический priority — pre-baked feature
    'owner_type',     # категориальный — превратим в one-hot
]

# Global context (state of the world)
CONTEXT_FEATURES = [
    'phase',
    'own_ship_ratio',
    'own_prod_ratio',
    'own_planet_ratio',
    'n_own_planets',
    'n_neutral_planets',
    'n_enemy_planets',
    'src_ships',
    'src_prod',
    'src_wnn',
    'src_ripeness',
]


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════

def load_choice_sets(sample_episodes: int | None = None,
                     seed: int = 42) -> pd.DataFrame:
    """Загружает choice_sets.csv. Опционально sample по episode_id."""
    print(f'Loading {CHOICE_SETS} …')
    t0 = time.time()
    df = pd.read_csv(CHOICE_SETS)
    print(f'  {len(df):,} строк за {time.time()-t0:.1f}s')

    df['episode_id'] = df['episode_id'].astype(str)
    print(f'  эпизодов: {df["episode_id"].nunique():,}')
    print(f'  choice_sets: {df["choice_set_id"].nunique():,}')

    if sample_episodes is not None and sample_episodes < df['episode_id'].nunique():
        rng = np.random.default_rng(seed)
        eids = df['episode_id'].unique()
        rng.shuffle(eids)
        keep = set(eids[:sample_episodes])
        df = df[df['episode_id'].isin(keep)].copy()
        print(f'  sample: {len(df):,} строк, {df["episode_id"].nunique()} эпизодов')

    # one-hot owner_type
    if 'owner_type' in df.columns:
        # owner_type ∈ {0, 1, 2}? проверим
        unique_owners = sorted(df['owner_type'].unique())
        print(f'  owner_type unique: {unique_owners}')
        # Простой numeric encoding: оставим как есть
    return df


def split_by_episode(df: pd.DataFrame, val_frac: float = 0.15,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    eids = df['episode_id'].unique()
    rng.shuffle(eids)
    n_val = max(1, int(len(eids) * val_frac))
    val_set = set(eids[:n_val])
    val_mask = df['episode_id'].isin(val_set)
    return df[~val_mask].copy(), df[val_mask].copy()


# ════════════════════════════════════════════════════════════════════════
# Tensor preparation
# ════════════════════════════════════════════════════════════════════════

def build_groups(df: pd.DataFrame, feat_cols: list, ctx_cols: list,
                 feat_stats: dict | None = None) -> tuple[list, dict]:
    """Группируем по choice_set_id. Для каждого:
        feats: (n_planets, n_features)
        ctx:   (n_context,)
        chosen_idx: int
    Также возвращаем feat_stats для нормализации.
    """
    # Если stats не даны — посчитать (для train)
    if feat_stats is None:
        feat_stats = {}
        for c in feat_cols + ctx_cols:
            if c in df.columns:
                vals = df[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
                feat_stats[c] = {'mean': float(vals.mean()),
                                  'std':  float(vals.std()) if vals.std() > 1e-6 else 1.0}

    groups = []
    for csid, g in df.groupby('choice_set_id'):
        n = len(g)
        if n < 2:
            continue  # нет выбора
        g = g.reset_index(drop=True)
        chosen_idx = g.index[g['is_chosen'] == 1].tolist()
        if not chosen_idx:
            continue
        chosen_idx = chosen_idx[0]

        # Per-planet features (нормированные)
        feats = np.zeros((n, len(feat_cols)), dtype=np.float32)
        for j, c in enumerate(feat_cols):
            if c not in g.columns:
                continue
            v = g[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
            s = feat_stats.get(c, {'mean': 0.0, 'std': 1.0})
            feats[:, j] = (v - s['mean']) / max(s['std'], 1e-6)

        # Context (одинаковый для всех планет в этом choice_set)
        ctx = np.zeros(len(ctx_cols), dtype=np.float32)
        first = g.iloc[0]
        for j, c in enumerate(ctx_cols):
            if c not in g.columns:
                continue
            v = float(first[c]) if not pd.isna(first[c]) else 0.0
            s = feat_stats.get(c, {'mean': 0.0, 'std': 1.0})
            ctx[j] = (v - s['mean']) / max(s['std'], 1e-6)

        groups.append({
            'feats':       feats,
            'ctx':         ctx,
            'chosen_idx':  chosen_idx,
            'n':           n,
            'episode_id':  str(first['episode_id']),
        })

    return groups, feat_stats


# ════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════

class LinearRanker(nn.Module):
    """Score(planet) = w · [planet_features ++ context_features].
    Архитектура линейная — для baseline ML-подхода. Готова к расширению на MLP."""

    def __init__(self, n_planet_features: int, n_context_features: int):
        super().__init__()
        self.n_planet = n_planet_features
        self.n_ctx    = n_context_features
        self.linear   = nn.Linear(n_planet_features + n_context_features, 1, bias=False)

    def forward(self, planet_feats: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        # planet_feats: (n_planets, n_planet_features)
        # ctx: (n_context,)
        n = planet_feats.size(0)
        ctx_expand = ctx.unsqueeze(0).expand(n, -1)
        x = torch.cat([planet_feats, ctx_expand], dim=-1)
        return self.linear(x).squeeze(-1)  # (n_planets,)


def listwise_loss(scores: torch.Tensor, chosen_idx: int) -> torch.Tensor:
    """Negative log-likelihood of chosen planet in softmax(scores)."""
    log_probs = F.log_softmax(scores, dim=0)
    return -log_probs[chosen_idx]


# ════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════

def train_loop(model: LinearRanker, train_groups: list, val_groups: list,
               n_epochs: int = 20, lr: float = 0.01,
               batch_size_groups: int = 64,
               weight_decay: float = 1e-4) -> dict:
    """One full training run. Возвращает best val metrics + history."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(0)

    history = []
    best_val_top3 = -1
    best_state = None

    for epoch in range(n_epochs):
        # ── train ──
        model.train()
        rng.shuffle(train_groups)
        train_loss_sum = 0
        n_batches = 0
        for i in range(0, len(train_groups), batch_size_groups):
            batch = train_groups[i:i + batch_size_groups]
            opt.zero_grad()
            batch_loss = 0
            for g in batch:
                feats = torch.from_numpy(g['feats'])
                ctx   = torch.from_numpy(g['ctx'])
                scores = model(feats, ctx)
                batch_loss = batch_loss + listwise_loss(scores, g['chosen_idx'])
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            opt.step()
            train_loss_sum += batch_loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        # ── val ──
        val_metrics = evaluate(model, val_groups)
        history.append({
            'epoch':     epoch,
            'train_loss': round(train_loss, 4),
            'val_top1':   val_metrics['top1'],
            'val_top3':   val_metrics['top3'],
            'val_top5':   val_metrics['top5'],
            'val_medrank': val_metrics['median_rank'],
        })
        print(f'  epoch {epoch:2d}  train_loss={train_loss:.4f}  '
              f'val: top1={val_metrics["top1"]:.4f}  '
              f'top3={val_metrics["top3"]:.4f}  '
              f'top5={val_metrics["top5"]:.4f}  '
              f'med_rank={val_metrics["median_rank"]:.1f}')

        if val_metrics['top3'] > best_val_top3:
            best_val_top3 = val_metrics['top3']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        'history':       history,
        'best_val_top3': best_val_top3,
        'best_metrics':  evaluate(model, val_groups),
    }


def evaluate(model: LinearRanker, groups: list) -> dict:
    """Topo-K accuracy на listе groups."""
    model.eval()
    top1 = top3 = top5 = top10 = 0
    ranks = []
    with torch.no_grad():
        for g in groups:
            feats = torch.from_numpy(g['feats'])
            ctx   = torch.from_numpy(g['ctx'])
            scores = model(feats, ctx).numpy()
            order = np.argsort(-scores)
            rank = int(np.where(order == g['chosen_idx'])[0][0]) + 1
            ranks.append(rank)
            if rank <= 1:  top1  += 1
            if rank <= 3:  top3  += 1
            if rank <= 5:  top5  += 1
            if rank <= 10: top10 += 1
    n = max(len(groups), 1)
    return {
        'top1':         top1 / n,
        'top3':         top3 / n,
        'top5':         top5 / n,
        'top10':        top10 / n,
        'median_rank':  float(np.median(ranks)) if ranks else 0,
        'mean_rank':    float(np.mean(ranks)) if ranks else 0,
        'n':            n,
    }


def baseline_priority_approx(groups: list) -> dict:
    """Baseline: использовать priority_approx (эвристический priority) напрямую
    как score без обучения. Это и есть baseline эвристики."""
    top1 = top3 = top5 = top10 = 0
    ranks = []
    for g in groups:
        # priority_approx — это feature index в PLANET_FEATURES, его номер ищем
        if 'priority_approx' not in PLANET_FEATURES:
            return {}
        idx = PLANET_FEATURES.index('priority_approx')
        scores = g['feats'][:, idx]   # уже нормированные
        order = np.argsort(-scores)
        rank = int(np.where(order == g['chosen_idx'])[0][0]) + 1
        ranks.append(rank)
        if rank <= 1:  top1  += 1
        if rank <= 3:  top3  += 1
        if rank <= 5:  top5  += 1
        if rank <= 10: top10 += 1
    n = max(len(groups), 1)
    return {
        'top1':         top1 / n,
        'top3':         top3 / n,
        'top5':         top5 / n,
        'top10':        top10 / n,
        'median_rank':  float(np.median(ranks)) if ranks else 0,
        'mean_rank':    float(np.mean(ranks)) if ranks else 0,
        'n':            n,
    }


# ════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-episodes', type=int, default=200,
                    help='Сколько эпизодов взять (None = все)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--batch-size', type=int, default=64,
                    help='Сколько choice_sets в одном gradient step')
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    args = ap.parse_args()

    df = load_choice_sets(sample_episodes=args.sample_episodes)
    train_df, val_df = split_by_episode(df, val_frac=0.15, seed=42)
    print(f'\nTrain: {len(train_df):,} строк, {train_df["episode_id"].nunique()} эпизодов')
    print(f'Val:   {len(val_df):,} строк, {val_df["episode_id"].nunique()} эпизодов')

    # Используем только колонки которые есть в датасете
    feat_cols = [c for c in PLANET_FEATURES if c in df.columns]
    ctx_cols  = [c for c in CONTEXT_FEATURES if c in df.columns]
    print(f'\nPer-planet features ({len(feat_cols)}): {feat_cols}')
    print(f'Context features ({len(ctx_cols)}): {ctx_cols}')

    print('\nПодготовка train groups …')
    t0 = time.time()
    train_groups, feat_stats = build_groups(train_df, feat_cols, ctx_cols)
    print(f'  {len(train_groups)} групп за {time.time()-t0:.1f}s')

    print('Подготовка val groups (using train stats) …')
    val_groups, _ = build_groups(val_df, feat_cols, ctx_cols, feat_stats=feat_stats)
    print(f'  {len(val_groups)} групп')

    # ── Baseline: priority_approx как score ──
    print('\n── Baseline: priority_approx как score ──')
    baseline = baseline_priority_approx(val_groups)
    if baseline:
        print(f'  top1={baseline["top1"]:.4f}  top3={baseline["top3"]:.4f}  '
              f'top5={baseline["top5"]:.4f}  med_rank={baseline["median_rank"]:.1f}')

    # ── Training ──
    print(f'\n── Training linear listwise ranker ──')
    model = LinearRanker(n_planet_features=len(feat_cols),
                          n_context_features=len(ctx_cols))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Параметров: {n_params}')

    result = train_loop(model, train_groups, val_groups,
                         n_epochs=args.epochs, lr=args.lr,
                         batch_size_groups=args.batch_size,
                         weight_decay=args.weight_decay)
    final = result['best_metrics']
    print(f'\n  Финальные метрики (best epoch):')
    print(f'    top1={final["top1"]:.4f}  top3={final["top3"]:.4f}  '
          f'top5={final["top5"]:.4f}  top10={final["top10"]:.4f}  '
          f'med_rank={final["median_rank"]:.1f}')

    # ── Сравнение с baseline ──
    if baseline:
        print(f'\n  Прибавка vs priority_approx baseline:')
        for k in ['top1', 'top3', 'top5', 'top10']:
            d = final[k] - baseline[k]
            print(f'    {k}: {baseline[k]:.4f} → {final[k]:.4f}  (Δ={d:+.4f})')

    # ── Сохранение ──
    w = model.linear.weight.detach().cpu().numpy().squeeze()
    n_planet = len(feat_cols)
    weights_planet = {feat_cols[i]: float(w[i]) for i in range(n_planet)}
    weights_ctx    = {ctx_cols[i]: float(w[n_planet + i]) for i in range(len(ctx_cols))}

    print(f'\n── Выученные веса (top-15 по |w|) ──')
    all_weights = [(f, weights_planet[f], 'planet') for f in feat_cols] + \
                  [(f, weights_ctx[f], 'context') for f in ctx_cols]
    all_weights.sort(key=lambda x: -abs(x[1]))
    for name, w, kind in all_weights[:15]:
        print(f'    {kind:<8} {name:<22} {w:+.4f}')

    torch.save(model.state_dict(), MODEL_DIR / 'ml_linear_weights.pt')
    meta = {
        'planet_features':   feat_cols,
        'context_features':  ctx_cols,
        'feat_stats':        feat_stats,
        'weights_planet':    weights_planet,
        'weights_ctx':       weights_ctx,
        'baseline_metrics':  baseline,
        'final_metrics':     final,
        'history':           result['history'],
        'n_train_groups':    len(train_groups),
        'n_val_groups':      len(val_groups),
    }
    (MODEL_DIR / 'ml_linear_meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\n✓ Сохранено:')
    print(f'  weights → {MODEL_DIR / "ml_linear_weights.pt"}')
    print(f'  meta    → {MODEL_DIR / "ml_linear_meta.json"}')

    # ── Markdown отчёт ──
    md = []
    md.append('# Linear listwise ranker — pro-matching baseline\n\n')
    md.append(f'**Данные**: {len(train_groups)} train + {len(val_groups)} val choice_sets\n')
    md.append(f'**Features**: {len(feat_cols)} per-planet + {len(ctx_cols)} context = '
              f'{len(feat_cols) + len(ctx_cols)} весов\n\n')
    md.append('## Метрики\n\n')
    md.append('| Model | top1 | top3 | top5 | top10 | median_rank |\n|---|---|---|---|---|---|\n')
    if baseline:
        md.append(f'| baseline (priority_approx) | {baseline["top1"]:.4f} | '
                  f'{baseline["top3"]:.4f} | {baseline["top5"]:.4f} | '
                  f'{baseline["top10"]:.4f} | {baseline["median_rank"]:.1f} |\n')
    md.append(f'| **linear listwise** | **{final["top1"]:.4f}** | '
              f'**{final["top3"]:.4f}** | **{final["top5"]:.4f}** | '
              f'**{final["top10"]:.4f}** | **{final["median_rank"]:.1f}** |\n')

    md.append('\n## Топ-15 выученных весов\n\n')
    md.append('| kind | feature | weight |\n|---|---|---|\n')
    for name, w, kind in all_weights[:15]:
        md.append(f'| {kind} | {name} | {w:+.4f} |\n')

    md.append('\n## История обучения\n\n')
    md.append('| epoch | train_loss | val_top1 | val_top3 | val_top5 |\n|---|---|---|---|---|\n')
    for h in result['history']:
        md.append(f'| {h["epoch"]} | {h["train_loss"]} | {h["val_top1"]:.4f} | '
                  f'{h["val_top3"]:.4f} | {h["val_top5"]:.4f} |\n')

    md.append('\n## Интерпретация\n\n')
    md.append('Если top-3 значимо лучше baseline (>+0.05) — listwise ML подход\n')
    md.append('даёт прирост даже на линейной модели. Это **proof of concept** для\n')
    md.append('перехода к MLP в шаге B.\n\n')
    md.append('Если top-3 примерно равен baseline — линейность не помогает,\n')
    md.append('и нужны нелинейности (MLP, attention) для дальнейших улучшений.\n')

    (REPORT_DIR / 'ml_linear_report.md').write_text(''.join(md))
    print(f'  отчёт   → {REPORT_DIR / "ml_linear_report.md"}')


if __name__ == '__main__':
    main()
