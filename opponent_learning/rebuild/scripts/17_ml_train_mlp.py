#!/usr/bin/env python3
"""
17_ml_train_mlp.py — MLP версия listwise ranker.

ПОЧЕМУ MLP
─────────
Linear (16_*) дал +14 pp top-1 vs baseline эвристики (6.4 → 20.7%). Но
loss застрял на ≈ ln(N_avg) с epoch 0 — линейная capacity исчерпана.
Линейка ловит только глобальные направления "близкие лучше" и "phase
зависит от ratio". Нелинейные паттерны типа
  "брать heavy targets ЕСЛИ src имеет много кораблей AND phase>0.5"
требуют нелинейностей.

АРХИТЕКТУРА
───────────
Per-planet features + context конкатенируются → MLP с 2 hidden layers,
ReLU, Dropout. На выходе скаляр — score планеты.
  x = [planet_feats; ctx]  (20-d)
  h1 = Dropout(ReLU(Linear_1(x)))  (64-d)
  h2 = Dropout(ReLU(Linear_2(h1)))  (32-d)
  score = Linear_3(h2)

Softmax по choice_set + cross-entropy на chosen — listwise loss.

ДОПОЛНИТЕЛЬНО
─────────────
- Cosine annealing LR schedule
- Early stopping по top-3
- Save best epoch weights

ВХОД / ВЫХОД — те же что в 16_*.
ТЯЖЁЛЫЙ скрипт (3-5 минут на CPU).
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
    sys.exit('PyTorch не установлен')

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

PLANET_FEATURES = [
    'prod', 'ships', 'dist_frac', 'ripeness', 'wnn', 'deepness',
    'late_aggression', 'priority_approx', 'owner_type',
]
CONTEXT_FEATURES = [
    'phase', 'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
    'n_own_planets', 'n_neutral_planets', 'n_enemy_planets',
    'src_ships', 'src_prod', 'src_wnn', 'src_ripeness',
]


def load_choice_sets(sample_episodes=None, seed=42):
    print(f'Loading {CHOICE_SETS} …')
    t0 = time.time()
    df = pd.read_csv(CHOICE_SETS)
    print(f'  {len(df):,} строк за {time.time()-t0:.1f}s')
    df['episode_id'] = df['episode_id'].astype(str)
    if sample_episodes is not None:
        rng = np.random.default_rng(seed)
        eids = df['episode_id'].unique()
        rng.shuffle(eids)
        keep = set(eids[:sample_episodes])
        df = df[df['episode_id'].isin(keep)].copy()
        print(f'  sample: {len(df):,} строк, {df["episode_id"].nunique()} эпизодов')
    return df


def split_by_episode(df, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    eids = df['episode_id'].unique()
    rng.shuffle(eids)
    n_val = max(1, int(len(eids) * val_frac))
    val_set = set(eids[:n_val])
    val_mask = df['episode_id'].isin(val_set)
    return df[~val_mask].copy(), df[val_mask].copy()


def build_groups(df, feat_cols, ctx_cols, feat_stats=None):
    if feat_stats is None:
        feat_stats = {}
        for c in feat_cols + ctx_cols:
            if c in df.columns:
                vals = df[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
                feat_stats[c] = {'mean': float(vals.mean()),
                                  'std': float(vals.std()) if vals.std() > 1e-6 else 1.0}
    groups = []
    for csid, g in df.groupby('choice_set_id'):
        if len(g) < 2:
            continue
        g = g.reset_index(drop=True)
        chosen_idx = g.index[g['is_chosen'] == 1].tolist()
        if not chosen_idx:
            continue
        chosen_idx = chosen_idx[0]
        n = len(g)
        feats = np.zeros((n, len(feat_cols)), dtype=np.float32)
        for j, c in enumerate(feat_cols):
            if c not in g.columns:
                continue
            v = g[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
            s = feat_stats[c]
            feats[:, j] = (v - s['mean']) / max(s['std'], 1e-6)
        ctx = np.zeros(len(ctx_cols), dtype=np.float32)
        first = g.iloc[0]
        for j, c in enumerate(ctx_cols):
            if c not in g.columns:
                continue
            v = float(first[c]) if not pd.isna(first[c]) else 0.0
            s = feat_stats[c]
            ctx[j] = (v - s['mean']) / max(s['std'], 1e-6)
        groups.append({
            'feats': feats, 'ctx': ctx, 'chosen_idx': chosen_idx,
            'n': n, 'episode_id': str(first['episode_id']),
        })
    return groups, feat_stats


# ════════════════════════════════════════════════════════════════════════
# MLP model
# ════════════════════════════════════════════════════════════════════════

class MLPRanker(nn.Module):
    def __init__(self, n_planet, n_ctx, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        d = n_planet + n_ctx
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, planet_feats, ctx):
        n = planet_feats.size(0)
        ctx_expand = ctx.unsqueeze(0).expand(n, -1)
        x = torch.cat([planet_feats, ctx_expand], dim=-1)
        return self.net(x).squeeze(-1)


def listwise_loss(scores, chosen_idx):
    log_probs = F.log_softmax(scores, dim=0)
    return -log_probs[chosen_idx]


def evaluate(model, groups):
    model.eval()
    top1 = top3 = top5 = top10 = 0
    ranks = []
    with torch.no_grad():
        for g in groups:
            feats = torch.from_numpy(g['feats'])
            ctx = torch.from_numpy(g['ctx'])
            scores = model(feats, ctx).numpy()
            order = np.argsort(-scores)
            rank = int(np.where(order == g['chosen_idx'])[0][0]) + 1
            ranks.append(rank)
            if rank <= 1:  top1 += 1
            if rank <= 3:  top3 += 1
            if rank <= 5:  top5 += 1
            if rank <= 10: top10 += 1
    n = max(len(groups), 1)
    return {
        'top1': top1/n, 'top3': top3/n, 'top5': top5/n, 'top10': top10/n,
        'median_rank': float(np.median(ranks)) if ranks else 0,
        'mean_rank':   float(np.mean(ranks)) if ranks else 0,
        'n': n,
    }


def train_loop(model, train_groups, val_groups, n_epochs=30, lr=0.001,
               batch_size=64, weight_decay=1e-4, patience=5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    rng = np.random.default_rng(0)
    history = []
    best_val_top3 = -1
    best_state = None
    no_improve = 0
    for epoch in range(n_epochs):
        model.train()
        rng.shuffle(train_groups)
        train_loss_sum = 0
        n_batches = 0
        for i in range(0, len(train_groups), batch_size):
            batch = train_groups[i:i + batch_size]
            opt.zero_grad()
            batch_loss = 0
            for g in batch:
                feats = torch.from_numpy(g['feats'])
                ctx = torch.from_numpy(g['ctx'])
                scores = model(feats, ctx)
                batch_loss = batch_loss + listwise_loss(scores, g['chosen_idx'])
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            opt.step()
            train_loss_sum += batch_loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        vm = evaluate(model, val_groups)
        history.append({'epoch': epoch, 'lr': scheduler.get_last_lr()[0],
                        'train_loss': round(train_loss, 4),
                        'val_top1': vm['top1'], 'val_top3': vm['top3'],
                        'val_top5': vm['top5'], 'val_medrank': vm['median_rank']})
        print(f'  epoch {epoch:2d}  lr={scheduler.get_last_lr()[0]:.5f}  '
              f'train_loss={train_loss:.4f}  '
              f'val: top1={vm["top1"]:.4f}  top3={vm["top3"]:.4f}  '
              f'top5={vm["top5"]:.4f}  med_rank={vm["median_rank"]:.1f}')
        if vm['top3'] > best_val_top3:
            best_val_top3 = vm['top3']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping на epoch {epoch} (no improve {patience})')
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {'history': history, 'best_val_top3': best_val_top3,
            'best_metrics': evaluate(model, val_groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-episodes', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--hidden', type=str, default='64,32',
                    help='Размеры скрытых слоёв через запятую')
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden.split(','))

    df = load_choice_sets(sample_episodes=args.sample_episodes)
    train_df, val_df = split_by_episode(df, val_frac=0.15)
    print(f'\nTrain: {len(train_df):,} строк, {train_df["episode_id"].nunique()} эпизодов')
    print(f'Val:   {len(val_df):,} строк, {val_df["episode_id"].nunique()} эпизодов')

    feat_cols = [c for c in PLANET_FEATURES if c in df.columns]
    ctx_cols = [c for c in CONTEXT_FEATURES if c in df.columns]
    print(f'\nFeatures: {len(feat_cols)} planet + {len(ctx_cols)} context')

    print('\nПодготовка train groups …')
    t0 = time.time()
    train_groups, feat_stats = build_groups(train_df, feat_cols, ctx_cols)
    print(f'  {len(train_groups)} групп за {time.time()-t0:.1f}s')

    print('Подготовка val groups …')
    val_groups, _ = build_groups(val_df, feat_cols, ctx_cols, feat_stats=feat_stats)
    print(f'  {len(val_groups)} групп')

    # ── Загрузим linear baseline для сравнения ──
    linear_baseline = None
    linear_path = MODEL_DIR / 'ml_linear_meta.json'
    if linear_path.exists():
        with open(linear_path) as f:
            linear_meta = json.load(f)
        linear_baseline = linear_meta.get('final_metrics', {})
        print(f'\n── Loaded linear baseline (16_*): top1={linear_baseline.get("top1", 0):.4f}  '
              f'top3={linear_baseline.get("top3", 0):.4f}')

    # ── MLP training ──
    print(f'\n── Training MLP listwise ranker  hidden={hidden_dims}  dropout={args.dropout} ──')
    model = MLPRanker(n_planet=len(feat_cols), n_ctx=len(ctx_cols),
                       hidden_dims=hidden_dims, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Параметров: {n_params}')

    result = train_loop(model, train_groups, val_groups,
                         n_epochs=args.epochs, lr=args.lr,
                         batch_size=args.batch_size,
                         weight_decay=args.weight_decay,
                         patience=args.patience)
    final = result['best_metrics']
    print(f'\n  Финальные метрики (best epoch):')
    print(f'    top1={final["top1"]:.4f}  top3={final["top3"]:.4f}  '
          f'top5={final["top5"]:.4f}  top10={final["top10"]:.4f}  '
          f'med_rank={final["median_rank"]:.1f}')

    if linear_baseline:
        print(f'\n  Прибавка vs LINEAR baseline (16):')
        for k in ['top1', 'top3', 'top5', 'top10']:
            d = final[k] - linear_baseline.get(k, 0)
            print(f'    {k}: {linear_baseline.get(k, 0):.4f} → {final[k]:.4f}  (Δ={d:+.4f})')

    torch.save(model.state_dict(), MODEL_DIR / 'ml_mlp_weights.pt')
    meta = {
        'architecture':     {'hidden_dims': list(hidden_dims), 'dropout': args.dropout,
                              'n_params': n_params},
        'planet_features':  feat_cols,
        'context_features': ctx_cols,
        'feat_stats':       feat_stats,
        'baseline_linear':  linear_baseline,
        'final_metrics':    final,
        'history':          result['history'],
        'config': {
            'sample_episodes': args.sample_episodes,
            'epochs': args.epochs, 'lr': args.lr,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
        },
    }
    (MODEL_DIR / 'ml_mlp_meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\n✓ Сохранено:')
    print(f'  weights → {MODEL_DIR / "ml_mlp_weights.pt"}')
    print(f'  meta    → {MODEL_DIR / "ml_mlp_meta.json"}')

    # Markdown
    md = []
    md.append('# MLP listwise ranker — нелинейный pro-matching\n\n')
    md.append(f'**Архитектура**: MLP {hidden_dims} + dropout={args.dropout}, '
              f'{n_params} параметров\n')
    md.append(f'**Данные**: {len(train_groups)} train + {len(val_groups)} val choice_sets\n\n')

    md.append('## Метрики\n\n')
    md.append('| Model | top1 | top3 | top5 | top10 | median_rank |\n|---|---|---|---|---|---|\n')
    md.append('| baseline (priority_approx) | 0.0641 | 0.1718 | 0.2705 | 0.4559 | 12.0 |\n')
    if linear_baseline:
        md.append(f'| linear (16_*) | {linear_baseline.get("top1", 0):.4f} | '
                  f'{linear_baseline.get("top3", 0):.4f} | '
                  f'{linear_baseline.get("top5", 0):.4f} | '
                  f'{linear_baseline.get("top10", 0):.4f} | '
                  f'{linear_baseline.get("median_rank", 0):.1f} |\n')
    md.append(f'| **MLP** | **{final["top1"]:.4f}** | **{final["top3"]:.4f}** | '
              f'**{final["top5"]:.4f}** | **{final["top10"]:.4f}** | '
              f'**{final["median_rank"]:.1f}** |\n')

    md.append('\n## История обучения\n\n')
    md.append('| epoch | lr | train_loss | top1 | top3 | top5 |\n|---|---|---|---|---|---|\n')
    for h in result['history']:
        md.append(f'| {h["epoch"]} | {h["lr"]:.5f} | {h["train_loss"]} | '
                  f'{h["val_top1"]:.4f} | {h["val_top3"]:.4f} | '
                  f'{h["val_top5"]:.4f} |\n')

    md.append('\n## Что дальше\n\n')
    md.append('Если MLP даёт значимый прирост vs linear (≥+0.05 top-3) — нелинейности\n')
    md.append('работают, и нелинейные паттерны существуют в данных. Следующие шаги:\n')
    md.append('1. Attention/Pointer Network для interaction между планетами\n')
    md.append('2. Residual integration: heuristic_priority + alpha·ML_score\n')
    md.append('3. Турниры в zones_tournament.py с интегрированной моделью\n')

    (REPORT_DIR / 'ml_mlp_report.md').write_text(''.join(md))
    print(f'  отчёт   → {REPORT_DIR / "ml_mlp_report.md"}')


if __name__ == '__main__':
    main()
