#!/usr/bin/env python3
"""
18_retrain_on_zones_features.py — переобучение listwise модели на фичах zones.

ПОЧЕМУ
──────
Модель из 16_/17_ обучилась на фичах из choice_sets.csv (prod, ships, dist_frac,
ripeness, wnn, deepness, late_aggression, priority_approx, owner_type). Эти фичи
**отсутствуют** в df возвращаемом `compute_zones_from_state` в runtime.
Без них модель нельзя интегрировать в эвристику.

ЧТО ДЕЛАЕМ
──────────
Для каждого pro launch:
  1. Загружаем obs из replay
  2. Запускаем compute_zones_from_state (как делает агент в runtime)
  3. Из df получаем 7 zones features per planet + priority + zone
  4. Используем их как X для обучения
Это даёт **полное соответствие** между обучением и runtime.

ФИЧИ
────
Per-planet (7): prod, ships, area_inv, wnn_close_res, mean_dist_all,
                n_cross, late_aggression
Context (11): phase, own_ship_ratio, own_prod_ratio, own_planet_ratio,
              n_own_planets, n_neutral_planets, n_enemy_planets,
              src_ships, src_prod, src_wnn, src_ripeness

ВХОД
────
  data/processed/choice_sets.csv  (для is_chosen меток и metadata)
  data/raw/*.json                  (для obs)

ВЫХОД
─────
  rebuild/data/zones_features_cache.parquet  — пересчитанные фичи
  rebuild/models/ml_zones_linear_weights.pt
  rebuild/models/ml_zones_linear_meta.json
  rebuild/models/ml_zones_mlp_weights.pt    (если --mlp)
  rebuild/reports/ml_zones_retrain_report.md

ТЯЖЁЛЫЙ скрипт. Compute features: ~5-15 минут на 100 эпизодов. Training: 2-5 мин.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
RAW_DIR    = OL_DIR / "data" / "raw"
DATA_DIR   = REBUILD / "data"
MODEL_DIR  = REBUILD / "models"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHOICE_SETS = PROC_DIR / "choice_sets.csv"
CACHE_PATH  = DATA_DIR / "zones_features_cache.csv"
UUID_MAP_PATH = DATA_DIR / "uuid_to_numeric_id.csv"

# Импорт agent_bundle_swarm 2 для compute_zones_from_state
AGENT_DIR = OL_DIR.parent / "agent_bundle_swarm 2"
sys.path.insert(0, str(AGENT_DIR))
from zones import compute_zones_from_state
from orbit_sim import GameState
from projection import project_state, PROJECTION_HORIZON

# Фичи которые есть в df_zones runtime
ZONES_PLANET_FEATURES = [
    'prod', 'ships',
    'area_inv', 'wnn_close_res', 'mean_dist_all',
    'n_cross', 'late_aggression',
]
# Контекст — тот же что и раньше (из choice_sets.csv)
CONTEXT_FEATURES = [
    'phase', 'own_ship_ratio', 'own_prod_ratio', 'own_planet_ratio',
    'n_own_planets', 'n_neutral_planets', 'n_enemy_planets',
    'src_ships', 'src_prod', 'src_wnn', 'src_ripeness',
]


# ════════════════════════════════════════════════════════════════════════
# UUID → numeric ID mapping
# ════════════════════════════════════════════════════════════════════════

def build_uuid_mapping() -> dict:
    """В choice_sets.csv episode_id это UUID (id поле эпизода). В raw/ имена
    файлов — это numeric ID (info.EpisodeId). Строим mapping один раз."""
    if UUID_MAP_PATH.exists():
        print(f'Загружаю UUID mapping из {UUID_MAP_PATH} …')
        m = pd.read_csv(UUID_MAP_PATH)
        return dict(zip(m['uuid'].astype(str), m['numeric'].astype(str)))

    print(f'Строю UUID mapping из raw/*.json (раз и навсегда) …')
    mapping = {}
    files = sorted(p for p in RAW_DIR.glob('*.json')
                   if p.name != 'episodes_index.json')
    t0 = time.time()
    for i, fp in enumerate(files):
        try:
            with open(fp) as f:
                ep = json.load(f)
        except Exception:
            continue
        uuid = ep.get('id')
        if uuid:
            mapping[str(uuid)] = fp.stem
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(files)} ({time.time()-t0:.0f}s)')
    pd.DataFrame({'uuid': list(mapping.keys()),
                  'numeric': list(mapping.values())}).to_csv(UUID_MAP_PATH, index=False)
    print(f'✓ Mapping {len(mapping)} → {UUID_MAP_PATH}')
    return mapping


# ════════════════════════════════════════════════════════════════════════
# Stage 1: Пересчёт фич через compute_zones_from_state
# ════════════════════════════════════════════════════════════════════════

def compute_features_for_episode(json_path: Path, group: pd.DataFrame,
                                  diag: dict | None = None) -> list[dict]:
    """Для одного эпизода: для каждого уникального (step, player) запускаем
    compute_zones_from_state и сохраняем фичи всех планет в choice_set."""
    if diag is None:
        diag = {}
    try:
        with open(json_path) as f:
            ep = json.load(f)
    except Exception as e:
        diag['json_load_err'] = diag.get('json_load_err', 0) + 1
        diag['last_err'] = f'json load: {e!r}'
        return []
    steps = ep.get('steps') or []
    if not steps:
        diag['no_steps'] = diag.get('no_steps', 0) + 1
        return []

    out = []
    by_state = group.groupby(['step', 'player_idx', 'choice_set_id'])
    state_cache = {}

    for (step, player_idx, csid), choices in by_state:
        cache_key = (step, player_idx)
        if cache_key not in state_cache:
            if step >= len(steps):
                diag['step_oob'] = diag.get('step_oob', 0) + 1
                continue
            s_now = steps[step]
            if not isinstance(s_now, list) or player_idx >= len(s_now):
                diag['bad_step'] = diag.get('bad_step', 0) + 1
                continue
            obs = (s_now[player_idx] or {}).get('observation') or {}
            if not obs.get('planets'):
                diag['no_planets'] = diag.get('no_planets', 0) + 1
                continue
            try:
                state = GameState.from_kaggle_obs(obs, step=step)
            except Exception as e:
                diag['kaggle_obs_err'] = diag.get('kaggle_obs_err', 0) + 1
                diag['last_err'] = f'from_kaggle_obs: {e!r}'
                continue
            try:
                state_proj = project_state(state, horizon=PROJECTION_HORIZON, player=player_idx)
            except Exception as e:
                diag['project_err'] = diag.get('project_err', 0) + 1
                diag['last_err'] = f'project_state: {e!r}'
                continue
            try:
                df_zones, _ = compute_zones_from_state(state_proj, player=player_idx)
            except Exception as e:
                diag['zones_err'] = diag.get('zones_err', 0) + 1
                diag['last_err'] = f'compute_zones: {e!r}'
                continue
            state_cache[cache_key] = df_zones

        df_zones = state_cache.get(cache_key)
        if df_zones is None:
            continue

        zones_by_pid = {row['pid']: row for _, row in df_zones.iterrows()}

        for _, ch in choices.iterrows():
            pid = ch['planet_id']
            if pid not in zones_by_pid:
                diag['pid_miss'] = diag.get('pid_miss', 0) + 1
                continue
            zr = zones_by_pid[pid]
            record = {
                'episode_id':    str(ch['episode_id']),
                'choice_set_id': csid,
                'step':          step,
                'player_idx':    player_idx,
                'planet_id':     pid,
                'is_chosen':     int(ch['is_chosen']),
                'zones_priority': float(zr.get('priority', 0.0)),
                'zones_zone':    str(zr.get('zone', '')),
            }
            for c in ZONES_PLANET_FEATURES:
                record[c] = float(zr.get(c, 0.0))
            for c in CONTEXT_FEATURES:
                if c in ch:
                    record[c] = float(ch[c]) if not pd.isna(ch[c]) else 0.0
                else:
                    record[c] = 0.0
            out.append(record)
    return out


def build_zones_features_dataset(sample_episodes: int = 100, seed: int = 42) -> pd.DataFrame:
    """Загружает choice_sets sample, пересчитывает фичи через compute_zones_from_state."""
    if CACHE_PATH.exists():
        ans = input(f'Cache существует {CACHE_PATH}, перезаписать? [y/N] ').strip().lower()
        if ans != 'y':
            print('Загружаю из кэша …')
            return pd.read_csv(CACHE_PATH)

    print(f'Loading {CHOICE_SETS} …')
    t0 = time.time()
    needed_cols = ['episode_id', 'player_idx', 'step', 'choice_set_id',
                    'planet_id', 'is_chosen'] + CONTEXT_FEATURES
    df = pd.read_csv(CHOICE_SETS, usecols=lambda c: c in needed_cols)
    print(f'  {len(df):,} строк за {time.time()-t0:.1f}s')
    df['episode_id'] = df['episode_id'].astype(str)

    rng = np.random.default_rng(seed)
    eids = df['episode_id'].unique()
    rng.shuffle(eids)
    keep = set(eids[:sample_episodes])
    df = df[df['episode_id'].isin(keep)].copy()
    print(f'  sample: {len(df):,} строк, {df["episode_id"].nunique()} эпизодов')

    # Загружаем UUID → numeric mapping
    uuid_map = build_uuid_mapping()
    print(f'  UUID mapping: {len(uuid_map)} эпизодов')

    results = []
    diag = {}
    no_json = 0
    not_in_map = 0
    t0 = time.time()
    by_ep = df.groupby('episode_id')
    n_eps = len(by_ep)
    print(f'  обрабатываю {n_eps} эпизодов …')
    for i, (eid, group) in enumerate(by_ep):
        # eid это UUID, мапим в numeric
        numeric_id = uuid_map.get(str(eid))
        if numeric_id is None:
            not_in_map += 1
            if not_in_map <= 3:
                print(f'    ⚠ UUID {eid} нет в mapping')
            continue
        json_path = RAW_DIR / f'{numeric_id}.json'
        if not json_path.exists():
            no_json += 1
            if no_json <= 3:
                print(f'    ⚠ JSON не найден: {json_path}')
            continue
        records = compute_features_for_episode(json_path, group, diag)
        results.extend(records)
        if (i + 1) % 5 == 0:
            print(f'    {i+1}/{n_eps}  ({time.time()-t0:.0f}s, {len(results)} records, diag={diag})')

    print(f'\n  Не нашлось в mapping: {not_in_map}')
    print(f'  no_json: {no_json}')

    print(f'\n  Diagnostic: no_json={no_json}')
    print(f'  Counters: {diag}')

    result_df = pd.DataFrame(results)
    if len(result_df) == 0:
        print('  ⚠ 0 records — что-то не так! Проверьте last_err выше.')
    result_df.to_csv(CACHE_PATH, index=False)
    print(f'\n✓ Cache сохранён: {CACHE_PATH}  ({len(result_df)} records за {time.time()-t0:.0f}s)')
    return result_df


# ════════════════════════════════════════════════════════════════════════
# Stage 2: Train на новых фичах
# ════════════════════════════════════════════════════════════════════════

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
            v = g[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
            s = feat_stats[c]
            feats[:, j] = (v - s['mean']) / max(s['std'], 1e-6)
        ctx = np.zeros(len(ctx_cols), dtype=np.float32)
        first = g.iloc[0]
        for j, c in enumerate(ctx_cols):
            v = float(first[c]) if not pd.isna(first[c]) else 0.0
            s = feat_stats[c]
            ctx[j] = (v - s['mean']) / max(s['std'], 1e-6)
        groups.append({'feats': feats, 'ctx': ctx, 'chosen_idx': chosen_idx,
                       'n': n, 'episode_id': str(first['episode_id'])})
    return groups, feat_stats


class LinearRanker(nn.Module):
    def __init__(self, n_planet, n_ctx):
        super().__init__()
        self.linear = nn.Linear(n_planet + n_ctx, 1, bias=False)

    def forward(self, planet_feats, ctx):
        n = planet_feats.size(0)
        ctx_exp = ctx.unsqueeze(0).expand(n, -1)
        x = torch.cat([planet_feats, ctx_exp], dim=-1)
        return self.linear(x).squeeze(-1)


class MLPRanker(nn.Module):
    def __init__(self, n_planet, n_ctx, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        d = n_planet + n_ctx
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, planet_feats, ctx):
        n = planet_feats.size(0)
        ctx_exp = ctx.unsqueeze(0).expand(n, -1)
        x = torch.cat([planet_feats, ctx_exp], dim=-1)
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
    return {'top1': top1/n, 'top3': top3/n, 'top5': top5/n, 'top10': top10/n,
            'median_rank': float(np.median(ranks)) if ranks else 0,
            'mean_rank': float(np.mean(ranks)) if ranks else 0,
            'n': n}


def baseline_zones_priority(groups, priority_idx=None):
    """Baseline: zones_priority (как считала эвристика) — это feature в df."""
    return evaluate_with_feature(groups, feature_idx=priority_idx)


def evaluate_with_feature(groups, feature_idx):
    """Sort by single feature (без обучения)."""
    top1 = top3 = top5 = top10 = 0
    ranks = []
    for g in groups:
        scores = g['feats'][:, feature_idx]
        order = np.argsort(-scores)
        rank = int(np.where(order == g['chosen_idx'])[0][0]) + 1
        ranks.append(rank)
        if rank <= 1:  top1 += 1
        if rank <= 3:  top3 += 1
        if rank <= 5:  top5 += 1
        if rank <= 10: top10 += 1
    n = max(len(groups), 1)
    return {'top1': top1/n, 'top3': top3/n, 'top5': top5/n, 'top10': top10/n,
            'median_rank': float(np.median(ranks)) if ranks else 0, 'n': n}


def train_loop(model, train_groups, val_groups, n_epochs=20, lr=0.01,
               batch_size=64, weight_decay=1e-4, patience=5,
               schedule=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if schedule:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    rng = np.random.default_rng(0)
    history = []
    best_top3 = -1
    best_state = None
    no_improve = 0
    for epoch in range(n_epochs):
        model.train()
        rng.shuffle(train_groups)
        loss_sum = 0
        n_batches = 0
        for i in range(0, len(train_groups), batch_size):
            batch = train_groups[i:i+batch_size]
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
            loss_sum += batch_loss.item()
            n_batches += 1
        if sched:
            sched.step()
        train_loss = loss_sum / max(n_batches, 1)
        vm = evaluate(model, val_groups)
        history.append({'epoch': epoch, 'train_loss': round(train_loss, 4),
                        'val_top1': vm['top1'], 'val_top3': vm['top3'],
                        'val_top5': vm['top5']})
        print(f'  epoch {epoch:2d}  train_loss={train_loss:.4f}  '
              f'val: top1={vm["top1"]:.4f}  top3={vm["top3"]:.4f}  '
              f'top5={vm["top5"]:.4f}')
        if vm['top3'] > best_top3:
            best_top3 = vm['top3']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping epoch {epoch}')
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {'history': history, 'best_metrics': evaluate(model, val_groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-episodes', type=int, default=80)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--mlp', action='store_true', help='Также обучить MLP версию')
    args = ap.parse_args()

    # Stage 1: build cache
    df_features = build_zones_features_dataset(sample_episodes=args.sample_episodes)
    print(f'\nDataset: {len(df_features)} records, {df_features["episode_id"].nunique()} эпизодов')

    # Stage 2: train
    train_df, val_df = split_by_episode(df_features, val_frac=0.15)
    print(f'Train: {len(train_df)}  Val: {len(val_df)}')

    feat_cols = [c for c in ZONES_PLANET_FEATURES if c in df_features.columns]
    ctx_cols  = [c for c in CONTEXT_FEATURES if c in df_features.columns]
    print(f'Features: {len(feat_cols)} planet + {len(ctx_cols)} context')

    train_groups, feat_stats = build_groups(train_df, feat_cols, ctx_cols)
    val_groups, _ = build_groups(val_df, feat_cols, ctx_cols, feat_stats=feat_stats)
    print(f'Groups: train={len(train_groups)}  val={len(val_groups)}')

    # ── Baseline: zones_priority (как ranking score) ──
    print('\n── Baseline: zones priority (без обучения) ──')
    # priority находится в df_features как отдельная колонка, не в feat_cols
    # Сделаем кастомную evaluation на zones_priority
    top1 = top3 = top5 = top10 = 0
    ranks_z = []
    for g_df_grp in val_df.groupby('choice_set_id'):
        csid, grp = g_df_grp
        if len(grp) < 2:
            continue
        grp = grp.reset_index(drop=True)
        chosen = grp.index[grp['is_chosen'] == 1].tolist()
        if not chosen:
            continue
        chosen = chosen[0]
        scores = grp['zones_priority'].values
        order = np.argsort(-scores)
        rank = int(np.where(order == chosen)[0][0]) + 1
        ranks_z.append(rank)
        if rank <= 1:  top1 += 1
        if rank <= 3:  top3 += 1
        if rank <= 5:  top5 += 1
        if rank <= 10: top10 += 1
    n = max(len(ranks_z), 1)
    baseline = {'top1': top1/n, 'top3': top3/n, 'top5': top5/n, 'top10': top10/n,
                'median_rank': float(np.median(ranks_z)) if ranks_z else 0, 'n': n}
    print(f'  top1={baseline["top1"]:.4f}  top3={baseline["top3"]:.4f}  '
          f'top5={baseline["top5"]:.4f}  med_rank={baseline["median_rank"]:.1f}')

    # ── Linear ──
    print('\n── Training LINEAR на zones features ──')
    model_lin = LinearRanker(n_planet=len(feat_cols), n_ctx=len(ctx_cols))
    result_lin = train_loop(model_lin, train_groups, val_groups,
                             n_epochs=args.epochs, lr=args.lr)
    final_lin = result_lin['best_metrics']
    print(f'  Best: top1={final_lin["top1"]:.4f}  top3={final_lin["top3"]:.4f}  '
          f'top5={final_lin["top5"]:.4f}')

    # Сохраняем веса
    w = model_lin.linear.weight.detach().cpu().numpy().squeeze()
    weights_planet = {feat_cols[i]: float(w[i]) for i in range(len(feat_cols))}
    weights_ctx    = {ctx_cols[i]: float(w[len(feat_cols) + i]) for i in range(len(ctx_cols))}
    torch.save(model_lin.state_dict(), MODEL_DIR / 'ml_zones_linear_weights.pt')

    meta = {
        'arch': 'linear',
        'planet_features': feat_cols,
        'context_features': ctx_cols,
        'feat_stats': feat_stats,
        'weights_planet': weights_planet,
        'weights_ctx': weights_ctx,
        'baseline_zones_priority': baseline,
        'final_metrics': final_lin,
        'history': result_lin['history'],
    }
    (MODEL_DIR / 'ml_zones_linear_meta.json').write_text(json.dumps(meta, indent=2))

    print(f'\n  Топ-10 весов по |w|:')
    all_w = [(c, w[i], 'planet') for i, c in enumerate(feat_cols)] + \
            [(c, w[len(feat_cols)+i], 'context') for i, c in enumerate(ctx_cols)]
    all_w.sort(key=lambda x: -abs(x[1]))
    for name, val, kind in all_w[:10]:
        print(f'    {kind:<8} {name:<22}  {val:+.4f}')

    # ── MLP опционально ──
    final_mlp = None
    if args.mlp:
        print('\n── Training MLP на zones features ──')
        model_mlp = MLPRanker(n_planet=len(feat_cols), n_ctx=len(ctx_cols),
                                hidden=(64, 32))
        result_mlp = train_loop(model_mlp, train_groups, val_groups,
                                 n_epochs=args.epochs, lr=args.lr*0.1,
                                 schedule=True)
        final_mlp = result_mlp['best_metrics']
        torch.save(model_mlp.state_dict(), MODEL_DIR / 'ml_zones_mlp_weights.pt')
        mlp_meta = {'arch': 'mlp_64_32', 'feat_stats': feat_stats,
                    'planet_features': feat_cols, 'context_features': ctx_cols,
                    'final_metrics': final_mlp, 'history': result_mlp['history']}
        (MODEL_DIR / 'ml_zones_mlp_meta.json').write_text(json.dumps(mlp_meta, indent=2))
        print(f'  MLP best: top1={final_mlp["top1"]:.4f}  top3={final_mlp["top3"]:.4f}')

    # ── Сравнение ──
    print('\n══════════ СВОДКА ══════════')
    print(f'  Model              top1     top3     top5     top10    med_rank')
    print(f'  baseline (zones)   {baseline["top1"]:.4f}   {baseline["top3"]:.4f}   '
          f'{baseline["top5"]:.4f}   {baseline["top10"]:.4f}   {baseline["median_rank"]:.1f}')
    print(f'  linear             {final_lin["top1"]:.4f}   {final_lin["top3"]:.4f}   '
          f'{final_lin["top5"]:.4f}   {final_lin["top10"]:.4f}   {final_lin["median_rank"]:.1f}')
    if final_mlp:
        print(f'  MLP                {final_mlp["top1"]:.4f}   {final_mlp["top3"]:.4f}   '
              f'{final_mlp["top5"]:.4f}   {final_mlp["top10"]:.4f}   {final_mlp["median_rank"]:.1f}')

    # Markdown
    md = []
    md.append('# Retrained ranker — zones features (готов к интеграции)\n\n')
    md.append(f'**Dataset**: {len(train_groups)} train + {len(val_groups)} val choice_sets\n')
    md.append(f'**Features**: {len(feat_cols)} planet + {len(ctx_cols)} context\n\n')
    md.append('## Метрики\n\n')
    md.append('| Model | top1 | top3 | top5 | top10 | median_rank |\n|---|---|---|---|---|---|\n')
    md.append(f'| baseline (zones priority) | {baseline["top1"]:.4f} | '
              f'{baseline["top3"]:.4f} | {baseline["top5"]:.4f} | '
              f'{baseline["top10"]:.4f} | {baseline["median_rank"]:.1f} |\n')
    md.append(f'| linear | {final_lin["top1"]:.4f} | {final_lin["top3"]:.4f} | '
              f'{final_lin["top5"]:.4f} | {final_lin["top10"]:.4f} | '
              f'{final_lin["median_rank"]:.1f} |\n')
    if final_mlp:
        md.append(f'| MLP | {final_mlp["top1"]:.4f} | {final_mlp["top3"]:.4f} | '
                  f'{final_mlp["top5"]:.4f} | {final_mlp["top10"]:.4f} | '
                  f'{final_mlp["median_rank"]:.1f} |\n')

    md.append('\n## Топ-10 весов linear (по |w|)\n\n')
    md.append('| kind | feature | weight |\n|---|---|---|\n')
    for name, val, kind in all_w[:10]:
        md.append(f'| {kind} | {name} | {val:+.4f} |\n')

    md.append('\n## Готов для интеграции\n\n')
    md.append('Эти веса можно подключить как residual в zones.compute_zones_from_state\n')
    md.append('через скрипт 19_tournament_eval.py.\n')

    (REPORT_DIR / 'ml_zones_retrain_report.md').write_text(''.join(md))
    print(f'\n✓ Сохранено:')
    print(f'  weights → {MODEL_DIR / "ml_zones_linear_weights.pt"}')
    print(f'  meta    → {MODEL_DIR / "ml_zones_linear_meta.json"}')
    print(f'  отчёт   → {REPORT_DIR / "ml_zones_retrain_report.md"}')


if __name__ == '__main__':
    main()
