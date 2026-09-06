#!/usr/bin/env python3
"""
train_gnn.py — Policy GNN для orbit_wars (GPU-ready, mini-batch training).

Запуск:
  python train_gnn.py --smoke-test            # проверить архитектуру
  python train_gnn.py --limit 50 --epochs 10  # быстрый тест
  python train_gnn.py --epochs 50             # полное обучение
  python train_gnn.py --epochs 50 --batch-size 64  # крупнее батч для GPU
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

HERE      = Path(__file__).parent
RAW_DIR   = HERE / "data" / "raw"
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NEUTRAL_OWNER = -1
NODE_DIM      = 10


# ═══════════════════════════════════════════════════════════════════
# Парсинг реплея → граф + метки
# ═══════════════════════════════════════════════════════════════════

def _parse_planet(p):
    if isinstance(p, (list, tuple)) and len(p) >= 7:
        return {'id': int(p[0]),
                'owner': int(p[1]) if p[1] is not None else NEUTRAL_OWNER,
                'x': float(p[2]), 'y': float(p[3]),
                'ships': max(0.0, float(p[5])), 'prod': float(p[6])}
    if isinstance(p, dict):
        owner = p.get('owner', NEUTRAL_OWNER)
        return {'id': int(p['id']),
                'owner': owner if owner is not None else NEUTRAL_OWNER,
                'x': float(p.get('x', 0)), 'y': float(p.get('y', 0)),
                'ships': max(0.0, float(p.get('ships', 0))),
                'prod': float(p.get('production', p.get('prod', 0)))}
    return None


def _find_target(fx, fy, angle, planets):
    dx, dy = math.cos(angle), math.sin(angle)
    best_id, best_score = None, float('inf')
    for p in planets:
        tdx, tdy = p['x'] - fx, p['y'] - fy
        dist = math.hypot(tdx, tdy)
        if dist < 1e-6:
            continue
        if (dx * tdx + dy * tdy) / dist < 0.2:
            continue
        ang = math.atan2(tdy, tdx)
        ad = abs(math.atan2(math.sin(angle - ang), math.cos(angle - ang)))
        if ad < best_score:
            best_score, best_id = ad, p['id']
    return best_id


def build_graph(planets, fleets, pidx):
    n = len(planets)
    if n == 0:
        return None

    xs     = np.array([p['x']     for p in planets], dtype=np.float32)
    ys     = np.array([p['y']     for p in planets], dtype=np.float32)
    ships  = np.array([p['ships'] for p in planets], dtype=np.float32)
    prods  = np.array([p['prod']  for p in planets], dtype=np.float32)
    owners = np.array([p['owner'] for p in planets], dtype=np.int32)
    ids    = np.array([p['id']    for p in planets], dtype=np.int32)
    id2idx = {int(pid): i for i, pid in enumerate(ids)}

    x_norm = (xs - xs.min()) / max(xs.max() - xs.min(), 1.0)
    y_norm = (ys - ys.min()) / max(ys.max() - ys.min(), 1.0)

    total_ships = ships.sum() or 1.0
    total_prod  = prods.sum() or 1.0
    ripeness    = prods / (1.0 + ships)
    ripe_max    = ripeness.max() or 1.0

    is_own     = (owners == pidx).astype(np.float32)
    is_neutral = (owners == NEUTRAL_OWNER).astype(np.float32)
    is_enemy   = ((owners != pidx) & (owners != NEUTRAL_OWNER)).astype(np.float32)

    inc_fri = np.zeros(n, dtype=np.float32)
    inc_enm = np.zeros(n, dtype=np.float32)
    for f in fleets:
        try:
            if isinstance(f, (list, tuple)):
                fo = int(f[1]); fx_ = float(f[2]); fy_ = float(f[3])
                fa = float(f[4]); fs = float(f[5])
            else:
                fo = int(f.get('owner', -1)); fx_ = float(f.get('x', 0))
                fy_ = float(f.get('y', 0)); fa = float(f.get('angle', 0))
                fs = float(f.get('ships', 0))
        except Exception:
            continue
        tid = _find_target(fx_, fy_, fa, planets)
        if tid is not None and tid in id2idx:
            ti = id2idx[tid]
            if fo == pidx: inc_fri[ti] += fs
            else:          inc_enm[ti] += fs

    H = np.stack([
        ships / total_ships,
        prods / total_prod,
        ripeness / ripe_max,
        is_own, is_neutral, is_enemy,
        x_norm, y_norm,
        inc_fri / (inc_fri.max() or 1.0),
        inc_enm / (inc_enm.max() or 1.0),
    ], axis=1).astype(np.float32)

    return {'H': H,
            'coords': np.stack([x_norm, y_norm], axis=1).astype(np.float32),
            'ids': ids, 'id2idx': id2idx,
            'own_mask': is_own.astype(bool),
            'planets': planets}


def build_action_labels(actions, graph):
    n      = len(graph['ids'])
    id2idx = graph['id2idx']
    pls    = {p['id']: p for p in graph['planets']}

    src_labels = np.zeros(n, dtype=np.float32)
    tgt_labels = np.full(n, -1, dtype=np.int32)
    ships_frac = np.zeros(n, dtype=np.float32)

    for act in (actions or []):
        try:
            src_id, angle, n_sent = int(act[0]), float(act[1]), float(act[2])
        except Exception:
            continue
        if src_id not in id2idx:
            continue
        si = id2idx[src_id]
        tid = _find_target(pls[src_id]['x'], pls[src_id]['y'], angle, graph['planets'])
        if tid is None or tid not in id2idx:
            continue
        ti = id2idx[tid]
        src_labels[si] = 1.0
        tgt_labels[si] = ti
        ships_frac[si] = min(n_sent / max(pls[src_id]['ships'], 1.0), 1.0)

    return {'did_act': bool(actions),
            'src_labels': src_labels,
            'tgt_labels': tgt_labels,
            'ships_frac': ships_frac}


def load_dataset(raw_dir, limit=None):
    files = sorted(fp for fp in raw_dir.glob('*.json') if fp.stem.isdigit())
    if limit:
        files = files[:limit]

    samples = []
    for fp in files:
        try:
            d = json.loads(fp.read_bytes())
        except Exception:
            continue
        if not isinstance(d, dict) or 'steps' not in d:
            continue

        rewards = d.get('rewards', [])
        eid     = str(d.get('id', fp.stem))

        for si, step in enumerate(d['steps']):
            for pidx in range(len(step)):
                reward  = float(rewards[pidx] or 0) if pidx < len(rewards) else 0.0
                obs     = step[pidx].get('observation', {})
                actions = step[pidx].get('action') or []
                pls_raw = obs.get('planets', [])
                if not pls_raw:
                    continue
                planets = [_parse_planet(p) for p in pls_raw]
                planets = [p for p in planets if p]
                if len(planets) < 3:
                    continue
                graph = build_graph(planets, obs.get('fleets', []), pidx)
                if graph is None:
                    continue
                samples.append({'graph':   graph,
                                'labels':  build_action_labels(actions, graph),
                                'reward':  reward,
                                'episode': eid})

    n_act = sum(1 for s in samples if s['labels']['did_act'])
    print(f"Loaded {len(samples):,} steps from {len(files)} replays  "
          f"(did_act={n_act/max(len(samples),1):.1%}  "
          f"avg_nodes={np.mean([s['graph']['H'].shape[0] for s in samples]):.1f})")
    return samples


def compute_baselines(samples):
    """
    Считает базовые линии эвристик чтобы контекстуализировать метрики модели:
      - always_act_bacc: balanced accuracy если всегда предсказывать 'атаковать'
      - nearest_tgt_acc: точность если всегда выбирать ближайшую не-свою планету
      - nearest_tgt_mrr:  MRR для той же эвристики
    """
    tp = tn = fp = fn = 0
    near_correct = near_total = 0
    near_rr_sum = 0.0

    for s in samples:
        act = s['labels']['did_act']
        # did_act: "always act" — предсказываем 1 всегда
        if act:  tp += 1
        else:    fp += 1  # ложная тревога

        # tgt: nearest non-own planet
        coords   = s['graph']['coords']          # [N, 2]
        own_mask = s['graph']['own_mask']         # [N] bool
        tgt_lbl  = s['labels']['tgt_labels']      # [N] int, -1 = not launched

        for si in range(len(own_mask)):
            if not own_mask[si]:
                continue
            ti = int(tgt_lbl[si])
            if ti < 0:
                continue
            # дистанции от si до всех планет
            dists = np.linalg.norm(coords - coords[si], axis=1)
            dists[si] = np.inf                    # не себя
            # сортируем: closest first
            ranked = np.argsort(dists)            # [N] indices by distance
            rank_of_ti = int(np.where(ranked == ti)[0][0]) + 1  # 1-based
            near_correct += int(rank_of_ti == 1)
            near_rr_sum  += 1.0 / rank_of_ti
            near_total   += 1

    # balanced accuracy для "always act"
    tpr = tp / max(tp + fn, 1)   # fn=0 т.к. мы всегда угадываем positives
    tnr = tn / max(tn + fp, 1)   # tn=0 т.к. мы никогда не предсказываем 0
    # правильнее: always_act → TPR=1, TNR=0 → bacc=0.5
    always_act_bacc = 0.5

    near_acc = near_correct / max(near_total, 1)
    near_mrr = near_rr_sum  / max(near_total, 1)
    return {
        'always_act_bacc':  always_act_bacc,
        'act_rate':         tp / max(tp + fp, 1),   # реальная доля шагов с атакой
        'nearest_tgt_acc':  near_acc,
        'nearest_tgt_mrr':  near_mrr,
        'n_launches':       near_total,
    }


# ═══════════════════════════════════════════════════════════════════
# Collate: паддинг + стек в батч
# ═══════════════════════════════════════════════════════════════════

def collate_fn(samples, max_N=None):
    """Паддит графы до max_N и стекает в numpy-батч."""
    if max_N is None:
        max_N = max(s['graph']['H'].shape[0] for s in samples)
    B = len(samples)

    H_b      = np.zeros((B, max_N, NODE_DIM), dtype=np.float32)
    coords_b = np.zeros((B, max_N, 2),        dtype=np.float32)
    pad_b    = np.zeros((B, max_N),            dtype=bool)
    own_b    = np.zeros((B, max_N),            dtype=bool)
    src_b    = np.zeros((B, max_N),            dtype=np.float32)
    tgt_b    = np.full ((B, max_N), -1,        dtype=np.int32)
    shp_b    = np.zeros((B, max_N),            dtype=np.float32)
    da_b     = np.zeros(B,                     dtype=np.float32)
    rew_b    = np.zeros(B,                     dtype=np.float32)

    for i, s in enumerate(samples):
        n = s['graph']['H'].shape[0]
        H_b[i, :n]   = s['graph']['H']
        coords_b[i,:n] = s['graph']['coords']
        pad_b[i, :n] = True
        own_b[i, :n] = s['graph']['own_mask']
        src_b[i, :n] = s['labels']['src_labels']
        tgt_b[i, :n] = s['labels']['tgt_labels']
        shp_b[i, :n] = s['labels']['ships_frac']
        da_b[i]      = float(s['labels']['did_act'])
        rew_b[i]     = s['reward']

    return {'H': H_b, 'coords': coords_b,
            'pad_mask': pad_b, 'own_mask': own_b,
            'src_labels': src_b, 'tgt_labels': tgt_b,
            'ships_frac': shp_b, 'did_act': da_b, 'reward': rew_b}


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

def _try_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError:
        return None, None, None


def build_model(node_dim=NODE_DIM, hidden=64, n_layers=2, dropout=0.2):
    torch, nn, F = _try_torch()
    if torch is None:
        raise ImportError("pip install torch")

    class MPLayer(nn.Module):
        """Message passing — работает и с [N,D] и с [B,N,D]."""
        def __init__(self, dim):
            super().__init__()
            self.W_self = nn.Linear(dim, dim, bias=False)
            self.W_msg  = nn.Linear(dim, dim, bias=False)
            self.norm   = nn.LayerNorm(dim)
            self.drop   = nn.Dropout(dropout)

        def forward(self, H, A):
            msg = torch.bmm(A, H) if H.dim() == 3 else A @ H
            return H + self.drop(F.relu(self.norm(self.W_self(H) + self.W_msg(msg))))

    class PolicyGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = nn.Sequential(nn.Linear(node_dim, hidden),
                                            nn.ReLU(), nn.LayerNorm(hidden))
            self.mp         = nn.ModuleList([MPLayer(hidden) for _ in range(n_layers)])
            self.drop       = nn.Dropout(dropout)
            self.did_act_head = nn.Sequential(
                nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.src_head   = nn.Sequential(
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
            self.tgt_head   = nn.Sequential(
                nn.Linear(hidden*2+2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.ships_head = nn.Sequential(
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
            self.attn_q     = nn.Linear(hidden, 1)
            self.val_head   = nn.Sequential(
                nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

        def _adj(self, coords, pad_mask=None):
            """Proximity matrix, row-normalized. coords: [N,2] или [B,N,2]."""
            if coords.dim() == 2:
                diff = coords[:, None, :] - coords[None, :, :]      # [N,N,2]
                D    = diff.norm(dim=-1).clamp(min=1e-3)
                A    = 1.0 / D
                A    = A * (1 - torch.eye(A.size(0), device=A.device))
            else:
                B, N, _ = coords.shape
                diff = coords[:, :, None, :] - coords[:, None, :, :]  # [B,N,N,2]
                D    = diff.norm(dim=-1).clamp(min=1e-3)
                A    = 1.0 / D
                A    = A * (1 - torch.eye(N, device=coords.device).unsqueeze(0))
                if pad_mask is not None:
                    v = pad_mask.float()
                    A = A * v[:, :, None] * v[:, None, :]
            return A / A.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        def _encode(self, H, coords, pad_mask=None):
            h = self.encoder(H)
            A = self._adj(coords, pad_mask)
            for layer in self.mp:
                h = layer(h, A)
            return h

        def _readout(self, h, pad_mask=None):
            """mean + attention pooling → [2*hidden] или [B, 2*hidden]."""
            if h.dim() == 2:
                mean_g = h.mean(0)
                attn   = torch.softmax(self.attn_q(h).squeeze(-1), 0)
                attn_g = (attn[:, None] * h).sum(0)
            else:
                valid  = (pad_mask.float().unsqueeze(-1)
                          if pad_mask is not None
                          else torch.ones_like(h[..., :1]))
                mean_g = (h * valid).sum(1) / valid.sum(1).clamp(1)
                raw    = self.attn_q(h).squeeze(-1)
                if pad_mask is not None:
                    raw = raw.masked_fill(~pad_mask, -1e9)
                attn   = torch.softmax(raw, dim=-1)
                attn_g = (attn.unsqueeze(-1) * h).sum(1)
            return torch.cat([mean_g, attn_g], dim=-1)

        def forward_batch(self, H, coords, pad_mask, own_mask):
            """
            Батчевый forward — основной для обучения на GPU.
            H: [B,N,D]  coords: [B,N,2]  pad_mask/own_mask: [B,N] bool
            """
            B, N, _ = H.shape
            h = self._encode(H, coords, pad_mask)        # [B, N, hidden]
            g = self._readout(h, pad_mask)               # [B, 2*hidden]

            did_act_logits = self.did_act_head(g).squeeze(-1)          # [B]
            values         = torch.sigmoid(self.val_head(g).squeeze(-1))  # [B]
            src_logits     = self.src_head(h).squeeze(-1)              # [B, N]
            ships_out      = torch.sigmoid(self.ships_head(h).squeeze(-1))  # [B, N]

            # Target scores: все пары (src_i, tgt_j) → [B, N, N]
            h_i   = h.unsqueeze(2).expand(B, N, N, -1)
            h_j   = h.unsqueeze(1).expand(B, N, N, -1)
            diff  = coords[:, :, None, :] - coords[:, None, :, :]      # [B,N,N,2]
            dist  = diff.norm(dim=-1).clamp(1e-3)
            max_d = dist.flatten(1).max(1)[0][:, None, None].clamp(1.0)
            ef    = torch.stack([1.0/(dist+1.0), dist/max_d], dim=-1)  # [B,N,N,2]
            tgt_logits = self.tgt_head(
                torch.cat([h_i, h_j, ef], dim=-1)).squeeze(-1)         # [B, N, N]

            return {'did_act_logits': did_act_logits,
                    'src_logits':     src_logits,
                    'tgt_logits':     tgt_logits,
                    'ships_frac':     ships_out,
                    'values':         values}

        def forward_all(self, H, coords, own_mask):
            """Single-graph forward для инференса."""
            h = self._encode(H, coords)
            g = self._readout(h)
            did_act_logit = self.did_act_head(g).squeeze(-1)
            value         = torch.sigmoid(self.val_head(g).squeeze(-1))
            src_logits    = self.src_head(h).squeeze(-1)
            ships_out     = torch.sigmoid(self.ships_head(h).squeeze(-1))
            own_idx   = own_mask.nonzero(as_tuple=True)[0]
            other_idx = (~own_mask).nonzero(as_tuple=True)[0]
            tgt_scores = {}
            if len(own_idx) > 0 and len(other_idx) > 0:
                for ii, si in enumerate(own_idx):
                    pairs = []
                    for oj in other_idx:
                        d    = (coords[oj] - coords[si]).norm().clamp(1e-3)
                        md   = (coords[:, None, :]-coords[None,:,:]).norm(dim=-1).max().clamp(1.)
                        ef   = torch.stack([1.0/(d+1.0), d/md])
                        pairs.append(torch.cat([h[si], h[oj], ef]))
                    tgt_scores[ii] = (self.tgt_head(torch.stack(pairs)).squeeze(-1), other_idx)
            return {'did_act_logit': did_act_logit, 'src_logits': src_logits,
                    'tgt_scores': tgt_scores, 'ships_frac': ships_out,
                    'value': value, 'own_idx': own_idx, 'other_idx': other_idx}

    return PolicyGNN()


# ═══════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════

def compute_loss_batch(out, batch, device, lam=(1.0, 0.8, 1.5, 0.5, 0.3)):
    import torch
    import torch.nn.functional as F
    λ_da, λ_src, λ_tgt, λ_ships, λ_val = lam

    def _t(arr, dtype=None):
        t = torch.tensor(arr, device=device)
        return t if dtype is None else t.to(dtype)

    pad = _t(batch['pad_mask'])
    own = _t(batch['own_mask'])

    # L1: did_act
    l_da = F.binary_cross_entropy_with_logits(
        out['did_act_logits'], _t(batch['did_act'], torch.float32))

    # L2: src multi-label BCE (только real own nodes)
    src_mask = own & pad
    l_src = (F.binary_cross_entropy_with_logits(
                out['src_logits'][src_mask], _t(batch['src_labels'])[src_mask])
             if src_mask.any() else torch.tensor(0., device=device))

    # L3: tgt CE per launched src
    # NOTE: в orbit_wars можно посылать флот на свои планеты (подкрепление),
    #       поэтому НЕ маскируем own-узлы как невалидные цели.
    #       Маскируем только: паддинг-узлы + планету-источник si.
    tgt_t = _t(batch['tgt_labels'], torch.long)
    l_tgt = torch.tensor(0., device=device)
    n_tgt = 0
    n_tgt_correct = 0
    B, N  = pad.shape
    for b in range(B):
        for si in range(N):
            if not own[b, si]:
                continue
            ti = int(tgt_t[b, si].item())
            if ti < 0:
                continue
            logits = out['tgt_logits'][b, si].clone()
            logits = logits.masked_fill(~pad[b], -1e9)     # паддинг → невалид
            logits[si] = -1e9                              # нельзя в себя
            l_tgt = l_tgt + F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor(ti, device=device, dtype=torch.long).unsqueeze(0))
            n_tgt += 1
            if int(logits.argmax().item()) == ti:
                n_tgt_correct += 1
    if n_tgt > 0:
        l_tgt = l_tgt / n_tgt

    # L4: ships MSE
    launched = (_t(batch['ships_frac']) > 0) & own & pad
    l_ships = (F.mse_loss(out['ships_frac'][launched], _t(batch['ships_frac'])[launched])
               if launched.any() else torch.tensor(0., device=device))

    # L5: value BCE
    y_val = _t((batch['reward'] + 1) / 2, torch.float32)
    l_val = F.binary_cross_entropy(out['values'].clamp(1e-6, 1-1e-6), y_val)

    total = λ_da*l_da + λ_src*l_src + λ_tgt*l_tgt + λ_ships*l_ships + λ_val*l_val
    return total, {'did_act': l_da.item(), 'src': l_src.item(),
                   'tgt': l_tgt.item(), 'n_tgt': n_tgt, 'n_tgt_correct': n_tgt_correct,
                   'ships': l_ships.item(), 'value': l_val.item()}


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def _save_checkpoint(model, out_dir, epoch, hidden, n_layers):
    import torch
    path = out_dir / f'policy_gnn_ep{epoch:03d}.pt'
    torch.save({'model_state': model.state_dict(),
                'node_dim':    NODE_DIM,
                'hidden':      hidden,
                'n_layers':    n_layers,
                'epoch':       epoch}, path)
    print(f"  💾 checkpoint: {path.name}")
    return path


def train(samples, n_epochs, lr, device, batch_size=32, hidden=64, n_layers=2,
          save_every=5, out_dir=None):
    import torch

    if out_dir is None:
        out_dir = MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model     = build_model(hidden=hidden, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    episodes = list({s['episode'] for s in samples})
    rng      = np.random.default_rng(42)
    rng.shuffle(episodes)
    val_eps  = set(episodes[:max(1, len(episodes)//8)])
    tr_data  = [s for s in samples if s['episode'] not in val_eps]
    va_data  = [s for s in samples if s['episode'] in val_eps]

    # Фиксируем max_N один раз — быстрее паддить
    max_N  = max(s['graph']['H'].shape[0] for s in samples)
    n_par  = sum(p.numel() for p in model.parameters())
    n_tr   = len(tr_data)
    print(f"\nTrain: {n_tr:,} | Val: {len(va_data):,} | "
          f"batch={batch_size} | max_N={max_N} | params={n_par:,} | device={device}")

    print("Считаем базовые линии эвристик...")
    bl = compute_baselines(va_data)
    print(f"  Базовые линии (val):")
    print(f"    did_act  — act_rate={bl['act_rate']:.3f}  always_act bacc={bl['always_act_bacc']:.3f}")
    print(f"    tgt      — nearest acc={bl['nearest_tgt_acc']:.3f}  nearest MRR={bl['nearest_tgt_mrr']:.3f}")
    print(f"    random   — acc={1/max(max_N-1,1):.3f}  MRR≈{sum(1/i for i in range(1,max_N))/(max_N-1):.3f}")
    print()

    for epoch in range(1, n_epochs + 1):
        model.train()
        rng.shuffle(tr_data)
        ep_loss = 0
        da_correct = tgt_correct = tgt_total = 0

        for i in range(0, n_tr, batch_size):
            bs    = tr_data[i:i+batch_size]
            batch = collate_fn(bs, max_N=max_N)

            H      = torch.tensor(batch['H'],         device=device)
            coords = torch.tensor(batch['coords'],    device=device)
            pad    = torch.tensor(batch['pad_mask'],  device=device)
            own    = torch.tensor(batch['own_mask'],  device=device)

            out  = model.forward_batch(H, coords, pad, own)
            loss, sub = compute_loss_batch(out, batch, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss    += loss.item() * len(bs)
            da_correct += int(((out['did_act_logits'] > 0).cpu().numpy()
                                == batch['did_act'].astype(bool)).sum())
            tgt_correct += sub['n_tgt_correct']
            tgt_total   += sub['n_tgt']

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            # val: balanced accuracy для did_act + MRR/top3 для tgt
            va_tp = va_tn = va_fp = va_fn = 0
            va_mrr = va_top3 = va_tgt_t = 0
            with torch.no_grad():
                for i in range(0, len(va_data), batch_size):
                    vb     = collate_fn(va_data[i:i+batch_size], max_N=max_N)
                    H_v    = torch.tensor(vb['H'],        device=device)
                    c_v    = torch.tensor(vb['coords'],   device=device)
                    p_v    = torch.tensor(vb['pad_mask'], device=device)
                    o_v    = torch.tensor(vb['own_mask'], device=device)
                    o_out  = model.forward_batch(H_v, c_v, p_v, o_v)

                    # balanced accuracy
                    pred_da = (o_out['did_act_logits'] > 0).cpu().numpy()
                    true_da = vb['did_act'].astype(bool)
                    va_tp += int(( pred_da &  true_da).sum())
                    va_tn += int((~pred_da & ~true_da).sum())
                    va_fp += int(( pred_da & ~true_da).sum())
                    va_fn += int((~pred_da &  true_da).sum())

                    # MRR + top-3 для tgt
                    tgt_logits_np = o_out['tgt_logits'].cpu().numpy()   # [B, N, N]
                    pad_np  = vb['pad_mask']   # [B, N]
                    own_np  = vb['own_mask']   # [B, N]
                    tgt_lbl = vb['tgt_labels'] # [B, N]
                    Bv, Nv  = pad_np.shape
                    for b in range(Bv):
                        for si in range(Nv):
                            if not own_np[b, si]:
                                continue
                            ti = int(tgt_lbl[b, si])
                            if ti < 0:
                                continue
                            logits = tgt_logits_np[b, si].copy()
                            logits[~pad_np[b]] = -1e9
                            logits[si]         = -1e9
                            # rank of correct target (1-based)
                            rank = int((logits > logits[ti]).sum()) + 1
                            va_mrr  += 1.0 / rank
                            va_top3 += int(rank <= 3)
                            va_tgt_t += 1

            tpr = va_tp / max(va_tp + va_fn, 1)
            tnr = va_tn / max(va_tn + va_fp, 1)
            va_bacc = (tpr + tnr) / 2
            va_mrr_avg  = va_mrr  / max(va_tgt_t, 1)
            va_top3_acc = va_top3 / max(va_tgt_t, 1)

            # train tgt_acc для справки
            tgt_acc = tgt_correct / max(tgt_total, 1)
            print(f"  ep {epoch:>3} | loss={ep_loss/n_tr:.4f} | "
                  f"tr tgt_acc={tgt_acc:.3f} | "
                  f"val: bacc={va_bacc:.3f}(tpr={tpr:.2f} tnr={tnr:.2f}) | "
                  f"tgt MRR={va_mrr_avg:.3f} top3={va_top3_acc:.3f}")

        # Сохраняем чекпоинт каждые save_every эпох
        if epoch % save_every == 0 or epoch == n_epochs:
            _save_checkpoint(model, out_dir, epoch, hidden, n_layers)

    return model


# ═══════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════

def smoke_test():
    torch, nn, F = _try_torch()
    if torch is None:
        print("PyTorch не установлен — см. setup_win.bat")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model().to(device)
    model.eval()
    B, N = 4, 24

    H      = torch.randn(B, N, NODE_DIM, device=device)
    coords = torch.rand(B, N, 2, device=device)
    pad    = torch.ones(B, N, dtype=torch.bool, device=device)
    own    = torch.zeros(B, N, dtype=torch.bool, device=device)
    own[:, :6] = True

    with torch.no_grad():
        out = model.forward_batch(H, coords, pad, own)

    print(f"\nInput:           B={B}, N={N}, node_dim={NODE_DIM}")
    print(f"did_act_logits:  {tuple(out['did_act_logits'].shape)}")
    print(f"src_logits:      {tuple(out['src_logits'].shape)}")
    print(f"tgt_logits:      {tuple(out['tgt_logits'].shape)}  ← src×tgt пары")
    print(f"ships_frac:      {tuple(out['ships_frac'].shape)}")
    print(f"values:          {tuple(out['values'].shape)}")
    print(f"Model params:    {sum(p.numel() for p in model.parameters()):,}")

    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"GPU mem used:    {mem:.1f} MB")

    print("\n✓ Smoke test passed")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test',  action='store_true')
    parser.add_argument('--limit',       type=int,   default=None,
                        help='Кол-во реплеев (None=все)')
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--batch-size',  type=int,   default=32,
                        help='Батч-сайз (64-128 оптимально для GPU)')
    parser.add_argument('--hidden',      type=int,   default=64)
    parser.add_argument('--layers',      type=int,   default=2)
    parser.add_argument('--save-every',  type=int,   default=5,
                        help='Сохранять чекпоинт каждые N эпох (default=5)')
    parser.add_argument('--out-dir',     default=None)
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    torch, _, _ = _try_torch()
    if torch is None:
        print("⚠ PyTorch не установлен:")
        print("  CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")
    else:
        print("CUDA не найдена — обучение на CPU (медленнее)")
        print("Для GPU установи: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    samples = load_dataset(RAW_DIR, limit=args.limit)
    if not samples:
        print("Нет данных в data/raw/ — запусти: python download_data.py")
        return

    out_dir = Path(args.out_dir) if args.out_dir else MODEL_DIR

    model = train(samples, n_epochs=args.epochs, lr=args.lr,
                  device=device, batch_size=args.batch_size,
                  hidden=args.hidden, n_layers=args.layers,
                  save_every=args.save_every, out_dir=out_dir)

    # финальная модель без номера эпохи — для агента
    path = out_dir / 'policy_gnn.pt'
    import torch as _torch
    _torch.save({'model_state': model.state_dict(),
                 'node_dim': NODE_DIM,
                 'hidden':   args.hidden,
                 'n_layers': args.layers}, path)
    print(f"\n✓ Финальная модель: {path}")


if __name__ == '__main__':
    main()
