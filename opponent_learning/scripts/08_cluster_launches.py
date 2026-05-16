#!/usr/bin/env python3
"""
08_cluster_launches.py — per-launch behavioral clustering using zone-value features.

What it does:
  1. Loads launches.csv (output of 02c_extract_launches.py)
  2. Splits launches by target type: neutral / enemy / reinforce (own)
  3. Clusters each type independently using zone-aware relative-choice features
  4. Profiles each cluster in human-readable zone-value language
  5. Saves:
       results/launch_clusters.csv          — launches + cluster labels per type
       results/launch_archetypes.json       — archetype profiles (for opponent model)
       results/launch_cluster_report.txt    — human-readable summary

Key insight:
  Relative features (rel_prod, rel_wnn, rel_late_agg, …) describe *why* a target
  was chosen relative to alternatives — they're independent of map size/phase,
  making them the cleanest signal for discovering true behavioral archetypes.

Run:
  python3 opponent_learning/scripts/08_cluster_launches.py
  python3 opponent_learning/scripts/08_cluster_launches.py --k-neutral 4 --k-enemy 5 --k-reinforce 3
  python3 opponent_learning/scripts/08_cluster_launches.py --no-plots
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.decomposition import PCA
except ImportError:
    raise ImportError("pip install scikit-learn --break-system-packages")

HERE     = Path(__file__).parent
OL_DIR   = HERE.parent
DATA_DIR = OL_DIR / "data" / "processed"
RES_DIR  = OL_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

LAUNCHES_CSV = DATA_DIR / "launches.csv"

# ── Feature sets for clustering ───────────────────────────────────────────────

# Relative choice features: how chosen target compares to alternatives
REL_COLS = [
    'rel_prod',        # chosen_prod - mean_avail_prod          (higher = richer pick)
    'rel_ships',       # chosen_ships - mean_avail_ships         (lower = safer pick)
    'rel_dist',        # chosen_dist - mean_avail_dist           (higher = farther pick)
    'rel_ripeness',    # chosen_ripeness - mean_avail_ripeness   (higher = juicier pick)
    'rel_wnn',         # chosen_wnn - mean_avail_wnn             (higher = better neighborhood)
    'rel_late_agg',    # chosen_late_agg - mean_avail_late_agg  (higher = deeper/riskier pick)
]

# Boolean "best of available" flags
BOOL_COLS = [
    'chose_cheapest',       # picked the cheapest ship-cost target
    'chose_richest_prod',   # picked the highest-production target
    'chose_best_ripeness',  # picked the best ripeness target
    'chose_best_wnn',       # picked the best neighborhood target
]

# Target absolute features (normalized within dataset)
TGT_COLS = [
    'tgt_dist_frac',       # distance as fraction of max map distance
    'tgt_deepness',        # mean dist from target to own planets (higher = deeper)
    'overkill',            # (ships_sent - tgt_ships) / max(tgt_ships, 1)
    'fleet_size_ratio',    # ships_sent / own_ships_total
]

# Context features that describe the game state
CTX_COLS = [
    'phase',
    'own_ship_ratio',
    'own_planet_ratio',
    'n_better_alternatives',  # how many alternatives were strictly better by priority
    'tgt_priority_rank',      # rank of chosen target by our priority_approx (1=best)
]

# Reinforce-specific features (rel_* are all 0 for own-planet targets by design)
# Use absolute zone-value features of the reinforced planet instead
REINFORCE_COLS = [
    'tgt_prod',          # production of reinforced planet
    'tgt_ships',         # current garrison (how depleted?)
    'tgt_dist_frac',     # distance to reinforced planet from source
    'tgt_wnn',           # neighborhood quality
    'tgt_deepness',      # how deep in our territory (frontier vs rear)
    'tgt_ripeness',      # low = depleted/under attack
    'tgt_late_aggression',
    'fleet_size_ratio',  # how big a fleet sent
    'overkill',          # how much above min needed
    'phase',
    'own_ship_ratio',
    'own_planet_ratio',
    'under_threat_count',
    'src_nearest_enemy_dist',
]

# All features used for clustering (neutral + enemy)
CLUSTER_FEATURES = REL_COLS + BOOL_COLS + TGT_COLS + CTX_COLS

# Owner type mapping
# Encoding from 02c_extract_launches.py line 415:
#   owner == pidx      → 2 = own planet  (reinforce target)
#   owner == NEUTRAL   → 0 = neutral
#   else               → 1 = enemy
OWNER_TYPE = {0: 'neutral', 1: 'enemy', 2: 'reinforce'}

# Clipping percentiles for outlier-robustness before scaling
CLIP_PCT = 1.0   # clip at 1st / 99th percentile


# ── Helpers ───────────────────────────────────────────────────────────────────

def clip_outliers(df: pd.DataFrame, cols: list, pct: float = CLIP_PCT) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns and df[c].dtype.kind in 'fi':
            lo = df[c].quantile(pct / 100)
            hi = df[c].quantile(1 - pct / 100)
            df[c] = df[c].clip(lo, hi)
    return df


def build_feature_matrix(df: pd.DataFrame,
                          feature_list: list = None) -> np.ndarray:
    """Return (n, p) scaled feature matrix for clustering."""
    cols = feature_list if feature_list is not None else CLUSTER_FEATURES
    available = [c for c in cols if c in df.columns]
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    # Clip extreme values before scaling
    p1  = np.nanpercentile(X, 1,  axis=0)
    p99 = np.nanpercentile(X, 99, axis=0)
    X   = np.clip(X, p1, p99)
    scaler = RobustScaler()
    return scaler.fit_transform(X), available, scaler


def cluster_split(df: pd.DataFrame, k: int, seed: int = 42,
                  feature_list: list = None) -> np.ndarray:
    """KMeans with multiple inits; returns cluster labels array."""
    X, _, _ = build_feature_matrix(df, feature_list)
    km = KMeans(n_clusters=k, n_init=15, random_state=seed, max_iter=500)
    return km.fit_predict(X)


def silhouette(df: pd.DataFrame, labels: np.ndarray,
               sample: int = 20_000, feature_list: list = None) -> float:
    X, _, _ = build_feature_matrix(df, feature_list)
    if len(df) > sample:
        idx = np.random.default_rng(0).choice(len(df), sample, replace=False)
        X, labels = X[idx], labels[idx]
    try:
        return silhouette_score(X, labels, sample_size=min(5000, len(df)))
    except Exception:
        return float('nan')


def scan_k(df: pd.DataFrame, k_range: range,
           feature_list: list = None) -> dict:
    """Scan K values, return {k: silhouette_score}."""
    scores = {}
    for k in k_range:
        if k >= len(df):
            break
        labels = cluster_split(df, k, feature_list=feature_list)
        scores[k] = silhouette(df, labels, feature_list=feature_list)
        print(f"    k={k}  sil={scores[k]:.4f}")
    return scores


def profile_cluster(sub: pd.DataFrame, all_df: pd.DataFrame) -> dict:
    """Build a human-readable profile dict for a single cluster."""
    n = len(sub)
    frac = n / len(all_df)

    profile = {
        'n': int(n),
        'frac': round(float(frac), 4),
    }

    # Relative choice means
    for c in REL_COLS:
        if c in sub.columns:
            profile[c] = round(float(sub[c].mean()), 4)

    # Boolean rates
    for c in BOOL_COLS:
        if c in sub.columns:
            profile[c] = round(float(sub[c].mean()), 4)

    # Target features
    for c in TGT_COLS + CTX_COLS:
        if c in sub.columns:
            profile[c] = round(float(sub[c].mean()), 4)

    # Zone label distribution
    if 'tgt_zone_approx' in sub.columns:
        vc = sub['tgt_zone_approx'].value_counts(normalize=True)
        profile['tgt_zone_dist'] = {k: round(float(v), 4) for k, v in vc.items()}

    # Source zone distribution
    if 'src_zone_approx' in sub.columns:
        vc = sub['src_zone_approx'].value_counts(normalize=True)
        profile['src_zone_dist'] = {k: round(float(v), 4) for k, v in vc.items()}

    # Label
    profile['label'] = _auto_label(profile)

    return profile


def _auto_label(p: dict) -> str:
    """Generate a short human label from profile statistics."""
    parts = []

    # Distance tendency
    rel_dist = p.get('rel_dist', 0)
    if rel_dist < -0.05:
        parts.append('close')
    elif rel_dist > 0.05:
        parts.append('far')

    # Production preference
    rel_prod = p.get('rel_prod', 0)
    if rel_prod > 0.15:
        parts.append('high-prod')
    elif rel_prod < -0.1:
        parts.append('low-prod')

    # Ship cost preference
    rel_ships = p.get('rel_ships', 0)
    if rel_ships < -5:
        parts.append('low-cost')
    elif rel_ships > 5:
        parts.append('high-cost')

    # Ripeness
    rel_ripe = p.get('rel_ripeness', 0)
    if rel_ripe > 0.05:
        parts.append('ripe')
    elif rel_ripe < -0.05:
        parts.append('unripe')

    # Wnn (neighborhood quality)
    rel_wnn = p.get('rel_wnn', 0)
    if rel_wnn > 0.02:
        parts.append('good-neighborhood')

    # Late aggression (deepness × ripeness)
    rel_la = p.get('rel_late_agg', 0)
    if rel_la > 0.3:
        parts.append('deep')
    elif rel_la < -0.3:
        parts.append('shallow')

    # Boolean tendencies
    if p.get('chose_cheapest', 0) > 0.2:
        parts.append('cheapest')
    if p.get('chose_richest_prod', 0) > 0.3:
        parts.append('richest')
    if p.get('chose_best_wnn', 0) > 0.2:
        parts.append('best-zone')
    if p.get('chose_best_ripeness', 0) > 0.2:
        parts.append('best-ripe')

    # Priority rank
    prank = p.get('tgt_priority_rank', 5)
    if prank <= 2:
        parts.append('top-priority')
    elif prank > 8:
        parts.append('ignores-priority')

    if not parts:
        parts = ['balanced']

    return '+'.join(parts)


def format_report_section(type_name: str, profiles: dict, k: int, sil: float) -> str:
    lines = []
    lines.append(f"\n{'═'*72}")
    lines.append(f"  {type_name.upper()} LAUNCHES   k={k}   silhouette={sil:.4f}")
    lines.append(f"{'═'*72}")

    for cid, p in sorted(profiles.items()):
        lines.append(f"\n  Cluster {cid}  [{p['label']}]  n={p['n']:,}  ({100*p['frac']:.1f}%)")
        lines.append(f"  {'─'*65}")
        lines.append(f"  Relative choice:")
        lines.append(f"    rel_prod={p.get('rel_prod', 0):+.3f}  rel_ships={p.get('rel_ships', 0):+.2f}  rel_dist={p.get('rel_dist', 0):+.3f}")
        lines.append(f"    rel_ripeness={p.get('rel_ripeness', 0):+.4f}  rel_wnn={p.get('rel_wnn', 0):+.4f}  rel_late_agg={p.get('rel_late_agg', 0):+.4f}")
        lines.append(f"  Boolean rates:")
        lines.append(f"    chose_cheapest={p.get('chose_cheapest', 0):.3f}  chose_richest_prod={p.get('chose_richest_prod', 0):.3f}")
        lines.append(f"    chose_best_ripeness={p.get('chose_best_ripeness', 0):.3f}  chose_best_wnn={p.get('chose_best_wnn', 0):.3f}")
        lines.append(f"  Target:")
        lines.append(f"    dist_frac={p.get('tgt_dist_frac', 0):.3f}  deepness={p.get('tgt_deepness', 0):.3f}  overkill={p.get('overkill', 0):.2f}")
        lines.append(f"    priority_rank={p.get('tgt_priority_rank', 0):.1f}  n_better_alts={p.get('n_better_alternatives', 0):.1f}")
        lines.append(f"  Context:")
        lines.append(f"    phase={p.get('phase', 0):.2f}  own_ship_ratio={p.get('own_ship_ratio', 0):.3f}  own_planet_ratio={p.get('own_planet_ratio', 0):.3f}")

        if 'tgt_zone_dist' in p:
            top_zones = sorted(p['tgt_zone_dist'].items(), key=lambda x: -x[1])[:4]
            zstr = '  '.join(f"{z}={100*v:.0f}%" for z, v in top_zones)
            lines.append(f"  Target zones: {zstr}")

    return '\n'.join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(LAUNCHES_CSV))
    parser.add_argument('--k-neutral',   type=int, default=None, help='K for neutral clusters (default: auto)')
    parser.add_argument('--k-enemy',     type=int, default=None, help='K for enemy clusters')
    parser.add_argument('--k-reinforce', type=int, default=None, help='K for reinforce clusters')
    parser.add_argument('--k-scan',      action='store_true', help='Scan K 2..8 and pick best silhouette')
    parser.add_argument('--k-default',   type=int, default=5,   help='Default K if not specified')
    parser.add_argument('--min-reward',  type=float, default=None,
                        help='Filter: only top-N%% reward episodes (e.g. 50 = top half)')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'═'*72}")
    print("08_cluster_launches.py  —  per-launch behavioral clustering")
    print(f"{'═'*72}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.input} …")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows  {len(df.columns)} cols")

    # Optional: filter to top-reward episodes only
    if args.min_reward is not None:
        thresh = df.groupby('episode_id')['reward'].mean().quantile(args.min_reward / 100)
        ep_ok = df.groupby('episode_id')['reward'].mean() >= thresh
        ok_eps = ep_ok[ep_ok].index
        df = df[df['episode_id'].isin(ok_eps)]
        print(f"  After top-{100-args.min_reward:.0f}% reward filter: {len(df):,} rows")

    # Clip outliers in relative features (huge ship counts skew things)
    df = clip_outliers(df, REL_COLS + TGT_COLS)

    # ── Split by target type ──────────────────────────────────────────────────
    # Encoding: 0=neutral, 1=enemy, 2=own/reinforce  (02c line 415)
    splits = {
        'neutral':   df[df['tgt_owner_type'] == 0].copy(),
        'enemy':     df[df['tgt_owner_type'] == 1].copy(),
        'reinforce': df[df['tgt_owner_type'] == 2].copy(),
    }
    for t, sub in splits.items():
        print(f"  {t}: {len(sub):,} launches")

    # ── K selection ───────────────────────────────────────────────────────────
    k_map = {
        'neutral':   args.k_neutral   or args.k_default,
        'enemy':     args.k_enemy     or args.k_default,
        'reinforce': args.k_reinforce or args.k_default,
    }

    feat_map = {'neutral': CLUSTER_FEATURES, 'enemy': CLUSTER_FEATURES,
                'reinforce': REINFORCE_COLS}

    if args.k_scan:
        print("\nScanning K values …")
        for t, sub in splits.items():
            print(f"\n  [{t}]")
            scores = scan_k(sub, range(2, 9), feature_list=feat_map[t])
            best_k = max(scores, key=scores.get)
            k_map[t] = best_k
            print(f"  → best k={best_k}  sil={scores[best_k]:.4f}")

    print(f"\nK selection: {k_map}")

    # ── Clustering ────────────────────────────────────────────────────────────
    all_profiles   = {}
    cluster_labels = {}
    silhouettes    = {}

    df['launch_cluster_neutral']   = -1
    df['launch_cluster_enemy']     = -1
    df['launch_cluster_reinforce'] = -1

    label_col_map = {'neutral': 'launch_cluster_neutral',
                     'enemy':   'launch_cluster_enemy',
                     'reinforce': 'launch_cluster_reinforce'}

    for ttype, sub in splits.items():
        k    = k_map[ttype]
        feat = feat_map[ttype]
        print(f"\nClustering {ttype} (k={k}, n={len(sub):,}) …")

        if len(sub) < k:
            print(f"  ⚠ Too few rows ({len(sub)}) for k={k}, skipping")
            continue

        labels = cluster_split(sub, k, seed=args.seed, feature_list=feat)
        sil    = silhouette(sub, labels, feature_list=feat)
        print(f"  silhouette = {sil:.4f}")

        # Write labels back into main df
        df.loc[sub.index, label_col_map[ttype]] = labels
        cluster_labels[ttype] = labels
        silhouettes[ttype]    = sil

        # Profile each cluster
        profiles = {}
        for cid in range(k):
            mask  = labels == cid
            csub  = sub.iloc[mask] if hasattr(sub, 'iloc') else sub[mask]
            csub  = sub[labels == cid]
            prof  = profile_cluster(csub, sub)
            profiles[cid] = prof
            print(f"    Cluster {cid}  [{prof['label']}]  n={prof['n']:,}  ({100*prof['frac']:.1f}%)")

        all_profiles[ttype] = {'k': k, 'silhouette': float(sil), 'clusters': profiles}

    # ── Save results ──────────────────────────────────────────────────────────

    # 1. launches + cluster labels
    out_csv = RES_DIR / "launch_clusters.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved: {out_csv}  ({len(df):,} rows)")

    # 2. Archetype profiles JSON
    out_json = RES_DIR / "launch_archetypes.json"
    # Convert int keys to str for JSON serialization
    json_profiles = {}
    for ttype, info in all_profiles.items():
        json_profiles[ttype] = {
            'k': info['k'],
            'silhouette': info['silhouette'],
            'clusters': {str(k): v for k, v in info['clusters'].items()}
        }
    out_json.write_text(json.dumps(json_profiles, indent=2))
    print(f"✓ Saved: {out_json}")

    # 3. Human-readable report
    report_lines = ["=" * 72,
                    "LAUNCH BEHAVIORAL ARCHETYPES  —  08_cluster_launches.py",
                    "=" * 72,
                    "",
                    "Clustering on relative-choice + zone-value features.",
                    "Key: rel_X = chosen_X - mean(available_X)  (signed, clipped at 1/99 pct)",
                    ""]

    for ttype in ('neutral', 'enemy', 'reinforce'):
        if ttype not in all_profiles:
            continue
        info = all_profiles[ttype]
        report_lines.append(format_report_section(
            ttype, info['clusters'], info['k'], info['silhouette']
        ))

    # Summary: what makes each archetype different from global mean
    report_lines += ["", "=" * 72, "GLOBAL BASELINE (all launches):", "=" * 72]
    for c in REL_COLS:
        if c in df.columns:
            report_lines.append(f"  {c}: {df[c].mean():+.4f}")

    report_txt = '\n'.join(report_lines)
    out_report = RES_DIR / "launch_cluster_report.txt"
    out_report.write_text(report_txt)
    print(f"✓ Saved: {out_report}")

    # Print report to stdout
    print("\n" + report_txt)

    # ── Quick discriminative features per cluster ─────────────────────────────
    print("\n" + "=" * 72)
    print("TOP DISCRIMINATIVE FEATURES PER CLUSTER (vs. type mean)")
    print("=" * 72)

    disc_cols = REL_COLS + BOOL_COLS + ['tgt_dist_frac', 'tgt_deepness', 'overkill', 'tgt_priority_rank']

    for ttype, sub in splits.items():
        if ttype not in all_profiles:
            continue
        info  = all_profiles[ttype]
        lc    = label_col_map[ttype]
        print(f"\n  [{ttype.upper()}]  (k={info['k']})")

        type_means = sub[[c for c in disc_cols if c in sub.columns]].mean()

        type_code = {v: k for k, v in OWNER_TYPE.items()}[ttype]
        for cid in range(info['k']):
            csub = df[(df[lc] == cid) & (df['tgt_owner_type'] == type_code)]
            if len(csub) == 0:
                continue
            cmeans = csub[[c for c in disc_cols if c in csub.columns]].mean()
            delta  = (cmeans - type_means).abs().sort_values(ascending=False)
            top5   = delta.head(5)
            prof   = info['clusters'][cid]
            print(f"    Cluster {cid} [{prof['label']}]  n={prof['n']:,}")
            for feat in top5.index:
                print(f"      {feat:<30} {type_means[feat]:+.4f} → {cmeans[feat]:+.4f}  (Δ={cmeans[feat]-type_means[feat]:+.4f})")

    print(f"\n{'═'*72}")
    print("Done. Next step:")
    print("  python3 opponent_learning/scripts/09_archetype_presets.py  (coming soon)")
    print(f"{'═'*72}\n")


if __name__ == '__main__':
    main()
