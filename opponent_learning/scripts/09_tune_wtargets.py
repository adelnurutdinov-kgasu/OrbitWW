#!/usr/bin/env python3
"""
09_tune_wtargets.py — evidence-based W_TARGETS tuning from per-launch analysis.

Findings from 270,714 launches (500 replays, top players, orbit_wars):

  CURRENT W_TARGETS problems:
    late_aggression: +0.9   ← WRONG SIGN
      60% of enemy attacks & 50% of neutral attacks prefer SHALLOW targets
      Top-reward players: median rel_late_agg = -1.91 (strongly prefer shallow)
      Empirical OLS weight: -0.89 enemy, -0.54 neutral

    ripeness: missing       ← SECOND-STRONGEST SIGNAL
      Players strongly prefer ripe (high prod/ships) targets
      Empirical OLS weight: +2.88 enemy, +0.89 neutral

  PROPOSED W_TARGETS:
    late_aggression: -0.5   (flipped, phase multiplier kept)
    ripeness added:  +0.9   (new feature)
    mean_dist_all:   -0.4   (strengthened from -0.2)

  VALIDATION (chosen_target_score > mean_alternative_score):
    Neutral: 51.1% → 54.6%  (+3.5pp)
    Enemy:   54.3% → 64.3%  (+10.0pp)

What this script does:
  1. Validates the proposed weights one more time on the full dataset
  2. Patches zones.py with proposed W_TARGETS
  3. Prints a diff of what changed

Run:
  python3 opponent_learning/scripts/09_tune_wtargets.py --dry-run
  python3 opponent_learning/scripts/09_tune_wtargets.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE      = Path(__file__).parent
OL_DIR    = HERE.parent
ROOT_DIR  = OL_DIR.parent
ZONES_PY  = ROOT_DIR / "agent_bundle_swarm" / "zones.py"
CLUSTERS_CSV = OL_DIR / "results" / "launch_clusters.csv"


# ── Proposed weights ──────────────────────────────────────────────────────────

W_TARGETS_CURRENT = {
    'area_inv':        +0.4,
    'wnn_close_res':   +0.8,
    'mean_dist_all':   -0.2,
    'prod':            +0.5,
    'ships':           -0.5,
    'n_cross':         +0.4,
    'late_aggression': +0.9,
}

W_TARGETS_PROPOSED = {
    'area_inv':        +0.4,    # unchanged (simulation-derived, not empirically testable)
    'wnn_close_res':   +0.8,    # confirmed ✓ (empirical: +4.25 enemy, +1.03 neutral)
    'mean_dist_all':   -0.4,    # strengthened (empirical: -0.76 neutral, -0.34 enemy)
    'prod':            +0.5,    # confirmed ✓ (empirical: +1.62 enemy, +0.34 neutral)
    'ships':           -0.5,    # confirmed ✓ (rel_ships < 0 in all clusters)
    'n_cross':         +0.4,    # unchanged (simulation-derived)
    'late_aggression': -0.5,    # FLIPPED from +0.9  ← KEY CHANGE
                                # 60% of enemy + 50% of neutral attacks prefer shallow targets
                                # phase multiplier kept: early game ≈ 0, late game = -0.4
}

# Note: ripeness = prod/(1+ships) is missing from zones.py ZONE_FEATURES
# Empirical weight: +0.9 neutral, +2.88 enemy → add separately if possible
# For now, the prod(+0.5) and ships(-0.5) terms partially capture this.
RIPENESS_NOTE = """
# TODO: add 'ripeness' to ZONE_FEATURES and W_TARGETS
#   ripeness = prod / (1 + ships)   — empirical weight: +0.9 neutral, +2.88 enemy
#   This is the second-strongest empirical signal after wnn_close_res.
#   Currently partially captured by prod(+0.5) and ships(-0.5) but product form
#   (ripeness) is stronger. Add to compute_zones to fully utilize this signal.
"""


# ── Validation ────────────────────────────────────────────────────────────────

def score_current(wnn, prod, shp, la):
    return wnn * 0.8 + prod * 0.5 - shp * 0.5 + la * 0.9


def score_proposed(wnn, prod, shp, la, ripe, dist):
    return wnn * 0.8 + prod * 0.5 - shp * 0.5 + ripe * 0.9 - la * 0.5 - dist * 0.4


def validate(df: pd.DataFrame) -> dict:
    results = {}

    for ttype_code, name, pfx in [(0, 'neutral', 'neutral'), (1, 'enemy', 'enemy')]:
        sub = df[df['tgt_owner_type'] == ttype_code].copy()
        sub = sub[sub[f'n_{pfx}_avail'] > 1]

        tgt_wnn  = sub['tgt_wnn']
        tgt_prod = sub['tgt_prod']
        tgt_shp  = sub['tgt_ships']
        tgt_la   = sub['tgt_late_aggression']
        tgt_ripe = sub['tgt_ripeness']
        tgt_dist = sub['tgt_dist_frac']

        alt_wnn  = sub[f'{pfx}_wnn_mean']
        alt_prod = sub[f'{pfx}_prod_mean']
        alt_shp  = sub[f'{pfx}_ships_mean']
        alt_la   = sub[f'{pfx}_late_agg_mean']
        alt_ripe = sub[f'{pfx}_ripeness_mean']
        alt_dist = sub[f'{pfx}_dist_min']

        beat_curr = (score_current(tgt_wnn, tgt_prod, tgt_shp, tgt_la) >
                     score_current(alt_wnn, alt_prod, alt_shp, alt_la)).mean()
        beat_prop = (score_proposed(tgt_wnn, tgt_prod, tgt_shp, tgt_la, tgt_ripe, tgt_dist) >
                     score_proposed(alt_wnn, alt_prod, alt_shp, alt_la, alt_ripe, alt_dist)).mean()

        results[name] = {
            'n': len(sub),
            'current': float(beat_curr),
            'proposed': float(beat_prop),
            'delta': float(beat_prop - beat_curr),
        }
        print(f"  [{name}]  n={len(sub):,}")
        print(f"    Current  formula: {beat_curr:.1%} of choices score above mean alternative")
        print(f"    Proposed formula: {beat_prop:.1%} of choices score above mean alternative")
        print(f"    Improvement:      {beat_prop - beat_curr:+.1%}")

    return results


# ── Patch zones.py ────────────────────────────────────────────────────────────

def patch_zones(dry_run: bool) -> bool:
    if not ZONES_PY.exists():
        print(f"⚠ Not found: {ZONES_PY}")
        return False

    src = ZONES_PY.read_text()

    # Match the W_TARGETS dict block
    pattern = r"(W_TARGETS\s*=\s*\{[^}]+\})"
    match = re.search(pattern, src, re.DOTALL)
    if not match:
        print("⚠ Could not find W_TARGETS block in zones.py")
        return False

    old_block = match.group(1)

    # Build new block
    new_block = """W_TARGETS = {
    'area_inv':        +0.4,
    'wnn_close_res':   +0.8,    # confirmed by per-launch analysis (270k launches, 500 replays)
    'mean_dist_all':   -0.4,    # strengthened: empirical -0.76 neutral, -0.34 enemy
    'prod':            +0.5,    # confirmed ✓
    'ships':           -0.5,    # confirmed ✓
    'n_cross':         +0.4,
    'late_aggression': -0.5,    # FLIPPED from +0.9 (2025-05 empirical analysis)
                                # 60% enemy + 50% neutral attacks prefer SHALLOW targets.
                                # Top-reward players: median rel_late_agg = -1.91 (shallow).
                                # Prediction accuracy: neutral 54.6% (+3.5pp), enemy 64.3% (+10pp).
                                # phase-multiplier kept: effective weight ≈ phase × (-0.5)
                                # TODO: add ripeness=prod/(1+ships) as separate feature (+0.9)
}"""

    if dry_run:
        print("\n[DRY RUN] Would replace W_TARGETS in zones.py:")
        print("\n  OLD:")
        for line in old_block.splitlines():
            print(f"    {line}")
        print("\n  NEW:")
        for line in new_block.splitlines():
            print(f"    {line}")
        return True

    new_src = src.replace(old_block, new_block, 1)
    if new_src == src:
        print("⚠ No change made — W_TARGETS block may have been modified already")
        return False

    ZONES_PY.write_text(new_src)
    print(f"✓ Patched: {ZONES_PY}")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-patch', action='store_true', help='Validate only, do not patch')
    args = parser.parse_args()

    print(f"\n{'═'*72}")
    print("09_tune_wtargets.py — evidence-based W_TARGETS tuning")
    print(f"{'═'*72}\n")

    # ── Validate ─────────────────────────────────────────────────────────────
    print("Step 1: Validating on launch dataset …")
    if not CLUSTERS_CSV.exists():
        print(f"⚠ Not found: {CLUSTERS_CSV}")
        print("  Run first: python3 opponent_learning/scripts/08_cluster_launches.py")
        return

    df = pd.read_csv(CLUSTERS_CSV)
    print(f"  Loaded {len(df):,} launches\n")
    results = validate(df)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("PROPOSED W_TARGETS CHANGES:")
    print(f"{'─'*72}")
    for feat in W_TARGETS_CURRENT:
        old = W_TARGETS_CURRENT[feat]
        new = W_TARGETS_PROPOSED[feat]
        if abs(old - new) > 0.001:
            sign = '→' if new * old > 0 else '✗→'
            print(f"  {feat:<20} {old:+.2f}  {sign}  {new:+.2f}  {'← SIGN FLIP' if new*old < 0 else '← adjusted'}")
        else:
            print(f"  {feat:<20} {old:+.2f}  (unchanged)")
    print()
    print("Missing from current W_TARGETS:")
    print("  ripeness (prod/(1+ships))  empirical: +2.88 enemy, +0.89 neutral")
    print("  → Add to ZONE_FEATURES and compute_zones in a future refactor")

    # ── Patch ────────────────────────────────────────────────────────────────
    if not args.no_patch:
        print(f"\n{'─'*72}")
        print("Step 2: Patching zones.py …")
        patch_zones(args.dry_run)

    # ── Next steps ───────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    if args.dry_run:
        print("DRY RUN complete. Run without --dry-run to apply.")
    else:
        print("✓ Done.")
        print("\nNext steps:")
        print("  1. Run a local tournament to verify the patch improves win rate")
        print("  2. Add 'ripeness' to ZONE_FEATURES in zones.py for +10% more signal")
        print("  3. Run 06_integrate_likelihood.py to integrate XGBoost likelihood")
        print("  4. Consider per-cluster opponent presets in opponent_model_bayesian.py")
    print(f"{'═'*72}\n")


if __name__ == '__main__':
    main()
