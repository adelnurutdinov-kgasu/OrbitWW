"""
analytics_bayes.py -- Post-game analytics for the Bayesian opponent predictor.

Usage (two modes):

  # Mode 1: read from agent debug log (produced when ORBIT_AGENT_LOG is set)
  python analytics_bayes.py agent_debug.log

  # Mode 2: pass model object directly from notebook
  from analytics_bayes import plot_from_model
  plot_from_model(our_mod._bayes_model)

Output:
  - 4-panel figure: surprise, entropy, preset probabilities (stackplot), match accuracy
  - Summary dict printed to stdout
"""

import sys
import json
import math
import os
from typing import List, Dict


# ══════════════════════════════════════════════════════════════════════════
# 1. Parse history from agent log
# ══════════════════════════════════════════════════════════════════════════

def parse_log(path: str) -> List[dict]:
    """Extract [BAYES/H] JSON lines from agent_debug.log."""
    records = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('[BAYES/H] '):
                payload = line[len('[BAYES/H] '):]
                try:
                    records.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return records


# ══════════════════════════════════════════════════════════════════════════
# 2. Build arrays from history
# ══════════════════════════════════════════════════════════════════════════

def _extract_arrays(history: List[dict]):
    steps        = [r['step']               for r in history]
    surprises    = [r['surprise']           for r in history]
    entropies    = [r['entropy']            for r in history]
    top_probs    = [r['top_prob']           for r in history]
    n_obs        = [r['n_opponent_actions'] for r in history]
    matches      = [r['match_count']        for r in history]

    # Accuracy: fraction of observed actions that matched top preset
    accuracy = []
    for r in history:
        n = r['n_opponent_actions']
        accuracy.append(r['match_count'] / max(1, n) if n > 0 else float('nan'))

    # Preset probability series
    preset_names = sorted(history[0]['priors'].keys()) if history else []
    preset_series = {k: [r['priors'].get(k, 0.0) for r in history]
                     for k in preset_names}

    return {
        'steps':         steps,
        'surprises':     surprises,
        'entropies':     entropies,
        'top_probs':     top_probs,
        'n_obs':         n_obs,
        'matches':       matches,
        'accuracy':      accuracy,
        'preset_names':  preset_names,
        'preset_series': preset_series,
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. Summary statistics
# ══════════════════════════════════════════════════════════════════════════

def compute_summary(history: List[dict]) -> dict:
    if not history:
        return {}

    arr = _extract_arrays(history)

    active = [r for r in history if r['n_opponent_actions'] > 0]

    avg_surprise_turn   = sum(arr['surprises']) / max(1, len(arr['surprises']))
    avg_surprise_action = (
        sum(r['surprise'] for r in active) / max(1, len(active))
        if active else float('nan')
    )
    avg_entropy = sum(arr['entropies']) / max(1, len(arr['entropies']))

    # Accuracy: turns where ALL observed actions matched top preset
    perfect = sum(
        1 for r in active
        if r['match_count'] >= r['n_opponent_actions'] > 0
    )
    accuracy_perfect = perfect / max(1, len(active))

    # Dominant preset
    from collections import Counter
    top_hist = Counter(r['top_preset'] for r in history)
    dominant = top_hist.most_common(1)[0] if top_hist else ('?', 0)

    return {
        'turns_total':           len(history),
        'turns_with_opp_action': len(active),
        'avg_surprise_per_turn':   round(avg_surprise_turn,   4),
        'avg_surprise_per_action': round(avg_surprise_action, 4),
        'avg_entropy':             round(avg_entropy,          4),
        'accuracy_perfect':        round(accuracy_perfect,     4),
        'dominant_preset':         dominant[0],
        'dominant_preset_turns':   dominant[1],
        'final_priors':            history[-1]['priors'] if history else {},
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. Plotting
# ══════════════════════════════════════════════════════════════════════════

def plot_history(history: List[dict], title: str = '', save_path: str = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print('matplotlib not available — skipping plot')
        return

    if not history:
        print('No analytics history to plot.')
        return

    arr = _extract_arrays(history)
    steps = arr['steps']

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(f'Bayesian opponent predictor analytics{" — " + title if title else ""}',
                 fontsize=13)

    # Panel 1: Surprise
    ax = axes[0]
    ax.plot(steps, arr['surprises'], color='#E84855', lw=1.5)
    ax.fill_between(steps, arr['surprises'], alpha=0.15, color='#E84855')
    ax.set_ylabel('Surprise  −log P(obs)', fontsize=9)
    ax.set_title('Surprise per turn (lower = model predicted correctly)', fontsize=10)
    ax.grid(alpha=0.25)

    # Panel 2: Entropy
    ax = axes[1]
    ax.plot(steps, arr['entropies'], color='#2176AE', lw=1.5)
    ax.fill_between(steps, arr['entropies'], alpha=0.15, color='#2176AE')
    # Theoretical max entropy for reference
    n_presets = len(arr['preset_names'])
    max_ent = math.log(n_presets) if n_presets > 0 else 1.0
    ax.axhline(max_ent, ls='--', lw=0.9, color='#7B7B7B', label=f'max H={max_ent:.2f}')
    ax.set_ylabel('Entropy H(presets)', fontsize=9)
    ax.set_title('Entropy (falling = model converging on opponent style)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3: Preset probabilities (stackplot)
    ax = axes[2]
    colors = ['#2176AE', '#27AE60', '#8E44AD', '#E67E22', '#E84855', '#6B6B6B']
    ys = [arr['preset_series'][k] for k in arr['preset_names']]
    ax.stackplot(steps, ys,
                 labels=arr['preset_names'],
                 colors=colors[:len(arr['preset_names'])],
                 alpha=0.82)
    ax.set_ylabel('P(preset)', fontsize=9)
    ax.set_title('Preset probability distribution over time', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', ncol=3)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.18)

    # Panel 4: Match accuracy
    ax = axes[3]
    acc_clean = [a if not (isinstance(a, float) and math.isnan(a)) else 0.0
                 for a in arr['accuracy']]
    ax.bar(steps, acc_clean, width=0.8, color='#27AE60', alpha=0.75)
    ax.axhline(1.0, ls='--', lw=0.9, color='#7B7B7B', label='perfect match')
    ax.set_ylabel('Fraction matched', fontsize=9)
    ax.set_title('Top-preset action match accuracy (per turn with opponent moves)', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Game step', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f'Saved: {save_path}')
    else:
        plt.show()

    return fig


# ══════════════════════════════════════════════════════════════════════════
# 5. Notebook-friendly entry point
# ══════════════════════════════════════════════════════════════════════════

def plot_from_model(model, title: str = '', save_path: str = None):
    """Call from notebook: plot_from_model(our_mod._bayes_model)"""
    history = model.dump_analytics()
    if not history:
        print('No analytics history — run with ANALYTICS_MODE=True '
              '(enable ORBIT_AGENT_LOG before the game).')
        return
    summary = compute_summary(history)
    print('=== Bayesian predictor summary ===')
    for k, v in summary.items():
        if k != 'final_priors':
            print(f'  {k:<30} {v}')
    print('  final_priors:')
    for preset, prob in sorted(summary.get('final_priors', {}).items(),
                               key=lambda x: -x[1]):
        print(f'    {preset:<20} {prob:.3f}')
    print()
    plot_history(history, title=title, save_path=save_path)


# ══════════════════════════════════════════════════════════════════════════
# 6. CLI entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python analytics_bayes.py <agent_debug.log> [save_path.png]')
        sys.exit(1)

    log_path  = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    history = parse_log(log_path)
    if not history:
        print(f'No [BAYES/H] records found in {log_path}')
        sys.exit(0)

    print(f'Loaded {len(history)} analytics records from {log_path}')
    summary = compute_summary(history)
    print('\n=== Summary ===')
    for k, v in summary.items():
        if k != 'final_priors':
            print(f'  {k:<30} {v}')
    print('  final_priors:')
    for preset, prob in sorted(summary.get('final_priors', {}).items(),
                               key=lambda x: -x[1]):
        print(f'    {preset:<20} {prob:.3f}')

    out = save_path or log_path.replace('.log', '_bayes_analytics.png')
    plot_history(history, title=os.path.basename(log_path), save_path=out)
