"""
opponent_model_bayesian.py -- Bayesian opponent behaviour model with analytics.

ANALYTICS_MODE (module-level bool, default False):
  When False  -- zero overhead: no history stored, no log/entropy computed.
  When True   -- full metrics collected in self.analytics_history each turn.
  Set from agent.py: import opponent_model_bayesian; opponent_model_bayesian.ANALYTICS_MODE = True

Public interface:
  ANALYTICS_MODE           : bool  (set before creating model instance)
  class OpponentModelBayesian
    .update(observed_actions, state, opp_id, step=0) -> dict|None
    .get_expected_actions(state, opp_id, threshold=0.05) -> List[dict]
    .top_preset() -> str
    .summary()   -> str
    .priors      : dict {preset_name: probability}
    ._history    : List[str]          -- top preset each turn (always)
    .analytics_history : List[dict]   -- full metrics (only if ANALYTICS_MODE)
"""

import math
from typing import List, Dict, Optional

from opponent_presets import PRESETS, get_preset_actions

# ── Module-level analytics flag ────────────────────────────────────────────
# Set to True from agent.py when _dbg.enabled() is True.
# All analytics code is guarded by `if ANALYTICS_MODE:` → zero cost in battle.
ANALYTICS_MODE: bool = False

# ── Parameters ─────────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 1.5
MIN_PROB            = 0.02   # probability floor per preset
MAX_VIRTUAL_FLEETS  = 3      # cap on virtual fleets per turn


class OpponentModelBayesian:
    """Bayesian distribution over opponent strategy presets.

    Fast path (ANALYTICS_MODE=False):
        Likelihoods computed, priors updated, top preset appended to _history.
        No log/entropy/history dict created.

    Full path (ANALYTICS_MODE=True):
        Additionally computes surprise, entropy, match_count, stores full
        metrics dict in analytics_history for post-game analysis.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        n = len(PRESETS)
        self.priors: Dict[str, float] = {k: 1.0 / n for k in PRESETS}
        self.temperature = temperature
        self._history: List[str] = []          # top preset name each turn (always)
        self.analytics_history: List[dict] = []  # full metrics (ANALYTICS_MODE only)

    # ── Bayesian update ────────────────────────────────────────────────────

    def update(self, observed_actions: List[dict],
               state, opp_id: int,
               step: int = 0) -> Optional[dict]:
        """Update priors given observed opponent actions.

        Args:
            observed_actions: list of {from_id, target_id, ships, action_type}
            state:            GameState when opponent acted (prev turn raw state)
            opp_id:           opponent player id
            step:             current game step (for analytics)

        Returns:
            metrics dict if ANALYTICS_MODE else None.
        """
        # ── No observed actions: opponent was idle ──────────────────────────
        if not observed_actions:
            top = self.top_preset()
            self._history.append(top)
            if ANALYTICS_MODE:
                entropy = -sum(p * math.log(max(p, 1e-12))
                               for p in self.priors.values())
                rec = {
                    'step': step,
                    'surprise': 0.0,        # idle = not surprising
                    'entropy': entropy,
                    'top_preset': top,
                    'top_prob': self.priors[top],
                    'n_opponent_actions': 0,
                    'match_count': 0,
                    'priors': dict(self.priors),
                }
                self.analytics_history.append(rec)
                return rec
            return None

        # ── Compute likelihoods ────────────────────────────────────────────
        n_obs      = max(1, len(observed_actions))
        obs_pairs  = {(a['from_id'], a['target_id']) for a in observed_actions}

        likelihoods: Dict[str, float] = {}
        pred_cache:  Dict[str, list]  = {}

        for preset_name in self.priors:
            predicted = get_preset_actions(preset_name, state, opp_id)
            pred_cache[preset_name] = predicted

            if not predicted:
                likelihoods[preset_name] = 1.0   # neutral: no prediction → no update
            else:
                pred_pairs = {(p['from_id'], p['target_id']) for p in predicted}
                pred_types = {p['action_type'] for p in predicted}

                matched_exact = sum(
                    1 for a in observed_actions
                    if (a['from_id'], a['target_id']) in pred_pairs
                )
                matched_type = sum(
                    1 for a in observed_actions
                    if a['action_type'] in pred_types
                )
                # Exact match weighs 2x type match
                score = (2 * matched_exact + matched_type) / (3 * n_obs)
                likelihoods[preset_name] = math.exp(self.temperature * score)

        # ── Analytics (pre-update) ─────────────────────────────────────────
        if ANALYTICS_MODE:
            # P(obs) = marginal likelihood (evidence term in Bayes rule)
            p_obs    = sum(self.priors[k] * likelihoods[k] for k in self.priors)
            surprise = -math.log(max(p_obs, 1e-12))

        # ── Bayesian update ────────────────────────────────────────────────
        new_priors = {k: self.priors[k] * likelihoods[k] for k in self.priors}
        total = sum(new_priors.values())

        if total < 1e-12:
            n = len(PRESETS)
            self.priors = {k: 1.0 / n for k in PRESETS}
        else:
            # Apply floor and re-normalise
            floored = {k: max(MIN_PROB, v / total) for k, v in new_priors.items()}
            total2  = sum(floored.values())
            self.priors = {k: v / total2 for k, v in floored.items()}

        top = self.top_preset()
        self._history.append(top)

        # ── Analytics (post-update) ────────────────────────────────────────
        if ANALYTICS_MODE:
            entropy = -sum(p * math.log(max(p, 1e-12))
                           for p in self.priors.values())

            top_pairs = {(p['from_id'], p['target_id'])
                         for p in pred_cache.get(top, [])}
            match_count = sum(
                1 for a in observed_actions
                if (a['from_id'], a['target_id']) in top_pairs
            )

            rec = {
                'step':               step,
                'surprise':           surprise,
                'entropy':            entropy,
                'top_preset':         top,
                'top_prob':           self.priors[top],
                'n_opponent_actions': len(observed_actions),
                'match_count':        match_count,
                'priors':             dict(self.priors),
            }
            self.analytics_history.append(rec)
            return rec

        return None

    # ── Query ──────────────────────────────────────────────────────────────

    def get_expected_actions(self, state, opp_id: int,
                              threshold: float = 0.05) -> List[dict]:
        """Return expected opponent actions from presets with prob >= threshold.

        Deduplicated by (from_id, target_id); capped at MAX_VIRTUAL_FLEETS.
        """
        candidates: List[dict] = []
        for preset_name, prob in sorted(self.priors.items(),
                                        key=lambda kv: -kv[1]):
            if prob < threshold:
                continue
            for a in get_preset_actions(preset_name, state, opp_id):
                candidates.append({**a, '_prob': prob, '_preset': preset_name})

        # Keep highest-prob per (from_id, target_id)
        seen: Dict[tuple, dict] = {}
        for a in candidates:
            key = (a['from_id'], a['target_id'])
            if key not in seen or a['_prob'] > seen[key]['_prob']:
                seen[key] = a

        unique = sorted(seen.values(), key=lambda a: (-a['_prob'], -a['ships']))
        return unique[:MAX_VIRTUAL_FLEETS]

    # ── Info ───────────────────────────────────────────────────────────────

    def top_preset(self) -> str:
        return max(self.priors, key=lambda k: self.priors[k])

    def summary(self) -> str:
        parts = [f'{k}={v:.2f}' for k, v in
                 sorted(self.priors.items(), key=lambda kv: -kv[1])]
        return '  '.join(parts)

    def dump_analytics(self) -> List[dict]:
        """Return analytics_history (empty list if ANALYTICS_MODE was False)."""
        return list(self.analytics_history)


__all__ = [
    'ANALYTICS_MODE',
    'OpponentModelBayesian',
    'MAX_VIRTUAL_FLEETS',
    'DEFAULT_TEMPERATURE',
]
