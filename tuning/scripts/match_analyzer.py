#!/usr/bin/env python3
"""
match_analyzer.py — собирает богатые данные с N матчей для кластеризации сидов.

ЧТО СОБИРАЕТ:
  1. Map-features (характеристики карты): omega, n_orbital/static, prod_total/var,
     min/mean dist home→neutral, mean inter-planet, sun_blockage, и пр.
  2. Per-turn time series: ships, prod, planet_count, fleet_count для обоих
     игроков на каждом ходу.
  3. Derived metrics из time series: max_lead_turn, first_lead_turn,
     dominance_area, avg_ships_diff по квартилям, и пр.
  4. Финальный исход: win/loss/draw, reward, final ships/planets diff.

ВЫХОД:
  • analysis_meta_<ts>.csv      — одна строка на seed: map-features + derived + outcome.
  • analysis_ts_<ts>.parquet    — long format: seed × turn × per-turn-метрики.
  • analysis_summary_<ts>.txt   — текстовая сводка для быстрого просмотра.

КЛАСТЕРИЗАЦИЯ:
  В конце показывается базовый k-means на map-features. Полноценную
  кластеризацию (с UMAP/PCA, train-test split, поиск корреляций) делаешь
  потом в notebook на собранных CSV/parquet.

ЗАПУСК:
  python3 match_analyzer.py
  (при необходимости поменяй N_SEEDS / N_WORKERS / MATCH_TIMEOUT в КОНФИГ-блоке)
"""

import os
import sys
import importlib.util
import math
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ══════════════════════════════════════════════════════════════════════════

N_SEEDS     = 600
SEED_OFFSET = 0
N_WORKERS   = max(1, os.cpu_count() // 2)

# Кластеризация: сколько кластеров, какие фичи использовать.
N_CLUSTERS  = 5
CLUSTER_FEATURES = [
    'omega', 'n_orbital', 'n_static', 'n_neutrals',
    'prod_total', 'prod_var',
    'min_dist_home_neutral', 'mean_dist_home_neutral',
    'mean_inter_dist', 'sun_blockage_pct',
]

# ══════════════════════════════════════════════════════════════════════════

# Скрипт лежит в <project>/tuning/scripts/. См. комментарий в zones_grid_search.py.
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUR_PATH    = os.path.join(PROJECT_ROOT, "agent_bundle", "agent.py")
SUB_PATH    = os.path.join(PROJECT_ROOT, "submission.py")
TS          = datetime.now().strftime('%Y%m%d_%H%M%S')
META_CSV    = os.path.join(RESULTS_DIR, f"analysis_meta_{TS}.csv")
TS_PARQUET  = os.path.join(RESULTS_DIR, f"analysis_ts_{TS}.parquet")
SUMMARY_TXT = os.path.join(RESULTS_DIR, f"analysis_summary_{TS}.txt")


# ──────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────

def _silence_debug_logs():
    for k in ("ORBIT_AGENT_LOG", "ORBIT_AGENT_LOG_PLANS_ALL",
              "ORBIT_AGENT_LOG_FLEETS",
              "SUB_AGENT_LOG", "SUB_AGENT_LOG_MISSIONS_ALL",
              "SUB_AGENT_LOG_FLEETS"):
        os.environ.pop(k, None)


def _load_module(path, name):
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_map_features(initial_obs, omega):
    """Считает характеристики карты по начальному состоянию."""
    planets = initial_obs.get('planets', []) or []

    if not planets:
        return {}

    # Owner-распределение
    our = [p for p in planets if p[1] == 0]
    sub = [p for p in planets if p[1] == 1]
    neut = [p for p in planets if p[1] == -1]

    # Production stats
    prods = [p[6] for p in planets]
    prod_neutrals = [p[6] for p in neut]
    prod_total = sum(prods)
    prod_var = float(np.var(prods)) if prods else 0.0
    prod_mean = prod_total / len(prods) if prods else 0.0

    # Orbital vs static (использует тот же критерий, что движок)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "agent_bundle"))
    from shooting import is_orbital, segment_hits_sun
    n_orbital = sum(1 for p in planets if is_orbital(p[2], p[3], p[4]))
    n_static = len(planets) - n_orbital

    # Distances home → neutrals
    if our and neut:
        home = our[0]
        dists_hn = [math.hypot(home[2] - p[2], home[3] - p[3]) for p in neut]
        min_dist_home_neutral = min(dists_hn)
        mean_dist_home_neutral = sum(dists_hn) / len(dists_hn)
    else:
        min_dist_home_neutral = mean_dist_home_neutral = 0.0

    # Mean inter-planet distance + max
    inter_d = []
    for i, p1 in enumerate(planets):
        for p2 in planets[i+1:]:
            inter_d.append(math.hypot(p1[2] - p2[2], p1[3] - p2[3]))
    mean_inter_dist = sum(inter_d) / len(inter_d) if inter_d else 0.0
    max_inter_dist = max(inter_d) if inter_d else 0.0

    # Sun blockage: доля пар планет, между которыми прямой путь блокируется
    blocked = 0
    for i, p1 in enumerate(planets):
        for p2 in planets[i+1:]:
            if segment_hits_sun(p1[2], p1[3], p2[2], p2[3]):
                blocked += 1
    sun_blockage_pct = blocked / len(inter_d) if inter_d else 0.0

    # Распределение prod по тирам (1, 2, 3, 4, 5)
    prod_tier_counts = {}
    for tier in (1, 2, 3, 4, 5):
        prod_tier_counts[f'n_prod_{tier}'] = sum(1 for p in prods if p == tier)

    # Сумма prod нейтралов — потенциал захвата
    prod_total_neutrals = sum(prod_neutrals)

    return {
        'omega': float(omega) if omega is not None else 0.0,
        'n_planets': len(planets),
        'n_orbital': n_orbital,
        'n_static': n_static,
        'n_our_start': len(our),
        'n_sub_start': len(sub),
        'n_neutrals': len(neut),
        'prod_total': prod_total,
        'prod_total_neutrals': prod_total_neutrals,
        'prod_mean': round(prod_mean, 3),
        'prod_var': round(prod_var, 3),
        'min_dist_home_neutral': round(min_dist_home_neutral, 2),
        'mean_dist_home_neutral': round(mean_dist_home_neutral, 2),
        'mean_inter_dist': round(mean_inter_dist, 2),
        'max_inter_dist': round(max_inter_dist, 2),
        'sun_blockage_pct': round(sun_blockage_pct, 4),
        **prod_tier_counts,
    }


def _extract_timeseries(env_steps, seed):
    """Достаёт per-turn series: ships, prod, n_planets, fleet stats."""
    rows = []
    for turn_idx, step in enumerate(env_steps):
        obs = step[1].get('observation') or {}
        planets = obs.get('planets') or []
        fleets = obs.get('fleets') or []

        ships_p0 = sum((p[5] or 0) for p in planets if p[1] == 0)
        ships_p1 = sum((p[5] or 0) for p in planets if p[1] == 1)
        prod_p0 = sum((p[6] or 0) for p in planets if p[1] == 0)
        prod_p1 = sum((p[6] or 0) for p in planets if p[1] == 1)
        n_p0 = sum(1 for p in planets if p[1] == 0)
        n_p1 = sum(1 for p in planets if p[1] == 1)
        n_neut = sum(1 for p in planets if p[1] == -1)
        fleet_ships_p0 = sum((f[6] or 0) for f in fleets if f[1] == 0)
        fleet_ships_p1 = sum((f[6] or 0) for f in fleets if f[1] == 1)
        n_fleets_p0 = sum(1 for f in fleets if f[1] == 0)
        n_fleets_p1 = sum(1 for f in fleets if f[1] == 1)

        rows.append({
            'seed': seed,
            'turn': turn_idx,
            'ships_p0': ships_p0,
            'ships_p1': ships_p1,
            'prod_p0': prod_p0,
            'prod_p1': prod_p1,
            'n_p0': n_p0,
            'n_p1': n_p1,
            'n_neutrals': n_neut,
            'fleet_ships_p0': fleet_ships_p0,
            'fleet_ships_p1': fleet_ships_p1,
            'n_fleets_p0': n_fleets_p0,
            'n_fleets_p1': n_fleets_p1,
        })
    return rows


def _compute_derived(ts_rows):
    """Из time series вычисляет резюмирующие признаки матча.
    Это главное сырьё для seed-кластеризации поверх map-features."""
    if not ts_rows:
        return {}
    ts = pd.DataFrame(ts_rows)
    ts['ships_diff'] = ts['ships_p0'] - ts['ships_p1']
    ts['prod_diff']  = ts['prod_p0'] - ts['prod_p1']
    ts['n_diff']     = ts['n_p0'] - ts['n_p1']
    ts['total_pool'] = ts['ships_p0'] + ts['ships_p1']
    ts['ratio_p0']   = ts['ships_p0'] / ts['total_pool'].clip(lower=1)

    n = len(ts)
    q = max(1, n // 4)

    def _safe_mean(s):
        return float(s.mean()) if len(s) else 0.0

    return {
        'final_ships_diff':    int(ts['ships_diff'].iloc[-1]),
        'final_prod_diff':     int(ts['prod_diff'].iloc[-1]),
        'final_n_diff':        int(ts['n_diff'].iloc[-1]),
        'max_ships_diff':      int(ts['ships_diff'].max()),
        'min_ships_diff':      int(ts['ships_diff'].min()),
        'max_lead_turn':       int(ts['ships_diff'].idxmax()),
        'min_lead_turn':       int(ts['ships_diff'].idxmin()),
        'first_lead_turn':     int((ts['ships_diff'] > 0).idxmax()) if (ts['ships_diff'] > 0).any() else -1,
        'first_lag_turn':      int((ts['ships_diff'] < 0).idxmax()) if (ts['ships_diff'] < 0).any() else -1,
        'lead_changes':        int(((ts['ships_diff'] > 0).astype(int).diff().abs() == 1).sum()),
        'avg_ships_diff_q1':   round(_safe_mean(ts['ships_diff'].iloc[:q]), 1),
        'avg_ships_diff_q2':   round(_safe_mean(ts['ships_diff'].iloc[q:2*q]), 1),
        'avg_ships_diff_q3':   round(_safe_mean(ts['ships_diff'].iloc[2*q:3*q]), 1),
        'avg_ships_diff_q4':   round(_safe_mean(ts['ships_diff'].iloc[3*q:]), 1),
        'avg_ratio_q1':        round(_safe_mean(ts['ratio_p0'].iloc[:q]), 3),
        'avg_ratio_q2':        round(_safe_mean(ts['ratio_p0'].iloc[q:2*q]), 3),
        'avg_ratio_q3':        round(_safe_mean(ts['ratio_p0'].iloc[2*q:3*q]), 3),
        'avg_ratio_q4':        round(_safe_mean(ts['ratio_p0'].iloc[3*q:]), 3),
        'dominance_area':      int(ts['ships_diff'].sum()),
        'mid_game_lead':       int(ts['ships_diff'].iloc[n // 2] if n > 0 else 0),
        'turn_total_war':      int((ts['n_neutrals'] == 0).idxmax()) if (ts['n_neutrals'] == 0).any() else -1,
    }


def run_one(seed):
    """Один матч; возвращает (meta_dict, list_of_ts_rows)."""
    _silence_debug_logs()
    bundle_dir = os.path.join(PROJECT_ROOT, "agent_bundle")
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    suffix = f"{seed}"
    our_mod = _load_module(OUR_PATH, f"_our_anal_{suffix}")
    sub_mod = _load_module(SUB_PATH, f"_sub_anal_{suffix}")

    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.time()
    env.run([our_mod.agent, sub_mod.agent])
    match_time = time.time() - t0

    # Map features (по initial state)
    initial_obs = env.steps[1][1].get('observation') or {}
    omega = initial_obs.get('angular_velocity', 0.0) or 0.0
    map_feat = _extract_map_features(initial_obs, omega)

    # Time series
    ts_rows = _extract_timeseries(env.steps, seed)

    # Derived
    derived = _compute_derived(ts_rows)

    # Outcome
    final = env.steps[-1]
    r0 = final[0].get('reward', 0) or 0
    r1 = final[1].get('reward', 0) or 0

    meta = {
        'seed': seed,
        'reward_our': r0,
        'reward_sub': r1,
        'win': int(r0 > r1),
        'draw': int(r0 == r1),
        'steps': len(env.steps),
        'match_time_sec': round(match_time, 1),
        **map_feat,
        **derived,
    }
    return meta, ts_rows


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def _print_box(title):
    print()
    print('═' * 70)
    print(title)
    print('═' * 70)


def _flush_partial(meta_rows, ts_rows):
    """Промежуточное сохранение — на случай ctrl+C."""
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(META_CSV, index=False)
    if ts_rows:
        try:
            pd.DataFrame(ts_rows).to_parquet(TS_PARQUET, index=False)
        except Exception:
            # parquet требует pyarrow; fallback на CSV
            pd.DataFrame(ts_rows).to_csv(TS_PARQUET.replace('.parquet', '.csv'), index=False)


def main():
    if not os.path.isfile(OUR_PATH):
        sys.exit(f"Не найден агент: {OUR_PATH}")
    if not os.path.isfile(SUB_PATH):
        sys.exit(f"Не найден baseline: {SUB_PATH}")

    seeds = list(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))

    _print_box(f"match_analyzer — сбор данных с {N_SEEDS} матчей")
    print(f"  диапазон seed: {seeds[0]}..{seeds[-1]}")
    print(f"  воркеров:      {N_WORKERS}")
    print(f"  meta CSV:      {META_CSV}")
    print(f"  ts parquet:    {TS_PARQUET}")
    print(f"  summary:       {SUMMARY_TXT}")

    meta_rows = []
    ts_rows = []
    started = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_one, s): s for s in seeds}
        for i, fut in enumerate(as_completed(futures), 1):
            seed = futures[fut]
            try:
                meta, ts = fut.result()
                meta_rows.append(meta)
                ts_rows.extend(ts)
                outcome = 'WIN ' if meta['win'] else ('DRAW' if meta['draw'] else 'LOSS')
                print(f"[{i:>4}/{len(seeds)}] seed={seed:>4}  {outcome}  "
                      f"steps={meta['steps']:>3}  "
                      f"final_diff={meta['final_ships_diff']:+6}  "
                      f"mid_lead={meta['mid_game_lead']:+5}  "
                      f"({meta['match_time_sec']}s)", flush=True)
            except Exception as e:
                print(f"[{i:>4}/{len(seeds)}] seed={seed} FAILED: {type(e).__name__}: {e}",
                      flush=True)

            # Каждые 25 матчей — промежуточный flush
            if i % 25 == 0:
                _flush_partial(meta_rows, ts_rows)

    elapsed = time.time() - started
    _flush_partial(meta_rows, ts_rows)

    # ── Сводка ──────────────────────────────────────────────────────────
    df = pd.DataFrame(meta_rows)
    if df.empty:
        print("\nНет данных для анализа.")
        return

    _print_box("СВОДКА ПО МАТЧАМ")
    n_total = len(df)
    n_wins = int(df['win'].sum())
    n_draws = int(df['draw'].sum())
    n_losses = n_total - n_wins - n_draws
    print(f"  total:    {n_total}")
    print(f"  wins:     {n_wins}  ({100*n_wins/n_total:.1f}%)")
    print(f"  draws:    {n_draws}")
    print(f"  losses:   {n_losses}  ({100*n_losses/n_total:.1f}%)")
    print(f"  avg steps: {df['steps'].mean():.1f}")
    print(f"  avg final ships_diff: {df['final_ships_diff'].mean():+.1f}")
    print(f"  median final ships_diff: {df['final_ships_diff'].median():+.0f}")

    # ── Кластеризация (базовая) ─────────────────────────────────────────
    _print_box(f"КЛАСТЕРИЗАЦИЯ КАРТ (k-means, k={N_CLUSTERS})")
    feat_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    print(f"  использованы фичи: {feat_cols}")

    X = df[feat_cols].fillna(0).values
    # Стандартизация
    mu, sigma = X.mean(0), X.std(0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    Xs = (X - mu) / sigma

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        df['cluster'] = km.fit_predict(Xs)
    except ImportError:
        print("  [skip] sklearn не установлен — пропускаем k-means")
        df['cluster'] = -1
    else:
        # Винрейт по кластерам
        cluster_stats = df.groupby('cluster').agg(
            n=('seed', 'count'),
            winrate=('win', 'mean'),
            avg_ships_diff=('final_ships_diff', 'mean'),
            avg_steps=('steps', 'mean'),
        ).round(3).sort_values('winrate', ascending=False)
        print("\n  Винрейт по кластерам:")
        print(cluster_stats.to_string())

        # Какие фичи отличают худший кластер от лучшего
        best_c = cluster_stats.index[0]
        worst_c = cluster_stats.index[-1]
        print(f"\n  Профиль фич по кластерам (mean):")
        cluster_profile = df.groupby('cluster')[feat_cols].mean().round(2)
        print(cluster_profile.to_string())
        print(f"\n  Лучший кластер (#{best_c}) vs худший (#{worst_c}):")
        diff = cluster_profile.loc[best_c] - cluster_profile.loc[worst_c]
        for f in diff.abs().sort_values(ascending=False).head(5).index:
            print(f"    {f:<28} {cluster_profile.loc[best_c, f]:>8.2f} vs "
                  f"{cluster_profile.loc[worst_c, f]:>8.2f}  Δ={diff[f]:+.2f}")

    # Корреляции map-features ↔ winrate
    _print_box("КОРРЕЛЯЦИИ map-features ↔ winrate")
    for f in feat_cols:
        if df[f].std() < 1e-9:
            continue
        c = float(df[[f, 'win']].corr().iloc[0, 1])
        print(f"  {f:<28}  corr(win) = {c:+.3f}")

    # Сохраняем обновлённый meta CSV (с cluster колонкой)
    df.to_csv(META_CSV, index=False)

    # ── Записываем сводку в txt ─────────────────────────────────────────
    with open(SUMMARY_TXT, 'w') as fh:
        fh.write(f"match_analyzer summary — {TS}\n")
        fh.write(f"seeds: {seeds[0]}..{seeds[-1]} ({n_total} матчей)\n")
        fh.write(f"winrate: {n_wins}/{n_total} = {n_wins/n_total:.3f}\n")
        fh.write(f"avg final ships_diff: {df['final_ships_diff'].mean():+.1f}\n")
        fh.write(f"\nFiles:\n  {META_CSV}\n  {TS_PARQUET}\n")

    print()
    print(f"Время: {elapsed:.0f}с ({elapsed/max(1,n_total):.1f}с/матч)")
    print(f"\nГотово. Файлы:")
    print(f"  {META_CSV}")
    print(f"  {TS_PARQUET}")
    print(f"  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
