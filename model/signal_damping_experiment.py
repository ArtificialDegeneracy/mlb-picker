"""Signal-disagreement damping experiment (Option B).

Hypothesis: when the model's prediction is supported by FEW of the underlying signals
(record, FIP, bullpen, wRC+, recent form), it's overconfident. Damp the probability
toward 50% based on how few signals agree.

Method:
  1. Re-run the production 5-feature model on every 2022-2025 finalized game (using
     season-aware data, no lookahead)
  2. For each game, compute the 5 signals and check which support the model's pick
  3. Slice by "signal count supporting pick" (0-5) and measure:
     - Predicted probability (mean, by pick tier)
     - Actual win rate
     - Calibration gap (predicted - actual)
  4. The slices with the biggest predicted-vs-actual gap tell us where damping helps
  5. Try several damping rules and report which performs best on a 2025 holdout

Limitations:
  - bullpen_era, wrc_plus are season-aggregates. For mid-season games (April-May),
    a 2025-05-18 pick would use full-season 2025 stats, which include games AFTER
    the pick was made. This is a mild lookahead bias for the signal-tag analysis
    only, not for the model itself. Helpers already have a season<=game_year filter
    via yesterday's fix, but they return the latest available value within that.
  - We only have 4 years of data. With 5 signals × probability tiers, slices get
    thin. We use a 2022-2024 fit / 2025 holdout split.

Usage:
    python -m model.signal_damping_experiment
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_db
from model.features import (
    FEATURE_NAMES,
    resolve_feature_names,
    _get_pitcher_fip,
    _get_team_quality,
    _get_park_factor,
    _get_offense_trend,
    _get_bullpen_era,
    _get_wrc_plus,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def _compute_signals(g, conn, season):
    """Compute the 5 signals for a game. Returns a dict of (signal_name → favored_side).

    favored_side is 'home', 'away', or None (no edge / no data).
    """
    signals = {}

    # 1. Record edge — who has the better blended W-L
    home_q = _get_team_quality(g["home_team"], g["game_date"], int(g["game_date"][5:7]), conn)
    away_q = _get_team_quality(g["away_team"], g["game_date"], int(g["game_date"][5:7]), conn)
    if abs(home_q - away_q) < 0.005:
        signals["record"] = None  # essentially tied
    else:
        signals["record"] = "home" if home_q > away_q else "away"

    # 2. FIP edge — who has the lower (better) starter FIP
    h_fip = _get_pitcher_fip(g["home_starter_id"], g["home_team"], conn)
    a_fip = _get_pitcher_fip(g["away_starter_id"], g["away_team"], conn)
    if h_fip is None or a_fip is None or abs(h_fip - a_fip) < 0.10:
        signals["fip"] = None
    else:
        signals["fip"] = "home" if h_fip < a_fip else "away"

    # 3. Bullpen edge — who has the lower bullpen ERA
    h_bp = _get_bullpen_era(g["home_team"], conn, season=season)
    a_bp = _get_bullpen_era(g["away_team"], conn, season=season)
    if h_bp is None or a_bp is None or abs(h_bp - a_bp) < 0.15:
        signals["bullpen"] = None
    else:
        signals["bullpen"] = "home" if h_bp < a_bp else "away"

    # 4. wRC+ edge — who has the better offense
    h_wrc = _get_wrc_plus(g["home_team"], conn, season=season)
    a_wrc = _get_wrc_plus(g["away_team"], conn, season=season)
    if h_wrc is None or a_wrc is None or abs(h_wrc - a_wrc) < 2.0:
        signals["wrc"] = None
    else:
        signals["wrc"] = "home" if h_wrc > a_wrc else "away"

    # 5. Recent form — who's hotter on offense lately
    h_trend = _get_offense_trend(g["home_team"], g["game_date"], conn)
    a_trend = _get_offense_trend(g["away_team"], g["game_date"], conn)
    if abs(h_trend - a_trend) < 0.3:
        signals["form"] = None
    else:
        signals["form"] = "home" if h_trend > a_trend else "away"

    return signals


def _build_feature_vector_for_backtest(g, conn):
    """Build the production feature vector — same as model.features.build_feature_vector
    but works on a single row efficiently."""
    feats = {}
    month = int(g["game_date"][5:7])
    h_fip = _get_pitcher_fip(g["home_starter_id"], g["home_team"], conn)
    a_fip = _get_pitcher_fip(g["away_starter_id"], g["away_team"], conn)
    feats["fip_diff"] = (h_fip - a_fip) if (h_fip is not None and a_fip is not None) else 0.0
    h_q = _get_team_quality(g["home_team"], g["game_date"], month, conn)
    a_q = _get_team_quality(g["away_team"], g["game_date"], month, conn)
    feats["team_quality_diff"] = h_q - a_q
    feats["park_factor"] = _get_park_factor(g)
    feats["home_offense_trend"] = _get_offense_trend(g["home_team"], g["game_date"], conn)
    feats["away_offense_trend"] = _get_offense_trend(g["away_team"], g["game_date"], conn)
    return feats


def main():
    print("=" * 70)
    print("  SIGNAL DAMPING EXPERIMENT (Option B)")
    print("=" * 70)

    # Load production model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Build dataset: for every 2022-2025 final game, compute features + signals + prediction
    rows = []
    with get_db() as conn:
        games = conn.execute("""
            SELECT * FROM games
            WHERE status = 'Final'
              AND game_date >= '2022-01-01' AND game_date <= '2025-12-31'
              AND winner IS NOT NULL
            ORDER BY game_date
        """).fetchall()
        print(f"\nProcessing {len(games)} finalized games...")

        for i, g in enumerate(games):
            season = int(g["game_date"][:4])
            feats = _build_feature_vector_for_backtest(g, conn)
            signals = _compute_signals(g, conn, season)

            # Predict using production model — resolve the schema it was trained on,
            # since FEATURE_NAMES may have grown past what this pickle expects.
            feat_df = pd.DataFrame([feats])[resolve_feature_names(model)].fillna(0)
            feat_scaled = scaler.transform(feat_df)
            home_win_prob = float(model.predict_proba(feat_scaled)[0][1])

            picked_side = "home" if home_win_prob >= 0.5 else "away"
            pick_prob = home_win_prob if picked_side == "home" else 1 - home_win_prob

            # Count signals supporting the picked side
            supporting = 0
            total_with_signal = 0
            for sig_name, fav in signals.items():
                if fav is None:
                    continue
                total_with_signal += 1
                if fav == picked_side:
                    supporting += 1

            won = (g["winner"] == picked_side)

            rows.append({
                "game_date": g["game_date"],
                "season": season,
                "game_id": g["game_id"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_win_prob": home_win_prob,
                "pick_prob": pick_prob,
                "picked_side": picked_side,
                "supporting": supporting,
                "total_with_signal": total_with_signal,
                "won": int(won),
                "sig_record": signals["record"],
                "sig_fip": signals["fip"],
                "sig_bullpen": signals["bullpen"],
                "sig_wrc": signals["wrc"],
                "sig_form": signals["form"],
            })

            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(games)}...")

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} games processed. Sample:")
    print(df[["game_date", "home_team", "away_team", "pick_prob", "supporting", "total_with_signal", "won"]].head(10).to_string(index=False))

    # === Slice by supporting-signal count, all tiers ===
    print("\n" + "=" * 70)
    print("  CALIBRATION BY SUPPORTING-SIGNAL COUNT (all picks)")
    print("=" * 70)
    print(f"\n  {'Sig count':>10}  {'N':>6}  {'avg pred':>10}  {'actual':>8}  {'gap':>7}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*7}")
    for n in range(6):
        sub = df[df["supporting"] == n]
        if len(sub) == 0:
            continue
        avg_pred = sub["pick_prob"].mean()
        actual = sub["won"].mean()
        gap = avg_pred - actual
        print(f"  {n}/5        {len(sub):>6}  {avg_pred:>10.3f}  {actual:>8.3f}  {gap:>+7.3f}")

    # === Same but for HIGH and MEDIUM tiers separately ===
    print("\n" + "=" * 70)
    print("  CALIBRATION BY SIGNAL COUNT × TIER")
    print("=" * 70)
    tiers = {
        "HIGH (≥0.63)": df[df["pick_prob"] >= 0.63],
        "MEDIUM (0.55-0.63)": df[(df["pick_prob"] >= 0.55) & (df["pick_prob"] < 0.63)],
        "LEAN (<0.55)": df[df["pick_prob"] < 0.55],
    }
    for tier_label, tier_df in tiers.items():
        print(f"\n  {tier_label}:  {len(tier_df)} picks")
        print(f"  {'Sig count':>10}  {'N':>6}  {'avg pred':>10}  {'actual':>8}  {'gap':>7}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*7}")
        for n in range(6):
            sub = tier_df[tier_df["supporting"] == n]
            if len(sub) == 0:
                continue
            avg_pred = sub["pick_prob"].mean()
            actual = sub["won"].mean()
            gap = avg_pred - actual
            flag = " ←" if abs(gap) > 0.04 and len(sub) >= 30 else ""
            print(f"  {n}/5        {len(sub):>6}  {avg_pred:>10.3f}  {actual:>8.3f}  {gap:>+7.3f}{flag}")

    # === Train: fit damping rule on 2022-2024, validate on 2025 ===
    train_df = df[df["season"] <= 2024].copy()
    val_df = df[df["season"] == 2025].copy()
    print(f"\n  Train (2022-2024): {len(train_df)} games, Val (2025): {len(val_df)} games")

    # Define candidate damping rules. Each rule maps supporting count → damping factor.
    # Damping factor d shrinks pick_prob toward 0.5 by fraction d:
    #   new_prob = 0.5 + (pick_prob - 0.5) * (1 - d)
    # d=0 means no damping; d=1 means collapse to 50%.
    # Initial hypothesis was: dampen mixed-signal picks (0-1/5).
    # 2022-2024 calibration showed the OPPOSITE: 5/5 HIGH picks systematically
    # overpredict by ~3pp on a large sample, while 2-3/5 HIGH picks OUTperform.
    # These rules now test the inverse: dampen ALIGNED picks where overconfidence is real.
    # Initial hypothesis was: dampen mixed-signal picks (0-1/5).
    # 2022-2024 calibration showed the OPPOSITE: aligned-signal picks overpredict.
    # Drilling deeper: the overprediction is concentrated in AWAY-team picks (gap +0.101)
    # while HOME-team picks are well-calibrated (gap -0.010). The market/model
    # systematically overrates away favorites when all signals align.
    # Test rules that target this asymmetry. Keys can be `n` or `(n, side)`.
    rules = {
        "no_damping":             {n: 0.0 for n in range(6)},
        # Damp ALL 5/5 picks equally (proves the asymmetry matters)
        "rule_5all":              {5: 0.15},
        # Damp ONLY away 5/5 picks (the asymmetric finding)
        "rule_away5_only":        {(5, "away"): 0.20},
        # Damp away 4/5 + 5/5 (broaden the asymmetric rule)
        "rule_away4and5":         {(4, "away"): 0.08, (5, "away"): 0.20},
        # Aggressive: damp all away HIGH-tier picks with 3+ signals
        "rule_away3to5":          {(3, "away"): 0.05, (4, "away"): 0.10, (5, "away"): 0.20},
    }

    def apply_rule(p, supporting, rule, picked_side=None):
        # Support rules that vary by picked_side; key format: (signals, side) or just signals
        if isinstance(rule, dict):
            keys = rule.get((supporting, picked_side), rule.get(supporting, 0.0))
            d = keys
        else:
            d = 0.0
        return 0.5 + (p - 0.5) * (1 - d)

    # Evaluate each rule on 2025 holdout
    print("\n" + "=" * 70)
    print("  DAMPING RULE EVALUATION (2025 holdout)")
    print("=" * 70)
    print(f"\n  {'Rule':<22} {'overall':>9}  {'HIGH acc':>16}  {'MED acc':>16}  {'Brier':>7}")
    print(f"  {'-'*22} {'-'*9}  {'-'*16}  {'-'*16}  {'-'*7}")
    for name, rule in rules.items():
        damped_probs = val_df.apply(
            lambda r: apply_rule(r["pick_prob"], r["supporting"], rule, r["picked_side"]), axis=1
        )
        damped_win = (val_df["won"]).values
        high_mask = (damped_probs >= 0.63).values
        med_mask = ((damped_probs >= 0.55) & (damped_probs < 0.63)).values
        overall = damped_win.mean()
        h_acc = damped_win[high_mask].mean() if high_mask.sum() else 0
        m_acc = damped_win[med_mask].mean() if med_mask.sum() else 0
        brier = ((damped_probs - val_df["won"].values) ** 2).mean()
        print(f"  {name:<22} {overall:>8.1%}  {h_acc:>10.1%} (n={high_mask.sum():>4})  {m_acc:>10.1%} (n={med_mask.sum():>4})  {brier:.4f}")

    # Show the count of picks that drop from HIGH→MED or MED→LEAN under each rule
    print("\n" + "=" * 70)
    print("  TIER MIGRATION UNDER DAMPING (2025 holdout)")
    print("=" * 70)
    for name, rule in rules.items():
        if name == "no_damping":
            continue
        original_high = val_df["pick_prob"] >= 0.63
        damped_high = val_df.apply(
            lambda r: apply_rule(r["pick_prob"], r["supporting"], rule, r["picked_side"]), axis=1
        ) >= 0.63
        downgrade_h_to_med = (original_high & ~damped_high).sum()
        print(f"  {name:<22}  HIGH→lower: {downgrade_h_to_med}/{original_high.sum()} ({100*downgrade_h_to_med/max(1,original_high.sum()):.0f}%)")

    # === Validate the asymmetric pattern on 2025 holdout ===
    print("\n" + "=" * 70)
    print("  ASYMMETRIC PATTERN: HIGH picks split by home/away (2025 holdout)")
    print("=" * 70)
    high_val = val_df[val_df["pick_prob"] >= 0.63]
    for side in ["home", "away"]:
        for sigs in [(2,3), (4,), (5,)]:
            b = high_val[(high_val["picked_side"] == side) & (high_val["supporting"].isin(sigs))]
            if len(b) < 5:
                continue
            sig_label = "+".join(str(s) for s in sigs)
            print(f"    {side:>4} pick, {sig_label}/5 signals: N={len(b):>3}  pred={b['pick_prob'].mean():.3f}  actual={b['won'].mean():.3f}  gap={b['pick_prob'].mean()-b['won'].mean():+.3f}")

    # === Investigate the 5/5 HIGH overconfidence pattern ===
    print("\n" + "=" * 70)
    print("  WHY DO 5/5 HIGH PICKS UNDERPERFORM? (training data only)")
    print("=" * 70)
    sub = train_df[(train_df["pick_prob"] >= 0.63) & (train_df["supporting"] == 5)]
    print(f"\n  All 5/5 HIGH picks (train): {len(sub)} picks")
    print(f"    avg predicted: {sub['pick_prob'].mean():.3f}, actual: {sub['won'].mean():.3f}, gap: {sub['pick_prob'].mean()-sub['won'].mean():+.3f}")

    # Slice by home/away pick
    home = sub[sub["picked_side"] == "home"]
    away = sub[sub["picked_side"] == "away"]
    print(f"\n  Home picks: {len(home)}  pred {home['pick_prob'].mean():.3f}  actual {home['won'].mean():.3f}  gap {home['pick_prob'].mean()-home['won'].mean():+.3f}")
    print(f"  Away picks: {len(away)}  pred {away['pick_prob'].mean():.3f}  actual {away['won'].mean():.3f}  gap {away['pick_prob'].mean()-away['won'].mean():+.3f}")

    # Slice by probability bucket within 5/5
    print(f"\n  Sub-buckets by predicted probability:")
    print(f"  {'prob range':>15}  {'N':>5}  {'pred':>7}  {'actual':>7}  {'gap':>7}")
    for lo, hi in [(0.63, 0.66), (0.66, 0.70), (0.70, 0.75), (0.75, 1.0)]:
        b = sub[(sub["pick_prob"] >= lo) & (sub["pick_prob"] < hi)]
        if len(b) < 10:
            continue
        print(f"  {lo:.2f}-{hi:.2f}   {len(b):>5}  {b['pick_prob'].mean():>.3f}  {b['won'].mean():>.3f}  {b['pick_prob'].mean()-b['won'].mean():>+.3f}")

    # Slice by year — is this consistent or just noise from one year?
    print(f"\n  By year:")
    for season in [2022, 2023, 2024, 2025]:
        b = train_df[(train_df["pick_prob"] >= 0.63) & (train_df["supporting"] == 5) & (train_df["season"] == season)] if season != 2025 else df[(df["pick_prob"] >= 0.63) & (df["supporting"] == 5) & (df["season"] == 2025)]
        if len(b) == 0:
            continue
        print(f"    {season}: N={len(b):>4}  pred={b['pick_prob'].mean():.3f}  actual={b['won'].mean():.3f}  gap={b['pick_prob'].mean()-b['won'].mean():+.3f}")

    # And what about 2/5 and 3/5 HIGH overperformance — is that consistent too?
    print(f"\n  HIGH picks with 2-3/5 supporting (the 'overperforming' bucket):")
    sub23 = train_df[(train_df["pick_prob"] >= 0.63) & (train_df["supporting"].isin([2, 3]))]
    print(f"    N={len(sub23)}  pred={sub23['pick_prob'].mean():.3f}  actual={sub23['won'].mean():.3f}  gap={sub23['pick_prob'].mean()-sub23['won'].mean():+.3f}")
    for season in [2022, 2023, 2024]:
        b = sub23[sub23["season"] == season]
        if len(b) == 0:
            continue
        print(f"    {season}: N={len(b):>4}  pred={b['pick_prob'].mean():.3f}  actual={b['won'].mean():.3f}  gap={b['pick_prob'].mean()-b['won'].mean():+.3f}")

    # === If ATL @ MIA from yesterday were in this dataset, what would damping do? ===
    print("\n" + "=" * 70)
    print("  TODAY'S EXAMPLE: ATL @ MIA")
    print("=" * 70)
    print("  Model says: ATL 69.5% (HIGH)")
    print("  Manual signal check from earlier deep-dive: 1/5 supports ATL")
    for name, rule in rules.items():
        d = rule.get(1, 0.0)
        new_prob = 0.5 + (0.695 - 0.5) * (1 - d)
        tier = "HIGH" if new_prob >= 0.63 else ("MED" if new_prob >= 0.55 else "LEAN")
        print(f"  {name:<22}  damping factor {d:.2f}  →  {new_prob:.1%} ({tier})")


if __name__ == "__main__":
    main()
