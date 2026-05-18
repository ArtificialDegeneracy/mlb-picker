"""Feature expansion experiment.

Compares model variants to decide whether bullpen_diff, wrc_plus_diff, and
platoon_wrc_diff should be added to the production feature set.

Method:
  - Train on 2022-2024 finalized games (~7,287)
  - Validate on 2025 full season (~2,428)
  - Same logistic regression hyperparams as production (class_weight=balanced, C=0.5)
  - Same feature matrix shared across variants — only the column selection changes
  - Helpers query season-appropriate team_stats rows (no lookahead)

Variants:
  A — baseline (current 6 features)
  B — baseline minus dead home_flag (5 features)
  C — B + bullpen_diff
  D — B + wrc_plus_diff
  E — B + platoon_wrc_diff
  F — B + all three

Usage:
    python -m model.feature_experiment
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
from db import get_db
from model.features import (
    _get_pitcher_fip,
    _get_team_quality,
    _get_park_factor,
    _get_offense_trend,
    _get_bullpen_era,
    _get_wrc_plus,
    _get_platoon_wrc,
)


# All candidate features. Variants pick subsets of these column names.
ALL_FEATURES = [
    "fip_diff",
    "home_flag",
    "team_quality_diff",
    "park_factor",
    "home_offense_trend",
    "away_offense_trend",
    "bullpen_diff",
    "wrc_plus_diff",
    "platoon_wrc_diff",
]

VARIANTS = {
    "A_baseline":           ["fip_diff", "home_flag", "team_quality_diff", "park_factor", "home_offense_trend", "away_offense_trend"],
    "B_no_home_flag":       ["fip_diff", "team_quality_diff", "park_factor", "home_offense_trend", "away_offense_trend"],
    "C_+bullpen":           ["fip_diff", "team_quality_diff", "park_factor", "home_offense_trend", "away_offense_trend", "bullpen_diff"],
    "D_+wrc":               ["fip_diff", "team_quality_diff", "park_factor", "home_offense_trend", "away_offense_trend", "wrc_plus_diff"],
    "E_+platoon":           ["fip_diff", "team_quality_diff", "park_factor", "home_offense_trend", "away_offense_trend", "platoon_wrc_diff"],
    "F_all_three":          ["fip_diff", "team_quality_diff", "park_factor", "home_offense_trend", "away_offense_trend", "bullpen_diff", "wrc_plus_diff", "platoon_wrc_diff"],
    # Diagnostic variants: probe whether team_quality_diff is dominant because it's
    # truly the best signal, or because it's a collinear aggregator of others.
    "G_no_team_quality":    ["fip_diff", "park_factor", "home_offense_trend", "away_offense_trend"],
    "H_components_only":    ["fip_diff", "park_factor", "home_offense_trend", "away_offense_trend", "bullpen_diff", "wrc_plus_diff", "platoon_wrc_diff"],
}


def _build_full_feature_vector(game_row, conn):
    """Compute every candidate feature for a single game, with season-appropriate data."""
    game_date = game_row["game_date"]
    if not game_date:
        return None
    season = int(game_date[:4])
    month = int(game_date[5:7])

    home_fip = _get_pitcher_fip(game_row["home_starter_id"], game_row["home_team"], conn)
    away_fip = _get_pitcher_fip(game_row["away_starter_id"], game_row["away_team"], conn)
    fip_diff = (home_fip - away_fip) if (home_fip is not None and away_fip is not None) else 0.0

    home_q = _get_team_quality(game_row["home_team"], game_date, month, conn)
    away_q = _get_team_quality(game_row["away_team"], game_date, month, conn)

    home_bp = _get_bullpen_era(game_row["home_team"], conn, season=season)
    away_bp = _get_bullpen_era(game_row["away_team"], conn, season=season)
    bullpen_diff = (home_bp - away_bp) if (home_bp is not None and away_bp is not None) else 0.0

    home_wrc = _get_wrc_plus(game_row["home_team"], conn, season=season)
    away_wrc = _get_wrc_plus(game_row["away_team"], conn, season=season)
    wrc_plus_diff = (home_wrc - away_wrc) if (home_wrc is not None and away_wrc is not None) else 0.0

    home_pl = _get_platoon_wrc(game_row["home_team"], game_row["away_starter_id"], conn, season=season)
    away_pl = _get_platoon_wrc(game_row["away_team"], game_row["home_starter_id"], conn, season=season)
    platoon_wrc_diff = (home_pl - away_pl) if (home_pl is not None and away_pl is not None) else 0.0

    return {
        "fip_diff": fip_diff,
        "home_flag": 1.0,
        "team_quality_diff": home_q - away_q,
        "park_factor": _get_park_factor(game_row),
        "home_offense_trend": _get_offense_trend(game_row["home_team"], game_date, conn),
        "away_offense_trend": _get_offense_trend(game_row["away_team"], game_date, conn),
        "bullpen_diff": bullpen_diff,
        "wrc_plus_diff": wrc_plus_diff,
        "platoon_wrc_diff": platoon_wrc_diff,
    }


def _build_dataset(start_year, end_year):
    """Compute all candidate features for every finalized game in the window."""
    feats, labels = [], []
    with get_db() as conn:
        games = conn.execute("""
            SELECT * FROM games
            WHERE status = 'Final'
              AND game_date >= ? AND game_date <= ?
              AND winner IS NOT NULL
            ORDER BY game_date
        """, (f"{start_year}-01-01", f"{end_year}-12-31")).fetchall()

        print(f"  Building features for {len(games)} games ({start_year}-{end_year})...")
        skipped = 0
        for i, g in enumerate(games):
            row = _build_full_feature_vector(g, conn)
            if row is None:
                skipped += 1
                continue
            feats.append(row)
            labels.append(1 if g["winner"] == "home" else 0)
            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{len(games)}...")
        if skipped:
            print(f"    Skipped {skipped} games")
    return pd.DataFrame(feats), np.array(labels)


def _evaluate(model, scaler, X_val_df, y_val, feature_cols):
    """Run model on val set; return metrics dict."""
    X = scaler.transform(X_val_df[feature_cols].fillna(0))
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    pick_correct = ((probs >= 0.5) & (y_val == 1)) | ((probs < 0.5) & (y_val == 0))
    high_mask = (probs >= HIGH_CONFIDENCE_THRESHOLD) | (probs <= 1 - HIGH_CONFIDENCE_THRESHOLD)
    med_mask = ((probs >= MEDIUM_CONFIDENCE_THRESHOLD) | (probs <= 1 - MEDIUM_CONFIDENCE_THRESHOLD)) & ~high_mask
    lean_mask = ~high_mask & ~med_mask
    return {
        "accuracy": float(accuracy_score(y_val, preds)),
        "high_acc": float(pick_correct[high_mask].mean()) if high_mask.sum() else None,
        "high_n": int(high_mask.sum()),
        "med_acc": float(pick_correct[med_mask].mean()) if med_mask.sum() else None,
        "med_n": int(med_mask.sum()),
        "lean_acc": float(pick_correct[lean_mask].mean()) if lean_mask.sum() else None,
        "lean_n": int(lean_mask.sum()),
        "brier": float(np.mean((probs - y_val) ** 2)),
    }


def main():
    print("=" * 70)
    print("  FEATURE EXPANSION EXPERIMENT")
    print("  Train: 2022-2024  |  Validate: 2025")
    print("=" * 70)

    print("\nBuilding training dataset...")
    X_train_df, y_train = _build_dataset(2022, 2024)
    print(f"  Training: {len(X_train_df)} games")

    print("\nBuilding validation dataset...")
    X_val_df, y_val = _build_dataset(2025, 2025)
    print(f"  Validation: {len(X_val_df)} games")

    # Feature coverage diagnostic — are the new columns actually non-zero?
    print("\nFeature coverage on training set (% non-zero, mean, std):")
    for col in ALL_FEATURES:
        s = X_train_df[col].fillna(0)
        nz = (s != 0).mean()
        print(f"  {col:<22} non-zero={nz:6.1%}  mean={s.mean():+7.3f}  std={s.std():6.3f}")

    # Collinearity diagnostic — how correlated are the candidate features?
    print("\nCorrelation matrix (training set):")
    diag_cols = ["fip_diff", "team_quality_diff", "bullpen_diff", "wrc_plus_diff", "platoon_wrc_diff"]
    corr = X_train_df[diag_cols].fillna(0).corr()
    print(f"  {'':22}" + "".join(f"{c[:14]:>15}" for c in diag_cols))
    for r in diag_cols:
        row = f"  {r:<22}"
        for c in diag_cols:
            row += f"{corr.loc[r, c]:>+15.3f}"
        print(row)

    results = {}
    for name, cols in VARIANTS.items():
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_df[cols].fillna(0))
        model = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)
        model.fit(X_tr, y_train)
        metrics = _evaluate(model, scaler, X_val_df, y_val, cols)
        coefs = dict(zip(cols, [float(c) for c in model.coef_[0]]))
        results[name] = {"cols": cols, "metrics": metrics, "coefs": coefs}

    # Headline comparison table
    print("\n" + "=" * 70)
    print("  RESULTS (2025 validation)")
    print("=" * 70)
    print(f"\n{'Variant':<18} {'Overall':>8} {'HIGH':>14} {'MED':>14} {'Brier':>8}")
    print("-" * 70)
    baseline_acc = results["A_baseline"]["metrics"]["accuracy"]
    for name, r in results.items():
        m = r["metrics"]
        h = f"{m['high_acc']:.1%} (n={m['high_n']})" if m["high_acc"] is not None else "—"
        md = f"{m['med_acc']:.1%} (n={m['med_n']})" if m["med_acc"] is not None else "—"
        delta = m["accuracy"] - baseline_acc
        delta_str = f"{delta:+.1%}" if name != "A_baseline" else "  —  "
        print(f"{name:<18} {m['accuracy']:>7.1%}  {h:>13}  {md:>13}  {m['brier']:.4f}   {delta_str}")

    # Coefficients table — focus on the new features
    print("\n" + "=" * 70)
    print("  COEFFICIENTS (standardized features)")
    print("=" * 70)
    print(f"\n{'Feature':<22} " + "".join(f"{name[:10]:>11}" for name in VARIANTS))
    print("-" * (22 + 11 * len(VARIANTS)))
    for feat in ALL_FEATURES:
        row = f"{feat:<22} "
        for name in VARIANTS:
            coef = results[name]["coefs"].get(feat)
            row += f"{coef:+11.4f}" if coef is not None else f"{'—':>11}"
        print(row)

    # Decision gate
    print("\n" + "=" * 70)
    print("  DECISION GATE")
    print("=" * 70)
    print("  Ship if: overall accuracy ≥ +1.0% AND HIGH/MED don't regress >0.5%\n")
    base = results["A_baseline"]["metrics"]
    for name, r in results.items():
        if name == "A_baseline":
            continue
        m = r["metrics"]
        overall_d = m["accuracy"] - base["accuracy"]
        high_d = (m["high_acc"] or 0) - (base["high_acc"] or 0) if base["high_acc"] else 0
        med_d = (m["med_acc"] or 0) - (base["med_acc"] or 0) if base["med_acc"] else 0
        passes = overall_d >= 0.01 and high_d >= -0.005 and med_d >= -0.005
        flag = "✓ SHIP" if passes else "✗ skip"
        print(f"  {name:<18}  overall {overall_d:+.1%}  HIGH {high_d:+.1%}  MED {med_d:+.1%}   → {flag}")

    # Tiered diagnostic: does the new feature help when team quality is ambiguous?
    # Split 2025 games by |team_quality_diff|: low = teams look similar (room for
    # other signals), high = one team clearly stronger (W-L is decisive).
    print("\n" + "=" * 70)
    print("  WHERE DO NEW FEATURES HELP? (split by |team_quality_diff|)")
    print("=" * 70)
    tqd = X_val_df["team_quality_diff"].fillna(0).abs()
    p33, p67 = tqd.quantile(0.33), tqd.quantile(0.67)
    masks = {
        f"close (|TQD|<{p33:.3f})":     (tqd <  p33).values,
        f"moderate ({p33:.3f}–{p67:.3f})": ((tqd >= p33) & (tqd < p67)).values,
        f"decisive (|TQD|≥{p67:.3f})":  (tqd >= p67).values,
    }
    print(f"\n{'Variant':<18}" + "".join(f"{m:>26}" for m in masks))
    print("-" * (18 + 26 * len(masks)))
    for name in ["A_baseline", "F_all_three", "H_components_only"]:
        cols = VARIANTS[name]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_df[cols].fillna(0))
        model = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)
        model.fit(X_tr, y_train)
        X_val = scaler.transform(X_val_df[cols].fillna(0))
        probs = model.predict_proba(X_val)[:, 1]
        correct = ((probs >= 0.5) & (y_val == 1)) | ((probs < 0.5) & (y_val == 0))
        row = f"{name:<18}"
        for mask in masks.values():
            acc = correct[mask].mean() if mask.sum() else 0
            row += f"  {acc:.1%} (n={mask.sum():4})        "[:26]
        print(row)


if __name__ == "__main__":
    main()
