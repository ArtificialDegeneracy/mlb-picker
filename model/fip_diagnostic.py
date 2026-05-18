"""Diagnose why fip_diff has such a small coefficient in the model.

Hypotheses tested:
  1. FIP truly isn't predictive at the game level (raw correlation with win)
  2. FIP is collinear with team_quality_diff (regression absorbs it elsewhere)
  3. FIP values are buggy — too many fall through to the 4.00 fallback
  4. FIP-only model vs FIP+team_quality model (does team_quality steal its signal?)

Usage:
    python -m model.fip_diagnostic
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

from db import get_db
from model.features import _get_pitcher_fip, _get_team_quality, _get_park_factor, _get_offense_trend


def _build_dataset(start_year, end_year):
    """Compute FIP + team quality + win for each game, tracking FIP source."""
    rows = []
    with get_db() as conn:
        games = conn.execute("""
            SELECT * FROM games
            WHERE status = 'Final'
              AND game_date >= ? AND game_date <= ?
              AND winner IS NOT NULL
            ORDER BY game_date
        """, (f"{start_year}-01-01", f"{end_year}-12-31")).fetchall()

        for g in games:
            if not g["game_date"]:
                continue
            month = int(g["game_date"][5:7])

            # Track whether FIP came from real data or fallback
            home_fip_raw = _raw_fip(g["home_starter_id"], g["home_team"], conn)
            away_fip_raw = _raw_fip(g["away_starter_id"], g["away_team"], conn)

            home_q = _get_team_quality(g["home_team"], g["game_date"], month, conn)
            away_q = _get_team_quality(g["away_team"], g["game_date"], month, conn)

            rows.append({
                "game_date": g["game_date"],
                "home_fip": home_fip_raw["value"],
                "home_fip_src": home_fip_raw["source"],
                "away_fip": away_fip_raw["value"],
                "away_fip_src": away_fip_raw["source"],
                "fip_diff": home_fip_raw["value"] - away_fip_raw["value"],
                "team_quality_diff": home_q - away_q,
                "park_factor": _get_park_factor(g),
                "home_offense_trend": _get_offense_trend(g["home_team"], g["game_date"], conn),
                "away_offense_trend": _get_offense_trend(g["away_team"], g["game_date"], conn),
                "home_win": 1 if g["winner"] == "home" else 0,
            })
    return pd.DataFrame(rows)


def _raw_fip(player_id, team_abbr, conn):
    """Replicate _get_pitcher_fip but track which branch produced the value."""
    if player_id:
        rows = conn.execute(
            "SELECT season, fip, innings_pitched FROM pitcher_stats WHERE player_id = ? AND fip IS NOT NULL ORDER BY season DESC LIMIT 2",
            (player_id,)
        ).fetchall()
        if rows:
            total_ip, weighted_fip = 0, 0
            for r in rows:
                ip = r["innings_pitched"] or 0
                if ip > 0 and r["fip"] is not None:
                    weighted_fip += r["fip"] * ip
                    total_ip += ip
            if total_ip > 0:
                return {"value": round(weighted_fip / total_ip, 2), "source": "pitcher_blended"}

    row = conn.execute(
        "SELECT AVG(fip) as avg_fip FROM pitcher_stats WHERE team = ? AND fip IS NOT NULL",
        (team_abbr,)
    ).fetchone()
    if row and row["avg_fip"]:
        return {"value": row["avg_fip"], "source": "team_avg"}
    return {"value": 4.00, "source": "default_4.00"}


def main():
    print("=" * 70)
    print("  FIP DIAGNOSTIC")
    print("  Train: 2022-2024  |  Validate: 2025")
    print("=" * 70)

    print("\nBuilding 2022-2024 training set...")
    df_train = _build_dataset(2022, 2024)
    print(f"  {len(df_train)} games")
    print("\nBuilding 2025 validation set...")
    df_val = _build_dataset(2025, 2025)
    print(f"  {len(df_val)} games")

    # === Hypothesis 3: FIP source distribution ===
    print("\n" + "=" * 70)
    print("  HYPOTHESIS 3: Are FIP values actually real, or falling through to fallback?")
    print("=" * 70)
    home_src = df_train["home_fip_src"].value_counts(normalize=True)
    away_src = df_train["away_fip_src"].value_counts(normalize=True)
    print("\nTraining set FIP source:")
    for src in ["pitcher_blended", "team_avg", "default_4.00"]:
        h = home_src.get(src, 0)
        a = away_src.get(src, 0)
        print(f"  {src:<20}  home: {h:6.1%}   away: {a:6.1%}")

    val_home_src = df_val["home_fip_src"].value_counts(normalize=True)
    print("\n2025 validation set FIP source:")
    for src in ["pitcher_blended", "team_avg", "default_4.00"]:
        h = val_home_src.get(src, 0)
        print(f"  {src:<20}  home: {h:6.1%}")

    # === Hypothesis 1: Raw FIP→win correlation ===
    print("\n" + "=" * 70)
    print("  HYPOTHESIS 1: Does FIP predict wins at all?")
    print("=" * 70)
    real_mask = (df_train["home_fip_src"] == "pitcher_blended") & (df_train["away_fip_src"] == "pitcher_blended")
    df_real = df_train[real_mask]
    print(f"\nFiltering to games with real FIP for both starters: {len(df_real)} / {len(df_train)}")

    corr_all = df_train[["fip_diff", "home_win"]].corr().loc["fip_diff", "home_win"]
    corr_real = df_real[["fip_diff", "home_win"]].corr().loc["fip_diff", "home_win"]
    print(f"\nCorrelation fip_diff vs home_win:")
    print(f"  All games:        {corr_all:+.4f}")
    print(f"  Real-FIP games:   {corr_real:+.4f}")
    print(f"  (Negative is correct: lower home FIP → more home wins)")

    # Bucket by fip_diff and look at home win rate
    print("\nHome win rate by fip_diff bucket (real-FIP games only):")
    df_real_sorted = df_real.copy()
    df_real_sorted["bucket"] = pd.qcut(df_real_sorted["fip_diff"], q=5, labels=["best home SP", "good", "even", "worse", "worst home SP"])
    bucket_stats = df_real_sorted.groupby("bucket", observed=True).agg(
        n=("home_win", "size"),
        win_rate=("home_win", "mean"),
        avg_fip_diff=("fip_diff", "mean"),
    )
    print(bucket_stats.to_string())

    # === Hypothesis 2 + 4: FIP-only vs FIP+TQD models ===
    print("\n" + "=" * 70)
    print("  HYPOTHESIS 2/4: Does team_quality_diff steal FIP's signal?")
    print("=" * 70)

    # Train several models on the same data and compare FIP's contribution
    submodels = {
        "FIP_only":           ["fip_diff"],
        "FIP_+_TQD":          ["fip_diff", "team_quality_diff"],
        "all_baseline":       ["fip_diff", "team_quality_diff", "park_factor", "home_offense_trend", "away_offense_trend"],
        "FIP_real_only":      ["fip_diff"],  # same cols but trained on real-FIP subset
    }

    y_train = df_train["home_win"].values
    y_val = df_val["home_win"].values

    print(f"\n{'Model':<22} {'FIP coef':>11} {'Val acc':>10} {'Val HIGH':>14}")
    print("-" * 60)
    for name, cols in submodels.items():
        if name == "FIP_real_only":
            X_tr_df = df_train[real_mask][cols].fillna(0)
            y_tr = df_train[real_mask]["home_win"].values
        else:
            X_tr_df = df_train[cols].fillna(0)
            y_tr = y_train
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_df)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)
        model.fit(X_tr, y_tr)

        X_val = scaler.transform(df_val[cols].fillna(0))
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds)
        high_mask = (probs >= 0.63) | (probs <= 0.37)
        correct = ((probs >= 0.5) & (y_val == 1)) | ((probs < 0.5) & (y_val == 0))
        high_acc = correct[high_mask].mean() if high_mask.sum() else None
        high_n = int(high_mask.sum())
        fip_coef = model.coef_[0][cols.index("fip_diff")]

        high_str = f"{high_acc:.1%} (n={high_n})" if high_acc is not None else "—"
        print(f"{name:<22} {fip_coef:>+11.4f} {acc:>9.1%}  {high_str:>13}")

    # === Hypothesis 4: FIP magnitude — is the standardized value squashing the signal? ===
    print("\n" + "=" * 70)
    print("  HYPOTHESIS 4: Is standardization squashing FIP's range?")
    print("=" * 70)
    print(f"\nfip_diff distribution (training, real-FIP games):")
    s = df_real["fip_diff"]
    print(f"  mean:  {s.mean():+.3f}")
    print(f"  std:   {s.std():.3f}")
    print(f"  min:   {s.min():+.3f}")
    print(f"  p10:   {s.quantile(0.1):+.3f}")
    print(f"  p90:   {s.quantile(0.9):+.3f}")
    print(f"  max:   {s.max():+.3f}")
    print(f"\nA 'big edge' game (top decile of |fip_diff|) is about {abs(s.quantile(0.9)):.2f} runs.")
    print(f"Standardized, that's {abs(s.quantile(0.9)) / s.std():.2f} std deviations from mean.")
    print(f"With coef -0.05, that contributes only {abs(-0.05 * abs(s.quantile(0.9)) / s.std()):.3f} to the logit.")
    print(f"For comparison, team_quality_diff's coefficient (+0.47) times its p90 std-units contributes:")
    tqd = df_train["team_quality_diff"]
    print(f"  TQD p90 = {tqd.quantile(0.9):.3f}, in std-units = {tqd.quantile(0.9) / tqd.std():.2f}")
    print(f"  Contribution to logit: {0.47 * tqd.quantile(0.9) / tqd.std():.3f}")


if __name__ == "__main__":
    main()
