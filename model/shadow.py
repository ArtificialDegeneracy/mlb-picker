"""
Champion/challenger shadow picks.

WHAT THIS IS
------------
Production (model/predict.py + scheduler.py + the daily GitHub Actions crons)
keeps making the real contest picks, untouched. This module runs an XGBoost
*challenger* in parallel: it predicts the same games, writes "shadow picks" to
the shadow_picks table, and a comparison report scores champion vs challenger
side by side over time.

Nothing here writes to the production `picks` table. Nothing in the production
path imports this module. The shadow test cannot affect a real contest pick.

WHY
---
The current model (logistic regression, 5 features) has hit its ceiling
(TODO.md "Path C"). Before swapping in XGBoost we want a LIVE forward
comparison, not just a backtest — a backtest can be fooled by overfitting; a
shadow test on games the model has never seen cannot.

WHAT IT COMPARES (and what it deliberately does NOT, yet)
---------------------------------------------------------
This runs XGBoost on the SAME 5 production features (model/features.py:
FEATURE_NAMES). That isolates one variable — "does the tree model beat the
linear model?" — independent of the separate balldontlie-features question.
Once the balldontlie trial proves out, a second model_version ('xgb_bdl')
can shadow with the staged features added.

It also does NOT apply production's 5-stage calibration stack (opener
dampening, away-overconfidence damping, LEAN-flip, lineup nudges). That stack
was tuned for the logreg output distribution; applying it to XGBoost would
confound the comparison. The honest first comparison is raw-model vs
raw-model. Tiering uses the same thresholds as production so HIGH/MED/LEAN
buckets line up.

USAGE
-----
    # train the challenger + shadow-pick a date (defaults to today)
    python -m model.shadow predict --date 2026-05-21

    # score past shadow picks against final results (mirrors run_results)
    python -m model.shadow score --date 2026-05-20

    # champion-vs-challenger accuracy report
    python -m model.shadow report

Wire `predict` into a daily cron alongside the production run to accumulate a
live comparison; `score` the morning after.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD, SEASON
from db import get_db
from model.features import FEATURE_NAMES, build_training_features, build_feature_vector

# The challenger identity. A future balldontlie-fed variant would be 'xgb_bdl'.
MODEL_VERSION = "xgb_5feat"

# Conservative XGBoost config — see model/xgb_experiment.py for the rationale
# (small dataset => shallow trees + regularization or it overfits instantly).
XGB_PARAMS = dict(
    max_depth=3, n_estimators=200, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=1.0, eval_metric="logloss", random_state=42, n_jobs=-1,
)


def _train_challenger(cutoff_date):
    """
    Train the XGBoost challenger on all Final games STRICTLY BEFORE cutoff_date.

    The cutoff guard is what makes the shadow test honest: the challenger is
    only ever scored on games it did not train on. When shadow-picking
    2026-05-21, the model trains on everything through 2026-05-20.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.exit("xgboost not installed — `pip install xgboost`.")

    feats, labels, _, dates = build_training_features(
        2022, SEASON, return_dates=True)
    # keep only games before the cutoff
    keep = [i for i, d in enumerate(dates) if d < cutoff_date]
    if len(keep) < 500:
        sys.exit(f"only {len(keep)} games before {cutoff_date} — too few to "
                 "train a challenger.")
    X = pd.DataFrame([feats[i] for i in keep])[FEATURE_NAMES].fillna(0)
    y = np.array([labels[i] for i in keep])

    pos = int(y.sum())
    neg = len(y) - pos
    model = XGBClassifier(scale_pos_weight=(neg / pos) if pos else 1.0,
                          **XGB_PARAMS)
    model.fit(X, y)
    print(f"  challenger trained on {len(keep)} games (all before {cutoff_date})")
    return model


def _tier(pick_prob):
    """Same tier thresholds as production, so HIGH/MED/LEAN buckets compare."""
    if pick_prob >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    if pick_prob >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "LEAN"


def predict(date_str, run_type="shadow"):
    """
    Generate challenger shadow picks for one date and store them in
    shadow_picks. Never writes to `picks`. Idempotent per (game, model_version)
    — re-running replaces that day's shadow rows.
    """
    print(f"=== shadow predict — {date_str} (model={MODEL_VERSION}) ===")
    model = _train_challenger(date_str)

    with get_db() as conn:
        games = conn.execute(
            "SELECT * FROM games WHERE game_date = ?", (date_str,)).fetchall()
        if not games:
            print(f"  no games on {date_str}")
            return []

        rows = []
        for g in games:
            feats = build_feature_vector(g, conn)
            if feats is None:
                continue
            X = pd.DataFrame([feats])[FEATURE_NAMES].fillna(0)
            home_prob = float(model.predict_proba(X)[0][1])

            if home_prob >= 0.5:
                winner, pick_prob = g["home_team"], home_prob
            else:
                winner, pick_prob = g["away_team"], 1 - home_prob
            conf = _tier(pick_prob)

            conn.execute("""
                INSERT OR REPLACE INTO shadow_picks
                (game_id, pick_date, run_type, model_version,
                 predicted_winner, home_win_prob, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (g["game_id"], date_str, run_type, MODEL_VERSION,
                  winner, round(home_prob, 4), conf))
            rows.append((g["away_team"], g["home_team"], winner, pick_prob, conf))

        print(f"  wrote {len(rows)} shadow picks:")
        for away, home, winner, prob, conf in rows:
            print(f"    {away}@{home:<5} -> {winner:<5} {prob:.0%}  {conf}")
        return rows


def score(date_str):
    """
    Score shadow picks for a date against final results. Mirrors
    scheduler.py:run_results but writes ONLY to shadow_picks.
    """
    print(f"=== shadow score — {date_str} ===")
    with get_db() as conn:
        finals = {
            g["game_id"]: g for g in conn.execute(
                "SELECT * FROM games WHERE game_date=? AND status='Final' "
                "AND winner IS NOT NULL", (date_str,))
        }
        picks = conn.execute(
            "SELECT * FROM shadow_picks WHERE pick_date=?", (date_str,)
        ).fetchall()
        if not picks:
            print(f"  no shadow picks on {date_str}")
            return

        scored = correct = 0
        for p in picks:
            g = finals.get(p["game_id"])
            if not g:
                continue
            actual = g["home_team"] if g["winner"] == "home" else g["away_team"]
            is_correct = 1 if p["predicted_winner"] == actual else 0
            conn.execute(
                "UPDATE shadow_picks SET actual_winner=?, correct=? "
                "WHERE game_id=? AND run_type=? AND model_version=?",
                (actual, is_correct, p["game_id"], p["run_type"],
                 p["model_version"]))
            scored += 1
            correct += is_correct
        pct = f"{correct/scored:.0%}" if scored else "N/A"
        print(f"  scored {scored} shadow picks: {correct}/{scored} ({pct})")


def report():
    """
    Champion (production `picks`) vs challenger (`shadow_picks`) on the games
    where BOTH have a scored prediction — the only fair comparison set.
    """
    print("=== champion vs challenger — scored-accuracy report ===")
    with get_db() as conn:
        # Champion: production picks, deduped to one row per game (prefer
        # lineup_lock over morning, the documented pattern).
        champ = {
            r["game_id"]: r for r in conn.execute("""
                SELECT p.game_id, p.predicted_winner, p.confidence, p.correct
                FROM picks p
                WHERE p.correct IS NOT NULL
                  AND p.run_type = (
                    SELECT p2.run_type FROM picks p2 WHERE p2.game_id = p.game_id
                    ORDER BY CASE p2.run_type WHEN 'lineup_lock' THEN 0 ELSE 1 END
                    LIMIT 1)
            """)
        }
        versions = [r[0] for r in conn.execute(
            "SELECT DISTINCT model_version FROM shadow_picks "
            "WHERE correct IS NOT NULL")]
        if not versions:
            print("  no scored shadow picks yet — run `predict` then `score` "
                  "for a few days first.")
            return

        for ver in versions:
            chal = {
                r["game_id"]: r for r in conn.execute(
                    "SELECT game_id, predicted_winner, confidence, correct "
                    "FROM shadow_picks WHERE model_version=? "
                    "AND correct IS NOT NULL", (ver,))
            }
            common = sorted(set(champ) & set(chal))
            if not common:
                print(f"\n  [{ver}] no overlap with scored production picks yet.")
                continue

            c_correct = sum(champ[g]["correct"] for g in common)
            x_correct = sum(chal[g]["correct"] for g in common)
            n = len(common)
            # tier breakdown for the challenger
            def tier_acc(picks_map, tier):
                t = [g for g in common if picks_map[g]["confidence"] == tier]
                if not t:
                    return None, 0
                return sum(picks_map[g]["correct"] for g in t) / len(t), len(t)

            print(f"\n  [{ver}] vs production — {n} games both models scored:")
            print(f"    champion (logreg):   {c_correct}/{n} = {c_correct/n:.1%}")
            print(f"    challenger ({ver}): {x_correct}/{n} = {x_correct/n:.1%}")
            delta = (x_correct - c_correct) / n
            print(f"    delta: {delta:+.1%} "
                  f"({'challenger ahead' if delta > 0 else 'champion ahead' if delta < 0 else 'tied'})")
            for tier in ("HIGH", "MEDIUM", "LEAN"):
                ca, cn = tier_acc(champ, tier)
                xa, xn = tier_acc(chal, tier)
                if cn or xn:
                    cs = f"{ca:.0%}({cn})" if ca is not None else "—"
                    xs = f"{xa:.0%}({xn})" if xa is not None else "—"
                    print(f"      {tier:<7} champ {cs:<10} challenger {xs}")
            if n < 30:
                print(f"    NOTE: n={n} is small — let the shadow test "
                      "accumulate 2-3 weeks before reading much into this.")


def main():
    ap = argparse.ArgumentParser(description="Champion/challenger shadow picks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pred = sub.add_parser("predict", help="generate shadow picks for a date")
    p_pred.add_argument("--date", default=date.today().isoformat())
    p_score = sub.add_parser("score", help="score shadow picks for a date")
    p_score.add_argument("--date", default=date.today().isoformat())
    sub.add_parser("report", help="champion vs challenger accuracy")
    args = ap.parse_args()

    if args.cmd == "predict":
        predict(args.date)
    elif args.cmd == "score":
        score(args.date)
    elif args.cmd == "report":
        report()


if __name__ == "__main__":
    main()
