"""Weekly retrain script.

Trains a fresh logistic regression model on 2022-prior_year + current_year-to-date,
validates against a held-out tail of recent games, and writes a report.

Usage:
    python -m model.retrain                  # full retrain + write report
    python -m model.retrain --dry-run        # validate only, don't write models

Outputs (when not --dry-run):
    model/trained_model.pkl                  # new production model
    model/scaler.pkl                         # new production scaler
    model/archive/trained_model_<ts>.pkl     # archived prior model
    model/archive/scaler_<ts>.pkl            # archived prior scaler
    model/retrain_report.json                # validation metrics + coefficients
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD, SEASON
from model.features import FEATURE_NAMES, build_training_features

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
REPORT_PATH = os.path.join(MODEL_DIR, "retrain_report.json")

# Validation gate: refuse to deploy if new model is meaningfully worse than the prior.
ACCURACY_REGRESSION_LIMIT = 0.02  # 2 percentage points


def _evaluate(model, scaler, val_feats, val_labels):
    """Run a model on a validation set and return accuracy + tier breakdown."""
    val_df = pd.DataFrame(val_feats)[FEATURE_NAMES].fillna(0)
    val_y = np.array(val_labels)
    val_X = scaler.transform(val_df)
    val_probs = model.predict_proba(val_X)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    overall = accuracy_score(val_y, val_preds)
    pick_correct = ((val_probs >= 0.5) & (val_y == 1)) | ((val_probs < 0.5) & (val_y == 0))
    high_mask = (val_probs >= HIGH_CONFIDENCE_THRESHOLD) | (val_probs <= 1 - HIGH_CONFIDENCE_THRESHOLD)
    med_mask = ((val_probs >= MEDIUM_CONFIDENCE_THRESHOLD) | (val_probs <= 1 - MEDIUM_CONFIDENCE_THRESHOLD)) & ~high_mask
    lean_mask = ~high_mask & ~med_mask
    return {
        "accuracy": float(overall),
        "high_acc": float(pick_correct[high_mask].mean()) if high_mask.sum() else None,
        "high_n": int(high_mask.sum()),
        "med_acc": float(pick_correct[med_mask].mean()) if med_mask.sum() else None,
        "med_n": int(med_mask.sum()),
        "lean_acc": float(pick_correct[lean_mask].mean()) if lean_mask.sum() else None,
        "lean_n": int(lean_mask.sum()),
        "brier": float(np.mean((val_probs - val_y) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser(description="Weekly model retrain")
    parser.add_argument("--dry-run", action="store_true", help="Don't write model files")
    parser.add_argument("--season", type=int, default=SEASON, help="Current season")
    parser.add_argument("--holdout-frac", type=float, default=0.2,
                        help="Fraction of current-season games to hold out for validation")
    args = parser.parse_args()

    print(f"=== Weekly retrain — {datetime.now().isoformat()} ===")
    print(f"Building training set: 2022-{args.season - 1} + {args.season}-to-date...")

    history_feats, history_labels, _ = build_training_features(2022, args.season - 1)
    current_feats, current_labels, _ = build_training_features(args.season, args.season)

    print(f"  History: {len(history_feats)} games")
    print(f"  Current season: {len(current_feats)} games")

    if len(current_feats) < 50:
        print(f"  WARNING: only {len(current_feats)} current-season games — too early to retrain meaningfully.")
        print(f"  Aborting. Try again once {args.season} has more data.")
        sys.exit(1)

    # Holdout = last N% of current season (chronologically the most recent — that's the realistic test)
    split = int(len(current_feats) * (1 - args.holdout_frac))
    train_feats = history_feats + current_feats[:split]
    train_labels = history_labels + current_labels[:split]
    holdout_feats = current_feats[split:]
    holdout_labels = current_labels[split:]

    print(f"  Training: {len(train_feats)}, Holdout: {len(holdout_feats)}")

    # Train candidate model
    train_df = pd.DataFrame(train_feats)[FEATURE_NAMES].fillna(0)
    train_y = np.array(train_labels)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df)
    candidate = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)
    candidate.fit(train_X, train_y)

    candidate_metrics = _evaluate(candidate, scaler, holdout_feats, holdout_labels)
    print(f"\nCandidate model on holdout ({len(holdout_feats)} games):")
    print(f"  Overall: {candidate_metrics['accuracy']:.1%}")
    if candidate_metrics["high_n"]:
        print(f"  HIGH:    {candidate_metrics['high_acc']:.1%} (n={candidate_metrics['high_n']})")
    if candidate_metrics["med_n"]:
        print(f"  MED:     {candidate_metrics['med_acc']:.1%} (n={candidate_metrics['med_n']})")
    if candidate_metrics["lean_n"]:
        print(f"  LEAN:    {candidate_metrics['lean_acc']:.1%} (n={candidate_metrics['lean_n']})")
    print(f"  Brier:   {candidate_metrics['brier']:.4f}")

    # Compare against current production model on the same holdout
    baseline_metrics = None
    regression_blocked = False
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            prod_model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            prod_scaler = pickle.load(f)
        baseline_metrics = _evaluate(prod_model, prod_scaler, holdout_feats, holdout_labels)
        print(f"\nProduction model on same holdout:")
        print(f"  Overall: {baseline_metrics['accuracy']:.1%}")
        delta = candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
        print(f"  Delta:   {delta:+.1%}")
        if delta < -ACCURACY_REGRESSION_LIMIT:
            print(f"\n  ⚠ REGRESSION GATE: candidate is {abs(delta):.1%} worse than production.")
            print(f"  Refusing to write new model files. Investigate before deploying.")
            regression_blocked = True

    coefficients = dict(zip(FEATURE_NAMES, [float(c) for c in candidate.coef_[0]]))
    print(f"\nCandidate coefficients:")
    for name, coef in coefficients.items():
        print(f"  {name:<22} {coef:+.4f}")

    # Final retrain on ALL data (no holdout) for production deploy
    if args.dry_run or regression_blocked:
        all_feats = train_feats
        all_labels = train_labels
        final_model = candidate
        final_scaler = scaler
        action = "dry-run" if args.dry_run else "blocked-by-regression-gate"
    else:
        all_feats = history_feats + current_feats
        all_labels = history_labels + current_labels
        all_df = pd.DataFrame(all_feats)[FEATURE_NAMES].fillna(0)
        all_y = np.array(all_labels)
        final_scaler = StandardScaler()
        all_X = final_scaler.fit_transform(all_df)
        final_model = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)
        final_model.fit(all_X, all_y)
        action = "deployed"
        print(f"\nFinal model retrained on all {len(all_feats)} games (no holdout) for production.")

        # Archive the existing prod model before overwriting
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(MODEL_PATH):
            shutil.copy(MODEL_PATH, os.path.join(ARCHIVE_DIR, f"trained_model_{ts}.pkl"))
        if os.path.exists(SCALER_PATH):
            shutil.copy(SCALER_PATH, os.path.join(ARCHIVE_DIR, f"scaler_{ts}.pkl"))
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(final_model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(final_scaler, f)
        print(f"  Wrote {MODEL_PATH} and {SCALER_PATH} (prior versions archived to {ARCHIVE_DIR}/)")

    # Write report regardless of action — useful for PR description / audit trail
    report = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "season": args.season,
        "training_set_size": len(all_feats),
        "holdout_size": len(holdout_feats),
        "candidate_metrics": candidate_metrics,
        "production_metrics": baseline_metrics,
        "accuracy_delta": (
            candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
            if baseline_metrics else None
        ),
        "coefficients": coefficients,
        "regression_gate_limit": ACCURACY_REGRESSION_LIMIT,
        "regression_blocked": regression_blocked,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {REPORT_PATH}")

    if regression_blocked:
        sys.exit(2)  # Distinct exit code so the workflow can detect this case


if __name__ == "__main__":
    main()
