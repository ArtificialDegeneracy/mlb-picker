"""
XGBoost candidate experiment — Path B model comparison harness.

WHAT THIS IS
------------
A STANDALONE harness that trains logistic regression (the current production
model type) and XGBoost on identical data and compares them head-to-head using
the same metrics retrain.py's regression gate uses.

It does NOT touch the production path. model/predict.py and model/retrain.py are
unchanged; this writes only to model/xgb_experiment_report.json. If XGBoost wins
here, wiring it into production is a separate, deliberate follow-up (the
integration path is spelled out in docs/balldontlie_evaluation.md).

WHY A SEPARATE HARNESS
----------------------
The 5-feature linear model has hit its ceiling (TODO.md "Path C"). XGBoost can
represent interactions and sample-size gating a linear model structurally
cannot — which is exactly what the balldontlie candidate features in
model/feature_staging.py need (most are tagged 'tree': they only pay off under
a tree model; testing them in logreg would falsely fail them, the 2026-05-18
trap). This harness is where those features get a fair test.

DRY-RUN (works TODAY, before the balldontlie trial)
---------------------------------------------------
With --dry-run-synthetic the harness runs on the 5 production features plus the
staged balldontlie features. The bdl_* cache is empty pre-trial, so the staged
columns are constant 0.0 — they add no signal, but the run PROVES the
logreg-vs-XGBoost plumbing works end to end on real game data. Once the trial
populates the bdl_* tables, drop the flag and the SAME code trains on real
staged features. No code change between dry-run and live.

USAGE
-----
    # Pre-trial: prove the plumbing, baseline logreg-vs-XGBoost on 5 features.
    python -m model.xgb_experiment --dry-run-synthetic

    # Post-trial: real staged features included.
    python -m model.xgb_experiment

    # Test a specific feature subset (grid-testing which features earn a slot):
    python -m model.xgb_experiment --features arsenal_matchup_score,h2h_ops_diff

    # Tune XGBoost depth/estimators.
    python -m model.xgb_experiment --max-depth 4 --n-estimators 300

OUTPUT
------
    model/xgb_experiment_report.json  — full metrics, both models, feature list.
    Console — side-by-side accuracy / tier / Brier, plus XGBoost feature
    importances so dead features are visible.
"""

import argparse
import json
import os
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
from db import get_db
from model.features import FEATURE_NAMES, build_training_features
from model.feature_staging import CANDIDATE_FEATURES, stage_features, cache_coverage

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(MODEL_DIR, "xgb_experiment_report.json")

# Same regression-gate threshold retrain.py uses, so "did XGBoost win" is
# judged on the same bar as a weekly retrain.
ACCURACY_REGRESSION_LIMIT = 0.02


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------

def build_dataset(staged_feature_names, season):
    """
    Build the full feature matrix: the 5 production features + the requested
    staged balldontlie features, for every Final game 2022..season.

    Returns (df, labels, dates) where df columns are
    FEATURE_NAMES + staged_feature_names, in that fixed order.
    """
    print(f"  building production features (2022-{season})...")
    feats, labels, game_ids, dates = build_training_features(
        2022, season, return_dates=True)

    base_df = pd.DataFrame(feats)[FEATURE_NAMES]

    if not staged_feature_names:
        return base_df.fillna(0), labels, dates

    # Stage the balldontlie candidate features. One pass over the same games,
    # joined by game_id so rows line up with the production matrix.
    print(f"  staging {len(staged_feature_names)} candidate feature(s) "
          f"for {len(game_ids)} games...")
    staged_rows = []
    with get_db() as conn:
        gid_set = set(game_ids)
        games_by_id = {
            g["game_id"]: g for g in conn.execute(
                "SELECT * FROM games WHERE status='Final' "
                "AND game_date >= '2022-01-01'")
            if g["game_id"] in gid_set
        }
        for gid in game_ids:
            g = games_by_id.get(gid)
            if g is None:
                staged_rows.append({n: 0.0 for n in staged_feature_names})
                continue
            yr = int(g["game_date"][:4])
            staged_rows.append(
                stage_features(g, conn, yr, staged_feature_names))

    staged_df = pd.DataFrame(staged_rows)[staged_feature_names]
    full = pd.concat([base_df.reset_index(drop=True),
                      staged_df.reset_index(drop=True)], axis=1)
    return full.fillna(0), labels, dates


def time_walked_split(df, labels, dates, holdout_frac=0.2):
    """
    Chronological holdout — the last `holdout_frac` of games by date. Both
    models train on the earlier games and are scored on the same later games,
    so neither has a leakage advantage (the comparison is apples-to-apples;
    this is a within-run fair split, simpler than retrain.py's mtime-based
    production-vs-candidate walk because here BOTH models are candidates).
    """
    order = np.argsort(dates)
    df = df.iloc[order].reset_index(drop=True)
    labels = np.array(labels)[order]
    n = len(df)
    split = int(n * (1 - holdout_frac))
    return (df.iloc[:split], labels[:split],
            df.iloc[split:], labels[split:],
            sorted(dates)[split] if split < n else None)


# ---------------------------------------------------------------------------
# model training
# ---------------------------------------------------------------------------

def train_logreg(train_X_raw, train_y):
    """Logistic regression — same config as predict.py / retrain.py."""
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X_raw)
    model = LogisticRegression(class_weight="balanced", max_iter=1000,
                               C=0.5, random_state=42)
    model.fit(train_X, train_y)
    return model, scaler


def train_xgboost(train_X_raw, train_y, max_depth, n_estimators, learning_rate):
    """
    XGBoost classifier. No scaler — trees are scale-invariant.

    Defaults are deliberately CONSERVATIVE: with only ~7-10k games and
    5-15 features, deep trees overfit instantly. Shallow trees (depth 3-4) +
    a moderate learning rate + subsampling keep it honest. Tune via CLI flags.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.exit("xgboost not installed — `pip install xgboost` "
                 "(or add it to requirements.txt).")

    pos = int(np.sum(train_y))
    neg = len(train_y) - pos
    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=0.8,            # row subsampling — regularization
        colsample_bytree=0.8,     # column subsampling — regularization
        min_child_weight=5,       # don't split on tiny leaves — regularization
        reg_lambda=1.0,
        scale_pos_weight=(neg / pos) if pos else 1.0,  # class balance
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_X_raw, train_y)
    return model


# ---------------------------------------------------------------------------
# evaluation — mirrors retrain.py:_evaluate so metrics are comparable
# ---------------------------------------------------------------------------

def evaluate(probs, val_y):
    """Accuracy + confidence-tier breakdown + Brier from P(home win)."""
    val_y = np.array(val_y)
    preds = (probs >= 0.5).astype(int)
    overall = accuracy_score(val_y, preds)
    pick_correct = ((probs >= 0.5) & (val_y == 1)) | ((probs < 0.5) & (val_y == 0))
    high = (probs >= HIGH_CONFIDENCE_THRESHOLD) | (probs <= 1 - HIGH_CONFIDENCE_THRESHOLD)
    med = ((probs >= MEDIUM_CONFIDENCE_THRESHOLD) | (probs <= 1 - MEDIUM_CONFIDENCE_THRESHOLD)) & ~high
    lean = ~high & ~med
    return {
        "accuracy": float(overall),
        "high_acc": float(pick_correct[high].mean()) if high.sum() else None,
        "high_n": int(high.sum()),
        "med_acc": float(pick_correct[med].mean()) if med.sum() else None,
        "med_n": int(med.sum()),
        "lean_acc": float(pick_correct[lean].mean()) if lean.sum() else None,
        "lean_n": int(lean.sum()),
        "brier": float(np.mean((probs - val_y) ** 2)),
    }


def _print_metrics(label, m):
    print(f"\n  {label}")
    print(f"    Overall accuracy: {m['accuracy']:.1%}")
    if m["high_n"]:
        print(f"    HIGH  {m['high_acc']:.1%}  (n={m['high_n']})")
    if m["med_n"]:
        print(f"    MED   {m['med_acc']:.1%}  (n={m['med_n']})")
    if m["lean_n"]:
        print(f"    LEAN  {m['lean_acc']:.1%}  (n={m['lean_n']})")
    print(f"    Brier {m['brier']:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="XGBoost vs logreg comparison")
    ap.add_argument("--dry-run-synthetic", action="store_true",
                    help="Run before the balldontlie trial: staged features "
                         "are constant 0.0 (empty cache). Proves the plumbing.")
    ap.add_argument("--features", type=str, default="all",
                    help="Comma-separated staged feature names, 'all', or "
                         "'none' for production-5-only. See "
                         "model/feature_staging.py:CANDIDATE_FEATURES.")
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    print(f"=== XGBoost experiment — {datetime.now().isoformat()} ===")

    # Resolve which staged features to include.
    if args.features == "none":
        staged = []
    elif args.features == "all":
        staged = list(CANDIDATE_FEATURES.keys())
    else:
        staged = [f.strip() for f in args.features.split(",") if f.strip()]
        unknown = [f for f in staged if f not in CANDIDATE_FEATURES]
        if unknown:
            sys.exit(f"unknown staged feature(s): {unknown}")

    # Cache-coverage check: tell the user whether staged features are real.
    with get_db() as conn:
        cov = cache_coverage(conn)
    cache_empty = all(v == 0 for v in cov.values() if isinstance(v, int))
    print(f"\n  bdl_* cache coverage: {cov}")
    if staged and cache_empty and not args.dry_run_synthetic:
        print("\n  WARNING: staged features requested but the bdl_* cache is "
              "EMPTY.\n  The staged columns will be constant 0.0 and add no "
              "signal. Re-run\n  with --dry-run-synthetic to acknowledge this, "
              "or load the trial data first.")
        sys.exit(1)
    if args.dry_run_synthetic:
        print("\n  DRY-RUN-SYNTHETIC: staged features are constant 0.0 "
              "(empty cache).\n  This run proves the logreg-vs-XGBoost "
              "plumbing only — not feature value.")

    print(f"\n  production features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    print(f"  staged features ({len(staged)}): {staged or '(none)'}")

    # Build dataset.
    df, labels, dates = build_dataset(staged, args.season)
    all_features = list(df.columns)
    print(f"\n  dataset: {len(df)} games x {len(all_features)} features")

    train_X, train_y, val_X, val_y, cutoff = time_walked_split(
        df, labels, dates, args.holdout_frac)
    print(f"  chronological split: {len(train_X)} train / {len(val_X)} "
          f"holdout (holdout starts {cutoff})")
    if len(val_X) < 30:
        sys.exit(f"holdout too small ({len(val_X)}) — need 30+ for a "
                 "meaningful comparison.")

    # --- train both models on identical data ---
    print("\n  training logistic regression (baseline)...")
    logreg, scaler = train_logreg(train_X, train_y)
    logreg_probs = logreg.predict_proba(scaler.transform(val_X))[:, 1]
    logreg_metrics = evaluate(logreg_probs, val_y)

    print(f"  training XGBoost (max_depth={args.max_depth}, "
          f"n_estimators={args.n_estimators}, lr={args.learning_rate})...")
    xgb = train_xgboost(train_X, train_y, args.max_depth,
                        args.n_estimators, args.learning_rate)
    xgb_probs = xgb.predict_proba(val_X)[:, 1]
    xgb_metrics = evaluate(xgb_probs, val_y)

    # --- compare ---
    print("\n" + "=" * 60)
    print("  HEAD-TO-HEAD on the same chronological holdout")
    print("=" * 60)
    _print_metrics("Logistic regression (current production type):", logreg_metrics)
    _print_metrics("XGBoost (candidate):", xgb_metrics)

    delta = xgb_metrics["accuracy"] - logreg_metrics["accuracy"]
    brier_delta = xgb_metrics["brier"] - logreg_metrics["brier"]
    print(f"\n  XGBoost - logreg accuracy delta: {delta:+.1%}")
    print(f"  XGBoost - logreg Brier delta:    {brier_delta:+.4f} "
          f"({'better' if brier_delta < 0 else 'worse'})")

    if args.dry_run_synthetic:
        verdict = ("DRY-RUN: plumbing verified. Feature value is UNTESTED "
                   "until the bdl_* cache holds real trial data.")
    elif delta > ACCURACY_REGRESSION_LIMIT:
        verdict = (f"XGBoost beats logreg by >{ACCURACY_REGRESSION_LIMIT:.0%} "
                   "— candidate for production. Proceed to the integration "
                   "path in docs/balldontlie_evaluation.md.")
    elif delta < -ACCURACY_REGRESSION_LIMIT:
        verdict = ("XGBoost is materially WORSE — keep logreg. The feature set "
                   "or hyperparameters need work.")
    else:
        verdict = ("within noise — no clear winner. Try richer features or "
                   "tune depth/estimators before committing to a swap.")
    print(f"\n  VERDICT: {verdict}")

    # --- XGBoost feature importance — which features actually did work ---
    importances = sorted(
        zip(all_features, (float(x) for x in xgb.feature_importances_)),
        key=lambda kv: kv[1], reverse=True)
    print("\n  XGBoost feature importance (gain-normalized):")
    for name, imp in importances:
        staged_tag = ""
        if name in CANDIDATE_FEATURES:
            staged_tag = f"  [staged/{CANDIDATE_FEATURES[name]['fit']}]"
        bar = "#" * int(imp * 50)
        print(f"    {name:<28} {imp:.4f} {bar}{staged_tag}")
    if staged and not args.dry_run_synthetic:
        dead = [n for n, i in importances
                if n in CANDIDATE_FEATURES and i < 0.01]
        if dead:
            print(f"\n  Near-zero-importance staged features (drop these): "
                  f"{dead}")

    # --- report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "dry_run_synthetic" if args.dry_run_synthetic else "live",
        "season": args.season,
        "production_features": FEATURE_NAMES,
        "staged_features": staged,
        "n_games": len(df),
        "n_train": len(train_X),
        "n_holdout": len(val_X),
        "holdout_cutoff": cutoff,
        "bdl_cache_coverage": cov,
        "xgb_hyperparams": {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
        },
        "logreg_metrics": logreg_metrics,
        "xgb_metrics": xgb_metrics,
        "accuracy_delta": delta,
        "brier_delta": brier_delta,
        "xgb_feature_importance": dict(importances),
        "verdict": verdict,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
