"""Logistic regression model for MLB win probability prediction."""

import os
import pickle
import logging

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD, SEASON
from db import get_db
from model.features import build_training_features, build_feature_vector, FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def train_model(train_start=2022, train_end=2024, val_start=2025, val_end=2025):
    """Train logistic regression and validate on held-out data.

    Args:
        train_start/end: Year range for training data
        val_start/end: Year range for validation data

    Returns:
        Dict with training results (accuracy, feature importance, etc.)
    """
    print(f"\n{'='*55}")
    print(f"  MODEL TRAINING")
    print(f"{'='*55}")

    # Build training features
    print(f"\n  Training set: {train_start}-{train_end}")
    train_feats, train_labels, train_ids = build_training_features(train_start, train_end)

    if len(train_feats) < 100:
        print(f"  ERROR: Only {len(train_feats)} training samples. Need at least 100.")
        return None

    # Build validation features
    print(f"\n  Validation set: {val_start}-{val_end}")
    val_feats, val_labels, val_ids = build_training_features(val_start, val_end)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_feats)[FEATURE_NAMES]
    val_df = pd.DataFrame(val_feats)[FEATURE_NAMES]
    train_y = np.array(train_labels)
    val_y = np.array(val_labels)

    # Fill any NaN values with 0
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)

    print(f"\n  Training: {len(train_df)} games, Validation: {len(val_df)} games")

    # Scale features
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df)
    val_X = scaler.transform(val_df)

    # Train logistic regression
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    model.fit(train_X, train_y)

    # Training accuracy
    train_preds = model.predict(train_X)
    train_acc = accuracy_score(train_y, train_preds)

    # Validation accuracy
    val_probs = model.predict_proba(val_X)[:, 1]  # P(home win)
    val_preds = (val_probs >= 0.5).astype(int)
    val_acc = accuracy_score(val_y, val_preds)

    # Accuracy by confidence tier
    high_mask = (val_probs >= HIGH_CONFIDENCE_THRESHOLD) | (val_probs <= 1 - HIGH_CONFIDENCE_THRESHOLD)
    med_mask = ((val_probs >= MEDIUM_CONFIDENCE_THRESHOLD) | (val_probs <= 1 - MEDIUM_CONFIDENCE_THRESHOLD)) & ~high_mask

    # For accuracy, the "pick" is whichever side has >50%
    pick_correct = ((val_probs >= 0.5) & (val_y == 1)) | ((val_probs < 0.5) & (val_y == 0))

    high_acc = pick_correct[high_mask].mean() if high_mask.sum() > 0 else 0
    med_acc = pick_correct[med_mask].mean() if med_mask.sum() > 0 else 0
    lean_mask = ~high_mask & ~med_mask
    lean_acc = pick_correct[lean_mask].mean() if lean_mask.sum() > 0 else 0

    # Feature importance (logistic regression coefficients)
    coef_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)

    # Print results
    print(f"\n  {'─'*50}")
    print(f"  Training accuracy:    {train_acc:.1%}")
    print(f"  Validation accuracy:  {val_acc:.1%}")
    print(f"  {'─'*50}")
    print(f"  By confidence tier ({val_start}-{val_end}):")
    print(f"    HIGH  (>{HIGH_CONFIDENCE_THRESHOLD:.0%}):  {high_acc:.1%}  ({high_mask.sum()} games)")
    print(f"    MED   ({MEDIUM_CONFIDENCE_THRESHOLD:.0%}-{HIGH_CONFIDENCE_THRESHOLD:.0%}): {med_acc:.1%}  ({med_mask.sum()} games)")
    print(f"    LEAN  (<{MEDIUM_CONFIDENCE_THRESHOLD:.0%}):  {lean_acc:.1%}  ({lean_mask.sum()} games)")
    print(f"  {'─'*50}")
    print(f"  Feature importance:")
    for _, row in coef_df.iterrows():
        direction = "+" if row["coefficient"] > 0 else ""
        print(f"    {row['feature']:<22} {direction}{row['coefficient']:.4f}")
    print(f"  {'─'*50}")

    # Brier score
    brier = np.mean((val_probs - val_y) ** 2)
    print(f"  Brier score:          {brier:.4f} (lower is better, baseline ~0.25)")

    # Save model and scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Model saved to {MODEL_PATH}")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "high_acc": high_acc,
        "med_acc": med_acc,
        "lean_acc": lean_acc,
        "brier": brier,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "coefficients": dict(zip(FEATURE_NAMES, model.coef_[0])),
    }


def load_model():
    """Load trained model and scaler from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "No trained model found. Run 'python main.py init' first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_games(date_str, run_type="morning"):
    """Run predictions for all games on a given date.

    Args:
        date_str: Date in YYYY-MM-DD format
        run_type: "morning" or "lineup_lock"

    Returns:
        List of pick dicts with keys: game_id, home_team, away_team,
        predicted_winner, home_win_prob, confidence
    """
    model, scaler = load_model()

    with get_db() as conn:
        games = conn.execute(
            "SELECT * FROM games WHERE game_date = ?", (date_str,)
        ).fetchall()

        if not games:
            print(f"  No games found for {date_str}")
            return []

        picks = []
        for game in games:
            feats = build_feature_vector(game, conn)
            if feats is None:
                continue

            feat_df = pd.DataFrame([feats])[FEATURE_NAMES].fillna(0)
            feat_scaled = scaler.transform(feat_df)

            home_win_prob = model.predict_proba(feat_scaled)[0][1]

            # Detect opener/spot starter situation
            home_opener = _is_probable_opener(game["home_starter_id"], conn)
            away_opener = _is_probable_opener(game["away_starter_id"], conn)
            opener_flag = None

            if home_opener or away_opener:
                # Dampen probability toward 50% — we don't know who's really pitching
                # Shrink distance from 0.5 by 40%
                DAMPEN = 0.40
                dampened = 0.5 + (home_win_prob - 0.5) * (1 - DAMPEN)
                if home_opener and away_opener:
                    opener_flag = "both"
                elif home_opener:
                    opener_flag = "home"
                else:
                    opener_flag = "away"
                logger.info(f"  Opener detected ({opener_flag}): {game['away_team']} @ {game['home_team']} — "
                           f"prob {home_win_prob:.0%} → {dampened:.0%}")
                home_win_prob = dampened

            # Determine pick and confidence
            if home_win_prob >= 0.5:
                predicted_winner = game["home_team"]
                pick_prob = home_win_prob
            else:
                predicted_winner = game["away_team"]
                pick_prob = 1 - home_win_prob

            if pick_prob >= HIGH_CONFIDENCE_THRESHOLD:
                confidence = "HIGH"
            elif pick_prob >= MEDIUM_CONFIDENCE_THRESHOLD:
                confidence = "MEDIUM"
            else:
                confidence = "LEAN"

            pick = {
                "game_id": game["game_id"],
                "pick_date": date_str,
                "run_type": run_type,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_starter_name": game["home_starter_name"],
                "away_starter_name": game["away_starter_name"],
                "game_time": game["game_time"],
                "predicted_winner": predicted_winner,
                "home_win_prob": round(home_win_prob, 4),
                "pick_prob": round(pick_prob, 4),
                "confidence": confidence,
                "opener_flag": opener_flag,
            }
            picks.append(pick)

            # Save to DB
            conn.execute("""
                INSERT OR REPLACE INTO picks
                (game_id, pick_date, run_type, predicted_winner, home_win_prob, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                game["game_id"], date_str, run_type,
                predicted_winner, round(home_win_prob, 4), confidence,
            ))

        # Sort by confidence level then probability
        conf_order = {"HIGH": 0, "MEDIUM": 1, "LEAN": 2}
        picks.sort(key=lambda p: (conf_order.get(p["confidence"], 3), -p["pick_prob"]))

        return picks


def print_predictions(picks, date_str, run_type="morning"):
    """Print formatted prediction output."""
    if not picks:
        print(f"  No predictions for {date_str}")
        return

    run_label = "Morning Run" if run_type == "morning" else "Lineup Lock"
    print(f"\n{'='*60}")
    print(f"  MLB PICKS — {date_str} ({run_label})")
    print(f"{'='*60}")
    print(f"  {'Time':<10} {'Matchup':<18} {'Pick':<6} {'Prob':>5}  {'Conf'}")
    print(f"  {'─'*55}")

    high_count = 0
    for p in picks:
        prob_display = f"{p['pick_prob']:.0%}"
        conf = p["confidence"]
        if conf == "HIGH":
            high_count += 1
            conf_display = f"\033[32m{conf}\033[0m"  # Green
        elif conf == "MEDIUM":
            conf_display = f"\033[33m{conf}\033[0m"  # Yellow
        else:
            conf_display = conf

        matchup = f"{p['away_team']} @ {p['home_team']}"
        print(f"  {p['game_time']:<10} {matchup:<18} → {p['predicted_winner']:<5} {prob_display:>5}  {conf_display}")

    print(f"  {'─'*55}")
    print(f"  High-confidence picks: {high_count} | Total games: {len(picks)}")
    opener_games = [p for p in picks if p.get("opener_flag")]
    if opener_games:
        print(f"  ⚠ Opener detected in {len(opener_games)} game(s) — confidence dampened")
    print()


# Opener detection thresholds
OPENER_MAX_IP = 20      # Pitcher with <20 career IP is likely an opener
OPENER_MAX_GS = 5       # Pitcher with <5 career starts is likely an opener
OPENER_MIN_GP_RATIO = 3 # GP/GS ratio >3 means mostly a reliever


def _is_probable_opener(pitcher_id, conn):
    """Detect if a listed starter is likely an opener or spot starter.

    Uses the best available season (checks current season first, falls back
    to previous season) to avoid false positives early in the year when
    legitimate starters haven't accumulated IP yet.

    Checks:
    - Less than 20 IP across best available season
    - Less than 5 starts across best available season

    Returns True if the pitcher looks like an opener.
    """
    if not pitcher_id:
        return False

    # Get stats from all available seasons, most recent first
    rows = conn.execute("""
        SELECT season, innings_pitched, games_started
        FROM pitcher_stats
        WHERE player_id = ?
        ORDER BY season DESC
    """, (pitcher_id,)).fetchall()

    if not rows:
        return True  # Unknown pitcher — treat as opener

    # Use the best season for evaluation:
    # If the most recent season has enough data, use it.
    # Otherwise fall back to the previous season.
    best_ip = 0
    best_gs = 0
    for row in rows:
        ip = row["innings_pitched"] or 0
        gs = row["games_started"] or 0
        # Take the season with the most IP (handles early-season 2026 vs full 2025)
        if ip > best_ip:
            best_ip = ip
            best_gs = gs

    if best_ip < OPENER_MAX_IP and best_gs < OPENER_MAX_GS:
        return True

    return False
