#!/usr/bin/env python3
"""Smart scheduler for MLB Picker — handles morning, lineup lock, and results runs.

Lineup lock runs only process games starting in the next ~3 hours whose
lineups are newly confirmed. Each game is only locked once.

Usage:
    python scheduler.py morning          # 8 AM: team-level preview picks
    python scheduler.py lineup_lock      # 11 AM/2 PM/5 PM/8 PM: lineup-aware updates
    python scheduler.py results          # 1 AM: score yesterday's picks
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from config import SEASON
from db import init_db, seed_priors, get_db
from data.mlb_api import get_schedule, get_pitcher_season_stats, get_all_team_records
from data.fangraphs import refresh_fangraphs_stats
from data.lineups import fetch_and_cache_lineup, LINEUP_WEAK_THRESHOLD, LINEUP_DAMPEN_FACTOR
from model.predict import predict_games, print_predictions, load_model, _is_probable_opener
from model.features import (
    build_feature_vector, FEATURE_NAMES,
    compute_signals, away_overconfidence_damping,
)
from output.dashboard import generate_dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_morning(date_str=None):
    """Morning run: refresh all data, generate team-level predictions."""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  MORNING RUN — {date_str}")
    print(f"{'='*55}")

    init_db()
    seed_priors()

    # Import and run the full refresh from main
    from main import refresh_data
    refresh_data(date_str)

    # Run predictions
    picks = predict_games(date_str, run_type="morning")
    print_predictions(picks, date_str, "morning")

    # Generate dashboard
    generate_dashboard(date_str)

    high = sum(1 for p in picks if p["confidence"] == "HIGH")
    return f"{len(picks)} picks ready ({high} HIGH confidence)"


def run_lineup_lock(date_str=None):
    """Lineup lock run: fetch confirmed lineups and update predictions for upcoming games.

    Only processes games starting in the next ~3 hours that haven't been locked yet.
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")

    # Convert UTC now to ET for proper window comparison
    from data.mlb_api import _get_et_offset
    now_utc = datetime.utcnow()
    now_et = now_utc - timedelta(hours=_get_et_offset(now_utc))

    print(f"\n{'='*55}")
    print(f"  LINEUP LOCK — {date_str} ({now_et.strftime('%I:%M %p')} ET)")
    print(f"{'='*55}")

    init_db()

    with get_db() as conn:
        # Get today's games
        games = conn.execute(
            "SELECT * FROM games WHERE game_date = ?", (date_str,)
        ).fetchall()

        if not games:
            print("  No games today.")
            return "No games today"

        # Determine which games to process:
        # Games starting in the next 3 hours that don't have a lineup_lock pick yet
        window_end = now_et + timedelta(hours=3)
        games_to_lock = []

        for g in games:
            # Check if already locked
            existing = conn.execute(
                "SELECT 1 FROM picks WHERE game_id = ? AND run_type = 'lineup_lock'",
                (g["game_id"],)
            ).fetchone()
            if existing:
                continue

            # Parse game time (stored as 24-hour HH:MM in ET)
            game_time_str = g["game_time"] or ""
            game_dt = _parse_game_time(game_time_str, date_str)

            if game_dt and game_dt <= window_end:
                games_to_lock.append(g)

        if not games_to_lock:
            print("  No unlocked games in the next 3 hours.")
            # Still regenerate dashboard in case results came in
            generate_dashboard(date_str)
            return "No games to lock"

        print(f"  {len(games_to_lock)} games to lock in this window\n")

        # Fetch lineups and batter splits for each game
        model, scaler = load_model()
        import pandas as pd
        locked = 0
        updated_picks = []

        for g in games_to_lock:
            lineup_data = fetch_and_cache_lineup(g["game_id"], conn, SEASON)

            if not lineup_data:
                print(f"  {g['away_team']} @ {g['home_team']}: lineups not available, using morning pick")
                morning = conn.execute(
                    "SELECT * FROM picks WHERE game_id = ? AND run_type = 'morning'",
                    (g["game_id"],)
                ).fetchone()
                if morning:
                    conn.execute("""
                        INSERT OR REPLACE INTO picks
                        (game_id, pick_date, run_type, predicted_winner, home_win_prob, confidence, opener_flag, pick_flipped)
                        VALUES (?, ?, 'lineup_lock', ?, ?, ?, ?, ?)
                    """, (g["game_id"], date_str, morning["predicted_winner"],
                          morning["home_win_prob"], morning["confidence"],
                          morning["opener_flag"] if "opener_flag" in morning.keys() else None,
                          morning["pick_flipped"] if "pick_flipped" in morning.keys() else 0))
                    locked += 1
                continue

            # Build features
            feats = build_feature_vector(g, conn)
            if feats is None:
                continue

            home_ld = lineup_data.get("home", {})
            away_ld = lineup_data.get("away", {})
            if home_ld.get("lineup_ops") and away_ld.get("lineup_ops"):
                logger.info(f"  {g['away_team']} @ {g['home_team']}: lineup OPS "
                           f"{away_ld['lineup_ops']:.3f} vs {home_ld['lineup_ops']:.3f}")

            # Predict
            feat_df = pd.DataFrame([feats])[FEATURE_NAMES].fillna(0)
            feat_scaled = scaler.transform(feat_df)
            home_win_prob = model.predict_proba(feat_scaled)[0][1]

            # Re-detect openers at lineup_lock time and dampen the raw model probability
            # toward 50%. Mirrors model.predict.predict_games() so morning and lineup_lock
            # are calibrated the same way.
            home_opener = _is_probable_opener(g["home_starter_id"], conn)
            away_opener = _is_probable_opener(g["away_starter_id"], conn)
            opener_flag = None
            if home_opener or away_opener:
                OPENER_DAMPEN = 0.40
                pre_dampen = home_win_prob
                home_win_prob = 0.5 + (home_win_prob - 0.5) * (1 - OPENER_DAMPEN)
                if home_opener and away_opener:
                    opener_flag = "both"
                elif home_opener:
                    opener_flag = "home"
                else:
                    opener_flag = "away"
                logger.info(f"    Opener detected ({opener_flag}): "
                            f"prob {pre_dampen:.0%} → {home_win_prob:.0%}")

            # Away-team overconfidence correction (see model/signal_damping_experiment.py)
            signals = compute_signals(g, conn)
            pre_signal = home_win_prob
            home_win_prob = away_overconfidence_damping(home_win_prob, signals)
            if pre_signal != home_win_prob:
                logger.info(f"    Away-overconfidence damping: "
                            f"prob {pre_signal:.0%} → {home_win_prob:.0%}")

            # Lineup-OPS nudge: shift home_win_prob toward the team with the stronger
            # lineup-vs-handedness OPS. Conservative K=0.40 with ±4pp cap so a single
            # bad OPS data point can't flip a HIGH pick. Stack position is after the
            # signal-based dampening so it can move probability in either direction,
            # before the weakened-lineup shrinkage which still gets the final say.
            home_ops = home_ld.get("lineup_ops")
            away_ops = away_ld.get("lineup_ops")
            if home_ops and away_ops:
                ops_gap = home_ops - away_ops
                nudge = max(-0.04, min(0.04, ops_gap * 0.40))
                if abs(nudge) >= 0.005:
                    pre_nudge = home_win_prob
                    home_win_prob = max(0.05, min(0.95, home_win_prob + nudge))
                    logger.info(f"    Lineup-OPS nudge: gap={ops_gap:+.3f} "
                                f"nudge={nudge:+.1%} → {pre_nudge:.0%} → {home_win_prob:.0%}")

            # Dampen for weakened lineups
            lineup_flags = []
            for side, ld, team in [("home", home_ld, g["home_team"]), ("away", away_ld, g["away_team"])]:
                if ld.get("is_weakened"):
                    pct = ld["lineup_pct_change"]
                    missing = ld.get("missing_regulars", [])
                    missing_names = ", ".join(m["player_name"] for m in missing[:2])
                    logger.info(f"    {team} lineup weakened ({pct:+.1%})"
                               f"{' — missing: ' + missing_names if missing_names else ''}")
                    lineup_flags.append(f"{team} weakened")

            if lineup_flags:
                dampened = 0.5 + (home_win_prob - 0.5) * (1 - LINEUP_DAMPEN_FACTOR)
                logger.info(f"    Lineup dampening: {home_win_prob:.0%} → {dampened:.0%}")
                home_win_prob = dampened

            from config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
            if home_win_prob >= 0.5:
                predicted_winner = g["home_team"]
                pick_prob = home_win_prob
            else:
                predicted_winner = g["away_team"]
                pick_prob = 1 - home_win_prob

            if pick_prob >= HIGH_CONFIDENCE_THRESHOLD:
                confidence = "HIGH"
            elif pick_prob >= MEDIUM_CONFIDENCE_THRESHOLD:
                confidence = "MEDIUM"
            else:
                confidence = "LEAN"

            # LEAN-flip (see model/predict.py for rationale)
            pick_flipped = 0
            if confidence == "LEAN":
                home_win_prob = 1 - home_win_prob
                if home_win_prob >= 0.5:
                    predicted_winner = g["home_team"]
                    pick_prob = home_win_prob
                else:
                    predicted_winner = g["away_team"]
                    pick_prob = 1 - home_win_prob
                pick_flipped = 1
                logger.info(f"    LEAN-flip: contrarian pick {predicted_winner} @ {pick_prob:.0%}")

            # Check if pick changed from morning
            morning = conn.execute(
                "SELECT predicted_winner FROM picks WHERE game_id = ? AND run_type = 'morning'",
                (g["game_id"],)
            ).fetchone()
            changed = morning and morning["predicted_winner"] != predicted_winner
            change_marker = " ** CHANGED **" if changed else ""

            print(f"  {g['away_team']} @ {g['home_team']}: {predicted_winner} {pick_prob:.0%} {confidence}{change_marker}")

            conn.execute("""
                INSERT OR REPLACE INTO picks
                (game_id, pick_date, run_type, predicted_winner, home_win_prob, confidence, opener_flag, pick_flipped)
                VALUES (?, ?, 'lineup_lock', ?, ?, ?, ?, ?)
            """, (g["game_id"], date_str, predicted_winner, round(home_win_prob, 4), confidence, opener_flag, pick_flipped))
            locked += 1

    # Regenerate dashboard
    generate_dashboard(date_str)

    print(f"\n  Locked {locked} games")
    return f"{locked} games locked"


def run_results(date_str=None):
    """Results run: fetch final scores and mark picks correct/incorrect."""
    # Score yesterday's games by default
    date_str = date_str or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  RESULTS — {date_str}")
    print(f"{'='*55}")

    init_db()

    from data.mlb_api import get_game_results

    results = get_game_results(date_str)
    if not results:
        print(f"  No final results for {date_str}")
        return "No results"

    with get_db() as conn:
        scored = 0
        correct = 0
        total = 0

        for g in results:
            # Update game with final score
            conn.execute("""
                UPDATE games SET home_score = ?, away_score = ?, winner = ?, status = 'Final'
                WHERE game_id = ?
            """, (g["home_score"], g["away_score"], g["winner"], g["game_id"]))

            actual_winner = g["home_team"] if g["winner"] == "home" else g["away_team"]

            # Score all picks for this game (morning and lineup_lock)
            picks = conn.execute(
                "SELECT * FROM picks WHERE game_id = ?", (g["game_id"],)
            ).fetchall()

            for p in picks:
                is_correct = 1 if p["predicted_winner"] == actual_winner else 0
                conn.execute("""
                    UPDATE picks SET actual_winner = ?, correct = ?
                    WHERE game_id = ? AND run_type = ?
                """, (actual_winner, is_correct, g["game_id"], p["run_type"]))

                # Only count the best run type (lineup_lock > morning)
                if p["run_type"] == "lineup_lock" or (p["run_type"] == "morning" and not any(
                    pp["run_type"] == "lineup_lock" for pp in picks
                )):
                    total += 1
                    if is_correct:
                        correct += 1
                    scored += 1

    pct = f"{correct/total:.0%}" if total > 0 else "N/A"
    print(f"\n  Scored {scored} games: {correct}/{total} correct ({pct})")

    # Regenerate dashboard for the scored date
    generate_dashboard(date_str)

    return f"{correct}/{total} correct ({pct})"


def _parse_game_time(time_str, date_str):
    """Parse a game time into a datetime. Handles both 24-hour 'HH:MM' and 12-hour 'HH:MM PM ET'."""
    if not time_str:
        return None
    try:
        time_str = time_str.replace(" ET", "").strip()
        # Try 24-hour format first (current storage format)
        try:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            pass
        # Fall back to 12-hour format (legacy)
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %I:%M %p")
    except (ValueError, AttributeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="MLB Picker Scheduler")
    parser.add_argument("mode", choices=["morning", "lineup_lock", "results"],
                        help="Run mode")
    parser.add_argument("--date", default=None,
                        help="Date override (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.mode == "morning":
        result = run_morning(args.date)
    elif args.mode == "lineup_lock":
        result = run_lineup_lock(args.date)
    elif args.mode == "results":
        result = run_results(args.date)

    print(f"\n  Done: {result}")


if __name__ == "__main__":
    main()
