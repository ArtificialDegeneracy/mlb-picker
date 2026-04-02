"""Feature engineering for the MLB win probability model."""

import logging
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PRIOR_WEIGHT_BY_MONTH, WIN_TOTAL_PRIORS, SEASON, PARK_FACTORS, VENUE_TO_TEAM
from db import get_db

logger = logging.getLogger(__name__)

# Feature names in order — must match what predict.py expects
FEATURE_NAMES = [
    "fip_diff",            # home starter FIP - away starter FIP (negative = home advantage)
    "home_flag",           # always 1 (model learns home field advantage weight)
    "team_quality_diff",   # blended win% difference (home - away)
    "bullpen_era_diff",    # home bullpen ERA - away bullpen ERA (negative = home advantage)
    "platoon_wrc_diff",    # home team wRC+ vs starter hand - away team wRC+ vs starter hand
    "park_factor",         # park run environment (>1 = hitter friendly, <1 = pitcher friendly)
]


def build_feature_vector(game_row, conn=None):
    """Build a feature vector for a single game.

    Args:
        game_row: sqlite3.Row or dict with game data
        conn: Optional DB connection (creates one if not provided)

    Returns:
        Dict of feature_name → float value, or None if critical data is missing.
    """
    own_conn = conn is None
    if own_conn:
        from db import get_db
        db = get_db()
        conn = db.__enter__()

    try:
        features = {}

        # 1. FIP differential
        home_fip = _get_pitcher_fip(game_row["home_starter_id"], game_row["home_team"], conn)
        away_fip = _get_pitcher_fip(game_row["away_starter_id"], game_row["away_team"], conn)

        if home_fip is not None and away_fip is not None:
            features["fip_diff"] = home_fip - away_fip
        else:
            features["fip_diff"] = 0.0

        # 2. Home field flag
        features["home_flag"] = 1.0

        # 3. Team quality differential (prior-blended win%)
        game_date = game_row["game_date"]
        month = int(game_date[5:7]) if game_date else 6
        home_quality = _get_team_quality(game_row["home_team"], game_date, month, conn)
        away_quality = _get_team_quality(game_row["away_team"], game_date, month, conn)
        features["team_quality_diff"] = home_quality - away_quality

        # 4. Bullpen ERA differential
        home_bp = _get_bullpen_era(game_row["home_team"], conn)
        away_bp = _get_bullpen_era(game_row["away_team"], conn)
        if home_bp is not None and away_bp is not None:
            features["bullpen_era_diff"] = home_bp - away_bp
        else:
            features["bullpen_era_diff"] = 0.0

        # 5. Platoon wRC+ differential (team batting vs opposing starter's hand)
        home_platoon = _get_platoon_wrc(game_row["home_team"], game_row["away_starter_id"], conn)
        away_platoon = _get_platoon_wrc(game_row["away_team"], game_row["home_starter_id"], conn)
        if home_platoon is not None and away_platoon is not None:
            features["platoon_wrc_diff"] = home_platoon - away_platoon
        else:
            features["platoon_wrc_diff"] = 0.0

        # 6. Park factor
        features["park_factor"] = _get_park_factor(game_row)

        return features

    finally:
        if own_conn:
            db.__exit__(None, None, None)


def build_training_features(start_year=2022, end_year=2025):
    """Build feature matrix and labels for all historical games.

    Returns:
        (features_list, labels_list, game_ids_list)
    """
    features_list = []
    labels_list = []
    game_ids = []

    with get_db() as conn:
        games = conn.execute("""
            SELECT * FROM games
            WHERE status = 'Final'
              AND game_date >= ? AND game_date <= ?
              AND winner IS NOT NULL
            ORDER BY game_date
        """, (f"{start_year}-01-01", f"{end_year}-12-31")).fetchall()

        print(f"  Building features for {len(games)} games...")
        skipped = 0

        for i, game in enumerate(games):
            feats = build_feature_vector(game, conn)
            if feats is None:
                skipped += 1
                continue

            features_list.append(feats)
            labels_list.append(1 if game["winner"] == "home" else 0)
            game_ids.append(game["game_id"])

            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{len(games)} games processed...")

        if skipped:
            print(f"    Skipped {skipped} games with missing data")

    return features_list, labels_list, game_ids


# --- Helper functions ---

def _get_pitcher_fip(player_id, team_abbr, conn):
    """Get a pitcher's blended FIP, weighting current + previous season by IP.

    Early in the season, a pitcher's 2026 FIP (based on 5 IP) is unreliable.
    We blend it with their 2025 FIP weighted by innings pitched:
      blended = (fip_2026 * ip_2026 + fip_2025 * ip_2025) / (ip_2026 + ip_2025)

    This naturally shifts toward current-season FIP as IP accumulates.
    By mid-season (~100 IP), the current year dominates.
    """
    if player_id:
        rows = conn.execute(
            "SELECT season, fip, innings_pitched FROM pitcher_stats WHERE player_id = ? AND fip IS NOT NULL ORDER BY season DESC LIMIT 2",
            (player_id,)
        ).fetchall()

        if rows:
            total_ip = 0
            weighted_fip = 0
            for r in rows:
                ip = r["innings_pitched"] or 0
                if ip > 0 and r["fip"] is not None:
                    weighted_fip += r["fip"] * ip
                    total_ip += ip

            if total_ip > 0:
                return round(weighted_fip / total_ip, 2)

    # Fallback: team average FIP
    row = conn.execute(
        "SELECT AVG(fip) as avg_fip FROM pitcher_stats WHERE team = ? AND fip IS NOT NULL",
        (team_abbr,)
    ).fetchone()
    if row and row["avg_fip"]:
        return row["avg_fip"]

    return 4.00


def _get_team_quality(team_abbr, game_date, month, conn):
    """Get blended team quality: prior-weighted projected win% vs actual win%.

    Requires at least 10 games before blending in actual records to avoid
    extreme swings from small samples (e.g., 1-0 = 100% win rate).
    """
    MIN_GAMES_FOR_BLEND = 10
    prior_weight = PRIOR_WEIGHT_BY_MONTH.get(month, 0.15)

    projected_wins = WIN_TOTAL_PRIORS.get(team_abbr, 81)
    prior_winpct = projected_wins / 162.0

    year = int(game_date[:4]) if game_date else SEASON
    row = conn.execute(
        "SELECT wins, losses FROM team_stats WHERE team_name = ? AND season = ? AND wins IS NOT NULL",
        (team_abbr, year)
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT wins, losses FROM team_stats WHERE team_name = ? AND wins IS NOT NULL ORDER BY season DESC LIMIT 1",
            (team_abbr,)
        ).fetchone()

    if row and row["wins"] is not None and row["losses"] is not None:
        games_played = row["wins"] + row["losses"]
        if games_played >= MIN_GAMES_FOR_BLEND:
            actual_winpct = row["wins"] / games_played
        else:
            # Not enough games — use prior only
            actual_winpct = prior_winpct
    else:
        actual_winpct = prior_winpct

    return prior_weight * prior_winpct + (1 - prior_weight) * actual_winpct


def _get_bullpen_era(team_abbr, conn):
    """Get team bullpen ERA from FanGraphs data."""
    row = conn.execute(
        "SELECT bullpen_era FROM team_stats WHERE team_name = ? AND bullpen_era IS NOT NULL ORDER BY season DESC LIMIT 1",
        (team_abbr,)
    ).fetchone()
    return row["bullpen_era"] if row else None


def _get_wrc_plus(team_abbr, conn):
    """Get team wRC+ from FanGraphs data."""
    row = conn.execute(
        "SELECT wrc_plus FROM team_stats WHERE team_name = ? AND wrc_plus IS NOT NULL ORDER BY season DESC LIMIT 1",
        (team_abbr,)
    ).fetchone()
    return row["wrc_plus"] if row else None


def _get_park_factor(game_row):
    """Get park factor for the game's venue."""
    # Try venue name mapping first
    venue = game_row.get("venue", "") if hasattr(game_row, "get") else (game_row["venue"] if "venue" in game_row.keys() else "")
    if venue and venue in VENUE_TO_TEAM:
        team = VENUE_TO_TEAM[venue]
        return PARK_FACTORS.get(team, 1.00)

    # Fall back to home team
    home_team = game_row["home_team"]
    return PARK_FACTORS.get(home_team, 1.00)


def _get_pitcher_hand(pitcher_id, conn):
    """Get a pitcher's throwing hand from the DB."""
    if not pitcher_id:
        return None
    row = conn.execute(
        "SELECT throw_hand FROM pitcher_stats WHERE player_id = ? AND throw_hand IS NOT NULL ORDER BY season DESC LIMIT 1",
        (pitcher_id,)
    ).fetchone()
    return row["throw_hand"] if row else None


def _get_platoon_wrc(team_abbr, opposing_starter_id, conn):
    """Get team's wRC+ against the opposing starter's throwing hand.

    If the opposing starter is a LHP, returns team's wRC+ vs LHP.
    If RHP, returns wRC+ vs RHP.
    Falls back to overall wRC+ if platoon data is missing.
    """
    hand = _get_pitcher_hand(opposing_starter_id, conn)

    if hand == "L":
        row = conn.execute(
            "SELECT wrc_plus_vs_lhp FROM team_stats WHERE team_name = ? AND wrc_plus_vs_lhp IS NOT NULL ORDER BY season DESC LIMIT 1",
            (team_abbr,)
        ).fetchone()
        if row and row["wrc_plus_vs_lhp"] is not None:
            return row["wrc_plus_vs_lhp"]
    elif hand == "R":
        row = conn.execute(
            "SELECT wrc_plus_vs_rhp FROM team_stats WHERE team_name = ? AND wrc_plus_vs_rhp IS NOT NULL ORDER BY season DESC LIMIT 1",
            (team_abbr,)
        ).fetchone()
        if row and row["wrc_plus_vs_rhp"] is not None:
            return row["wrc_plus_vs_rhp"]

    return _get_wrc_plus(team_abbr, conn)
