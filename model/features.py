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
    "team_quality_diff",   # blended win% difference (home - away)
    "park_factor",         # park run environment (>1 = hitter friendly, <1 = pitcher friendly)
    "home_offense_trend",  # home team recent runs scored vs season avg (+ = hot, - = cold)
    "away_offense_trend",  # away team recent runs scored vs season avg
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

        # 2. Team quality differential (prior-blended win%)
        game_date = game_row["game_date"]
        month = int(game_date[5:7]) if game_date else 6
        home_quality = _get_team_quality(game_row["home_team"], game_date, month, conn)
        away_quality = _get_team_quality(game_row["away_team"], game_date, month, conn)
        features["team_quality_diff"] = home_quality - away_quality

        # 3. Park factor
        features["park_factor"] = _get_park_factor(game_row)

        # 4-5. Offense trends (rolling 7-game runs scored vs season average)
        game_date = game_row["game_date"]
        features["home_offense_trend"] = _get_offense_trend(game_row["home_team"], game_date, conn)
        features["away_offense_trend"] = _get_offense_trend(game_row["away_team"], game_date, conn)

        return features

    finally:
        if own_conn:
            db.__exit__(None, None, None)


def build_training_features(start_year=2022, end_year=2025, return_dates=False):
    """Build feature matrix and labels for all historical games.

    Args:
        start_year, end_year: inclusive season range to pull Final games from.
        return_dates: when True, also return each game's game_date as a 4th
            element — needed by callers that do a chronological train/holdout
            split (model/xgb_experiment.py). Default False keeps the original
            3-tuple contract for every existing caller.

    Returns:
        (features_list, labels_list, game_ids_list) — or, when return_dates
        is True, (features_list, labels_list, game_ids_list, dates_list).
    """
    features_list = []
    labels_list = []
    game_ids = []
    dates = []

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
            dates.append(game["game_date"])

            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{len(games)} games processed...")

        if skipped:
            print(f"    Skipped {skipped} games with missing data")

    if return_dates:
        return features_list, labels_list, game_ids, dates
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
    MIN_GAMES_FOR_BLEND = 5
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


def _get_bullpen_era(team_abbr, conn, season=None):
    """Get team bullpen ERA from FanGraphs data for the given season.

    If season is provided, use that season's row (falling back to most recent
    prior season if missing). If season is None, use most recent overall.
    Avoids lookahead bias when used in historical training.
    """
    if season is not None:
        row = conn.execute(
            "SELECT bullpen_era FROM team_stats WHERE team_name = ? AND season <= ? AND bullpen_era IS NOT NULL ORDER BY season DESC LIMIT 1",
            (team_abbr, season)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT bullpen_era FROM team_stats WHERE team_name = ? AND bullpen_era IS NOT NULL ORDER BY season DESC LIMIT 1",
            (team_abbr,)
        ).fetchone()
    return row["bullpen_era"] if row else None


def _get_wrc_plus(team_abbr, conn, season=None):
    """Get team wRC+ from FanGraphs data for the given season.

    If season is provided, use that season's row (falling back to most recent
    prior season if missing). If season is None, use most recent overall.
    Avoids lookahead bias when used in historical training.
    """
    if season is not None:
        row = conn.execute(
            "SELECT wrc_plus FROM team_stats WHERE team_name = ? AND season <= ? AND wrc_plus IS NOT NULL ORDER BY season DESC LIMIT 1",
            (team_abbr, season)
        ).fetchone()
    else:
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


def _get_offense_trend(team_abbr, game_date, conn):
    """Get team's offensive trend: rolling 7-game runs scored minus season average.

    Positive = team is hot (scoring above average).
    Negative = team is cold (scoring below average).
    Returns 0.0 if insufficient data.
    """
    if not game_date:
        return 0.0

    # Get runs scored in all games before this date (same year)
    year = game_date[:4]
    rows = conn.execute("""
        SELECT
            CASE WHEN home_team = ? THEN home_score ELSE away_score END as runs
        FROM games
        WHERE status = 'Final'
          AND (home_team = ? OR away_team = ?)
          AND game_date < ? AND game_date >= ?
          AND home_score IS NOT NULL
        ORDER BY game_date DESC
    """, (team_abbr, team_abbr, team_abbr, game_date, f"{year}-01-01")).fetchall()

    if len(rows) < 10:
        return 0.0  # Need at least 10 games for meaningful season avg

    runs = [r["runs"] for r in rows if r["runs"] is not None]
    if len(runs) < 10:
        return 0.0

    recent_7 = runs[:7]
    season_avg = sum(runs) / len(runs)
    recent_avg = sum(recent_7) / len(recent_7)

    return round(recent_avg - season_avg, 2)


def compute_signals(game_row, conn, season=None):
    """Compute the 5 underlying signals used by the away-overconfidence damping rule.

    Each signal reports which side ('home', 'away', or None) the metric favors.
    A None means the metric is tied within a small threshold or the data is missing.

    Signals:
      - record:  who has the better blended team_quality (W-L)
      - fip:     who has the lower (better) blended starter FIP
      - bullpen: who has the lower bullpen ERA
      - wrc:     who has the higher wRC+ (offense)
      - form:    who's hotter on recent offense trend

    Returns:
        Dict mapping signal name → 'home' | 'away' | None
    """
    if season is None:
        season = int(game_row["game_date"][:4]) if game_row["game_date"] else SEASON
    month = int(game_row["game_date"][5:7]) if game_row["game_date"] else 6

    signals = {}

    # Record edge
    home_q = _get_team_quality(game_row["home_team"], game_row["game_date"], month, conn)
    away_q = _get_team_quality(game_row["away_team"], game_row["game_date"], month, conn)
    signals["record"] = None if abs(home_q - away_q) < 0.005 else ("home" if home_q > away_q else "away")

    # FIP edge — lower is better
    h_fip = _get_pitcher_fip(game_row["home_starter_id"], game_row["home_team"], conn)
    a_fip = _get_pitcher_fip(game_row["away_starter_id"], game_row["away_team"], conn)
    if h_fip is None or a_fip is None or abs(h_fip - a_fip) < 0.10:
        signals["fip"] = None
    else:
        signals["fip"] = "home" if h_fip < a_fip else "away"

    # Bullpen edge — lower ERA is better
    h_bp = _get_bullpen_era(game_row["home_team"], conn, season=season)
    a_bp = _get_bullpen_era(game_row["away_team"], conn, season=season)
    if h_bp is None or a_bp is None or abs(h_bp - a_bp) < 0.15:
        signals["bullpen"] = None
    else:
        signals["bullpen"] = "home" if h_bp < a_bp else "away"

    # wRC+ edge — higher is better
    h_wrc = _get_wrc_plus(game_row["home_team"], conn, season=season)
    a_wrc = _get_wrc_plus(game_row["away_team"], conn, season=season)
    if h_wrc is None or a_wrc is None or abs(h_wrc - a_wrc) < 2.0:
        signals["wrc"] = None
    else:
        signals["wrc"] = "home" if h_wrc > a_wrc else "away"

    # Recent form edge
    h_trend = _get_offense_trend(game_row["home_team"], game_row["game_date"], conn)
    a_trend = _get_offense_trend(game_row["away_team"], game_row["game_date"], conn)
    signals["form"] = None if abs(h_trend - a_trend) < 0.3 else ("home" if h_trend > a_trend else "away")

    return signals


def count_supporting_signals(signals, picked_side):
    """Given a signal dict and a picked side, return count of signals supporting that side.

    None signals (no edge / no data) do not count for or against.
    """
    return sum(1 for s in signals.values() if s == picked_side)


def away_overconfidence_damping(home_win_prob, signals):
    """Apply the away-team-favorite overconfidence correction.

    Empirically validated rule (see model/signal_damping_experiment.py): when the
    model picks the AWAY team with HIGH confidence (>=63%) and 3+ of 5 underlying
    signals support that pick, the model systematically overrates the away team.
    Training data (2022-2024, n=251 at 5/5): +10.1pp overprediction. The 2025
    holdout confirms direction at 4/5 (+5pp) and 2-3/5 (+12.5pp).

    Damping factor by supporting-signal count (away picks only, HIGH tier only):
       3/5: 0.05    (gentle nudge)
       4/5: 0.10    (moderate)
       5/5: 0.20    (strong — largest training-data overprediction)

    Args:
        home_win_prob: P(home team wins) from the model
        signals: dict from compute_signals()

    Returns:
        Adjusted home_win_prob (only changed if pick is away + HIGH tier + 3+/5 support)
    """
    picked_side = "home" if home_win_prob >= 0.5 else "away"
    pick_prob = home_win_prob if picked_side == "home" else 1 - home_win_prob

    # Only damp away picks at HIGH tier
    if picked_side != "away" or pick_prob < 0.63:
        return home_win_prob

    supporting = count_supporting_signals(signals, "away")
    damping_by_count = {3: 0.05, 4: 0.10, 5: 0.20}
    d = damping_by_count.get(supporting, 0.0)
    if d == 0.0:
        return home_win_prob

    # Shrink the away pick probability toward 50%; equivalently, push home_win_prob
    # toward 50% from below.
    damped_pick = 0.5 + (pick_prob - 0.5) * (1 - d)
    return 1 - damped_pick  # convert back to home_win_prob


def _get_platoon_wrc(team_abbr, opposing_starter_id, conn, season=None):
    """Get team's wRC+ against the opposing starter's throwing hand.

    If the opposing starter is a LHP, returns team's wRC+ vs LHP.
    If RHP, returns wRC+ vs RHP.
    Falls back to overall wRC+ if platoon data is missing.
    Season-aware to avoid lookahead bias in historical training.
    """
    hand = _get_pitcher_hand(opposing_starter_id, conn)

    if hand == "L":
        if season is not None:
            row = conn.execute(
                "SELECT wrc_plus_vs_lhp FROM team_stats WHERE team_name = ? AND season <= ? AND wrc_plus_vs_lhp IS NOT NULL ORDER BY season DESC LIMIT 1",
                (team_abbr, season)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT wrc_plus_vs_lhp FROM team_stats WHERE team_name = ? AND wrc_plus_vs_lhp IS NOT NULL ORDER BY season DESC LIMIT 1",
                (team_abbr,)
            ).fetchone()
        if row and row["wrc_plus_vs_lhp"] is not None:
            return row["wrc_plus_vs_lhp"]
    elif hand == "R":
        if season is not None:
            row = conn.execute(
                "SELECT wrc_plus_vs_rhp FROM team_stats WHERE team_name = ? AND season <= ? AND wrc_plus_vs_rhp IS NOT NULL ORDER BY season DESC LIMIT 1",
                (team_abbr, season)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT wrc_plus_vs_rhp FROM team_stats WHERE team_name = ? AND wrc_plus_vs_rhp IS NOT NULL ORDER BY season DESC LIMIT 1",
                (team_abbr,)
            ).fetchone()
        if row and row["wrc_plus_vs_rhp"] is not None:
            return row["wrc_plus_vs_rhp"]

    return _get_wrc_plus(team_abbr, conn, season=season)
