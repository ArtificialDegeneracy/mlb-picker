#!/usr/bin/env python3
"""MLB Game Picker — daily win probability model for game-picking contests.

Usage:
    python main.py refresh --date 2026-03-26   Pull today's schedule + stats into DB
    python main.py status                       Show DB row counts and last update
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import SEASON, TEAM_ID_TO_ABBR, REQUEST_DELAY
from db import init_db, seed_priors, get_row_counts, get_db
from data.mlb_api import get_schedule, get_pitcher_season_stats, get_all_team_records
from data.fip import compute_fip_from_stats, get_fip_constant_for_season, update_fip_constant_from_api
from data.fangraphs import refresh_fangraphs_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def refresh_data(date_str, season=None):
    """Pull schedule, pitcher stats, team records, and FanGraphs data for a date."""
    season = season or SEASON
    print(f"\n{'='*55}")
    print(f"  MLB PICKER — Data Refresh for {date_str}")
    print(f"{'='*55}\n")

    # 1. Initialize DB + seed priors
    init_db()
    seed_priors()

    with get_db() as conn:
        # 2. Pull schedule
        print("Fetching schedule...")
        games = get_schedule(date_str)
        if not games:
            print(f"  No games found for {date_str}")
            return

        print(f"  Found {len(games)} games\n")

        # Insert games into DB and fetch weather
        from data.mlb_api import get_game_weather
        import time as _time
        for g in games:
            # Fetch weather for non-final games at open-air parks
            weather_temp = None
            weather_wind = None
            weather_condition = None
            roof_type = g.get("roof_type", "")

            if g["status"] != "Final" and roof_type in ("Open", "Retractable", ""):
                weather = get_game_weather(g["game_id"])
                if weather:
                    weather_temp = weather.get("temp")
                    weather_wind = weather.get("wind")
                    weather_condition = weather.get("condition")
                    if not roof_type:
                        roof_type = weather.get("roof_type", "")
                    _time.sleep(REQUEST_DELAY)

            conn.execute("""
                INSERT OR REPLACE INTO games
                (game_id, game_date, home_team, away_team, home_team_id, away_team_id,
                 home_starter_id, away_starter_id, home_starter_name, away_starter_name,
                 game_time, venue, roof_type, weather_temp, weather_wind, weather_condition,
                 home_score, away_score, winner, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                g["game_id"], g["game_date"], g["home_team"], g["away_team"],
                g["home_team_id"], g["away_team_id"],
                g["home_starter_id"], g["away_starter_id"],
                g["home_starter_name"], g["away_starter_name"],
                g["game_time"], g["venue"], roof_type,
                weather_temp, weather_wind, weather_condition,
                g["home_score"], g["away_score"], g["winner"], g["status"],
            ))

        # 3a. Refresh FIP constant cache for current and previous season (FIP cancels
        # out in fip_diff but stays consistent within a single pitcher's stat row).
        print("Refreshing FIP constants...")
        update_fip_constant_from_api(season, conn)
        update_fip_constant_from_api(season - 1, conn)

        # 3. Pull pitcher stats for all starters
        print("Fetching pitcher stats...")
        starter_ids = set()
        starter_names = {}
        for g in games:
            if g["home_starter_id"]:
                starter_ids.add(g["home_starter_id"])
                starter_names[g["home_starter_id"]] = g["home_starter_name"]
            if g["away_starter_id"]:
                starter_ids.add(g["away_starter_id"])
                starter_names[g["away_starter_id"]] = g["away_starter_name"]

        pitchers_updated = 0
        for pid in starter_ids:
            stats = get_pitcher_season_stats(pid, season)
            if stats is None:
                logger.warning(f"  No stats for pitcher {pid} ({starter_names.get(pid, 'Unknown')})")
                continue

            actual_season = stats.get("actual_season", season)
            fip_constant = get_fip_constant_for_season(actual_season, conn)
            fip = compute_fip_from_stats(stats, fip_constant=fip_constant)
            name = starter_names.get(pid, "Unknown")

            # Store under the actual season the data came from
            conn.execute("""
                INSERT OR REPLACE INTO pitcher_stats
                (player_id, player_name, team, season, era, fip,
                 k_per_9, bb_per_9, innings_pitched,
                 home_runs, walks, hbp, strikeouts, hits, games_started)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid, name, None, actual_season,
                float(stats["era"]) if stats["era"] else None,
                fip,
                stats["k_per_9"], stats["bb_per_9"], stats["ip"],
                stats["hr"], stats["bb"], stats["hbp"], stats["k"],
                stats["hits"], stats["games_started"],
            ))

            # If we used a fallback season, also store a current-season row
            # so the model can find the pitcher for the current year
            if actual_season != season:
                conn.execute("""
                    INSERT OR IGNORE INTO pitcher_stats
                    (player_id, player_name, team, season, era, fip,
                     k_per_9, bb_per_9, innings_pitched,
                     home_runs, walks, hbp, strikeouts, hits, games_started)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pid, name, None, season,
                    float(stats["era"]) if stats["era"] else None,
                    fip,
                    stats["k_per_9"], stats["bb_per_9"], stats["ip"],
                    stats["hr"], stats["bb"], stats["hbp"], stats["k"],
                    stats["hits"], stats["games_started"],
                ))
            pitchers_updated += 1

        print(f"  Updated {pitchers_updated}/{len(starter_ids)} pitchers")

        # 3b. Ensure previous season stats exist for opener detection
        prev_season = season - 1
        missing_prev = conn.execute("""
            SELECT DISTINCT ps.player_id, ps.player_name FROM pitcher_stats ps
            WHERE ps.player_id IN ({})
              AND NOT EXISTS (SELECT 1 FROM pitcher_stats ps2 WHERE ps2.player_id = ps.player_id AND ps2.season = ?)
        """.format(",".join(str(pid) for pid in starter_ids if pid)), (prev_season,)).fetchall()
        if missing_prev:
            prev_fetched = 0
            for row in missing_prev:
                prev_stats = get_pitcher_season_stats(row["player_id"], prev_season)
                if prev_stats and prev_stats.get("actual_season") == prev_season and prev_stats["ip"] > 0:
                    prev_fip_constant = get_fip_constant_for_season(prev_season, conn)
                    prev_fip = compute_fip_from_stats(prev_stats, fip_constant=prev_fip_constant)
                    conn.execute("""
                        INSERT OR IGNORE INTO pitcher_stats
                        (player_id, player_name, team, season, era, fip,
                         k_per_9, bb_per_9, innings_pitched,
                         home_runs, walks, hbp, strikeouts, hits, games_started)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["player_id"], row["player_name"], None, prev_season,
                        float(prev_stats["era"]) if prev_stats["era"] else None, prev_fip,
                        prev_stats["k_per_9"], prev_stats["bb_per_9"], prev_stats["ip"],
                        prev_stats["hr"], prev_stats["bb"], prev_stats["hbp"], prev_stats["k"],
                        prev_stats["hits"], prev_stats["games_started"],
                    ))
                    prev_fetched += 1
            if prev_fetched:
                print(f"  Backfilled {prev_fetched} pitcher {prev_season} stats for opener detection")

        # 3c. Fetch throwing hand for any pitcher missing it
        from data.mlb_api import get_pitcher_hand
        missing_hand = conn.execute(
            "SELECT DISTINCT player_id FROM pitcher_stats WHERE throw_hand IS NULL AND player_id IS NOT NULL"
        ).fetchall()
        if missing_hand:
            hands_fetched = 0
            for row in missing_hand:
                hand = get_pitcher_hand(row["player_id"])
                if hand:
                    conn.execute("UPDATE pitcher_stats SET throw_hand = ? WHERE player_id = ?",
                                 (hand, row["player_id"]))
                    hands_fetched += 1
            print(f"  Fetched throwing hand for {hands_fetched} pitchers")
        print()

        # 4. Pull team records
        print("Fetching team records...")
        records = get_all_team_records(season)
        for team_id, rec in records.items():
            abbr = TEAM_ID_TO_ABBR.get(team_id, str(team_id))
            conn.execute("""
                INSERT INTO team_stats (team_id, team_name, season, wins, losses, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(team_id, season) DO UPDATE SET
                    wins = excluded.wins,
                    losses = excluded.losses,
                    updated_at = datetime('now')
            """, (team_id, abbr, season, rec["wins"], rec["losses"]))
        print(f"  Updated {len(records)} team records\n")

        # 5. Refresh FanGraphs stats (wRC+ and bullpen ERA)
        print("Fetching FanGraphs stats...")
        refresh_fangraphs_stats(conn, season)
        print()

    # 6. Print summary
    _print_schedule_summary(games)


def _print_schedule_summary(games):
    """Print today's schedule with starters and stats."""
    print(f"\n{'─'*60}")
    print(f"  {'Time':<10} {'Matchup':<22} {'Home SP':<16} {'Away SP':<16}")
    print(f"{'─'*60}")

    with get_db() as conn:
        for g in games:
            # Look up FIP for each starter
            home_fip = ""
            away_fip = ""
            if g["home_starter_id"]:
                row = conn.execute(
                    "SELECT fip FROM pitcher_stats WHERE player_id = ? AND season = ?",
                    (g["home_starter_id"], SEASON)
                ).fetchone()
                if row and row["fip"]:
                    home_fip = f" ({row['fip']:.2f})"

            if g["away_starter_id"]:
                row = conn.execute(
                    "SELECT fip FROM pitcher_stats WHERE player_id = ? AND season = ?",
                    (g["away_starter_id"], SEASON)
                ).fetchone()
                if row and row["fip"]:
                    away_fip = f" ({row['fip']:.2f})"

            matchup = f"{g['away_team']} @ {g['home_team']}"
            home_sp = f"{g['home_starter_name'][:12]}{home_fip}"
            away_sp = f"{g['away_starter_name'][:12]}{away_fip}"

            from model.predict import _format_time_12h
            display_time = _format_time_12h(g['game_time'])
            print(f"  {display_time:<12} {matchup:<22} {home_sp:<16} {away_sp:<16}")

    print(f"{'─'*60}")
    print(f"  Total games: {len(games)}\n")


def run_init(force=False):
    """One-time setup: pull historical data and train the model."""
    init_db()
    seed_priors()

    from data.historical import build_training_set
    from data.fangraphs import refresh_fangraphs_stats

    print(f"\n{'='*55}")
    print(f"  MLB PICKER — Initialization")
    print(f"{'='*55}")

    # Pull FanGraphs data and team records for historical seasons
    with get_db() as conn:
        for year in [2022, 2023, 2024, 2025]:
            print(f"\n  Fetching FanGraphs stats for {year}...")
            refresh_fangraphs_stats(conn, year, force=True)
            print(f"  Fetching team records for {year}...")
            records = get_all_team_records(year)
            for team_id, rec in records.items():
                abbr = TEAM_ID_TO_ABBR.get(team_id, str(team_id))
                conn.execute("""
                    INSERT INTO team_stats (team_id, team_name, season, wins, losses, updated_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(team_id, season) DO UPDATE SET
                        wins = excluded.wins,
                        losses = excluded.losses,
                        updated_at = datetime('now')
                """, (team_id, abbr, year, rec["wins"], rec["losses"]))
            print(f"    {len(records)} teams updated")

    # Pull historical game data
    print("\n  Pulling historical game data from MLB API...")
    build_training_set(2022, 2025)

    # Train model
    from model.predict import train_model
    results = train_model(train_start=2022, train_end=2024, val_start=2025, val_end=2025)

    if results:
        print("\n  Initialization complete. You can now run predictions.")
    else:
        print("\n  WARNING: Model training failed. Check data.")


def run_predict(date_str, run_type="morning", open_dashboard=True):
    """Refresh data for a date and run predictions, then generate dashboard."""
    refresh_data(date_str)

    from model.predict import predict_games, print_predictions
    picks = predict_games(date_str, run_type)
    print_predictions(picks, date_str, run_type)

    # Generate and open dashboard
    from output.dashboard import generate_dashboard
    path = generate_dashboard(date_str)
    if open_dashboard:
        import subprocess
        subprocess.Popen(["open", path])

    return picks


def run_dashboard(date_str):
    """Generate and open the HTML dashboard."""
    from output.dashboard import generate_dashboard
    import subprocess
    path = generate_dashboard(date_str)
    subprocess.Popen(["open", path])


def show_status():
    """Print database row counts."""
    init_db()
    print(f"\nMLB Picker — Database Status (Season {SEASON})")
    print(f"{'─'*40}")
    get_row_counts()

    import os
    model_path = os.path.join(os.path.dirname(__file__), "model", "trained_model.pkl")
    if os.path.exists(model_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"  model: trained ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print(f"  model: not trained (run 'python main.py init')")
    print()


def main():
    parser = argparse.ArgumentParser(description="MLB Game Picker")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="One-time setup: pull history + train model")
    init_parser.add_argument("--force", action="store_true", help="Force re-download of historical data")

    # refresh command
    refresh_parser = subparsers.add_parser("refresh", help="Pull schedule + stats for a date")
    refresh_parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                                help="Date in YYYY-MM-DD format (default: today)")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run predictions for a date")
    predict_parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                                help="Date in YYYY-MM-DD format (default: today)")
    predict_parser.add_argument("--run", choices=["morning", "lineup_lock"], default="morning",
                                help="Run type (default: morning)")

    # dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Generate and open the HTML dashboard")
    dash_parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                             help="Date in YYYY-MM-DD format (default: today)")

    # status command
    subparsers.add_parser("status", help="Show database status")

    args = parser.parse_args()

    if args.command == "init":
        run_init(force=getattr(args, "force", False))
    elif args.command == "refresh":
        refresh_data(args.date)
    elif args.command == "dashboard":
        run_dashboard(args.date)
    elif args.command == "predict":
        run_predict(args.date, args.run)
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
