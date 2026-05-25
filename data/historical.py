"""Pull historical MLB game data (2022-2025) from the Stats API for model training."""

import time
import logging
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MLB_API_BASE, REQUEST_DELAY, TEAM_ID_TO_ABBR
from data.mlb_api import _api_get, get_pitcher_season_stats
from data.fip import compute_fip_from_stats, get_fip_constant_for_season, update_fip_constant_from_api
from db import get_db

logger = logging.getLogger(__name__)

# Regular season date ranges by year
SEASON_DATES = {
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}


def build_training_set(start_year=2022, end_year=2025):
    """Pull historical games and pitcher stats for model training.

    Fetches schedules in 7-day chunks, caches pitcher stats by (player_id, season)
    to minimize API calls. Inserts into games and pitcher_stats tables.
    """
    with get_db() as conn:
        # Check if we already have historical data
        existing = conn.execute(
            "SELECT COUNT(*) FROM games WHERE game_date < '2026-01-01'"
        ).fetchone()[0]
        if existing > 5000:
            print(f"  Historical data already loaded ({existing} games). Use --force to reload.")
            return

        total_games = 0
        pitcher_cache = {}  # (player_id, season) → stats dict

        for year in range(start_year, end_year + 1):
            if year not in SEASON_DATES:
                logger.warning(f"No date range defined for {year}, skipping")
                continue

            start_date, end_date = SEASON_DATES[year]
            print(f"\n  Pulling {year} season ({start_date} to {end_date})...")

            year_games = 0
            current = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            while current <= end:
                chunk_end = min(current + timedelta(days=6), end)
                date_start = current.strftime("%Y-%m-%d")
                date_end = chunk_end.strftime("%Y-%m-%d")

                data = _api_get("/schedule", params={
                    "sportId": 1,
                    "startDate": date_start,
                    "endDate": date_end,
                    "hydrate": "probablePitcher,linescore",
                    "gameType": "R",  # Regular season only
                })
                time.sleep(REQUEST_DELAY)

                if not data or not data.get("dates"):
                    current = chunk_end + timedelta(days=1)
                    continue

                for date_entry in data["dates"]:
                    for game in date_entry.get("games", []):
                        status = game.get("status", {}).get("abstractGameState", "")
                        if status != "Final":
                            continue

                        home = game.get("teams", {}).get("home", {})
                        away = game.get("teams", {}).get("away", {})
                        linescore = game.get("linescore", {})

                        home_team_id = home.get("team", {}).get("id")
                        away_team_id = away.get("team", {}).get("id")
                        home_starter_id = home.get("probablePitcher", {}).get("id")
                        away_starter_id = away.get("probablePitcher", {}).get("id")
                        home_starter_name = home.get("probablePitcher", {}).get("fullName", "TBD")
                        away_starter_name = away.get("probablePitcher", {}).get("fullName", "TBD")

                        home_score = linescore.get("teams", {}).get("home", {}).get("runs")
                        away_score = linescore.get("teams", {}).get("away", {}).get("runs")

                        if home_score is None or away_score is None:
                            continue

                        winner = "home" if home_score > away_score else "away"
                        game_date = date_entry.get("date", "")

                        conn.execute("""
                            INSERT OR IGNORE INTO games
                            (game_id, game_date, home_team, away_team, home_team_id, away_team_id,
                             home_starter_id, away_starter_id, home_starter_name, away_starter_name,
                             home_score, away_score, winner, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(game.get("gamePk", "")), game_date,
                            TEAM_ID_TO_ABBR.get(home_team_id, str(home_team_id)),
                            TEAM_ID_TO_ABBR.get(away_team_id, str(away_team_id)),
                            home_team_id, away_team_id,
                            home_starter_id, away_starter_id,
                            home_starter_name, away_starter_name,
                            home_score, away_score, winner, "Final",
                        ))
                        year_games += 1

                        # Track unique pitchers for stats pull
                        for pid in [home_starter_id, away_starter_id]:
                            if pid and (pid, year) not in pitcher_cache:
                                pitcher_cache[(pid, year)] = None  # Mark for fetching

                current = chunk_end + timedelta(days=1)

            total_games += year_games
            print(f"    {year}: {year_games} games loaded")

        # Populate FIP constant cache for each season we're about to load
        seasons_to_cache = sorted({yr for (_, yr) in pitcher_cache})
        print(f"\n  Refreshing FIP constants for seasons {seasons_to_cache}...")
        for yr in seasons_to_cache:
            update_fip_constant_from_api(yr, conn)

        # Fetch stats for all unique pitchers
        pitchers_to_fetch = [(pid, yr) for (pid, yr) in pitcher_cache if pitcher_cache[(pid, yr)] is None]
        print(f"\n  Fetching stats for {len(pitchers_to_fetch)} unique pitcher-seasons...")

        fetched = 0
        for i, (pid, yr) in enumerate(pitchers_to_fetch):
            stats = get_pitcher_season_stats(pid, yr)
            if stats is None:
                continue

            fip_constant = get_fip_constant_for_season(yr, conn)
            fip = compute_fip_from_stats(stats, fip_constant=fip_constant)

            conn.execute("""
                INSERT OR IGNORE INTO pitcher_stats
                (player_id, player_name, team, season, era, fip,
                 k_per_9, bb_per_9, innings_pitched,
                 home_runs, walks, hbp, strikeouts, hits, games_started)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid, stats.get("player_name", "Unknown"), None, yr,
                float(stats["era"]) if stats["era"] else None,
                fip,
                stats["k_per_9"], stats["bb_per_9"], stats["ip"],
                stats["hr"], stats["bb"], stats["hbp"], stats["k"],
                stats["hits"], stats["games_started"],
            ))
            fetched += 1
            pitcher_cache[(pid, yr)] = stats

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(pitchers_to_fetch)} pitchers fetched...")
                conn.commit()  # Intermediate commit

        print(f"    {fetched} pitcher stat records saved")
        print(f"\n  Total: {total_games} historical games loaded")


def get_historical_team_records():
    """Build a lookup of team records by (team_abbr, season, month).

    Returns cumulative W-L through each month for prior-weight blending.
    """
    records = {}
    with get_db() as conn:
        rows = conn.execute("""
            SELECT home_team, away_team, winner, game_date
            FROM games WHERE status = 'Final' AND game_date < '2026-01-01'
            ORDER BY game_date
        """).fetchall()

    for row in rows:
        date = row["game_date"]
        year = int(date[:4])
        month = int(date[5:7])

        for team, is_home in [(row["home_team"], True), (row["away_team"], False)]:
            key = (team, year)
            if key not in records:
                records[key] = {"wins": 0, "losses": 0}

            won = (is_home and row["winner"] == "home") or (not is_home and row["winner"] == "away")
            if won:
                records[key]["wins"] += 1
            else:
                records[key]["losses"] += 1

    return records
