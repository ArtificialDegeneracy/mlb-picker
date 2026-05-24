"""Backfill pitcher_stats for historical seasons.

The pitcher_stats table is missing 2023 and 2024 entirely, and has gaps in 2022.
This script finds every (pitcher_id, season) pair that started a game in 2022-2024
but is missing a corresponding pitcher_stats row, and fetches it from the MLB API.

Usage:
    python -m data.backfill_pitcher_stats
    python -m data.backfill_pitcher_stats --year 2023      # single year
    python -m data.backfill_pitcher_stats --dry-run        # show what would be fetched
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mlb_api import get_pitcher_season_stats
from data.fip import compute_fip_from_stats, get_fip_constant_for_season, update_fip_constant_from_api
from db import get_db


def find_missing(years):
    """Return list of (player_id, season, player_name) needing backfill."""
    missing = []
    with get_db() as conn:
        for year in years:
            # All distinct starters in games for this year
            rows = conn.execute("""
                SELECT DISTINCT pid, name FROM (
                    SELECT home_starter_id as pid, home_starter_name as name
                    FROM games
                    WHERE substr(game_date, 1, 4) = ? AND home_starter_id IS NOT NULL
                    UNION
                    SELECT away_starter_id as pid, away_starter_name as name
                    FROM games
                    WHERE substr(game_date, 1, 4) = ? AND away_starter_id IS NOT NULL
                )
            """, (str(year), str(year))).fetchall()

            for r in rows:
                pid = r["pid"]
                # Already have it for this season?
                existing = conn.execute(
                    "SELECT 1 FROM pitcher_stats WHERE player_id = ? AND season = ? LIMIT 1",
                    (pid, year)
                ).fetchone()
                if not existing:
                    missing.append((pid, year, r["name"]))
    return missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, action="append",
                   help="Year to backfill (repeatable). Default: 2022, 2023, 2024.")
    p.add_argument("--dry-run", action="store_true", help="Don't write, just report scope.")
    args = p.parse_args()

    years = args.year or [2022, 2023, 2024]
    print(f"Finding missing pitcher-seasons for years: {years}")

    missing = find_missing(years)
    print(f"\n{len(missing)} (pitcher, season) pairs need backfill.")

    by_year = {}
    for pid, yr, name in missing:
        by_year.setdefault(yr, []).append((pid, name))
    for yr in sorted(by_year):
        print(f"  {yr}: {len(by_year[yr])} pitchers")

    if args.dry_run:
        print("\n(dry-run — exiting)")
        return

    if not missing:
        print("Nothing to backfill.")
        return

    print(f"\nFetching... (~0.4s per pitcher, ETA ~{len(missing) * 0.4 / 60:.1f} min)")
    fetched, failed, no_data = 0, 0, 0
    with get_db() as conn:
        # Ensure FIP constant cache covers every season we're about to write.
        seasons_to_cache = sorted({yr for (_, yr, _) in missing})
        print(f"  Refreshing FIP constants for seasons {seasons_to_cache}...")
        for yr in seasons_to_cache:
            update_fip_constant_from_api(yr, conn)

        for i, (pid, year, name) in enumerate(missing):
            try:
                stats = get_pitcher_season_stats(pid, year)
            except Exception as e:
                print(f"  ERROR fetching pid={pid} season={year}: {e}")
                failed += 1
                continue

            if stats is None:
                no_data += 1
                continue

            # The API helper may return data from a different season if requested
            # season has no data. We want to insert under the year we asked for —
            # the season the game was played in — so the historical lookup is right.
            fip_constant = get_fip_constant_for_season(year, conn)
            fip = compute_fip_from_stats(stats, fip_constant=fip_constant)

            conn.execute("""
                INSERT OR IGNORE INTO pitcher_stats
                (player_id, player_name, team, season, era, fip,
                 k_per_9, bb_per_9, innings_pitched,
                 home_runs, walks, hbp, strikeouts, hits, games_started)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid, name or stats.get("player_name") or "Unknown", None, year,
                float(stats["era"]) if stats["era"] else None,
                fip,
                stats["k_per_9"], stats["bb_per_9"], stats["ip"],
                stats["hr"], stats["bb"], stats["hbp"], stats["k"],
                stats["hits"], stats["games_started"],
            ))
            fetched += 1

            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  {i + 1}/{len(missing)}  (fetched={fetched}, no_data={no_data}, failed={failed})")

        conn.commit()

    print(f"\nDone.")
    print(f"  Fetched and inserted: {fetched}")
    print(f"  No data from API:     {no_data}")
    print(f"  Errors:               {failed}")


if __name__ == "__main__":
    main()
