"""Audit which games fall through to the FIP default (4.00) and why.

For each 2022-2024 game where home or away starter hits the default-4.00 path,
record: pitcher_id, name, season of game, whether pitcher exists in pitcher_stats
at all, and if so, what data is missing.

Usage:
    python -m model.fip_fallback_audit
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from db import get_db


def main():
    with get_db() as conn:
        # Pull all 2022-2024 final games
        games = conn.execute("""
            SELECT game_id, game_date, home_team, away_team,
                   home_starter_id, away_starter_id,
                   home_starter_name, away_starter_name
            FROM games
            WHERE status = 'Final'
              AND game_date >= '2022-01-01' AND game_date <= '2024-12-31'
              AND winner IS NOT NULL
            ORDER BY game_date
        """).fetchall()
        print(f"Auditing {len(games)} games (2022-2024)...")

        # For each game starter, classify why it falls back
        # Categories:
        #   "real"          — has fip with ip>0 in pitcher_stats
        #   "no_id"         — game has no starter_id
        #   "not_in_db"     — starter_id not in pitcher_stats at all
        #   "in_db_no_fip"  — in pitcher_stats but no fip or no ip
        causes = Counter()
        fallback_examples = []  # (date, name, cause)
        fallback_by_year = Counter()
        fallback_by_year_total = Counter()

        for g in games:
            year = g["game_date"][:4]
            for side in ["home", "away"]:
                fallback_by_year_total[year] += 1
                pid = g[f"{side}_starter_id"]
                name = g[f"{side}_starter_name"] or "?"

                if not pid:
                    causes["no_id"] += 1
                    fallback_by_year[year] += 1
                    if len(fallback_examples) < 20:
                        fallback_examples.append((g["game_date"], name, "no_id"))
                    continue

                rows = conn.execute(
                    "SELECT season, fip, innings_pitched FROM pitcher_stats WHERE player_id = ? ORDER BY season DESC LIMIT 5",
                    (pid,)
                ).fetchall()

                if not rows:
                    causes["not_in_db"] += 1
                    fallback_by_year[year] += 1
                    if len(fallback_examples) < 40:
                        fallback_examples.append((g["game_date"], f"{name} (pid={pid})", "not_in_db"))
                    continue

                # In DB. Does any row have ip>0 and fip?
                has_usable = any(
                    (r["fip"] is not None and (r["innings_pitched"] or 0) > 0)
                    for r in rows
                )
                if has_usable:
                    causes["real"] += 1
                else:
                    causes["in_db_no_fip"] += 1
                    fallback_by_year[year] += 1
                    if len(fallback_examples) < 60:
                        seasons = [r["season"] for r in rows]
                        fallback_examples.append((g["game_date"], f"{name} (seasons={seasons})", "in_db_no_fip"))

        total_starters = sum(causes.values())
        print(f"\nTotal starter-slots: {total_starters}")
        print(f"\nCauses:")
        for cause, n in causes.most_common():
            pct = n / total_starters
            print(f"  {cause:<18} {n:6} ({pct:6.1%})")

        print(f"\nFallback rate by year:")
        for year in sorted(fallback_by_year_total):
            fb = fallback_by_year[year]
            total = fallback_by_year_total[year]
            print(f"  {year}: {fb:5} / {total:5}  ({fb/total:.1%})")

        print(f"\nExample fallback games:")
        for date, name, cause in fallback_examples[:30]:
            print(f"  {date}  [{cause:<14}] {name}")

        # Now: of the not_in_db pitchers, are they openers? Recent call-ups?
        # Let's look at the distinct missing pitcher_ids
        missing_pids = set()
        for g in games:
            for side in ["home", "away"]:
                pid = g[f"{side}_starter_id"]
                if pid:
                    row = conn.execute(
                        "SELECT 1 FROM pitcher_stats WHERE player_id = ? LIMIT 1",
                        (pid,)
                    ).fetchone()
                    if not row:
                        missing_pids.add(pid)

        print(f"\nDistinct missing pitcher_ids: {len(missing_pids)}")

        # How many of these missing pitchers started multiple games in 2022-2024?
        # That tells us if they're real recurring starters we should backfill.
        if missing_pids:
            pid_counts = Counter()
            for g in games:
                for side in ["home", "away"]:
                    pid = g[f"{side}_starter_id"]
                    if pid in missing_pids:
                        pid_counts[pid] += 1
            print(f"\nTop 20 missing pitchers by # games started (these are the biggest backfill wins):")
            for pid, n in pid_counts.most_common(20):
                # Find a name
                name_row = conn.execute("""
                    SELECT home_starter_name as name FROM games
                    WHERE home_starter_id = ? LIMIT 1
                """, (pid,)).fetchone()
                if not name_row or not name_row["name"]:
                    name_row = conn.execute("""
                        SELECT away_starter_name as name FROM games
                        WHERE away_starter_id = ? LIMIT 1
                    """, (pid,)).fetchone()
                name = name_row["name"] if name_row else "?"
                print(f"  pid={pid:8}  {name:30}  {n:4} starts")


if __name__ == "__main__":
    main()
