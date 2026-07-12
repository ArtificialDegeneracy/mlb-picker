"""FanGraphs API client for team-level wRC+ and bullpen ERA."""

import logging
import time
from datetime import datetime, timedelta

import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SEASON

logger = logging.getLogger(__name__)

FANGRAPHS_API_BASE = "https://www.fangraphs.com/api/leaders/major-league/data"

FANGRAPHS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# FanGraphs API uses different abbreviations for some teams
FANGRAPHS_ABBR_MAP = {
    "ATH": "OAK", "CHW": "CWS", "KCR": "KC",
    "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH",
}

# Plausibility bounds — values outside these are rejected as bad data rather
# than persisted. Observed 2026-05-09 bug: NYY+SF bullpen_era stuck at 0.0
# for 9+ days, inflating their bullpen signal. ERA of 0 is implausible for
# any MLB bullpen across a season; wRC+ of 0 means a team didn't score in
# any tracked PAs, also implausible.
BULLPEN_ERA_MIN = 0.5     # below this = bad data
BULLPEN_ERA_MAX = 12.0    # above this = bad data
WRC_PLUS_MIN = 30         # below this = bad data
WRC_PLUS_MAX = 200        # above this = bad data


def _normalize_abbr(fg_abbr):
    """Convert FanGraphs abbreviation to our standard abbreviation."""
    return FANGRAPHS_ABBR_MAP.get(fg_abbr, fg_abbr)


def _http_get_with_retry(url, params, label, retries=3):
    """GET with exponential backoff. FanGraphs intermittently returns 429/5xx;
    silent failures here are how the platoon-splits columns went NULL for all
    30 teams. Returns parsed JSON or None.
    """
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=FANGRAPHS_HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            if attempt < retries - 1:
                logger.info(f"FanGraphs {label} attempt {attempt+1} failed ({e}); retry in {delay:.1f}s")
                time.sleep(delay)
                delay *= 2
            else:
                logger.warning(f"FanGraphs {label} failed after {retries} attempts: {e}")
    return None


def _fetch_team_stats(stats_type, season):
    """Fetch team-level stats from the FanGraphs JSON API.

    Args:
        stats_type: "bat" for batting, "rel" for relievers
        season: MLB season year

    Returns:
        List of team stat dicts, or empty list on failure.
    """
    params = {
        "pos": "all",
        "stats": stats_type,
        "lg": "all",
        "qual": 0,
        "type": 8,
        "season": season,
        "month": 0,
        "season1": season,
        "ind": 0,
        "team": "0,ts",
        "rost": 0,
        "age": 0,
        "filter": "",
        "players": 0,
        "pageitems": 2147483647,
        "pagenum": 1,
    }

    data = _http_get_with_retry(FANGRAPHS_API_BASE, params, f"_fetch_team_stats({stats_type})")
    return data.get("data", []) if data else []


def get_team_wrc_plus(season=None):
    """Get team wRC+ from FanGraphs API.

    Returns:
        Dict mapping team abbreviation → wRC+ value, or empty dict on failure.
        Values outside [WRC_PLUS_MIN, WRC_PLUS_MAX] are rejected.
    """
    season = season or SEASON
    teams = _fetch_team_stats("bat", season)

    results = {}
    rejected = []
    for team in teams:
        abbr = _normalize_abbr(team.get("TeamNameAbb", ""))
        wrc_plus = team.get("wRC+")
        if abbr and wrc_plus is not None:
            try:
                v = float(wrc_plus)
            except (ValueError, TypeError):
                continue
            if WRC_PLUS_MIN <= v <= WRC_PLUS_MAX:
                results[abbr] = v
            else:
                rejected.append((abbr, v))
    if rejected:
        logger.warning(f"get_team_wrc_plus({season}): rejected implausible values: {rejected}")
    return results


def get_team_wrc_plus_vs_hand(hand, season=None):
    """Get team wRC+ vs LHP or vs RHP from FanGraphs API.

    Args:
        hand: "L" or "R" — the pitcher's throwing hand
        season: MLB season year

    Returns:
        Dict mapping team abbreviation → wRC+ value vs that hand.
        Values outside [WRC_PLUS_MIN, WRC_PLUS_MAX] are rejected.
    """
    season = season or SEASON
    # FanGraphs: month=13 is vs LHP, month=14 is vs RHP
    month_code = 13 if hand == "L" else 14

    params = {
        "pos": "all", "stats": "bat", "lg": "all", "qual": 0, "type": 1,
        "season": season, "month": month_code, "season1": season,
        "ind": 0, "team": "0,ts", "rost": 0, "age": 0,
        "pageitems": 2147483647, "pagenum": 1,
    }

    data = _http_get_with_retry(FANGRAPHS_API_BASE, params, f"platoon (vs {hand}HP)")
    if not data:
        return {}

    rows = data.get("data", [])
    results = {}
    rejected = []
    for team in rows:
        abbr = _normalize_abbr(team.get("TeamNameAbb", ""))
        wrc = team.get("wRC+")
        if abbr and wrc is not None:
            try:
                v = float(wrc)
            except (ValueError, TypeError):
                continue
            if WRC_PLUS_MIN <= v <= WRC_PLUS_MAX:
                results[abbr] = v
            else:
                rejected.append((abbr, v))
    if rejected:
        logger.warning(f"get_team_wrc_plus_vs_hand({hand}, {season}): rejected implausible: {rejected}")
    return results


def get_bullpen_era(season=None):
    """Get team bullpen (reliever) ERA from FanGraphs API.

    Returns:
        Dict mapping team abbreviation → bullpen ERA value, or empty dict on failure.
        Values outside [BULLPEN_ERA_MIN, BULLPEN_ERA_MAX] are rejected — observed
        2026-05-09 bug had NYY+SF persistently at 0.0, inflating their bullpen signal.
    """
    season = season or SEASON
    teams = _fetch_team_stats("rel", season)

    results = {}
    rejected = []
    for team in teams:
        abbr = _normalize_abbr(team.get("TeamNameAbb", ""))
        era = team.get("ERA")
        if abbr and era is not None:
            try:
                v = float(era)
            except (ValueError, TypeError):
                continue
            if BULLPEN_ERA_MIN <= v <= BULLPEN_ERA_MAX:
                results[abbr] = v
            else:
                rejected.append((abbr, v))
    if rejected:
        logger.warning(f"get_bullpen_era({season}): rejected implausible values: {rejected}")
    return results


def _ensure_refresh_log(db_conn):
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS fangraphs_refresh_log (
            season INTEGER PRIMARY KEY,
            refreshed_at TEXT NOT NULL
        )
    """)


def refresh_fangraphs_stats(db_conn, season=None, force=False):
    """Pull wRC+ and bullpen ERA from FanGraphs and update team_stats table.

    Skips refresh if FanGraphs was already pulled within the last 12 hours
    (unless force=True) — enough to dedupe same-day re-runs while letting the
    daily morning cron refresh every day.

    Freshness MUST be tracked in fangraphs_refresh_log, never via
    team_stats.updated_at: the team-records write in refresh_data bumps
    updated_at for every team right before this runs, so an updated_at-based
    check reads "fresh" forever and this function never executes. That is how
    wrc_plus_vs_lhp/vs_rhp sat NULL for all 30 teams for 2 months in 2026 and
    stuck bullpen_era rows never self-repaired.
    """
    season = season or SEASON
    _ensure_refresh_log(db_conn)

    if not force:
        row = db_conn.execute(
            "SELECT refreshed_at FROM fangraphs_refresh_log WHERE season = ?", (season,)
        ).fetchone()
        if row and row[0]:
            try:
                last_update = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                if datetime.now() - last_update < timedelta(hours=12):
                    logger.info("FanGraphs data is fresh (< 12 hours), skipping refresh")
                    return
            except ValueError:
                pass  # Proceed with refresh if timestamp is unparseable

    wrc = get_team_wrc_plus(season)
    bullpen = get_bullpen_era(season)
    wrc_vs_lhp = get_team_wrc_plus_vs_hand("L", season)
    wrc_vs_rhp = get_team_wrc_plus_vs_hand("R", season)

    # Fall back to previous season if current season has sparse data
    if len(wrc) < 15 or len(bullpen) < 15:
        logger.info(f"Sparse {season} FanGraphs data ({len(wrc)} wRC+, {len(bullpen)} ERA) — falling back to {season - 1}")
        for getter, target in [
            (lambda: get_team_wrc_plus(season - 1), wrc),
            (lambda: get_bullpen_era(season - 1), bullpen),
            (lambda: get_team_wrc_plus_vs_hand("L", season - 1), wrc_vs_lhp),
            (lambda: get_team_wrc_plus_vs_hand("R", season - 1), wrc_vs_rhp),
        ]:
            prev = getter()
            for team, val in prev.items():
                target.setdefault(team, val)

    if not wrc and not bullpen:
        logger.warning("FanGraphs returned no data — skipping update")
        return

    from config import ABBR_TO_TEAM_ID

    all_teams = set(list(wrc.keys()) + list(bullpen.keys()) + list(wrc_vs_lhp.keys()) + list(wrc_vs_rhp.keys()))
    updated = 0
    for abbr in all_teams:
        team_id = ABBR_TO_TEAM_ID.get(abbr)
        if team_id is None:
            continue

        db_conn.execute("""
            INSERT INTO team_stats (team_id, team_name, season, wrc_plus, wrc_plus_vs_lhp, wrc_plus_vs_rhp, bullpen_era, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(team_id, season) DO UPDATE SET
                wrc_plus = COALESCE(excluded.wrc_plus, wrc_plus),
                wrc_plus_vs_lhp = COALESCE(excluded.wrc_plus_vs_lhp, wrc_plus_vs_lhp),
                wrc_plus_vs_rhp = COALESCE(excluded.wrc_plus_vs_rhp, wrc_plus_vs_rhp),
                bullpen_era = COALESCE(excluded.bullpen_era, bullpen_era),
                updated_at = datetime('now')
        """, (team_id, abbr, season, wrc.get(abbr), wrc_vs_lhp.get(abbr), wrc_vs_rhp.get(abbr), bullpen.get(abbr)))
        updated += 1

    db_conn.execute("""
        INSERT INTO fangraphs_refresh_log (season, refreshed_at) VALUES (?, ?)
        ON CONFLICT(season) DO UPDATE SET refreshed_at = excluded.refreshed_at
    """, (season, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    logger.info(f"Updated FanGraphs stats for {updated} teams")
    print(f"  FanGraphs: updated {updated} teams (wRC+: {len(wrc)}, bullpen ERA: {len(bullpen)}, platoon splits: {len(wrc_vs_lhp)}L/{len(wrc_vs_rhp)}R)")
