"""balldontlie MLB ingest — dashboard data only.

Public surface:
    resolve_crosswalk(conn)       — refresh the bdl_id_map (teams + players).
                                    Nightly is enough; rosters change rarely.
    ingest_for_date(conn, date)   — pull everything the Goose dashboard needs
                                    for the given slate date. Idempotent.

This module is for the DASHBOARD only. The 2026-05-22 evaluation
(docs/balldontlie_evaluation.md) found balldontlie data does not improve the
predictor — do NOT wire any of these tables into the model.

GOAT-tier rate (600 req/min) makes the ingest a ~1 min synchronous job.
Designed to run inline in the GitHub Actions cron — no background workers.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_current_season

logger = logging.getLogger(__name__)

BASE = "https://api.balldontlie.io/mlb/v1"

# balldontlie uses CHW for the White Sox; the rest of this project uses CWS.
# bdl_id_map records the project abbr so downstream queries stay consistent.
ABBR_REMAP = {"CHW": "CWS"}


# --- HTTP --------------------------------------------------------------------

def _api_key() -> str:
    """Read the API key from env (production) or data/.bdl_key (local dev)."""
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if key:
        return key.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    keyfile = os.path.join(here, ".bdl_key")
    if os.path.exists(keyfile):
        with open(keyfile) as f:
            return f.read().strip()
    raise RuntimeError(
        "balldontlie API key not found. Set BALLDONTLIE_API_KEY or "
        f"create {keyfile}")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = _api_key()
    return s


def _get(session: requests.Session, path: str, params: Optional[dict] = None,
         max_retries: int = 4) -> dict:
    """GET a balldontlie endpoint with retry-on-429.

    GOAT rate is 600/min; we shouldn't hit 429 in normal operation, but a
    burst of parallel jobs or a transient throttle could. Honor `retry-after`
    when present, otherwise exponential backoff capped at ~10s.
    """
    url = f"{BASE}/{path.lstrip('/')}"
    backoff = 1.0
    for attempt in range(max_retries + 1):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after") or backoff)
            logger.warning("429 on %s — waiting %.1fs (attempt %d/%d)",
                           path, wait, attempt + 1, max_retries + 1)
            time.sleep(min(wait, 10))
            backoff = min(backoff * 2, 10)
            continue
        if r.status_code >= 500:
            logger.warning("%d on %s — backing off %.1fs", r.status_code, path, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {path} failed after {max_retries + 1} attempts")


def _paginate(session, path, params):
    """Iterate every record on a cursor-paginated endpoint."""
    cursor = None
    while True:
        p = dict(params or {})
        if cursor is not None:
            p["cursor"] = cursor
        p.setdefault("per_page", 100)
        resp = _get(session, path, p)
        for row in resp.get("data") or []:
            yield row
        meta = resp.get("meta") or {}
        cursor = meta.get("next_cursor")
        if not cursor:
            break


# --- crosswalk ---------------------------------------------------------------

def resolve_crosswalk(conn) -> Tuple[int, int]:
    """Refresh bdl_id_map for teams + all known MLB players.

    Teams are 30 rows — trivial. Players are resolved by querying
    /players?first_name=X&last_name=Y for each unique player name in the
    project's local tables (game_lineups, pitcher_stats).

    The 2026-05 trial discovered that `search=` matches first_name only,
    which is why it only resolved ~290 of ~414 players. Using
    first_name+last_name in tandem is much more reliable (~95%+ exact match).

    Returns (teams_resolved, players_resolved).
    """
    session = _session()
    teams_n = _ingest_teams(session, conn)
    players_n = _ingest_players(session, conn)
    conn.commit()
    logger.info("crosswalk: %d teams, %d players resolved", teams_n, players_n)
    return teams_n, players_n


def _ingest_teams(session, conn) -> int:
    """Pull every MLB team and map it to its project abbreviation."""
    resp = _get(session, "teams")
    n = 0
    for t in resp.get("data") or []:
        abbr_raw = t["abbreviation"]
        abbr = ABBR_REMAP.get(abbr_raw, abbr_raw)
        conn.execute(
            "INSERT OR REPLACE INTO bdl_id_map "
            "(mlb_id, bdl_id, entity_type, name, match_quality) "
            "VALUES (?, ?, 'team', ?, 'exact')",
            (None, t["id"], abbr))
        n += 1
    return n


def _player_names(conn) -> List[Tuple[int, str, str]]:
    """Distinct (mlb_id, first, last) tuples this project knows about.

    Pulled from game_lineups (batters) + pitcher_stats + games (starters).
    Splits on the first space — handles ~99% of names. The trickier cases
    (suffixes, multi-word last names) fall through to no-match, which is
    fine: the dashboard already degrades gracefully when a player is
    unmapped.
    """
    seen: Dict[int, Tuple[str, str]] = {}
    queries = [
        "SELECT player_id, player_name FROM game_lineups "
        "WHERE player_name IS NOT NULL",
        "SELECT player_id, player_name FROM pitcher_stats "
        "WHERE player_name IS NOT NULL",
        "SELECT home_starter_id, home_starter_name FROM games "
        "WHERE home_starter_id IS NOT NULL AND home_starter_name IS NOT NULL",
        "SELECT away_starter_id, away_starter_name FROM games "
        "WHERE away_starter_id IS NOT NULL AND away_starter_name IS NOT NULL",
    ]
    for q in queries:
        for pid, name in conn.execute(q).fetchall():
            if pid in seen or not name:
                continue
            parts = name.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            seen[pid] = (parts[0], parts[1])
    return [(pid, f, l) for pid, (f, l) in seen.items()]


def _ingest_players(session, conn) -> int:
    """Resolve each known player to a balldontlie id and cache the result.

    Skips players already mapped (idempotent). A nightly crosswalk run picks
    up new lineup members as they appear.
    """
    existing = {
        r[0] for r in conn.execute(
            "SELECT mlb_id FROM bdl_id_map WHERE entity_type='player'").fetchall()
    }
    todo = [t for t in _player_names(conn) if t[0] not in existing]
    n = 0
    for mlb_id, first, last in todo:
        try:
            resp = _get(session, "players",
                        {"first_name": first, "last_name": last, "per_page": 5})
        except Exception as e:
            logger.warning("crosswalk: %s %s failed: %s", first, last, e)
            continue
        rows = resp.get("data") or []
        # exact match preferred
        match = next((r for r in rows
                      if r["first_name"].lower() == first.lower()
                      and r["last_name"].lower() == last.lower()), None)
        if not match and rows:
            match = rows[0]  # fuzzy: first-page first-result
            quality = "fuzzy"
        elif match:
            quality = "exact"
        else:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO bdl_id_map "
            "(mlb_id, bdl_id, entity_type, name, match_quality) "
            "VALUES (?, ?, 'player', ?, ?)",
            (mlb_id, match["id"], f"{first} {last}", quality))
        n += 1
    return n


# --- ingest_for_date ---------------------------------------------------------

def ingest_for_date(conn, date_str: str) -> Dict[str, int]:
    """Pull everything the dashboard needs for date_str. Idempotent.

    Safe to re-run mid-day (lineup-lock cron) — INSERT OR REPLACE keys on
    `game_date` so the rows for *this* date get refreshed, leaving other
    dates' snapshots intact.

    Returns counts per table (for log readability).
    """
    season = get_current_season()
    session = _session()
    counts: Dict[str, int] = defaultdict(int)

    games = _ingest_games_and_odds(session, conn, date_str, counts)
    if not games:
        logger.warning("no balldontlie games for %s — slate not yet posted", date_str)
        return dict(counts)

    # Build the player set we need stats / form / arsenals / pitch-vs for.
    starter_bdl_ids = _starter_bdl_ids(conn, games)
    lineup_bdl_ids = _lineup_bdl_ids(conn, games, date_str)
    all_player_ids = list(starter_bdl_ids | lineup_bdl_ids)

    _ingest_pitch_type_stats(session, conn, sorted(starter_bdl_ids),
                             season, role="pitcher", counts=counts)
    _ingest_pitch_type_stats(session, conn, sorted(lineup_bdl_ids),
                             season, role="hitter", counts=counts)
    _ingest_season_stats(session, conn, all_player_ids,
                         season, date_str, counts)
    _ingest_form_splits(session, conn, all_player_ids,
                        season, date_str, counts)
    _ingest_injuries(session, conn, games, date_str, counts)

    conn.commit()
    logger.info("ingest_for_date(%s): %s", date_str, dict(counts))
    return dict(counts)


# --- /games + /odds ---------------------------------------------------------

def _ingest_games_and_odds(session, conn, date_str, counts) -> List[dict]:
    """Pull today's games (in-memory map) + per-game odds (bdl_odds_today).

    Returns the games list so downstream steps can build the player set.
    """
    games = list(_paginate(session, "games", {"dates[]": date_str}))
    counts["games"] = len(games)
    if not games:
        return []

    game_ids = [g["id"] for g in games]
    odds = list(_paginate(session, "odds", {"game_ids[]": game_ids}))

    by_game = {g["id"]: g for g in games}
    for o in odds:
        g = by_game.get(o["game_id"])
        if not g:
            continue
        home_abbr = ABBR_REMAP.get(g["home_team"]["abbreviation"], g["home_team"]["abbreviation"])
        away_abbr = ABBR_REMAP.get(g["away_team"]["abbreviation"], g["away_team"]["abbreviation"])
        # total_value comes back as a string in some books; coerce to float
        tv = o.get("total_value")
        if isinstance(tv, str):
            try:
                tv = float(tv)
            except (TypeError, ValueError):
                tv = None
        conn.execute(
            "INSERT OR REPLACE INTO bdl_odds_today "
            "(bdl_game_id, game_date, home_team, away_team, vendor, "
            " moneyline_home_odds, moneyline_away_odds, total_value, "
            " total_over_odds, total_under_odds, spread_home_value, "
            " spread_home_odds, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (o["game_id"], date_str, home_abbr, away_abbr, o["vendor"],
             o.get("moneyline_home_odds"), o.get("moneyline_away_odds"), tv,
             o.get("total_over_odds"), o.get("total_under_odds"),
             _coerce_float(o.get("spread_home_value")),
             o.get("spread_home_odds")))
        counts["odds"] += 1
    return games


def _coerce_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- player set --------------------------------------------------------------

def _starter_bdl_ids(conn, games) -> set:
    """Resolve today's starters (from the project's games table) to bdl ids.

    The dashboard wants the home + away starters for each game in the slate.
    We pull MLB starter ids from `games` filtered by team abbr matchups
    coming from the balldontlie schedule.
    """
    pairs = []
    for g in games:
        home = ABBR_REMAP.get(g["home_team"]["abbreviation"], g["home_team"]["abbreviation"])
        away = ABBR_REMAP.get(g["away_team"]["abbreviation"], g["away_team"]["abbreviation"])
        pairs.append((home, away))
    if not pairs:
        return set()
    placeholders = ",".join("(?, ?)" for _ in pairs)
    flat = [v for pair in pairs for v in pair]
    rows = conn.execute(
        f"SELECT home_starter_id, away_starter_id FROM games "
        f"WHERE (home_team, away_team) IN ({placeholders})", flat).fetchall()
    mlb_ids = set()
    for r in rows:
        if r["home_starter_id"] is not None:
            mlb_ids.add(r["home_starter_id"])
        if r["away_starter_id"] is not None:
            mlb_ids.add(r["away_starter_id"])
    if not mlb_ids:
        return set()
    return _mlb_to_bdl(conn, mlb_ids, "player")


def _lineup_bdl_ids(conn, games, date_str) -> set:
    """Resolve the most recent stored lineup (on or before date_str) for each
    team in the slate to bdl ids."""
    abbrs = set()
    for g in games:
        abbrs.add(ABBR_REMAP.get(g["home_team"]["abbreviation"], g["home_team"]["abbreviation"]))
        abbrs.add(ABBR_REMAP.get(g["away_team"]["abbreviation"], g["away_team"]["abbreviation"]))
    if not abbrs:
        return set()
    mlb_ids = set()
    for team in abbrs:
        d = conn.execute(
            "SELECT MAX(lineup_date) d FROM game_lineups "
            "WHERE team=? AND lineup_date<=?", (team, date_str)).fetchone()["d"]
        if not d:
            continue
        for r in conn.execute(
                "SELECT player_id FROM game_lineups "
                "WHERE team=? AND lineup_date=?", (team, d)).fetchall():
            if r["player_id"] is not None:
                mlb_ids.add(r["player_id"])
    if not mlb_ids:
        return set()
    return _mlb_to_bdl(conn, mlb_ids, "player")


def _mlb_to_bdl(conn, mlb_ids: Iterable[int], entity_type: str) -> set:
    placeholders = ",".join("?" * len(list(mlb_ids)))
    if not placeholders:
        return set()
    mlb_ids = list(mlb_ids)
    placeholders = ",".join("?" * len(mlb_ids))
    rows = conn.execute(
        f"SELECT bdl_id FROM bdl_id_map "
        f"WHERE entity_type=? AND mlb_id IN ({placeholders})",
        [entity_type] + mlb_ids).fetchall()
    return {r["bdl_id"] for r in rows}


# --- /pitcher_pitch_type + /hitter_pitch_type -------------------------------

_BATCH = 25  # players per call — balldontlie accepts player_ids[]


def _ingest_pitch_type_stats(session, conn, player_ids, season, role, counts):
    """Pull pitch-type season stats for the given roster of players."""
    if not player_ids:
        return
    endpoint = ("pitcher_pitch_type_season_stats" if role == "pitcher"
                else "hitter_pitch_type_season_stats")
    for chunk in _chunks(player_ids, _BATCH):
        for row in _paginate(session, endpoint,
                             {"season": season, "player_ids[]": chunk}):
            conn.execute(
                "INSERT OR REPLACE INTO bdl_pitch_type_stats "
                "(player_id, season, role, pitch_type, pitch_usage_percent, "
                " whiff_percent, chase_percent, zone_percent, ba, slg, woba, "
                " xwoba, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (row["player_id"], season, role, row.get("pitch_type"),
                 row.get("pitch_usage_percent"), row.get("whiff_percent"),
                 row.get("chase_percent"), row.get("zone_percent"),
                 row.get("ba"), row.get("slg"), row.get("woba"),
                 row.get("xwoba")))
            counts[f"pitch_type_{role}"] += 1


def _chunks(xs, n):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# --- /season_stats ----------------------------------------------------------

def _ingest_season_stats(session, conn, player_ids, season, date_str, counts):
    """Pull season aggregates — batting (OPS/HR) + pitching (WAR/QS)."""
    if not player_ids:
        return
    for chunk in _chunks(player_ids, _BATCH):
        for row in _paginate(session, "season_stats",
                             {"season": season, "player_ids[]": chunk}):
            # Season-level WAR/QS — historical reference, not date-scoped
            conn.execute(
                "INSERT OR REPLACE INTO bdl_season_stats "
                "(player_id, season, pitching_war, batting_war, pitching_qs, "
                " pitching_gs, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                (row["player"]["id"], season,
                 row.get("pitching_war"), row.get("batting_war"),
                 row.get("pitching_qs"), row.get("pitching_gs")))
            counts["season_stats"] += 1

            # bdl_batting_today snapshot — what the dashboard reads
            ops = _ops_from_season_stats(row)
            conn.execute(
                "INSERT OR REPLACE INTO bdl_batting_today "
                "(player_id, game_date, full_name, team, gp, avg, obp, slg, "
                " ops, hr, rbi, sb, war, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (row["player"]["id"], date_str,
                 row["player"].get("full_name"),
                 row.get("team_name"),
                 row.get("batting_gp"), row.get("batting_avg"),
                 row.get("batting_obp"), row.get("batting_slg"),
                 ops, row.get("batting_hr"), row.get("batting_rbi"),
                 row.get("batting_sb"), row.get("batting_war")))
            counts["batting_today"] += 1


def _ops_from_season_stats(row):
    """OPS is OBP + SLG. balldontlie returns batting_obp / batting_slg."""
    obp = row.get("batting_obp")
    slg = row.get("batting_slg")
    if obp is None or slg is None:
        return None
    try:
        return float(obp) + float(slg)
    except (TypeError, ValueError):
        return None


# --- /players/splits → form -------------------------------------------------

_FORM_KEYS = {
    "Last 7 Days": ("last7_ops", "last7_ab"),
    "Last 15 Days": ("last15_ops", "last15_ab"),
    "Last 30 Days": ("last30_ops", "last30_ab"),
}
_SEASON_SPLIT = ("All Splits",)  # split_category='split', split_name='All Splits'


def _ingest_form_splits(session, conn, player_ids, season, date_str, counts):
    """Pull last-7/15/30-day OPS via /players/splits.

    The /players/splits endpoint takes a singular `player_id` — no batching.
    Each call is one player at GOAT cost. For a 270-player slate this is
    ~30 sec at the GOAT rate (well under any cron window).
    """
    if not player_ids:
        return
    for pid in player_ids:
        try:
            resp = _get(session, "players/splits",
                        {"player_id": pid, "season": season})
        except Exception as e:
            logger.warning("splits %s failed: %s", pid, e)
            continue
        data = (resp.get("data") or {}) if isinstance(resp.get("data"), dict) else {}
        # season OPS (split_category='split', split_name='All Splits')
        season_ops, season_ab = None, None
        for r in data.get("split") or []:
            if r.get("split_name") in _SEASON_SPLIT:
                season_ops = r.get("ops")
                season_ab = r.get("at_bats")
                break
        # recent-window OPS via byDayMonth
        form: Dict[str, Tuple[Optional[float], Optional[int]]] = {
            v[0]: (None, None) for v in _FORM_KEYS.values()
        }
        for r in data.get("byDayMonth") or []:
            mapping = _FORM_KEYS.get(r.get("split_name"))
            if mapping:
                form[mapping[0]] = (r.get("ops"), r.get("at_bats"))
        # write the row — even if everything is None (the dashboard tolerates
        # nulls and the row's presence prevents stale day-N-1 rows from
        # masking)
        conn.execute(
            "INSERT OR REPLACE INTO bdl_form_today "
            "(player_id, game_date, season_ops, season_ab, "
            " last7_ops, last7_ab, last15_ops, last15_ab, "
            " last30_ops, last30_ab, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (pid, date_str, season_ops, season_ab,
             form["last7_ops"][0], form["last7_ops"][1],
             form["last15_ops"][0], form["last15_ops"][1],
             form["last30_ops"][0], form["last30_ops"][1]))
        counts["form_today"] += 1


# --- /player_injuries -------------------------------------------------------

def _ingest_injuries(session, conn, games, date_str, counts):
    """Pull current IL list for every team in today's slate."""
    team_bdl_ids = set()
    for g in games:
        team_bdl_ids.add(g["home_team"]["id"])
        team_bdl_ids.add(g["away_team"]["id"])
    if not team_bdl_ids:
        return
    # team_ids[] is batchable — one call covers the whole slate.
    for row in _paginate(session, "player_injuries",
                         {"team_ids[]": sorted(team_bdl_ids)}):
        player = row.get("player") or {}
        team = player.get("team") or {}
        conn.execute(
            "INSERT OR REPLACE INTO bdl_injuries "
            "(player_id, team_id, snapshot_date, injury_date, return_date, "
            " injury_type, detail, side, status, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (player.get("id"), team.get("id"), date_str,
             row.get("date"), row.get("return_date"), row.get("type"),
             row.get("detail"), row.get("side"), row.get("status")))
        counts["injuries"] += 1


# --- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from db import get_db

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Run balldontlie ingest.")
    ap.add_argument("mode", choices=["crosswalk", "ingest"])
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    with get_db() as conn:
        if args.mode == "crosswalk":
            resolve_crosswalk(conn)
        else:
            d = args.date or datetime.now().strftime("%Y-%m-%d")
            counts = ingest_for_date(conn, d)
            print(f"  done: {counts}")
