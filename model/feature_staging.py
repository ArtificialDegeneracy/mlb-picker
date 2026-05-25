"""
Path B feature staging — candidate features from balldontlie API data.

WHAT THIS IS
------------
A staging area for new model-input candidates sourced from balldontlie
(see docs/balldontlie_evaluation.md). It is SEPARATE from model/features.py:
- model/features.py owns the 5 PRODUCTION features and is untouched.
- this module computes ~12 EXPERIMENTAL features for the XGBoost candidate
  (model/xgb_experiment.py) to train and grid-test.

Nothing here is wired into the live prediction path. A feature graduates from
here into model/features.py:FEATURE_NAMES only after the XGBoost experiment
shows it earns its place.

HOW IT STAYS RUNNABLE BEFORE THE TRIAL
--------------------------------------
Every feature reads from the bdl_* SQLite cache tables (db.py). Those tables are
EMPTY until a balldontlie ingest populates them. Each feature function returns a
documented NEUTRAL value (0.0 for a diff, or None) when its data is missing, so
the whole module runs today on an empty cache — the XGBoost harness can prove
its plumbing now and the same code produces real features once data lands.

MODEL-FIT TAGS (the important part)
-----------------------------------
Each candidate is tagged 'linear' or 'tree':
  - 'linear'  — a clean monotonic diff; logistic regression CAN use it.
  - 'tree'    — an interaction or sample-size-gated signal; only a tree model
                (XGBoost) can extract it. Testing a 'tree' feature in logreg
                will falsely fail it — that was the 2026-05-18 trap.

And tagged by training-eligibility:
  - 'training'   — expected to have 2022-2024 history (confirm via the probe).
  - 'inference'  — only recent data; usable for live picks, NOT for the
                   2022-2024 training set.

These tags are PROVISIONAL until the trial probe confirms historical depth.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

# Neutral fallbacks. A "diff" feature is 0.0 when data is absent (no edge
# either way); a rate feature is None so the harness can impute per-column.
NEUTRAL_DIFF = 0.0


# ---------------------------------------------------------------------------
# bdl id resolution
# ---------------------------------------------------------------------------

def _bdl_id(conn, mlb_id, entity_type):
    """MLB Stats API id -> balldontlie id via the bdl_id_map crosswalk.
    Returns None if unmapped (cache not yet populated)."""
    if mlb_id is None:
        return None
    row = conn.execute(
        "SELECT bdl_id FROM bdl_id_map WHERE mlb_id=? AND entity_type=?",
        (mlb_id, entity_type),
    ).fetchone()
    return row[0] if row else None


def _lineup_batter_ids(conn, game_id, team):
    """MLB player_ids of the stored lineup for one team in one game."""
    return [r[0] for r in conn.execute(
        "SELECT player_id FROM game_lineups WHERE game_id=? AND team=? "
        "ORDER BY lineup_position",
        (game_id, team),
    )]


# ---------------------------------------------------------------------------
# pitch-type helpers (shared by several features)
# ---------------------------------------------------------------------------

# Contact-quality field, best-to-worst. The balldontlie docs show xwoba and
# woba both EXIST on pitch-type rows but are frequently null (seen null even in
# a real sample response). slg is always populated. arsenal features try this
# chain so the marquee feature degrades instead of dying when xwoba is absent.
QUALITY_FIELD_CHAIN = ["xwoba", "woba", "slg"]


def _best_quality(row_or_dict):
    """First populated value from QUALITY_FIELD_CHAIN. (value, field_used) or
    (None, None). Accepts a sqlite3.Row or a dict."""
    for f in QUALITY_FIELD_CHAIN:
        try:
            v = row_or_dict[f]
        except (KeyError, IndexError):
            v = None
        if v is not None:
            return v, f
    return None, None


def _pitcher_arsenal(conn, mlb_pitcher_id, season):
    """All pitch-type rows for a starter that season. List of sqlite3.Row."""
    bdl = _bdl_id(conn, mlb_pitcher_id, "player")
    if bdl is None:
        return []
    return conn.execute(
        "SELECT pitch_type, pitch_usage_percent, whiff_percent, "
        "xwoba, woba, slg FROM bdl_pitch_type_stats "
        "WHERE player_id=? AND season=? AND role='pitcher'",
        (bdl, season),
    ).fetchall()


def _hitter_pitch_profile(conn, mlb_batter_id, season):
    """Map pitch_type -> best-available contact-quality value for one hitter.
    Uses the xwoba->woba->slg chain so the profile is populated even when
    Statcast xwoba is missing."""
    bdl = _bdl_id(conn, mlb_batter_id, "player")
    if bdl is None:
        return {}
    rows = conn.execute(
        "SELECT pitch_type, xwoba, woba, slg FROM bdl_pitch_type_stats "
        "WHERE player_id=? AND season=? AND role='hitter'",
        (bdl, season),
    ).fetchall()
    out = {}
    for r in rows:
        v, _ = _best_quality(r)
        if v is not None:
            out[r["pitch_type"]] = v
    return out


def _usage_weighted(rows, value_key):
    """Usage-weighted mean of `value_key` across arsenal rows. None if no data.
    Pass value_key='_quality' to use the xwoba->woba->slg fallback chain."""
    num = den = 0.0
    for r in rows:
        if value_key == "_quality":
            v, _ = _best_quality(r)
        else:
            v = r[value_key]
        u = r["pitch_usage_percent"]
        if v is not None and u is not None:
            num += v * u
            den += u
    return num / den if den > 0 else None


# ===========================================================================
# CANDIDATE FEATURES
# Each: (game_row, conn, season) -> float. Neutral fallback when data absent.
# ===========================================================================

# --- arsenal: pitcher_pitch_type_season_stats ------------------------------

def feat_arsenal_xwoba_diff(game, conn, season):
    """Usage-weighted contact-quality allowed: away starter - home starter.
    Positive => the home lineup faces a more hittable starter => home edge.
    Uses the xwoba->woba->slg chain (see QUALITY_FIELD_CHAIN) — named
    'xwoba' for continuity, but degrades gracefully when xwoba is null."""
    home = _usage_weighted(
        _pitcher_arsenal(conn, game["home_starter_id"], season), "_quality")
    away = _usage_weighted(
        _pitcher_arsenal(conn, game["away_starter_id"], season), "_quality")
    if home is None or away is None:
        return NEUTRAL_DIFF
    return away - home


def feat_starter_whiff_diff(game, conn, season):
    """Usage-weighted whiff%: home starter - away starter.
    Positive => home starter misses more bats => home edge."""
    home = _usage_weighted(_pitcher_arsenal(conn, game["home_starter_id"], season), "whiff_percent")
    away = _usage_weighted(_pitcher_arsenal(conn, game["away_starter_id"], season), "whiff_percent")
    if home is None or away is None:
        return NEUTRAL_DIFF
    return home - away


def feat_arsenal_diversity_diff(game, conn, season):
    """Count of pitch types thrown >=10%: home starter - away starter.
    A deep arsenal interacts nonlinearly with quality — tree-only."""
    def diversity(pid):
        rows = _pitcher_arsenal(conn, pid, season)
        return sum(1 for r in rows
                   if (r["pitch_usage_percent"] or 0) >= 10.0)
    home_rows = _pitcher_arsenal(conn, game["home_starter_id"], season)
    away_rows = _pitcher_arsenal(conn, game["away_starter_id"], season)
    if not home_rows or not away_rows:
        return NEUTRAL_DIFF
    return diversity(game["home_starter_id"]) - diversity(game["away_starter_id"])


# --- the marquee feature: arsenal x lineup interaction ---------------------

def _arsenal_matchup_for_side(conn, starter_id, batter_ids, season):
    """
    For one starter vs one lineup: sum over the starter's pitch types of
    (pitch usage%) x (lineup's mean xwoba vs that pitch type).
    Higher => this lineup hits this starter's mix hard. None if data missing.
    """
    arsenal = _pitcher_arsenal(conn, starter_id, season)
    if not arsenal:
        return None
    profiles = [_hitter_pitch_profile(conn, b, season) for b in batter_ids]
    profiles = [p for p in profiles if p]
    if not profiles:
        return None
    score = 0.0
    used = 0.0
    for r in arsenal:
        pt, usage = r["pitch_type"], r["pitch_usage_percent"]
        if usage is None:
            continue
        lineup_xw = [p[pt] for p in profiles if pt in p]
        if not lineup_xw:
            continue
        score += usage * (sum(lineup_xw) / len(lineup_xw))
        used += usage
    return score / used if used > 0 else None


def feat_arsenal_matchup_score(game, conn, season):
    """
    THE marquee Path B feature. home_lineup-vs-away_starter matchup minus
    away_lineup-vs-home_starter matchup. Positive => the home lineup is better
    set up against what it will see => home edge.

    This is a pitch-mix x lineup INTERACTION. A linear model structurally
    cannot represent it; it exists for the tree model. Needs both lineups'
    per-pitch-type xwoba — the most data-hungry candidate.
    """
    home_bats = _lineup_batter_ids(conn, game["game_id"], game["home_team"])
    away_bats = _lineup_batter_ids(conn, game["game_id"], game["away_team"])
    if not home_bats or not away_bats:
        return NEUTRAL_DIFF
    home_vs = _arsenal_matchup_for_side(conn, game["away_starter_id"], home_bats, season)
    away_vs = _arsenal_matchup_for_side(conn, game["home_starter_id"], away_bats, season)
    if home_vs is None or away_vs is None:
        return NEUTRAL_DIFF
    return home_vs - away_vs


# --- H2H: players/versus ---------------------------------------------------

def _lineup_h2h(conn, batter_ids, opp_team_mlb_id):
    """AB-weighted aggregate H2H OPS for a lineup vs a team.
    Returns (weighted_ops, total_at_bats). (None, 0) when no data."""
    opp_bdl = _bdl_id(conn, opp_team_mlb_id, "team")
    if opp_bdl is None:
        return None, 0
    num = 0.0
    total_ab = 0
    for b in batter_ids:
        bdl = _bdl_id(conn, b, "player")
        if bdl is None:
            continue
        row = conn.execute(
            "SELECT ops, at_bats FROM bdl_h2h "
            "WHERE batter_id=? AND opponent_team_id=?",
            (bdl, opp_bdl),
        ).fetchone()
        if row and row[0] is not None and row[1]:
            num += row[0] * row[1]
            total_ab += row[1]
    return (num / total_ab if total_ab else None), total_ab


def feat_h2h_ops_diff(game, conn, season):
    """AB-weighted lineup H2H OPS: home lineup vs away team minus the reverse.
    Tree-only — its reliability depends on h2h_sample_size below."""
    home_bats = _lineup_batter_ids(conn, game["game_id"], game["home_team"])
    away_bats = _lineup_batter_ids(conn, game["game_id"], game["away_team"])
    if not home_bats or not away_bats:
        return NEUTRAL_DIFF
    home_ops, _ = _lineup_h2h(conn, home_bats, game["away_team_id"])
    away_ops, _ = _lineup_h2h(conn, away_bats, game["home_team_id"])
    if home_ops is None or away_ops is None:
        return NEUTRAL_DIFF
    return home_ops - away_ops


def feat_h2h_sample_size(game, conn, season):
    """Total H2H at-bats behind feat_h2h_ops_diff (both lineups summed).
    NOT a directional feature — it lets the tree GATE on H2H reliability:
    trust h2h_ops_diff only where this is large. Useless to a linear model."""
    home_bats = _lineup_batter_ids(conn, game["game_id"], game["home_team"])
    away_bats = _lineup_batter_ids(conn, game["game_id"], game["away_team"])
    if not home_bats or not away_bats:
        return 0.0
    _, home_ab = _lineup_h2h(conn, home_bats, game["away_team_id"])
    _, away_ab = _lineup_h2h(conn, away_bats, game["home_team_id"])
    return float(home_ab + away_ab)


# --- splits: players/splits ------------------------------------------------

def _starter_split(conn, mlb_pitcher_id, season, category, name, field):
    """
    Pull one pitching-split field for a starter. None if absent.

    `category` / `name` map to the API's split_category / split_name. Their
    exact string values are PROBE-CONFIRMED (the docs truncate before listing
    them) — SPLIT_CATEGORY_* / SPLIT_NAME_* below hold the current best guess;
    update them once the probe reports the real strings.
    """
    bdl = _bdl_id(conn, mlb_pitcher_id, "player")
    if bdl is None:
        return None
    row = conn.execute(
        "SELECT " + field + " FROM bdl_player_splits "
        "WHERE player_id=? AND season=? AND split_category=? "
        "AND split_name=? AND role='pitching'",
        (bdl, season, category, name),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


# --- split category/name strings — CONFIRMED from the live API 2026-05-21 ----
# balldontlie /players/splits categories: byArena, byBreakdown, byDayMonth,
# byOpponent, split. Key finding: there is NO vs-RHP/vs-LHP handedness split —
# byBreakdown is Home/Away/Day/Night only. So a pitcher-platoon-split feature
# CANNOT be built from this endpoint (see feat_starter_platoon_split_diff).
SPLIT_CAT_RECENT = "byDayMonth"       # holds 'Last 7/15/30 Days', month names
SPLIT_CAT_SEASON = "split"            # the season-total category
SPLIT_NAME_RECENT = "Last 30 Days"    # recent-form window
SPLIT_NAME_SEASON = "All Splits"      # season baseline row


def feat_starter_platoon_split_diff(game, conn, season):
    """
    DISABLED — balldontlie's /players/splits has NO vs-RHP/vs-LHP handedness
    split (confirmed live 2026-05-21: byBreakdown is Home/Away/Day/Night only).
    A pitcher platoon-split feature cannot be built from this data source.
    Returns neutral so it drops out of the model harmlessly. Kept registered
    for audit visibility; remove from CANDIDATE_FEATURES if you want it gone.
    """
    return NEUTRAL_DIFF


def feat_starter_recent_form_diff(game, conn, season):
    """
    Recent-form ERA minus season ERA, per starter, then home-minus-away of
    that delta. A genuine pitcher recency signal — the model only has crude
    team-level offense_trend today. Tree-only: recency interacts with quality.
    Uses balldontlie's 'Last 30 Days' split vs the season baseline.
    """
    def form(pid):
        recent = _starter_split(conn, pid, season, SPLIT_CAT_RECENT,
                                SPLIT_NAME_RECENT, "era")
        season_era = _starter_split(conn, pid, season, SPLIT_CAT_SEASON,
                                    SPLIT_NAME_SEASON, "era")
        if recent is None or season_era is None:
            return None
        return recent - season_era  # negative => pitching better than usual
    h, a = form(game["home_starter_id"]), form(game["away_starter_id"])
    if h is None or a is None:
        return NEUTRAL_DIFF
    # home better-than-usual (more negative) should help home => away - home
    return a - h


def feat_lineup_platoon_edge(game, conn, season):
    """
    Count of opposing batters holding the platoon advantage vs each starter
    (L bat vs R pitcher, or R bat vs L pitcher; switch hitters always have it).
    Feature = home lineup's platoon-advantaged count vs away starter minus the
    reverse. Uses bat_side / throw_hand already in the DB — no balldontlie data
    needed, so this one works TODAY. Linear-friendly.
    """
    def advantaged(batter_ids, starter_mlb_id):
        hand_row = conn.execute(
            "SELECT throw_hand FROM pitcher_stats WHERE player_id=? "
            "AND throw_hand IS NOT NULL ORDER BY season DESC LIMIT 1",
            (starter_mlb_id,),
        ).fetchone()
        if not hand_row:
            return None
        p_hand = hand_row[0]
        count = 0
        for b in batter_ids:
            br = conn.execute(
                "SELECT bat_side FROM game_lineups WHERE player_id=? "
                "AND bat_side IS NOT NULL LIMIT 1", (b,),
            ).fetchone()
            if not br:
                continue
            bs = br[0]
            if bs == "S" or (bs and p_hand and bs != p_hand):
                count += 1
        return count
    home_bats = _lineup_batter_ids(conn, game["game_id"], game["home_team"])
    away_bats = _lineup_batter_ids(conn, game["game_id"], game["away_team"])
    if not home_bats or not away_bats:
        return NEUTRAL_DIFF
    h = advantaged(home_bats, game["away_starter_id"])
    a = advantaged(away_bats, game["home_starter_id"])
    if h is None or a is None:
        return NEUTRAL_DIFF
    return float(h - a)


# --- season_stats: WAR / QS ------------------------------------------------

def feat_starter_qs_rate_diff(game, conn, season):
    """Quality-start rate (QS / GS): home starter - away starter."""
    def qs_rate(pid):
        bdl = _bdl_id(conn, pid, "player")
        if bdl is None:
            return None
        row = conn.execute(
            "SELECT pitching_qs, pitching_gs FROM bdl_season_stats "
            "WHERE player_id=? AND season=?", (bdl, season),
        ).fetchone()
        if not row or not row[1]:
            return None
        return (row[0] or 0) / row[1]
    h, a = qs_rate(game["home_starter_id"]), qs_rate(game["away_starter_id"])
    if h is None or a is None:
        return NEUTRAL_DIFF
    return h - a


# ===========================================================================
# REGISTRY — single source of truth for the harness
# ===========================================================================

# fn / fit / eligibility / endpoint. `fit` and `eligibility` are PROVISIONAL
# until the trial probe confirms historical depth (docs/balldontlie_evaluation.md).
CANDIDATE_FEATURES = {
    "arsenal_xwoba_diff": {
        "fn": feat_arsenal_xwoba_diff, "fit": "linear",
        "eligibility": "unknown", "endpoint": "pitcher_pitch_type_season_stats"},
    "starter_whiff_diff": {
        "fn": feat_starter_whiff_diff, "fit": "linear",
        "eligibility": "unknown", "endpoint": "pitcher_pitch_type_season_stats"},
    "arsenal_diversity_diff": {
        "fn": feat_arsenal_diversity_diff, "fit": "tree",
        "eligibility": "unknown", "endpoint": "pitcher_pitch_type_season_stats"},
    "arsenal_matchup_score": {
        "fn": feat_arsenal_matchup_score, "fit": "tree",
        "eligibility": "unknown",
        "endpoint": "pitcher_+hitter_pitch_type_season_stats"},
    "h2h_ops_diff": {
        "fn": feat_h2h_ops_diff, "fit": "tree",
        "eligibility": "unknown", "endpoint": "players/versus"},
    "h2h_sample_size": {
        "fn": feat_h2h_sample_size, "fit": "tree",
        "eligibility": "unknown", "endpoint": "players/versus"},
    "starter_platoon_split_diff": {
        "fn": feat_starter_platoon_split_diff, "fit": "linear",
        "eligibility": "unknown", "endpoint": "players/splits"},
    "starter_recent_form_diff": {
        "fn": feat_starter_recent_form_diff, "fit": "tree",
        "eligibility": "unknown", "endpoint": "players/splits"},
    "lineup_platoon_edge": {
        "fn": feat_lineup_platoon_edge, "fit": "linear",
        "eligibility": "training",  # uses DB handedness only — no balldontlie dependency
        "endpoint": "(none — local DB)"},
    "starter_qs_rate_diff": {
        "fn": feat_starter_qs_rate_diff, "fit": "linear",
        "eligibility": "unknown", "endpoint": "season_stats"},
}


def stage_features(game, conn, season, feature_names=None):
    """
    Compute the requested candidate features for one game.

    Args:
        game: sqlite3.Row / dict — must have game_id, *_starter_id, *_team,
              *_team_id, game_date.
        conn: open DB connection.
        season: season int for the season-keyed bdl_* lookups.
        feature_names: subset of CANDIDATE_FEATURES keys; None => all.

    Returns:
        dict feature_name -> float. Missing balldontlie data yields the
        documented neutral fallback, never a crash.
    """
    names = feature_names or list(CANDIDATE_FEATURES.keys())
    out = {}
    for name in names:
        spec = CANDIDATE_FEATURES.get(name)
        if spec is None:
            raise KeyError(f"unknown candidate feature: {name}")
        try:
            out[name] = float(spec["fn"](game, conn, season))
        except Exception as e:  # a staging feature must never break training
            logger.warning("feature %s failed for game %s: %r",
                           name, game["game_id"], e)
            out[name] = NEUTRAL_DIFF
    return out


def cache_coverage(conn):
    """Row counts of the bdl_* cache tables — quick 'is the trial loaded?' check."""
    tables = ["bdl_id_map", "bdl_pitch_type_stats", "bdl_h2h",
              "bdl_player_splits", "bdl_injuries", "bdl_season_stats"]
    cov = {}
    for t in tables:
        try:
            cov[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            cov[t] = "MISSING (run migrate.py)"
    return cov


if __name__ == "__main__":
    # Smoke test: run every feature on one real game against the current cache.
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from db import get_db
    with get_db() as c:
        print("bdl_* cache coverage:", cache_coverage(c))
        g = c.execute(
            "SELECT * FROM games WHERE status='Final' AND home_starter_id "
            "IS NOT NULL ORDER BY game_date DESC LIMIT 1").fetchone()
        if g:
            season = int(g["game_date"][:4])
            feats = stage_features(g, c, season)
            print(f"\nstaged features for {g['away_team']}@{g['home_team']} "
                  f"{g['game_date']}:")
            for k, v in feats.items():
                tag = CANDIDATE_FEATURES[k]["fit"]
                print(f"  {k:<28} {v:+.4f}   [{tag}]")
            print("\n(all neutral 0.0 expected until the bdl_* cache is "
                  "populated by the trial — except lineup_platoon_edge, "
                  "which uses local DB handedness.)")
