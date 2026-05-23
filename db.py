"""SQLite database connection and schema management."""

import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "mlb_picker.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    game_date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_starter_id INTEGER,
    away_starter_id INTEGER,
    home_starter_name TEXT,
    away_starter_name TEXT,
    game_time TEXT,
    venue TEXT,
    roof_type TEXT,
    weather_temp INTEGER,
    weather_wind TEXT,
    weather_condition TEXT,
    home_score INTEGER,
    away_score INTEGER,
    winner TEXT,
    status TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS game_lineups (
    game_id TEXT,
    team TEXT,
    player_id INTEGER,
    lineup_position INTEGER,
    player_name TEXT,
    bat_side TEXT,
    ops_vs_lhp REAL,
    ops_vs_rhp REAL,
    lineup_date TEXT,
    PRIMARY KEY (game_id, team, lineup_position)
);

CREATE TABLE IF NOT EXISTS pitcher_stats (
    player_id INTEGER,
    player_name TEXT,
    team TEXT,
    season INTEGER,
    era REAL,
    fip REAL,
    xfip REAL,
    k_per_9 REAL,
    bb_per_9 REAL,
    innings_pitched REAL,
    hits INTEGER,
    home_runs INTEGER,
    walks INTEGER,
    hbp INTEGER,
    strikeouts INTEGER,
    games_started INTEGER,
    throw_hand TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS team_stats (
    team_id INTEGER,
    team_name TEXT,
    season INTEGER,
    wins INTEGER,
    losses INTEGER,
    wrc_plus REAL,
    wrc_plus_vs_lhp REAL,
    wrc_plus_vs_rhp REAL,
    bullpen_era REAL,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (team_id, season)
);

CREATE TABLE IF NOT EXISTS batter_splits (
    player_id INTEGER,
    player_name TEXT,
    bat_side TEXT,
    season INTEGER,
    ops_vs_lhp REAL,
    ops_vs_rhp REAL,
    ab_vs_lhp INTEGER,
    ab_vs_rhp INTEGER,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS picks (
    game_id TEXT,
    pick_date TEXT,
    run_type TEXT,
    predicted_winner TEXT,
    home_win_prob REAL,
    confidence TEXT,
    actual_winner TEXT,
    correct INTEGER,
    opener_flag TEXT,
    pick_flipped INTEGER DEFAULT 0,
    PRIMARY KEY (game_id, run_type)
);

CREATE TABLE IF NOT EXISTS win_total_priors (
    team_name TEXT PRIMARY KEY,
    projected_wins INTEGER,
    season INTEGER
);

-- Champion/challenger shadow picks. Production keeps writing to `picks`;
-- challenger models (XGBoost, balldontlie-fed variants) write here instead.
-- This table is NEVER read by the dashboard, run_results, or model training —
-- it exists purely so a challenger can be scored alongside production without
-- any risk of contaminating the real contest tracker. model_version is part
-- of the key, so multiple challengers can coexist for the same game.
CREATE TABLE IF NOT EXISTS shadow_picks (
    game_id TEXT,
    pick_date TEXT,
    run_type TEXT,
    model_version TEXT,        -- e.g. 'xgb_5feat', 'xgb_bdl', 'logreg_baseline'
    predicted_winner TEXT,
    home_win_prob REAL,
    confidence TEXT,
    actual_winner TEXT,        -- filled by the shadow scorer (mirrors run_results)
    correct INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (game_id, run_type, model_version)
);

CREATE TABLE IF NOT EXISTS fip_constants (
    season INTEGER PRIMARY KEY,
    fip_constant REAL NOT NULL,
    computed_at TEXT
);

-- balldontlie API cache tables (Path B feature expansion).
-- These store advanced data balldontlie provides that the MLB Stats API /
-- FanGraphs sources don't. They are SEPARATE from the core pipeline tables:
-- model/feature_staging.py reads from them, but data/mlb_api.py and
-- data/fangraphs.py never touch them. Populated by a (future) balldontlie
-- ingest; until then they are empty and feature_staging falls back gracefully.
--
-- player_id / team_id here are balldontlie's OWN ids, not MLB Stats API ids.
-- The crosswalk lives in bdl_id_map.

CREATE TABLE IF NOT EXISTS bdl_id_map (
    mlb_id INTEGER,            -- MLB Stats API id (what the rest of the DB uses)
    bdl_id INTEGER,            -- balldontlie's internal id
    entity_type TEXT,          -- 'player' | 'team'
    name TEXT,                 -- for audit / fuzzy-match review
    resolved_at TEXT DEFAULT (datetime('now')),
    match_quality TEXT,        -- 'exact' | 'fuzzy'
    PRIMARY KEY (mlb_id, entity_type)
);

CREATE TABLE IF NOT EXISTS bdl_pitch_type_stats (
    player_id INTEGER,         -- balldontlie id
    season INTEGER,
    role TEXT,                 -- 'pitcher' | 'hitter' (which endpoint it came from)
    pitch_type TEXT,
    pitch_usage_percent REAL,
    whiff_percent REAL,
    chase_percent REAL,
    zone_percent REAL,
    ba REAL,
    slg REAL,
    woba REAL,
    xwoba REAL,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, season, role, pitch_type)
);

CREATE TABLE IF NOT EXISTS bdl_h2h (
    batter_id INTEGER,         -- balldontlie id
    opponent_team_id INTEGER,  -- balldontlie id
    at_bats INTEGER,
    hits INTEGER,
    home_runs INTEGER,
    avg REAL,
    obp REAL,
    slg REAL,
    ops REAL,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (batter_id, opponent_team_id)
);

-- /players/splits returns data keyed by category (e.g. byArena confirmed in
-- the docs; vs-hand / by-month keys exist but their exact strings are not in
-- the truncated docs — the probe confirms them). Each split row carries
-- split_category / split_name / split_abbreviation plus batting AND pitching
-- fields. split_name holds values like 'vs RHP'; PROBE-CONFIRM the exact
-- strings before the ingest relies on them.
CREATE TABLE IF NOT EXISTS bdl_player_splits (
    player_id INTEGER,         -- balldontlie id
    season INTEGER,
    split_category TEXT,       -- API 'split_category' (the grouping)
    split_name TEXT,           -- API 'split_name' e.g. 'vs RHP', 'April'
    split_abbreviation TEXT,   -- API 'split_abbreviation'
    role TEXT,                 -- 'pitching' | 'batting' (which fields are populated)
    era REAL,
    avg REAL,
    obp REAL,
    slg REAL,
    ops REAL,
    woba REAL,                 -- may be null — balldontlie does not always populate
    innings_pitched REAL,
    at_bats INTEGER,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, season, split_category, split_name, role)
);

CREATE TABLE IF NOT EXISTS bdl_injuries (
    player_id INTEGER,         -- balldontlie id
    team_id INTEGER,           -- balldontlie id
    snapshot_date TEXT,        -- date this injury snapshot was pulled
    injury_date TEXT,
    return_date TEXT,
    injury_type TEXT,
    detail TEXT,
    side TEXT,
    status TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, snapshot_date)
);

-- Column names mirror the API's season_stats field names exactly (pitching_gs,
-- pitching_qs, ...) so the ingest is a direct field->column copy.
CREATE TABLE IF NOT EXISTS bdl_season_stats (
    player_id INTEGER,         -- balldontlie id
    season INTEGER,
    pitching_war REAL,
    batting_war REAL,
    pitching_qs INTEGER,
    pitching_gs INTEGER,       -- API 'pitching_gs' (games started)
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, season)
);

-- "_today" tables — per-day balldontlie snapshots populated by
-- data/balldontlie.py:ingest_for_date. The game_date column is mandatory: the
-- dashboard scopes its reads by game_date, otherwise day-2 reads serve stale
-- day-1 rows. The PK includes game_date so re-ingest on the same day is
-- idempotent (INSERT OR REPLACE) but different days coexist.
CREATE TABLE IF NOT EXISTS bdl_odds_today (
    bdl_game_id INTEGER,
    game_date TEXT,            -- 'YYYY-MM-DD' — the slate this row belongs to
    home_team TEXT,            -- project abbr (CWS not CHW)
    away_team TEXT,            -- project abbr
    vendor TEXT,
    moneyline_home_odds INTEGER,
    moneyline_away_odds INTEGER,
    total_value REAL,
    total_over_odds INTEGER,
    total_under_odds INTEGER,
    spread_home_value REAL,
    spread_home_odds INTEGER,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (bdl_game_id, vendor, game_date)
);

CREATE TABLE IF NOT EXISTS bdl_batting_today (
    player_id INTEGER,         -- balldontlie id
    game_date TEXT,
    full_name TEXT,
    team TEXT,
    gp INTEGER,
    avg REAL,
    obp REAL,
    slg REAL,
    ops REAL,
    hr INTEGER,
    rbi INTEGER,
    sb INTEGER,
    war REAL,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, game_date)
);

CREATE TABLE IF NOT EXISTS bdl_form_today (
    player_id INTEGER,         -- balldontlie id
    game_date TEXT,
    season_ops REAL,
    season_ab INTEGER,
    last7_ops REAL,
    last7_ab INTEGER,
    last15_ops REAL,
    last15_ab INTEGER,
    last30_ops REAL,
    last30_ab INTEGER,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(pick_date);
CREATE INDEX IF NOT EXISTS idx_picks_game_id ON picks(game_id);
CREATE INDEX IF NOT EXISTS idx_pitcher_stats_player ON pitcher_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_team_stats_name ON team_stats(team_name);
CREATE INDEX IF NOT EXISTS idx_lineups_team_date ON game_lineups(team, lineup_date);
CREATE INDEX IF NOT EXISTS idx_bdl_id_map_bdl ON bdl_id_map(bdl_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_bdl_pitch_type_season ON bdl_pitch_type_stats(season, role);
CREATE INDEX IF NOT EXISTS idx_bdl_splits_player ON bdl_player_splits(player_id, season);
CREATE INDEX IF NOT EXISTS idx_shadow_picks_date ON shadow_picks(pick_date, model_version);
CREATE INDEX IF NOT EXISTS idx_bdl_odds_today_date ON bdl_odds_today(game_date);
CREATE INDEX IF NOT EXISTS idx_bdl_batting_today_date ON bdl_batting_today(game_date);
CREATE INDEX IF NOT EXISTS idx_bdl_form_today_date ON bdl_form_today(game_date);
"""


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables and indexes."""
    with get_db() as conn:
        conn.executescript(SCHEMA)
    print(f"Database initialized at {DB_PATH}")


def seed_priors():
    """Insert preseason win total priors from config."""
    from config import WIN_TOTAL_PRIORS, SEASON
    with get_db() as conn:
        for team, wins in WIN_TOTAL_PRIORS.items():
            conn.execute(
                "INSERT OR REPLACE INTO win_total_priors (team_name, projected_wins, season) VALUES (?, ?, ?)",
                (team, wins, SEASON),
            )
    print(f"Seeded {len(WIN_TOTAL_PRIORS)} team priors for {SEASON}")


def get_row_counts():
    """Print row counts for all tables."""
    tables = ["games", "pitcher_stats", "team_stats", "picks", "win_total_priors"]
    with get_db() as conn:
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} rows")
