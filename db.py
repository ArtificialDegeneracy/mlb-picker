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
    home_score INTEGER,
    away_score INTEGER,
    winner TEXT,
    status TEXT,
    created_at TEXT DEFAULT (datetime('now'))
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
    ops REAL,
    wrc_plus REAL,
    wrc_plus_vs_lhp REAL,
    wrc_plus_vs_rhp REAL,
    team_era REAL,
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
    PRIMARY KEY (game_id, run_type)
);

CREATE TABLE IF NOT EXISTS win_total_priors (
    team_name TEXT PRIMARY KEY,
    projected_wins INTEGER,
    season INTEGER
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(pick_date);
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
