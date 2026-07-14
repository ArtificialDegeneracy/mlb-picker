"""Database schema migration — safely adds new columns and tables."""

import sqlite3
import sys

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "mlb_picker.db"

# Column migrations: (table, column_name, column_type)
COLUMN_MIGRATIONS = [
    ("picks", "opener_flag", "TEXT"),
    ("picks", "pick_flipped", "INTEGER DEFAULT 0"),
    ("pitcher_stats", "throw_hand", "TEXT"),
    ("team_stats", "wrc_plus_vs_lhp", "REAL"),
    ("team_stats", "wrc_plus_vs_rhp", "REAL"),
    ("games", "roof_type", "TEXT"),
    ("games", "weather_temp", "INTEGER"),
    ("games", "weather_wind", "TEXT"),
    ("games", "weather_condition", "TEXT"),
    # Per-day balldontlie snapshots — without game_date the dashboard would
    # serve stale data after day 1. home_team/away_team on bdl_odds_today let
    # the dashboard join by matchup without going through bdl_id_map.
    ("bdl_odds_today", "game_date", "TEXT"),
    ("bdl_odds_today", "home_team", "TEXT"),
    ("bdl_odds_today", "away_team", "TEXT"),
    ("bdl_batting_today", "game_date", "TEXT"),
    ("bdl_form_today", "game_date", "TEXT"),
]

# Standalone table migrations (CREATE TABLE statements that should run on existing DBs).
# For consistency with init_db's schema string in db.py.
TABLE_MIGRATIONS_EXTRA = [
    ("fip_constants", """
        CREATE TABLE IF NOT EXISTS fip_constants (
            season INTEGER PRIMARY KEY,
            fip_constant REAL NOT NULL,
            computed_at TEXT
        )
    """),
    ("fangraphs_refresh_log", """
        CREATE TABLE IF NOT EXISTS fangraphs_refresh_log (
            season INTEGER PRIMARY KEY,
            refreshed_at TEXT NOT NULL
        )
    """),
    # balldontlie API cache tables (Path B feature expansion). Empty until a
    # balldontlie ingest populates them; model/feature_staging.py reads them.
    # Must mirror the definitions in db.py:SCHEMA exactly.
    ("bdl_id_map", """
        CREATE TABLE IF NOT EXISTS bdl_id_map (
            mlb_id INTEGER,
            bdl_id INTEGER,
            entity_type TEXT,
            name TEXT,
            resolved_at TEXT DEFAULT (datetime('now')),
            match_quality TEXT,
            PRIMARY KEY (mlb_id, entity_type)
        )
    """),
    ("bdl_pitch_type_stats", """
        CREATE TABLE IF NOT EXISTS bdl_pitch_type_stats (
            player_id INTEGER,
            season INTEGER,
            role TEXT,
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
        )
    """),
    ("bdl_h2h", """
        CREATE TABLE IF NOT EXISTS bdl_h2h (
            batter_id INTEGER,
            opponent_team_id INTEGER,
            at_bats INTEGER,
            hits INTEGER,
            home_runs INTEGER,
            avg REAL,
            obp REAL,
            slg REAL,
            ops REAL,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (batter_id, opponent_team_id)
        )
    """),
    ("bdl_player_splits", """
        CREATE TABLE IF NOT EXISTS bdl_player_splits (
            player_id INTEGER,
            season INTEGER,
            split_category TEXT,
            split_name TEXT,
            split_abbreviation TEXT,
            role TEXT,
            era REAL,
            avg REAL,
            obp REAL,
            slg REAL,
            ops REAL,
            woba REAL,
            innings_pitched REAL,
            at_bats INTEGER,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (player_id, season, split_category, split_name, role)
        )
    """),
    ("bdl_injuries", """
        CREATE TABLE IF NOT EXISTS bdl_injuries (
            player_id INTEGER,
            team_id INTEGER,
            snapshot_date TEXT,
            injury_date TEXT,
            return_date TEXT,
            injury_type TEXT,
            detail TEXT,
            side TEXT,
            status TEXT,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (player_id, snapshot_date)
        )
    """),
    ("bdl_season_stats", """
        CREATE TABLE IF NOT EXISTS bdl_season_stats (
            player_id INTEGER,
            season INTEGER,
            pitching_war REAL,
            batting_war REAL,
            pitching_qs INTEGER,
            pitching_gs INTEGER,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (player_id, season)
        )
    """),
    # Per-day balldontlie snapshots — mirror db.py:SCHEMA exactly.
    ("bdl_odds_today", """
        CREATE TABLE IF NOT EXISTS bdl_odds_today (
            bdl_game_id INTEGER,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
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
        )
    """),
    ("bdl_batting_today", """
        CREATE TABLE IF NOT EXISTS bdl_batting_today (
            player_id INTEGER,
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
        )
    """),
    ("bdl_form_today", """
        CREATE TABLE IF NOT EXISTS bdl_form_today (
            player_id INTEGER,
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
        )
    """),
    # Champion/challenger shadow picks — isolated from the production `picks`
    # table. Mirror db.py:SCHEMA exactly.
    ("shadow_picks", """
        CREATE TABLE IF NOT EXISTS shadow_picks (
            game_id TEXT,
            pick_date TEXT,
            run_type TEXT,
            model_version TEXT,
            predicted_winner TEXT,
            home_win_prob REAL,
            confidence TEXT,
            actual_winner TEXT,
            correct INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (game_id, run_type, model_version)
        )
    """),
]

# Table migrations: (table_name, create_statement)
TABLE_MIGRATIONS = [
    ("game_lineups", """
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
        )
    """),
]

INDEX_MIGRATIONS = [
    "CREATE INDEX IF NOT EXISTS idx_lineups_team_date ON game_lineups(team, lineup_date)",
    "CREATE INDEX IF NOT EXISTS idx_bdl_id_map_bdl ON bdl_id_map(bdl_id, entity_type)",
    "CREATE INDEX IF NOT EXISTS idx_bdl_pitch_type_season ON bdl_pitch_type_stats(season, role)",
    "CREATE INDEX IF NOT EXISTS idx_bdl_splits_player ON bdl_player_splits(player_id, season)",
    "CREATE INDEX IF NOT EXISTS idx_shadow_picks_date ON shadow_picks(pick_date, model_version)",
    "CREATE INDEX IF NOT EXISTS idx_bdl_odds_today_date ON bdl_odds_today(game_date)",
    "CREATE INDEX IF NOT EXISTS idx_bdl_batting_today_date ON bdl_batting_today(game_date)",
    "CREATE INDEX IF NOT EXISTS idx_bdl_form_today_date ON bdl_form_today(game_date)",
]


def get_existing_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def get_existing_tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row[0] for row in rows}


# Tables that may be DROPPED and recreated if their schema is stale AND they
# hold no rows. CREATE TABLE IF NOT EXISTS cannot alter existing columns, so an
# early build of these (empty) bdl_* cache tables would otherwise keep its old
# column set forever. Safe only because these are pure caches — never source of
# truth — and the rebuild is gated on row count 0.
REBUILD_IF_EMPTY = {
    "bdl_id_map", "bdl_pitch_type_stats", "bdl_h2h",
    "bdl_player_splits", "bdl_injuries", "bdl_season_stats",
}


def run_migrations(db_path):
    conn = sqlite3.connect(db_path)
    applied = 0

    # Rebuild stale-but-empty bdl_* cache tables before the create pass.
    # Expected columns are obtained by actually creating each table in a
    # throwaway in-memory DB and asking SQLite — no fragile SQL hand-parsing.
    all_table_migrations = TABLE_MIGRATIONS + TABLE_MIGRATIONS_EXTRA
    expected_cols = {}
    _mem = sqlite3.connect(":memory:")
    for tname, csql in all_table_migrations:
        _mem.execute(csql)
        expected_cols[tname] = {
            row[1] for row in _mem.execute(f"PRAGMA table_info({tname})")
        }
    _mem.close()

    existing_tables = get_existing_tables(conn)
    for tname in REBUILD_IF_EMPTY:
        if tname in existing_tables and tname in expected_cols:
            have = get_existing_columns(conn, tname)
            if have != expected_cols[tname]:
                n = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                if n == 0:
                    conn.execute(f"DROP TABLE {tname}")
                    print(f"  Rebuilding empty table {tname} (schema changed)")
                    existing_tables.discard(tname)
                    applied += 1
                else:
                    print(f"  WARNING: {tname} schema is stale but has {n} "
                          f"rows — NOT auto-rebuilt. Migrate it by hand.")

    # Add new tables
    for table_name, create_sql in all_table_migrations:
        if table_name not in existing_tables:
            conn.execute(create_sql)
            print(f"  Created table {table_name}")
            applied += 1

    # Add new columns
    for table, column, col_type in COLUMN_MIGRATIONS:
        existing = get_existing_columns(conn, table)
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            print(f"  Added {table}.{column} ({col_type})")
            applied += 1

    # Add indices
    for idx_sql in INDEX_MIGRATIONS:
        try:
            conn.execute(idx_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

    if applied:
        print(f"  {applied} migration(s) applied")
    else:
        print("  Schema up to date")


if __name__ == "__main__":
    run_migrations(DB_PATH)
