"""Generate self-contained HTML dashboard for MLB Game Picker."""

import json
import os
import logging
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SEASON
from db import get_db

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.dirname(__file__)


def generate_dashboard(date_str=None, output_path=None):
    """Generate a self-contained HTML dashboard.

    Args:
        date_str: Date to highlight as "today" (default: actual today)
        output_path: Where to save the HTML file (default: output/mlb_picks.html)
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    output_path = output_path or os.path.join(OUTPUT_DIR, "mlb_picks.html")

    data = _gather_dashboard_data(date_str)
    html = _render_html(data, date_str)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"  Dashboard saved to {output_path}")
    return output_path


def _gather_dashboard_data(date_str):
    """Pull all data needed for the dashboard from the DB."""
    with get_db() as conn:
        # Today's picks with game info (use subqueries to avoid cartesian product)
        today_picks = conn.execute("""
            SELECT p.*, g.home_team, g.away_team, g.home_starter_name, g.away_starter_name,
                   g.game_time, g.venue, g.home_score, g.away_score, g.winner as game_winner,
                   g.status,
                   (SELECT fip FROM pitcher_stats WHERE player_id = g.home_starter_id ORDER BY season DESC LIMIT 1) as home_fip,
                   (SELECT era FROM pitcher_stats WHERE player_id = g.home_starter_id ORDER BY season DESC LIMIT 1) as home_era,
                   (SELECT MAX(innings_pitched) FROM pitcher_stats WHERE player_id = g.home_starter_id) as home_ip,
                   (SELECT k_per_9 FROM pitcher_stats WHERE player_id = g.home_starter_id ORDER BY season DESC LIMIT 1) as home_k9,
                   (SELECT throw_hand FROM pitcher_stats WHERE player_id = g.home_starter_id AND throw_hand IS NOT NULL ORDER BY season DESC LIMIT 1) as home_throw_hand,
                   (SELECT fip FROM pitcher_stats WHERE player_id = g.away_starter_id ORDER BY season DESC LIMIT 1) as away_fip,
                   (SELECT era FROM pitcher_stats WHERE player_id = g.away_starter_id ORDER BY season DESC LIMIT 1) as away_era,
                   (SELECT MAX(innings_pitched) FROM pitcher_stats WHERE player_id = g.away_starter_id) as away_ip,
                   (SELECT k_per_9 FROM pitcher_stats WHERE player_id = g.away_starter_id ORDER BY season DESC LIMIT 1) as away_k9,
                   (SELECT throw_hand FROM pitcher_stats WHERE player_id = g.away_starter_id AND throw_hand IS NOT NULL ORDER BY season DESC LIMIT 1) as away_throw_hand,
                   (SELECT wrc_plus FROM team_stats WHERE team_name = g.home_team AND wrc_plus IS NOT NULL ORDER BY season DESC LIMIT 1) as home_wrc,
                   (SELECT bullpen_era FROM team_stats WHERE team_name = g.home_team AND bullpen_era IS NOT NULL ORDER BY season DESC LIMIT 1) as home_bp_era,
                   (SELECT wrc_plus_vs_lhp FROM team_stats WHERE team_name = g.home_team AND wrc_plus_vs_lhp IS NOT NULL ORDER BY season DESC LIMIT 1) as home_wrc_vs_lhp,
                   (SELECT wrc_plus_vs_rhp FROM team_stats WHERE team_name = g.home_team AND wrc_plus_vs_rhp IS NOT NULL ORDER BY season DESC LIMIT 1) as home_wrc_vs_rhp,
                   (SELECT wrc_plus FROM team_stats WHERE team_name = g.away_team AND wrc_plus IS NOT NULL ORDER BY season DESC LIMIT 1) as away_wrc,
                   (SELECT bullpen_era FROM team_stats WHERE team_name = g.away_team AND bullpen_era IS NOT NULL ORDER BY season DESC LIMIT 1) as away_bp_era,
                   (SELECT wrc_plus_vs_lhp FROM team_stats WHERE team_name = g.away_team AND wrc_plus_vs_lhp IS NOT NULL ORDER BY season DESC LIMIT 1) as away_wrc_vs_lhp,
                   (SELECT wrc_plus_vs_rhp FROM team_stats WHERE team_name = g.away_team AND wrc_plus_vs_rhp IS NOT NULL ORDER BY season DESC LIMIT 1) as away_wrc_vs_rhp
            FROM picks p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.pick_date = ?
              AND p.run_type = (
                SELECT p2.run_type FROM picks p2
                WHERE p2.game_id = p.game_id
                ORDER BY CASE p2.run_type WHEN 'lineup_lock' THEN 0 ELSE 1 END
                LIMIT 1
              )
            ORDER BY g.game_time ASC
        """, (date_str,)).fetchall()

        # Season stats (prefer lineup_lock picks when available, else morning)
        season_stats = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN correct IS NULL THEN 1 ELSE 0 END) as pending
            FROM picks p
            WHERE pick_date >= ?
              AND run_type = (
                SELECT MAX(p2.run_type) FROM picks p2 WHERE p2.game_id = p.game_id
              )
        """, (f"{SEASON}-01-01",)).fetchone()

        # Accuracy by tier (best pick per game only)
        tier_stats = conn.execute("""
            SELECT
                confidence,
                COUNT(*) as total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as losses
            FROM picks p
            WHERE correct IS NOT NULL AND pick_date >= ?
              AND run_type = (
                SELECT p2.run_type FROM picks p2
                WHERE p2.game_id = p.game_id
                ORDER BY CASE p2.run_type WHEN 'lineup_lock' THEN 0 ELSE 1 END
                LIMIT 1
              )
            GROUP BY confidence
        """, (f"{SEASON}-01-01",)).fetchall()

        # Recent results (last 14 days with results, best pick per game)
        recent = conn.execute("""
            SELECT pick_date,
                COUNT(*) as total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
            FROM picks p
            WHERE correct IS NOT NULL AND pick_date >= ?
              AND run_type = (
                SELECT p2.run_type FROM picks p2
                WHERE p2.game_id = p.game_id
                ORDER BY CASE p2.run_type WHEN 'lineup_lock' THEN 0 ELSE 1 END
                LIMIT 1
              )
            GROUP BY pick_date
            ORDER BY pick_date DESC
            LIMIT 14
        """, (f"{SEASON}-01-01",)).fetchall()

        # All dates with picks (for history tab)
        pick_dates = conn.execute("""
            SELECT DISTINCT pick_date FROM picks
            WHERE pick_date >= ?
            ORDER BY pick_date DESC
        """, (f"{SEASON}-01-01",)).fetchall()

        # All picks for history (best pick per game, grouped by date)
        all_picks = conn.execute("""
            SELECT p.*, g.home_team, g.away_team, g.home_starter_name, g.away_starter_name,
                   g.game_time, g.home_score, g.away_score, g.winner as game_winner, g.status
            FROM picks p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.pick_date >= ?
              AND p.run_type = (
                SELECT p2.run_type FROM picks p2
                WHERE p2.game_id = p.game_id
                ORDER BY CASE p2.run_type WHEN 'lineup_lock' THEN 0 ELSE 1 END
                LIMIT 1
              )
            ORDER BY p.pick_date DESC, g.game_time ASC
        """, (f"{SEASON}-01-01",)).fetchall()

        # Current streak
        streak = _compute_streak(conn)

    # Enrich picks with series state
    today_picks_dicts = [dict(r) for r in today_picks]
    all_picks_dicts = [dict(r) for r in all_picks]
    _enrich_series_state(today_picks_dicts, date_str)
    _enrich_series_state(all_picks_dicts)

    return {
        "today_picks": today_picks_dicts,
        "season_stats": dict(season_stats) if season_stats else {"total": 0, "wins": 0, "losses": 0, "pending": 0},
        "tier_stats": {r["confidence"]: dict(r) for r in tier_stats},
        "recent": [dict(r) for r in recent],
        "pick_dates": [r["pick_date"] for r in pick_dates],
        "all_picks": all_picks_dicts,
        "streak": streak,
    }


def _enrich_series_state(picks_list, default_date=None):
    """Add series_info to each pick dict by looking at recent games between the same teams."""
    if not picks_list:
        return

    with get_db() as conn:
        for pick in picks_list:
            game_date = pick.get("pick_date") or default_date
            home = pick.get("home_team")
            away = pick.get("away_team")

            if not game_date or not home or not away:
                pick["series_info"] = None
                continue

            # Look back up to 4 days for games between these teams at this venue
            prior_games = conn.execute("""
                SELECT game_date, winner FROM games
                WHERE home_team = ? AND away_team = ?
                  AND game_date < ? AND game_date >= date(?, '-4 days')
                  AND status = 'Final' AND winner IS NOT NULL
                ORDER BY game_date ASC
            """, (home, away, game_date, game_date)).fetchall()

            if not prior_games:
                pick["series_info"] = {"game_num": 1, "label": "Game 1", "flag": None}
                continue

            # Filter to consecutive days (actual series, not random games weeks apart)
            from datetime import datetime, timedelta
            game_dt = datetime.strptime(game_date, "%Y-%m-%d")
            series = []
            for i, pg in enumerate(reversed(prior_games)):
                expected_date = (game_dt - timedelta(days=i + 1)).strftime("%Y-%m-%d")
                if pg["game_date"] == expected_date:
                    series.insert(0, pg["winner"])
                else:
                    break

            game_num = len(series) + 1
            home_wins = sum(1 for w in series if w == "home")
            away_wins = sum(1 for w in series if w == "away")

            # Build label
            if game_num == 1:
                label = "Game 1"
                flag = None
            elif game_num == 2:
                g1_winner = home if series[0] == "home" else away
                label = f"Game 2 ({g1_winner} won G1)"
                flag = "momentum"
            elif game_num == 3:
                label = f"Game 3 ({home} {home_wins}-{away_wins})"
                if home_wins == 2 or away_wins == 2:
                    sweeper = home if home_wins == 2 else away
                    label = f"Game 3 ({sweeper} sweep attempt)"
                    flag = "sweep"
                elif home_wins == 1 and away_wins == 1:
                    label = "Game 3 (series tied 1-1)"
                    flag = "rubber"
                else:
                    flag = None
            else:
                label = f"Game {game_num} ({home} {home_wins}-{away_wins})"
                flag = "sweep" if home_wins == 3 or away_wins == 3 else None

            pick["series_info"] = {
                "game_num": game_num,
                "home_wins": home_wins,
                "away_wins": away_wins,
                "label": label,
                "flag": flag,
            }


def _compute_streak(conn):
    """Compute current win/loss streak based on most recent games by date and time."""
    rows = conn.execute("""
        SELECT p.correct FROM picks p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.correct IS NOT NULL
          AND p.run_type = (
            SELECT p2.run_type FROM picks p2
            WHERE p2.game_id = p.game_id
            ORDER BY CASE p2.run_type WHEN 'lineup_lock' THEN 0 ELSE 1 END
            LIMIT 1
          )
        ORDER BY p.pick_date DESC, g.game_time DESC
        LIMIT 50
    """).fetchall()

    if not rows:
        return {"type": "none", "count": 0}

    first = rows[0]["correct"]
    count = 0
    for r in rows:
        if r["correct"] == first:
            count += 1
        else:
            break

    return {"type": "W" if first == 1 else "L", "count": count}


def _render_html(data, date_str):
    """Render the full HTML dashboard."""
    data_json = json.dumps(data, default=str)
    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLB Picker &mdash; {formatted_date}</title>
<style>
{_get_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <div class="logo">
            <span class="logo-icon">&#9918;</span>
            <div>
                <h1>MLB Picker</h1>
                <p class="subtitle">{formatted_date}</p>
            </div>
        </div>
        <div class="header-stats" id="headerStats"></div>
    </header>

    <nav class="tabs">
        <button class="tab active" data-tab="today">Today's Picks</button>
        <button class="tab" data-tab="season">Season Tracker</button>
        <button class="tab" data-tab="history">Pick History</button>
    </nav>

    <main>
        <section id="today" class="tab-content active"></section>
        <section id="season" class="tab-content"></section>
        <section id="history" class="tab-content"></section>
    </main>
</div>

<script>
const DATA = {data_json};
const TODAY = "{date_str}";
{_get_js()}
</script>
</body>
</html>"""


def _get_css():
    return """
:root {
    --bg: #0f1923;
    --surface: #1a2733;
    --surface-hover: #213040;
    --border: #2a3a4a;
    --text: #e8edf2;
    --text-dim: #8899aa;
    --accent: #4d9fff;
    --green: #34d399;
    --green-bg: rgba(52, 211, 153, 0.12);
    --yellow: #fbbf24;
    --yellow-bg: rgba(251, 191, 36, 0.12);
    --red: #f87171;
    --red-bg: rgba(248, 113, 113, 0.12);
    --gray: #6b7b8d;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    min-height: 100vh;
}

.container { max-width: 1100px; margin: 0 auto; padding: 24px 20px; }

header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid var(--border);
}
.logo { display: flex; align-items: center; gap: 14px; }
.logo-icon { font-size: 36px; }
h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
.subtitle { color: var(--text-dim); font-size: 14px; margin-top: 2px; }
.header-stats { display: flex; gap: 20px; }
.header-stat { text-align: center; }
.header-stat-value { font-size: 22px; font-weight: 700; }
.header-stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }

.tabs {
    display: flex; gap: 4px; margin-bottom: 24px;
    background: var(--surface); border-radius: 10px; padding: 4px;
}
.tab {
    flex: 1; padding: 10px 16px; border: none; background: none;
    color: var(--text-dim); font-size: 14px; font-weight: 500;
    cursor: pointer; border-radius: 8px; transition: all 0.2s;
}
.tab:hover { color: var(--text); background: var(--surface-hover); }
.tab.active { background: var(--accent); color: #fff; }

.tab-content { display: none; }
.tab-content.active { display: block; }

/* Pick Cards */
.picks-summary {
    display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;
}
.summary-badge {
    padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;
}
.summary-badge { cursor: pointer; transition: all 0.2s; opacity: 0.6; }
.summary-badge:hover { opacity: 0.85; }
.summary-badge.active { opacity: 1; box-shadow: 0 0 0 2px currentColor; }
.summary-all { background: rgba(77, 159, 255, 0.15); color: var(--accent); }
.summary-high { background: var(--green-bg); color: var(--green); }
.summary-med { background: var(--yellow-bg); color: var(--yellow); }
.summary-lean { background: rgba(107, 123, 141, 0.15); color: var(--gray); }

.cards-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 14px;
}

.pick-card {
    background: var(--surface); border-radius: 12px; padding: 18px;
    border: 1px solid var(--border); transition: transform 0.15s, border-color 0.15s;
}
.pick-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.pick-card.high { border-left: 3px solid var(--green); }
.pick-card.medium { border-left: 3px solid var(--yellow); }
.pick-card.lean { border-left: 3px solid var(--gray); }

.card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 14px;
}
.card-time { font-size: 12px; color: var(--text-dim); }
.card-confidence {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    padding: 3px 10px; border-radius: 12px; letter-spacing: 0.5px;
}
.conf-high { background: var(--green-bg); color: var(--green); }
.conf-medium { background: var(--yellow-bg); color: var(--yellow); }
.conf-lean { background: rgba(107, 123, 141, 0.15); color: var(--gray); }

.card-matchup {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 14px;
}
.team {
    flex: 1; text-align: center; padding: 8px; border-radius: 8px;
}
.team.picked { background: rgba(77, 159, 255, 0.12); border: 1px solid rgba(77, 159, 255, 0.3); }
.team-abbr { font-size: 20px; font-weight: 700; }
.team-starter { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
.team-fip { font-size: 12px; font-weight: 600; margin-top: 2px; }
.team-fip.good { color: var(--green); }
.team-fip.avg { color: var(--yellow); }
.team-fip.bad { color: var(--red); }
.vs { color: var(--text-dim); font-size: 12px; font-weight: 600; }

.card-prob {
    display: flex; align-items: center; gap: 10px; margin-top: 12px;
}
.prob-bar-bg {
    flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
}
.prob-bar {
    height: 100%; border-radius: 3px; transition: width 0.4s ease;
}
.prob-bar.high { background: var(--green); }
.prob-bar.medium { background: var(--yellow); }
.prob-bar.lean { background: var(--gray); }
.prob-value { font-size: 18px; font-weight: 700; min-width: 48px; text-align: right; }

.card-details {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border);
    font-size: 11px; color: var(--text-dim);
}
.detail-item { display: flex; justify-content: space-between; }
.detail-value { color: var(--text); font-weight: 500; }

.card-opener-warning {
    margin-top: 10px; padding: 8px 12px; border-radius: 8px;
    background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.25);
    font-size: 11px; font-weight: 500; color: var(--yellow);
    line-height: 1.4;
}

.card-reasoning {
    margin-top: 12px; padding: 10px 12px; border-radius: 8px;
    background: rgba(77, 159, 255, 0.06); border: 1px solid rgba(77, 159, 255, 0.12);
    font-size: 11px;
}
.reasoning-title {
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--accent); margin-bottom: 8px;
}
.reasoning-factors { display: flex; flex-direction: column; gap: 5px; }
.factor {
    display: flex; justify-content: space-between; align-items: center;
}
.factor-label { color: var(--text-dim); }
.factor-value { font-weight: 600; }
.factor-value.favors-pick { color: var(--green); }
.factor-value.neutral { color: var(--text-dim); }
.factor-value.against-pick { color: var(--red); }
.factor-bar {
    width: 60px; height: 4px; background: var(--border); border-radius: 2px;
    overflow: hidden; margin: 0 8px;
    flex-shrink: 0;
}
.factor-fill {
    height: 100%; border-radius: 2px;
}

.card-result {
    margin-top: 12px; padding: 8px 12px; border-radius: 8px;
    font-size: 13px; font-weight: 600; text-align: center;
}
.result-correct { background: var(--green-bg); color: var(--green); }
.result-incorrect { background: var(--red-bg); color: var(--red); }
.result-pending { background: rgba(107, 123, 141, 0.1); color: var(--text-dim); }

/* Season Tracker */
.season-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px; margin-bottom: 24px;
}
.stat-card {
    background: var(--surface); border-radius: 12px; padding: 20px;
    border: 1px solid var(--border); text-align: center;
}
.stat-value { font-size: 36px; font-weight: 800; letter-spacing: -1px; }
.stat-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

.tier-breakdown {
    background: var(--surface); border-radius: 12px; padding: 20px;
    border: 1px solid var(--border); margin-bottom: 24px;
}
.tier-breakdown h3 { font-size: 16px; margin-bottom: 16px; }
.tier-row {
    display: flex; align-items: center; gap: 14px; padding: 10px 0;
    border-bottom: 1px solid var(--border);
}
.tier-row:last-child { border-bottom: none; }
.tier-label { width: 70px; font-weight: 600; font-size: 13px; }
.tier-bar-bg { flex: 1; height: 24px; background: var(--border); border-radius: 6px; overflow: hidden; position: relative; }
.tier-bar { height: 100%; border-radius: 6px; min-width: 2px; transition: width 0.5s ease; }
.tier-bar-text { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: 600; }
.tier-record { width: 60px; text-align: right; font-size: 13px; color: var(--text-dim); }

.recent-section {
    background: var(--surface); border-radius: 12px; padding: 20px;
    border: 1px solid var(--border);
}
.recent-section h3 { font-size: 16px; margin-bottom: 16px; }
.recent-days { display: flex; gap: 6px; flex-wrap: wrap; }
.recent-day {
    width: 44px; height: 44px; border-radius: 8px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; font-size: 10px; font-weight: 600;
}
.recent-day .day-date { font-size: 9px; color: var(--text-dim); }
.recent-day.good { background: var(--green-bg); color: var(--green); }
.recent-day.ok { background: var(--yellow-bg); color: var(--yellow); }
.recent-day.bad { background: var(--red-bg); color: var(--red); }

/* History */
.history-controls {
    display: flex; gap: 12px; margin-bottom: 20px; align-items: center; flex-wrap: wrap;
}
.date-select {
    padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); font-size: 14px;
    cursor: pointer;
}
.date-select:focus { outline: none; border-color: var(--accent); }
.filter-btn {
    padding: 6px 14px; border-radius: 20px; border: 1px solid var(--border);
    background: none; color: var(--text-dim); font-size: 12px; cursor: pointer;
    transition: all 0.2s;
}
.filter-btn:hover, .filter-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

.history-summary {
    display: flex; gap: 16px; margin-bottom: 16px; font-size: 13px; color: var(--text-dim);
}
.history-summary span { font-weight: 600; color: var(--text); }

.no-data {
    text-align: center; padding: 60px 20px; color: var(--text-dim);
    font-size: 15px;
}
.no-data-icon { font-size: 48px; margin-bottom: 12px; }

@media (max-width: 700px) {
    .cards-grid { grid-template-columns: 1fr; }
    .season-grid { grid-template-columns: repeat(2, 1fr); }
    header { flex-direction: column; align-items: flex-start; gap: 12px; }
}
"""


def _get_js():
    return """
// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
    });
});

function fipClass(fip) {
    if (!fip) return 'avg';
    if (fip < 3.5) return 'good';
    if (fip < 4.2) return 'avg';
    return 'bad';
}

function truncName(name, max) {
    if (!name) return 'TBD';
    return name.length > max ? name.substring(0, max) + '.' : name;
}

const PARK_FACTORS = {
    "COL": 1.32, "ARI": 1.09, "BOS": 1.08, "CIN": 1.07, "TEX": 1.06,
    "CHC": 1.05, "PHI": 1.04, "ATL": 1.03, "MIL": 1.02, "TOR": 1.02,
    "LAA": 1.01, "MIN": 1.01, "BAL": 1.01, "NYY": 1.00, "CLE": 1.00,
    "DET": 1.00, "HOU": 0.99, "KC": 0.99, "STL": 0.99, "CWS": 0.99,
    "WSH": 0.98, "PIT": 0.98, "SF": 0.97, "NYM": 0.97, "SD": 0.96,
    "LAD": 0.96, "SEA": 0.96, "TB": 0.95, "MIA": 0.94, "OAK": 0.95
};

function buildReasoning(p, isHomePick) {
    const pick = p.predicted_winner;
    const opp = isHomePick ? p.away_team : p.home_team;
    const factors = [];

    // 1. Starting pitching (FIP)
    if (p.home_fip && p.away_fip) {
        const diff = Math.abs(p.home_fip - p.away_fip);
        const pickFip = isHomePick ? p.home_fip : p.away_fip;
        const oppFip = isHomePick ? p.away_fip : p.home_fip;
        const favors = pickFip < oppFip; // lower FIP is better
        let strength = diff < 0.3 ? 'neutral' : (diff < 0.8 ? 'mild' : 'strong');
        let label = `SP: ${pickFip.toFixed(2)} vs ${oppFip.toFixed(2)} FIP`;
        if (diff >= 0.3) label += favors ? ` (${pick} edge)` : ` (${opp} edge)`;
        factors.push({ label, favors: diff < 0.3 ? 'neutral' : (favors ? 'pick' : 'against'), strength: diff, maxStrength: 2.0 });
    }

    // 2. Platoon matchup
    const awayHand = p.away_throw_hand;
    const homeHand = p.home_throw_hand;
    if (awayHand || homeHand) {
        // Home team hits vs away starter's hand, away team hits vs home starter's hand
        const homeWrcVsHand = awayHand === 'L' ? p.home_wrc_vs_lhp : p.home_wrc_vs_rhp;
        const awayWrcVsHand = homeHand === 'L' ? p.away_wrc_vs_lhp : p.away_wrc_vs_rhp;

        if (homeWrcVsHand && awayWrcVsHand) {
            const pickWrc = isHomePick ? homeWrcVsHand : awayWrcVsHand;
            const oppWrc = isHomePick ? awayWrcVsHand : homeWrcVsHand;
            const diff = Math.abs(pickWrc - oppWrc);
            const favors = pickWrc > oppWrc;
            const pickHand = isHomePick ? (awayHand || '?') : (homeHand || '?');
            const oppHand = isHomePick ? (homeHand || '?') : (awayHand || '?');
            let label = `Platoon: ${pick} ${pickWrc.toFixed(0)} wRC+ vs ${pickHand}HP`;
            factors.push({ label, favors: diff < 5 ? 'neutral' : (favors ? 'pick' : 'against'), strength: diff, maxStrength: 40 });
        }
    }

    // 3. Bullpen
    if (p.home_bp_era && p.away_bp_era) {
        const diff = Math.abs(p.home_bp_era - p.away_bp_era);
        const pickBp = isHomePick ? p.home_bp_era : p.away_bp_era;
        const oppBp = isHomePick ? p.away_bp_era : p.home_bp_era;
        const favors = pickBp < oppBp; // lower ERA is better
        let label = `Bullpen: ${pickBp.toFixed(2)} vs ${oppBp.toFixed(2)} ERA`;
        factors.push({ label, favors: diff < 0.2 ? 'neutral' : (favors ? 'pick' : 'against'), strength: diff, maxStrength: 1.5 });
    }

    // 4. Park factor
    const pf = PARK_FACTORS[p.home_team] || 1.0;
    if (Math.abs(pf - 1.0) >= 0.03) {
        const parkDesc = pf > 1.02 ? 'Hitter-friendly' : 'Pitcher-friendly';
        const label = `Park: ${parkDesc} (${(pf * 100).toFixed(0)}%)`;
        factors.push({ label, favors: 'neutral', strength: 0, maxStrength: 1 });
    }

    // 5. Home field
    const hfLabel = isHomePick ? `Home field: ${pick} at home` : `Road pick: ${pick} away`;
    factors.push({ label: hfLabel, favors: isHomePick ? 'pick' : 'against', strength: isHomePick ? 0.5 : 0.5, maxStrength: 1 });

    // 6. Series context
    if (p.series_info && p.series_info.game_num > 1) {
        let seriesLabel = p.series_info.label;
        let seriesFavors = 'neutral';
        if (p.series_info.flag === 'sweep') {
            // Sweep attempts favor the trailing team (sweep resistance)
            const sweeper = p.series_info.home_wins > p.series_info.away_wins ? p.home_team : p.away_team;
            seriesFavors = (pick === sweeper) ? 'against' : 'pick';
            seriesLabel += ' ⚠';
        } else if (p.series_info.flag === 'momentum' && p.series_info.game_num === 2) {
            // G1 winner has momentum in G2
            const g1Winner = p.series_info.home_wins > 0 ? p.home_team : p.away_team;
            seriesFavors = (pick === g1Winner) ? 'pick' : 'against';
        }
        factors.push({ label: seriesLabel, favors: seriesFavors, strength: 0.5, maxStrength: 1 });
    }

    if (factors.length === 0) return '';

    let html = `<div class="card-reasoning"><div class="reasoning-title">Model Reasoning</div><div class="reasoning-factors">`;
    factors.forEach(f => {
        const cls = f.favors === 'pick' ? 'favors-pick' : (f.favors === 'against' ? 'against-pick' : 'neutral');
        const icon = f.favors === 'pick' ? '&#9650;' : (f.favors === 'against' ? '&#9660;' : '&#9679;');
        html += `<div class="factor">
            <span class="factor-label">${f.label}</span>
            <span class="factor-value ${cls}">${icon}</span>
        </div>`;
    });
    html += `</div></div>`;
    return html;
}

function renderHeaderStats() {
    const s = DATA.season_stats;
    const total = (s.wins || 0) + (s.losses || 0);
    const pct = total > 0 ? ((s.wins / total) * 100).toFixed(1) : '--';
    const streak = DATA.streak;
    const streakText = streak.count > 0 ? streak.type + streak.count : '--';

    document.getElementById('headerStats').innerHTML = `
        <div class="header-stat">
            <div class="header-stat-value">${total > 0 ? s.wins + '-' + s.losses : '--'}</div>
            <div class="header-stat-label">Record</div>
        </div>
        <div class="header-stat">
            <div class="header-stat-value">${pct}%</div>
            <div class="header-stat-label">Accuracy</div>
        </div>
        <div class="header-stat">
            <div class="header-stat-value">${streakText}</div>
            <div class="header-stat-label">Streak</div>
        </div>
    `;
}

function renderTodayTab() {
    const picks = DATA.today_picks;
    if (!picks || picks.length === 0) {
        document.getElementById('today').innerHTML = `
            <div class="no-data">
                <div class="no-data-icon">&#9918;</div>
                No picks for today yet.<br>Run <code>python main.py predict</code> to generate picks.
            </div>`;
        return;
    }

    const counts = { HIGH: 0, MEDIUM: 0, LEAN: 0 };
    picks.forEach(p => counts[p.confidence] = (counts[p.confidence] || 0) + 1);

    let html = `<div class="picks-summary">
        <span class="summary-badge summary-all active" data-filter="ALL" onclick="filterTodayPicks('ALL')">All ${picks.length}</span>
        <span class="summary-badge summary-high" data-filter="HIGH" onclick="filterTodayPicks('HIGH')">${counts.HIGH} High</span>
        <span class="summary-badge summary-med" data-filter="MEDIUM" onclick="filterTodayPicks('MEDIUM')">${counts.MEDIUM} Medium</span>
        <span class="summary-badge summary-lean" data-filter="LEAN" onclick="filterTodayPicks('LEAN')">${counts.LEAN} Lean</span>
    </div><div class="cards-grid" id="todayCards">`;

    picks.forEach(p => {
        const conf = p.confidence.toLowerCase();
        const isHomePick = p.predicted_winner === p.home_team;
        const pickProb = isHomePick ? p.home_win_prob : (1 - p.home_win_prob);
        const probPct = (pickProb * 100).toFixed(0);
        const barWidth = Math.max(probPct - 40, 5);

        // Opener detection — uses flag from Python model (checks career stats across seasons)
        const openerFlag = p.opener_flag;
        let openerHtml = '';
        if (openerFlag) {
            let who, team;
            if (openerFlag === 'home' || openerFlag === 'both') {
                who = p.home_starter_name;
                team = p.home_team;
            } else {
                who = p.away_starter_name;
                team = p.away_team;
            }
            const extra = openerFlag === 'both' ? ' (both teams)' : '';
            openerHtml = `<div class="card-opener-warning">&#9888; Probable opener: ${who} (${team})${extra} — confidence dampened, check who pitches after</div>`;
        }

        let resultHtml = '';
        if (p.status === 'Final' && p.game_winner) {
            const correct = p.predicted_winner === (p.game_winner === 'home' ? p.home_team : p.away_team);
            resultHtml = correct
                ? `<div class="card-result result-correct">&#10003; Correct &mdash; ${p.away_team} ${p.away_score}, ${p.home_team} ${p.home_score}</div>`
                : `<div class="card-result result-incorrect">&#10007; Incorrect &mdash; ${p.away_team} ${p.away_score}, ${p.home_team} ${p.home_score}</div>`;
        } else {
            resultHtml = `<div class="card-result result-pending">Scheduled</div>`;
        }

        // Build model reasoning factors
        const reasoningHtml = buildReasoning(p, isHomePick);

        html += `
        <div class="pick-card ${conf}" data-confidence="${p.confidence}">
            <div class="card-header">
                <span class="card-time">${p.game_time || ''} &bull; ${p.venue || ''}</span>
                <span class="card-confidence conf-${conf}">${p.confidence}</span>
            </div>
            <div class="card-matchup">
                <div class="team ${!isHomePick ? 'picked' : ''}">
                    <div class="team-abbr">${p.away_team}</div>
                    <div class="team-starter">${truncName(p.away_starter_name, 14)}</div>
                    <div class="team-fip ${fipClass(p.away_fip)}">${p.away_fip ? p.away_fip.toFixed(2) + ' FIP' : ''}</div>
                </div>
                <div class="vs">@</div>
                <div class="team ${isHomePick ? 'picked' : ''}">
                    <div class="team-abbr">${p.home_team}</div>
                    <div class="team-starter">${truncName(p.home_starter_name, 14)}</div>
                    <div class="team-fip ${fipClass(p.home_fip)}">${p.home_fip ? p.home_fip.toFixed(2) + ' FIP' : ''}</div>
                </div>
            </div>
            <div class="card-prob">
                <div class="prob-bar-bg">
                    <div class="prob-bar ${conf}" style="width: ${barWidth}%"></div>
                </div>
                <div class="prob-value">${probPct}%</div>
            </div>
            ${openerHtml}
            ${reasoningHtml}
            ${resultHtml}
        </div>`;
    });

    html += '</div>';
    document.getElementById('today').innerHTML = html;
}

function renderSeasonTab() {
    const s = DATA.season_stats;
    const total = (s.wins || 0) + (s.losses || 0);
    const pct = total > 0 ? ((s.wins / total) * 100).toFixed(1) : '0.0';
    const streak = DATA.streak;
    const pending = s.pending || 0;

    let html = `<div class="season-grid">
        <div class="stat-card">
            <div class="stat-value">${total > 0 ? s.wins + '-' + s.losses : '0-0'}</div>
            <div class="stat-label">Season Record</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${pct}%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${total + pending}</div>
            <div class="stat-label">Total Picks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: ${streak.type === 'W' ? 'var(--green)' : streak.type === 'L' ? 'var(--red)' : 'var(--text-dim)'}">${streak.count > 0 ? streak.type + streak.count : '--'}</div>
            <div class="stat-label">Current Streak</div>
        </div>
    </div>`;

    // Tier breakdown
    const tiers = ['HIGH', 'MEDIUM', 'LEAN'];
    const tierColors = { HIGH: 'var(--green)', MEDIUM: 'var(--yellow)', LEAN: 'var(--gray)' };
    html += `<div class="tier-breakdown"><h3>Accuracy by Confidence Tier</h3>`;

    tiers.forEach(tier => {
        const t = DATA.tier_stats[tier] || { total: 0, wins: 0, losses: 0 };
        const pct = t.total > 0 ? ((t.wins / t.total) * 100).toFixed(1) : 0;
        html += `
        <div class="tier-row">
            <div class="tier-label" style="color: ${tierColors[tier]}">${tier}</div>
            <div class="tier-bar-bg">
                <div class="tier-bar" style="width: ${pct}%; background: ${tierColors[tier]}"></div>
                <span class="tier-bar-text">${pct}%</span>
            </div>
            <div class="tier-record">${t.wins}-${t.losses}</div>
        </div>`;
    });
    html += '</div>';

    // Recent results heatmap
    if (DATA.recent.length > 0) {
        html += `<div class="recent-section"><h3>Recent Days</h3><div class="recent-days">`;
        DATA.recent.forEach(d => {
            const pct = d.total > 0 ? d.wins / d.total : 0;
            const cls = pct >= 0.65 ? 'good' : pct >= 0.5 ? 'ok' : 'bad';
            const dateShort = d.pick_date.substring(5); // MM-DD
            html += `<div class="recent-day ${cls}" title="${d.pick_date}: ${d.wins}/${d.total}">
                ${d.wins}/${d.total}
                <span class="day-date">${dateShort}</span>
            </div>`;
        });
        html += '</div></div>';
    } else {
        html += `<div class="no-data"><div class="no-data-icon">&#128202;</div>No results yet. Check back after games are scored.</div>`;
    }

    document.getElementById('season').innerHTML = html;
}

function renderHistoryTab() {
    const dates = DATA.pick_dates;
    if (!dates || dates.length === 0) {
        document.getElementById('history').innerHTML = `
            <div class="no-data"><div class="no-data-icon">&#128197;</div>No pick history yet.</div>`;
        return;
    }

    let html = `<div class="history-controls">
        <select class="date-select" id="historyDate">
            ${dates.map(d => `<option value="${d}" ${d === TODAY ? 'selected' : ''}>${d}</option>`).join('')}
        </select>
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="HIGH">High</button>
        <button class="filter-btn" data-filter="MEDIUM">Medium</button>
        <button class="filter-btn" data-filter="LEAN">Lean</button>
    </div>
    <div class="history-summary" id="historySummary"></div>
    <div class="cards-grid" id="historyCards"></div>`;

    document.getElementById('history').innerHTML = html;

    // Wire up controls
    document.getElementById('historyDate').addEventListener('change', renderHistoryCards);
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderHistoryCards();
        });
    });

    renderHistoryCards();
}

function renderHistoryCards() {
    const selectedDate = document.getElementById('historyDate').value;
    const filter = document.querySelector('.filter-btn.active').dataset.filter;

    let picks = DATA.all_picks.filter(p => p.pick_date === selectedDate);
    if (filter !== 'all') picks = picks.filter(p => p.confidence === filter);

    const scored = picks.filter(p => p.correct !== null);
    const wins = scored.filter(p => p.correct === 1).length;
    const losses = scored.filter(p => p.correct === 0).length;

    document.getElementById('historySummary').innerHTML = picks.length > 0
        ? `<div>${picks.length} picks &bull; <span>${wins}W-${losses}L</span>${scored.length < picks.length ? ` &bull; ${picks.length - scored.length} pending` : ''}</div>`
        : '';

    let html = '';
    picks.forEach(p => {
        const conf = p.confidence.toLowerCase();
        const isHomePick = p.predicted_winner === p.home_team;
        const pickProb = isHomePick ? p.home_win_prob : (1 - p.home_win_prob);
        const probPct = (pickProb * 100).toFixed(0);

        let resultHtml = '';
        if (p.correct === 1) {
            resultHtml = `<div class="card-result result-correct">&#10003; Correct &mdash; ${p.away_team} ${p.away_score || '?'}, ${p.home_team} ${p.home_score || '?'}</div>`;
        } else if (p.correct === 0) {
            resultHtml = `<div class="card-result result-incorrect">&#10007; Incorrect &mdash; ${p.away_team} ${p.away_score || '?'}, ${p.home_team} ${p.home_score || '?'}</div>`;
        } else {
            resultHtml = `<div class="card-result result-pending">Pending</div>`;
        }

        html += `
        <div class="pick-card ${conf}">
            <div class="card-header">
                <span class="card-time">${p.game_time || ''}</span>
                <span class="card-confidence conf-${conf}">${p.confidence}</span>
            </div>
            <div class="card-matchup">
                <div class="team ${!isHomePick ? 'picked' : ''}">
                    <div class="team-abbr">${p.away_team}</div>
                    <div class="team-starter">${truncName(p.away_starter_name, 14)}</div>
                </div>
                <div class="vs">@</div>
                <div class="team ${isHomePick ? 'picked' : ''}">
                    <div class="team-abbr">${p.home_team}</div>
                    <div class="team-starter">${truncName(p.home_starter_name, 14)}</div>
                </div>
            </div>
            <div class="card-prob">
                <div class="prob-bar-bg">
                    <div class="prob-bar ${conf}" style="width: ${Math.max(probPct - 40, 5)}%"></div>
                </div>
                <div class="prob-value">${probPct}%</div>
            </div>
            ${resultHtml}
        </div>`;
    });

    document.getElementById('historyCards').innerHTML = html || '<div class="no-data">No picks match this filter.</div>';
}

function filterTodayPicks(tier) {
    // Update active badge
    document.querySelectorAll('.picks-summary .summary-badge').forEach(b => {
        b.classList.toggle('active', b.dataset.filter === tier);
    });

    // Show/hide cards
    document.querySelectorAll('#todayCards .pick-card').forEach(card => {
        if (tier === 'ALL' || card.dataset.confidence === tier) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });
}

// Init
renderHeaderStats();
renderTodayTab();
renderSeasonTab();
renderHistoryTab();
"""
