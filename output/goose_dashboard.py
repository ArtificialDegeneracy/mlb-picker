"""Goose's Projection System — branded dashboard.

Mobile-first pub-themed predictions site, populated with cached balldontlie +
model picks from the production pipeline.

Design source: goose-s-projection-system-design (Claude Design bundle).
  - Edge Meter: model-vs-Vegas magnitude + comparison track
  - Typed edges: Model / Lineup / Pitching / Form / Injury, each colored
  - 5-pint confidence scale (the brand's signature indicator)
  - Two tabs: Tonight's slate + Season Tracker

Pure renderer. Reads the picks table (deduped lineup_lock > morning), games,
game_lineups, and the balldontlie cache tables (bdl_odds_today,
bdl_batting_today, bdl_pitch_type_stats, bdl_form_today, bdl_injuries).
Self-contained HTML output (logo embedded as base64).
"""
import base64
import datetime as _dt
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_current_season, PARK_FACTORS
from db import get_db
from model.features import _get_pitcher_fip, _get_offense_trend
from output.goose_props import (gather_prop_board, gather_prop_edges_for_game,
                                K_OVER_THRESHOLD, K_UNDER_THRESHOLD)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets", "goose")
DEFAULT_OUT = os.path.join(HERE, "goose_dashboard.html")
LOGO = os.path.join(ASSETS, "hat-logo-sm.png")
TOKENS_CSS = os.path.join(ASSETS, "colors_and_type.css")

PITCH_NAME = {"FF": "4-seam", "SI": "sinker", "FC": "cutter", "SL": "slider",
              "ST": "sweeper", "CH": "change", "CU": "curve", "FS": "splitter",
              "KC": "knuckle-curve", "SV": "slurve"}

# Prefer lineup_lock over morning for a given game — the production dedup
# pattern documented in CLAUDE.md.
_DEDUP = ("p.run_type = (SELECT p2.run_type FROM picks p2 "
          "WHERE p2.game_id=p.game_id ORDER BY CASE p2.run_type "
          "WHEN 'lineup_lock' THEN 0 ELSE 1 END LIMIT 1)")


# --- odds helpers ----------------------------------------------------------

def implied(odds):
    if odds is None:
        return None
    return (-odds) / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def prob_to_american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def median_odds(vals):
    probs = sorted(implied(o) for o in vals if o is not None)
    if not probs:
        return None
    return prob_to_american(probs[len(probs) // 2])


def vegas_consensus(conn, away, home, date_str):
    """Vegas consensus odds for an (away, home) matchup on a given date.

    Joins bdl_odds_today through bdl_id_map to resolve team abbreviations to
    balldontlie team ids, then medians across the books. Returns None if no
    odds rows exist for the slate (game hasn't been posted yet).
    """
    rows = conn.execute(
        "SELECT o.moneyline_home_odds, o.moneyline_away_odds, o.total_value "
        "FROM bdl_odds_today o "
        "WHERE o.game_date = ? AND o.home_team = ? AND o.away_team = ?",
        (date_str, home, away)).fetchall()
    hi, ai, tot, hml, aml = [], [], [], [], []
    for r in rows:
        a_i, h_i = implied(r["moneyline_away_odds"]), implied(r["moneyline_home_odds"])
        if a_i and h_i:
            hi.append(h_i)
            ai.append(a_i)
            hml.append(r["moneyline_home_odds"])
            aml.append(r["moneyline_away_odds"])
        if r["total_value"] is not None:
            tot.append(r["total_value"])
    if not hi:
        return None
    h, a = sum(hi) / len(hi), sum(ai) / len(ai)
    return {"home_prob": h / (h + a),
            "total": sum(tot) / len(tot) if tot else None,
            "home_ml": median_odds(hml), "away_ml": median_odds(aml)}


# --- stats helpers ---------------------------------------------------------

def starter_arsenal(conn, mlb_id, season):
    """Top-4 pitches by usage. Each: pitch name, usage%, contact-quality (xrv)."""
    if mlb_id is None:
        return []
    b = conn.execute("SELECT bdl_id FROM bdl_id_map WHERE mlb_id=? "
                     "AND entity_type='player'", (mlb_id,)).fetchone()
    if not b:
        return []
    rows = conn.execute(
        "SELECT pitch_type, pitch_usage_percent, slg, xwoba, woba "
        "FROM bdl_pitch_type_stats WHERE player_id=? AND role='pitcher' "
        "AND season=? ORDER BY pitch_usage_percent DESC",
        (b["bdl_id"], season)).fetchall()
    out = []
    for r in rows[:4]:
        q = r["xwoba"] if r["xwoba"] is not None else (
            r["woba"] if r["woba"] is not None else r["slg"])
        out.append({"pitch": PITCH_NAME.get(r["pitch_type"], r["pitch_type"]),
                    "raw": r["pitch_type"],
                    "usage": round(r["pitch_usage_percent"] or 0),
                    "xrv": round(q, 2) if q is not None else 0.0})
    return out


def starter_hand(conn, mlb_id):
    if mlb_id is None:
        return None
    r = conn.execute("SELECT throw_hand FROM pitcher_stats WHERE player_id=? "
                     "AND throw_hand IS NOT NULL ORDER BY season DESC LIMIT 1",
                     (mlb_id,)).fetchone()
    return r["throw_hand"] if r else None


def fip_tier(fip):
    """Color band for a starter's FIP. Mirrors the legacy dashboard's bands."""
    if fip is None:
        return "na"
    if fip < 3.50:
        return "good"
    if fip < 4.20:
        return "avg"
    return "bad"


# Projected total, display only. The 2026-07-12 backtest (260 night games
# 6/13-7/11 with clean pre-game odds; day games excluded for the
# bdl_odds_today mid-game-overwrite bug) showed the standalone heuristic had
# no edge: MAE 3.75 vs actual, worse than the Vegas total (3.66) and even a
# constant (3.65). Regressing (actual - vegas_total) on the FIP/xwOBA/
# bullpen/offense components found no signal in any of them (R^2 = 0.007,
# all |t| < 1.1). So the projection is now ANCHORED to the Vegas consensus
# total and only expresses a small clipped lean (<= half a run) from those
# components — calibrated by construction, never pretending to beat the
# line. When no Vegas total is posted yet, it falls back to a park-adjusted
# league base plus the same clipped lean.
_TOTAL_BASE = 9.0       # no-Vegas fallback anchor; fitted on 2026 night games
_FIP_BASELINE = 4.00    # league-average FIP
_FIP_PER_RUN = 0.45     # ~half a run per FIP point above league avg (per starter)
_XWOBA_BASELINE = 0.320 # league-average xwOBA against (allowed by pitchers)
_XWOBA_PER_RUN = 8.0    # per-unit xwOBA delta -> runs (scaled per starter)
_BULLPEN_BASELINE = 3.80
_BULLPEN_PER_RUN = 0.35 # ~1/3 run/game per ERA point above league avg, per team
_LEAGUE_AVG_BULLPEN = 3.80  # fallback when bullpen ERA missing or implausible
_LEAN_SCALE = 0.30      # raw component lean -> runs of adjustment
_LEAN_CAP = 0.5         # adjustment never moves more than half a run off anchor
# Real MLB totals live ~6.5-13. Anything outside this is almost certainly the
# bdl_odds_today day-game contamination bug (in-game/settled odds overwrite
# the pre-game line) — don't anchor to it.
_ANCHOR_MIN, _ANCHOR_MAX = 6.0, 14.0


def _starter_xwoba_against(conn, mlb_id, season):
    """Usage-weighted xwOBA against, across the starter's pitch arsenal.

    Returns None if there's no balldontlie mapping or no pitch-type rows.
    Falls back through xwOBA -> wOBA -> SLG, mirroring starter_arsenal().
    """
    if mlb_id is None:
        return None
    b = conn.execute(
        "SELECT bdl_id FROM bdl_id_map WHERE mlb_id=? AND entity_type='player'",
        (mlb_id,)).fetchone()
    if not b:
        return None
    rows = conn.execute(
        "SELECT pitch_usage_percent, xwoba, woba, slg "
        "FROM bdl_pitch_type_stats WHERE player_id=? AND role='pitcher' "
        "AND season=?",
        (b["bdl_id"], season)).fetchall()
    total_usage = 0.0
    weighted = 0.0
    for r in rows:
        u = r["pitch_usage_percent"] or 0
        if u <= 0:
            continue
        q = r["xwoba"] if r["xwoba"] is not None else (
            r["woba"] if r["woba"] is not None else r["slg"])
        if q is None:
            continue
        weighted += q * u
        total_usage += u
    if total_usage <= 0:
        return None
    return weighted / total_usage


def _team_bullpen_era(conn, team_abbr, season):
    """Team bullpen ERA for the season. Implausible values (<= 1.0) treated
    as missing — Bug 3 from CLAUDE.md left NYY/SF stuck at 0.0; rather than
    let that poison the projection we fall back to league avg."""
    row = conn.execute(
        "SELECT bullpen_era FROM team_stats WHERE team_name=? AND season=? "
        "AND bullpen_era IS NOT NULL",
        (team_abbr, season)).fetchone()
    if row and row["bullpen_era"] is not None and row["bullpen_era"] > 1.0:
        return row["bullpen_era"]
    return None


def _component_lean(conn, game_row, away_fip, home_fip):
    """Raw runs lean from starter FIP, starter xwOBA-against, bullpen ERA and
    recent offense. Missing inputs contribute 0. Same component weights as the
    old standalone heuristic — the 7/12 backtest found no per-component signal
    to refit, so they only set the relative shape of a small lean."""
    date = game_row["game_date"]
    season = int(date[:4]) if date else get_current_season()

    lean = 0.0
    if away_fip is not None:
        lean += (away_fip - _FIP_BASELINE) * _FIP_PER_RUN
    if home_fip is not None:
        lean += (home_fip - _FIP_BASELINE) * _FIP_PER_RUN

    away_xwoba = _starter_xwoba_against(conn, game_row["away_starter_id"], season)
    home_xwoba = _starter_xwoba_against(conn, game_row["home_starter_id"], season)
    if away_xwoba is not None:
        lean += (away_xwoba - _XWOBA_BASELINE) * _XWOBA_PER_RUN
    if home_xwoba is not None:
        lean += (home_xwoba - _XWOBA_BASELINE) * _XWOBA_PER_RUN

    home_bp = _team_bullpen_era(conn, game_row["home_team"], season) or _LEAGUE_AVG_BULLPEN
    away_bp = _team_bullpen_era(conn, game_row["away_team"], season) or _LEAGUE_AVG_BULLPEN
    lean += ((home_bp - _BULLPEN_BASELINE)
             + (away_bp - _BULLPEN_BASELINE)) * _BULLPEN_PER_RUN

    home_trend = _get_offense_trend(game_row["home_team"], date, conn) or 0.0
    away_trend = _get_offense_trend(game_row["away_team"], date, conn) or 0.0
    lean += max(-1.5, min(1.5, (home_trend + away_trend) * 0.6))
    return lean


def projected_total(conn, game_row, away_fip, home_fip, vegas_total=None):
    """Return Goose's projected runs total for the game, or None if missing input.

    Vegas-anchored: consensus total plus a clipped component lean of at most
    +/- _LEAN_CAP runs. Without a posted total, park-adjusted _TOTAL_BASE plus
    the same lean (requires both starter FIPs so the fallback isn't blind).
    """
    lean_adj = max(-_LEAN_CAP, min(_LEAN_CAP,
                   _component_lean(conn, game_row, away_fip, home_fip) * _LEAN_SCALE))
    if vegas_total is not None and _ANCHOR_MIN <= vegas_total <= _ANCHOR_MAX:
        raw = vegas_total + lean_adj
    elif away_fip is not None and home_fip is not None:
        park = PARK_FACTORS.get(game_row["home_team"], 1.00)
        raw = _TOTAL_BASE * park + lean_adj
    else:
        return None
    return round(raw * 2) / 2  # nearest 0.5


def hitter_vs_pitch(conn, bdl_id, pitch_type, season):
    r = conn.execute(
        "SELECT xwoba, woba, slg FROM bdl_pitch_type_stats "
        "WHERE player_id=? AND role='hitter' AND season=? AND pitch_type=?",
        (bdl_id, season, pitch_type)).fetchone()
    if not r:
        return None
    for f in ("xwoba", "woba", "slg"):
        if r[f] is not None:
            return r[f]
    return None


def team_hitters(conn, team, opp_starter_id, opp_arsenal, date_str, season, limit=4):
    """Top-N hitters by OPS, each with form/platoon/pitch tags.

    Uses the most recent stored lineup for the team on or before date_str.
    Reads batting / form snapshots scoped to date_str (the _today tables carry
    a game_date column post-Step-4).
    """
    d = conn.execute(
        "SELECT MAX(lineup_date) d FROM game_lineups "
        "WHERE team=? AND lineup_date<=?", (team, date_str)).fetchone()["d"]
    if not d:
        return []
    p_hand = starter_hand(conn, opp_starter_id)
    top_pitch = opp_arsenal[0]["raw"] if opp_arsenal else None
    top_pitch_name = opp_arsenal[0]["pitch"] if opp_arsenal else None
    lineup = conn.execute(
        "SELECT player_id, player_name, bat_side FROM game_lineups "
        "WHERE team=? AND lineup_date=?", (team, d)).fetchall()
    hitters = []
    for p in lineup:
        b = conn.execute("SELECT bdl_id FROM bdl_id_map WHERE mlb_id=? "
                         "AND entity_type='player'", (p["player_id"],)).fetchone()
        if not b:
            continue
        s = conn.execute("SELECT full_name, ops, hr FROM bdl_batting_today "
                         "WHERE player_id=? AND game_date=?",
                         (b["bdl_id"], date_str)).fetchone()
        if not s or s["ops"] is None:
            continue
        tags = []
        # form
        f = conn.execute("SELECT last15_ops, season_ops, last15_ab "
                         "FROM bdl_form_today WHERE player_id=? AND game_date=?",
                         (b["bdl_id"], date_str)).fetchone()
        if f and f["last15_ops"] is not None and f["season_ops"] is not None \
                and (f["last15_ab"] or 0) >= 10:
            d_ops = f["last15_ops"] - f["season_ops"]
            if d_ops >= 0.120:
                tags.append({"kind": "hot", "val": f"+{d_ops:.2f}"})
            elif d_ops <= -0.120:
                tags.append({"kind": "cold", "val": f"{d_ops:.2f}"})
        # platoon
        bs = p["bat_side"]
        if p_hand and (bs == "S" or (bs and bs != p_hand)):
            tags.append({"kind": "plt"})
        # pitch matchup — only flag if they HANDLE the top pitch well
        if top_pitch:
            vq = hitter_vs_pitch(conn, b["bdl_id"], top_pitch, season)
            if vq is not None and vq >= 0.360:
                tags.append({"kind": "pitch",
                             "val": f"{top_pitch_name} {vq:.2f}"})
        hitters.append({"name": s["full_name"] or p["player_name"],
                        "ops": s["ops"], "hr": s["hr"] or 0, "tags": tags})
    hitters.sort(key=lambda x: x["ops"], reverse=True)
    return hitters[:limit]


def team_form(conn, team, date_str, n=10):
    """L10 record + run differential, restricted to games before date_str."""
    rows = conn.execute(
        "SELECT home_team, away_team, home_score, away_score, winner "
        "FROM games WHERE status='Final' AND winner IS NOT NULL "
        "AND (home_team=? OR away_team=?) AND game_date < ? "
        "ORDER BY game_date DESC LIMIT ?", (team, team, date_str, n)).fetchall()
    w = rd = 0
    for r in rows:
        is_home = r["home_team"] == team
        if (r["winner"] == "home") == is_home:
            w += 1
        ts = r["home_score"] if is_home else r["away_score"]
        os_ = r["away_score"] if is_home else r["home_score"]
        if ts is not None and os_ is not None:
            rd += ts - os_
    return {"L10": f"{w}-{len(rows) - w}", "RD": rd, "n": len(rows)}


def team_injuries(conn, team):
    """IL players for a team — named where the crosswalk resolves."""
    tm = conn.execute("SELECT bdl_id FROM bdl_id_map WHERE name=? "
                      "AND entity_type='team'", (team,)).fetchone()
    if not tm:
        return []
    rows = conn.execute(
        "SELECT i.injury_type, m.name FROM bdl_injuries i "
        "LEFT JOIN bdl_id_map m ON m.bdl_id=i.player_id "
        "  AND m.entity_type='player' "
        "WHERE i.team_id=? AND i.status LIKE '%IL%'", (tm["bdl_id"],)).fetchall()
    named = [(r["name"], r["injury_type"]) for r in rows if r["name"]]
    total = len(rows)
    if not named:
        return [f"{team}: {total} on IL"] if total else []
    head = ", ".join(f"{n} ({t})" for n, t in named[:2])
    extra = total - min(2, len(named))
    return [f"{team} IL: {head}" + (f" +{extra}" if extra > 0 else "")]


# --- edge typing -----------------------------------------------------------
# The design's 5 typed edges. We DERIVE them from real data, then keep the
# strongest 2-3 per game. Each: kind, label, detail, strength (1-5).

def derive_edges(g):
    """Build the typed-edge list for one assembled game dict."""
    edges = []
    pick_is_home = g["pick"] == g["home"]
    pick_abbr = g["pick"]
    opp_abbr = g["away"] if pick_is_home else g["home"]

    # --- Model edge: model prob vs de-vigged Vegas prob ---
    if g["vegasPct"] is not None:
        gap = g["modelPct"] - g["vegasPct"]
        # orient to the pick
        pick_gap = gap if pick_is_home else -gap
        if abs(pick_gap) >= 1:
            strength = 5 if abs(pick_gap) >= 8 else 4 if abs(pick_gap) >= 4 \
                else 3 if abs(pick_gap) >= 2 else 1
            if pick_gap >= 1:
                detail = f"Model has {pick_abbr} {pick_gap:+d} pts over the Vegas line"
            else:
                detail = f"Model fades the market — {pick_gap:+d} pts vs Vegas on {pick_abbr}"
            edges.append({"kind": "model", "label": "Model edge",
                          "detail": detail, "strength": strength})

    # --- Form edge: L10 + run differential of the pick's team ---
    pf = g["_form"][pick_abbr]
    of = g["_form"][opp_abbr]
    pw = int(pf["L10"].split("-")[0])
    ow = int(of["L10"].split("-")[0])
    if pw - ow >= 2 or pf["RD"] - of["RD"] >= 12:
        strength = 4 if (pw - ow >= 3 and pf["RD"] - of["RD"] >= 15) else 3
        edges.append({"kind": "form", "label": "Form edge",
                      "detail": f"{pick_abbr} {pf['L10']} L10 (RD {pf['RD']:+d}) "
                                f"vs {opp_abbr} {of['L10']}",
                      "strength": strength})
    elif of["RD"] - pf["RD"] >= 12:
        # form actually leans against the pick — surface as a caution
        edges.append({"kind": "form", "label": "Form mismatch",
                      "detail": f"{opp_abbr} RD {of['RD']:+d} dwarfs "
                                f"{pick_abbr}'s {pf['RD']:+d}",
                      "strength": 2})

    # --- Lineup edge: platoon-advantaged + hot bats in the pick's lineup ---
    pick_bats = g["_bats"][pick_abbr]
    n_plt = sum(1 for h in pick_bats
                for t in h["tags"] if t["kind"] == "plt")
    n_hot = sum(1 for h in pick_bats
                for t in h["tags"] if t["kind"] == "hot")
    if n_plt >= 3 or n_hot >= 2:
        bits = []
        if n_plt >= 3:
            bits.append(f"{n_plt} {pick_abbr} bats with the platoon edge")
        if n_hot >= 2:
            bits.append(f"{n_hot} hot ({pick_abbr})")
        edges.append({"kind": "lineup", "label": "Lineup edge",
                      "detail": " · ".join(bits),
                      "strength": 4 if (n_plt >= 4 or n_hot >= 3) else 3})

    # --- Pitching edge: opposing starter's most-used pitch gets hit ---
    opp_arsenal = g["_arsenal"][opp_abbr]
    if opp_arsenal:
        top = opp_arsenal[0]
        # how the pick's bats handle that pitch
        hits = [t["val"] for h in pick_bats for t in h["tags"]
                if t["kind"] == "pitch"]
        if top["xrv"] >= 0.42 or len(hits) >= 2:
            opp_sp = g["_pitcher_name"][opp_abbr]
            edges.append({"kind": "pitching", "label": "Pitching edge",
                          "detail": f"{opp_sp}'s {top['pitch']} ({top['usage']}% "
                                    f"usage) is hittable — {pick_abbr} bats sit on it",
                          "strength": 4 if top["xrv"] >= 0.45 else 3})

    # --- Injury edge: opposing team meaningfully banged up ---
    opp_inj = g["_injuries"][opp_abbr]
    pick_inj = g["_injuries"][pick_abbr]
    if pick_inj:
        # if the PICK is the hurt one, surface as a caution
        txt = pick_inj[0]
        if "IL:" in txt or "on IL" in txt:
            n = _il_count(txt)
            if n >= 6:
                edges.append({"kind": "injury", "label": "Injury caution",
                              "detail": f"{txt} — depth is thin",
                              "strength": 3})
    if opp_inj:
        n = _il_count(opp_inj[0])
        if n >= 6:
            edges.append({"kind": "injury", "label": "Injury edge",
                          "detail": f"{opp_abbr} banged up — {opp_inj[0]}",
                          "strength": 3})

    # keep the strongest 3, sorted by strength desc
    edges.sort(key=lambda e: e["strength"], reverse=True)
    return edges[:3]


def _il_count(txt):
    """Pull the IL count out of an injuries string."""
    import re
    m = re.search(r"(\d+)\s*on IL", txt)
    if m:
        return int(m.group(1))
    # "TEAM IL: a, b +N" — 2 named + N
    m = re.search(r"\+(\d+)", txt)
    return (2 + int(m.group(1))) if m else (1 if "IL:" in txt else 0)


# --- pints: 5-level confidence ---------------------------------------------
# The model has 4 tiers (HIGH/MEDIUM/LEAN/COIN FLIP). The design wants a 1-5
# pint scale, so we split on the pick's probability WITHIN tiers to get a real
# 5th step rather than padding a 4-tier system.
#   5  Lock      — HIGH tier, pick prob >= 0.70
#   4  Strong    — HIGH tier below 0.70, or MEDIUM at >= 0.62
#   3  Lean      — MEDIUM tier
#   2  Slight    — LEAN tier
#   1  Coin Flip — COIN FLIP / model within 3% of 50

# 5-pint confidence ladder — the brand's signature scale.
#   1  Wet the Beak     5  Flying V
PINT_LABELS = {1: "Wet the Beak", 2: "Pub Lean", 3: "Goose Approved",
               4: "Goose Is Loose", 5: "Flying V"}


def pints_and_label(tier, pick_prob):
    """Return (pints 1-5, label) from the model tier + pick win probability."""
    if tier == "COIN FLIP":
        pints = 1
    elif tier == "LEAN":
        pints = 2
    elif tier == "MEDIUM":
        pints = 4 if pick_prob >= 0.62 else 3
    elif tier == "HIGH":
        pints = 5 if pick_prob >= 0.70 else 4
    else:
        pints = 1
    return pints, PINT_LABELS[pints]


# --- game assembly ---------------------------------------------------------

def assemble_games(date_str, season):
    """Assemble every game on date_str into the design's card data shape.

    Reads the picks table (deduped lineup_lock > morning), joined to games.
    Picks are produced upstream by scheduler.py; this is a pure renderer.
    """
    games = []
    with get_db() as conn:
        rows = conn.execute(f"""
            SELECT g.*, p.predicted_winner, p.home_win_prob, p.confidence,
                   p.opener_flag, p.pick_flipped
            FROM picks p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.pick_date = ? AND {_DEDUP}
            ORDER BY g.game_time ASC
        """, (date_str,)).fetchall()
        for g in rows:
            away, home = g["away_team"], g["home_team"]
            v = vegas_consensus(conn, away, home, date_str)
            mp = g["home_win_prob"]
            pick = g["predicted_winner"]
            pick_is_home = pick == home
            tier = g["confidence"]
            pick_prob = mp if pick_is_home else (1 - mp)
            pints, plabel = pints_and_label(tier, pick_prob)

            away_ars = starter_arsenal(conn, g["away_starter_id"], season)
            home_ars = starter_arsenal(conn, g["home_starter_id"], season)
            away_bats = team_hitters(conn, away, g["home_starter_id"],
                                     home_ars, date_str, season)
            home_bats = team_hitters(conn, home, g["away_starter_id"],
                                     away_ars, date_str, season)

            # Same FIP the model consumes — blended current+prior season by IP
            # (see model.features._get_pitcher_fip). Display only; the model
            # already used these via fip_diff to produce the pick.
            away_fip = _get_pitcher_fip(g["away_starter_id"], away, conn)
            home_fip = _get_pitcher_fip(g["home_starter_id"], home, conn)
            away_hand = starter_hand(conn, g["away_starter_id"])
            home_hand = starter_hand(conn, g["home_starter_id"])
            goose_total = projected_total(conn, g, away_fip, home_fip,
                                          vegas_total=v["total"] if v else None)

            # model & vegas as P(pick wins), to 0-100 ints
            model_pick_pct = round(pick_prob * 100)
            vegas_pick_pct = None
            pick_ml = None
            if v:
                vp_home = v["home_prob"]
                vegas_pick_pct = round((vp_home if pick_is_home else 1 - vp_home) * 100)
                pick_ml = v["home_ml"] if pick_is_home else v["away_ml"]

            entry = {
                "id": g["game_id"],
                "when": _fmt_time(g["game_time"]),
                "_sort_time": g["game_time"] or "99:99",
                "away": away, "home": home,
                "pick": pick, "pickOdds": pick_ml,
                "confidence": tier, "confLabel": plabel, "pints": pints,
                "modelPct": model_pick_pct, "vegasPct": vegas_pick_pct,
                "total": v["total"] if v else None,
                "gooseTotal": goose_total,
                "pitcherAway": g["away_starter_name"],
                "pitcherHome": g["home_starter_name"],
                "fipAway": away_fip,
                "fipHome": home_fip,
                "handAway": away_hand,
                "handHome": home_hand,
                # private fields for edge derivation
                "_form": {away: team_form(conn, away, date_str),
                          home: team_form(conn, home, date_str)},
                "_bats": {away: away_bats, home: home_bats},
                "_arsenal": {away: away_ars, home: home_ars},
                "_injuries": {away: team_injuries(conn, away),
                              home: team_injuries(conn, home)},
                "_pitcher_name": {away: g["away_starter_name"],
                                  home: g["home_starter_name"]},
            }
            entry["edges"] = derive_edges(entry)
            # Prop edges — display-only commentary surfaced in the expanded
            # card section, separate from the winner-pick edge typing above.
            entry["prop_edges"] = gather_prop_edges_for_game(
                conn, g, season, date_str)
            games.append(entry)
    return games


def _fmt_time(hhmm):
    """24h 'HH:MM' -> '7:15 PM ET'."""
    if not hhmm or ":" not in hhmm:
        return hhmm or ""
    try:
        h, m = int(hhmm[:2]), hhmm[3:5]
        ap = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12}:{m} {ap} ET"
    except (ValueError, IndexError):
        return hhmm


def season_tracker(season):
    """Season Tracker data — a 1:1 match of the production dashboard's tab:
    overall record / win rate / total picks / current streak, per-tier accuracy,
    and the Recent Days heatmap (per-day W/total, green/yellow/red graded)."""
    season_start = f"{season}-01-01"
    with get_db() as conn:
        overall = conn.execute(f"""
            SELECT SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) wins,
                   SUM(CASE WHEN correct=0 THEN 1 ELSE 0 END) losses,
                   SUM(CASE WHEN correct IS NULL THEN 1 ELSE 0 END) pending
            FROM picks p
            WHERE pick_date >= ? AND {_DEDUP}""", (season_start,)).fetchone()
        tiers = conn.execute(f"""
            SELECT confidence,
                   SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) wins,
                   SUM(CASE WHEN correct=0 THEN 1 ELSE 0 END) losses
            FROM picks p
            WHERE correct IS NOT NULL AND pick_date >= ? AND {_DEDUP}
            GROUP BY confidence""", (season_start,)).fetchall()
        # Recent Days — per-day W/total, last 14 days (the production query)
        recent = conn.execute(f"""
            SELECT pick_date,
                   COUNT(*) total,
                   SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) wins
            FROM picks p
            WHERE correct IS NOT NULL AND pick_date >= ? AND {_DEDUP}
            GROUP BY pick_date ORDER BY pick_date DESC LIMIT 14
        """, (season_start,)).fetchall()
        # Current streak — most recent graded picks by date+game time
        streak_rows = conn.execute(f"""
            SELECT p.correct FROM picks p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.correct IS NOT NULL AND {_DEDUP}
            ORDER BY p.pick_date DESC, g.game_time DESC LIMIT 50""").fetchall()
        # Flying V hit rate — graded HIGH-tier picks where the pick's win
        # probability cleared the 5-pint threshold (>= 0.70).
        fv_rows = conn.execute(f"""
            SELECT p.predicted_winner, p.home_win_prob, p.correct, g.home_team
            FROM picks p JOIN games g ON g.game_id = p.game_id
            WHERE p.confidence='HIGH' AND p.correct IS NOT NULL AND {_DEDUP}
        """).fetchall()

    wins = overall["wins"] or 0
    losses = overall["losses"] or 0
    pending = overall["pending"] or 0
    total = wins + losses

    # streak
    streak = {"type": "none", "count": 0}
    if streak_rows:
        first = streak_rows[0]["correct"]
        cnt = 0
        for r in streak_rows:
            if r["correct"] == first:
                cnt += 1
            else:
                break
        streak = {"type": "W" if first == 1 else "L", "count": cnt}

    tier_order = {"HIGH": 0, "MEDIUM": 1, "LEAN": 2, "COIN FLIP": 3}
    tier_rows = sorted(
        [{"tier": t["confidence"], "wins": t["wins"] or 0, "losses": t["losses"] or 0,
          "n": (t["wins"] or 0) + (t["losses"] or 0),
          "acc": ((t["wins"] or 0) / ((t["wins"] or 0) + (t["losses"] or 0)))
                 if ((t["wins"] or 0) + (t["losses"] or 0)) else 0}
         for t in tiers],
        key=lambda x: tier_order.get(x["tier"], 9))

    # Flying V hit rate — 5-pint picks (HIGH tier, pick prob >= 0.70)
    fv_n = fv_w = 0
    for r in fv_rows:
        pick_prob = (r["home_win_prob"] if r["predicted_winner"] == r["home_team"]
                     else 1 - r["home_win_prob"])
        if pick_prob >= 0.70:
            fv_n += 1
            fv_w += r["correct"]

    return {
        "wins": wins, "losses": losses, "total": total, "pending": pending,
        "acc": wins / total if total else 0,
        "streak": streak,
        "tiers": tier_rows,
        "fv_wins": fv_w, "fv_n": fv_n,
        "fv_acc": fv_w / fv_n if fv_n else 0,
        "recent": [{"date": r["pick_date"], "total": r["total"],
                    "wins": r["wins"] or 0} for r in recent],
    }


# ---------------------------------------------------------------------------
# Goose Status — the signature lore bank. A mood is chosen from how the model
# did last time out + how strong tonight's slate is; the exact line rotates
# day to day so it never goes stale. {rec} = last graded record, {fv} = a
# Flying-V mention when one is on the board.
# ---------------------------------------------------------------------------
GOOSE_LORE = {
    # --- hot: won big last time ---
    "LOOSE": [
        "{rec} last time out. Pints flowing, GPS locked in.{fv}",
        "{rec} last time out. Confidence high, beak fully wet.{fv}",
        "{rec} last time out. The Goose is feeling himself tonight.{fv}",
    ],
    "FLYING": [   # hot AND multiple Flying Vs
        "Multiple Flying Vs tonight. Wet the beak accordingly.",
        "{rec} last time out and the Vs are out. Goose is airborne.",
        "More than one Flying V on the board. This is the dream slate.",
    ],
    # --- neutral: steady last time ---
    "BACK AT GOOSE'S": [
        "Steady card. Goose likes a few spots.{fv}",
        "{rec} last time out, model holding. Goose likes a few spots.{fv}",
        "{rec} last time out. Morale holding — back on the stool.{fv}",
    ],
    "WETTING THE BEAK": [   # neutral but a light/low-conviction slate
        "Light slate. Proceed casually.",
        "Quiet card tonight — Goose is sipping, not chugging.",
        "Not much conviction on the board. Wet the beak gently.",
    ],
    # --- cold: rough last time ---
    "RE-CALIBRATING": [
        "Yesterday's pints may have been overpoured.",
        "{rec} last time out. Model's having a coffee, sobering up.",
        "{rec} last time out. Confidence under review — but a few spots remain.{fv}",
    ],
    "LOST SIGNAL": [
        "The Goose got cooked yesterday.",
        "{rec} last time out. GPS is currently unavailable for comment.",
        "{rec} last time out. The Goose flew south for the winter.",
    ],
}


def goose_status(season, games, date_str):
    """A one-line 'Goose Status' from the lore bank — meme-fuel commentary
    derived from how the model did last time out + tonight's slate strength.
    Returns (mood, line). The mood is picked deterministically; the exact
    line rotates by date so it varies day to day."""
    recent = season["recent"]
    n_flying_v = sum(1 for g in games if g["pints"] == 5)
    n_strong = sum(1 for g in games if g["pints"] >= 4)
    light_slate = n_strong == 0  # nothing above 3 pints

    if recent:
        last = recent[0]
        rate = last["wins"] / last["total"] if last["total"] else 0.5
        rec = f"{last['wins']}–{last['total'] - last['wins']}"
    else:
        rate, rec = 0.5, "0–0"

    # pick the mood
    if rate >= 0.60:
        mood = "FLYING" if n_flying_v >= 2 else "LOOSE"
    elif rate >= 0.47:
        mood = "WETTING THE BEAK" if light_slate else "BACK AT GOOSE'S"
    else:
        mood = "RE-CALIBRATING" if (n_flying_v or n_strong >= 2) else "LOST SIGNAL"

    # rotate the line within the mood by day-of-year (stable within a day)
    lines = GOOSE_LORE[mood]
    y, m, d = (int(x) for x in date_str.split("-"))
    idx = _dt.date(y, m, d).timetuple().tm_yday % len(lines)
    fv = (" One Flying V on the board." if n_flying_v == 1
          else f" {n_flying_v} Flying Vs on the board." if n_flying_v >= 2
          else "")
    return mood, lines[idx].format(rec=rec, fv=fv)


# --- icon SVGs (from the design's Icons.jsx) -------------------------------

def pint_svg(filled, size=14):
    """One Guinness pint glass — the brand's signature confidence unit."""
    w = size
    h = round(size * 1.35)
    if not filled:
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 48 64" fill="none" '
                f'style="opacity:.4">'
                f'<path d="M 9 6 C 9 11, 11 12, 11.5 17 L 14 55 Q 14 58, 17 58 '
                f'L 31 58 Q 34 58, 34 55 L 36.5 17 C 37 12, 39 11, 39 6" '
                f'stroke="currentColor" stroke-width="1.8" fill="none" '
                f'stroke-linejoin="round" stroke-linecap="round"/></svg>')
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 48 64">'
            f'<defs><clipPath id="pg{size}"><path d="M 9 6 C 9 11, 11 12, 11.5 17 '
            f'L 14 55 Q 14 58, 17 58 L 31 58 Q 34 58, 34 55 L 36.5 17 C 37 12, '
            f'39 11, 39 6 Z"/></clipPath></defs>'
            f'<g clip-path="url(#pg{size})">'
            f'<rect x="0" y="14" width="48" height="50" fill="#0a0503"/>'
            f'<ellipse cx="14.5" cy="38" rx="3" ry="18" fill="#5a2210" opacity="0.45"/>'
            f'<rect x="0" y="6" width="48" height="10" fill="#f3e3b4"/>'
            f'<path d="M 9 7 Q 24 1.5 39 7 L 39 14 L 9 14 Z" fill="#f3e3b4"/>'
            f'<rect x="0" y="14" width="48" height="1.2" fill="#8a6628" opacity="0.5"/>'
            f'<circle cx="15.5" cy="9" r="1.4" fill="#fff8df"/>'
            f'<circle cx="22" cy="6" r="1.6" fill="#fffaea"/>'
            f'<circle cx="29" cy="8" r="1.2" fill="#fff8df"/>'
            f'<circle cx="33" cy="6" r="0.9" fill="#fffaea"/>'
            f'<path d="M 11.6 18 Q 11.8 36 13.6 54" stroke="rgba(255,255,255,0.22)" '
            f'stroke-width="1.6" stroke-linecap="round" fill="none"/></g>'
            f'<path d="M 9 6.5 Q 24 1 39 6.5" stroke="rgba(255,248,223,0.5)" '
            f'stroke-width="0.8" fill="none" stroke-linecap="round"/>'
            f'<path d="M 9 6 C 9 11, 11 12, 11.5 17 L 14 55 Q 14 58, 17 58 L 31 58 '
            f'Q 34 58, 34 55 L 36.5 17 C 37 12, 39 11, 39 6" stroke="currentColor" '
            f'stroke-width="1.8" fill="none" stroke-linejoin="round" '
            f'stroke-linecap="round"/></svg>')


def pint_row(value, size=14):
    pints = "".join(pint_svg(i <= value, size) for i in range(1, 6))
    return f'<span class="pint-row">{pints}</span>'


_EDGE_GLYPH = {
    "model": '<path d="M 8 1 L 15 8 L 8 15 L 1 8 Z" fill="currentColor"/>',
    "lineup": '<path d="M 8 2 L 15 14 L 1 14 Z" fill="currentColor"/>',
    "pitching": '<path d="M 1 2 L 15 2 L 8 14 Z" fill="currentColor"/>',
    "form": ('<circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" '
             'stroke-width="2"/><path d="M 8 2 A 6 6 0 0 1 8 14 Z" fill="currentColor"/>'),
    "injury": ('<path d="M 6 2 H 10 V 6 H 14 V 10 H 10 V 14 H 6 V 10 H 2 V 6 H 6 Z" '
               'fill="currentColor"/>'),
}


def edge_icon(kind, size=14):
    g = _EDGE_GLYPH.get(kind, "")
    return f'<svg width="{size}" height="{size}" viewBox="0 0 16 16">{g}</svg>'


# --- HTML rendering --------------------------------------------------------

def fmt_odds(n):
    if n is None:
        return ""
    return f"+{n}" if n > 0 else str(n)


def edge_meter_html(model_pct, vegas_pct):
    """The signature Model-vs-Vegas visualization (from EdgeMeter.jsx)."""
    if vegas_pct is None:
        return ('<div class="edge-strip no-line">'
                '<span class="edge-strip-label">Model vs Vegas</span>'
                '<span class="edge-strip-note">no line posted yet</span></div>')
    edge = model_pct - vegas_pct
    edge_abs = abs(edge)
    sign = "+" if edge > 0 else ("−" if edge < 0 else "")
    tone = "edge-strong" if edge_abs >= 4 else "edge-mid" if edge_abs >= 2 else "edge-slim"

    lo = max(0, min(model_pct, vegas_pct) - 18)
    hi = min(100, max(model_pct, vegas_pct) + 18)
    span = (hi - lo) or 1
    x_model = ((model_pct - lo) / span) * 100
    x_vegas = ((vegas_pct - lo) / span) * 100
    fill_start = min(x_model, x_vegas)
    fill_end = max(x_model, x_vegas)
    # Compact strip: small inline magnitude + a slim comparison track.
    # The edge is supporting context now, not the card's hero.
    return f"""<div class="edge-strip {tone}">
      <div class="edge-strip-mag">
        <span class="edge-strip-num">{sign}{edge_abs}</span>
        <span class="edge-strip-cap">pt vs Vegas</span>
      </div>
      <div class="edge-strip-track">
        <div class="edge-track-bar">
          <div class="edge-fill" style="left:{fill_start}%;width:{fill_end - fill_start}%"></div>
          <div class="edge-marker edge-marker-vegas" style="left:{x_vegas}%">
            <span class="edge-marker-dot"></span>
          </div>
          <div class="edge-marker edge-marker-model" style="left:{x_model}%">
            <span class="edge-marker-dot"></span>
          </div>
        </div>
        <div class="edge-strip-legend">
          <span class="legend-model">● Goose {model_pct}%</span>
          <span class="legend-vegas">● Vegas {vegas_pct}%</span>
        </div>
      </div>
    </div>"""


def edge_row_html(e):
    return f"""<li class="edge-row edge-kind-{e['kind']}">
      <span class="edge-row-icon">{edge_icon(e['kind'], 14)}</span>
      <span class="edge-row-label">{e['label']}</span>
      <span class="edge-row-detail">{e['detail']}</span>
    </li>"""


_TAG_HTML = {
    "hot": lambda v: f'<span class="tag tag-hot">Hot {v}</span>',
    "cold": lambda v: f'<span class="tag tag-cold">Cold {v}</span>',
    "plt": lambda v: '<span class="tag tag-plt">PLT</span>',
    "pitch": lambda v: f'<span class="tag tag-pitch">{v}</span>',
}


def hitter_html(h):
    tags = "".join(_TAG_HTML[t["kind"]](t.get("val", "")) for t in h["tags"]
                   if t["kind"] in _TAG_HTML)
    tags_html = f'<span class="hitter-tags">{tags}</span>' if tags else ""
    return f"""<li class="hitter">
      <span class="hitter-name">{h['name']}</span>
      <span class="hitter-stats">
        <span class="hitter-ops">{h['ops']:.3f}</span><span class="hitter-ops-l">OPS</span>
        <span class="hitter-hr">{h['hr']}</span><span class="hitter-hr-l">HR</span>
      </span>{tags_html}
    </li>"""


def pitching_strip_html(g):
    """Compact two-row pitcher matchup for the main card face.

    Shows each starter's name, hand badge, team, and FIP (the same blended
    value the model uses). An EDGE caret marks whichever side has the lower
    FIP — FIP is the dominant pitching input into the pick.
    """
    away_fip, home_fip = g["fipAway"], g["fipHome"]

    edge_side = None
    if away_fip is not None and home_fip is not None and abs(away_fip - home_fip) >= 0.10:
        edge_side = "away" if away_fip < home_fip else "home"

    def row(name, abbr, hand, fip, side):
        if not name:
            return ""
        hand_badge = (f'<span class="ps-hand ps-hand-{hand.lower()}">{hand}</span>'
                      if hand else "")
        tier = fip_tier(fip)
        fip_txt = f"{fip:.2f}" if fip is not None else "—"
        edge_marker = ('<span class="ps-edge" title="Better FIP">EDGE ◀</span>'
                       if edge_side == side else "")
        return (f'<div class="ps-row">'
                f'<span class="ps-name">{name}</span>'
                f'{hand_badge}'
                f'<span class="ps-team">{abbr}</span>'
                f'<span class="ps-spacer"></span>'
                f'{edge_marker}'
                f'<span class="ps-fip ps-fip-{tier}">'
                f'<span class="ps-fip-val">{fip_txt}</span>'
                f'<span class="ps-fip-lbl">FIP</span>'
                f'</span>'
                f'</div>')

    away_row = row(g["pitcherAway"], g["away"], g["handAway"], away_fip, "away")
    home_row = row(g["pitcherHome"], g["home"], g["handHome"], home_fip, "home")
    if not away_row and not home_row:
        return ""
    return f'<div class="pitching-strip">{away_row}{home_row}</div>'


def arsenal_html(name, abbr, arsenal):
    if not arsenal:
        body = '<div class="arsenal-empty">no arsenal data</div>'
    else:
        cells = "".join(
            f'<div class="arsenal-cell"><span class="ars-pitch">{p["pitch"]}</span>'
            f'<span class="ars-usage">{p["usage"]}%</span>'
            f'<span class="ars-xrv {"warn" if p["xrv"] >= 0.42 else ""}">'
            f'{p["xrv"]:.2f}</span></div>'
            for p in arsenal)
        body = f'<div class="arsenal">{cells}</div>'
    return (f'<div class="exp-pitcher"><div class="exp-pitcher-name">{name}'
            f'<span class="exp-pitcher-team"> · {abbr}</span></div>{body}</div>')


# --- prop edges rendering (per-card + slate-wide Prop Board tab) ----------

_PROP_KIND_LABEL = {
    "HR": "HR",
    "HITS": "Hits/TB",
    "K-OVER": "K's · over",
    "K-UNDER": "K's · under",
}


# Unicode minus from the design handoff — used for the lineup-order chip.
_MINUS = "−"


# --- chalk tally-mark glyph -------------------------------------------------
#
# The matchup-quality chip on the chalkboard. Hand-drawn chalk tally marks —
# `tier` of them solid (strong) strokes, the remaining strokes dotted/light
# to show "out of 5." Intentionally DIFFERENT from the 5-pint confidence
# indicator used on moneyline picks: that one implies "this pick will win,"
# the matchup chip only implies "this is among the better matchups on the
# board." The visual difference matters: tally marks read as a count, not a
# guarantee.

def chalk_tally_row(tier, height=16):
    """Five chalk strokes — `tier` of them solid (strong), the rest dotted.

    Each stroke is a short slightly-rotated vertical line, like keeping count
    on a chalkboard. Rotated and offset by index so they look hand-drawn, not
    typeset. SVG path: each stroke is two lines at slight angles to feel like
    a chalk stroke rather than a typeset pipe.
    """
    parts = []
    width = 6
    for i in range(5):
        on = i < tier
        # Tiny stroke-angle jitter so the five marks aren't perfectly parallel.
        angle = (-2, 3, -1, 4, -3)[i]
        color_class = "tally-on" if on else "tally-off"
        opacity = 1.0 if on else 0.35
        stroke = "1.6" if on else "1.0"
        dash = "" if on else 'stroke-dasharray="2 2"'
        parts.append(
            f'<svg class="ctally {color_class}" width="{width}" '
            f'height="{height}" viewBox="0 0 6 16">'
            f'<line x1="3" y1="2" x2="3" y2="14" '
            f'stroke="currentColor" stroke-width="{stroke}" '
            f'stroke-linecap="round" {dash} opacity="{opacity}" '
            f'transform="rotate({angle} 3 8)"/></svg>')
    return f'<span class="ctally-row">{"".join(parts)}</span>'


# Backward-compatible alias for any caller that imported the old name.
chalk_pint_row = chalk_tally_row


# --- per-card prop edges panel (in-card Breadcrumbs) -----------------------

def _prop_edge_row_html(e):
    """Per-card 'Prop Edges' panel row — pub-card aesthetic. Uses the same
    matchup-quality tally chip the chalkboard board uses (NOT the moneyline
    Flying-V pints) so the visual reminds the user this is matchup ranking,
    not outcome prediction."""
    kind = e["kind"]
    kind_class = ("prop-kind-hr" if kind == "HR"
                  else "prop-kind-hits" if kind == "HITS"
                  else "prop-kind-kover" if kind == "K-OVER"
                  else "prop-kind-kunder")
    why = e.get("why") or ""
    tier = e.get("tier", e.get("pints", 1))
    tier_label = e.get("tier_label") or e.get("pint_label") or ""
    tally = chalk_tally_row(tier, height=14)
    return f"""<li class="prop-edge-row {kind_class}">
      <span class="prop-edge-kind">{_PROP_KIND_LABEL.get(kind, kind)}</span>
      <span class="prop-edge-body">
        <span class="prop-edge-primary">{e['primary']}</span>
        <span class="prop-edge-secondary">{e['secondary']}</span>
        {f'<span class="prop-edge-why">{why}</span>' if why else ''}
        <span class="prop-edge-tier">{tally}<span class="prop-edge-tier-label">{tier_label}</span></span>
      </span>
    </li>"""


# --- chalkboard table renderers --------------------------------------------

def _tier_cell(c):
    """Render the matchup-quality cell — tally marks above a small label."""
    tier = c.get("tier", c.get("pints", 1))
    label = c.get("tier_label") or c.get("pint_label") or ""
    return (f'<td class="cb-cell-tier">'
            f'{chalk_tally_row(tier)}'
            f'<span class="cb-tier-label">{label}</span>'
            f'</td>')


def _batter_row_html(c):
    """One <tr> for a batter prop table (HR / Hits-TB). Columns:
    Matchup · Player · Matchup · Why."""
    order = (f' <span class="cb-order">{_MINUS}#{c["order"]}</span>'
             if c.get("order") else "")
    return f"""<tr class="cb-row">
      {_tier_cell(c)}
      <td class="cb-cell-player">
        <span class="cb-name">{c.get("batter", "—")}</span>{order}
        <span class="cb-team">{c.get("bat_team", "")}</span>
      </td>
      <td class="cb-cell-matchup">
        <span class="cb-vs">vs</span> {c.get("opp_starter", "?")}
        <span class="cb-game">{c.get("matchup", "")}</span>
      </td>
      <td class="cb-cell-why">{c.get("why", "")}</td>
    </tr>"""


def _k_row_html(c):
    """One <tr> for the K-leans table. Columns:
    Matchup · Direction · Pitcher · Matchup · Why."""
    direction = c.get("direction", "neutral")
    dir_class = ("cb-dir-over" if direction == "OVER"
                 else "cb-dir-under" if direction == "UNDER"
                 else "cb-dir-neutral")
    return f"""<tr class="cb-row">
      {_tier_cell(c)}
      <td class="cb-cell-direction">
        <span class="cb-dir {dir_class}">{direction}</span>
      </td>
      <td class="cb-cell-player">
        <span class="cb-name">{c.get("pitcher", "—")}</span>
        <span class="cb-team">{c.get("pitcher_team", "")}</span>
      </td>
      <td class="cb-cell-matchup">
        <span class="cb-vs">vs</span> {c.get("opp_team", "?")} lineup
        <span class="cb-game">{c.get("matchup", "")}</span>
      </td>
      <td class="cb-cell-why">{c.get("why", "")}</td>
    </tr>"""


def _batter_table_html(rows, empty_msg):
    if not rows:
        return f'<div class="cb-empty">{empty_msg}</div>'
    body = "".join(_batter_row_html(r) for r in rows)
    return f"""<table class="cb-table cb-table-batter">
      <thead>
        <tr>
          <th class="cb-th-tier">Matchup</th>
          <th class="cb-th-player">Player</th>
          <th class="cb-th-matchup">Facing</th>
          <th class="cb-th-why">Why</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>"""


def _k_table_html(rows, empty_msg):
    if not rows:
        return f'<div class="cb-empty">{empty_msg}</div>'
    body = "".join(_k_row_html(r) for r in rows)
    return f"""<table class="cb-table cb-table-k">
      <thead>
        <tr>
          <th class="cb-th-tier">Matchup</th>
          <th class="cb-th-dir">Lean</th>
          <th class="cb-th-player">Pitcher</th>
          <th class="cb-th-matchup">Facing</th>
          <th class="cb-th-why">Why</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>"""


def _chalkboard_card(title: str, subtitle: str, table_html: str) -> str:
    """A single chalkboard card — title + subtitle in chalk, then the body
    table on the slate. Matches the design handoff's chalkboard primitive
    (oak frame, slate surface, chalked title with -1deg rotation)."""
    return f"""<section class="chalkboard chalkboard-card">
      <div class="cb-head">
        <div class="cb-title">{title}</div>
        <div class="cb-subtitle">{subtitle}</div>
      </div>
      <div class="cb-rule"></div>
      {table_html}
    </section>"""


def prop_board_html(board):
    """Render the slate-wide Prop Board as three chalkboard cards (HR, Hits/TB,
    K leans), each containing a 4-or-5-column table. Plus a chalk-styled
    explainer card at the top and a small footer with pool counts.
    """
    if board.get("reason"):
        return f"""<section class="chalkboard">
          <div class="cb-title">No lines yet</div>
          <div class="cb-rule"></div>
          <div class="cb-empty">{board['reason']}</div>
        </section>"""

    t = board.get("totals", {})

    # Compact one-line explainer — sits just above the cards, no chalkboard
    # surface of its own. Big "How to read this board" toggle expands to the
    # longer copy on demand.
    explainer = """<details class="prop-explainer">
      <summary class="prop-explainer-summary">
        <span class="prop-explainer-eyebrow">Matchup rankings</span>
        <span class="prop-explainer-line">
          Each card below ranks tonight&rsquo;s best matchups for one prop &mdash;
          hints, not guarantees.
        </span>
        <span class="prop-explainer-toggle">How to read &rsaquo;</span>
      </summary>
      <div class="prop-explainer-body">
        <p><strong>Matchup tally:</strong> 5 chalk strokes show how the matchup
        ranks across the whole slate. More strokes = better matchup signal.
        <em>It is NOT a probability that a HR / hit / K total will land.</em></p>
        <p><strong>Batter cards:</strong> we&rsquo;re looking at how well the
        hitter handles the pitch types the opposing starter throws most often,
        whether his bat is hot lately, and his season power profile.</p>
        <p><strong>Pitcher card:</strong> <em>OVER</em> = pitcher misses bats
        AND opposing lineup whiffs (K-rich spot). <em>UNDER</em> = contact
        pitcher facing a contact lineup (low-K spot).</p>
        <p class="cb-caveat">A hint, not a guarantee. Sample sizes are
        real-season-small. Wet the beak responsibly.</p>
      </div>
    </details>"""

    hr_card = _chalkboard_card(
        "Tonight&rsquo;s HR spots",
        "Best home-run upside matchups across the slate.",
        _batter_table_html(board.get("hr", []),
                           "No standout HR spots tonight."))
    hits_card = _chalkboard_card(
        "Tonight&rsquo;s hit spots",
        "Best hit / total-bases matchups across the slate.",
        _batter_table_html(board.get("hits_tb", []),
                           "No standout hit spots tonight."))
    k_card = _chalkboard_card(
        "Tonight&rsquo;s K leans",
        "Strikeout over/under leans by starter.",
        _k_table_html(board.get("k", []),
                      "No standout K leans tonight."))

    footer = f"""<section class="chalkboard chalkboard-foot">
      <div class="cb-footer">
        {t.get("games", 0)} games &middot;
        {t.get("hr_pool", 0)} HR candidates &middot;
        {t.get("hits_pool", 0)} hits candidates &middot;
        {t.get("k_pool", 0)} K candidates
      </div>
    </section>"""

    return "\n".join([explainer, hr_card, hits_card, k_card, footer])


def card_html(g, is_biggest=False):
    conf_slug = g["confidence"].lower().replace(" ", "-")
    # the design's pill classes differ from the rail slug
    pill_cls = {"HIGH": "conf-high", "MEDIUM": "conf-med",
                "LEAN": "conf-lean", "COIN FLIP": "conf-flip"}.get(
                    g["confidence"], "conf-flip")
    edges = "".join(edge_row_html(e) for e in g["edges"])
    away_bats = "".join(hitter_html(h) for h in g["_bats"][g["away"]])
    home_bats = "".join(hitter_html(h) for h in g["_bats"][g["home"]])
    away_il = g["_injuries"][g["away"]]
    home_il = g["_injuries"][g["home"]]
    il_items = "".join(f"<li>{x}</li>" for x in (away_il + home_il))
    il_section = (f"""<section class="exp-section">
          <h3 class="exp-title"><span class="exp-icon">🏥</span>Injury Notes</h3>
          <ul class="il-list">{il_items}</ul>
        </section>""" if il_items else "")
    # Prop edges panel — surfaces strongest batter/pitcher prop matchups for
    # this game from the bdl_*_today cache. Display-only commentary, NOT a
    # model input.
    prop_edges = g.get("prop_edges") or []
    prop_section = ""
    if prop_edges:
        prop_items = "".join(_prop_edge_row_html(e) for e in prop_edges)
        prop_section = (f"""<section class="exp-section prop-edges-section">
          <h3 class="exp-title"><span class="exp-icon">🎯</span>Prop Edges</h3>
          <ul class="prop-edges-list">{prop_items}</ul>
        </section>""")
    af, hf = g["_form"][g["away"]], g["_form"][g["home"]]

    total_chip = ""
    if g["total"] is not None:
        t = g["total"]
        tc = "hot" if t >= 9.0 else "cool" if t < 7.5 else ""
        total_chip = f'<span class="ou-chip {tc}">Vegas O/U {t:.1f}</span>'

    goose_total_chip = ""
    gt = g.get("gooseTotal")
    if gt is not None:
        # Lean class: compared to Vegas, are we projecting over/under/match?
        lean_cls = ""
        if g["total"] is not None:
            diff = gt - g["total"]
            if diff >= 0.5:
                lean_cls = "over"
            elif diff <= -0.5:
                lean_cls = "under"
        goose_total_chip = f'<span class="goose-total {lean_cls}">Goose {gt:.1f}</span>'

    pitching_strip = pitching_strip_html(g)

    biggest_badge = ('<span class="biggest-badge">★ Biggest edge</span>'
                     if is_biggest else "")
    # 5-pint cards get the Flying V treatment — gold glow, the works
    flying_v = g["pints"] == 5
    fv_class = " flying-v" if flying_v else ""
    fv_badge = ('<span class="fv-badge">🪿 Flying V</span>'
                if flying_v else "")

    return f"""
    <article class="pred conf-rail-{conf_slug}{fv_class}">
      <header class="pred-head">
        <span class="pred-matchup">{g['away']} <span class="at">@</span> {g['home']}</span>
        <span class="pred-head-right">
          {fv_badge}
          {biggest_badge}
          <span class="pred-time">{g['when']}</span>
        </span>
      </header>
      <div class="pred-pick">
        <div class="pred-pick-line">
          <span class="pred-pick-team">{g['pick']}</span>
          <span class="pred-pick-odds">{fmt_odds(g['pickOdds'])}</span>
        </div>
        <div class="pred-pick-conf">
          <span class="conf-pill {pill_cls}">{g['confLabel']}</span>
          {pint_row(g['pints'], 13)}
        </div>
      </div>
      {pitching_strip}
      <div class="why-block">
        <div class="why-head">Goose's Gander</div>
        <ul class="edges-list">{edges or '<li class="edge-empty">A quiet one — no standout edge.</li>'}</ul>
      </div>
      {edge_meter_html(g['modelPct'], g['vegasPct'])}
      <div class="form-strip">
        <span class="form-pill"><span class="form-team">{g['away']}</span>
          <span class="form-rec">{af['L10']}</span>
          <span class="form-rd {'pos' if af['RD'] >= 0 else 'neg'}">RD {af['RD']:+d}</span></span>
        <span class="form-pill"><span class="form-team">{g['home']}</span>
          <span class="form-rec">{hf['L10']}</span>
          <span class="form-rd {'pos' if hf['RD'] >= 0 else 'neg'}">RD {hf['RD']:+d}</span></span>
      </div>
      <div class="totals-strip">
        {total_chip}
        {goose_total_chip}
      </div>
      <button class="expand-btn" onclick="this.parentElement.classList.toggle('is-open')">
        <span class="expand-show">Breadcrumbs</span>
        <span class="expand-hide">Hide breadcrumbs</span>
        <span class="chev">▾</span>
      </button>
      <div class="expanded">
        <section class="exp-section">
          <h3 class="exp-title"><span class="exp-icon">⚾</span>Pitching</h3>
          {arsenal_html(g['pitcherAway'], g['away'], g['_arsenal'][g['away']])}
          {arsenal_html(g['pitcherHome'], g['home'], g['_arsenal'][g['home']])}
        </section>
        <section class="exp-section">
          <h3 class="exp-title"><span class="exp-icon">🔥</span>Hitters to Watch</h3>
          <div class="team-batters">
            <div class="team-batters-head">{g['away']} bats</div>
            <ul class="hitter-list">{away_bats or '<li class="hitter-empty">no batting data</li>'}</ul>
          </div>
          <div class="team-batters">
            <div class="team-batters-head">{g['home']} bats</div>
            <ul class="hitter-list">{home_bats or '<li class="hitter-empty">no batting data</li>'}</ul>
          </div>
        </section>
        {prop_section}
        {il_section}
      </div>
    </article>"""


def season_html(s):
    """Season Tracker — a 1:1 recreation of the production dashboard's tab,
    re-skinned in the pub theme: 4 stat cards, tier breakdown, Recent Days
    heatmap (per-day, green/yellow/red graded)."""
    total, pending = s["total"], s["pending"]
    rec = f"{s['wins']}-{s['losses']}" if total else "0-0"
    win_rate = f"{s['acc'] * 100:.1f}"
    streak = s["streak"]
    streak_txt = (streak["type"] + str(streak["count"])
                  if streak["count"] > 0 else "--")
    streak_cls = ("streak-w" if streak["type"] == "W"
                  else "streak-l" if streak["type"] == "L" else "")

    # tier breakdown — same colors as production (green / yellow / gray)
    tier_colors = {"HIGH": "var(--shamrock-bright)",
                   "MEDIUM": "var(--brass-bright)",
                   "LEAN": "var(--foam-dim)"}
    tier_rows = ""
    for t in s["tiers"]:
        pct = f"{t['acc'] * 100:.1f}"
        col = tier_colors.get(t["tier"], "var(--foam-dim)")
        tier_rows += f"""<div class="tier-row">
          <div class="tier-label" style="color:{col}">{t['tier']}</div>
          <div class="tier-bar-bg">
            <div class="tier-bar" style="width:{pct}%;background:{col}"></div>
            <span class="tier-bar-text">{pct}%</span>
          </div>
          <div class="tier-record">{t['wins']}-{t['losses']}</div>
        </div>"""

    # Recent Days heatmap — per-day W/total, good >=65% / ok >=50% / bad <50%
    day_cells = ""
    for d in s["recent"]:
        pct = d["wins"] / d["total"] if d["total"] else 0
        cls = "good" if pct >= 0.65 else "ok" if pct >= 0.5 else "bad"
        short = d["date"][5:]  # MM-DD
        day_cells += f"""<div class="recent-day {cls}" title="{d['date']}: {d['wins']}/{d['total']}">
          <span class="day-rec">{d['wins']}/{d['total']}</span>
          <span class="day-date">{short}</span>
        </div>"""
    recent_block = (f"""<div class="recent-section">
        <h3 class="season-h3">Recent Days</h3>
        <div class="recent-days">{day_cells}</div></div>"""
        if s["recent"] else
        '<div class="no-data">No results yet — check back after games are scored.</div>')

    return f"""
    <section class="season-grid">
      <div class="stat-card">
        <div class="stat-value">{rec}</div>
        <div class="stat-label">Season Record</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{win_rate}%</div>
        <div class="stat-label">Win Rate</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{total + pending}</div>
        <div class="stat-label">Total Picks</div>
      </div>
      <div class="stat-card">
        <div class="stat-value {streak_cls}">{streak_txt}</div>
        <div class="stat-label">Current Streak</div>
      </div>
    </section>
    <div class="tier-breakdown">
      <h3 class="season-h3">Accuracy by Confidence Tier</h3>
      {tier_rows}
    </div>
    {recent_block}"""


def _edge_pts(g):
    """Absolute model-vs-Vegas gap, or -1 when there's no line."""
    if g["vegasPct"] is None:
        return -1
    return abs(g["modelPct"] - g["vegasPct"])


def render(games, season, date_str, prop_board):
    # which game carries the biggest edge — flagged on its own card
    biggest = max(games, key=_edge_pts, default=None)
    top_edge = _edge_pts(biggest) if biggest else 0
    # cards run in chronological order by first pitch
    games = sorted(games, key=lambda g: g.get("_sort_time", "99:99"))
    y, m, d = (int(x) for x in date_str.split("-"))
    date = datetime(y, m, d).strftime("%a, %b %-d")

    cards = ""
    for g in games:
        cards += card_html(g, is_biggest=(g is biggest))

    with open(LOGO, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    with open(TOKENS_CSS) as f:
        tokens_css = f.read()

    # slate-bar stats — overall record / win% / streak / HIGH-tier win rate
    season_rec = (f"{season['wins']}-{season['losses']}"
                  if season["total"] else "0-0")
    streak = season["streak"]
    streak_txt = (streak["type"] + str(streak["count"])
                  if streak["count"] > 0 else "--")
    streak_cls = ("streak-w" if streak["type"] == "W"
                  else "streak-l" if streak["type"] == "L" else "")
    fv_acc = f"{season['fv_acc'] * 100:.0f}" if season["fv_n"] else "—"

    gs_mood, gs_line = goose_status(season, games, date_str)

    return _PAGE.format(
        date=date, logo=logo_b64, tokens=tokens_css, css=_CSS,
        n_games=len(games), top_edge=top_edge,
        season_acc=f"{season['acc'] * 100:.1f}",
        season_rec=season_rec, streak_txt=streak_txt, streak_cls=streak_cls,
        fv_acc=fv_acc, gs_mood=gs_mood, gs_line=gs_line,
        cards=cards, season=season_html(season),
        prop_board=prop_board_html(prop_board))


# --- page template ---------------------------------------------------------

_CSS = """/* ============================================================
   Goose's Projection System — Mobile-first UI
   Single-column feed, max-width 460px, expands gracefully.
   ============================================================ */

* { box-sizing: border-box; }
html, body, #root { margin: 0; padding: 0; }
body {
  background: var(--stout);
  color: var(--fg1);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
  /* wood grain background on desktop wraps the centered phone-width content */
  background-image:
    repeating-linear-gradient(90deg, rgba(255,220,160,.02) 0 1px, transparent 1px 3px, rgba(0,0,0,0.03) 3px 6px),
    radial-gradient(ellipse at 50% 0%, rgba(201,162,74,0.06), transparent 60%);
}

.app {
  max-width: 460px;
  margin: 0 auto;
  min-height: 100vh;
  background:
    repeating-linear-gradient(90deg, rgba(255,220,160,.022) 0 1px, transparent 1px 3px, rgba(0,0,0,0.025) 3px 6px),
    radial-gradient(ellipse at 50% 0%, rgba(201,162,74,0.07), transparent 65%),
    var(--stout);
  position: relative;
}
@media (min-width: 600px) {
  .app {
    box-shadow: 0 0 60px rgba(0,0,0,0.6), 0 0 0 1px var(--line);
  }
}
/* Once the viewport can fit 2 cards (~360px each + gaps), widen the app. */
@media (min-width: 760px) {
  .app { max-width: 760px; }
}
@media (min-width: 1100px) {
  .app { max-width: 1120px; }
}
@media (min-width: 1400px) {
  .app { max-width: 1340px; }
}

/* ============================================================
   HEADER
   ============================================================ */
.header { position: sticky; top: 0; z-index: 50; }
.brass-rail {
  height: 3px;
  background: linear-gradient(180deg, #e6c073, #c9a24a 50%, #8c6e2c);
  box-shadow: 0 1px 0 rgba(0,0,0,0.5);
}
.header-inner {
  background: linear-gradient(180deg, rgba(12,10,8,0.97), rgba(12,10,8,0.92));
  border-bottom: 1px solid var(--line);
  padding: 10px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  backdrop-filter: blur(8px);
}
.brand { text-decoration: none; display: flex; align-items: center; gap: 10px; min-width: 0; }
.brand-patch {
  width: 38px; height: 38px; border-radius: 50%;
  overflow: hidden; flex-shrink: 0;
  box-shadow: 0 0 0 1.5px var(--brass-deep), 0 0 0 2.5px rgba(201,162,74,0.2), 0 4px 10px rgba(0,0,0,0.6);
  background: var(--stout);
}
.brand-patch img { width: 100%; height: 100%; object-fit: cover; display: block; }
.brand-text { display: flex; flex-direction: column; line-height: 1; min-width: 0; }
.brand-1 { font-family: var(--font-display); font-size: 13px; color: var(--foam); text-transform: uppercase; letter-spacing: .005em; }
.brand-2 { font-family: var(--font-display); font-size: 13px; color: var(--brass); text-transform: uppercase; letter-spacing: .005em; margin-top: 1px; }

.header-meta { flex-shrink: 0; }
.date-chip {
  font-family: var(--font-headline); font-weight: 600; font-size: 10px;
  letter-spacing: .14em; text-transform: uppercase; color: var(--brass);
  border: 1px solid var(--line); padding: 5px 9px; border-radius: 2px;
}

/* ============================================================
   MAIN
   ============================================================ */
.main { padding: 14px; }

/* ====== INTRO ====== */
.intro {
  padding: 12px 4px 18px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 14px;
}
.intro-eyebrow {
  display: inline-flex; gap: 8px; align-items: center;
  font-family: var(--font-headline); font-weight: 700;
  font-size: 10px; letter-spacing: .22em; text-transform: uppercase;
  color: var(--brass);
}
.intro-dot {
  width: 6px; height: 6px; border-radius: 999px; background: var(--shamrock-bright);
  box-shadow: 0 0 8px var(--shamrock-bright);
}
.intro-title {
  font-family: var(--font-display);
  font-size: 36px; line-height: 0.94;
  text-transform: uppercase; color: var(--foam); margin: 8px 0 14px;
  letter-spacing: .005em; text-shadow: 0 2px 0 rgba(0,0,0,0.5);
}
.intro-title .gold { color: var(--brass); }
.intro-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.ks { padding: 8px 10px; background: rgba(0,0,0,0.25); border: 1px solid var(--hairline); border-radius: 3px; }
.ks-val {
  font-family: var(--font-stats); font-weight: 700; font-size: 20px;
  color: var(--foam); line-height: 1; font-variant-numeric: tabular-nums;
}
.ks-val.pos { color: var(--shamrock-bright); }
.ks-val.gold { color: var(--brass-bright); }
.ks-val.streak-w { color: var(--shamrock-bright); }
.ks-val.streak-l { color: var(--loss-fg); }
.ks-val .ks-pct { font-size: 12px; color: var(--fg3); margin-left: 1px; }
.ks-lab {
  font-family: var(--font-headline); font-size: 9px;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--fg3); margin-top: 4px;
}

/* ====== SLATE BAR (headline removed — compact stat strip only) ====== */
.slate-bar {
  padding: 12px 4px 14px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 14px;
}
.slate-bar-head {
  display: inline-flex; gap: 8px; align-items: center;
  font-family: var(--font-headline); font-weight: 700;
  font-size: 10px; letter-spacing: .22em; text-transform: uppercase;
  color: var(--brass); margin-bottom: 12px;
}
.slate-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.slate-stats .ks { text-align: center; padding: 9px 6px; }
/* Flying V hit-rate chip — glowing gold, ties to the Flying V cards */
.slate-stats .ks-flyingv {
  border: 1px solid var(--brass);
  background:
    radial-gradient(ellipse at 50% 0%, rgba(201,162,74,0.22), transparent 70%),
    rgba(201,162,74,0.06);
  box-shadow:
    0 0 0 1px rgba(201,162,74,0.25),
    0 0 16px rgba(201,162,74,0.30);
  animation: fv-pulse 3.4s var(--ease-pour) infinite;
}
.slate-stats .ks-flyingv .ks-val {
  color: var(--brass-bright);
  text-shadow: 0 0 14px rgba(230,192,115,0.5);
}
.slate-stats .ks-flyingv .ks-lab { color: var(--brass); }

/* ====== GOOSE STATUS — the meme-fuel line ====== */
.goose-status {
  display: flex; align-items: center; gap: 11px;
  margin: 2px 0 12px;
  padding: 11px 13px;
  background:
    linear-gradient(90deg, rgba(31,107,58,0.14), rgba(201,162,74,0.10) 60%, transparent);
  border: 1px solid var(--line);
  border-left: 3px solid var(--shamrock-bright);
  border-radius: 4px;
}
.gs-goose {
  font-size: 22px; line-height: 1; flex-shrink: 0;
  filter: drop-shadow(0 1px 2px rgba(0,0,0,0.5));
}
.gs-text { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.gs-label {
  font-family: var(--font-headline); font-weight: 600; font-size: 10px;
  letter-spacing: .16em; text-transform: uppercase; color: var(--fg2);
}
.gs-label b {
  color: var(--brass-bright); font-weight: 800; letter-spacing: .1em;
}
.gs-line {
  font-family: var(--font-body); font-size: 12.5px; line-height: 1.4;
  color: var(--foam-dim); font-style: italic;
}

/* ====== BEST EDGE STAMP ====== */
.best-edge-stamp {
  text-align: center;
  font-family: var(--font-headline); font-weight: 700; font-size: 10px;
  letter-spacing: .22em; text-transform: uppercase;
  color: var(--brass-bright);
  padding: 4px 0 8px;
  position: relative;
}
.best-edge-stamp::before, .best-edge-stamp::after {
  content: ""; position: absolute; top: 50%; width: 60px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--brass));
}
.best-edge-stamp::before { left: 0; }
.best-edge-stamp::after { right: 0; transform: scaleX(-1); }

/* ============================================================
   PREDICTION CARD
   ============================================================ */
.feed { display: flex; flex-direction: column; gap: 14px; }

/* Constrain hero/recap/footer to a narrower column on wide screens
   so they don't stretch awkwardly while the card grid uses the full width. */
@media (min-width: 760px) {
  .intro, .recap-section, .footer {
    max-width: 640px;
    margin-left: auto;
    margin-right: auto;
  }
}

/* Auto-fill grid: as many ~360px+ columns as fit, otherwise stack. */
@media (min-width: 760px) {
  .feed {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
    align-items: start;
  }
  .best-edge-stamp { grid-column: 1 / -1; }
}
@media (min-width: 1100px) {
  .feed { gap: 18px; }
}

.pred {
  background:
    radial-gradient(ellipse at 0% 0%, rgba(201,162,74,0.05), transparent 60%),
    var(--bg-card);
  border: 1px solid var(--line);
  border-radius: 5px;
  overflow: hidden;
  position: relative;
  box-shadow: 0 1px 0 rgba(0,0,0,0.5), 0 6px 14px rgba(0,0,0,0.45);
}
/* confidence rail */
.pred::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
}
.conf-rail-high::before    { background: var(--shamrock-bright); box-shadow: 0 0 14px var(--shamrock); }
.conf-rail-medium::before  { background: var(--steel-bright); }
.conf-rail-lean::before    { background: var(--brass); }
.conf-rail-coin-flip::before { background: var(--foam-dim); opacity: .35; }

.pred-head {
  padding: 12px 14px 0;
  display: flex; justify-content: space-between; align-items: baseline;
}
.pred-matchup {
  font-family: var(--font-headline); font-weight: 700; font-size: 12px;
  letter-spacing: .18em; text-transform: uppercase; color: var(--foam-dim);
}
.pred-matchup .at { color: var(--fg3); padding: 0 4px; }
.pred-time {
  font-family: var(--font-stats); font-size: 11px; color: var(--fg3);
  font-variant-numeric: tabular-nums; white-space: nowrap;
}

/* PICK ROW — big team & odds */
.pred-pick {
  padding: 4px 14px 10px;
  display: flex; align-items: flex-end; justify-content: space-between; gap: 10px;
}
.pred-pick-line { display: flex; align-items: baseline; gap: 10px; }
.pred-pick-team {
  font-family: var(--font-display); font-size: 36px; line-height: 0.95;
  text-transform: uppercase; color: var(--foam); letter-spacing: .005em;
}
.pred-pick-odds {
  font-family: var(--font-stats); font-weight: 700; font-size: 20px;
  color: var(--brass); font-variant-numeric: tabular-nums; line-height: 1;
}
.pred-pick-conf {
  display: flex; flex-direction: column; align-items: flex-end; gap: 6px;
  flex-shrink: 0;
}
.pint-row { display: inline-flex; gap: 2px; }
.conf-pill {
  display: inline-flex; align-items: center;
  font-family: var(--font-headline); font-weight: 700;
  font-size: 9px; letter-spacing: .2em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 2px; line-height: 1;
  border: 1px solid transparent;
}
.conf-high { background: rgba(31,107,58,0.22);  color: var(--shamrock-bright); border-color: rgba(58,154,92,0.5); box-shadow: 0 0 10px rgba(58,154,92,0.15); }
.conf-med  { background: rgba(108,138,168,0.18); color: var(--steel-bright);   border-color: rgba(142,176,212,0.5); }
.conf-lean { background: rgba(201,162,74,0.16);  color: var(--brass-bright);   border-color: rgba(201,162,74,0.5); }
.conf-flip { background: rgba(217,205,168,0.10); color: var(--foam-dim);       border-color: rgba(217,205,168,0.3); }

/* ============================================================
   PITCHING STRIP — starters + FIP on the main card face
   ============================================================ */
.pitching-strip {
  margin: 0 14px 10px;
  padding: 8px 10px;
  background: rgba(0,0,0,0.22);
  border: 1px solid var(--line-dim);
  border-radius: 3px;
  display: flex; flex-direction: column; gap: 4px;
}
.ps-row {
  display: flex; align-items: center; gap: 7px;
  font-family: var(--font-body); font-size: 12px; line-height: 1.2;
  color: var(--foam-dim);
}
.ps-name { color: var(--foam); font-weight: 600; }
.ps-hand {
  display: inline-flex; align-items: center; justify-content: center;
  width: 14px; height: 14px; border-radius: 2px;
  font-family: var(--font-headline); font-weight: 700; font-size: 9px;
  letter-spacing: 0;
}
.ps-hand-l { background: rgba(142,176,212,0.22); color: var(--steel-bright); }
.ps-hand-r { background: rgba(201,162,74,0.20);  color: var(--brass-bright); }
.ps-team {
  font-family: var(--font-headline); font-weight: 700; font-size: 9px;
  letter-spacing: .18em; color: var(--fg3);
}
.ps-spacer { flex: 1; }
.ps-edge {
  font-family: var(--font-headline); font-weight: 700; font-size: 9px;
  letter-spacing: .2em; color: var(--brass-bright);
  text-transform: uppercase;
}
.ps-fip {
  display: inline-flex; align-items: baseline; gap: 4px;
  font-family: var(--font-stats); font-variant-numeric: tabular-nums;
  padding: 2px 7px; border-radius: 2px; border: 1px solid transparent;
}
.ps-fip-val { font-weight: 700; font-size: 13px; }
.ps-fip-lbl { font-size: 9px; letter-spacing: .15em; opacity: .75; }
.ps-fip-good { color: var(--shamrock-bright); background: rgba(31,107,58,0.18); border-color: rgba(58,154,92,0.35); }
.ps-fip-avg  { color: var(--brass-bright);    background: rgba(201,162,74,0.14); border-color: rgba(201,162,74,0.35); }
.ps-fip-bad  { color: var(--ember);           background: rgba(217,122,78,0.14); border-color: rgba(217,122,78,0.35); }
.ps-fip-na   { color: var(--fg3); }

/* ============================================================
   EDGE METER — model vs vegas
   ============================================================ */
.edge-meter {
  margin: 0 14px;
  padding: 14px;
  background:
    radial-gradient(ellipse at 50% 0%, rgba(201,162,74,0.10), transparent 70%),
    rgba(0,0,0,0.3);
  border: 1px solid rgba(201,162,74,0.25);
  border-radius: 4px;
  position: relative;
  overflow: hidden;
}
.edge-meter.edge-strong {
  border-color: rgba(201,162,74,0.55);
  box-shadow: 0 0 0 1px rgba(201,162,74,0.15), inset 0 0 30px rgba(201,162,74,0.06);
}
.edge-meter.edge-strong::before {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% 0%, rgba(201,162,74,0.18), transparent 60%);
  pointer-events: none;
}

.edge-magnitude {
  display: flex; align-items: baseline; gap: 6px;
  margin-bottom: 14px;
}
.edge-sign {
  font-family: var(--font-display); font-size: 44px; line-height: 0.85;
  color: var(--brass);
}
.edge-num {
  font-family: var(--font-display); font-size: 64px; line-height: 0.85;
  color: var(--foam);
  text-shadow: 0 2px 0 rgba(0,0,0,0.5);
  font-variant-numeric: tabular-nums;
}
.edge-meter.edge-strong .edge-num { color: var(--brass-bright); text-shadow: 0 0 18px rgba(230,192,115,0.4), 0 2px 0 rgba(0,0,0,0.5); }
.edge-meter.edge-slim .edge-num { color: var(--foam-dim); }
.edge-cap {
  display: flex; flex-direction: column; gap: 2px; padding-bottom: 6px;
}
.edge-cap-1 {
  font-family: var(--font-headline); font-weight: 700; font-size: 13px;
  letter-spacing: .14em; text-transform: uppercase; color: var(--brass);
}
.edge-cap-2 {
  font-family: var(--font-headline); font-size: 10px;
  letter-spacing: .16em; text-transform: uppercase; color: var(--fg3);
}

/* track */
.edge-track { padding-top: 22px; padding-bottom: 6px; }
.edge-track-bar {
  position: relative;
  height: 6px;
  background: var(--tar);
  border-radius: 999px;
  box-shadow: inset 0 1px 0 rgba(0,0,0,0.6);
}
.edge-ticks { position: absolute; inset: 0; }
.edge-tick {
  position: absolute; top: -2px; width: 1px; height: 10px;
  background: rgba(243,233,201,0.1); transform: translateX(-50%);
}
.edge-fill {
  position: absolute; top: 0; bottom: 0;
  background: linear-gradient(90deg, var(--steel-bright), var(--brass-bright));
  border-radius: 999px;
  box-shadow: 0 0 12px rgba(230,192,115,0.5);
}
.edge-meter.edge-strong .edge-fill { box-shadow: 0 0 18px rgba(230,192,115,0.7); }

.edge-marker {
  position: absolute; top: 50%; transform: translate(-50%, -50%);
}
.edge-marker-dot {
  display: block;
  width: 12px; height: 12px; border-radius: 999px;
  border: 2px solid var(--bg-card);
  box-shadow: 0 0 0 1.5px currentColor, 0 0 8px currentColor;
}
.edge-marker-label {
  position: absolute; left: 50%; transform: translateX(-50%);
  white-space: nowrap;
  font-family: var(--font-headline); font-weight: 700; font-size: 10px;
  letter-spacing: .12em; text-transform: uppercase;
  font-variant-numeric: tabular-nums;
}
.edge-marker-vegas { color: var(--brass-bright); }
.edge-marker-vegas .edge-marker-label { bottom: 16px; }
.edge-marker-model { color: var(--steel-bright); }
.edge-marker-model .edge-marker-label { top: 16px; }

.edge-scale {
  display: flex; justify-content: space-between;
  margin-top: 18px;
  font-family: var(--font-stats); font-size: 10px; color: var(--fg3);
  font-variant-numeric: tabular-nums;
}

/* ============================================================
   EDGE STRIP — compact model-vs-Vegas (supporting, not hero)
   ============================================================ */
.edge-strip {
  margin: 0 14px 4px;
  padding: 10px 12px;
  display: flex; align-items: center; gap: 14px;
  background: rgba(0,0,0,0.28);
  border: 1px solid var(--hairline);
  border-radius: 4px;
}
.edge-strip.no-line { color: var(--fg3); justify-content: space-between; }
.edge-strip-label {
  font-family: var(--font-headline); font-weight: 700; font-size: 10px;
  letter-spacing: .14em; text-transform: uppercase; color: var(--fg3);
}
.edge-strip-note { font-family: var(--font-body); font-style: italic; font-size: 12px; }

.edge-strip-mag {
  display: flex; flex-direction: column; align-items: center;
  flex-shrink: 0; min-width: 52px;
}
.edge-strip-num {
  font-family: var(--font-stats); font-weight: 700; font-size: 22px;
  line-height: 1; color: var(--foam-dim); font-variant-numeric: tabular-nums;
}
.edge-strip.edge-strong .edge-strip-num { color: var(--brass-bright); }
.edge-strip.edge-mid .edge-strip-num { color: var(--foam); }
.edge-strip-cap {
  font-family: var(--font-headline); font-size: 8px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--fg3); margin-top: 3px;
}
.edge-strip-track { flex: 1; min-width: 0; }
.edge-strip-track .edge-track-bar {
  position: relative; height: 5px; background: var(--tar);
  border-radius: 999px; box-shadow: inset 0 1px 0 rgba(0,0,0,0.6);
}
.edge-strip-track .edge-fill {
  position: absolute; top: 0; bottom: 0;
  background: linear-gradient(90deg, var(--steel-bright), var(--brass-bright));
  border-radius: 999px; opacity: .8;
}
.edge-strip-track .edge-marker {
  position: absolute; top: 50%; transform: translate(-50%, -50%);
}
.edge-strip-track .edge-marker-dot {
  display: block; width: 9px; height: 9px; border-radius: 999px;
  border: 2px solid var(--bg-card); box-shadow: 0 0 0 1.5px currentColor;
}
.edge-strip-track .edge-marker-vegas { color: var(--brass-bright); }
.edge-strip-track .edge-marker-model { color: var(--steel-bright); }
.edge-strip-legend {
  display: flex; justify-content: space-between; margin-top: 6px;
  font-family: var(--font-headline); font-weight: 600; font-size: 9px;
  letter-spacing: .06em; font-variant-numeric: tabular-nums;
}
.edge-strip-legend .legend-model { color: var(--steel-bright); }
.edge-strip-legend .legend-vegas { color: var(--brass-bright); }

/* biggest-edge badge — on the card header, not a banner */
.pred-head-right { display: flex; align-items: center; gap: 8px; }
.biggest-badge {
  font-family: var(--font-headline); font-weight: 700; font-size: 8.5px;
  letter-spacing: .12em; text-transform: uppercase;
  color: var(--brass-bright);
  background: rgba(201,162,74,0.14);
  border: 1px solid rgba(201,162,74,0.45);
  padding: 3px 7px; border-radius: 2px; white-space: nowrap; line-height: 1;
}

/* ============================================================
   FLYING V — the 5-pint card. Make it scream.
   ============================================================ */
.pred.flying-v {
  border: 2px solid var(--brass);
  background:
    radial-gradient(ellipse at 50% 0%, rgba(201,162,74,0.16), transparent 62%),
    radial-gradient(ellipse at 100% 100%, rgba(58,154,92,0.10), transparent 55%),
    var(--bg-card);
  box-shadow:
    0 0 0 1px rgba(201,162,74,0.3),
    0 0 26px rgba(201,162,74,0.32),
    0 6px 18px rgba(0,0,0,0.55);
  animation: fv-pulse 3.4s var(--ease-pour) infinite;
}
@keyframes fv-pulse {
  0%, 100% { box-shadow: 0 0 0 1px rgba(201,162,74,0.3),
             0 0 22px rgba(201,162,74,0.26), 0 6px 18px rgba(0,0,0,0.55); }
  50%      { box-shadow: 0 0 0 1px rgba(201,162,74,0.45),
             0 0 34px rgba(201,162,74,0.42), 0 6px 18px rgba(0,0,0,0.55); }
}
/* the confidence rail goes brass+green on a Flying V */
.pred.flying-v::before {
  width: 4px;
  background: linear-gradient(180deg, var(--brass-bright), var(--shamrock-bright));
  box-shadow: 0 0 16px rgba(201,162,74,0.6);
}
/* the pick team name catches the gold */
.pred.flying-v .pred-pick-team {
  color: var(--brass-bright);
  text-shadow: 0 0 18px rgba(230,192,115,0.45), 0 2px 0 rgba(0,0,0,0.5);
}
/* the 🪿 Flying V badge */
.fv-badge {
  font-family: var(--font-headline); font-weight: 800; font-size: 8.5px;
  letter-spacing: .12em; text-transform: uppercase;
  color: var(--stout);
  background: linear-gradient(180deg, var(--brass-bright), var(--brass));
  border: 1px solid var(--brass-bright);
  padding: 3px 8px; border-radius: 2px; white-space: nowrap; line-height: 1;
  box-shadow: 0 0 12px rgba(230,192,115,0.5);
}

/* ============================================================
   WHY BLOCK — edges list (the card's primary content)
   ============================================================ */
.why-block { padding: 14px 14px 12px; }
.why-head {
  font-family: var(--font-headline); font-weight: 700; font-size: 11px;
  letter-spacing: .18em; text-transform: uppercase;
  color: var(--brass); margin-bottom: 10px;
}
.edges-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px; }
.edge-row {
  display: grid;
  grid-template-columns: 18px auto 1fr;
  gap: 10px;
  align-items: baseline;
  padding: 6px 0 6px 8px;
  border-left: 2px solid transparent;
  font-size: 13px; line-height: 1.4;
}
.edge-row-icon {
  display: inline-flex; align-items: center;
  align-self: flex-start; padding-top: 3px;
}
.edge-row-label {
  font-family: var(--font-headline); font-weight: 700; font-size: 10px;
  letter-spacing: .14em; text-transform: uppercase;
  white-space: nowrap;
}
.edge-row-detail {
  font-family: var(--font-body); font-size: 13px; color: var(--foam-dim); line-height: 1.4;
}

.edge-kind-model    { border-left-color: var(--brass);          }
.edge-kind-model .edge-row-icon, .edge-kind-model .edge-row-label    { color: var(--brass-bright); }
.edge-kind-lineup   { border-left-color: var(--steel-bright);   }
.edge-kind-lineup .edge-row-icon, .edge-kind-lineup .edge-row-label  { color: var(--steel-bright); }
.edge-kind-pitching { border-left-color: var(--ember);          }
.edge-kind-pitching .edge-row-icon, .edge-kind-pitching .edge-row-label { color: var(--ember); }
.edge-kind-form     { border-left-color: var(--shamrock-bright);}
.edge-kind-form .edge-row-icon, .edge-kind-form .edge-row-label    { color: var(--shamrock-bright); }
.edge-kind-injury   { border-left-color: var(--loss-fg);        }
.edge-kind-injury .edge-row-icon, .edge-kind-injury .edge-row-label  { color: var(--loss-fg); }

/* ============================================================
   FORM STRIP (L10 + run differential)
   ============================================================ */
.form-strip {
  display: flex; gap: 8px; padding: 12px 14px 0;
}
.totals-strip {
  display: flex; gap: 8px; padding: 8px 14px 0;
}
.totals-strip .ou-chip,
.totals-strip .goose-total {
  flex: 1; justify-content: center;
}
.form-pill {
  flex: 1;
  display: flex; align-items: baseline; gap: 6px; justify-content: center;
  background: var(--tar); border: 1px solid var(--line);
  border-radius: 2px; padding: 6px 8px;
  font-family: var(--font-stats); font-size: 11px;
  font-variant-numeric: tabular-nums; white-space: nowrap;
  box-shadow: inset 0 1px 0 rgba(0,0,0,0.4);
}
.form-team { font-family: var(--font-headline); font-weight: 700; letter-spacing: .1em; color: var(--foam-dim); font-size: 10px; }
.form-rec { color: var(--foam); font-weight: 700; }
.form-rd { font-family: var(--font-headline); font-size: 9px; letter-spacing: .08em; }
.form-rd.pos { color: var(--shamrock-bright); }
.form-rd.neg { color: var(--loss-fg); }

/* ============================================================
   EXPAND BUTTON
   ============================================================ */
.expand-btn {
  width: 100%;
  margin-top: 12px;
  padding: 12px 14px;
  background: rgba(0,0,0,0.25);
  border: 0; border-top: 1px solid var(--hairline);
  color: var(--brass); cursor: pointer;
  font-family: var(--font-headline); font-weight: 600;
  font-size: 11px; letter-spacing: .18em; text-transform: uppercase;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  transition: color 120ms, background 120ms;
}
.expand-btn:hover { color: var(--brass-bright); background: rgba(201,162,74,0.06); }
.expand-btn .chev { transition: transform 200ms; display: inline-block; }
.expand-btn .chev.is-open { transform: rotate(180deg); }

/* ============================================================
   EXPANDED DETAILS
   ============================================================ */
.expanded {
  background: rgba(0,0,0,0.25);
  border-top: 1px solid var(--line-dim);
}
.exp-section { padding: 14px; border-bottom: 1px solid var(--line-dim); }
.exp-section:last-child { border-bottom: 0; }
.exp-title {
  font-family: var(--font-headline); font-weight: 700; font-size: 10px;
  letter-spacing: .2em; text-transform: uppercase;
  color: var(--brass); margin: 0 0 10px;
  display: flex; align-items: center; gap: 7px;
}
.exp-icon {
  font-size: 13px; line-height: 1;
  filter: saturate(0.85);
}
.exp-pitcher { margin-bottom: 12px; }
.exp-pitcher:last-child { margin-bottom: 0; }
.exp-pitcher-name {
  font-family: var(--font-headline); font-weight: 700; font-size: 13px;
  color: var(--foam); margin-bottom: 6px; letter-spacing: .04em;
}
.exp-pitcher-team { color: var(--fg3); font-weight: 500; font-size: 11px; letter-spacing: .08em; }

.arsenal { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; }
.arsenal-cell { display: flex; align-items: baseline; gap: 5px; font-size: 12px; }
.ars-pitch { font-family: var(--font-body); font-weight: 600; color: var(--foam); min-width: 48px; }
.ars-usage { font-family: var(--font-stats); color: var(--foam-dim); font-variant-numeric: tabular-nums; min-width: 32px; }
.ars-xrv { font-family: var(--font-stats); color: var(--fg3); font-variant-numeric: tabular-nums; }
.ars-xrv.warn { color: var(--ember); font-weight: 700; }
.arsenal-empty { font-family: var(--font-body); font-style: italic; font-size: 11px; color: var(--fg3); }

.watch {
  background: rgba(201,162,74,0.10);
  border: 1px solid rgba(201,162,74,0.4);
  border-left: 3px solid var(--brass);
  border-radius: 3px;
  padding: 8px 10px;
  margin: 10px 0 0;
  display: flex; gap: 8px; align-items: flex-start;
  font-family: var(--font-body); font-size: 12px; line-height: 1.4;
}
.watch-tag {
  font-family: var(--font-headline); font-weight: 700;
  font-size: 9px; letter-spacing: .18em; text-transform: uppercase;
  color: var(--brass); background: rgba(201,162,74,0.15);
  padding: 3px 6px; border-radius: 2px;
  white-space: nowrap; line-height: 1;
}
.watch-body { color: var(--foam-dim); }
.watch-body strong { color: var(--foam); font-weight: 600; }
.il-list {
  list-style: none; padding: 0; margin: 10px 0 0;
  font-family: var(--font-body); font-size: 11px; line-height: 1.5;
  color: var(--loss-fg);
}

.team-batters { margin-bottom: 10px; }
.team-batters:last-child { margin-bottom: 0; }
.team-batters-head {
  font-family: var(--font-headline); font-weight: 700; font-size: 10px;
  letter-spacing: .16em; text-transform: uppercase;
  color: var(--brass); margin-bottom: 4px;
  padding-bottom: 3px; border-bottom: 1px solid var(--hairline);
}
.hitter-list { list-style: none; padding: 0; margin: 0; }
.hitter {
  padding: 5px 0; border-bottom: 1px solid var(--hairline);
  display: grid; grid-template-columns: 1fr auto; gap: 8px 12px;
  align-items: baseline;
}
.hitter:last-child { border-bottom: 0; }
.hitter-name {
  font-family: var(--font-body); font-weight: 500; font-size: 13px;
  color: var(--foam);
  min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.hitter-stats {
  display: inline-flex; gap: 5px; align-items: baseline;
  font-family: var(--font-stats); font-variant-numeric: tabular-nums;
  flex-shrink: 0; white-space: nowrap;
}
.hitter-ops { color: var(--steel-bright); font-weight: 700; font-size: 12px; }
.hitter-ops-l, .hitter-hr-l {
  font-family: var(--font-headline); font-size: 8px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--fg3);
}
.hitter-hr { color: var(--foam-dim); font-weight: 500; font-size: 12px; }
.hitter-tags { grid-column: 1 / -1; display: flex; gap: 4px; flex-wrap: wrap; margin-top: 2px; }
.tag {
  font-family: var(--font-headline); font-weight: 700;
  font-size: 9px; letter-spacing: .12em; text-transform: uppercase;
  padding: 2px 5px; border-radius: 2px; line-height: 1;
  font-variant-numeric: tabular-nums;
}
.tag-hot   { background: rgba(217,122,78,0.18); color: var(--ember); border: 1px solid rgba(217,122,78,0.4); }
.tag-cold  { background: rgba(108,138,168,0.15); color: var(--steel-bright); border: 1px solid rgba(142,176,212,0.4); }
.tag-plt   { background: rgba(108,138,168,0.12); color: var(--steel-bright); border: 1px solid rgba(142,176,212,0.3); }
.tag-pitch { background: rgba(217,122,78,0.10); color: var(--ember); border: 1px solid rgba(217,122,78,0.3); }

/* ============================================================
   RECAP (small)
   ============================================================ */
.recap-section {
  margin-top: 24px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
}
.recap-head { display: flex; justify-content: space-between; align-items: baseline; padding: 0 4px 10px; }
.recap-eyebrow {
  font-family: var(--font-headline); font-weight: 700; font-size: 11px;
  letter-spacing: .2em; text-transform: uppercase; color: var(--brass);
}
.recap-summary {
  font-family: var(--font-stats); font-size: 12px; font-variant-numeric: tabular-nums;
}
.recap-w { color: var(--shamrock-bright); font-weight: 700; }
.recap-l { color: var(--loss-fg); font-weight: 700; }
.recap-units { font-weight: 700; }
.recap-units.pos { color: var(--shamrock-bright); }
.recap-sep { color: var(--fg3); padding: 0 5px; }

.recap-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px; }
.recap-item {
  display: grid; grid-template-columns: 22px 1fr auto;
  gap: 10px; align-items: center; padding: 7px 10px;
  background: rgba(0,0,0,0.2); border: 1px solid var(--hairline); border-radius: 3px;
}
.recap-mark {
  font-family: var(--font-display); font-size: 13px; line-height: 1;
  text-align: center;
}
.recap-item.is-w .recap-mark { color: var(--shamrock-bright); }
.recap-item.is-l .recap-mark { color: var(--loss-fg); }
.recap-pick { font-family: var(--font-body); font-size: 12px; color: var(--foam-dim); }
.recap-odds { font-family: var(--font-stats); font-weight: 700; font-size: 11px; color: var(--brass); font-variant-numeric: tabular-nums; }

/* ============================================================
   FOOTER
   ============================================================ */
.footer {
  margin-top: 28px; padding: 20px 4px 24px;
  border-top: 1px solid var(--line);
  display: flex; gap: 12px; align-items: flex-start;
}
.footer-patch { width: 40px; height: 40px; border-radius: 50%; box-shadow: 0 0 0 1.5px var(--brass-deep); flex-shrink: 0; }
.footer-text { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.footer-mark { font-family: var(--font-display); font-size: 14px; color: var(--foam); text-transform: uppercase; line-height: 1.1; }
.footer-mark .gold { color: var(--brass); }
.footer-est { font-family: var(--font-headline); font-size: 9px; color: var(--fg3); letter-spacing: .2em; text-transform: uppercase; }
.footer-disc { font-family: var(--font-body); font-size: 11px; color: var(--fg3); font-style: italic; line-height: 1.4; margin-top: 4px; }


/* ============================================================
   ADDITIONS — tabs, season tracker, expand toggle
   (extends the design; same tokens / motifs)
   ============================================================ */

/* Tab nav — chalkboard-style strip under the header */
.tabs {
  display: flex; gap: 2px;
  background: var(--bg-chalk);
  border-bottom: 1px solid var(--line);
  padding: 0 10px;
  position: sticky; top: 49px; z-index: 40;
}
.tab-btn {
  flex: 1;
  background: transparent; border: 0; cursor: pointer;
  font-family: var(--font-headline); font-weight: 700;
  font-size: 11px; letter-spacing: .16em; text-transform: uppercase;
  color: var(--chalk-dim);
  padding: 12px 8px;
  border-bottom: 2px solid transparent;
  transition: color 120ms, border-color 120ms;
}
.tab-btn:hover { color: var(--chalk); }
.tab-btn.is-active { color: var(--brass-bright); border-bottom-color: var(--brass); }

.tab-panel { display: none; }
.tab-panel.is-active { display: block; }

/* expand button show/hide label toggle */
.expand-hide { display: none; }
.pred.is-open .expand-show { display: none; }
.pred.is-open .expand-hide { display: inline; }
.pred.is-open .chev { transform: rotate(180deg); }
.pred .expanded { display: none; }
.pred.is-open .expanded { display: block; }

/* o/u total chip in the form strip */
.ou-chip {
  display: inline-flex; align-items: center;
  font-family: var(--font-stats); font-size: 10px; font-weight: 500;
  color: var(--fg3); border: 1px solid var(--line);
  border-radius: 2px; padding: 5px 8px; white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.ou-chip.hot  { color: var(--ember); border-color: rgba(217,122,78,0.4); }
.ou-chip.cool { color: var(--steel-bright); border-color: rgba(142,176,212,0.4); }

/* Goose's projected total. Lean is computed vs Vegas (over/under/match). */
.goose-total {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--font-stats); font-size: 10px; font-weight: 700;
  color: var(--brass-bright);
  border: 1px solid rgba(201,162,74,0.5);
  background: rgba(201,162,74,0.10);
  border-radius: 2px; padding: 5px 8px; white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.goose-total.over  { color: var(--ember);         border-color: rgba(217,122,78,0.5);  background: rgba(217,122,78,0.10); }
.goose-total.under { color: var(--steel-bright);  border-color: rgba(142,176,212,0.5); background: rgba(142,176,212,0.10); }

/* empty states */
.edge-empty, .hitter-empty {
  font-family: var(--font-body); font-style: italic; font-size: 12px;
  color: var(--fg3); list-style: none; padding: 4px 0;
}

/* no-line edge meter */
.edge-meter.no-line .edge-num { font-size: 40px; color: var(--fg3); }

/* ====== SEASON TRACKER — mirrors the production dashboard tab ====== */
.season-grid {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
  margin-bottom: 16px;
}
@media (min-width: 600px) {
  .season-grid { grid-template-columns: repeat(4, 1fr); }
}
.stat-card {
  background: rgba(0,0,0,0.25); border: 1px solid var(--line);
  border-radius: 4px; padding: 18px 10px; text-align: center;
}
.stat-value {
  font-family: var(--font-stats); font-weight: 700; font-size: 32px;
  color: var(--foam); line-height: 1; font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
.stat-value.streak-w { color: var(--shamrock-bright); }
.stat-value.streak-l { color: var(--loss-fg); }
.stat-label {
  font-family: var(--font-headline); font-weight: 600; font-size: 9.5px;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--fg3); margin-top: 7px;
}
.season-h3 {
  font-family: var(--font-headline); font-weight: 700; font-size: 13px;
  letter-spacing: .08em; text-transform: uppercase;
  color: var(--foam); margin: 0 0 14px;
}
.tier-breakdown {
  background: rgba(0,0,0,0.25); border: 1px solid var(--line);
  border-radius: 4px; padding: 18px 16px; margin-bottom: 16px;
}
.tier-row {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 0; border-bottom: 1px solid var(--hairline);
}
.tier-row:last-child { border-bottom: 0; }
.tier-label {
  width: 78px; flex-shrink: 0;
  font-family: var(--font-headline); font-weight: 700; font-size: 10px;
  letter-spacing: .12em; text-transform: uppercase; color: var(--foam-dim);
}
.tier-bar-bg {
  flex: 1; height: 20px; background: var(--tar);
  border-radius: 2px; overflow: hidden; position: relative;
  box-shadow: inset 0 1px 0 rgba(0,0,0,0.5);
}
.tier-bar { height: 100%; border-radius: 2px; min-width: 2px; }
.tier-bar-text {
  position: absolute; right: 7px; top: 50%; transform: translateY(-50%);
  font-family: var(--font-stats); font-weight: 700; font-size: 10px;
  color: var(--foam); font-variant-numeric: tabular-nums;
}
.tier-record {
  width: 50px; flex-shrink: 0; text-align: right;
  font-family: var(--font-stats); font-size: 11px; color: var(--fg3);
  font-variant-numeric: tabular-nums;
}
/* Recent Days heatmap — per-day W/total, green/yellow/red graded */
.recent-section {
  background: rgba(0,0,0,0.25); border: 1px solid var(--line);
  border-radius: 4px; padding: 18px 16px;
}
.recent-days { display: flex; gap: 7px; flex-wrap: wrap; }
.recent-day {
  display: flex; flex-direction: column; align-items: center;
  gap: 2px; padding: 8px 10px; border-radius: 4px;
  min-width: 52px; border: 1px solid transparent;
}
.recent-day .day-rec {
  font-family: var(--font-stats); font-weight: 700; font-size: 13px;
  font-variant-numeric: tabular-nums;
}
.recent-day .day-date {
  font-family: var(--font-stats); font-size: 9px; color: var(--fg3);
}
.recent-day.good {
  background: rgba(31,107,58,0.22); border-color: rgba(58,154,92,0.4);
}
.recent-day.good .day-rec { color: var(--shamrock-bright); }
.recent-day.ok {
  background: rgba(201,162,74,0.16); border-color: rgba(201,162,74,0.4);
}
.recent-day.ok .day-rec { color: var(--brass-bright); }
.recent-day.bad {
  background: rgba(138,42,42,0.20); border-color: rgba(196,72,72,0.4);
}
.recent-day.bad .day-rec { color: var(--loss-fg); }
.no-data {
  font-family: var(--font-body); font-style: italic; font-size: 13px;
  color: var(--fg3); text-align: center; padding: 24px;
}

/* ============================================================
   PROP EDGES (per-card expanded section)
   Display layer over the bdl_*_today cache. Stays in the pub-card
   aesthetic so the nested block doesn't clash with its parent card.
   The standalone Prop Board tab uses the chalkboard treatment below.
   ============================================================ */

.prop-edges-section .exp-title .exp-icon { filter: hue-rotate(-20deg); }
.prop-edges-list {
  list-style: none; margin: 0; padding: 0;
  display: flex; flex-direction: column; gap: 8px;
}
.prop-edge-row {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 9px 10px;
  background: rgba(0,0,0,0.22);
  border: 1px solid var(--hairline); border-radius: 3px;
  border-left: 3px solid var(--brass);
}
.prop-edge-row.prop-kind-hr      { border-left-color: var(--shamrock-bright); }
.prop-edge-row.prop-kind-hits    { border-left-color: var(--brass-bright); }
.prop-edge-row.prop-kind-kover   { border-left-color: var(--steel-bright); }
.prop-edge-row.prop-kind-kunder  { border-left-color: var(--ember); }
.prop-edge-kind {
  flex-shrink: 0; min-width: 60px;
  font-family: var(--font-headline); font-weight: 800;
  font-size: 9px; letter-spacing: .14em; text-transform: uppercase;
  padding: 4px 8px; border-radius: 2px;
  background: rgba(201,162,74,0.12); color: var(--brass-bright);
  border: 1px solid rgba(201,162,74,0.35); text-align: center; line-height: 1;
}
.prop-kind-hr .prop-edge-kind     { background: rgba(31,107,58,0.22); color: var(--shamrock-bright); border-color: rgba(58,154,92,0.5); }
.prop-kind-hits .prop-edge-kind   { background: rgba(201,162,74,0.16); color: var(--brass-bright); border-color: rgba(201,162,74,0.5); }
.prop-kind-kover .prop-edge-kind  { background: rgba(108,138,168,0.20); color: var(--steel-bright); border-color: rgba(142,176,212,0.5); }
.prop-kind-kunder .prop-edge-kind { background: rgba(217,138,78,0.20); color: var(--ember); border-color: rgba(217,138,78,0.45); }
.prop-edge-body {
  display: flex; flex-direction: column; gap: 4px; min-width: 0; flex: 1;
}
.prop-edge-primary {
  font-family: var(--font-body); font-size: 13.5px; font-weight: 700;
  color: var(--foam); line-height: 1.25;
}
.prop-edge-secondary {
  font-family: var(--font-body); font-size: 11px;
  color: var(--fg3); line-height: 1.3; letter-spacing: .04em;
  text-transform: uppercase;
}
.prop-edge-why {
  font-family: var(--font-body); font-size: 13px; line-height: 1.45;
  color: var(--foam-dim); margin-top: 2px;
}
.prop-edge-tier {
  display: inline-flex; align-items: center; gap: 8px; margin-top: 4px;
  color: var(--brass-bright);
}
.prop-edge-tier-label {
  font-family: var(--font-headline); font-weight: 700;
  font-size: 9px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--brass);
}

/* ============================================================
   PROP BOARD — chalkboard surface (design handoff)
   ------------------------------------------------------------
   Slate-green chalkboard on an oak frame, hung on the pub wall.
   Permanent Marker chalk for all text on the chalkboard surface.
   Brass numbers for the score (matches the design's price-in-brass
   convention from the betting-lines variant of the chalkboard).
   ============================================================ */

.intro-sub {
  font-family: var(--font-body); font-size: 12.5px; line-height: 1.5;
  color: var(--fg2); margin: 0 0 18px; max-width: 38em;
}

.prop-board {
  padding: 4px 0 16px;
  display: flex; flex-direction: column; gap: 22px;
}

/* Compact one-line explainer above the cards. Renders as a <details> so the
   long-form copy is one-tap available without dominating the page. Closed by
   default; the summary row sits as a thin chalk-styled strip. */
.prop-explainer {
  margin: -6px 0 -4px;
  padding: 10px 14px;
  background:
    radial-gradient(circle at 30% 20%, rgba(255,255,255,0.025), transparent 60%),
    var(--chalkboard);
  border-left: 2px solid var(--brass-deep);
  border-radius: 2px;
}
.prop-explainer[open] { padding-bottom: 14px; }
.prop-explainer-summary {
  cursor: pointer;
  list-style: none;
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px;
  color: var(--chalk);
}
.prop-explainer-summary::-webkit-details-marker { display: none; }
.prop-explainer-eyebrow {
  font-family: var(--font-headline);
  font-weight: 700; font-size: 9px;
  letter-spacing: .18em; text-transform: uppercase;
  color: var(--brass);
  flex-shrink: 0;
}
.prop-explainer-line {
  font-family: var(--font-chalk);
  font-size: 14px; line-height: 1.35;
  color: var(--chalk-dim);
  flex: 1; min-width: 0;
}
.prop-explainer-toggle {
  font-family: var(--font-headline);
  font-weight: 700; font-size: 9px;
  letter-spacing: .12em; text-transform: uppercase;
  color: var(--brass-bright);
  flex-shrink: 0;
  white-space: nowrap;
}
.prop-explainer[open] .prop-explainer-toggle::after {
  content: " (close)";
}
.prop-explainer-body {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px dashed rgba(236,231,212,0.18);
  font-family: var(--font-chalk);
  font-size: 14px; line-height: 1.5;
  color: var(--chalk);
}
.prop-explainer-body p { margin: 0 0 8px; }
.prop-explainer-body p:last-child { margin-bottom: 0; }
.prop-explainer-body strong {
  color: var(--brass);
  font-weight: 400;
  font-family: var(--font-chalk);
}
.prop-explainer-body em {
  color: var(--chalk-dim);
  font-style: italic;
}
.prop-explainer-body .cb-caveat {
  font-size: 12.5px; color: var(--chalk-dim);
  border-top: 1px dashed rgba(236,231,212,0.18);
  padding-top: 8px; margin-top: 4px;
}

/* The chalkboard surface — slate green, oak frame, inset shadow,
   subtle radial highlights so it reads as a physical board hung on a wall. */
.chalkboard {
  background:
    radial-gradient(circle at 30% 20%, rgba(255,255,255,0.04), transparent 60%),
    radial-gradient(circle at 70% 80%, rgba(255,255,255,0.03), transparent 50%),
    var(--chalkboard);
  border: 8px solid var(--oak);
  border-radius: var(--r-2);
  padding: 18px 22px 22px;
  box-shadow:
    inset 0 0 60px rgba(0,0,0,0.5),
    0 1px 0 rgba(0,0,0,0.5),
    0 10px 20px rgba(0,0,0,0.55);
  position: relative;
}

/* Chalk dust at the edges — purely cosmetic; doesn't impede the read. */
.chalkboard::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background:
    radial-gradient(ellipse at 0% 100%, rgba(236,231,212,0.015), transparent 30%),
    radial-gradient(ellipse at 100% 0%, rgba(236,231,212,0.012), transparent 30%);
}

/* Card head: title + subtitle stacked in chalk. Subtitle gives the user a
   one-line "what's in this card" prose so they don't have to learn columns. */
.cb-head { margin-bottom: 4px; }

.cb-title {
  font-family: var(--font-chalk);
  font-size: 28px; line-height: 1.05;
  color: var(--chalk);
  transform: rotate(-1deg);
  display: inline-block;
  border-bottom: 2px solid var(--chalk);
  padding-bottom: 2px;
}
.cb-subtitle {
  margin-top: 8px;
  font-family: var(--font-chalk);
  font-size: 14px; line-height: 1.35;
  color: var(--chalk-dim);
}

.cb-rule {
  border-bottom: 1px dashed rgba(236,231,212,0.3);
  margin: 8px 0 10px;
}

/* Tables sit directly on the slate (no extra panel). Borders are dashed
   chalk lines between rows, not solid pixel rules. */
.cb-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 6px;
  font-family: var(--font-chalk);
}

.cb-table thead th {
  font-family: var(--font-headline);
  font-weight: 600; font-size: 10px; letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--chalk-dim);
  text-align: left;
  padding: 4px 10px 8px;
  border-bottom: 1px dashed rgba(236,231,212,0.3);
}
.cb-table thead th.cb-th-tier   { width: 130px; }
.cb-table thead th.cb-th-dir    { width: 78px; text-align: center; }
.cb-table thead th.cb-th-player { width: 22%; }
.cb-table thead th.cb-th-matchup { width: 26%; }
.cb-table thead th.cb-th-why { }

.cb-row td {
  padding: 12px 10px;
  vertical-align: middle;
  border-bottom: 1px dashed rgba(236,231,212,0.10);
}
.cb-row:last-child td { border-bottom: 0; }
.cb-row:hover td { background: rgba(236,231,212,0.025); }

.cb-cell-tier {
  white-space: nowrap;
  vertical-align: middle;
  color: var(--chalk);
}
.cb-cell-tier .ctally-row {
  display: inline-flex; gap: 4px; align-items: center;
  color: var(--chalk);
}
.cb-tier-label {
  display: block;
  font-family: var(--font-chalk);
  font-size: 13px;
  color: var(--chalk-dim);
  margin-top: 4px;
  letter-spacing: 0.01em;
}

.cb-cell-player {
  font-family: var(--font-chalk);
  font-size: 19px;
  color: var(--chalk);
  line-height: 1.15;
}
.cb-cell-player .cb-name { color: var(--chalk); }
.cb-cell-player .cb-team {
  display: block;
  font-family: var(--font-headline);
  font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--chalk-dim); margin-top: 3px;
}
.cb-cell-player .cb-order {
  color: var(--chalk-dim); font-size: 14px;
  font-family: var(--font-chalk);
}

.cb-cell-matchup {
  font-family: var(--font-chalk);
  font-size: 17px;
  color: var(--chalk-dim);
  line-height: 1.2;
}
.cb-cell-matchup .cb-vs {
  font-family: var(--font-chalk);
  color: var(--chalk-dim);
  font-size: 14px;
}
.cb-cell-matchup .cb-game {
  display: block;
  font-family: var(--font-headline);
  font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--brass); margin-top: 3px;
}

.cb-cell-why {
  font-family: var(--font-chalk);
  font-size: 14.5px; line-height: 1.4;
  color: var(--chalk);
}

.cb-cell-direction { text-align: center; }
.cb-dir {
  font-family: var(--font-chalk);
  font-size: 16px;
  padding: 2px 8px;
  border: 2px solid currentColor;
  border-radius: 2px;
  transform: rotate(-1deg); display: inline-block;
  line-height: 1.1;
  letter-spacing: .04em;
}
.cb-dir-over    { color: var(--shamrock-bright); }
.cb-dir-under   { color: var(--ember); }
.cb-dir-neutral { color: var(--chalk-dim); }

.cb-empty {
  font-family: var(--font-chalk); font-size: 18px;
  color: var(--chalk-dim); text-align: center;
  padding: 24px 0 8px;
  transform: rotate(-0.5deg);
}

/* Chalk tally marks — hand-drawn vertical strokes (see chalk_tally_row() in
   Python). Filled strokes take the chalk color at full opacity; empty strokes
   are chalk-dim with a dashed line, like a fading first stroke that hasn't
   been completed yet. Each mark is also slightly rotated at the source so the
   five marks don't read as a typeset bar code. */
.ctally { display: inline-block; vertical-align: middle; }
.ctally-row { display: inline-flex; gap: 4px; align-items: center; }
.ctally.tally-on { color: var(--chalk); }
.ctally.tally-off { color: var(--chalk-dim); }

/* Explainer chalkboard at the top — chalk paragraph variant. */
.chalkboard-note .cb-note {
  font-family: var(--font-chalk);
  font-size: 16px; line-height: 1.5;
  color: var(--chalk);
}
.chalkboard-note .cb-note p { margin: 0 0 10px; }
.chalkboard-note .cb-note p:last-child { margin-bottom: 0; }
.chalkboard-note .cb-key {
  color: var(--brass);
  font-family: var(--font-chalk);
}
.chalkboard-note .cb-caveat {
  font-size: 14px; color: var(--chalk-dim); margin-top: 8px;
  border-top: 1px dashed rgba(236,231,212,0.25);
  padding-top: 8px;
}

/* Footer chalkboard — small chalked metadata strip. */
.chalkboard-foot {
  padding: 10px 22px;
}
.chalkboard-foot .cb-footer {
  font-family: var(--font-chalk);
  font-size: 13px; color: var(--chalk-dim);
  text-align: center;
  letter-spacing: .02em;
}

/* Narrow viewport — shrink the title and tighten the table. The chalkboard
   surface intentionally does NOT collapse to stacked cards on mobile (per
   the design handoff: "the chalkboard metaphor breaks if the columns wrap").
   Instead the matchup column hides its small game label and the why column
   becomes the dominant cell. */
@media (max-width: 600px) {
  .chalkboard { padding: 14px 14px 18px; }
  .cb-title { font-size: 22px; }
  .cb-subtitle { font-size: 13px; }
  .cb-table thead th { font-size: 9px; }
  .cb-table thead th.cb-th-tier { width: 108px; }
  .cb-row td { padding: 10px 6px; }
  .cb-cell-player { font-size: 16px; }
  .cb-cell-matchup { font-size: 14px; }
  .cb-cell-why { font-size: 13px; }
  .cb-cell-matchup .cb-game { display: none; }
}
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Goose's Projection System — Tonight's Predictions</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{tokens}
{css}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <div class="brass-rail"></div>
    <div class="header-inner">
      <a class="brand" href="#" onclick="return false">
        <span class="brand-patch"><img src="data:image/png;base64,{logo}" alt=""></span>
        <span class="brand-text">
          <span class="brand-1">Goose's</span>
          <span class="brand-2">Projection System</span>
        </span>
      </a>
      <div class="header-meta"><span class="date-chip">{date}</span></div>
    </div>
  </header>

  <nav class="tabs">
    <button class="tab-btn is-active" data-tab="tonight">Tonight's Slate</button>
    <button class="tab-btn" data-tab="props">Prop Board</button>
    <button class="tab-btn" data-tab="season">Season Tracker</button>
  </nav>

  <main class="main">
    <div class="tab-panel is-active" id="tab-tonight">
      <div class="goose-status">
        <span class="gs-goose">🪿</span>
        <span class="gs-text">
          <span class="gs-label">Goose Status: <b>{gs_mood}</b></span>
          <span class="gs-line">{gs_line}</span>
        </span>
      </div>
      <section class="slate-bar">
        <div class="slate-bar-head">
          <span class="intro-dot"></span>Tonight's Slate &middot; {n_games} games
        </div>
        <div class="slate-stats">
          <div class="ks"><div class="ks-val">{season_rec}</div>
            <div class="ks-lab">Record</div></div>
          <div class="ks"><div class="ks-val gold">{season_acc}<span class="ks-pct">%</span></div>
            <div class="ks-lab">Win rate</div></div>
          <div class="ks"><div class="ks-val {streak_cls}">{streak_txt}</div>
            <div class="ks-lab">Streak</div></div>
          <div class="ks ks-flyingv"><div class="ks-val gold">{fv_acc}<span class="ks-pct">%</span></div>
            <div class="ks-lab">Flying V hit rate</div></div>
        </div>
      </section>
      <section class="feed">{cards}</section>
    </div>

    <div class="tab-panel" id="tab-props">
      <section class="intro">
        <div class="intro-eyebrow"><span class="intro-dot"></span>Tonight's Slate &middot; Prop Board</div>
        <h1 class="intro-title">Where the <span class="gold">edge</span><br>shows up.</h1>
        <p class="intro-sub">Batter-vs-pitcher matchup signals from balldontlie. Display only &mdash; no model wiring. Sample-gated arsenal xRV, last-15 form, season power. K leans use weighted whiff% &times; opposing lineup vulnerability.</p>
      </section>
      <div class="prop-board">{prop_board}</div>
    </div>

    <div class="tab-panel" id="tab-season">
      <section class="intro">
        <div class="intro-eyebrow"><span class="intro-dot"></span>2026 Season</div>
        <h1 class="intro-title">The <span class="gold">bar tab</span><br>so far.</h1>
      </section>
      {season}
    </div>

    <footer class="footer">
      <img class="footer-patch" src="data:image/png;base64,{logo}" alt="">
      <div class="footer-text">
        <span class="footer-mark">Goose's <span class="gold">Projection System</span></span>
        <span class="footer-est">Est. Largo, FL &middot; 2026</span>
        <span class="footer-disc">GPS is not financial advice. Wet your beak responsibly.</span>
      </div>
    </footer>
  </main>
</div>

<script>
  document.querySelectorAll(".tab-btn").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      var tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(function(b) {{
        b.classList.toggle("is-active", b === btn);
      }});
      document.querySelectorAll(".tab-panel").forEach(function(p) {{
        p.classList.toggle("is-active", p.id === "tab-" + tab);
      }});
      window.scrollTo(0, 0);
    }});
  }});
</script>
</body>
</html>"""


def generate_goose_dashboard(date_str=None, output_path=None):
    """Render the Goose dashboard HTML for date_str (default: today, ET).

    Reads the picks table (deduped lineup_lock > morning) joined to games,
    plus the bdl_* cache tables for odds, arsenals, hitters, form, injuries.
    Writes a self-contained HTML file.
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    output_path = output_path or DEFAULT_OUT
    season = get_current_season()

    games = assemble_games(date_str, season)
    season_data = season_tracker(season)
    prop_board = gather_prop_board(date_str, season)
    html = render(games, season_data, date_str, prop_board)

    with open(output_path, "w") as f:
        f.write(html)

    edge_games = [g for g in games if g["vegasPct"] is not None]
    total_edges = sum(len(g["edges"]) for g in games)
    total_props = sum(len(g.get("prop_edges") or []) for g in games)
    pb_t = prop_board.get("totals", {}) if prop_board else {}
    print(f"  Goose dashboard saved to {output_path}")
    print(f"  {len(games)} games, {len(edge_games)} with a Vegas line, "
          f"{total_edges} typed edges")
    print(f"  Prop edges: {total_props} per-card; board pools "
          f"HR={pb_t.get('hr_pool', 0)} Hits={pb_t.get('hits_pool', 0)} "
          f"K={pb_t.get('k_pool', 0)}")
    return output_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render the Goose dashboard.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--out", help="output HTML path (default: output/goose_dashboard.html)")
    args = ap.parse_args()
    generate_goose_dashboard(date_str=args.date, output_path=args.out)
