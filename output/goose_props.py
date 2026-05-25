"""Goose player-prop ranker — pure display layer over bdl_*_today + bdl_pitch_type_stats.

This is *signal surfacing*, not modeling. The 2026-05-22 evaluation
(docs/balldontlie_evaluation.md) said balldontlie data does not improve the
winner-pick model and should NOT be wired into the predictor. Player props are
a different game: we surface batter-vs-pitcher matchup signals as commentary
the user can act on (or ignore) alongside the model pick.

Two public entry points consumed by output/goose_dashboard.py:

    gather_prop_board(date_str, season)
        Slate-wide ranking: top 10 HR / Hits-TB / Pitcher K matchups across
        every game on the slate. Powers the new "Prop Board" tab.

    gather_prop_edges_for_game(conn, game_row, season)
        Per-game callouts: 3-5 prop edges to surface inside the existing card's
        expanded "Breadcrumbs" section. The cheapest signal the user can act on
        next to the model pick.

Both read from the production bdl_* cache populated by data/balldontlie.py:
  - bdl_pitch_type_stats (pitcher arsenals + hitter pitch-type performance)
  - bdl_batting_today (season batting line, scoped by game_date)
  - bdl_form_today (last7/15/30 OPS, scoped by game_date)
  - game_lineups (most recent stored lineup per team)
  - bdl_id_map (MLB-id <-> balldontlie-id crosswalk)

Scoring is league-percentile rank across the slate, weighted-sum across signals.
NO model lookups, NO Vegas line consumption, NO new features wired into picks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# How many rows per category to surface on the Prop Board tab.
BOARD_LIMIT = 10
# Min lineup size to even attempt prop scoring for a game (most are 9).
MIN_LINEUP = 6
# K-prop edge thresholds: above OVER_THRESHOLD lean over, below UNDER_THRESHOLD
# lean under. Center band (50 ± 15) is "neutral lean" with no directional call.
K_OVER_THRESHOLD = 65
K_UNDER_THRESHOLD = 35

# --- bdl_pitch_type_stats helpers (production has one row per (player, pitch_type)) ---

# Production stores xwoba but it's frequently NULL — degrade through this
# chain. Goose dashboard already uses the same fallback in starter_arsenal().
_QUALITY_CHAIN = ("xwoba", "woba", "slg")


def _best_quality(row) -> Optional[float]:
    for f in _QUALITY_CHAIN:
        v = row[f] if f in row.keys() else None
        if v is not None:
            return v
    return None


def _pitcher_arsenal_rows(conn, mlb_pitcher_id: int, season: int) -> List[dict]:
    """All pitch-type rows for one starter, joined with the contact-quality
    fallback. Each dict has: pitch_type, usage, quality, whiff_percent.
    """
    if mlb_pitcher_id is None:
        return []
    b = conn.execute("SELECT bdl_id FROM bdl_id_map WHERE mlb_id=? "
                     "AND entity_type='player'", (mlb_pitcher_id,)).fetchone()
    if not b:
        return []
    rows = conn.execute(
        "SELECT pitch_type, pitch_usage_percent, whiff_percent, "
        "xwoba, woba, slg, ba FROM bdl_pitch_type_stats "
        "WHERE player_id=? AND role='pitcher' AND season=?",
        (b["bdl_id"], season)).fetchall()
    out = []
    for r in rows:
        usage = r["pitch_usage_percent"]
        if usage is None or usage < 1:
            continue
        out.append({
            "pitch_type": r["pitch_type"],
            "usage": usage,
            "quality_allowed": _best_quality(r),    # contact quality opp gets vs this pitch
            "whiff_percent": r["whiff_percent"],
        })
    return out


def _hitter_pitch_type_profile(conn, bdl_player_id: int, season: int) -> Dict[str, dict]:
    """Map pitch_type -> {quality, whiff_percent} for one hitter.
    Returns empty dict when hitter has no pitch-type rows."""
    if bdl_player_id is None:
        return {}
    rows = conn.execute(
        "SELECT pitch_type, whiff_percent, xwoba, woba, slg "
        "FROM bdl_pitch_type_stats "
        "WHERE player_id=? AND role='hitter' AND season=?",
        (bdl_player_id, season)).fetchall()
    profile = {}
    for r in rows:
        q = _best_quality(r)
        if q is not None or r["whiff_percent"] is not None:
            profile[r["pitch_type"]] = {
                "quality": q, "whiff_percent": r["whiff_percent"]}
    return profile


def _bdl_id(conn, mlb_id: int, entity_type: str) -> Optional[int]:
    if mlb_id is None:
        return None
    r = conn.execute(
        "SELECT bdl_id FROM bdl_id_map WHERE mlb_id=? AND entity_type=?",
        (mlb_id, entity_type)).fetchone()
    return r["bdl_id"] if r else None


# --- arsenal-match aggregates ----------------------------------------------

def _hitter_arsenal_match(
    hitter_profile: Dict[str, dict],
    pitcher_arsenal: List[dict],
) -> Optional[Dict[str, float]]:
    """Weighted-by-usage hitter performance vs this pitcher's full arsenal.

    Returns {quality, whiff_against, coverage} or None when there's no overlap
    between the pitcher's pitches and pitches the hitter has sample on. Coverage
    is the fraction of arsenal usage% covered by hitter sample — a low coverage
    means the score is built on partial information.
    """
    if not hitter_profile or not pitcher_arsenal:
        return None
    q_num = w_num = denom = 0.0
    for p in pitcher_arsenal:
        h = hitter_profile.get(p["pitch_type"])
        if not h:
            continue
        u = p["usage"]
        denom += u
        if h["quality"] is not None:
            q_num += h["quality"] * u
        if h["whiff_percent"] is not None:
            w_num += h["whiff_percent"] * u
    if denom == 0:
        return None
    # coverage relative to the FULL arsenal (including uncovered pitches), so
    # we don't reward a feature built on a single 10%-usage pitch.
    full = sum(p["usage"] for p in pitcher_arsenal) or 1
    return {
        "quality": q_num / denom,
        "whiff_against": w_num / denom,
        "coverage": denom / full,
    }


def _pitcher_weighted_whiff(arsenal: List[dict]) -> Optional[float]:
    """Usage-weighted whiff% across the pitcher's arsenal. None if no whiff
    data."""
    num = denom = 0.0
    for p in arsenal:
        if p["whiff_percent"] is None:
            continue
        num += p["whiff_percent"] * p["usage"]
        denom += p["usage"]
    return num / denom if denom else None


def _pitcher_weighted_quality_allowed(arsenal: List[dict]) -> Optional[float]:
    """Usage-weighted contact-quality allowed (lower = better pitcher)."""
    num = denom = 0.0
    for p in arsenal:
        if p["quality_allowed"] is None:
            continue
        num += p["quality_allowed"] * p["usage"]
        denom += p["usage"]
    return num / denom if denom else None


# --- form / batting-line helpers -------------------------------------------

def _form_row(conn, bdl_id: int, date_str: str) -> Optional[dict]:
    r = conn.execute(
        "SELECT season_ops, season_ab, last7_ops, last7_ab, "
        "last15_ops, last15_ab, last30_ops, last30_ab "
        "FROM bdl_form_today WHERE player_id=? AND game_date=?",
        (bdl_id, date_str)).fetchone()
    return dict(r) if r else None


def _batting_row(conn, bdl_id: int, date_str: str) -> Optional[dict]:
    r = conn.execute(
        "SELECT full_name, team, gp, avg, obp, slg, ops, hr, rbi, sb "
        "FROM bdl_batting_today WHERE player_id=? AND game_date=?",
        (bdl_id, date_str)).fetchone()
    return dict(r) if r else None


def _form_delta_ops(form: dict) -> Optional[float]:
    """L15 OPS minus season OPS, when L15 has >= 10 AB. Returns None when the
    sample doesn't support a meaningful delta."""
    if not form or form.get("season_ops") is None:
        return None
    if form.get("last15_ops") is None or (form.get("last15_ab") or 0) < 10:
        return None
    return form["last15_ops"] - form["season_ops"]


# --- lineup lookup (mirrors Goose's team_hitters fallback) -----------------

def _team_lineup(conn, team: str, date_str: str) -> List[dict]:
    """Most recent stored lineup for `team` on or before `date_str`. Each
    entry: {player_id, name, bat_side, bdl_id}. Empty list when no lineup
    is on file."""
    d = conn.execute(
        "SELECT MAX(lineup_date) d FROM game_lineups "
        "WHERE team=? AND lineup_date<=?", (team, date_str)).fetchone()
    if not d or not d["d"]:
        return []
    rows = conn.execute(
        "SELECT gl.player_id, gl.player_name, gl.bat_side, gl.lineup_position, "
        "m.bdl_id FROM game_lineups gl "
        "LEFT JOIN bdl_id_map m ON m.mlb_id=gl.player_id "
        "  AND m.entity_type='player' "
        "WHERE gl.team=? AND gl.lineup_date=? "
        "ORDER BY gl.lineup_position",
        (team, d["d"])).fetchall()
    return [{"player_id": r["player_id"], "name": r["player_name"],
             "bat_side": r["bat_side"], "order": r["lineup_position"],
             "bdl_id": r["bdl_id"]} for r in rows]


def _starter_hand(conn, mlb_id: int) -> Optional[str]:
    """'L' or 'R' for a starter, from pitcher_stats."""
    if mlb_id is None:
        return None
    r = conn.execute(
        "SELECT throw_hand FROM pitcher_stats WHERE player_id=? "
        "AND throw_hand IS NOT NULL ORDER BY season DESC LIMIT 1",
        (mlb_id,)).fetchone()
    return r["throw_hand"] if r else None


def _has_platoon_edge(bat_side: Optional[str], throw_hand: Optional[str]) -> bool:
    """Switch hitters always have it; otherwise opposite-handedness wins."""
    if not bat_side or not throw_hand:
        return False
    return bat_side == "S" or bat_side != throw_hand


# --- HR candidates ----------------------------------------------------------

def _build_hr_candidates_for_game(conn, game: dict, season: int,
                                  date_str: str) -> List[dict]:
    """One HR candidate per (lineup hitter, opposing starter). Raw signals
    only — percentile-rank / edge-score is applied across the slate later."""
    out = []
    for side in ("home", "away"):
        bat_team = game[f"{side}_team"]
        opp_mlb = game["away_starter_id" if side == "home" else "home_starter_id"]
        opp_name = game["away_starter_name" if side == "home" else "home_starter_name"]
        if opp_mlb is None:
            continue
        arsenal = _pitcher_arsenal_rows(conn, opp_mlb, season)
        if not arsenal:
            continue
        opp_hand = _starter_hand(conn, opp_mlb)
        lineup = _team_lineup(conn, bat_team, date_str)
        if len(lineup) < MIN_LINEUP:
            continue
        for b in lineup:
            if b["bdl_id"] is None:
                continue
            bat_row = _batting_row(conn, b["bdl_id"], date_str)
            form_row = _form_row(conn, b["bdl_id"], date_str)
            profile = _hitter_pitch_type_profile(conn, b["bdl_id"], season)
            am = _hitter_arsenal_match(profile, arsenal)
            form_delta = _form_delta_ops(form_row)
            # season HR-per-game (per game played) — proxy for HR rate when
            # bdl_batting_today doesn't expose PA.
            hr_per_g = None
            if bat_row and bat_row.get("hr") is not None and (bat_row.get("gp") or 0) >= 10:
                hr_per_g = bat_row["hr"] / bat_row["gp"]
            out.append({
                "game_id": game["game_id"],
                "matchup": f"{game['away_team']}@{game['home_team']}",
                "bat_team": bat_team,
                "batter": (bat_row or {}).get("full_name") or b["name"],
                "order": b["order"],
                "opp_starter": opp_name,
                # raw signals (percentile-ranked across slate later):
                "arsenal_quality": (am or {}).get("quality"),
                "arsenal_coverage": (am or {}).get("coverage", 0.0),
                "form_delta_ops": form_delta,
                "hr_per_g": hr_per_g,
                "platoon": _has_platoon_edge(b["bat_side"], opp_hand),
            })
    return out


# --- Hits/TB candidates -----------------------------------------------------

def _build_hits_candidates_for_game(conn, game: dict, season: int,
                                    date_str: str) -> List[dict]:
    """Same shape as HR but the underlying signal is SLG-weighted contact
    rather than HR-leaning xwoba, and the percentile weights shift."""
    out = []
    for side in ("home", "away"):
        bat_team = game[f"{side}_team"]
        opp_mlb = game["away_starter_id" if side == "home" else "home_starter_id"]
        opp_name = game["away_starter_name" if side == "home" else "home_starter_name"]
        if opp_mlb is None:
            continue
        arsenal = _pitcher_arsenal_rows(conn, opp_mlb, season)
        if not arsenal:
            continue
        lineup = _team_lineup(conn, bat_team, date_str)
        if len(lineup) < MIN_LINEUP:
            continue
        for b in lineup:
            if b["bdl_id"] is None:
                continue
            bat_row = _batting_row(conn, b["bdl_id"], date_str)
            form_row = _form_row(conn, b["bdl_id"], date_str)
            profile = _hitter_pitch_type_profile(conn, b["bdl_id"], season)
            am = _hitter_arsenal_match(profile, arsenal)
            form_delta = _form_delta_ops(form_row)
            out.append({
                "game_id": game["game_id"],
                "matchup": f"{game['away_team']}@{game['home_team']}",
                "bat_team": bat_team,
                "batter": (bat_row or {}).get("full_name") or b["name"],
                "order": b["order"],
                "opp_starter": opp_name,
                "arsenal_quality": (am or {}).get("quality"),
                "arsenal_coverage": (am or {}).get("coverage", 0.0),
                "form_delta_ops": form_delta,
                "season_slg": (bat_row or {}).get("slg"),
            })
    return out


# --- Pitcher K candidates ---------------------------------------------------

def _build_k_candidates_for_game(conn, game: dict, season: int,
                                 date_str: str) -> List[dict]:
    """One K candidate per starter on the slate (2 per game)."""
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        pid = game[f"{side}_starter_id"]
        pname = game[f"{side}_starter_name"]
        opp_team = game[f"{opp}_team"]
        if pid is None:
            continue
        arsenal = _pitcher_arsenal_rows(conn, pid, season)
        if not arsenal:
            continue
        weighted_whiff = _pitcher_weighted_whiff(arsenal)
        # Lineup whiff vulnerability vs THIS arsenal: weighted whiff% the
        # opposing lineup has historically had vs each pitch type.
        lineup = _team_lineup(conn, opp_team, date_str)
        if len(lineup) < MIN_LINEUP:
            continue
        lineup_whiffs = []
        for b in lineup:
            if b["bdl_id"] is None:
                continue
            profile = _hitter_pitch_type_profile(conn, b["bdl_id"], season)
            am = _hitter_arsenal_match(profile, arsenal)
            if am and am.get("whiff_against") is not None and am["coverage"] >= 0.3:
                lineup_whiffs.append(am["whiff_against"])
        lineup_avg_whiff = (sum(lineup_whiffs) / len(lineup_whiffs)
                            if lineup_whiffs else None)
        out.append({
            "game_id": game["game_id"],
            "matchup": f"{game['away_team']}@{game['home_team']}",
            "pitcher_team": game[f"{side}_team"],
            "pitcher": pname,
            "opp_team": opp_team,
            "weighted_whiff": weighted_whiff,
            "lineup_avg_whiff": lineup_avg_whiff,
            "lineup_sample": len(lineup_whiffs),
        })
    return out


# --- percentile-rank scoring (slate-wide) -----------------------------------

def _percentile(values: List[float], target: Optional[float]) -> Optional[float]:
    """Return target's percentile (0-100) among non-None values. None when
    target itself is None or the pool is too small to be meaningful."""
    if target is None:
        return None
    pool = sorted(v for v in values if v is not None)
    if len(pool) < 3:
        return None
    below = sum(1 for v in pool if v < target)
    return 100.0 * below / len(pool)


# --- plain-English why generators + confidence mapping ----------------------

def _last_name(full_name: str) -> str:
    """Conservative last-name pull. 'Bryce Harper' -> 'Harper'. Falls back to
    the whole string when there's no space (single-name players, etc.)."""
    if not full_name:
        return "the pitcher"
    parts = full_name.strip().split()
    return parts[-1] if parts else full_name


# Matchup-quality tier labels. Deliberately DIFFERENT vocabulary from the
# 5-pint confidence ladder used on the moneyline pick cards — those imply a
# specific outcome prediction; these only rank how favorable the matchup is
# relative to the rest of the slate. A "premier spot" is not a guarantee the
# batter homers; it just means the underlying signals line up better than
# almost any other prop on the board tonight.
TIER_LABELS = {
    5: "Premier spot",
    4: "Strong matchup",
    3: "Solid lean",
    2: "Worth watching",
    1: "Thin edge",
}

# K-prop labels need direction-aware variants since "premier" can mean
# "premier OVER" or "premier UNDER." Strength stays the same; the row's
# `direction` field carries OVER/UNDER and the table renders both.
TIER_LABELS_K_OVER = {
    5: "Premier over",
    4: "Strong over",
    3: "Solid over lean",
    2: "Worth watching",
    1: "Thin edge",
}
TIER_LABELS_K_UNDER = {
    5: "Premier under",
    4: "Strong under",
    3: "Solid under lean",
    2: "Worth watching",
    1: "Thin edge",
}


def _tier_for_batter(score: float) -> int:
    """Map a HR/Hits prop score (0-100, higher=stronger) to a 1-5 tier."""
    if score >= 90:
        return 5
    if score >= 75:
        return 4
    if score >= 60:
        return 3
    if score >= 45:
        return 2
    return 1


def _tier_for_k(score: float) -> int:
    """Map a K-prop score (centered at 50; high=OVER, low=UNDER) to 1-5 tiers
    by absolute distance from neutral."""
    dist = abs(score - 50.0)
    if dist >= 40:
        return 5
    if dist >= 30:
        return 4
    if dist >= 20:
        return 3
    if dist >= 10:
        return 2
    return 1


def _tier_label_k(tier: int, direction: str) -> str:
    """Pick the OVER- or UNDER-flavored label for a K-prop tier."""
    if direction == "OVER":
        return TIER_LABELS_K_OVER[tier]
    if direction == "UNDER":
        return TIER_LABELS_K_UNDER[tier]
    return TIER_LABELS[tier]


# --- Legacy aliases kept until callers fully renamed ------------------------
PINT_LABELS = TIER_LABELS
_pints_for_batter = _tier_for_batter
_pints_for_k = _tier_for_k


def _join_two(a: str, b: str) -> str:
    """Cap a sentence at two clauses so the row stays scannable in a table cell."""
    a_cap = a[0].upper() + a[1:]
    return f"{a_cap}, and {b}."


def _hr_why(c: dict, q_p, f_p, h_p, plt_p) -> str:
    """Plain-English HR 'why' with numbers folded inline.

    Sentence builds from the 2 strongest narrative threads; supporting numbers
    are embedded so the row tells a complete story in one sentence (no
    separate stat line)."""
    bat = c["batter"]
    pit = _last_name(c["opp_starter"])
    clauses = []
    if q_p is not None and q_p >= 70 and c.get("arsenal_quality") is not None:
        clauses.append(
            f"{bat} hits {pit}'s pitch mix well (xRV {c['arsenal_quality']:.2f})")
    if f_p is not None and f_p >= 70 and c.get("form_delta_ops") is not None:
        d = c["form_delta_ops"]
        clauses.append(
            f"he's hot — last 15 games up {d * 1000:.0f} OPS points")
    if h_p is not None and h_p >= 75 and c.get("hr_per_g"):
        clauses.append(
            f"he's at {c['hr_per_g']:.2f} HR per game this year")
    if plt_p is not None:
        clauses.append(f"he has the platoon edge")

    if not clauses:
        return f"{bat} projects above the slate average for HR upside."
    if len(clauses) == 1:
        c1 = clauses[0]
        return c1[0].upper() + c1[1:] + "."
    return _join_two(clauses[0], clauses[1])


def _hits_why(c: dict, q_p, f_p, s_p) -> str:
    bat = c["batter"]
    pit = _last_name(c["opp_starter"])
    clauses = []
    if q_p is not None and q_p >= 70 and c.get("arsenal_quality") is not None:
        clauses.append(
            f"{bat} handles {pit}'s pitch mix (xRV {c['arsenal_quality']:.2f})")
    if f_p is not None and f_p >= 70 and c.get("form_delta_ops") is not None:
        d = c["form_delta_ops"]
        clauses.append(
            f"he's hot — last 15 games up {d * 1000:.0f} OPS points")
    if s_p is not None and s_p >= 75 and c.get("season_slg"):
        clauses.append(
            f"he's slugging {c['season_slg']:.3f} on the year")

    if not clauses:
        return f"{bat} projects above the slate average for a hit or extra base."
    if len(clauses) == 1:
        c1 = clauses[0]
        return c1[0].upper() + c1[1:] + "."
    return _join_two(clauses[0], clauses[1])


def _k_why(c: dict, w_p, v_p) -> str:
    pit = c["pitcher"]
    direction = c.get("direction", "neutral")
    opp = c.get("opp_team", "the opposing lineup")
    clauses = []
    if direction == "OVER":
        if w_p is not None and w_p >= 75 and c.get("weighted_whiff") is not None:
            clauses.append(
                f"{pit} misses bats at {c['weighted_whiff']:.0f}% (top of the slate)")
        if v_p is not None and v_p >= 75 and c.get("lineup_avg_whiff") is not None:
            clauses.append(
                f"{opp}'s lineup is vulnerable to this arsenal "
                f"(projects {c['lineup_avg_whiff']:.0f}% whiff)")
        if not clauses:
            return f"{pit} projects to rack up K's above the slate average."
    elif direction == "UNDER":
        if w_p is not None and w_p <= 25 and c.get("weighted_whiff") is not None:
            clauses.append(
                f"{pit} is a contact pitcher (only {c['weighted_whiff']:.0f}% whiff)")
        if v_p is not None and v_p <= 25 and c.get("lineup_avg_whiff") is not None:
            clauses.append(
                f"{opp}'s lineup makes contact "
                f"(projects {c['lineup_avg_whiff']:.0f}% whiff vs this arsenal)")
        if not clauses:
            return f"{pit} projects to fall short of the slate's average K total."
    else:
        return f"{pit} projects roughly average K production tonight."

    if len(clauses) == 1:
        c1 = clauses[0]
        return c1[0].upper() + c1[1:] + "."
    return _join_two(clauses[0], clauses[1])


# --- scoring (assigns edge_score, tags, why, numbers) -----------------------

def _score_hr_candidates(cands: List[dict]) -> List[dict]:
    quality_pool = [c["arsenal_quality"] for c in cands
                    if c.get("arsenal_coverage", 0) >= 0.4]
    form_pool = [c["form_delta_ops"] for c in cands]
    hr_pool = [c["hr_per_g"] for c in cands]
    for c in cands:
        q_p = (_percentile(quality_pool, c["arsenal_quality"])
               if c.get("arsenal_coverage", 0) >= 0.4 else None)
        f_p = _percentile(form_pool, c["form_delta_ops"])
        h_p = _percentile(hr_pool, c["hr_per_g"])
        # Platoon: +10 percentile-equivalent points when active, 0 otherwise
        plt_p = 60.0 if c.get("platoon") else None  # 60 = "slightly above neutral"

        weights = {"q": 0.40, "f": 0.25, "h": 0.20, "plt": 0.15}
        parts = {"q": q_p, "f": f_p, "h": h_p, "plt": plt_p}
        num = den = 0.0
        for k, v in parts.items():
            if v is not None:
                num += weights[k] * v
                den += weights[k]
        c["edge_score"] = round(num / den, 1) if den > 0 else 0.0
        # Tags: only signals that materially helped (>= 70th percentile)
        tags = []
        if q_p is not None and q_p >= 70:
            tags.append({"kind": "arsenal",
                         "text": f"Crushes arsenal · xRV {c['arsenal_quality']:.2f}"})
        if f_p is not None and f_p >= 70 and c.get("form_delta_ops"):
            tags.append({"kind": "form",
                         "text": f"Hot · L15 {c['form_delta_ops']:+.2f}"})
        if h_p is not None and h_p >= 75 and c.get("hr_per_g"):
            tags.append({"kind": "power",
                         "text": f"Power · {c['hr_per_g']:.2f} HR/G"})
        if plt_p is not None:
            tags.append({"kind": "platoon", "text": "PLT edge"})
        c["tags"] = tags
        # Plain-English why/numbers lines for the chalkboard renderer.
        c["why"] = _hr_why(c, q_p, f_p, h_p, plt_p)
        c["tier"] = _tier_for_batter(c["edge_score"])
        c["tier_label"] = TIER_LABELS[c["tier"]]
        # Legacy aliases for any caller still using the old names.
        c["pints"] = c["tier"]
        c["pint_label"] = c["tier_label"]
    return cands


def _score_hits_candidates(cands: List[dict]) -> List[dict]:
    quality_pool = [c["arsenal_quality"] for c in cands
                    if c.get("arsenal_coverage", 0) >= 0.4]
    form_pool = [c["form_delta_ops"] for c in cands]
    slg_pool = [c["season_slg"] for c in cands]
    for c in cands:
        q_p = (_percentile(quality_pool, c["arsenal_quality"])
               if c.get("arsenal_coverage", 0) >= 0.4 else None)
        f_p = _percentile(form_pool, c["form_delta_ops"])
        s_p = _percentile(slg_pool, c["season_slg"])
        weights = {"q": 0.50, "f": 0.30, "s": 0.20}
        parts = {"q": q_p, "f": f_p, "s": s_p}
        num = den = 0.0
        for k, v in parts.items():
            if v is not None:
                num += weights[k] * v
                den += weights[k]
        c["edge_score"] = round(num / den, 1) if den > 0 else 0.0
        tags = []
        if q_p is not None and q_p >= 70:
            tags.append({"kind": "arsenal",
                         "text": f"Strong vs arsenal · xRV {c['arsenal_quality']:.2f}"})
        if f_p is not None and f_p >= 70 and c.get("form_delta_ops"):
            tags.append({"kind": "form",
                         "text": f"Hot · L15 {c['form_delta_ops']:+.2f}"})
        if s_p is not None and s_p >= 75 and c.get("season_slg"):
            tags.append({"kind": "slg",
                         "text": f"SLG {c['season_slg']:.3f}"})
        c["tags"] = tags
        c["why"] = _hits_why(c, q_p, f_p, s_p)
        c["tier"] = _tier_for_batter(c["edge_score"])
        c["tier_label"] = TIER_LABELS[c["tier"]]
        c["pints"] = c["tier"]
        c["pint_label"] = c["tier_label"]
    return cands


def _score_k_candidates(cands: List[dict]) -> List[dict]:
    """K direction (over/under). Score CENTERED at 50; high = over, low = under."""
    whiff_pool = [c["weighted_whiff"] for c in cands]
    vuln_pool = [c["lineup_avg_whiff"] for c in cands]
    for c in cands:
        w_p = _percentile(whiff_pool, c["weighted_whiff"])
        v_p = _percentile(vuln_pool, c["lineup_avg_whiff"])
        weights = {"w": 0.55, "v": 0.45}
        parts = {"w": w_p, "v": v_p}
        num = den = 0.0
        for k, v in parts.items():
            if v is not None:
                num += weights[k] * v
                den += weights[k]
        c["edge_score"] = round(num / den, 1) if den > 0 else 50.0
        # Direction call from the centered score.
        if c["edge_score"] >= K_OVER_THRESHOLD:
            c["direction"] = "OVER"
        elif c["edge_score"] <= K_UNDER_THRESHOLD:
            c["direction"] = "UNDER"
        else:
            c["direction"] = "neutral"
        tags = []
        if w_p is not None and w_p >= 75:
            tags.append({"kind": "whiff",
                         "text": f"Whiff {c['weighted_whiff']:.0f}%"})
        if v_p is not None and v_p >= 75:
            tags.append({"kind": "vuln",
                         "text": f"Lineup vuln {c['lineup_avg_whiff']:.0f}%"})
        if w_p is not None and w_p <= 25:
            tags.append({"kind": "whiff-low",
                         "text": f"Low whiff {c['weighted_whiff']:.0f}%"})
        if v_p is not None and v_p <= 25:
            tags.append({"kind": "vuln-low",
                         "text": f"Contact lineup {c['lineup_avg_whiff']:.0f}%"})
        c["tags"] = tags
        c["why"] = _k_why(c, w_p, v_p)
        c["tier"] = _tier_for_k(c["edge_score"])
        c["tier_label"] = _tier_label_k(c["tier"], c.get("direction", "neutral"))
        c["pints"] = c["tier"]
        c["pint_label"] = c["tier_label"]
    return cands


# --- public API -------------------------------------------------------------

def gather_prop_board(date_str: str, season: int) -> dict:
    """Slate-wide prop ranking for the 'Prop Board' tab.

    Returns {"hr": [top10], "hits_tb": [top10], "k": [top10 by abs-from-50],
             "totals": {games, hr_pool, ...}, "reason": optional message}.
    """
    from db import get_db
    with get_db() as conn:
        games = conn.execute("""
            SELECT g.game_id, g.game_date, g.home_team, g.away_team,
                   g.home_team_id, g.away_team_id,
                   g.home_starter_id, g.away_starter_id,
                   g.home_starter_name, g.away_starter_name
            FROM games g
            WHERE g.game_date = ?
              AND g.home_starter_id IS NOT NULL
              AND g.away_starter_id IS NOT NULL
        """, (date_str,)).fetchall()
        games = [dict(g) for g in games]
        if not games:
            return {"hr": [], "hits_tb": [], "k": [],
                    "totals": {"games": 0},
                    "reason": f"no games scheduled on {date_str}"}

        hr_all, hits_all, k_all = [], [], []
        for g in games:
            hr_all.extend(_build_hr_candidates_for_game(conn, g, season, date_str))
            hits_all.extend(_build_hits_candidates_for_game(conn, g, season, date_str))
            k_all.extend(_build_k_candidates_for_game(conn, g, season, date_str))

        hr_scored = _score_hr_candidates(hr_all)
        hits_scored = _score_hits_candidates(hits_all)
        k_scored = _score_k_candidates(k_all)

        hr_scored.sort(key=lambda c: c["edge_score"], reverse=True)
        hits_scored.sort(key=lambda c: c["edge_score"], reverse=True)
        # K-prop ranking is by distance from neutral (so both OVER and UNDER
        # leans surface in the top 10).
        k_scored.sort(key=lambda c: abs(c["edge_score"] - 50), reverse=True)

        totals = {"games": len(games),
                  "hr_pool": len(hr_all), "hits_pool": len(hits_all),
                  "k_pool": len(k_all)}
        # Reason for empty board (most common case: lineups not loaded yet).
        reason = None
        if not hr_all and not hits_all:
            # K can populate without lineups (it only needs lineup *whiff* if
            # available — falls back when not). If both batter pools empty,
            # almost always lineups missing.
            n_with_lineups = sum(1 for g in games if _team_lineup(
                conn, g["home_team"], date_str) and _team_lineup(
                conn, g["away_team"], date_str))
            if n_with_lineups == 0:
                reason = ("lineups not yet posted — props populate after "
                          "lineup_lock crons fill the day's lineups")

        return {
            "hr": hr_scored[:BOARD_LIMIT],
            "hits_tb": hits_scored[:BOARD_LIMIT],
            "k": k_scored[:BOARD_LIMIT],
            "totals": totals,
            "reason": reason,
        }


def gather_prop_edges_for_game(conn, game: dict, season: int,
                               date_str: str) -> List[dict]:
    """Per-game prop callouts (3-5) for the expanded section of a game card.

    Returns a list of {kind, primary, secondary, tags, edge_score} where:
      kind: 'HR' | 'HITS' | 'K-OVER' | 'K-UNDER'
      primary: the headline (batter or pitcher name + matchup)
      secondary: short context line
      tags: list of {kind, text} for chip rendering
      edge_score: 0-100

    These are the strongest 3-5 prop edges WITHIN this game, NOT slate-wide
    ranked — that's the Prop Board tab's job. Picked from each game's own
    candidate list to keep the per-card section relevant to that matchup.
    """
    hr = _score_hr_candidates(_build_hr_candidates_for_game(conn, game, season, date_str))
    hits = _score_hits_candidates(_build_hits_candidates_for_game(conn, game, season, date_str))
    k = _score_k_candidates(_build_k_candidates_for_game(conn, game, season, date_str))

    out = []
    # 2 strongest HR candidates in this game (one per side preferred)
    hr.sort(key=lambda c: c["edge_score"], reverse=True)
    for c in hr[:2]:
        if c["edge_score"] >= 55:
            out.append({
                "kind": "HR",
                "primary": f"{c['batter']} · HR upside vs {_last_name(c['opp_starter'])}",
                "secondary": f"{c['bat_team']} bats",
                "why": c.get("why", ""),
                "tier": c.get("tier", 1),
                "tier_label": c.get("tier_label", ""),
                "tags": c["tags"],
                "edge_score": c["edge_score"],
            })
    # 1 strongest Hits/TB candidate
    hits.sort(key=lambda c: c["edge_score"], reverse=True)
    for c in hits[:1]:
        if c["edge_score"] >= 55:
            out.append({
                "kind": "HITS",
                "primary": f"{c['batter']} · Hits/TB vs {_last_name(c['opp_starter'])}",
                "secondary": f"{c['bat_team']} bats",
                "why": c.get("why", ""),
                "tier": c.get("tier", 1),
                "tier_label": c.get("tier_label", ""),
                "tags": c["tags"],
                "edge_score": c["edge_score"],
            })
    # K leans — surface ONE direction per starter (so up to 2 per game)
    for c in sorted(k, key=lambda c: abs(c["edge_score"] - 50), reverse=True):
        if c["direction"] != "neutral":
            out.append({
                "kind": f"K-{c['direction']}",
                "primary": (f"{c['pitcher']} · K's "
                            f"{'over' if c['direction'] == 'OVER' else 'under'} the line"),
                "secondary": f"vs {c['opp_team']} lineup",
                "why": c.get("why", ""),
                "tier": c.get("tier", 1),
                "tier_label": c.get("tier_label", ""),
                "tags": c["tags"],
                "edge_score": c["edge_score"],
            })
    # Cap at 5 — best edges, in a stable order (by absolute score deviation
    # from neutral so OVER/UNDER K leans rank by strength alongside HR/HITS).
    out.sort(key=lambda e: abs(e["edge_score"] - 50), reverse=True)
    return out[:5]


# --- CLI for ad-hoc testing -------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datetime import datetime
    from config import get_current_season
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--season", type=int, default=get_current_season())
    args = ap.parse_args()
    d = args.date or datetime.now().strftime("%Y-%m-%d")
    board = gather_prop_board(d, args.season)
    print(f"\nProp Board — {d}")
    if board.get("reason"):
        print(f"  {board['reason']}")
    t = board["totals"]
    print(f"  {t.get('games', 0)} games · pools: "
          f"HR={t.get('hr_pool', 0)} Hits={t.get('hits_pool', 0)} K={t.get('k_pool', 0)}")
    for cat, header in [("hr", "TOP HR"), ("hits_tb", "TOP HITS/TB"),
                        ("k", "PITCHER K LEANS")]:
        print(f"\n  --- {header} ---")
        for i, c in enumerate(board[cat], 1):
            if cat == "k":
                print(f"  {i:>2}. [{c['edge_score']:>5.1f} {c.get('direction','-'):>5}] "
                      f"{c['pitcher']:<22} ({c['pitcher_team']}) vs {c['opp_team']} "
                      f"lineup  {c['matchup']}")
            else:
                print(f"  {i:>2}. [{c['edge_score']:>5.1f}] {c['batter']:<24} "
                      f"({c['bat_team']}) vs {c.get('opp_starter','?'):<22} "
                      f"{c['matchup']}")
            tier = c.get("tier", c.get("pints", 0))
            label = c.get("tier_label", c.get("pint_label", ""))
            tally = "|" * tier + "·" * (5 - tier)
            print(f"        [{tally}] {label}")
            if c.get("why"):
                print(f"        why: {c['why']}")
