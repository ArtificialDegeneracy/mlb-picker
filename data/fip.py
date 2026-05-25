"""FIP (Fielding Independent Pitching) computation from raw pitching stats."""

import logging
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DEFAULT_FIP_CONSTANT, SEASON

logger = logging.getLogger(__name__)


def compute_fip(hr, bb, hbp, k, ip, fip_constant=None):
    """Compute FIP from pitching components.

    Formula: ((13*HR + 3*(BB+HBP) - 2*K) / IP) + FIP_constant

    Args:
        hr: Home runs allowed
        bb: Walks
        hbp: Hit by pitch
        k: Strikeouts
        ip: Innings pitched
        fip_constant: League-specific constant (~3.10). Uses default if None.

    Returns:
        FIP value (float), or None if insufficient innings.
    """
    if fip_constant is None:
        fip_constant = DEFAULT_FIP_CONSTANT

    if ip is None or ip < 1.0:
        return None

    try:
        fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + fip_constant
        return round(fip, 2)
    except (TypeError, ZeroDivisionError):
        return None


def compute_fip_from_stats(pitcher_stats, fip_constant=None):
    """Compute FIP from a pitcher stats dict (as returned by mlb_api).

    Args:
        pitcher_stats: Dict with keys hr, bb, hbp, k, ip
        fip_constant: Override constant

    Returns:
        FIP value or None
    """
    if pitcher_stats is None:
        return None
    return compute_fip(
        hr=pitcher_stats.get("hr", 0),
        bb=pitcher_stats.get("bb", 0),
        hbp=pitcher_stats.get("hbp", 0),
        k=pitcher_stats.get("k", 0),
        ip=pitcher_stats.get("ip", 0),
        fip_constant=fip_constant,
    )


def update_fip_constant_from_api(season=None, conn=None):
    """Fetch league-wide pitching totals, compute the season FIP constant,
    and persist it in the `fip_constants` table.

    Returns the computed constant, or DEFAULT_FIP_CONSTANT on failure.
    """
    from data.mlb_api import _api_get
    season = season or SEASON

    data = _api_get("/teams/stats", params={
        "stats": "season",
        "season": season,
        "group": "pitching",
        "sportIds": 1,
    })

    if not data:
        logger.warning(f"FIP constant: no API data for {season}, falling back to DEFAULT_FIP_CONSTANT")
        return DEFAULT_FIP_CONSTANT

    # Sum league totals
    lg_era = lg_hr = lg_bb = lg_hbp = lg_k = lg_ip = 0
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split.get("stat", {})
        try:
            lg_ip += float(s.get("inningsPitched", 0))
        except (ValueError, TypeError):
            continue
        lg_era_val = s.get("era")
        if lg_era_val:
            try:
                lg_era += float(lg_era_val) * float(s.get("inningsPitched", 0))
            except (ValueError, TypeError):
                pass
        lg_hr += s.get("homeRuns", 0)
        lg_bb += s.get("baseOnBalls", 0)
        lg_hbp += s.get("hitByPitch", 0)
        lg_k += s.get("strikeOuts", 0)

    # Early season: not enough innings for a reliable constant. Use prior season.
    if lg_ip < 4000:
        logger.info(f"FIP constant: {season} only has {lg_ip:.0f} IP, using prior season constant")
        return _resolve_constant(season - 1, conn) if conn else DEFAULT_FIP_CONSTANT

    if lg_ip <= 0:
        return DEFAULT_FIP_CONSTANT

    lg_era_weighted = lg_era / lg_ip
    constant = compute_league_fip_constant(lg_era_weighted, lg_hr, lg_bb, lg_hbp, lg_k, lg_ip)
    logger.info(f"FIP constant for {season}: {constant} (lgERA={lg_era_weighted:.3f}, lgIP={lg_ip:.0f})")

    if conn is not None:
        conn.execute("""
            INSERT INTO fip_constants (season, fip_constant, computed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(season) DO UPDATE SET
                fip_constant = excluded.fip_constant,
                computed_at = excluded.computed_at
        """, (season, constant, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return constant


def get_fip_constant_for_season(season, conn):
    """Read the cached FIP constant for `season`. Falls back through:
      1. fip_constants table row for `season`
      2. fip_constants table row for `season - 1` (early-season fallback)
      3. DEFAULT_FIP_CONSTANT

    Does NOT make API calls — pure read. To populate the cache, call
    update_fip_constant_from_api() during the daily refresh.
    """
    row = conn.execute(
        "SELECT fip_constant FROM fip_constants WHERE season = ?", (season,)
    ).fetchone()
    if row is not None:
        return row[0]

    row = conn.execute(
        "SELECT fip_constant FROM fip_constants WHERE season = ?", (season - 1,)
    ).fetchone()
    if row is not None:
        return row[0]

    return DEFAULT_FIP_CONSTANT


def _resolve_constant(season, conn):
    """Internal helper used by update_fip_constant_from_api during early-season fallback."""
    row = conn.execute(
        "SELECT fip_constant FROM fip_constants WHERE season = ?", (season,)
    ).fetchone()
    return row[0] if row else DEFAULT_FIP_CONSTANT


def compute_league_fip_constant(league_era, league_hr, league_bb, league_hbp, league_k, league_ip):
    """Derive the FIP constant from league-wide totals.

    FIP_constant = lgERA - ((13*lgHR + 3*(lgBB+lgHBP) - 2*lgK) / lgIP)

    This is typically ~3.10 but varies year to year.
    """
    if league_ip is None or league_ip < 100:
        return DEFAULT_FIP_CONSTANT
    try:
        raw = (13 * league_hr + 3 * (league_bb + league_hbp) - 2 * league_k) / league_ip
        constant = league_era - raw
        return round(constant, 2)
    except (TypeError, ZeroDivisionError):
        return DEFAULT_FIP_CONSTANT
