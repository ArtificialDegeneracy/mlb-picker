"""MLB Game Picker configuration."""

from datetime import datetime


def get_current_season():
    """Derive the MLB season year from the current date.

    MLB seasons run roughly late March through October. From January through
    mid-February there's no current-season data, so default to the prior season
    (whose data is complete and queryable). Late February onward, reference
    the year that's starting / in progress.
    """
    now = datetime.now()
    if now.month >= 3:
        return now.year
    if now.month == 2 and now.day >= 15:
        return now.year  # Late Feb: spring training has started
    return now.year - 1  # Jan or early Feb: prior season is the complete one


SEASON = get_current_season()

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.67
MEDIUM_CONFIDENCE_THRESHOLD = 0.55

# Prior weight schedule: how much to weight preseason projections vs actual record
# Fades as the season progresses and actual data becomes more reliable
PRIOR_WEIGHT_BY_MONTH = {
    3: 0.70, 4: 0.50, 5: 0.35, 6: 0.25,
    7: 0.15, 8: 0.10, 9: 0.05, 10: 0.05,
}

# MLB Stats API
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.2  # seconds between API calls

# FIP constant — use previous year's value at season start, recompute monthly
DEFAULT_FIP_CONSTANT = 3.10

# Team ID → abbreviation mapping (MLB Stats API IDs)
TEAM_ID_TO_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

ABBR_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBR.items()}

# 2026 preseason projected win totals (approximate — from Vegas/PECOTA consensus)
WIN_TOTAL_PRIORS = {
    "LAD": 98, "ATL": 93, "NYY": 92, "HOU": 91, "PHI": 90,
    "BAL": 89, "SD":  88, "NYM": 87, "MIN": 86, "SEA": 85,
    "CLE": 85, "SF":  84, "BOS": 84, "TEX": 83, "MIL": 83,
    "ARI": 82, "TB":  81, "CHC": 81, "TOR": 80, "CIN": 79,
    "KC":  78, "STL": 77, "DET": 77, "PIT": 75, "LAA": 74,
    "WSH": 73, "MIA": 70, "COL": 68, "CWS": 67, "OAK": 65,
}

# Park factors (runs per game relative to league average, 1.00 = neutral)
# Source: ESPN Park Factors / FanGraphs, averaged over 2022-2025
PARK_FACTORS = {
    "COL": 1.32,  # Coors Field
    "ARI": 1.09,  # Chase Field
    "BOS": 1.08,  # Fenway Park
    "CIN": 1.07,  # Great American Ball Park
    "TEX": 1.06,  # Globe Life Field
    "CHC": 1.05,  # Wrigley Field
    "PHI": 1.04,  # Citizens Bank Park
    "ATL": 1.03,  # Truist Park
    "MIL": 1.02,  # American Family Field
    "TOR": 1.02,  # Rogers Centre
    "LAA": 1.01,  # Angel Stadium
    "MIN": 1.01,  # Target Field
    "BAL": 1.01,  # Camden Yards
    "NYY": 1.00,  # Yankee Stadium
    "CLE": 1.00,  # Progressive Field
    "DET": 1.00,  # Comerica Park
    "HOU": 0.99,  # Minute Maid Park
    "KC":  0.99,  # Kauffman Stadium
    "STL": 0.99,  # Busch Stadium
    "CWS": 0.99,  # Guaranteed Rate Field
    "WSH": 0.98,  # Nationals Park
    "PIT": 0.98,  # PNC Park
    "SF":  0.97,  # Oracle Park
    "NYM": 0.97,  # Citi Field
    "SD":  0.96,  # Petco Park
    "LAD": 0.96,  # Dodger Stadium
    "SEA": 0.96,  # T-Mobile Park
    "TB":  0.95,  # Tropicana Field
    "MIA": 0.94,  # LoanDepot Park
    "OAK": 0.95,  # Oakland Coliseum
}

# Venue name → team abbreviation (for mapping API venue names to park factors)
VENUE_TO_TEAM = {
    "Coors Field": "COL", "Chase Field": "ARI", "Fenway Park": "BOS",
    "Great American Ball Park": "CIN", "Globe Life Field": "TEX",
    "Wrigley Field": "CHC", "Citizens Bank Park": "PHI", "Truist Park": "ATL",
    "American Family Field": "MIL", "Rogers Centre": "TOR",
    "Angel Stadium": "LAA", "Target Field": "MIN", "Oriole Park at Camden Yards": "BAL",
    "Yankee Stadium": "NYY", "Progressive Field": "CLE", "Comerica Park": "DET",
    "Minute Maid Park": "HOU", "Kauffman Stadium": "KC", "Busch Stadium": "STL",
    "Guaranteed Rate Field": "CWS", "Rate Field": "CWS",
    "Nationals Park": "WSH", "PNC Park": "PIT",
    "Oracle Park": "SF", "Citi Field": "NYM", "Petco Park": "SD",
    "Dodger Stadium": "LAD", "T-Mobile Park": "SEA",
    "Tropicana Field": "TB", "loanDepot park": "MIA", "LoanDepot Park": "MIA",
    "Oakland Coliseum": "OAK", "RingCentral Coliseum": "OAK",
}
