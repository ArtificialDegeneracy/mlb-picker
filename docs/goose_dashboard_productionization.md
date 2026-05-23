# Goose's Projection System — Productionization Plan

**Status:** prototype complete (in `scratch/`), not yet wired to the pipeline.
**Decision made (2026-05-22):** subscribing to **balldontlie GOAT tier for MLB**
($39.99/mo, 600 req/min). Other sports / data points TBD later.
**Goal:** the new branded dashboard regenerates itself daily, deployed to
GitHub Pages, with no manual steps.

---

## TL;DR — what has to happen, in order

1. **Activate the GOAT subscription**, put the API key in GitHub Actions secrets.
2. **Promote the prototype** out of `scratch/` into real `data/` + `output/` modules.
3. **Build the balldontlie ingest job** — a proper, idempotent module (the GOAT
   rate limit makes this straightforward; no more 5-req/24s crawling).
4. **Wire it into `.github/workflows/daily-picks.yml`** as new steps.
5. **Fix the results-scoring gap** (graded picks stopped at 5/4 — a real
   pre-existing bug, independent of this work).
6. **Decide deploy model** — replace the current dashboard, or run side-by-side.

Estimated effort: ~1 focused day of work. None of it is hard now that the
rate limit is gone; it's mostly turning throwaway scripts into durable code.

---

## Where things stand today

Everything is in the `balldontlie-trial` git worktree, under `scratch/`
(gitignored — throwaway by design):

| File | What it is | Fate |
|------|-----------|------|
| `scratch/goose_dashboard.py` | THE dashboard generator (final) | → promote to `output/` |
| `scratch/goose_assets/` | logo + `colors_and_type.css` | → move to `output/assets/` |
| `scratch/bdl_dashboard*.py`, `market_board.py` | earlier prototypes | → delete |
| `scratch/fetch_*.py`, `odds_backtest.py` | one-off trial scripts | → delete (logic moves into the ingest module) |
| `bdl_*` SQLite tables | already in `db.py` schema | keep — production uses them |
| `bdl_odds_today`, `bdl_batting_today`, `bdl_form_today` | created ad-hoc by trial scripts | → formalize in `db.py` SCHEMA |

**The dashboard works today only because the data was pulled by hand.** It is
not self-updating. That is the whole gap this plan closes.

---

## Step 1 — Subscription & secrets

1. Activate balldontlie **GOAT tier, MLB**. Confirm the production API key
   (may differ from the trial key).
2. Add it to the GitHub repo: **Settings → Secrets and variables → Actions →
   New repository secret**, name `BALLDONTLIE_API_KEY`.
3. Locally, keep it in `data/.bdl_key` (already gitignored) for dev runs.

GOAT = 600 req/min. The daily ingest is ~600 calls total → it finishes in
about a minute. None of the trial's rate-limit fragility carries over.

---

## Step 2 — Promote the prototype to real modules

| From (scratch) | To (production) |
|----------------|-----------------|
| `scratch/goose_dashboard.py` | `output/goose_dashboard.py` |
| `scratch/goose_assets/hat-logo-sm.png` | `output/assets/goose/hat-logo-sm.png` |
| `scratch/goose_assets/colors_and_type.css` | `output/assets/goose/colors_and_type.css` |

Changes needed when promoting `goose_dashboard.py`:
- It currently **hard-codes `2026-05-22`** in ~5 places (`assemble_games`,
  `season_tracker`, `_PAGE` date, `goose_status`). Replace with a `date_str`
  parameter, defaulting to `get_current_season()`-aware "today", exactly like
  `output/dashboard.py:generate_dashboard(date_str)`.
- It **hard-codes the 12→15 `BDL_GAME` map** (balldontlie game-id → team
  abbrs). This must be built dynamically each run from the balldontlie
  `/games?dates[]=` response — see Step 3.
- Keep it a pure renderer: `generate_goose_dashboard(date_str)` reads the DB,
  writes the HTML. No API calls inside it.

---

## Step 3 — The balldontlie ingest module (`data/balldontlie.py`)

This is the real new code. One module, two public functions:

```
data/balldontlie.py
  ├─ _client()            — auth, GOAT-rate-aware session, retry/backoff
  ├─ resolve_crosswalk(conn)        — refresh bdl_id_map (teams + players)
  └─ ingest_for_date(conn, date)    — pull everything today's slate needs
```

### `resolve_crosswalk(conn)`
The trial used `search=` and only resolved ~290 players. **Use the
`first_name` + `last_name` params instead** — that was the fix discovered
during the trial; `search=` matches first name only. Also pull `/teams` once
(crosswalk by abbreviation; note balldontlie uses `CHW`, project uses `CWS` —
the id-keyed `bdl_id_map` handles it). Run this nightly; rosters change.

### `ingest_for_date(conn, date)` — populates, per the day's slate:
| Endpoint | Fills table | Notes |
|----------|-------------|-------|
| `/games?dates[]=` | (in-memory game-id map) | maps bdl game-id → teams |
| `/odds?game_ids[]=` | `bdl_odds_today` | moneyline + total, 6 books |
| `/pitcher_pitch_type_season_stats` | `bdl_pitch_type_stats` | starters' arsenals |
| `/hitter_pitch_type_season_stats` | `bdl_pitch_type_stats` | lineup hitters |
| `/season_stats` | `bdl_batting_today`, `bdl_season_stats` | OPS/HR/WAR/QS |
| `/players/splits` | `bdl_form_today` | last-15 OPS (hot/cold) |
| `/player_injuries` | `bdl_injuries` | IL list |

Requirements for production-grade ingest (the trial scripts had none of these):
- **Idempotent** — `INSERT OR REPLACE`; safe to re-run mid-day for the
  lineup-lock passes.
- **Per-day scoping** — `bdl_odds_today` / `bdl_batting_today` / `bdl_form_today`
  should carry a `game_date` column and the dashboard should read only today's
  rows (the trial's `_today` tables had no date column — fine for one day,
  wrong for a daily job). **Add `game_date` to these three tables in `db.py`.**
- **Graceful partial failure** — if odds aren't posted yet for a late game,
  log and continue; the dashboard already degrades (no-line state).
- **A liveness-safe runner** — the trial's background jobs zombied silently.
  At GOAT rate the ingest is ~1 minute synchronous; just run it inline in the
  Action, no backgrounding.

### Formalize the schema
Add to `db.py` `SCHEMA` (currently created ad-hoc by scratch scripts):
- `bdl_odds_today` — **+ `game_date` column**
- `bdl_batting_today` — **+ `game_date` column**
- `bdl_form_today` — **+ `game_date` column**

---

## Step 4 — Wire into the GitHub Actions cron

`.github/workflows/daily-picks.yml` runs 6 crons/day. Add an ingest +
regenerate step. The new daily shape:

| ET time | Existing step | NEW steps |
|---------|--------------|-----------|
| 8 AM | morning picks | `balldontlie.resolve_crosswalk` + `ingest_for_date` + generate goose dashboard |
| 11/2/5/8 | lineup lock | re-run `ingest_for_date` (odds move, lineups post) + regenerate |
| 1 AM | results scoring | regenerate (Season Tracker picks up new grades) |

Concretely, after the existing `python scheduler.py <mode>` step, add:

```yaml
- name: Ingest balldontlie data
  env:
    BALLDONTLIE_API_KEY: ${{ secrets.BALLDONTLIE_API_KEY }}
  run: python -c "from data.balldontlie import resolve_crosswalk, ingest_for_date; \
    from db import get_db; from datetime import datetime; \
    d=datetime.now().strftime('%Y-%m-%d'); \
    conn=get_db().__enter__(); resolve_crosswalk(conn); ingest_for_date(conn, d)"

- name: Regenerate Goose dashboard
  run: python -c "from output.goose_dashboard import generate_goose_dashboard; \
    from datetime import datetime; \
    generate_goose_dashboard(datetime.now().strftime('%Y-%m-%d'))"
```

The `bdl_*` tables persist between runs via the existing DB artifact
upload/download — no new artifact plumbing needed.

### Deploy
The workflow already builds a Pages artifact and has a `deploy-dashboard` job.
Two options (Step 6 decision):
- **Replace:** point Pages at `goose_dashboard.html` instead of
  `mlb_picks.html`.
- **Side-by-side:** publish both; `goose_dashboard.html` at a new path while
  the old one stays as a fallback during a soak period. **Recommended** — let
  the new one run a week before retiring the old.

---

## Step 5 — Fix the results-scoring gap (pre-existing bug)

Independent of balldontlie, but it caps the Season Tracker. **Graded picks in
the live DB stop at 5/4** even though games are Final later — the 1 AM results
cron either stopped grading or the `games` rows never got their final scores.
During the trial we manually backfilled 5/5–5/20 with `scheduler.run_results`.

To fix properly:
1. Check the GitHub Actions run history for the 1 AM `results` job — has it
   been failing, or running but not grading?
2. The symptom (games stored but `status` not `Final`, `winner` NULL) points
   at `get_game_results` / `run_results` not being called for those dates, or
   the morning run inserting games that the results run never revisits.
3. One-time: run `scheduler.run_results` for every ungraded past date to
   backfill. Then ensure the cron grades reliably going forward.

This is worth doing regardless — it fixes the *current* live dashboard too.

---

## Step 6 — Open decisions for you

1. **Replace vs. side-by-side** the existing dashboard (Step 4 deploy).
   Recommendation: side-by-side soak for ~1 week.
2. **Lineups.** The dashboard's "Hitters to Watch" uses each team's most
   recent *stored* lineup. Production should pull the day's confirmed lineup
   during the lineup-lock runs (the pipeline already fetches lineups for
   `run_lineup_lock` — just feed them to the ingest).
3. **Injury filter.** `bdl_injuries` currently includes long-term/60-day IL,
   so counts look high. Consider filtering to `return_date` within ~10 days
   for the "Injury Notes" panel.
4. **Edge backtest caveat.** The Edge Meter is shown as a decision aid; the
   trial proved balldontlie's *historical* odds are post-game-contaminated, so
   "does the edge predict" was never validated. Fine for a display feature —
   just don't let it silently become a model input without a clean backtest.

---

## What NOT to do

- **Don't wire balldontlie data into the model.** The trial tested it as model
  features four different ways — all four failed (see
  `docs/balldontlie_evaluation.md`). The GOAT subscription is for the
  *dashboard*, not the predictor. The 5-feature logistic-regression model and
  its calibration stack stay exactly as they are.
- **Don't skip the `game_date` columns.** Without them the `_today` tables
  silently serve stale data on the second day.
- **Don't background the ingest.** At GOAT rate it's a fast synchronous job;
  the trial's background runners zombied twice.
