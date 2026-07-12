# MLB Game Picker

Daily MLB win probability model for a head-to-head pick accuracy contest.

## Stack
- Python 3.9+, SQLite, scikit-learn (logistic regression), pandas
- Data: MLB Stats API (free), FanGraphs JSON API (free)
- Dashboard: Self-contained HTML deployed to GitHub Pages
- Automation: GitHub Actions (6 daily cron runs)
- Repo: github.com/ArtificialDegeneracy/mlb-picker

## Architecture

```
main.py              CLI entry point (refresh, predict, init, dashboard, status)
scheduler.py         Automation brain (morning, lineup_lock, results modes)
config.py            Season config, thresholds, team mappings, park factors
db.py                SQLite schema + connection management

data/
  mlb_api.py         MLB Stats API client (schedule, pitcher stats, team records, lineups, batter splits)
  fip.py             FIP computation from pitching components
  fangraphs.py       FanGraphs JSON API for wRC+, bullpen ERA, platoon splits
  historical.py      Pull 2022-2025 games for model training
  lineups.py         Lineup-aware features for lineup lock runs

model/
  features.py        Feature engineering (FEATURE_NAMES defines the model input)
  predict.py         Logistic regression training, prediction, opener detection
  retrain.py         Weekly auto-retrain entry point (regression gate + PR creation)

output/
  dashboard.py       HTML dashboard generator (Today's Picks, Season Tracker, Pick History)

.github/workflows/
  daily-picks.yml      GitHub Actions: 6 cron runs + manual dispatch with date override
  weekly-retrain.yml   Sunday 6am ET retrain, opens PR if regression gate passes
```

## Model
- Logistic regression trained on 2022-2024 MLB data (7,287 games)
- Validated on 2025: 68.2% HIGH tier accuracy at 63% threshold (was 67.4% pre-away-overconfidence damping)
- **5 features** (`FEATURE_NAMES` in `model/features.py`): `fip_diff`, `team_quality_diff`, `park_factor`, `home_offense_trend`, `away_offense_trend`
- **wRC+ / bullpen ERA / platoon splits are NOT model inputs** — they feed the dashboard signal-tag system and the away-overconfidence damping rule only. Linear regression couldn't extract clean signal from them (r=0.74-0.92 collinear with team_quality), see TODO 2026-05-18.

## Calibration stack
The raw logreg probability is only the first stage. Each prediction passes through up to 4 post-model corrections, in this order:
1. **Opener dampening** — if listed starter has <10 career GS, shrink (prob - 0.5) by 40%.
2. **Away-overconfidence dampening** — for HIGH-tier away picks with 3+/5 supporting signals (record/FIP/bullpen/wrc/form), shrink toward 50% by 5/10/20% based on signal count. Shipped 2026-05-20.
3. **Lineup-OPS nudge** (lineup_lock only) — `nudge = clip(home_ops - away_ops) * 0.40, ±0.04`, applied additively. Shipped 2026-05-20.
4. **Weakened-lineup dampening** (lineup_lock only) — if a team is missing regulars and their lineup OPS is >4% below their 10-game rolling baseline, shrink (prob - 0.5) by 25%.

**LEAN-flip is DISABLED as of 2026-06-13** — the 5/20 backtest (471 LEAN picks at 45.9% raw, 54.1% if flipped) did not generalize. Across 5/20–6/12 the flip cost -2.6pp on 113 picks; on the June-only subset the flip lost 12 wins out of 62 picks (raw 59.7% → flipped 40.3%). The original signal was likely overfit to early-2026 LEAN picks. The `pick_flipped` column is preserved but now always written as 0. Code is at `model/predict.py:242` and `scheduler.py:241` — both reduced to `pick_flipped = 0`.

`scheduler.py:run_lineup_lock` and `model/predict.py:predict_games` each re-implement this stack. They have drifted before (Bug 2: opener dampening missing from lineup_lock until 5/19) — keep them in sync.

## Daily Run Schedule (GitHub Actions)
All times ET (crons are UTC, DST-adjusted for summer only):
- 8 AM: Morning picks (team-level)
- 11 AM / 2 PM / 5 PM / 8 PM: Lineup lock (batter-level splits, 3-hour window)
- 1 AM: Results (score previous day)

Dashboard: https://artificialdegeneracy.github.io/mlb-picker/

## Key Patterns
- **Per-game dedup**: Dashboard queries must prefer lineup_lock over morning picks using:
  ```sql
  AND p.run_type = (SELECT p2.run_type FROM picks p2 WHERE p2.game_id = p.game_id
    ORDER BY CASE p2.run_type WHEN 'lineup_lock' THEN 0 ELSE 1 END LIMIT 1)
  ```
- **Pitcher stats fallback**: Current season → previous season. Stores both rows keyed by actual season.
- **Early-season guard**: Team W-L records need 10+ games before blending with priors.
- **Artifacts**: DB + model pkl files persist between GitHub Actions runs via upload/download artifacts. Seed files in `seed/` as fallback.
- **Model deploys**: use the `force_seed_model=true` workflow_dispatch input — restores the DB from the artifact as normal but uses the repo checkout's `model/trained_model.pkl` + `scaler.pkl` (the path retrain PRs commit to), so DB history is never touched. Flow: merge the retrain PR, dispatch daily-picks with `force_seed_model=true`. Rollback: revert on main, re-dispatch. `force_seed=true` resets BOTH DB and model to `seed/` and is now guarded: the run fails if the artifact DB has more picks than the seed DB (the 6/13 wipe scenario) unless `allow_history_loss=true` is also set.

## Known Issues (as of 2026-05-20)
- Game time UTC→ET uses month approximation for DST (wrong March 1-13, late Oct)
- GitHub Actions crons are DST-only — off by 1 hour Nov-Mar
- No doubleheader awareness (Game 2 treated same as Game 1)
- ~~Model is never retrained with in-season data~~ — stale, was already wired (see Recently fixed)
- ~~FIP constant hardcoded at 3.10~~ — FIXED 2026-05-20 (see Recently fixed)
- ~~SEASON is hardcoded to 2026~~ — stale, was already wired via `get_current_season()`. The Nov-Feb branch was dead code (returned current year regardless); FIXED 2026-05-20 to correctly return prior year in Jan / early Feb.
- ~~Picks dedup uses MAX(run_type) somewhere~~ — stale, audited 2026-05-20: all 6 dashboard dedup queries use the documented `ORDER BY CASE ... LIMIT 1` pattern.
- ~~Missing DB indices on game_id, player_id, team_name~~ — stale, audited 2026-05-20: all three indexed via primary keys or explicit indices; query plans confirm no table scans on the hot dashboard queries.

## Recently fixed
- 2026-07-12 — Full-system audit + retrain (data through 7/11). Findings:
  - **6/13 force_seed deploy silently wiped 5/25-6/12 history** (464 picks, 248 Final games): `force_seed=true` resets the cloud DB to `seed/mlb_picker.db`, which was dated 5/24. Restored from `mlb_picker.db.cloud-snapshot-20260613` and refreshed `seed/mlb_picker.db` to the merged current DB. **Rule: refresh the seed DB from the latest artifact before every force_seed deploy.**
  - **bdl_odds_today day-game contamination**: lineup_lock crons re-ingest odds up to ~5-6 PM ET with last-write-wins, so day games store in-game/settled odds (stored favorites won 90.1% with 85.5% mean implied prob on 131 day games vs 58.9%/56.7% on night games). Edge Meter is garbage for day games; filter any odds backtest to night games until fixed.
  - **Clean model-vs-Vegas baseline** (night games 6/13-7/11, n=263): Vegas favorite 58.9%, model 57.8%, model-contrarian picks 47.9% (n=71) — the "contrarian edge" hypothesis is NOT supported on clean data.
  - **Goose projected O/U has no signal**: MAE 3.77 vs Vegas 3.65 vs constant-9.2 3.72 on clean games; betting its lean vs the line is ~50%. Display flavor only until recalibrated.
  - **Platoon wRC+ regression**: all 30 teams NULL for 2026 (broken since before 6/13 despite the 5/20 fix). NYY/SF `bullpen_era=0.0` was stuck again — NULLed in the deployed DB (falls back to 2025); the fangraphs plausibility filter blocks bad writes but never repairs stuck rows, and `features._get_bullpen_era` has no <=1.0 guard.
  - **Retrain deployed**: 11,129 games (2022-2026-07-11), time-walked holdout n=390 since 6/13: candidate 57.7% vs prior prod 57.4% (+0.3pp), gate passed. Coefficients stable (team_quality +0.426, fip -0.113; away_offense_trend ≈ 0 — candidate for removal).
  - Shadow XGB (n=390 graded, 6/13-7/12): 53.6%, ≈ production — no swap warranted.
- 2026-06-13 — LEAN-flip DISABLED. June dashboard accuracy cratered (~43% overall vs ~52% YTD). Counterfactual on June 1-12 production picks: raw model 50.7%, post-stack 43.4% (-7.2pp from the stack, dominated by the LEAN-flip). Of 46 LEAN picks flipped in June, the flips lost 10 net wins (raw would have won 60.9%, flipped won 39.1%). The 5/20 backtest of 471 LEAN picks at 45.9% raw → 54.1% flipped did not generalize forward. Stack diff: `pick_flipped = 0` everywhere; column preserved for audit. Retrained model on data through 6/12 (delta vs May-18 model: -0.9pp on 340-game time-walked holdout, so retraining was largely irrelevant — the model itself was fine, the stack was wrong). Also: three stale retrain PRs (#3, #7, #8) were opened weekly but never deployed because daily-picks.yml restores the model from the *artifact*, not from main — so merging the PR wouldn't have helped anyway. Deployed via `force_seed=true` after staging `model/{trained_model,scaler}.pkl` to `seed/`.
- 2026-05-20 — S2 sweep audit: three "known issues" turned out to be stale (picks dedup MAX, missing DB indices, SEASON hardcoded). All three were already fixed or never bugs in the first place; the SEASON note had one real bug embedded (`get_current_season()` Jan-Feb branch was dead code, would have returned wrong year during the offseason).
- 2026-05-20 — LEAN-flip: contest requires picking every game; historical LEAN-tier picks were 45.9% accurate (below coin flip). Flipping yields 54.1%. Applied automatically in `model/predict.py:predict_games` and `scheduler.py:run_lineup_lock` after the rest of the calibration stack. Audit via `picks.pick_flipped` column. Dashboard shows a "Contrarian" badge. **Reverted 2026-06-13 — see above.**
- 2026-05-20 — Retrain gate data leakage: the weekly retrain regression gate was using the last 20% of current season as holdout, but the production model was already trained on those games (commit date > holdout dates) → gate fired spuriously every week, blocking 4 consecutive retrains 5/10-5/19. Fix: use model file mtime as the time-walked cutoff. Validation = games AFTER production was trained; both models evaluated fairly. Skip gate when no time-walked window is available (no leakage advantage either way).
- 2026-05-20 — Bug 4: FIP constant was hardcoded at 3.10. Added per-season `fip_constants` table, fixed broken `update_fip_constant_from_api` (missing SEASON import + no persistence), wired `get_fip_constant_for_season(season, conn)` into all 4 FIP write sites. Daily refresh now populates the cache. Observed range across 2022-2026: 3.10-3.25. The constant cancels in `fip_diff`, so existing model coefficients remain valid; the benefit is consistency when stats from different seasons appear in the same comparison.
- 2026-05-20 — Bug 1: `scheduler.py` was writing lineup OPS into a feature key (`platoon_wrc_diff`) that wasn't in `FEATURE_NAMES`. Lineup OPS was silently discarded. Replaced with a post-model lineup-OPS nudge in the calibration stack.
- 2026-05-20 — Bug 3: NYY+SF had `bullpen_era=0.0` stuck in the DB for 9+ days, inflating their bullpen signal. All 30 teams had NULL `wrc_plus_vs_lhp/vs_rhp` due to silent FanGraphs request failures. Added retry-with-backoff and plausibility bounds in `data/fangraphs.py`.
- 2026-05-19 — Bug 2: `scheduler.py:run_lineup_lock` did not re-run opener detection. Now mirrors `predict_games`.
- 2026-05-18 — Backfilled `pitcher_stats` for 2022-2024 (~26% of training games were falling through to FIP=4.0). FIP coefficient strengthened from -0.05 to -0.12.

## Important Context
- Backfill runs for completed games have lookahead bias (team W-L includes game outcomes before predicting). Only run morning for future/same-day games.
- If cloud artifacts get corrupted, delete via `gh api -X DELETE` and re-run to fall back to seed.
- The user observed the model mostly picks favorites but underdog picks hit well — contrarian picks may be the real edge.
- Series context (sweep attempts, G1 momentum) is shown on cards but NOT used in the model (tested, didn't improve accuracy).
