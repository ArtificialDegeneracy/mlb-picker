# Open Items

Living list of model/dashboard improvements we've identified but not yet shipped.
Most-recent additions at the top.

---

## Champion/challenger shadow testing (set up 2026-05-21)

Production (logreg, `model/predict.py` + `scheduler.py` + the daily crons) keeps
making the real contest picks, untouched. An XGBoost **challenger** now runs in
parallel and is scored alongside production — a live forward comparison, so we
don't swap models on backtest evidence alone (a backtest can be fooled by
overfitting; a shadow test on unseen games cannot).

**Built (this commit):**
- `shadow_picks` table (`db.py` + `migrate.py`) — `model_version`-tagged,
  structurally isolated. Nothing in the production path (dashboard, run_results,
  retrain) reads it. Verified: full predict→score→report cycle leaves the
  `picks` table byte-identical.
- `model/shadow.py` — `predict` / `score` / `report` subcommands. Trains an
  XGBoost challenger time-walked (only ever scored on games it didn't train on),
  tiers picks with the same HIGH/MED/LEAN thresholds as production.

**Deliberate scope of the FIRST comparison:**
- Challenger uses the SAME 5 production features — isolates "does the tree model
  beat the linear model?" from the separate "do balldontlie features help?"
  question. A second `model_version='xgb_bdl'` shadows the balldontlie features
  once the trial proves them out.
- Challenger does NOT apply production's 5-stage calibration stack (opener
  dampening, LEAN-flip, etc.) — that stack is logreg-tuned; applying it to
  XGBoost would confound the comparison. First comparison is raw-model vs
  raw-model.

**Next step:** wire `python -m model.shadow predict` into a daily cron next to
the production run, and `score` the morning after. Let it accumulate 2-3 weeks
(`model/shadow.py report` flags small-n), then decide on the swap with both the
backtest AND the live shadow record in hand.

---

## Path B sub-item: balldontlie feature expansion (evaluated 2026-05-21)

Groundwork for **Path B (XGBoost)**. The linear model has hit its feature-set
ceiling (see "Path C" below); balldontlie is an *additive* source for richer,
nonlinear features the current sources (MLB Stats API, FanGraphs) don't expose.

**Not a migration.** MLB Stats API stays (it's the origin; balldontlie is a
republisher one hop behind, and free-tier rate limits can't serve the
~270-call `lineup_lock` path). FanGraphs stays (no wRC+/bullpen ERA in
balldontlie). FIP stays local (`data/fip.py`).

**Evaluation artifacts (this commit) — two scripts, run in order:**
- `scratch/balldontlie_probe.py` — data-availability probe. "Does the data
  exist and is it clean?" Resolves real DB players → balldontlie IDs, probes 6
  endpoints + a historical-depth sweep. Free-tier-safe (~30-40 calls).
- `scratch/balldontlie_backtest.py` — predictive-value backtest. "Does the data
  make a *better* product?" Measures INCREMENTAL signal vs the current model on
  scored 2026 games: residual correlation, collinearity, disagreement-case win
  rate. H2H is the FULL lineup-weighted version (every batter in both stored
  lineups, AB-weighted) — ~1,100 calls, ~3.5-4 hrs at the trial's 5 req/min, run
  unattended. Sample restricted to the 141 games with stored lineups (covers
  ≈2026-04-12..05-05). **This is the buy/no-buy decider.**
- `docs/balldontlie_evaluation.md` — findings doc: 4 critical questions up top,
  per-endpoint table, predictive-value backtest table, feature catalog, the
  XGBoost integration path, a 3-tier call-volume/cost comparison, and a
  BUY/NO-BUY recommendation template.
- Both scratch/ scripts gitignored, not wired into the pipeline.

**Path B model + feature scaffolding (this commit) — production path UNTOUCHED:**
- `bdl_*` SQLite tables (`db.py` schema + `migrate.py`) — cache for balldontlie
  data: pitch-type stats, H2H, splits, injuries, season WAR/QS, plus an
  `bdl_id_map` crosswalk (balldontlie uses its own ids). Empty until a future
  ingest populates them; nothing in the live pipeline reads them yet.
- `model/feature_staging.py` — ~10 candidate features, each an isolated,
  toggleable function in a `CANDIDATE_FEATURES` registry. Tagged `linear` vs
  `tree` (a `tree` feature only pays off under XGBoost — testing it in logreg
  is the 5/18 trap) and by training-eligibility. Returns neutral 0.0 on an
  empty cache so it runs today; same code yields real features post-trial.
  `lineup_platoon_edge` works NOW (uses local DB handedness, no balldontlie).
- `model/xgb_experiment.py` — STANDALONE logreg-vs-XGBoost harness. Does NOT
  touch predict.py/retrain.py. Trains both on identical data, chronological
  holdout, same metrics as the retrain gate, reports XGBoost feature importance.
  `--dry-run-synthetic` runs today (verified 2026-05-21): on the 5 production
  features XGBoost is +0.2-0.3% vs logreg = within noise. That's the honest
  baseline — with no new features XGBoost is NOT better; the Path B bet is that
  the staged `tree` features unlock it.
- These three are wired together: trial data → ingest into `bdl_*` → harness
  trains XGBoost on real staged features → buy/no-buy decided on a real model
  comparison, not assumption.

**Don't assume "more data = better product."** The 2026-05-18 experiment added
bullpen/wRC+/platoon features that were real and clean but did NOT improve
accuracy (collinear with team_quality_diff, r=0.74-0.92). The backtest exists
specifically to catch a repeat of that — a feature must beat the current model
on signal it doesn't already have, or it's not worth integrating.

**Pre-trial finding (already known, don't re-discover):** balldontlie does NOT
use MLB Stats API IDs. The DB stores MLB IDs; balldontlie has its own. Any
integration must maintain a name/abbrev → balldontlie-ID crosswalk. The probe
resolves via `/players?search=` and `/teams`.

**The decisive unknown — historical depth.** The model trains on 2022–2024. Any
balldontlie feature lacking 2022–2024 data is **inference-only** (usable for
live picks, NOT a training feature). The probe's history sweep answers this per
endpoint; the integration plan below is contingent on it.

**Candidate data sources (7), ranked by expected Path B value:**

1. **Pitch-type season stats** (`pitcher_` + `hitter_pitch_type_season_stats`) —
   highest value. Enables `arsenal_matchup_score` (starter pitch-mix × opposing
   lineup xwoba-per-pitch-type), the exact nonlinear interaction a linear model
   can't represent. Marquee new field: `xwoba`. **Training-eligibility TBD —
   gated on the depth sweep.**
2. **Batter-vs-team H2H** (`/players/versus`) — `lineup_h2h_ops_diff` +
   `h2h_sample_size`. Zero individual-matchup signal in the model today. Tree
   can gate on sample size. **Training-eligibility TBD.**
3. **Player splits** (`/players/splits`) — pitcher splits + monthly form, neither
   of which the project has (only hitter LHP/RHP splits). `starter_recent_month_era`,
   `starter_platoon_split`.
4. **Structured injuries** (`/player_injuries`) — NOT a model feature; improves
   the weakened-lineup damping stage. More reliable + earlier than the current
   lineup-history inference in `data/lineups.py:_detect_missing_regulars`.
5. **WAR / QS** (`/season_stats`) — `team_war_diff`, `starter_qs_rate`. Cleaner
   than W-L-derived `team_quality_diff`. Minor; free if already calling the endpoint.
6. **Plate appearances** (`/plate_appearances`) — Statcast-grade, but one call/game
   and the heaviest payload. NOT a live feature. Only consider as an *offline*
   historical xwoba backfill, and only if the cheaper pitch-type season endpoints
   lack 2022 depth.

**Recommended feature list when XGBoost work begins** (finalize after trial):
`arsenal_matchup_score`, `starter_arsenal_xwoba`, `starter_whiff_rate`,
`lineup_h2h_ops_diff`, `h2h_sample_size`, `starter_recent_month_era`,
`starter_platoon_split` — each tagged training-eligible vs inference-only once
the probe reports earliest-available season. Injury data wires into the damping
stack, not `FEATURE_NAMES`.

**Trial caveat (verified from balldontlie docs 2026-05-21):** the 48-hour trial
is "a 48-hour trial of the GOAT tier" but "5 req/min during the trial" — it
unlocks GOAT-tier *endpoints* at the *free* 5 req/min *rate*, NOT a paid rate.
So the trial proves data quality and predictive value; whether the All-Star
($9.99, 60/min) tier is fast enough for a daily production refresh is a
*calculation* from the probe's per-feature call counts, not something the trial
can measure directly.

**Trial-prep hardening (this commit) — the chain is now complete end to end:**
- `scratch/balldontlie_ingest.py` — field-adaptive ingest, endpoints → `bdl_*`
  tables. Trial-scoped by default to exactly the 40 games the backtest scores
  (~1,200 calls / ~4 hrs); `BALLDONTLIE_INGEST_FULL=1` for the full pool.
  Resumable, rate-limited, budget-capped, every field read with `.get()`.
- Probe + backtest hardened: output `_Tee`'d to timestamped logs (a closed
  terminal can't lose trial calls), and a 1-call auth preflight that fails fast
  on a 401 instead of discovering it on call #1,100.
- API details confirmed from balldontlie's official docs (mlb.balldontlie.io):
  auth is `Authorization: <raw key>` (no Bearer); cursor pagination; field
  names corrected (`pitching_gs`, `split_category`/`split_name`). `xwoba` EXISTS
  on pitch-type rows but is often `null` — `feature_staging.py` now applies an
  `xwoba`→`woba`→`slg` fallback so `arsenal_matchup_score` degrades, not dies.
- `migrate.py` now rebuilds stale-but-empty `bdl_*` tables (CREATE IF NOT EXISTS
  can't alter columns); idempotent.
- `docs/balldontlie_evaluation.md` has a step-by-step RUN ORDER for the 48h.

**The full trial chain (all built, ~2,300 calls, fits 48h at 5 req/min):**
`migrate.py` → `balldontlie_probe.py` → `balldontlie_backtest.py` →
`balldontlie_ingest.py` → `python -m model.xgb_experiment` (real features).

**Next step:** start the trial, follow the RUN ORDER in the eval doc, fill in
its tables, then make the BUY/NO-BUY call and (if buy) pick the cheapest tier
that clears the call volume of the features that passed.

---

## LEAN-flip — SHIPPED 2026-05-20

Discovery during the LEAN deep-dive: 471 historical 2026 LEAN picks scored at 45.9%,
*below* coin flip. Flipping them yields 54.1% on the same population. The model has
genuine anti-signal in its low-confidence (0.45-0.55 home_win_prob) zone. The narrow
0.48-0.52 band is even stronger — model picks home at 0.51-0.52 win only ~37%, model
picks away at 0.48-0.49 win 50-61%.

Effect on contest scoring (1127 scored 2026 picks):
- Status quo: 601 correct / 1127 = 53.3%
- LEAN-flip: 640 correct / 1127 = 56.8% (+3.5pp)

Contest requires a pick on every game, so "sit out" isn't an option.

Implementation: in `model/predict.py:predict_games` and `scheduler.py:run_lineup_lock`,
after the full calibration stack determines confidence, if confidence == "LEAN" then
`home_win_prob = 1 - home_win_prob`, recompute pick. Sets `picks.pick_flipped = 1`.
Original model output is recoverable by `1 - home_win_prob` for flipped rows.

**Watch for the next 2-3 weeks:** if flipped picks drop below 50% accuracy, the
inversion was an artifact and we should disable the flip. Bullpen+platoon data was
broken until 5/20 which may have inflated LEAN frequency; on cleaner data the flip
may be less reliable.

---

## Retrain regression gate data leakage — FIXED 2026-05-20

Weekly retrain had been blocked by the regression gate for 4 consecutive runs
(5/10, 5/17, 5/18 manual, 5/19) before discovery. Root cause: the "production model"
in the comparison was being evaluated on its own training data while the candidate
was held out, so production won by 1-2pp every time → gate fired.

Specifically: holdout = last 20% of current season chronologically. But production
was trained on data through its commit date, which included those games. The gate
was comparing candidate-on-real-holdout vs production-on-training-data.

Fix: use the production model file's mtime as a time-walked cutoff. Validation set
is now games on/after that cutoff — neither model trained on them. If no such games
exist (e.g. retraining right after a deploy), skip the gate entirely rather than
running an unfair comparison.

Side effect: dry-run of the retrain on local data now correctly returns "skip gate"
instead of "candidate is 1.1% worse." Cloud Sunday cron should now deploy correctly.

---

## Away-team overconfidence damping — SHIPPED 2026-05-20

Backtest of 2022-2024 revealed a systematic pattern: the model overrates away-team
HIGH picks by ~10pp when all 5 signals (record, FIP, bullpen, wRC+, form) support
the pick. Home picks at the same signal-count level are well-calibrated.

Shipped rule (`away_overconfidence_damping` in `model/features.py`): for away picks
at HIGH tier (>=63%) with 3+/5 signals supporting, shrink toward 50% by:
  - 3/5 supporting: 5%
  - 4/5 supporting: 10%
  - 5/5 supporting: 20%

Validation (2025 holdout, n=2428): HIGH-tier accuracy +0.8pp (67.4% → 68.2%),
Brier improved 0.2405 → 0.2403. About 12% of historical HIGH picks would have
been downgraded from HIGH to MEDIUM.

Applied at both morning (`model/predict.py`) and lineup_lock (`scheduler.py`).
Experiment script retained: `model/signal_damping_experiment.py`.

**Key finding it does NOT address:** the data didn't have enough away HIGH picks
with 0-2/5 signals to fit a rule for those. Today's 2026-05-19 ATL @ MIA pick
(HIGH at 1/5 signals) is exactly the sort of case my intuition flagged but the
training data couldn't validate. Option D (XGBoost) may pick this up natively.

---

## lineup_lock pipeline bugs — 3 of 4 fixed 2026-05-20

Found during a "what data are we missing?" audit. lineup_lock was supposed to be the
strongest-tier pick path (closest to game time, most recent data, lineup-aware) but
several pieces of the pipeline were silently broken.

**Bug 2 FIXED 2026-05-19:** scheduler.py:lineup_lock did NOT re-run opener detection
or apply opener dampening. Morning picks dampen 40% toward 50% when an opener is detected;
lineup_lock picks did not, so HIGH/MEDIUM tier assignments were inconsistent between
the two runs for opener games. Lineup_lock probabilities looked more confident than
they should have. Now re-detects and dampens the same way `predict_games` does.

**Bug 1 FIXED 2026-05-20 (Option B):** Replaced the dead `feats["platoon_wrc_diff"]` write
with a post-model lineup-OPS nudge in the calibration stack. Stack position is after
opener-dampen and away-overconfidence-dampen, before weakened-lineup dampening.

Nudge formula: `nudge = clip(ops_gap * 0.40, ±0.04)`, applied as `home_win_prob += nudge`
with the result clipped to [0.05, 0.95]. Dead zone of 0.5pp suppresses tiny noise nudges.
Conservative K=0.40 chosen first-principles (50 OPS pts ≈ 2pp shift in published research)
rather than backtested — only 152 scored lineup_lock picks exist in 2026, too few for
grid-search tuning. Plan: re-evaluate K in ~3 weeks once n is larger.

Why Option B over Option A (add feature + retrain): lineup OPS only exists at lineup_lock,
not morning. Option A would require either two models or a costly historical lineup
backfill via rate-limited MLB API. Option B keeps one model serving both paths.
The 5-OPS-feature collinearity (r=0.74-0.92 with team_quality) documented 2026-05-18
also suggests the linear model couldn't extract clean OPS signal anyway.

**Bug 3 STILL OPEN:** Weather data fetched only in morning via `main.py:refresh_data`,
but MLB Stats API returns `weather: {}` for games in `Preview` status (not yet started).
At 8 AM ET when morning runs, 0 of 15 games have weather. Season-to-date: 48/639
(7.5%) games have weather. Fix: also call `get_game_weather` in scheduler's lineup_lock
path, which runs 2-3 hours before first pitch when MLB has actually published wind/temp.

**Bug 4 RESOLVED 2026-05-20:** Resolved as a side effect of Bug 1's Option B fix.
The dangling `platoon_wrc_diff` write was removed entirely; no orphan-producer pattern
remains. Other producer-side code should still be audited periodically — grep for
`feats[` assignments outside of `model/features.py`.

---

## NYY bullpen ERA = 0.0 data bug — FIXED 2026-05-20

Bug 3 from the lineup_lock audit. NYY pick (vs TOR) on 5/18 was inflated by a
phantom "best bullpen in baseball" value of 0.0. Audit found SF was also stuck at 0.0,
and a second related bug: **all 30 teams had NULL `wrc_plus_vs_lhp` / `wrc_plus_vs_rhp`**
in the DB despite the live FanGraphs API returning valid platoon splits — silent
RequestException failures with no retry, masked by `except: return {}`.

Two distinct root causes:

1. **0.0 bullpen ERAs persisted across 9+ daily runs.** Live API returned plausible
   values for NYY/SF every time we tested, but historical writes had landed 0.0 and
   `COALESCE(excluded.bullpen_era, bullpen_era)` only replaces NULL, not 0.0.
2. **Platoon splits never persisted** for any team. Sequential 4x FanGraphs requests
   in `refresh_fangraphs_stats` hit intermittent failures; the per-request `except:
   return {}` meant we silently shipped empty dicts that updated nothing in the UPSERT.

Fix in `data/fangraphs.py`:
- Added `_http_get_with_retry()` with 3 attempts and exponential backoff (1s/2s/4s).
- Added plausibility bounds: `BULLPEN_ERA_MIN=0.5`, `BULLPEN_ERA_MAX=12.0`,
  `WRC_PLUS_MIN=30`, `WRC_PLUS_MAX=200`. Values outside are rejected with a warning
  log and don't get written.
- Cleanup: nulled out the 2 bad bullpen rows and reset `updated_at` to 2020-01-01
  on all 30 2026 rows to bypass the 7-day staleness gate. Re-ran refresh against
  local DB and `seed/mlb_picker.db`. All 30 teams now have plausible bullpen ERA
  and populated platoon splits.

Verification: NYY bullpen 0.0 → 3.54; SF bullpen 0.0 → 3.54; platoon splits
0/30 populated → 30/30 populated.

**Followup:** Recompute signal tags + away-overconfidence damping for picks made
between 5/9 and 5/20 — those used a broken bullpen signal for NYY/SF games and
flat NULL platoon for all games. We can't retroactively change posted picks but
it's worth quantifying how many HIGH-tier picks were misclassified.

---

## ~~Expand model feature set~~ — RESOLVED 2026-05-18

**Conclusion:** Adding `bullpen_diff`, `wrc_plus_diff`, `platoon_wrc_diff` to the regression
does NOT improve accuracy. They're heavily collinear with `team_quality_diff` (r=0.74-0.85)
and with each other (wrc↔platoon r=0.92). The W-L record is effectively a clean aggregator
of the underlying offense/pitching/bullpen signals, so adding them as separate inputs just
creates feature-fighting in the regression. The signal-tag system on the dashboard still
has value because it evaluates these features independently — that's something regression
coefficients can't do.

Experiment scripts retained in repo: `model/feature_experiment.py`, `model/fip_diagnostic.py`,
`model/fip_fallback_audit.py`. Re-run if revisiting.

**What we DID ship from this investigation:**

1. **Dropped `home_flag`** — confirmed no-op (constant 1.0 → coef always 0.0).
   Feature count: 6 → 5.

2. **Fixed lookahead bias in `_get_bullpen_era` / `_get_wrc_plus` / `_get_platoon_wrc`** —
   they were doing `ORDER BY season DESC LIMIT 1` regardless of game date, so historical
   training would use 2025 splits to predict 2022 games. Now season-aware (helpers are
   currently unused by the model but matter if anyone wires them in later).

3. **Backfilled `pitcher_stats` for 2022-2024** — this was the marquee finding. The table
   was missing 2023 (0 rows), 2024 (0 rows), and had gaps in 2022. **~26% of training games
   were silently falling through to `fip=4.00` for both starters**, suppressing FIP's signal.
   After backfilling 966 pitcher-seasons via the MLB Stats API:
   - FIP coverage: 74% real → 99.6% real
   - FIP coefficient: -0.05 → -0.12 (more than 2x)
   - Bucketed analysis shows a 12.7pp win-rate spread across FIP-edge quintiles (was 6.7)
   - Holdout (2026 last 95 games): 63.2% overall, 80% HIGH (n=10), 66.7% MED (n=42)

   Seed DB updated with backfilled data. New script: `data/backfill_pitcher_stats.py`.

4. **Added schema-mismatch handling to `retrain.py`** — previously a FEATURE_NAMES change
   would crash the regression gate with a ValueError. Now it logs and skips gracefully.

**Followup needed:** investigate why `data/historical.py` failed to populate pitcher_stats
for 2023-2024 originally. Probably the early-return guard at line 37 ("data already loaded")
fired before pitcher fetches could complete. Same hole could open again next year if not fixed.

---

## Expand signal tags to Pick History tab (MEDIUM priority)

Currently signal tags only render on today's pick cards. The history tab doesn't have them
because the `all_picks` SQL query in `output/dashboard.py` doesn't pull the joined FIP /
bullpen / wRC+ data (would slow page load).

**Options:**
- Expand the query (slower load but full retroactive validation)
- Pre-compute tags during scoring run and store them in the picks table
- Skip it — accept that history shows ✓/✗ only

If we ship the model feature expansion above, history-tag visibility becomes more valuable
because users will want to see whether the new features actually shifted picks correctly.

---

## Path C (in-season retrain) — TESTED 2026-05-20, no measurable lift on current data

Discovery: the weekly retrain was ALREADY using `2022-{prior_year} + {current_year}-to-date`
as the training set since at least 5/18 (model deployed that day was trained on
10,188 games including ~95 of 2026). I'd misread it as "static 2022-2024."

Tested whether the Bug 4 FIP-constant fix changes the model:
- Recomputed all 1,359 historical pitcher FIPs using per-season constants
  (range 3.10-3.25). Mean shift +0.066, max +0.16. All FIPs in a single season
  shift by the same constant, so `fip_diff` (the actual model input) is unchanged
  within-season; cross-season comparisons get a small correction.
- Dry-run retrain: 63.2% accuracy on 95-game holdout, FIP coefficient went from
  -0.117 → -0.117 (no change). Candidate was -1.1% vs production but within noise.
- StandardScaler absorbs the constant shift, so the FIP fix is invisible to the
  trained model unless we also have cross-season FIP comparisons (which we don't —
  every prediction uses both pitchers' current-season FIPs).

Conclusion: data is now clean, but **the linear model has hit its ceiling on this
feature set**. May 2026 in-season HIGH accuracy (62%) is below 2025-validated HIGH
accuracy (68%); LEAN accuracy is 45.9% (below coin flip).

**Next decision:** Path B (XGBoost) or accept the structural ceiling.
- Path B may pick up nonlinearity the linear model can't (e.g. "HIGH with 1/5 signals"
  case from TODO 2026-05-20).
- Pre-Path-B: the data fixes from 5/20 (NYY/SF bullpen, platoon splits) haven't
  affected ANY scored game yet — they only impact the post-model damping stack and
  signal tags. Wait 2-3 weeks of post-fix games before judging the current stack.

## Decide whether to switch weekly retrain from PR-based to direct-commit (LOW)

After 4 weeks of stable PRs (target: ~2026-06-02), evaluate:
- Did the regression gate ever incorrectly flag a healthy retrain? (false positive)
- Did the gate ever miss a bad retrain? (false negative — would need backfill check)
- Are the proposed retrains stable enough to skip review?

If clean: switch to direct-to-main commits in `weekly-retrain.yml` (remove the PR creation
step, just push to main). Keeps automation honest without ceremony.

---

## Investigate the "FIP signal degrades with more starts" finding (LOW)

From historical audit: SP with 0-1 prior starts had 57.8% pick hit rate, while SP with 4+
starts had 49.2%. That's backwards from intuition. Theories:
- Are we using current-season FIP too aggressively in May? Maybe blend with prior season longer.
- ~~Is the FIP constant (3.10 hardcoded in `data/fip.py`) drifting from actual league FIP?~~
  Ruled out 2026-05-20: constant now per-season in `fip_constants` table, range across
  2022-2026 is only 3.10-3.25 (~0.05 stddev) and cancels out in `fip_diff` anyway. Whatever
  is causing the degradation is something else — probably the current-season blend in May.

Worth investigating once we have ~50% of the season's data (~July).

---

## GitHub Actions Node.js 20 deprecation (LOW)

Both `daily-picks.yml` and `weekly-retrain.yml` use `actions/checkout@v4`,
`actions/setup-python@v5`, etc. — all on Node.js 20. Forced removal Sept 16, 2026.

**Action:** Update workflows to actions versions that support Node 24 before September.
Probably just bumping `@v4` → `@v5` on a few lines.

---

## CLAUDE.md is out of date (LOW)

Current FEATURE_NAMES (post-2026-05-18): `fip_diff`, `team_quality_diff`, `park_factor`,
`home_offense_trend`, `away_offense_trend` (5 features).

CLAUDE.md should be updated to match — both the features list and any references to
bullpen/wRC+ as model inputs (they're dashboard signal-tag inputs only).
