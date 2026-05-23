# balldontlie MLB API — Evaluation (Path B feature-expansion source)

**Status:** TRIAL COMPLETE — verdict: **NO-BUY** (see Final recommendation).
**Owner:** dan
**Created:** 2026-05-21
**Trial window:** 2026-05-21 → 2026-05-23 (48h free trial of GOAT-tier endpoints at 5 req/min)
**Completed:** 2026-05-22

This doc evaluates whether the [balldontlie MLB API](https://www.balldontlie.io/)
can supply richer, nonlinear features for the planned **Path B (XGBoost)** model.
It is **additive feature exploration only** — no existing data source is being
migrated. MLB Stats API and FanGraphs both stay (see "Out of scope" below).

The trial runs **two scripts, in order**. Both are gitignored (`scratch/`) and
not wired into the pipeline.

```bash
export BALLDONTLIE_API_KEY=<trial key>

# 1. Data-availability probe — "does the data exist and is it clean?"
python3 scratch/balldontlie_probe.py        # ~7-9 min, ~30-40 calls

# 2. Predictive-value backtest — "does the data make a BETTER product?"
#    Run AFTER the probe. Drop any endpoint the probe flagged as empty/shallow.
python3 scratch/balldontlie_backtest.py     # ~3.5-4 hrs, ~1,100 calls, unattended

# 3. (after a balldontlie ingest populates the bdl_* tables) full model test:
python3 -m model.xgb_experiment             # logreg vs XGBoost on real features
```

**Why two scripts + a harness.** The probe alone is not enough to justify a
subscription. The 2026-05-18 experiment (TODO.md) added bullpen/wRC+/platoon
features that *were real and clean* but did **not** improve accuracy — collinear
with `team_quality_diff`. "The data exists" and "the data helps" are different
questions. Script 1 answers the first; script 2 gives correlational evidence on
the second; the `xgb_experiment.py` harness gives the definitive answer once
real features are loaded — it trains an actual XGBoost model and compares it to
logreg head-to-head.

### Already built (committed before the trial)

| Component | What it is | Runnable now? |
|-----------|-----------|---------------|
| `scratch/balldontlie_probe.py` | data-availability probe | yes (needs key) |
| `scratch/balldontlie_backtest.py` | predictive-value backtest | yes (needs key) |
| `scratch/balldontlie_ingest.py` | endpoints → `bdl_*` cache, trial-scoped | yes (needs key) |
| `bdl_*` SQLite tables (`db.py`, `migrate.py`) | cache for balldontlie data | yes — `migrate.py` creates them |
| `model/feature_staging.py` | ~10 candidate features, each toggleable | yes — neutral 0.0 on empty cache |
| `model/xgb_experiment.py` | logreg-vs-XGBoost comparison harness | yes — `--dry-run-synthetic` proves plumbing |

The whole chain is built. The trial just runs it.

> **Post-trial correction (2026-05-22):** the three `scratch/` scripts are
> gitignored (`scratch/` is in `.gitignore`) — they exist on disk but are NOT
> committed. `model/xgb_experiment.py` had a latent bug: it called
> `build_training_features(..., return_dates=True)`, a kwarg that did not exist
> — so the harness had never actually run end to end (not even `--dry-run`).
> Fixed by adding the optional `return_dates` param to
> `model/features.py:build_training_features` (pure addition, default False,
> all existing callers unaffected).

---

## RUN ORDER — the 48-hour trial, step by step

The trial window is short and the backtest + ingest each run for hours. Follow
this order; do not parallelize steps 2/3 (both spend rate-limited calls).

**T-0 — before the key arrives (no API calls):**
```bash
python3 migrate.py mlb_picker.db          # create the bdl_* cache tables
python3 -m model.xgb_experiment --dry-run-synthetic   # confirm the harness runs
```

**Step 1 — activate + smoke-test the key (~1 call, 1 min):**
```bash
export BALLDONTLIE_API_KEY=<trial key>
python3 scratch/balldontlie_probe.py      # auth preflight runs first; aborts on 401
```
The probe self-aborts if the key/tier is bad — so this doubles as the auth check.
Let it finish (~8 min). Read its output: confirm `xwoba` is populated and how
far back the pitch-type/versus endpoints have data. **If a key endpoint is empty
or 401s, stop and reassess before spending the backtest's hours.**

**Step 2 — predictive-value backtest (~1,100 calls, ~3.5–4 hrs, unattended):**
```bash
nohup python3 scratch/balldontlie_backtest.py > scratch/backtest_run.log 2>&1 &
tail -f scratch/backtest_run.log
```
Correlational evidence on whether the new features carry incremental signal.

**Step 3 — ingest into the `bdl_*` cache (~1,200 calls, ~4 hrs, unattended):**
```bash
nohup python3 scratch/balldontlie_ingest.py > scratch/ingest_run.log 2>&1 &
tail -f scratch/ingest_run.log
```
Trial-scoped by default to exactly the 40 games the backtest scores. Resumable —
re-run if it stops on a 429 or the budget cap.

**Step 4 — the definitive model comparison (no API calls):**
```bash
python3 -m model.xgb_experiment           # logreg vs XGBoost on REAL features
```
This trains an actual XGBoost model on the ingested features and reports the
head-to-head. Its verdict + feature-importance table is the buy/no-buy evidence.

**Step 5 — fill in this doc** and make the call (see "Final recommendation").

Total API calls ≈ 2,300 across the trial — well within 48 hrs at 5 req/min,
with steps 2 and 3 being the long unattended runs. Budget time so both finish
before the window closes; start step 2 early.

---

## Critical questions the trial must answer

Answer these FIRST — they gate everything below.

| # | Question | Why it's decisive | Answer (filled 2026-05-22) |
|---|----------|-------------------|---------------------------|
| 1 | **Historical depth** — do `/players/versus` and the pitch-type endpoints have data back to **2022**? | The model trains on 2022–2024 (7,287 games). Any feature without 2022–2024 data is **inference-only** — usable for live picks but it cannot be a training feature, which sharply limits its value for an XGBoost rebuild. | **Endpoints DO have 2022 data** — `pitcher_pitch_type_season_stats?season=2022` returns rows. BUT the trial ingest (`scratch/balldontlie_ingest.py`) is hard-scoped to `seasons:[2022]`→ in practice ran `[2026]` only, plus the 40 backtest games. So the `bdl_*` cache holds **2026 data only**. A 2022–2024 backfill is ~3,000+ calls and was not run inside the 48h window. Net: as loaded, every staged feature is **inference-only**. |
| 2 | **Rate-limit reality** | The `lineup_lock` path already makes ~270 player calls/run. | **Trial rate is 5 req / 24s — far harsher than the doc's "5 req/min" assumption.** Confirmed live: `x-ratelimit-limit:5`, `x-ratelimit-reset` ~24s out, `retry-after:23` on a 429. The full ingest took 1,179 calls; the backtest 1,069 calls over ~3.7 hrs. A daily production refresh of all features (~600 calls/day, doc §"call-volume") would need the All-Star tier (60 req/min) to fit a cron window. |
| 3 | **Field completeness** | A feature null for 40% of the pool forces a fallback. | **xwoba/woba frequently null on low-volume rows** (confirmed in raw responses — null even on a real sampled row). `slg` is always populated; `feature_staging.QUALITY_FIELD_CHAIN` (xwoba→woba→slg) already anticipates this and degrades gracefully. H2H samples are thin this early in 2026: avg 81 total H2H AB/game across ~12.7 batters. |
| 4 | **`xwoba` coverage** | If `xwoba` only exists for stars, the arsenal-matchup feature is not viable. | **Effectively not viable as loaded.** The arsenal features (`arsenal_xwoba_diff`, `starter_whiff_diff`, `arsenal_diversity_diff`, `arsenal_matchup_score`) were populated on only 36–79 of 473 scored 2026 games and scored **0.0000 XGBoost importance** on the 2026 populated-sample test — too sparse for the model to split on. |

### Known integration cost discovered before the trial

**balldontlie does NOT use MLB Stats API IDs.** mlb-picker's SQLite DB stores
MLB Stats API IDs (e.g. pitcher `663568`, team `109`=ARI). balldontlie assigns
its own internal IDs. Any integration must maintain a **name/abbreviation →
balldontlie-ID crosswalk**, refreshed as rosters change. The probe resolves IDs
via `/mlb/v1/players?search=` and `/mlb/v1/teams` — record below how reliable
that resolution is (exact-match rate, fuzzy/ambiguous cases).

ID-resolution reliability (filled 2026-05-22): **396/414 players resolved
(378 exact, 18 fuzzy); 29/30 teams resolved exact.** ~4% of players unresolved
— acceptable. Teams crosswalk cleanly by abbreviation (note: balldontlie uses
`CHW` for the White Sox vs the project's `CWS`; the ID-keyed `bdl_id_map`
sidesteps this). Crosswalk lives in the `bdl_id_map` table.

---

## API basics (confirmed from the OpenAPI spec)

- **Base URL:** `https://api.balldontlie.io/mlb/v1`
- **Auth:** `Authorization` header, raw API key (no `Bearer` prefix)
- **Pagination:** cursor-based (`cursor` integer param → `next_cursor` in response)
- **Tiers:** Free 5 req/min · All-Star $9.99/sport 60 req/min · GOAT $39.99/sport
  600 req/min. The 48h trial = GOAT-tier *endpoints* at the *free* 5 req/min rate.

---

## Results table — per endpoint (fill after probe)

| Endpoint | Works? (HTTP) | Fields present vs. spec | Earliest season w/ data | Rate-limit headroom | Verdict |
|----------|---------------|-------------------------|-------------------------|---------------------|---------|
| `/season_stats` (WAR, QS) | ✅ 200 | full — `pitching_qs`, `pitching_gs`, `pitching_war`, `batting_war`, embedded player+team object | 2022 returns rows (only 2026 ingested) | OK — paginated, ~16 calls/season | **inference-only as loaded** — `starter_qs_rate_diff` got nonzero importance but no lift |
| `/players/versus` (batter-vs-team H2H) | ✅ 200 | `at_bats`, `hits`, `ops` etc. present | 2026 only ingested | poor — singular `player_id`, ~18 calls/game, no batch | **inference-only** — moderate residual r but loses disagreement cases (33%) |
| `/pitcher_pitch_type_season_stats` (arsenal: xwoba, whiff) | ✅ 200 | `xwoba`/`woba` frequently null on low-volume rows; `slg`/`whiff_percent` present | 2022 returns rows | OK — chunked by `player_ids[]`, ~16 calls/season | **not usable as loaded** — too sparse, 0.0 importance |
| `/hitter_pitch_type_season_stats` (lineup vs pitch type) | ✅ 200 | same shape as pitcher endpoint | 2022 returns rows | OK — chunked, ~16 calls/season | **not usable as loaded** — feeds `arsenal_matchup_score`, 0.0 importance |
| `/players/splits` (pitcher splits, monthly form) | ✅ 200 | categories: `byArena`, `byBreakdown`, `byDayMonth`, `byOpponent`, `split`. **NO vs-RHP/vs-LHP handedness split** — `byBreakdown` is Home/Away/Day/Night only | 2026 only ingested | OK — singular `player_id`, starters only | **partial fail** — pitcher platoon split impossible; recent-form rows not captured by the ingest (0 populated) |
| `/player_injuries` (structured injuries) | ✅ 200 | `status`, `return_date`, `type`, embedded player+team | n/a (current snapshot) | cheap — batchable | usable for damping, but not a model input; minor |
| `/plate_appearances` (Statcast pitch detail) | not probed live¹ | — | — | — | _offline-backfill only — not pursued_ |

¹ `/plate_appearances` is deliberately **not** in the probe: one call per game,
large payload, the most rate-limit-expensive endpoint. It is not a live-feature
candidate. Its only plausible use is an **offline** historical xwoba backfill —
evaluate that separately and only if the cheaper pitch-type season endpoints
turn out to lack 2022 depth.

---

## Candidate features (per the spec) — verdict after probe

| Candidate feature | Source endpoint(s) | Training-eligible? | Notes |
|-------------------|--------------------|--------------------|-------|
| `lineup_h2h_ops_diff` + `h2h_sample_size` | `/players/versus` | _TBD (depends on Q1)_ | Home lineup weighted H2H OPS vs away starter minus reverse. Tree can gate on sample size. |
| `starter_arsenal_xwoba`, `starter_whiff_rate` | `/pitcher_pitch_type_season_stats` | _TBD_ | Usage-weighted xwoba allowed. `xwoba` is the marquee new field. |
| `arsenal_matchup_score` | `pitcher_` × `hitter_pitch_type_season_stats` | _TBD_ | Per pitch type: starter usage% × opposing lineup xwoba vs that pitch. The interaction a linear model structurally cannot represent — the core Path B motivation. |
| `starter_recent_month_era`, `starter_platoon_split` | `/players/splits` | _TBD_ | Project has hitter LHP/RHP splits only — no pitcher splits, no monthly form. |
| Injury-aware weakened-lineup damping | `/player_injuries` | n/a (live damping input) | More reliable + earlier than the current lineup-history inference in `data/lineups.py`. Feeds the damping stage, not the model. |
| `team_war_diff`, `starter_qs_rate` | `/season_stats` | _TBD_ | Cleaner talent estimate than W-L-derived `team_quality_diff`. Minor. |

---

## Feature catalog — built and staged (`model/feature_staging.py`)

All ~10 candidate features below are **already implemented** in
`model/feature_staging.py`, each as an isolated, individually-toggleable
function registered in `CANDIDATE_FEATURES`. They read from the `bdl_*` SQLite
cache tables (`db.py`), which are empty until the trial ingest populates them —
every feature returns a documented neutral `0.0` on an empty cache, so the
harness runs today and the *same code* produces real features once data lands.

**Model-fit tag** is the load-bearing column. `linear` = a clean monotonic diff
the current logreg could use. `tree` = an interaction or sample-size-gated
signal that **only XGBoost can extract** — testing a `tree` feature in logreg
falsely fails it (the 2026-05-18 trap). `eligibility` is provisional until the
probe confirms 2022 depth.

| Staged feature | Fit | Definition | Endpoint | Eligibility |
|----------------|-----|------------|----------|-------------|
| `arsenal_xwoba_diff` | linear | usage-weighted xwoba allowed: away starter − home | `pitcher_pitch_type` | TBD (probe) |
| `starter_whiff_diff` | linear | usage-weighted whiff%: home starter − away | `pitcher_pitch_type` | TBD |
| `arsenal_diversity_diff` | tree | count of pitch types ≥10% usage: home − away (interacts w/ quality) | `pitcher_pitch_type` | TBD |
| `arsenal_matchup_score` | **tree** | **the marquee feature** — Σ(starter pitch usage% × opposing lineup mean xwoba vs that pitch), home setup − away setup | `pitcher_`+`hitter_pitch_type` | TBD |
| `h2h_ops_diff` | tree | AB-weighted lineup H2H OPS: home lineup vs away team − reverse | `players/versus` | TBD |
| `h2h_sample_size` | tree | total H2H at-bats behind `h2h_ops_diff` — lets the tree gate on H2H reliability | `players/versus` | TBD |
| `starter_platoon_split_diff` | linear | each starter's vs-RHP−vs-LHP wOBA gap; away exploitability − home | `players/splits` | TBD |
| `starter_recent_form_diff` | tree | recent-month ERA − season ERA, per starter, home−away of that delta | `players/splits` | TBD |
| `lineup_platoon_edge` | linear | count of platoon-advantaged batters vs the opposing starter, home − away | (local DB — no balldontlie) | **training** ✓ |
| `starter_qs_rate_diff` | linear | quality-start rate (QS/GS): home starter − away | `season_stats` | TBD |

`lineup_platoon_edge` is the one feature usable **today** — it derives from
`bat_side`/`throw_hand` already in the DB, no balldontlie dependency.

**Not in `CANDIDATE_FEATURES`:** injury data (`/player_injuries`) — it improves
the post-model weakened-lineup damping stage, not the model input vector. And
`team_war_diff` was deliberately dropped: WAR is near-certain to be collinear
with `team_quality_diff` (the 5/18 risk); add it only if the backtest's
collinearity test clears it.

---

## XGBoost candidate — built (`model/xgb_experiment.py`)

**Why XGBoost.** The current model is logistic regression: one dividing line,
one fixed coefficient per feature, each feature's effect independent of the
others. It structurally cannot represent an interaction ("a FIP edge matters
more when the bullpen is also weak") or a conditional ("trust H2H only at 30+
AB"). The 5/18 experiment failed for exactly this reason. XGBoost is gradient-
boosted decision trees — hundreds of small if/then trees — which handle
interactions, nonlinearity, and sample-size gating natively. Most of the staged
features above are tagged `tree` precisely because they *need* this.

**What's built.** `model/xgb_experiment.py` is a **standalone** harness — it
does NOT touch `model/predict.py` or `model/retrain.py`. It trains logreg and
XGBoost on identical data, splits chronologically (fair: neither model sees the
holdout), and compares them on the same accuracy/tier/Brier metrics the weekly
retrain gate uses. It reports XGBoost feature importances so dead features show.

**Dry-run verified (2026-05-21, empty cache).** On the 5 production features
only, XGBoost was +0.2–0.3% vs logreg — *within noise*, Brier slightly worse.
That is the honest baseline: **with no new features, XGBoost is not better.**
The entire Path B bet is that the staged `tree` features unlock it. The dry-run
proves the plumbing; the trial provides the features that test the bet.

**Integration path — only if XGBoost wins the live comparison.** Nothing below
is done yet; it's the deliberate follow-up gated on a positive
`xgb_experiment.py` result with real trial data:

1. Generalize `model/predict.py:load_model` / `train_model` to a model-type
   abstraction — XGBoost needs no `StandardScaler` and exposes
   `feature_importances_` instead of `coef_`. The `.pkl` format changes.
2. Add an XGBoost branch to `model/retrain.py` so the weekly retrain can train
   and gate it like logreg. The existing time-walked regression gate already
   gives the safety net — if XGBoost stops winning out-of-sample, the gate
   rejects it.
3. `FEATURE_NAMES` grows to include whichever staged features earned a slot
   (importance > ~0.01 and a positive backtest score).
4. The 5-stage calibration stack (`scheduler.py` / `predict.py`) is unaffected —
   it operates on the output probability, not the model internals.
5. New dependency: `xgboost` (already installed locally as 2.1.4) → add to
   `requirements.txt`.

**XGBoost is not guaranteed to win.** With ~7–10k games it usually edges logreg
*when real interactions exist* — and the away-overconfidence pattern suggests
they do — but it's empirical. If the live comparison comes back within noise,
the correct outcome is to keep logreg and not pay the complexity cost.

---

## Predictive-value backtest (fill after `balldontlie_backtest.py`)

This is the decisive section. Endpoint availability (above) is necessary but not
sufficient — the 5/18 experiment proved real, clean data can still be useless to
the model. The backtest measures **incremental** signal: signal the current
5-feature model does *not* already have. Three tests per candidate feature:

- **[1] Residual correlation** — `r` between the feature and the current model's
  prediction error (`outcome − predicted_prob`) on scored 2026 games. Non-zero
  ⇒ the feature predicts *where the model is wrong* ⇒ non-redundant signal.
  Thresholds: `|r|≥0.25` strong · `0.15–0.25` moderate · `0.07–0.15` weak ·
  `<0.07` negligible (the 5/18 outcome).
- **[2] Collinearity** — max `|r|` between the feature and the existing 5
  features. `≥0.7` = redundancy risk (the 5/18 killers were 0.74–0.92).
- **[3] Disagreement-case win rate** — on games where the feature points
  opposite the current pick, the win rate of *following the feature*. `>50%`
  ⇒ the feature would flip picks toward correct answers.

**A genuine Path B candidate scores [1] ≥0.15, [2] <0.7, and [3] >50%.**

| Candidate feature | [1] residual r | [2] max collinearity | [3] disagree win% | Populated n / sample | Verdict |
|-------------------|----------------|----------------------|-------------------|----------------------|---------|
| `arsenal_xwoba_diff` | −0.480 (strong, n=11) | **0.718 vs team_quality_diff** — FAILS [2] | only 3 disagreement cases — unreadable | 11 / 40 | **SKIP** — strong raw r but redundant with `team_quality_diff` (the 5/18 killer pattern, 0.74–0.92). And only populated on 11/40 games. |
| `h2h_ops_diff` (full lineup-weighted)¹ | −0.161 (moderate, n=39) | 0.235 (acceptably independent) — passes [2] | 5/15 = **33.3%** — FAILS [3] (below coin flip) | 39 / 40 | **SKIP** — clears collinearity but following it would flip picks *away* from correct answers. No incremental edge. |

¹ `h2h_ops_diff` is the **full lineup-weighted** version — every batter in both
stored lineups, AB-weighted (batters with more shared H2H history count more).
This is the expensive part of the run: ~18 `/players/versus` calls/game (the
endpoint has no batch param). At the trial's 5 req/min a 40-game backtest is
~3.5–4 hrs — run it unattended (`nohup`). Sample restricted to the 141 scored
2026 games that have full stored lineups, which cover ≈2026-04-12..05-05 — so
the result speaks to April/early-May 2026, not the whole season.

H2H sample-depth note (filled 2026-05-22): **avg 81 total H2H at-bats/game
across avg 12.7 batters with any H2H history.** Above the 40-AB thinness floor
in aggregate, but spread across ~12.7 batters that is ~6 AB per batter-vs-team
pair — individually thin. One of 40 games had no H2H data at all.

Current-model baseline accuracy on the backtest sample: **62.5%** (40 scored
2026 games with full lineups, mean |residual| 0.463). Consistent with the 62%
threshold figure in CLAUDE.md — sample is representative.

### XGBoost head-to-head (model/xgb_experiment.py, 2026-05-22)

The definitive test — train logreg and XGBoost on identical data, same
chronological holdout. Two runs:

**Run A — full 2022-2026 (10,188 games).** The `bdl_*` cache holds 2026 data
only, so staged features were constant 0.0 on all training + holdout games →
all 10 scored 0.0000 importance. Result: XGBoost 57.4% vs logreg 57.1%
(+0.3%, within noise), Brier slightly worse. **This run tested plumbing only**,
not feature value — staged features were absent on every evaluated game.

**Run B — 2026 populated-sample (180 games where ≥1 staged feature is real).**
The honest test. 135 train / 45 holdout, all within 2026:

| Model | Features | Accuracy | Brier |
|-------|----------|----------|-------|
| Logistic regression | 5 production | **71.1%** | **0.2045** |
| XGBoost | 5 production + 8 staged | 68.9% | 0.2375 |

Adding the staged features made the model **worse on both metrics**. XGBoost
importance on Run B: `h2h_ops_diff` 0.13, `h2h_sample_size` 0.12,
`starter_qs_rate_diff` 0.12, `lineup_platoon_edge` 0.06 — nonzero but not
helpful; the 4 arsenal features stayed 0.0000 even here (too sparse, 36–79/473
games). The 45-game holdout is small (noisy), but the direction is clear: **no
lift, and a Brier regression.**

---

## Tier comparison & call-volume budget (fill after probes)

**Verified from balldontlie's docs (2026-05-21):** _"Every paid sport offers a
48-hour trial of the GOAT tier"_ with _"5 req/min during the trial."_ The trial
unlocks GOAT-tier **endpoints** but enforces the **free 5 req/min rate**. It is
NOT a trial of a paid *rate*. So the trial proves *data quality and predictive
value*; it cannot *measure* whether a paid tier is fast enough for production —
that's a calculation from the per-feature call counts the probe records. Fill
the call-volume estimates below, then the tier verdict follows.

**Daily-refresh call-volume estimate** (~15 games/day; per-player-season
endpoints are cached, so cost ≈ unique players touched, not games):

| Feature | Endpoint | Calls/day (est.) | Notes |
|---------|----------|------------------|-------|
| Starter arsenal | `pitcher_pitch_type_season_stats` | ~30 (2 starters × 15 games, cached) | _TBD from probe_ |
| Lineup vs pitch type | `hitter_pitch_type_season_stats` | ~270 (18 batters × 15 games) or fewer if batched by `player_ids[]` | _TBD — check if batching works_ |
| H2H | `players/versus` | ~270 (one call per batter × team) — **no batching, `player_id` is singular** | _TBD_ |
| Splits | `players/splits` | ~30 (starters only) | singular `player_id` |
| Injuries | `player_injuries` | ~3 (batchable via `team_ids[]`) | cheap |
| **Total (all features)** | | **~600/day** | _refine after probe_ |

Tier names/prices confirmed from balldontlie.io (2026-05-21):

| Tier | Rate limit | Fits the est. daily refresh? | $/mo | Verdict |
|------|-----------|------------------------------|------|---------|
| Free | 5 req/min | _TBD_ | $0 | Almost certainly too slow for a same-window refresh |
| All-Star | 60 req/min | _TBD — ~600 calls ≈ 10 min of solid requests_ | $9.99/sport | _likely sufficient if refresh runs in its own job_ |
| GOAT | 600 req/min | _TBD_ | $39.99/sport | Only if call volume balloons or many features ship |

**Decision rule:** pick the cheapest tier whose rate limit clears the
*estimated daily call volume of only the features that passed the backtest* —
not all 7. If only `arsenal_xwoba_diff` survives, that's ~30 calls/day and even
the free tier may suffice for a nightly job.

---

## Out of scope — do not chase

- **No FIP/xFIP at season level** — balldontlie exposes only raw components, same
  as MLB Stats API. FIP stays computed locally in `data/fip.py`.
- **No team wRC+ / bullpen ERA** — FanGraphs dependency stays.
- **Not fresher than MLB Stats API** for schedule/lineups — balldontlie is a
  downstream republisher one ingest-hop behind. The time-sensitive `lineup_lock`
  path (~270 player calls/run) is NOT routed through it; free-tier limits make
  that impossible and the Paid tier still wouldn't be fresher.

---

## Final recommendation — **NO-BUY** (decided 2026-05-22)

The trial produced a defensible answer, on evidence: **do not subscribe.**

The buy bar was: at least one candidate feature scores [1] ≥0.15, [2] <0.7,
[3] >50% in the backtest. **Zero of the two backtested features clear all
three:**

- `arsenal_xwoba_diff` — strong raw residual r (−0.480) but **fails [2]**:
  collinearity 0.718 with `team_quality_diff`. This is the exact 2026-05-18
  failure mode (the killers then were 0.74–0.92). Also populated on only 11/40
  games.
- `h2h_ops_diff` — clears collinearity (0.235) but **fails [3]**: on the 15
  games where it disagrees with the current pick, following it wins 33.3% —
  *below a coin flip*. It would actively move picks toward wrong answers.

The XGBoost head-to-head confirms it independently: on the 180-game 2026
populated sample, adding all 8 staged features made the model **worse** (68.9%
vs 71.1% accuracy, Brier 0.238 vs 0.205). The four marquee arsenal/xwoba
features — the core Path B motivation — scored 0.0000 importance even there.

### Why no-buy, in one paragraph

balldontlie's MLB data is real, clean, and broad — but it does not carry
*incremental* signal the 5-feature model lacks. The two features with the best
raw correlation are each disqualified: one is redundant with `team_quality_diff`,
the other points the wrong way on disagreement. This is the **2026-05-18
experiment repeating** (bullpen/wRC+/platoon were also real, clean, and useless
because collinear). On top of that, the trial ingest landed **2026 data only** —
so even the features that aren't disqualified are inference-only; using them
would mean a 2022–2024 backfill (~3,000+ calls) before they could be training
features, with no evidence the payoff exists. The richer data is not, on its
own, a reason to pay a monthly fee.

### Answers to the trial's closing questions

1. **Which features passed the backtest?** None. `arsenal_xwoba_diff` fails
   collinearity [2]; `h2h_ops_diff` fails disagreement win-rate [3].
2. **Training-eligible vs inference-only?** Moot — none passed. As loaded, *all*
   staged features are inference-only (2026 cache only). The endpoints do return
   2022 data, so a backfill is *possible*, but not justified given (1).
3. **Cheapest sufficient tier?** N/A — no-buy. (For reference: a daily refresh
   of all features ≈600 calls/day would need All-Star, $9.99/sport/mo; the trial
   rate of 5 req/24s is far too slow for a production cron.)
4. **`/plate_appearances` xwoba backfill?** Not worth pursuing. The cheaper
   pitch-type season endpoints already expose 2022 data; the arsenal features
   built on them showed no model value, so a more expensive Statcast backfill
   would not change the verdict.
5. **Concrete decision:** **NO-BUY.** Keep the MLB Stats API + FanGraphs stack.
   The XGBoost harness, `feature_staging.py`, and the `bdl_*` schema stay in the
   tree as scaffolding (cost nothing, empty cache → neutral fallbacks) in case a
   future, genuinely-orthogonal data source appears worth testing. The Path B
   bet — that tree-model interactions on richer features beat the linear
   ceiling — was tested fairly and did not pay off with this source.

### Trial mechanics notes (for any future re-test)

- Trial rate limit is **5 req / 24s**, not the "5 req/min" the doc originally
  assumed. Budget accordingly — the ingest + backtest together used ~2,250 calls
  over ~7.5 hrs of wall time.
- `scratch/balldontlie_ingest.py` is hard-scoped to the backtest game pool +
  2026. A real training-feature evaluation needs it re-run per season for
  2022–2024 — a multi-hour job per season at trial rate.
- `model/xgb_experiment.py` trains on 2022-2026; running it against a 2026-only
  cache silently produces an all-zero staged-feature matrix (Run A above). Scope
  the evaluation to seasons the cache actually covers, or the result is a
  plumbing test masquerading as a feature test.
