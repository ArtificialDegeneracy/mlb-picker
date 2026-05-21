# balldontlie MLB API — Evaluation (Path B feature-expansion source)

**Status:** PROBE NOT YET RUN — awaiting the 48-hour free trial.
**Owner:** dan
**Created:** 2026-05-21
**Trial window:** _<fill in: trial start → start+48h>_

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

| # | Question | Why it's decisive | Answer (fill after probe) |
|---|----------|-------------------|---------------------------|
| 1 | **Historical depth** — do `/players/versus` and the pitch-type endpoints have data back to **2022**? | The model trains on 2022–2024 (7,287 games). Any feature without 2022–2024 data is **inference-only** — usable for live picks but it cannot be a training feature, which sharply limits its value for an XGBoost rebuild. | _TBD_ |
| 2 | **Rate-limit reality** — at 60 req/min (Paid $9.99 tier), can a daily refresh add these features without blowing the GitHub Actions cron window? | The `lineup_lock` path already makes ~270 player calls/run. New per-player endpoints multiply that. If a feature needs a call per starter + per lineup batter, that's 2 starters + ~18 batters = ~20 calls/game × 15 games = **~300 calls/day** just for one feature. | _TBD_ |
| 3 | **Field completeness** — does the API actually return `xwoba`, populated H2H samples, etc., or are fields frequently null for non-star players? | A feature that's null for 40% of the player pool forces a fallback path and weakens the signal. | _TBD_ |
| 4 | **`xwoba` coverage** — is `xwoba` populated for enough of the player pool to be usable, or only high-volume players? | `xwoba` (Statcast-derived) is the single most valuable field balldontlie exposes that the current sources don't. If it only exists for stars, the arsenal-matchup feature is not viable. | _TBD_ |

### Known integration cost discovered before the trial

**balldontlie does NOT use MLB Stats API IDs.** mlb-picker's SQLite DB stores
MLB Stats API IDs (e.g. pitcher `663568`, team `109`=ARI). balldontlie assigns
its own internal IDs. Any integration must maintain a **name/abbreviation →
balldontlie-ID crosswalk**, refreshed as rosters change. The probe resolves IDs
via `/mlb/v1/players?search=` and `/mlb/v1/teams` — record below how reliable
that resolution is (exact-match rate, fuzzy/ambiguous cases).

ID-resolution reliability (fill after probe): _TBD — e.g. "N/N players resolved
exact, M fuzzy"_

---

## API basics (confirmed from the OpenAPI spec)

- **Base URL:** `https://api.balldontlie.io/mlb/v1`
- **Auth:** `Authorization` header, raw API key (no `Bearer` prefix)
- **Pagination:** cursor-based (`cursor` integer param → `next_cursor` in response)
- **Tiers:** Free 5 req/min · All-Star $9.99/sport 60 req/min · GOAT $39.99/sport
  600 req/min. The 48h trial = GOAT-tier *endpoints* at the *free* 5 req/min rate.

---

## Results table — per endpoint (fill after probe)

| Endpoint | Works? (HTTP) | Fields present vs. spec | Earliest season w/ data | Latency | Rate-limit headroom | Verdict |
|----------|---------------|-------------------------|-------------------------|---------|---------------------|---------|
| `/season_stats` (WAR, QS) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _training / inference-only / not usable_ |
| `/players/versus` (batter-vs-team H2H) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `/pitcher_pitch_type_season_stats` (arsenal: xwoba, whiff) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `/hitter_pitch_type_season_stats` (lineup vs pitch type) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `/players/splits` (pitcher splits, monthly form) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `/player_injuries` (structured injuries) | _TBD_ | _TBD_ | n/a (current) | _TBD_ | _TBD_ | _TBD_ |
| `/plate_appearances` (Statcast pitch detail) | not probed live¹ | — | — | — | — | _offline-backfill only — see below_ |

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
| `arsenal_xwoba_diff` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _build / skip / inconclusive_ |
| `h2h_ops_diff` (full lineup-weighted)¹ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

¹ `h2h_ops_diff` is the **full lineup-weighted** version — every batter in both
stored lineups, AB-weighted (batters with more shared H2H history count more).
This is the expensive part of the run: ~18 `/players/versus` calls/game (the
endpoint has no batch param). At the trial's 5 req/min a 40-game backtest is
~3.5–4 hrs — run it unattended (`nohup`). Sample restricted to the 141 scored
2026 games that have full stored lineups, which cover ≈2026-04-12..05-05 — so
the result speaks to April/early-May 2026, not the whole season.

H2H sample-depth note (fill after run): _avg total H2H at-bats/game = TBD._ If
this is low (<40), many batter-vs-team pairs simply lack shared history this
early in 2026 and the feature is thin regardless of its correlation score.

Current-model baseline accuracy on the backtest sample: _TBD_ (compare to the
68.2% / 62% figures in CLAUDE.md to confirm the sample is representative).

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

## Final recommendation — BUY / NO-BUY (write after both scripts)

_The trial succeeds if it produces a defensible answer to this, on evidence:_

**Buy if:** at least one candidate feature scores [1] ≥0.15, [2] <0.7, [3] >50%
in the backtest **and** the cheapest sufficient tier is justified by the number
of features that passed. The data being "richer" is not, on its own, a reason
to buy — the 5/18 experiment is the standing counter-example.

**No-buy if:** every candidate feature is negligible/redundant (the 5/18
outcome repeats), or the only features with signal are inference-only (no 2022
depth) *and* the live-only lift doesn't justify a monthly cost.

Complete after the trial:

1. _Which candidate features passed the backtest (the [1]/[2]/[3] table)._
2. _Which passing features are training-eligible (2022+ data) vs inference-only._
3. _The cheapest tier that clears the call volume of the passing features only._
4. _Whether the offline `/plate_appearances` xwoba backfill is worth pursuing
   (only relevant if pitch-type season data lacks 2022 depth)._
5. _The concrete feature list + tier to commit to when XGBoost work begins,
   or an explicit NO-BUY with the numbers that justify it._
