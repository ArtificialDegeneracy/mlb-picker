# Open Items

Living list of model/dashboard improvements we've identified but not yet shipped.
Most-recent additions at the top.

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
- Is the FIP constant (3.10 hardcoded in `data/fip.py`) drifting from actual league FIP?

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
