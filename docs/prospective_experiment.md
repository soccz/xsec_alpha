# Fixed SHORT Ranking Trial

This is a prospective ranking experiment, not a backtest, trade executor, or
replacement for the live recommendation policy. Production retraining, the
SHORT safety gates, LONG KILL, and mandatory Telegram reports are unchanged.

September 30 status: the original 60-window trial finished `inconclusive` on
September 23 KST. Its final report and ledger remain unchanged. Before editing
registered operational source, `seal_completed()` preserved and verified the
complete checkpoint. `utils.experiment_supervisor.review_summary()` now checks
that sealed evidence separately from current live runtime. Direct calls to the
old `experiment_summary()` still truthfully report current-source mismatch;
that is not a rewritten historical protocol. See
[separate regime observation](regime_observation.md) for the new, non-switching
diagnostic stream. On September 30 the operator explicitly accepted local-only
recovery: the verified primary checkpoint and restore drill remain required,
but a second disk is no longer mandatory. Disk-failure protection is absent.

## Registration

`python scripts/prospective_experiment.py --start` copies the current 6h model
into `output/prospective/model.pkl` and stores a checksummed protocol in
`output/prospective/ledger.sqlite`. Registration cannot reset an existing trial.
The actual registration timestamp and model hash are authoritative in the ledger.
No old recommendation, reconstructed historical score, or test fixture is admitted.

The next 05/11/17/23 UTC anchor after registration is slot one. Exactly 60
scheduled slots are followed, not "until 60 profitable/valid results appear".
Changes to feature/scoring/selection source files, relevant package versions or
the frozen model block new snapshots. A production model file replacement alone
does not affect this frozen trial. A changed protocol requires separate review;
do not delete or overwrite the original experiment to restart the count.

## Comparisons

At each anchor, use the current active liquidity universe intersected with the
observed Bitget-tradable set. Store that membership before outcomes are known.
All comparators use the same finite-input pool. Preserve original missing-input
flags even when the existing live inference pipeline has supplied neutral values.
This trial does not introduce a new imputation or feature pipeline.

| Strategy | Frozen Selection Rule |
| --- | --- |
| `model_short5` | Lowest five frozen-model scores, equal weight |
| `reversal_short5` | Lowest five `reversal_4h` values, equal weight |
| `universe_short` | Equal-weight SHORT of the entire saved comparable pool |

Ties use alphabetical market order. No buffer, sigma selection, or execution
gate is applied to these **shadow baskets**. This isolates ranking value, not
the efficacy of the live buffered/gated portfolio. Actual live selection and
actionability, current model hash and gate/preflight context are stored separately.
The experiment never sends additional trade suggestions or adds trading-ledger rows.

## Timing and Outcomes

- Record before anchor + 45 minutes; no post-hoc capture after that deadline.
- Entry proxy: raw Upbit candle open at anchor + 1h, strictly after recording.
- Exit proxy: raw open exactly 6h after entry. Do not settle before that exit
  candle is complete. Scheduled collection can make settlement appear one run later.
- Read original SQLite `crypto_data` rows, not forward/back-filled pivots. Include
  saved markets even if they later leave the active universe or are delisted.
- Every frozen pool member needs both positive finite, unique raw prices. Missing
  data is pending for up to 48h after exit, then the whole comparison is ineligible.
  Never drop a losing/missing coin, substitute another, or reweight survivors.
- Duplicate signals and matured results are append-only/idempotent. Existing
  results are not recomputed when the source price database is revised.
- SHORT gross is minus the raw coin return. Net deducts the registered round-trip
  fee/slippage plus the per-window short-extra assumption for each comparator.
  This is a raw-open **paper/proxy** return, not Upbit short execution or Bitget
  fills/funding. It cannot establish executable profitability.

## Interpretation

The primary endpoint is the paired per-window net difference between the frozen
model and the reversal baseline. The entire five-coin basket is one observation.
Model and baseline Spearman IC use the same saved cross-section and future return
vector; no asymptotic per-coin p-value is reported.

Means, IC and SHORT-gate breakdowns remain descriptive while collecting. At the
single final review, require all 60 paired windows for the registered 95% interval:
a circular moving-block bootstrap of four adjacent 6h windows, 2,000 draws and
seed 42. Missing slots are counted and not replaced; incomplete trials report
`incomplete_evidence`, not success. The four-window dependence assumption is a
declared approximation, not proof of independence or significance under all regimes.
There is no automatic threshold tuning, model promotion or trade activation.

For each selected model coin, save the three largest absolute changes between
its original score and the score after replacing one feature with that timestamp's
universe median. Negative deltas support the low/SHORT score. These are model
sensitivity checks, **not SHAP, additive contributions or causal explanations**;
correlated inputs can make the substitutions unrealistic. Link these saved checks
to each coin's subsequent proxy return, without regenerating the explanation.

## Visibility and Maintenance

- GitHub dashboard: private encrypted summary contains `prospective_experiment`,
  counts, comparisons, recent evidence/outcomes and protocol. Public introductory
  JSON excludes individual experiment data. No fixture outcomes are exported.
- Local API: `GET /api/experiment`.
- Status: `python scripts/prospective_experiment.py`.
- Settle saved observations: `python scripts/prospective_experiment.py --settle`.
- Normal `fetch_and_rank` records after attempting Telegram delivery, then settles
  in `finally` before dashboard export. Experiment errors cannot suppress the
  recommendation report. A wholly missed operating run is accounted for on the
  next maintenance pass; the one-minute Telegram retry does not run the experiment.
- Keep the complete experiment directory in backups. The local SQLite ledger is
  not an externally timestamped or tamper-proof audit service; checksums/version
  checks detect accidental changes but do not establish independent attestation.

The separate ten-minute supervisor now settles saved observations, accounts for
missed slots, creates integrity-checked checkpoints on a second filesystem and
performs temporary restore drills. A pre-first-slot review addendum classifies
the fixed trial at completion or its hard data deadline; it never rewrites the
original protocol. See [validation direction and supervision](validation_direction.md).

## Method References

Repeatedly selecting a winner on historical data can overstate out-of-sample
performance. This trial responds by preregistering one primary comparison and
using future data; it does not implement or claim a PBO estimate.
[Bailey, Borwein, Lopez de Prado and Zhu, The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

SciPy cautions that the usual Spearman p-value approximation is accurate only
for very large samples. This trial reports the IC coefficient descriptively and
does not use a 90-coin cross-section to claim a significant directional edge.
[SciPy spearmanr documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).
