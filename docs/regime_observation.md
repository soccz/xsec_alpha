# Regime Observation, Not Automatic Rotation

## Why This Exists

The fixed September trial finished with all 60 paired windows. Its registered
model-minus-reversal interval included zero: mean +0.500371 percentage points,
95% interval [-0.235116, +1.318287]. The verdict remains `inconclusive`.
Chronological thirds differed, but those post-hoc slices do not establish
regime dependence and are not admitted to this new prospective ledger.

The hypothesis is that relative ranking quality changes with market conditions.
This is distinct from periodic model retraining and the IC safety gate.
Regime models have a literature, but it does not validate this crypto strategy:
[Ang and Timmermann, Regime Changes and Financial Markets](https://www.nber.org/papers/w17182).

## Implemented Contract

Run `python scripts/observe_regimes.py --register` once. Registration creates
`output/regime_observation/ledger.sqlite` and refuses an in-place reset.
The ledger protocol, registration timestamp and first signal are authoritative.

- Four labels: BTC 7-day return sign crossed with high/low 7-day hourly volatility.
- Volatility reference: shifted, trailing 30-day median of the 7-day volatility
  series, requiring at least 7 days of reference observations. No full-sample fit.
- Read raw BTC closes, not the forward-filled live feature pivot. Only candles
  strictly preceding the signal hour enter labels. Gaps and insufficient warmup
  yield `unknown`; no zero/forward/back filling. This is a BTC proxy, not a
  comprehensive altcoin-season or liquidity regime classifier.
- Capture at 05/11/17/23 UTC before +45 minutes, after the normal Telegram report.
- Save current model scores and raw reversal scores in the same finite, observed
  Bitget-tradable pool. Select each bottom five with alphabetical tie-breaking.
  These are ungated/unbuffered shadow baskets, not the actual live portfolio.
- Save the live model hash each time. Periodic retraining remains active; pooled
  figures describe the ranking system, not one frozen model. Version counts are
  exposed, and any later single-model claim needs version-specific analysis.
- Reuse the strict raw next-open +6h evaluator: frozen membership, whole-window
  invalidation after a 48h price grace period, no dropping/replacing losing coins.
  Cost assumptions are fixed at registration; returns are Upbit paper/proxy,
  not Bitget fills or executable profit.
- The independent 10-minute supervisor settles saved snapshots and accounts for
  missed slots. Existing observations and outcomes are never overwritten.
- Weekly descriptive review uses only outcomes already evaluated by Monday
  00:00 UTC (09:00 KST), within the preceding 90 days. It may first appear at the
  next supervisor publication. Newer outcomes wait for the next weekly boundary.
- A regime needs 30 windows across at least 8 UTC dates to be marked available
  for descriptive review. These minimums are NOT a power calculation, evidence
  of significance, or permission to switch strategies.
- No automatic switching, promotion, LONG reactivation, orders or extra trade
  notifications. Mandatory normal coin reports and observation fallbacks remain.

Frozen observation/evaluation source fingerprints stop new recording if the
contract code changes. A revised contract needs a separately reviewed version,
not deleting the original evidence.

## What Is Deliberately Not Implemented

A profitable retrospective regime selector is not a validated adaptive strategy.
Before allowing rotation, specify a separate prospective comparison against the
unchanged policy. Fix the candidate strategies, decision timing, sample-size
calculation, outcome horizon, multiple-comparison handling, switching costs,
minimum holding period, and a single endpoint before starting.

A conservative candidate design is weekly reassessment, two consecutive eligible
reviews before a change, a cooldown, and a limited initial challenger allocation.
Those are proposals, not tuned or proven rules. The present observer supplies
timely evidence for choosing such a follow-up; it does not establish causality,
forecast future regimes or turn explanation into an accuracy claim.

## Operations Recovery

The completed original trial was sealed before changing its registered runtime.
`review_summary()` verifies the checkpoint, live ledger semantics, frozen model,
review plan and original final report, while exposing live-code mismatch separately.
Neither protocol nor final decision is rewritten to match the new runtime.

Dashboard publication now writes explicit xsec paths onto the latest remote tree
using an isolated bare Git cache. A concurrent remote change causes a bounded
retry without force pushing. The interactive website checkout, including its
untracked files and divergent unpublished commits, is untouched.

Health includes storage reserve, supervisor age, verified backup and publication
age. Operational notices retry unacknowledged failures and suppress unchanged
acknowledged incidents. Recovery gets a separate notice. Public HTML deployment
still needs GitHub Pages success verification; a Git push is not that verification.

As of the September 30 audit, the secondary backup disk has less than 1 GiB free.
The 5 GiB reserve remains enforced. The full primary checkpoint exists, but the
completed secondary copy is not repaired by merely changing reporting code.
