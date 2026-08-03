# xsec_alpha continuity

Updated 2026-08-03.

- Runtime contract: SHORT emits at most five manual-review signals. The repository has no broker/order API.
- LONG execution and the recommendation ledger are hard-blocked at zero by `LIVE_WATCH_LONG_N=0`.
- `strong_shadow_watch_v1` may send at most one non-actionable Telegram `LONG WATCH` at the 11/23 UTC slot, only after the strict BTC regime, current-contract IC, preflight, Bitget tradability, consensus, probability, expected-return, sigma, and trust gates all pass.
- The 2026-08-03 repair added run locks and scheduling order, exact-anchor/next-open/non-overlap measurement, horizon-purged training, cost-aware LONG promotion, OOS calibration, exact observed entry times, and a separate prospective LONG shadow ledger.
- Historical WATCH_LONG rows are legacy evidence. The 205 rows emitted after the 2026-07-11 KILL declaration document a policy-to-runtime wiring failure and must not count toward reactivation.
- Reactivation is never automatic. Review only after 60 independent 12h baskets, positive mean net, positive block-bootstrap 95% lower bound, and positive net in all three temporal folds and allowed regimes; then require a separate reviewed code change.
- Public narrative and aggregate dashboard live in `soccz.github.io/projects/xsec-alpha/`; keep paper/proxy returns clearly separated from real fills.

Next decision: accumulate the new prospective LONG shadow evidence and apply the preregistered gates without lowering thresholds.
