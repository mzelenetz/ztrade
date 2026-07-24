# Multi-Structure Spread Scanner — Design

**Date:** 2026-07-24
**Status:** Approved

## Problem

The Spreads scan only produces buy/sell pairs: long the most undervalued leg,
short the most overvalued leg. Structures like a long straddle (buy call + buy
put when both are cheap) or a short straddle (sell both when both are rich) are
structurally impossible — the delta-offset filter and the edge formula in
`build_spreads` assume exactly one long and one short leg.

## Goal

Extend the scanner to generate three two-leg structures, ranked together by net
edge, without a combinatorial blowup:

- **buy/sell** (unchanged): long cheap leg, short rich leg, same-sign option
  deltas so the position deltas offset.
- **buy/buy**: long a call + long a put, when combined undervaluation is
  positive.
- **sell/sell**: short a call + short a put, when combined overvaluation is
  positive.

All structures remain delta-offsetting vol trades. Same-side pairs must pair a
call with a put (opposite raw delta signs). Quantities stay delta-balanced
using the existing anchor-10 / try-both-anchors logic. The existing
`max_abs_net_delta`, `max_contract_ratio`, and `max_straddle_ratio` caps apply
to every structure.

## Algorithm

Edge is additive per leg (a buy leg contributes its undervaluation, a sell leg
its overvaluation), so the best combos can only come from the best individual
legs. The pipeline in `build_spreads` becomes:

1. **Score once.** Existing `compute_overvalued` + delta/price filters, sorted
   by `%Overvalued`.
2. **Buckets.** Four top-K pools (K = `max_legs_per_side`): undervalued calls,
   undervalued puts, overvalued calls, overvalued puts. buy/sell keeps its
   current head/tail pools (any type).
3. **Vectorized pairing.** Per structure, cross-join the two relevant buckets
   in polars and compute as columns: both anchor quantities, contract ratio,
   straddle ratio, net delta, and gross edge. Filter caps, `edge > 0`, and
   same-contract exclusion inside the join result.
4. **Deferred margin.** Concatenate all structures' survivors, rank by gross
   edge dollars, keep the top `3 × max_results` (slack because carry cost can
   reorder), run the Python margin/carry/executable-edge math only on those,
   re-rank by net edge dollars, return `max_results`.

Complexity: O(n log n) scoring plus a few thousand vectorized candidate rows,
independent of chain size. Expensive margin math runs on at most
`3 × max_results` rows.

## Margin

- **buy/sell:** unchanged (Reg T short-leg requirement or portfolio margin).
- **buy/buy:** no margin requirement; capital employed = net debit.
- **sell/sell (Reg T):** short-straddle rule — the greater of the two legs'
  individual requirements plus the premium of the other leg.
- **sell/sell (portfolio):** existing `portfolio_margin_requirement` already
  handles arbitrary leg lists.

Net debit stays signed; sell/sell results show a net credit.

## API

Each spread result gains:

- `structure`: `"buy_sell" | "buy_buy" | "sell_sell"`.
- `leg1` / `leg2` replacing `buy`/`sell`/`buyQty`/`sellQty`/`buyLeg`/`sellLeg`:
  each is `{ side: "buy" | "sell", contract, qty, detail }` where `detail` is
  the existing leg-detail payload.

No new query parameters; structure filtering is client-side.

## Frontend

`SpreadsView`:

- Columns become Structure, Leg 1, Leg 2 (side baked into the label, e.g.
  "B 10× NVDA Jan27 190c"), plus the existing economics columns.
- Client-side structure filter: All / Buy–Sell / Buy–Buy / Sell–Sell.
- Leg cards titled by side ("Leg 1 (Buy)", "Leg 2 (Sell)", etc.).
- `types.ts` updated to the new `Spread` shape.

## Testing

Extend `tests/test_services.py`:

- Synthetic chain with a cheap call + cheap put surfaces a buy/buy with
  positive edge; rich call + rich put surfaces a sell/sell.
- Delta offsetting and all caps enforced per structure.
- sell/sell Reg T margin equals greater-requirement + other-leg premium.
- buy/sell results unchanged from today on a mixed chain (regression).

## Out of scope

- Manual two-leg builder UI (may follow later).
- Structures with >2 legs.
- Same-type same-side pairs (e.g. two long calls) — excluded by the
  call+put rule.
