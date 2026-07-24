# Multi-Structure Spread Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Spreads scan to generate buy/buy and sell/sell structures alongside the existing buy/sell, with a vectorized polars pipeline and margin computed only for final survivors.

**Architecture:** `build_spreads` scores every leg once, builds top-K candidate buckets per (role, option type), cross-joins buckets per structure in polars with every cheap filter vectorized, then runs Python margin/carry math on only the top `3 × max_results` candidates ranked by gross edge. The API result shape generalizes `buy`/`sell` fields into `leg1`/`leg2` objects with a `structure` tag; the Ideas service and both frontend views consume the new shape.

**Tech Stack:** Python 3 + polars + FastAPI (backend), React + TypeScript + AG Grid (frontend), pytest, vitest not used (frontend has no tests).

**Spec:** `docs/superpowers/specs/2026-07-24-multi-structure-spreads-design.md`

## Global Constraints

- All structures are delta-offsetting vol trades: position deltas (side × option delta) of the two legs must have opposite signs.
- Same-side structures (buy/buy, sell/sell) therefore always pair a call with a put.
- Existing caps apply to every structure: `max_contract_ratio`, `max_straddle_ratio` (same expiry+strike call/put pair), `max_abs_net_delta`.
- Quantity anchoring unchanged: anchor 10 contracts on each leg in turn, delta-balance the other, keep both distinct variants.
- No new API query parameters; structure filtering is client-side.
- Result rows ranked by `netEdgeDollars` descending, capped at `max_results`.
- `netDebit` stays signed (sell/sell shows a credit as a negative number).
- Run backend tests with `uv run pytest -q`; typecheck/build frontend with `cd web && npx tsc -b`.
- The repo has unrelated uncommitted WIP (`src/api/data.py`, `src/api/routers/meta.py`, `src/api/schemas.py`, `src/ingest/job.py`, `src/ingest/rates.py`, `tests/test_rates.py`). Never `git add -A`; stage only the files each task names.

---

### Task 1: Reg T margin for a two-short-leg pair

**Files:**
- Modify: `src/services/margin_service.py` (add one function at the end)
- Test: `tests/test_margin.py` (append a test class)

**Interfaces:**
- Consumes: `short_margin_requirement(spot, strike, opt_type, premium, qty) -> float` and `CONTRACT_MULTIPLIER` (both already in `margin_service.py`).
- Produces: `short_pair_margin_requirement(legs: list[dict]) -> float` where each leg dict has keys `spot: float, strike: float, type: str ("C"/"P"), premium: float, qty: int`. Exactly two legs. Task 2 imports this.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_margin.py`:

```python
from src.services.margin_service import short_pair_margin_requirement


class TestShortPairMargin:
    def test_greater_requirement_plus_other_premium(self):
        # Call: spot 100, strike 110 → max(20−10, 10) + 5 = 15/share → 15,000 for 10 contracts.
        # Put: strike 100 → max(20, 10) + 20 = 40/share → 40,000 for 10 contracts.
        call = {"spot": 100.0, "strike": 110.0, "type": "C", "premium": 5.0, "qty": 10}
        put = {"spot": 100.0, "strike": 100.0, "type": "P", "premium": 20.0, "qty": 10}
        # Put side dominates: 40,000 + call premium dollars (5 · 100 · 10 = 5,000).
        assert short_pair_margin_requirement([call, put]) == pytest.approx(45_000)
        # Order must not matter.
        assert short_pair_margin_requirement([put, call]) == pytest.approx(45_000)

    def test_symmetric_legs(self):
        # Identical requirements both sides: 40/share → 40,000; + other premium 20,000.
        call = {"spot": 100.0, "strike": 100.0, "type": "C", "premium": 20.0, "qty": 10}
        put = {"spot": 100.0, "strike": 100.0, "type": "P", "premium": 20.0, "qty": 10}
        assert short_pair_margin_requirement([call, put]) == pytest.approx(60_000)
```

(`import pytest` already exists at the top of `tests/test_margin.py`; if not, add it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_margin.py::TestShortPairMargin -v`
Expected: FAIL with `ImportError: cannot import name 'short_pair_margin_requirement'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/services/margin_service.py`:

```python
def short_pair_margin_requirement(legs: list[dict]) -> float:
    """Reg T requirement for two naked short legs (short straddle/strangle):
    the greater leg's naked requirement plus the current premium of the other
    leg. Each leg: {spot, strike, type, premium, qty}."""
    reqs = [
        short_margin_requirement(
            spot=leg["spot"],
            strike=leg["strike"],
            opt_type=leg["type"],
            premium=leg["premium"],
            qty=leg["qty"],
        )
        for leg in legs
    ]
    premiums = [leg["premium"] * CONTRACT_MULTIPLIER * leg["qty"] for leg in legs]
    if reqs[0] >= reqs[1]:
        return reqs[0] + premiums[1]
    return reqs[1] + premiums[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_margin.py -v`
Expected: all PASS (new class plus existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/services/margin_service.py tests/test_margin.py
git commit -m "feat: add Reg T margin for two-short-leg pairs"
```

---

### Task 2: Vectorized multi-structure build_spreads

**Files:**
- Modify: `src/services/spreads_service.py` (replace `build_spreads`; keep `format_contract` and `_leg_detail` unchanged)
- Test: `tests/test_services.py` (update `TestBuildSpreads`, add structure tests)

**Interfaces:**
- Consumes: `short_pair_margin_requirement` from Task 1; existing `compute_overvalued`, `filter_by_delta_and_price`, `leg_summary`, `short_margin_requirement`, `portfolio_margin_requirement`, `CONTRACT_MULTIPLIER`.
- Produces: `build_spreads(...)` with the **same signature** as today, but each result dict is now:
  ```python
  {
    "structure": "buy_sell" | "buy_buy" | "sell_sell",
    "leg1": {"side": "buy"|"sell", "contract": str, "qty": int, "detail": <leg_summary payload>},
    "leg2": {...},
    "netDelta": float, "edge": float, "marginRequirement": float, "netDebit": float,
    "carryCost": float, "grossEdgeDollars": float, "netEdgeDollars": float,
    "execEdgeDollars": float | None, "capitalEmployed": float,
  }
  ```
  The old keys `buy`, `sell`, `buyQty`, `sellQty`, `buyLeg`, `sellLeg` are **removed**. For buy_sell, leg1 is the buy and leg2 the sell; for buy_buy/sell_sell, leg1 is the call and leg2 the put. Tasks 3–5 rely on this shape.
- Behavior deltas vs today (intentional, covered by tests): exact duplicate variants (same legs, same quantities from both anchors) are deduplicated; candidates must also have `grossEdgeDollars > 0`.

- [ ] **Step 1: Update existing tests to the new shape and add structure tests**

In `tests/test_services.py`, replace the body of `TestBuildSpreads` (keep `spreads_df` and `run_build_spreads` as they are) with:

```python
class TestBuildSpreads:
    def test_pairing_and_delta_neutral_sizing(self):
        spreads = run_build_spreads()
        # (buy A, sell B) with both anchor variants; reverse pair has negative edge.
        assert len(spreads) == 2
        for s in spreads:
            assert s["structure"] == "buy_sell"
            assert s["leg1"]["side"] == "buy" and s["leg1"]["contract"].endswith("100c")
            assert s["leg2"]["side"] == "sell" and s["leg2"]["contract"].endswith("110c")
            assert s["netDelta"] == pytest.approx(0.0)
            assert s["edge"] == pytest.approx(1.5)  # 1.0 − (−0.5)

        by_anchor = {s["leg1"]["qty"]: s for s in spreads}
        assert set(by_anchor) == {10, 5}
        assert by_anchor[10]["leg2"]["qty"] == 20  # 0.50·10 / 0.25
        assert by_anchor[5]["leg2"]["qty"] == 10

    def test_mispriced_call_put_pair_is_not_a_buy_sell(self):
        # Cheap call + rich put: same-sign-delta rule blocks buy/sell (it would be
        # directional), buy_buy edge is negative, and sell_sell nets to zero gross
        # dollars — so nothing qualifies.
        assert build_spreads(
            _call_put_df(call_last=10.0, call_fmv=20.0, put_last=20.0, put_fmv=10.0),
            **_default_args(),
        ) == []

    def test_buy_buy_long_straddle_when_both_legs_cheap(self):
        spreads = build_spreads(
            _call_put_df(call_last=10.0, call_fmv=20.0, put_last=10.0, put_fmv=20.0),
            **_default_args(),
        )
        assert len(spreads) == 1  # identical anchor variants deduplicate
        s = spreads[0]
        assert s["structure"] == "buy_buy"
        assert s["leg1"]["side"] == "buy" and s["leg1"]["contract"].endswith("100c")
        assert s["leg2"]["side"] == "buy" and s["leg2"]["contract"].endswith("100p")
        assert s["leg1"]["qty"] == 10 and s["leg2"]["qty"] == 10
        assert s["netDelta"] == pytest.approx(0.0)
        assert s["edge"] == pytest.approx(1.0)  # 0.5 undervaluation per leg
        # Fully paid: no margin, capital is the debit.
        assert s["marginRequirement"] == 0.0
        assert s["netDebit"] == pytest.approx(20_000)
        assert s["capitalEmployed"] == pytest.approx(20_000)
        assert s["grossEdgeDollars"] == pytest.approx(20_000)
        # carry = 20,000 · 10% · 0.5y
        assert s["carryCost"] == pytest.approx(1_000)
        assert s["netEdgeDollars"] == pytest.approx(19_000)

    def test_sell_sell_short_straddle_when_both_legs_rich(self):
        spreads = build_spreads(
            _call_put_df(call_last=20.0, call_fmv=10.0, put_last=20.0, put_fmv=10.0),
            **_default_args(),
        )
        assert len(spreads) == 1
        s = spreads[0]
        assert s["structure"] == "sell_sell"
        assert s["leg1"]["side"] == "sell" and s["leg2"]["side"] == "sell"
        assert s["netDelta"] == pytest.approx(0.0)
        assert s["edge"] == pytest.approx(2.0)
        # Each naked leg: max(20% · 100, floor) + 20 = 40/share → 40,000 per 10 contracts.
        # Pair rule: greater requirement + other leg premium = 40,000 + 20,000.
        assert s["marginRequirement"] == pytest.approx(60_000)
        assert s["netDebit"] == pytest.approx(-40_000)  # credit
        # capital = (60,000 − 40,000 received) + 0
        assert s["capitalEmployed"] == pytest.approx(20_000)
        assert s["grossEdgeDollars"] == pytest.approx(20_000)
        assert s["carryCost"] == pytest.approx(1_000)
        assert s["netEdgeDollars"] == pytest.approx(19_000)

    def test_straddle_ratio_cap_applies_to_buy_buy(self):
        # Same strike/expiry call+put with 2:1 delta imbalance → qty ratio 2
        # exceeds max_straddle_ratio 1.5 → excluded.
        df = _call_put_df(
            call_last=10.0, call_fmv=20.0, put_last=10.0, put_fmv=20.0,
            call_delta=0.50, put_delta=-0.25,
        )
        assert build_spreads(df, **_default_args()) == []

    def test_reg_t_dollar_economics(self):
        s = next(x for x in run_build_spreads() if x["leg1"]["qty"] == 10)
        # Reg T short leg: S=100, K=110 → max(20−10, 10)=10; +prem 20 → 30/share × 100 × 20
        assert s["marginRequirement"] == pytest.approx(30 * 100 * 20)
        # net debit: 10·100·10 − 20·100·20 = −30,000 (credit)
        assert s["netDebit"] == pytest.approx(-30_000)
        # capital: (60,000 − 40,000) + 0 = 20,000 → carry at 10% for 0.5y = 1,000
        assert s["carryCost"] == pytest.approx(20_000 * 0.10 * 0.5)
        # gross: (20−10)·100·20 + (20−10)·100·10 = 30,000
        assert s["grossEdgeDollars"] == pytest.approx(30_000)
        assert s["netEdgeDollars"] == pytest.approx(30_000 - 1_000)

    def test_carry_identity_and_zero_rate(self):
        for s in run_build_spreads():
            assert s["grossEdgeDollars"] - s["carryCost"] == pytest.approx(s["netEdgeDollars"])
        for s in run_build_spreads(margin_rate=0.0):
            assert s["carryCost"] == 0.0
            assert s["netEdgeDollars"] == pytest.approx(s["grossEdgeDollars"])

    def test_sorted_by_net_edge_desc(self):
        vals = [s["netEdgeDollars"] for s in run_build_spreads()]
        assert vals == sorted(vals, reverse=True)

    def test_portfolio_margin_lower_than_reg_t_for_hedged_pair(self):
        reg_t = next(x for x in run_build_spreads(margin_style="reg_t") if x["leg1"]["qty"] == 10)
        pm = next(x for x in run_build_spreads(margin_style="portfolio") if x["leg1"]["qty"] == 10)
        # Long 10× 100-call hedges short 20× 110-call under stress → PM well below naked Reg T.
        assert pm["marginRequirement"] < reg_t["marginRequirement"]
        assert pm["carryCost"] < reg_t["carryCost"]
        assert pm["netEdgeDollars"] > reg_t["netEdgeDollars"]
        # PM capital (net credit) = requirement → carry = req · rate · T
        assert pm["carryCost"] == pytest.approx(pm["marginRequirement"] * 0.10 * 0.5)

    def test_net_delta_filter_excludes(self):
        assert run_build_spreads(max_abs_net_delta=0.0) != []  # netDelta is exactly 0 here
        # Tighten contract ratio below 2 → both variants (ratio 2) excluded.
        assert run_build_spreads(max_contract_ratio=1.9) == []
```

Above `class TestBuildSpreads`, add the two helpers it uses (next to `spreads_df`):

```python
def _call_put_df(
    call_last: float,
    call_fmv: float,
    put_last: float,
    put_fmv: float,
    call_delta: float = 0.50,
    put_delta: float = -0.50,
) -> pl.DataFrame:
    """One call and one put, same strike/expiry, spot 100, T=0.5."""
    expiry = datetime(2026, 12, 18, 16, 0)
    lasts = [call_last, put_last]
    return pl.DataFrame(
        {
            "Ticker": ["T", "T"],
            "Expiry": [expiry, expiry],
            "Type": ["C", "P"],
            "Strike": [100.0, 100.0],
            "Delta": [call_delta, put_delta],
            "Last": lasts,
            "FMV": [call_fmv, put_fmv],
            "Bid": [v - 0.1 for v in lasts],
            "Ask": [v + 0.1 for v in lasts],
            "Mid": lasts,
            "Spot": [100.0, 100.0],
            "Vol30d": [0.25, 0.25],
            "Rate": [0.05, 0.05],
            "DividendYield": [0.0, 0.0],
            "T": [0.5, 0.5],
        }
    )


def _default_args(**overrides) -> dict:
    args = dict(
        metric_choice="Last",
        delta_min=0,
        delta_max=100,
        max_contract_ratio=2.5,
        max_straddle_ratio=1.5,
        min_option_price=0.0,
        max_last_price=1e9,
        max_abs_net_delta=10.0,
        max_legs_per_side=50,
        max_results=50,
        margin_rate=0.10,
        margin_style="reg_t",
    )
    args.update(overrides)
    return args
```

Also delete the old `test_call_put_pairs_are_excluded_as_directional` method (its inline DataFrame is superseded by `_call_put_df`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_services.py::TestBuildSpreads -v`
Expected: FAIL — old implementation returns dicts with `buy`/`sell` keys (KeyError `structure` / `leg1`).

- [ ] **Step 3: Replace build_spreads with the vectorized multi-structure pipeline**

In `src/services/spreads_service.py`, keep the imports, `format_contract`, and `_leg_detail`, add `short_pair_margin_requirement` to the margin imports, and replace everything from `def build_spreads(` down with:

```python
BASE_QTY = 10  # anchor quantity on one side
MARGIN_SLACK = 3  # gross-edge-ranked candidates margined per requested result

# structure → (side of leg 1, side of leg 2); +1 = buy, −1 = sell.
# For buy_sell leg1 is the buy; for same-side structures leg1 is the call.
STRUCTURES = {
    "buy_sell": (1, -1),
    "buy_buy": (1, 1),
    "sell_sell": (-1, -1),
}

_PAIR_COLS = ["RowId", "%Overvalued", "Delta", "Last", "FMV", "Expiry", "Strike", "Type"]


def _pair_frame(
    a: pl.DataFrame,
    b: pl.DataFrame,
    structure: str,
    max_contract_ratio: float,
    max_straddle_ratio: float,
    max_abs_net_delta: float,
) -> pl.DataFrame | None:
    """Cross-join two candidate pools and apply every cheap filter vectorized.

    Columns *_1 come from `a`, *_2 from `b`. Survivors carry both anchor-quantity
    variants (deduplicated), all caps applied, and positive edge both as a ratio
    and in gross dollars."""
    side1, side2 = STRUCTURES[structure]
    if a.is_empty() or b.is_empty():
        return None

    pairs = (
        a.select(_PAIR_COLS)
        .rename({c: f"{c}_1" for c in _PAIR_COLS})
        .join(
            b.select(_PAIR_COLS).rename({c: f"{c}_2" for c in _PAIR_COLS}),
            how="cross",
        )
        .filter(pl.col("RowId_1") != pl.col("RowId_2"))
        .filter((pl.col("Delta_1") != 0) & (pl.col("Delta_2") != 0))
        # Position deltas (side · option delta) must offset so the pair is a
        # volatility spread rather than a directional bet.
        .filter((side1 * pl.col("Delta_1")) * (side2 * pl.col("Delta_2")) < 0)
    )
    if pairs.is_empty():
        return None

    variants = []
    for anchor in (1, 2):
        anchored, balanced = ("1", "2") if anchor == 1 else ("2", "1")
        qty_balanced = (
            (pl.col(f"Delta_{anchored}").abs() * BASE_QTY / pl.col(f"Delta_{balanced}").abs())
            .round(0)
            .clip(lower_bound=1)
            .cast(pl.Int64)
        )
        variants.append(
            pairs.with_columns(
                pl.lit(BASE_QTY, dtype=pl.Int64).alias(f"Qty_{anchored}"),
                qty_balanced.alias(f"Qty_{balanced}"),
            ).select(*pairs.columns, "Qty_1", "Qty_2")
        )

    is_straddle = (
        (pl.col("Expiry_1") == pl.col("Expiry_2"))
        & (pl.col("Strike_1") == pl.col("Strike_2"))
        & (pl.col("Type_1") != pl.col("Type_2"))
    )

    return (
        pl.concat(variants)
        .unique(subset=["RowId_1", "RowId_2", "Qty_1", "Qty_2"])
        .with_columns(
            (
                pl.max_horizontal("Qty_1", "Qty_2") / pl.min_horizontal("Qty_1", "Qty_2")
            ).alias("ContractRatio"),
            (
                side1 * pl.col("Delta_1") * pl.col("Qty_1")
                + side2 * pl.col("Delta_2") * pl.col("Qty_2")
            ).alias("NetDelta"),
            (
                -side1 * pl.col("%Overvalued_1") - side2 * pl.col("%Overvalued_2")
            ).alias("Edge"),
            (
                (
                    side1 * (pl.col("FMV_1") - pl.col("Last_1")) * pl.col("Qty_1")
                    + side2 * (pl.col("FMV_2") - pl.col("Last_2")) * pl.col("Qty_2")
                )
                * CONTRACT_MULTIPLIER
            ).alias("GrossEdgeDollars"),
        )
        .filter(pl.col("ContractRatio") <= max_contract_ratio)
        .filter(~is_straddle | (pl.col("ContractRatio") <= max_straddle_ratio))
        .filter(pl.col("NetDelta").abs() <= max_abs_net_delta)
        .filter(pl.col("Edge") > 0)
        .filter(pl.col("GrossEdgeDollars") > 0)
        .with_columns(pl.lit(structure).alias("Structure"))
        .select(
            "Structure", "RowId_1", "RowId_2", "Qty_1", "Qty_2",
            "NetDelta", "Edge", "GrossEdgeDollars",
        )
    )


def _price_candidate(
    cand: dict, rows: dict[int, dict], margin_rate: float, margin_style: str
) -> dict:
    """Full economics (margin, carry, executable edge) for one surviving pair."""
    structure = cand["Structure"]
    sides = STRUCTURES[structure]
    legs = [
        (sides[0], rows[cand["RowId_1"]], int(cand["Qty_1"])),
        (sides[1], rows[cand["RowId_2"]], int(cand["Qty_2"])),
    ]

    net_debit = sum(
        side * float(leg["Last"]) * CONTRACT_MULTIPLIER * qty for side, leg, qty in legs
    )
    premium_received = sum(
        float(leg["Last"]) * CONTRACT_MULTIPLIER * qty for side, leg, qty in legs if side < 0
    )

    if structure == "buy_buy":
        # Fully paid position: no requirement, capital is the debit itself.
        margin_requirement = 0.0
        capital = max(net_debit, 0.0)
    elif margin_style == "portfolio":
        margin_requirement = portfolio_margin_requirement(
            [
                {
                    "spot": float(leg["Spot"]),
                    "strike": float(leg["Strike"]),
                    "type": leg["Type"],
                    "qty": qty,
                    "side": side,
                    "vol": float(leg["Vol30d"]),
                    "rate": float(leg["Rate"]),
                    "dividend": float(leg["DividendYield"]),
                    "tenor": float(leg["T"]),
                }
                for side, leg, qty in legs
            ]
        )
        # PM requirement is a pure worst-case loss (no premium component).
        capital = margin_requirement + max(net_debit, 0.0)
    else:
        shorts = [
            {
                "spot": float(leg["Spot"]),
                "strike": float(leg["Strike"]),
                "type": leg["Type"],
                "premium": float(leg["Last"]),
                "qty": qty,
            }
            for side, leg, qty in legs
            if side < 0
        ]
        if len(shorts) == 2:
            margin_requirement = short_pair_margin_requirement(shorts)
        else:
            margin_requirement = short_margin_requirement(
                spot=shorts[0]["spot"],
                strike=shorts[0]["strike"],
                opt_type=shorts[0]["type"],
                premium=shorts[0]["premium"],
                qty=shorts[0]["qty"],
            )
        # The premium-received component of the Reg T requirement is
        # self-funding, and a net credit doesn't reduce the margin held.
        capital = (margin_requirement - premium_received) + max(net_debit, 0.0)

    holding_years = min(float(leg["T"]) for _, leg, _ in legs)
    carry_cost = capital * margin_rate * holding_years

    gross_edge_dollars = float(cand["GrossEdgeDollars"])
    net_edge_dollars = gross_edge_dollars - carry_cost

    # Executable edge: what survives crossing the spread — buys fill at the ask,
    # sells at the bid. Edge that only exists at mid/last quotes is often phantom.
    exec_edge_dollars = None
    exec_prices = [leg.get("Ask") if side > 0 else leg.get("Bid") for side, leg, _ in legs]
    if all(p is not None and p > 0 for p in exec_prices):
        exec_edge_dollars = (
            sum(
                side * (float(leg["FMV"]) - float(price)) * CONTRACT_MULTIPLIER * qty
                for (side, leg, qty), price in zip(legs, exec_prices)
            )
            - carry_cost
        )

    def leg_payload(side: int, leg: dict, qty: int) -> dict:
        return {
            "side": "buy" if side > 0 else "sell",
            "contract": format_contract(leg["Ticker"], leg["Expiry"], leg["Strike"], leg["Type"]),
            "qty": qty,
            "detail": _leg_detail(leg),
        }

    return {
        "structure": structure,
        "leg1": leg_payload(*legs[0]),
        "leg2": leg_payload(*legs[1]),
        "netDelta": float(cand["NetDelta"]),
        "edge": float(cand["Edge"]),
        "marginRequirement": margin_requirement,
        "netDebit": net_debit,
        "carryCost": carry_cost,
        "grossEdgeDollars": gross_edge_dollars,
        "netEdgeDollars": net_edge_dollars,
        "execEdgeDollars": exec_edge_dollars,
        "capitalEmployed": capital,
    }


def build_spreads(
    df: pl.DataFrame,
    metric_choice: str,
    delta_min: float,
    delta_max: float,
    max_contract_ratio: float,
    max_straddle_ratio: float,
    min_option_price: float,
    max_last_price: float,
    max_abs_net_delta: float,
    max_legs_per_side: int,
    max_results: int,
    margin_rate: float = 0.11325,
    margin_style: str = "reg_t",
) -> list[dict]:
    sdf = (
        filter_by_delta_and_price(
            compute_overvalued(df, metric_choice), delta_min, delta_max, min_option_price
        )
        .filter(pl.col("Last") <= max_last_price)
        .filter(pl.col("%Overvalued").is_not_null())
        .sort("%Overvalued")
        .with_row_index("RowId")
    )
    if sdf.is_empty():
        return []

    # Edge is additive per leg, so the best combos can only come from the best
    # individual legs: top-K pools per (role, type). sdf is sorted ascending by
    # %Overvalued → head = most undervalued, tail = most overvalued.
    k = max_legs_per_side
    is_call = pl.col("Type") == "C"
    frames = [
        _pair_frame(
            sdf.head(k), sdf.tail(k), "buy_sell",
            max_contract_ratio, max_straddle_ratio, max_abs_net_delta,
        ),
        _pair_frame(
            sdf.filter(is_call).head(k), sdf.filter(~is_call).head(k), "buy_buy",
            max_contract_ratio, max_straddle_ratio, max_abs_net_delta,
        ),
        _pair_frame(
            sdf.filter(is_call).tail(k), sdf.filter(~is_call).tail(k), "sell_sell",
            max_contract_ratio, max_straddle_ratio, max_abs_net_delta,
        ),
    ]
    frames = [f for f in frames if f is not None and not f.is_empty()]
    if not frames:
        return []

    # Margin/carry is the expensive part: run it only on the strongest
    # candidates by gross edge, with slack because carry can reorder them.
    candidates = (
        pl.concat(frames)
        .sort("GrossEdgeDollars", descending=True)
        .head(MARGIN_SLACK * max_results)
    )

    rows = {r["RowId"]: r for r in sdf.to_dicts()}
    spreads = [
        _price_candidate(cand, rows, margin_rate, margin_style)
        for cand in candidates.to_dicts()
    ]
    return sorted(spreads, key=lambda r: r["netEdgeDollars"], reverse=True)[:max_results]
```

Delete the old `build_spreads` body entirely (the `base_qty` loop, `maybe_add_spread`, etc.).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_services.py -v`
Expected: all PASS. If a polars API mismatch surfaces (`clip(lower_bound=1)` vs `clip(1)` depends on version), fix to whatever `uv run python -c "import polars; print(polars.__version__)"` supports.

- [ ] **Step 5: Commit**

```bash
git add src/services/spreads_service.py tests/test_services.py
git commit -m "feat: multi-structure spread scanner (buy/sell, buy/buy, sell/sell)"
```

---

### Task 3: Ideas service consumes the new leg shape

**Files:**
- Modify: `src/services/ideas_service.py:61-64` (the `legs` tuple in `assess_spread`)
- Test: `tests/test_ideas.py:21-34` (the `spread` fixture helper)

**Interfaces:**
- Consumes: the Task 2 result shape (`leg1`/`leg2` objects with `side`, `qty`, `detail`).
- Produces: `assess_spread` and `build_ideas` behavior unchanged; ideas passed through keep `structure`, `leg1`, `leg2` (spread dict is spread into the idea via `**spread`).

- [ ] **Step 1: Update the test fixture to the new shape**

In `tests/test_ideas.py`, the `spread()` helper currently builds `buyQty`/`sellQty`/`buyLeg`/`sellLeg` keys (lines ~21–34). Change those four keys to:

```python
        "structure": "buy_sell",
        "leg1": {"side": "buy", "contract": "T Dec26 100c", "qty": buy_qty, "detail": buy_leg or leg()},
        "leg2": {"side": "sell", "contract": "T Dec26 110c", "qty": sell_qty, "detail": sell_leg or leg()},
```

Keep the helper's parameter names (`buy_qty`, `sell_qty`, `buy_leg`, `sell_leg`) so call sites don't change.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ideas.py -v`
Expected: FAIL with `KeyError: 'buyLeg'` in `assess_spread`.

- [ ] **Step 3: Update assess_spread**

In `src/services/ideas_service.py`, replace:

```python
    legs = (
        ("buy", spread["buyLeg"], spread.get("buyQty") or 0),
        ("sell", spread["sellLeg"], spread.get("sellQty") or 0),
    )
```

with:

```python
    legs = tuple(
        (leg["side"], leg["detail"], leg.get("qty") or 0)
        for leg in (spread["leg1"], spread["leg2"])
    )
```

(The flag strings become e.g. "buy leg volume 10" for each leg by its side — same wording as today for buy/sell; for same-side structures both legs share a side name, which is fine.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ideas.py tests/test_services.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/ideas_service.py tests/test_ideas.py
git commit -m "feat: ideas service reads generalized spread legs"
```

---

### Task 4: Frontend types and SpreadsView

**Files:**
- Modify: `web/src/types.ts:111-127` (the `Spread` interface)
- Modify: `web/src/components/SpreadsView.tsx` (full component update)

**Interfaces:**
- Consumes: API result shape from Task 2.
- Produces: `SpreadLeg` and updated `Spread` types; `legLabel(leg: SpreadLeg): string` and `STRUCTURE_LABELS` exported from `SpreadsView.tsx` for Task 5's `IdeasView`.

- [ ] **Step 1: Update types**

In `web/src/types.ts`, replace the `Spread` interface with:

```ts
export interface SpreadLeg {
  side: "buy" | "sell"
  contract: string
  qty: number
  detail: LegDetail
}

export interface Spread {
  structure: "buy_sell" | "buy_buy" | "sell_sell"
  leg1: SpreadLeg
  leg2: SpreadLeg
  netDelta: number
  edge: number
  marginRequirement: number
  netDebit: number
  carryCost: number
  grossEdgeDollars: number
  netEdgeDollars: number
  execEdgeDollars: number | null
  capitalEmployed: number
}
```

(`Idea extends Spread` picks the new shape up automatically.)

- [ ] **Step 2: Update SpreadsView**

Replace the imports/column section and component of `web/src/components/SpreadsView.tsx` so the file becomes:

```tsx
import { useMemo, useState } from "react"
import { AgGridReact } from "ag-grid-react"
import type { ColDef, ValueFormatterParams } from "ag-grid-community"
import { useTheme } from "next-themes"
import { gridThemeDark, gridThemeLight } from "@/lib/ag-grid-setup"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { Spread, SpreadLeg } from "@/types"
import { LegCard } from "@/components/LegCard"

export const STRUCTURE_LABELS: Record<Spread["structure"], string> = {
  buy_sell: "Buy–Sell",
  buy_buy: "Buy–Buy",
  sell_sell: "Sell–Sell",
}

export function legLabel(leg: SpreadLeg) {
  return `${leg.side === "buy" ? "B" : "S"} ${leg.qty}× ${leg.contract}`
}

function fmtNum(digits = 2) {
  return (params: ValueFormatterParams) =>
    typeof params.value === "number" ? params.value.toFixed(digits) : "-"
}

function fmtDollars(params: ValueFormatterParams) {
  return typeof params.value === "number" ? dollars(params.value) : "-"
}

const columnDefs: ColDef<Spread>[] = [
  {
    field: "structure",
    headerName: "Structure",
    width: 110,
    valueFormatter: (p) => STRUCTURE_LABELS[p.value as Spread["structure"]] ?? "-",
  },
  {
    colId: "leg1",
    headerName: "Leg 1",
    flex: 1,
    minWidth: 170,
    valueGetter: (p) => (p.data ? legLabel(p.data.leg1) : ""),
  },
  {
    colId: "leg2",
    headerName: "Leg 2",
    flex: 1,
    minWidth: 170,
    valueGetter: (p) => (p.data ? legLabel(p.data.leg2) : ""),
  },
  { field: "netDelta", headerName: "Net Delta", valueFormatter: fmtNum(3), width: 100 },
  { field: "edge", headerName: "Edge", valueFormatter: fmtNum(4), width: 90 },
  { field: "marginRequirement", headerName: "Margin $", valueFormatter: fmtDollars, width: 110 },
  { field: "netDebit", headerName: "Net Debit $", valueFormatter: fmtDollars, width: 110 },
  { field: "carryCost", headerName: "Carry $", valueFormatter: fmtDollars, width: 100 },
  { field: "grossEdgeDollars", headerName: "Gross Edge $", valueFormatter: fmtDollars, width: 120 },
  { field: "netEdgeDollars", headerName: "Net Edge $", valueFormatter: fmtDollars, width: 120, sort: "desc" },
]

function dollars(value: number) {
  const rounded = Math.round(value)
  return rounded < 0 ? `-$${Math.abs(rounded).toLocaleString()}` : `$${rounded.toLocaleString()}`
}

function Field({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium">{value ?? "-"}</p>
    </div>
  )
}

const FILTERS = ["all", "buy_sell", "buy_buy", "sell_sell"] as const

export function SpreadsView({ spreads, loading }: { spreads: Spread[]; loading: boolean }) {
  const { resolvedTheme } = useTheme()
  const theme = resolvedTheme === "dark" ? gridThemeDark : gridThemeLight
  const [selected, setSelected] = useState<Spread | null>(null)
  const [structureFilter, setStructureFilter] = useState<(typeof FILTERS)[number]>("all")

  const filtered = useMemo(
    () => (structureFilter === "all" ? spreads : spreads.filter((s) => s.structure === structureFilter)),
    [spreads, structureFilter],
  )

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading spreads…</p>
  }

  if (!spreads.length) {
    return <p className="text-sm text-muted-foreground">No spreads matched the current filters.</p>
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-1">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setStructureFilter(f)}
            className={`rounded-md px-3 py-1 text-sm ${
              structureFilter === f
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:text-foreground"
            }`}
          >
            {f === "all" ? "All" : STRUCTURE_LABELS[f]}
          </button>
        ))}
      </div>

      <div style={{ height: Math.min(500, 42 + filtered.length * 36) }}>
        <AgGridReact
          theme={theme}
          rowData={filtered}
          columnDefs={columnDefs}
          defaultColDef={{ sortable: true, resizable: true }}
          rowSelection={{ mode: "singleRow" }}
          onRowClicked={(e) => setSelected(e.data ?? null)}
        />
      </div>

      {selected ? (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Position economics</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-5">
              <Field label="Margin requirement" value={dollars(selected.marginRequirement)} />
              <Field label="Net debit" value={dollars(selected.netDebit)} />
              <Field label="Carry cost" value={dollars(selected.carryCost)} />
              <Field label="Gross edge" value={dollars(selected.grossEdgeDollars)} />
              <Field label="Net edge" value={dollars(selected.netEdgeDollars)} />
            </CardContent>
          </Card>
          <div className="grid gap-4 lg:grid-cols-2">
            <LegCard
              title={`Leg 1 (${selected.leg1.side === "buy" ? "Buy" : "Sell"} ${selected.leg1.qty}×)`}
              leg={selected.leg1.detail}
            />
            <LegCard
              title={`Leg 2 (${selected.leg2.side === "buy" ? "Buy" : "Sell"} ${selected.leg2.qty}×)`}
              leg={selected.leg2.detail}
            />
          </div>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">Select a spread to view leg details.</p>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Typecheck (IdeasView errors expected)**

Run: `cd web && npx tsc -b 2>&1 | head -30`
Expected: errors ONLY in `IdeasView.tsx` (`buy`, `buyQty`, `buyLeg` no longer exist) — those are Task 5. No errors in `types.ts` or `SpreadsView.tsx`.

- [ ] **Step 4: Commit**

```bash
git add web/src/types.ts web/src/components/SpreadsView.tsx
git commit -m "feat: SpreadsView renders multi-structure legs with a structure filter"
```

---

### Task 5: IdeasView on the new leg shape

**Files:**
- Modify: `web/src/components/IdeasView.tsx`

**Interfaces:**
- Consumes: `legLabel`, `STRUCTURE_LABELS` from `@/components/SpreadsView`; new `Idea` shape (extends `Spread`).
- Produces: compiling frontend; no API changes.

- [ ] **Step 1: Update IdeasView**

In `web/src/components/IdeasView.tsx`:

Add to the imports:

```tsx
import { legLabel, STRUCTURE_LABELS } from "@/components/SpreadsView"
```

Replace the four leg columns (lines with `field: "buy"`, `"sell"`, `"buyQty"`, `"sellQty"`) with:

```tsx
  {
    field: "structure",
    headerName: "Structure",
    width: 110,
    valueFormatter: (p) => STRUCTURE_LABELS[p.value as Idea["structure"]] ?? "-",
  },
  {
    colId: "leg1",
    headerName: "Leg 1",
    flex: 1,
    minWidth: 170,
    valueGetter: (p) => (p.data ? legLabel(p.data.leg1) : ""),
  },
  {
    colId: "leg2",
    headerName: "Leg 2",
    flex: 1,
    minWidth: 170,
    valueGetter: (p) => (p.data ? legLabel(p.data.leg2) : ""),
  },
```

Replace the selected-card title contents (`{selected.ticker}: buy {selected.buyQty}× …`) with:

```tsx
              {selected.ticker}: {legLabel(selected.leg1)}, {legLabel(selected.leg2)}
```

Replace the two `LegCard` lines at the bottom with:

```tsx
          <LegCard
            title={`${selected.leg1.side === "buy" ? "Buy" : "Sell"} ${selected.leg1.qty}×`}
            leg={selected.leg1.detail}
          />
          <LegCard
            title={`${selected.leg2.side === "buy" ? "Buy" : "Sell"} ${selected.leg2.qty}×`}
            leg={selected.leg2.detail}
          />
```

Also update the explanatory paragraph ("Ideas pair same-sign deltas only…") to:

```tsx
        Top {ideas.length} spreads across the universe, ranked by confidence then{" "}
        <strong className="text-foreground">executable edge</strong> — the profit left after
        filling buys at the ask, sells at the bid, and paying carry. Every structure pairs legs
        whose position deltas offset, isolating relative volatility rather than adding call/put
        delta exposure. High confidence = the edge survives realistic fills on tight, traded
        markets, priced off the fitted surface.
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: clean, no errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/IdeasView.tsx
git commit -m "feat: IdeasView renders generalized spread legs"
```

---

### Task 6: Full verification

**Files:** none new — verification only.

- [ ] **Step 1: Full backend suite**

Run: `uv run pytest -q`
Expected: all PASS (test_services, test_ideas, test_margin, and the untouched suites).

- [ ] **Step 2: Frontend build**

Run: `cd web && npm run build`
Expected: `tsc -b` clean and vite build succeeds.

- [ ] **Step 3: Live check in the dev preview**

Start the `backend` and `frontend` launch configs (`.claude/launch.json`), sign in via the magic link printed to backend stdout (request one for `dev@ztrade.local` at `/api/auth/magic/request`), open the Spreads tab, set Min option price to 0, and verify:
- the grid shows a Structure column and the All / Buy–Sell / Buy–Buy / Sell–Sell filter buttons work,
- selecting a row shows Leg 1/Leg 2 cards titled with side and quantity,
- the Ideas tab renders without errors.

- [ ] **Step 4: Commit any fixes found**

```bash
git add <specific files touched>
git commit -m "fix: <what the live check surfaced>"
```
