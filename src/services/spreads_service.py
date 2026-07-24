from __future__ import annotations

import polars as pl

from src.services.chain_service import compute_overvalued, filter_by_delta_and_price, leg_summary


from src.services.margin_service import (
    CONTRACT_MULTIPLIER,
    MarginLeg,
    portfolio_margin_requirement,
    short_margin_requirement,
    short_pair_margin_requirement,
)


def format_contract(ticker, expiry, strike, opt_type) -> str:
    expiry_label = expiry.strftime("%b%y")
    opt_label = "c" if opt_type == "C" else "p"
    return f"{ticker} {expiry_label} {strike:g}{opt_label}"


def _leg_detail(row: dict) -> dict:
    summary = leg_summary(row)
    summary["bidAsk"] = f"{row['Bid']} – {row['Ask']}"
    return {
        "ticker": row["Ticker"],
        "expiry": row["Expiry"].strftime("%Y-%m-%d"),
        "strike": row["Strike"],
        "type": row["Type"],
        **summary,
    }


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
