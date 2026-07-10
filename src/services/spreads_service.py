from __future__ import annotations

import polars as pl

from src.services.chain_service import compute_overvalued, filter_by_delta_and_price, leg_summary


from src.services.margin_service import (
    CONTRACT_MULTIPLIER,
    MarginLeg,
    portfolio_margin_requirement,
    short_margin_requirement,
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
    )

    if sdf.is_empty():
        return []

    sorted_legs = sdf.sort("%Overvalued")
    buys = sorted_legs.head(max_legs_per_side)
    sells = sorted_legs.tail(max_legs_per_side)

    spreads = []
    base_qty = 10  # anchor quantity on one side

    for buy in buys.to_dicts():
        buy_delta = float(buy["Delta"])
        if buy_delta == 0:
            continue

        for sell in sells.to_dicts():
            if (
                buy["Expiry"] == sell["Expiry"]
                and buy["Strike"] == sell["Strike"]
                and buy["Type"] == sell["Type"]
            ):
                continue

            sell_delta = float(sell["Delta"])
            if sell_delta == 0:
                continue

            def maybe_add_spread(anchor_buy: bool):
                if anchor_buy:
                    buy_qty = base_qty
                    sell_qty = max(1, int(round(abs(buy_delta) * buy_qty / abs(sell_delta))))
                else:
                    sell_qty = base_qty
                    buy_qty = max(1, int(round(abs(sell_delta) * sell_qty / abs(buy_delta))))

                if sell_qty <= 0 or buy_qty <= 0:
                    return

                contract_ratio = max(buy_qty, sell_qty) / min(buy_qty, sell_qty)
                if contract_ratio > max_contract_ratio:
                    return

                is_straddle = (
                    buy["Expiry"] == sell["Expiry"]
                    and buy["Strike"] == sell["Strike"]
                    and buy["Type"] != sell["Type"]
                )
                if is_straddle and contract_ratio > max_straddle_ratio:
                    return

                net_delta = buy_delta * buy_qty - sell_delta * sell_qty
                if abs(net_delta) > max_abs_net_delta:
                    return

                edge = float(sell["%Overvalued"]) - float(buy["%Overvalued"])
                if edge <= 0:
                    return

                sell_premium = float(sell["Last"]) * CONTRACT_MULTIPLIER * sell_qty
                buy_premium = float(buy["Last"]) * CONTRACT_MULTIPLIER * buy_qty
                net_debit = buy_premium - sell_premium

                if margin_style == "portfolio":
                    legs: list[MarginLeg] = [
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
                        for leg, qty, side in ((buy, buy_qty, 1), (sell, sell_qty, -1))
                    ]
                    margin_requirement = portfolio_margin_requirement(legs)
                    # PM requirement is a pure worst-case loss (no premium component).
                    capital = margin_requirement + max(net_debit, 0.0)
                else:
                    margin_requirement = short_margin_requirement(
                        spot=float(sell["Spot"]),
                        strike=float(sell["Strike"]),
                        opt_type=sell["Type"],
                        premium=float(sell["Last"]),
                        qty=sell_qty,
                    )
                    # The premium-received component of the Reg T requirement is
                    # self-funding, and a net credit doesn't reduce the margin held.
                    capital = (margin_requirement - sell_premium) + max(net_debit, 0.0)

                holding_years = min(float(buy["T"]), float(sell["T"]))
                carry_cost = capital * margin_rate * holding_years

                gross_edge_dollars = (
                    (float(sell["Last"]) - float(sell["FMV"])) * CONTRACT_MULTIPLIER * sell_qty
                    + (float(buy["FMV"]) - float(buy["Last"])) * CONTRACT_MULTIPLIER * buy_qty
                )
                net_edge_dollars = gross_edge_dollars - carry_cost

                spreads.append(
                    {
                        "buy": format_contract(buy["Ticker"], buy["Expiry"], buy["Strike"], buy["Type"]),
                        "sell": format_contract(sell["Ticker"], sell["Expiry"], sell["Strike"], sell["Type"]),
                        "buyQty": buy_qty,
                        "sellQty": sell_qty,
                        "netDelta": net_delta,
                        "edge": edge,
                        "marginRequirement": margin_requirement,
                        "netDebit": net_debit,
                        "carryCost": carry_cost,
                        "grossEdgeDollars": gross_edge_dollars,
                        "netEdgeDollars": net_edge_dollars,
                        "buyLeg": _leg_detail(buy),
                        "sellLeg": _leg_detail(sell),
                    }
                )

            maybe_add_spread(anchor_buy=True)
            maybe_add_spread(anchor_buy=False)

    return sorted(spreads, key=lambda r: r["netEdgeDollars"], reverse=True)[:max_results]
