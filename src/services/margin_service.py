from __future__ import annotations

from math import erf, exp, log, sqrt
from typing import TypedDict

CONTRACT_MULTIPLIER = 100

# TIMS-style portfolio margin parameters for individual equities.
PM_SHOCK = 0.15
PM_SCENARIOS = 11
PM_MIN_PER_SHORT_CONTRACT = 37.50


class MarginLeg(TypedDict):
    spot: float
    strike: float
    type: str  # "C" or "P"
    qty: int
    side: int  # +1 long, -1 short
    vol: float
    rate: float
    dividend: float
    tenor: float  # years


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    tenor: float,
    opt_type: str,
) -> float:
    """Black-Scholes-Merton European price, used to reprice legs under PM stress scenarios."""
    if tenor <= 0 or vol <= 0:
        intrinsic = spot - strike if opt_type == "C" else strike - spot
        return max(0.0, intrinsic)

    d1 = (log(spot / strike) + (rate - dividend + 0.5 * vol**2) * tenor) / (vol * sqrt(tenor))
    d2 = d1 - vol * sqrt(tenor)

    call = spot * exp(-dividend * tenor) * _norm_cdf(d1) - strike * exp(-rate * tenor) * _norm_cdf(d2)
    if opt_type == "C":
        return call
    # put via put-call parity
    return call - spot * exp(-dividend * tenor) + strike * exp(-rate * tenor)


def short_margin_requirement(spot: float, strike: float, opt_type: str, premium: float, qty: int) -> float:
    """Fidelity/Reg T naked short option requirement.

    Greater of (20% of underlying − OTM amount) or (10% of underlying for calls /
    10% of strike for puts), plus the premium, per contract.
    """
    if opt_type == "C":
        otm = max(0.0, strike - spot)
        floor = 0.10 * spot
    else:
        otm = max(0.0, spot - strike)
        floor = 0.10 * strike

    per_share = max(0.20 * spot - otm, floor) + premium
    return per_share * CONTRACT_MULTIPLIER * qty


def portfolio_margin_requirement(
    legs: list[MarginLeg],
    shock: float = PM_SHOCK,
    scenarios: int = PM_SCENARIOS,
    min_per_short_contract: float = PM_MIN_PER_SHORT_CONTRACT,
) -> float:
    """TIMS-style portfolio margin: worst-case loss of the combined position when the
    underlying is shocked across [−shock, +shock], with a per-short-contract minimum.

    Legs are repriced with Black-Scholes at each shocked spot (vol, rates, and time
    to expiry held constant), and the requirement is the largest mark-to-model loss
    relative to the unshocked value.
    """

    def position_value(spot_multiplier: float) -> float:
        total = 0.0
        for leg in legs:
            price = bs_price(
                spot=leg["spot"] * spot_multiplier,
                strike=leg["strike"],
                rate=leg["rate"],
                dividend=leg["dividend"],
                vol=leg["vol"],
                tenor=leg["tenor"],
                opt_type=leg["type"],
            )
            total += leg["side"] * leg["qty"] * CONTRACT_MULTIPLIER * price
        return total

    base_value = position_value(1.0)

    worst_loss = 0.0
    for i in range(scenarios):
        multiplier = 1.0 - shock + (2.0 * shock) * i / (scenarios - 1)
        worst_loss = max(worst_loss, base_value - position_value(multiplier))

    short_contracts = sum(leg["qty"] for leg in legs if leg["side"] < 0)
    return max(worst_loss, min_per_short_contract * short_contracts)
