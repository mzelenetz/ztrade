from __future__ import annotations

from math import erf, log, sqrt

import polars as pl


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def add_greeks(df: pl.DataFrame) -> pl.DataFrame:
    """Closed-form BSM greeks at the model vol (FittedVol). Delta is set by the
    pricing engines; gamma/vega/theta/rho come from here for both. Conventions:
    vega per 1 vol point, theta per calendar day, rho per 1% rate move."""
    from math import exp, log, pi, sqrt

    def greeks(row: dict) -> dict:
        S, K = row["Spot"], row["Strike"]
        r, q = row["Rate"], row["DividendYield"]
        sigma, t = row["FittedVol"], row["T"]
        if not all(v is not None for v in (S, K, r, q, sigma, t)) or min(S, K, sigma, t) <= 0:
            return {"Gamma": None, "Vega": None, "Theta": None, "Rho": None}

        sqt = sqrt(t)
        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqt)
        d2 = d1 - sigma * sqt
        pdf = exp(-0.5 * d1 * d1) / sqrt(2 * pi)
        nd1, nd2 = normal_cdf(d1), normal_cdf(d2)

        gamma = exp(-q * t) * pdf / (S * sigma * sqt)
        vega = S * exp(-q * t) * pdf * sqt / 100.0

        if row["Type"] == "C":
            theta_yr = (
                -S * exp(-q * t) * pdf * sigma / (2 * sqt)
                - r * K * exp(-r * t) * nd2
                + q * S * exp(-q * t) * nd1
            )
            rho = K * t * exp(-r * t) * nd2 / 100.0
        else:
            theta_yr = (
                -S * exp(-q * t) * pdf * sigma / (2 * sqt)
                + r * K * exp(-r * t) * normal_cdf(-d2)
                - q * S * exp(-q * t) * normal_cdf(-d1)
            )
            rho = -K * t * exp(-r * t) * normal_cdf(-d2) / 100.0

        return {"Gamma": gamma, "Vega": vega, "Theta": theta_yr / 365.0, "Rho": rho}

    out = df.with_columns(
        pl.struct(["Spot", "Strike", "Rate", "DividendYield", "FittedVol", "T", "Type"])
        .map_elements(
            greeks,
            return_dtype=pl.Struct(
                {"Gamma": pl.Float64, "Vega": pl.Float64, "Theta": pl.Float64, "Rho": pl.Float64}
            ),
        )
        .alias("_greeks")
    )
    return out.unnest("_greeks")


def add_probabilities(df: pl.DataFrame) -> pl.DataFrame:
    def prob_itm(row: dict) -> float | None:
        spot = float(row["Spot"])
        strike = float(row["Strike"])
        rate = float(row["Rate"])
        dividend = float(row["DividendYield"])
        vol = float(row["Vol30d"])
        tenor = float(row["T"])

        if spot <= 0 or strike <= 0 or vol <= 0 or tenor <= 0:
            return None

        d2 = (log(spot / strike) + (rate - dividend - 0.5 * vol**2) * tenor) / (
            vol * sqrt(tenor)
        )

        if row["Type"] == "C":
            return float(normal_cdf(d2))

        return float(normal_cdf(-d2))

    def prob_otm(row: dict) -> float | None:
        itm = prob_itm(row)
        if itm is None:
            return None
        return 1.0 - itm

    inputs = ["Spot", "Strike", "Rate", "DividendYield", "Vol30d", "T", "Type"]
    return df.with_columns(
        pl.struct(inputs).map_elements(prob_itm, return_dtype=pl.Float64).alias("Prob ITM"),
        pl.struct(inputs).map_elements(prob_otm, return_dtype=pl.Float64).alias("Prob OTM"),
    )


# Below this FMV the %Overvalued ratio is numerically meaningless (a stale $2
# quote against a near-zero model value reads as millions of percent).
MIN_FMV_FOR_OVERVALUED = 0.05


def compute_overvalued(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("FMV") >= MIN_FMV_FOR_OVERVALUED)
        .then((pl.col(metric) / pl.col("FMV")) - 1.0)
        .otherwise(None)
        .alias("%Overvalued")
    )


def bid_ask(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("Bid").cast(str) + " – " + pl.col("Ask").cast(str)).alias("BidAsk")
    )


def leg_summary(row: dict) -> dict:
    """Shape a single call/put row (as produced by bid_ask + compute_overvalued) for the API."""
    return {
        "fmv": row.get("FMV"),
        "last": row.get("Last"),
        "bid": row.get("Bid"),
        "ask": row.get("Ask"),
        "mid": row.get("Mid"),
        "bidAsk": row.get("BidAsk"),
        "overvalued": row.get("%Overvalued"),
        "delta": row.get("Delta"),
        "probItm": row.get("Prob ITM"),
        "probOtm": row.get("Prob OTM"),
        "marketIv": row.get("MarketIV"),
        "modelVol": row.get("FittedVol"),
        "volume": row.get("Volume"),
        "volFromSurface": row.get("VolFromSurface"),
        "openInterest": row.get("OpenInterest"),
        "bidSize": row.get("BidSize"),
        "askSize": row.get("AskSize"),
        "gamma": row.get("Gamma"),
        "vega": row.get("Vega"),
        "theta": row.get("Theta"),
        "rho": row.get("Rho"),
    }


def filter_by_delta_and_price(
    df: pl.DataFrame, delta_min: float, delta_max: float, min_price: float
) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.when(pl.col("Type") == "C")
            .then(pl.col("Delta"))
            .otherwise(-pl.col("Delta"))
            .alias("CallDelta")
        )
        .filter(
            (pl.col("CallDelta") >= delta_min / 100)
            & (pl.col("CallDelta") <= delta_max / 100)
            & (pl.col("Last") >= min_price)
        )
    )


def build_chain(
    df: pl.DataFrame,
    metric: str,
    delta_min: float,
    delta_max: float,
    min_price: float,
) -> dict:
    """Build an expiry-grouped, call/put-paired chain payload for a single ticker."""
    sdf = filter_by_delta_and_price(compute_overvalued(df, metric), delta_min, delta_max, min_price)

    expiries = []
    for (expiry,), group in sdf.sort("Expiry").group_by("Expiry", maintain_order=True):
        calls = bid_ask(group.filter(pl.col("Type") == "C")).sort("Strike")
        puts = bid_ask(group.filter(pl.col("Type") == "P")).sort("Strike")

        calls_by_strike = {row["Strike"]: row for row in calls.to_dicts()}
        puts_by_strike = {row["Strike"]: row for row in puts.to_dicts()}

        rows = [
            {
                "strike": strike,
                "call": leg_summary(calls_by_strike[strike]),
                "put": leg_summary(puts_by_strike[strike]),
            }
            for strike in sorted(set(calls_by_strike) & set(puts_by_strike))
        ]

        expiries.append({"expiry": expiry.strftime("%Y-%m-%d"), "rows": rows})

    return {"expiries": expiries}
