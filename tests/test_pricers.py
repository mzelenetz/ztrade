"""Pricing-engine correctness: QuantLib vs the closed-form Black-Scholes-Merton
formula computed independently here, put-call parity, and mzpricer vs QuantLib
(skipped when mzpricer isn't installed — it is only built in the Docker image)."""

from datetime import datetime
from math import erf, exp, log, sqrt
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import QuantLib as ql

from src.pricing_utils import OptionsPrices, ql_black_scholes_price_and_delta

NY = ZoneInfo("America/New_York")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bsm_reference(S, K, r, q, sigma, T, is_call):
    """Independent closed-form BSM price + delta for cross-checking the engines."""
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    call_delta = exp(-q * T) * norm_cdf(d1)
    if is_call:
        return call, call_delta
    put = call - S * exp(-q * T) + K * exp(-r * T)
    put_delta = call_delta - exp(-q * T)
    return put, put_delta


CASES = [
    # S, K, r, q, sigma, is_call
    (100.0, 100.0, 0.05, 0.0, 0.20, True),
    (100.0, 100.0, 0.05, 0.0, 0.20, False),
    (150.0, 120.0, 0.05, 0.02, 0.35, True),   # ITM call with dividend yield
    (150.0, 180.0, 0.03, 0.01, 0.25, False),  # ITM put
    (50.0, 65.0, 0.05, 0.0, 0.40, True),      # OTM call, high vol
]


@pytest.mark.parametrize("S,K,r,q,sigma,is_call", CASES)
def test_quantlib_matches_closed_form(S, K, r, q, sigma, is_call):
    valuation = ql.Date(1, 1, 2026)
    expiry = ql.Date(1, 1, 2027)  # 365 days → T = 1.0 under Actual365Fixed
    T = 1.0

    price, delta = ql_black_scholes_price_and_delta(
        S=S, K=K, r=r, q=q, sigma=sigma, valuation_dt=valuation, expiry_dt=expiry, is_call=is_call
    )
    ref_price, ref_delta = bsm_reference(S, K, r, q, sigma, T, is_call)

    assert price == pytest.approx(ref_price, abs=1e-8)
    assert delta == pytest.approx(ref_delta, abs=1e-8)


def test_quantlib_put_call_parity():
    valuation = ql.Date(1, 1, 2026)
    expiry = ql.Date(1, 1, 2027)
    S, K, r, q, sigma, T = 100.0, 105.0, 0.05, 0.01, 0.30, 1.0

    call, _ = ql_black_scholes_price_and_delta(S, K, r, q, sigma, valuation, expiry, is_call=True)
    put, _ = ql_black_scholes_price_and_delta(S, K, r, q, sigma, valuation, expiry, is_call=False)

    parity_rhs = S * exp(-q * T) - K * exp(-r * T)
    assert call - put == pytest.approx(parity_rhs, abs=1e-8)


def make_options_frame() -> pl.DataFrame:
    """A small frame in the shape OptionsPrices expects (post CBOEOptionsData)."""
    valuation = datetime(2026, 1, 2, 16, 0, tzinfo=NY)
    expiry = datetime(2027, 1, 2, 16, 0, tzinfo=NY)
    rows = [
        {"Type": "C", "Strike": 100.0},
        {"Type": "P", "Strike": 100.0},
        {"Type": "C", "Strike": 120.0},
        {"Type": "P", "Strike": 80.0},
    ]
    return pl.DataFrame(
        {
            "Ticker": ["TEST"] * len(rows),
            "ValuationTime": [valuation] * len(rows),
            "Expiry": [expiry] * len(rows),
            "Spot": [100.0] * len(rows),
            "Type": [r["Type"] for r in rows],
            "Strike": [r["Strike"] for r in rows],
            "Rate": [0.05] * len(rows),
            "DividendYield": [0.0] * len(rows),
            "Vol30d": [0.25] * len(rows),
            "T": [1.0] * len(rows),
            "Last": [10.0] * len(rows),
        }
    )


def test_options_prices_quantlib_pipeline_matches_closed_form():
    df = make_options_frame()
    out = OptionsPrices(df, model="quantlib").price_options()

    for row in out.to_dicts():
        ref_price, ref_delta = bsm_reference(
            S=row["Spot"], K=row["Strike"], r=row["Rate"], q=row["DividendYield"],
            sigma=row["Vol30d"], T=row["T"], is_call=row["Type"] == "C",
        )
        assert row["FMV"] == pytest.approx(ref_price, abs=1e-6)
        assert row["Delta"] == pytest.approx(ref_delta, abs=1e-6)
        assert row["%Overvalued"] == pytest.approx(row["Last"] / row["FMV"] - 1, abs=1e-12)


def test_mzpricer_agrees_with_quantlib():
    pytest.importorskip("mzpricer")

    df = make_options_frame()
    mz = OptionsPrices(df, model="mzpricer")._price_with_mzpricer()
    qlb = OptionsPrices(df, model="quantlib")._price_with_quantlib()

    for mz_row, ql_row in zip(mz.to_dicts(), qlb.to_dicts()):
        if mz_row["Type"] == "C":
            # With zero dividends an American call is never exercised early, so the
            # 500-step binomial must match the analytic European price closely.
            assert mz_row["FMV"] == pytest.approx(ql_row["FMV"], rel=0.01, abs=0.05)
            assert mz_row["Delta"] == pytest.approx(ql_row["Delta"], abs=0.02)
        else:
            # American puts carry an early-exercise premium over European ones,
            # bounded above by K·(1 − e^(−rT)).
            max_premium = mz_row["Strike"] * (1 - exp(-mz_row["Rate"] * mz_row["T"]))
            assert mz_row["FMV"] >= ql_row["FMV"] - 0.01
            assert mz_row["FMV"] <= ql_row["FMV"] + max_premium + 0.05
            assert mz_row["Delta"] == pytest.approx(ql_row["Delta"], abs=0.06)
