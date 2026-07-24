"""Chain/spread service math: probabilities, overvaluation, chain pairing,
spread sizing, edge, and the carry/net-edge identities."""

from datetime import datetime
from math import erf, log, sqrt

import polars as pl
import pytest

from src.services.chain_service import (
    add_probabilities,
    bid_ask,
    build_chain,
    compute_overvalued,
    normal_cdf,
)
from src.services.spreads_service import build_spreads, format_contract


def norm_cdf_ref(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


class TestProbabilities:
    def make_df(self, opt_type: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Spot": [100.0],
                "Strike": [90.0],
                "Rate": [0.05],
                "DividendYield": [0.0],
                "Vol30d": [0.20],
                "T": [0.5],
                "Type": [opt_type],
            }
        )

    def d2(self):
        return (log(100.0 / 90.0) + (0.05 - 0.0 - 0.5 * 0.2**2) * 0.5) / (0.2 * sqrt(0.5))

    def test_call_prob_itm_is_n_d2(self):
        row = add_probabilities(self.make_df("C")).to_dicts()[0]
        assert row["Prob ITM"] == pytest.approx(norm_cdf_ref(self.d2()), abs=1e-12)

    def test_put_prob_itm_is_n_minus_d2(self):
        row = add_probabilities(self.make_df("P")).to_dicts()[0]
        assert row["Prob ITM"] == pytest.approx(norm_cdf_ref(-self.d2()), abs=1e-12)

    def test_itm_and_otm_sum_to_one(self):
        for t in ("C", "P"):
            row = add_probabilities(self.make_df(t)).to_dicts()[0]
            assert row["Prob ITM"] + row["Prob OTM"] == pytest.approx(1.0, abs=1e-12)

    def test_invalid_inputs_yield_null(self):
        df = self.make_df("C").with_columns(pl.lit(0.0).alias("Vol30d"))
        row = add_probabilities(df).to_dicts()[0]
        assert row["Prob ITM"] is None and row["Prob OTM"] is None

    def test_normal_cdf_known_values(self):
        assert normal_cdf(0.0) == pytest.approx(0.5)
        assert normal_cdf(1.6448536269514722) == pytest.approx(0.95, abs=1e-9)


class TestOvervaluedAndBidAsk:
    def test_compute_overvalued(self):
        df = pl.DataFrame({"Last": [12.0], "FMV": [10.0]})
        row = compute_overvalued(df, "Last").to_dicts()[0]
        assert row["%Overvalued"] == pytest.approx(0.2)

    def test_bid_ask_string(self):
        df = pl.DataFrame({"Bid": [1.5], "Ask": [2.5]})
        assert bid_ask(df).to_dicts()[0]["BidAsk"] == "1.5 – 2.5"


class TestBuildChain:
    def make_df(self) -> pl.DataFrame:
        expiry = datetime(2026, 12, 18, 16, 0)
        rows = [
            # strike 100 has both a call and a put; 110 has only a call.
            ("C", 100.0, 0.6), ("P", 100.0, -0.4), ("C", 110.0, 0.3),
        ]
        n = len(rows)
        return pl.DataFrame(
            {
                "Ticker": ["T"] * n,
                "Expiry": [expiry] * n,
                "Type": [r[0] for r in rows],
                "Strike": [r[1] for r in rows],
                "Delta": [r[2] for r in rows],
                "Last": [5.0] * n,
                "FMV": [4.0] * n,
                "Bid": [4.9] * n,
                "Ask": [5.1] * n,
                "Mid": [5.0] * n,
            }
        )

    def test_only_paired_strikes_survive(self):
        chain = build_chain(self.make_df(), "Last", delta_min=0, delta_max=100, min_price=0)
        assert len(chain["expiries"]) == 1
        rows = chain["expiries"][0]["rows"]
        assert [r["strike"] for r in rows] == [100.0]
        assert rows[0]["call"]["overvalued"] == pytest.approx(0.25)
        assert rows[0]["put"]["overvalued"] == pytest.approx(0.25)

    def test_delta_band_filters(self):
        # Delta band 50-100 keeps only the 0.6-delta call; its put pair is dropped
        # (put CallDelta = -(-0.4) = 0.4 < 0.5) so no paired row remains.
        chain = build_chain(self.make_df(), "Last", delta_min=50, delta_max=100, min_price=0)
        assert chain["expiries"][0]["rows"] == []


def spreads_df() -> pl.DataFrame:
    """Two calls, same expiry: A is cheap (buy), B is rich (sell)."""
    expiry = datetime(2026, 12, 18, 16, 0)
    rows = [
        # name, strike, delta, last, fmv
        ("A", 100.0, 0.50, 10.0, 20.0),  # 50% undervalued
        ("B", 110.0, 0.25, 20.0, 10.0),  # 100% overvalued
    ]
    n = len(rows)
    return pl.DataFrame(
        {
            "Ticker": ["T"] * n,
            "Expiry": [expiry] * n,
            "Type": ["C"] * n,
            "Strike": [r[1] for r in rows],
            "Delta": [r[2] for r in rows],
            "Last": [r[3] for r in rows],
            "FMV": [r[4] for r in rows],
            "Bid": [r[3] - 0.1 for r in rows],
            "Ask": [r[3] + 0.1 for r in rows],
            "Mid": [r[3] for r in rows],
            "Spot": [100.0] * n,
            "Vol30d": [0.25] * n,
            "Rate": [0.05] * n,
            "DividendYield": [0.0] * n,
            "T": [0.5] * n,
        }
    )


def run_build_spreads(margin_rate=0.10, margin_style="reg_t", **overrides):
    kwargs = dict(
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
        margin_rate=margin_rate,
        margin_style=margin_style,
    )
    kwargs.update(overrides)
    return build_spreads(spreads_df(), **kwargs)


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


class TestGreeks:
    def test_matches_finite_differences(self):
        import polars as pl_mod

        from src.services.chain_service import add_greeks
        from src.services.margin_service import bs_price

        S, K, r, q, vol, T = 100.0, 105.0, 0.04, 0.01, 0.35, 0.6
        for typ in ("C", "P"):
            df = pl_mod.DataFrame(
                {
                    "Spot": [S], "Strike": [K], "Rate": [r], "DividendYield": [q],
                    "FittedVol": [vol], "T": [T], "Type": [typ],
                }
            )
            g = add_greeks(df).to_dicts()[0]

            h = 0.01
            def price(s=S, sig=vol, t=T, rr=r):
                return bs_price(s, K, rr, q, sig, t, typ)

            gamma_fd = (price(S + h) - 2 * price(S) + price(S - h)) / h**2
            vega_fd = (price(sig=vol + 0.0001) - price(sig=vol - 0.0001)) / 0.0002 / 100
            theta_fd = (price(t=T - 1 / 365) - price(t=T)) / 1.0  # per day, decay
            rho_fd = (price(rr=r + 0.0001) - price(rr=r - 0.0001)) / 0.0002 / 100

            assert g["Gamma"] == pytest.approx(gamma_fd, rel=1e-3)
            assert g["Vega"] == pytest.approx(vega_fd, rel=1e-3)
            assert g["Theta"] == pytest.approx(theta_fd, rel=2e-2)
            assert g["Rho"] == pytest.approx(rho_fd, rel=1e-3)

    def test_bad_inputs_yield_nulls(self):
        import polars as pl_mod

        from src.services.chain_service import add_greeks

        df = pl_mod.DataFrame(
            {
                "Spot": [100.0], "Strike": [100.0], "Rate": [0.04], "DividendYield": [0.0],
                "FittedVol": [None], "T": [0.5], "Type": ["C"],
            }
        )
        g = add_greeks(df).to_dicts()[0]
        assert g["Gamma"] is None and g["Vega"] is None


def test_format_contract():
    expiry = datetime(2026, 12, 18)
    assert format_contract("NVDA", expiry, 110.0, "C") == "NVDA Dec26 110c"
    assert format_contract("AAPL", expiry, 95.5, "P") == "AAPL Dec26 95.5p"
