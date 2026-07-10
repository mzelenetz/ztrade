"""Model-input machinery: valuation date, rate curve, smile fit, dividends,
spot correction, and the edge-sanity regression on the sample file."""

from datetime import date, datetime
from math import exp, log, sqrt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.pricing_utils import CBOEOptionsData, OptionsPrices
from src.services.chain_service import add_probabilities
from src.services.model_inputs import (
    DEFAULT_RATE_CURVE,
    apply_model_inputs,
    fit_smile,
    interpolate_rate,
)

NY = ZoneInfo("America/New_York")


def make_raw(quote_date="2026-01-11", expiration="2026-06-19") -> pl.DataFrame:
    """Minimal CBOE-shaped raw frame for CBOEOptionsData."""
    rows = [
        ("C", 100.0, 12.0, 0.40),
        ("P", 100.0, 8.0, 0.42),
        ("C", 110.0, 7.0, 0.38),
        ("P", 90.0, 4.0, 0.45),
    ]
    n = len(rows)
    return pl.DataFrame(
        {
            "underlying_symbol": ["TST"] * n,
            "quote_date": [quote_date] * n,
            "expiration": [expiration] * n,
            "strike": [r[1] for r in rows],
            "option_type": [r[0] for r in rows],
            "close": [r[2] for r in rows],
            "bid_1545": [r[2] - 0.1 for r in rows],
            "ask_1545": [r[2] + 0.1 for r in rows],
            "underlying_bid_1545": [99.9] * n,
            "underlying_ask_1545": [100.1] * n,
            "implied_volatility_1545": [r[3] for r in rows],
        }
    )


class TestValuationDate:
    def test_derived_from_quote_date(self):
        df = CBOEOptionsData(dataframe=make_raw()).get_data()
        vt = df["ValuationTime"][0]
        assert vt.astimezone(NY).date() == date(2026, 1, 11)
        assert vt.astimezone(NY).hour == 16

    def test_explicit_as_of_overrides_quote_date(self):
        df = CBOEOptionsData(dataframe=make_raw(), as_of="2026-03-01").get_data()
        assert df["ValuationTime"][0].astimezone(NY).date() == date(2026, 3, 1)

    def test_expiry_filter_relative_to_quote_date(self):
        # Expiry between the quote date and today: must survive (it was alive
        # on the quote date), which fails if valuation defaults to "today".
        df = CBOEOptionsData(dataframe=make_raw(expiration="2026-02-20")).get_data()
        assert df.height > 0

    def test_tenor_measured_from_quote_date(self):
        df = CBOEOptionsData(dataframe=make_raw()).get_data()
        expected_t = (datetime(2026, 6, 19, 16, tzinfo=NY) - datetime(2026, 1, 11, 16, tzinfo=NY)).days / 365
        assert df["T"][0] == pytest.approx(expected_t, abs=0.01)


class TestRateCurve:
    CURVE = [(30, 0.04), (365, 0.05), (730, 0.03)]

    def test_exact_knots(self):
        assert interpolate_rate(self.CURVE, 30 / 365) == pytest.approx(0.04)
        assert interpolate_rate(self.CURVE, 1.0) == pytest.approx(0.05)

    def test_interior_linear(self):
        # halfway between 30d and 365d
        mid_days = (30 + 365) / 2
        assert interpolate_rate(self.CURVE, mid_days / 365) == pytest.approx(0.045)

    def test_flat_extrapolation(self):
        assert interpolate_rate(self.CURVE, 1 / 365) == pytest.approx(0.04)
        assert interpolate_rate(self.CURVE, 10.0) == pytest.approx(0.03)

    def test_empty_curve_raises(self):
        with pytest.raises(ValueError):
            interpolate_rate([], 1.0)


class TestFitSmile:
    def synthetic(self, a=0.30, b=-0.03, c=0.05, forward=100.0, tenor=0.25, n=20):
        scale_guess = a * sqrt(tenor)
        strikes, ivs, types = [], [], []
        for i in range(n):
            z = -3 + 6 * i / (n - 1)
            x = z * scale_guess
            k = forward * exp(x)
            strikes.append(k)
            ivs.append(a + b * z + c * z**2)
            types.append("P" if k <= forward else "C")
        return strikes, ivs, types, forward, tenor

    def test_recovers_synthetic_smile(self):
        strikes, ivs, types, f, t = self.synthetic()
        smile = fit_smile(strikes, ivs, types, f, t)
        assert smile is not None
        for k, iv in zip(strikes, ivs):
            assert smile(k, f) == pytest.approx(iv, abs=0.01)

    def test_too_few_points_returns_none(self):
        strikes, ivs, types, f, t = self.synthetic(n=4)
        assert fit_smile(strikes, ivs, types, f, t) is None

    def test_itm_quotes_ignored(self):
        # Garbage ITM-side IVs must not perturb the OTM fit.
        strikes, ivs, types, f, t = self.synthetic()
        strikes += [80.0, 120.0]
        ivs += [1.9, 1.9]  # junk, but "valid"
        types += ["C", "P"]  # ITM call below F / ITM put above F → excluded
        smile = fit_smile(strikes, ivs, types, f, t)
        assert smile(f, f) == pytest.approx(0.30, abs=0.02)

    def test_out_of_range_flagged(self):
        strikes, ivs, types, f, t = self.synthetic()
        smile = fit_smile(strikes, ivs, types, f, t)
        assert not smile.in_range(f * 3, f)
        assert smile.in_range(f, f)


class TestDividends:
    def base_df(self, q: float) -> pl.DataFrame:
        raw = make_raw()
        df = CBOEOptionsData(dataframe=raw).get_data()
        return apply_model_inputs(df, "flat", {"TST": q}, list(DEFAULT_RATE_CURVE), 0.25)

    def test_dividend_yield_applied_per_ticker(self):
        df = self.base_df(0.03)
        assert df["DividendYield"].unique().to_list() == [0.03]

    def test_quantlib_call_cheaper_put_richer_with_dividends(self):
        no_div = OptionsPrices(self.base_df(0.0), model="quantlib").price_options()
        with_div = OptionsPrices(self.base_df(0.04), model="quantlib").price_options()

        for r0, r1 in zip(no_div.to_dicts(), with_div.to_dicts()):
            if r0["Type"] == "C":
                assert r1["FMV"] < r0["FMV"]
            else:
                assert r1["FMV"] > r0["FMV"]

    def test_mzpricer_dividend_spot_adjustment(self):
        pytest.importorskip("mzpricer")
        # European-equivalent check on a call (no early exercise when q < r is
        # not guaranteed for American calls with dividends, so allow tolerance).
        df = self.base_df(0.04)
        mz = OptionsPrices(df, model="mzpricer")._price_with_mzpricer()
        qlb = OptionsPrices(df, model="quantlib")._price_with_quantlib()
        call_mz = [r for r in mz.to_dicts() if r["Type"] == "C"]
        call_ql = [r for r in qlb.to_dicts() if r["Type"] == "C"]
        for m, q in zip(call_mz, call_ql):
            assert m["FMV"] == pytest.approx(q["FMV"], rel=0.05, abs=0.1)


class TestSpotCorrection:
    def parity_frame(self, underlying: float) -> pl.DataFrame:
        """Three C/P strike pairs whose prices are parity-consistent with S≈100."""
        rows = [
            # type, strike, last — C−P+K ≈ 101 at every strike (r·T discount ≈ −1)
            ("C", 95.0, 9.0), ("P", 95.0, 3.0),
            ("C", 100.0, 6.0), ("P", 100.0, 5.0),
            ("C", 105.0, 3.5), ("P", 105.0, 7.5),
        ]
        n = len(rows)
        return pl.DataFrame(
            {
                "underlying_symbol": ["TST"] * n,
                "quote_date": ["2026-01-11"] * n,
                "expiration": ["2026-06-19"] * n,
                "strike": [r[1] for r in rows],
                "option_type": [r[0] for r in rows],
                "close": [r[2] for r in rows],
                "bid_1545": [r[2] - 0.1 for r in rows],
                "ask_1545": [r[2] + 0.1 for r in rows],
                "underlying_bid_1545": [underlying - 0.1] * n,
                "underlying_ask_1545": [underlying + 0.1] * n,
                "implied_volatility_1545": [0.3] * n,
            }
        )

    def test_bad_underlying_quote_overridden_by_parity(self):
        # Options price a ~100 spot; the quoted underlying says 130 → override.
        df = CBOEOptionsData(dataframe=self.parity_frame(130.0)).get_data()
        assert df["Spot"][0] == pytest.approx(100.0, abs=2.0)

    def test_good_quote_left_alone(self):
        # Quote agrees with parity within 2% → untouched.
        df = CBOEOptionsData(dataframe=self.parity_frame(100.0)).get_data()
        assert df["Spot"][0] == pytest.approx(100.0, abs=0.2)

    def test_incoherent_parity_never_overrides(self):
        # Corrupt option data: the three strikes imply wildly different spots
        # (C−P+K of ~101, ~140, ~60). No coherent implied spot exists — the
        # quoted underlying must be left alone, however wrong it may be.
        raw = self.parity_frame(100.0)
        raw = raw.with_columns(
            pl.Series("close", [9.0, 3.0, 45.0, 5.0, 3.5, 48.5])  # scrambled prices
        )
        df = CBOEOptionsData(dataframe=raw).get_data()
        assert df["Spot"][0] == pytest.approx(100.0, abs=0.2)


class TestImpliedDividend:
    def synthetic_chain(self, r=0.05, q=0.02, spot=100.0, tenor=0.5):
        """European mid prices generated at known r, q — parity holds exactly."""
        from src.services.margin_service import bs_price

        strikes, types, mids = [], [], []
        for k in [85, 90, 95, 100, 105, 110, 115]:
            for t in ("C", "P"):
                strikes.append(float(k))
                types.append(t)
                mids.append(bs_price(spot, float(k), r, q, 0.30, tenor, t))
        return strikes, types, mids, spot, tenor

    def test_recovers_known_dividend_yield(self):
        from src.services.model_inputs import fit_implied_dividend

        strikes, types, mids, spot, tenor = self.synthetic_chain(r=0.05, q=0.02)
        got = fit_implied_dividend(strikes, types, mids, spot, tenor, rate=0.05)
        assert got == pytest.approx(0.02, abs=1e-4)

    def test_negative_yield_recovered(self):
        from src.services.model_inputs import fit_implied_dividend

        # Hard-to-borrow name: forward below carry → negative implied q.
        strikes, types, mids, spot, tenor = self.synthetic_chain(r=0.05, q=-0.03)
        got = fit_implied_dividend(strikes, types, mids, spot, tenor, rate=0.05)
        assert got == pytest.approx(-0.03, abs=1e-4)

    def test_too_few_pairs_returns_none(self):
        from src.services.model_inputs import fit_implied_dividend

        strikes, types, mids, spot, tenor = self.synthetic_chain()
        assert fit_implied_dividend(strikes[:4], types[:4], mids[:4], spot, tenor, 0.05) is None

    def test_out_of_band_returns_none(self):
        from src.services.model_inputs import fit_implied_dividend

        strikes, types, mids, spot, tenor = self.synthetic_chain(q=0.30)  # absurd yield
        assert fit_implied_dividend(strikes, types, mids, spot, tenor, 0.05) is None

    def test_apply_model_inputs_implied_mode(self):
        raw = make_raw()
        df = CBOEOptionsData(dataframe=raw).get_data()
        implied = apply_model_inputs(
            df, "flat", {"TST": 0.07}, list(DEFAULT_RATE_CURVE), 0.25, carry_mode="implied"
        )
        manual = apply_model_inputs(
            df, "flat", {"TST": 0.07}, list(DEFAULT_RATE_CURVE), 0.25, carry_mode="manual"
        )
        # make_raw has < MIN_PARITY_PAIRS shared strikes → implied falls back to manual
        assert implied["DividendYield"].to_list() == manual["DividendYield"].to_list()
        assert implied["DivSource"].unique().to_list() == ["manual"]
        assert manual["DividendYield"].unique().to_list() == [0.07]


class TestDividendSchedule:
    def test_dividend_pv_boundaries_and_discounting(self):
        from datetime import date
        from math import exp

        from src.services.model_inputs import dividend_pv

        valuation = date(2026, 1, 11)
        expiry = date(2026, 6, 19)
        schedule = [
            ("2026-01-05", 1.00),  # before valuation → excluded
            ("2026-02-09", 0.26),  # inside → included, discounted
            ("2026-06-19", 0.26),  # ex-date == expiry → included
            ("2026-07-01", 0.26),  # after expiry → excluded
        ]
        rate = 0.05
        expected = 0.26 * exp(-rate * 29 / 365) + 0.26 * exp(-rate * 159 / 365)
        assert dividend_pv(schedule, valuation, expiry, rate) == pytest.approx(expected)
        assert dividend_pv([], valuation, expiry, rate) == 0.0

    def test_escrowed_equivalent_yield_reproduces_escrowed_price(self):
        from math import exp

        from src.services.margin_service import bs_price
        from src.services.model_inputs import escrowed_equivalent_yield

        S, K, r, vol, T, pv = 100.0, 100.0, 0.05, 0.30, 0.5, 1.30
        q_eff = escrowed_equivalent_yield(pv, S, T)
        # q_eff pricing must equal pricing off the dividend-reduced spot exactly
        via_yield = bs_price(S, K, r, q_eff, vol, T, "C")
        via_escrow = bs_price(S - pv, K, r, 0.0, vol, T, "C")
        assert via_yield == pytest.approx(via_escrow, abs=1e-10)
        assert S * exp(-q_eff * T) == pytest.approx(S - pv, abs=1e-10)

    def test_equivalent_yield_guards(self):
        from src.services.model_inputs import escrowed_equivalent_yield

        assert escrowed_equivalent_yield(0.0, 100.0, 0.5) == 0.0
        assert escrowed_equivalent_yield(95.0, 100.0, 0.5) is None  # PV ≥ 90% of spot
        assert escrowed_equivalent_yield(1.0, 100.0, 0.0) is None

    def test_schedule_priority_and_expiry_dependence(self):
        # Two expiries; dividend ex-date falls between them → only the later
        # expiry gets an adjustment, and its DivSource is "schedule".
        rows_feb = make_raw(expiration="2026-02-06")
        rows_jun = make_raw(expiration="2026-06-19")
        import polars as pl_mod

        raw = pl_mod.concat([rows_feb, rows_jun])
        df = CBOEOptionsData(dataframe=raw).get_data()

        out = apply_model_inputs(
            df, "flat", {"TST": 0.07}, list(DEFAULT_RATE_CURVE), 0.25,
            carry_mode="manual",
            dividend_schedule={"TST": [("2026-03-01", 2.00)]},
        )
        feb = out.filter(pl.col("T") < 0.2)
        jun = out.filter(pl.col("T") > 0.2)
        assert feb["DivSource"].unique().to_list() == ["schedule"]
        assert feb["DividendYield"].unique().to_list() == [0.0]  # no div before Feb expiry
        assert jun["DivSource"].unique().to_list() == ["schedule"]
        assert jun["DividendYield"][0] > 0.03  # $2 on a ~$100 spot over ~0.44y

    def test_unscheduled_ticker_keeps_manual(self):
        df = CBOEOptionsData(dataframe=make_raw()).get_data()
        out = apply_model_inputs(
            df, "flat", {"TST": 0.07}, list(DEFAULT_RATE_CURVE), 0.25,
            carry_mode="manual",
            dividend_schedule={"OTHER": [("2026-03-01", 1.00)]},
        )
        assert out["DivSource"].unique().to_list() == ["manual"]
        assert out["DividendYield"].unique().to_list() == [0.07]


class TestImpliedVolRecompute:
    def test_round_trip(self):
        from src.services.margin_service import bs_price
        from src.services.model_inputs import implied_vol_from_price

        for vol in (0.15, 0.40, 0.90):
            for typ, k in (("C", 110.0), ("P", 90.0), ("C", 100.0)):
                price = bs_price(100.0, k, 0.04, 0.01, vol, 0.7, typ)
                got = implied_vol_from_price(price, 100.0, k, 0.04, 0.01, 0.7, typ)
                assert got == pytest.approx(vol, abs=1e-4)

    def test_junk_prices_return_none(self):
        from src.services.model_inputs import implied_vol_from_price

        # ITM call priced below its carry-adjusted intrinsic floor → no solution
        assert implied_vol_from_price(40.0, 100, 50, 0.04, 0.0, 0.5, "C") is None
        # above any vol's value → no solution
        assert implied_vol_from_price(500.0, 100, 100, 0.04, 0.0, 0.5, "C") is None
        assert implied_vol_from_price(None, 100, 100, 0.04, 0.0, 0.5, "C") is None

    def test_same_strike_call_put_align_under_true_carry(self):
        """Prices generated at one vol with carry (r*, q*): inverting with the
        SAME carry recovers identical C/P IVs; inverting with the wrong carry
        splits them — the vendor-IV pathology this fix removes."""
        from src.services.margin_service import bs_price
        from src.services.model_inputs import implied_vol_from_price

        S, K, r_true, q_true, vol, T = 100.0, 100.0, 0.04, 0.03, 0.35, 1.9
        call = bs_price(S, K, r_true, q_true, vol, T, "C")
        put = bs_price(S, K, r_true, q_true, vol, T, "P")

        iv_c = implied_vol_from_price(call, S, K, r_true, q_true, T, "C")
        iv_p = implied_vol_from_price(put, S, K, r_true, q_true, T, "P")
        assert abs(iv_c - iv_p) < 1e-4

        iv_c_bad = implied_vol_from_price(call, S, K, r_true, 0.0, T, "C")
        iv_p_bad = implied_vol_from_price(put, S, K, r_true, 0.0, T, "P")
        assert abs(iv_c_bad - iv_p_bad) > 0.03  # wrong carry → split branches


class TestGlobalVolSurface:
    """The surface is fit jointly across expiries (partial pooling): dense
    expiries keep their own smile; sparse long-dated ones are shrunk toward
    the pooled shape instead of chasing a few noisy quotes."""

    def surface_truth(self, tenor):
        from math import log
        # smoothly evolving skew/curvature in ln-tenor
        return -0.10 - 0.02 * log(tenor), 0.04 + 0.005 * log(tenor)

    def make_points(self, tenor, n, noise, rng):
        b, c = self.surface_truth(tenor)
        pts = []
        for i in range(n):
            z = -2.5 + 5.0 * i / (n - 1)
            y = b * z + c * z * z + rng.gauss(0, noise)
            pts.append((z, y))
        return pts

    def test_sparse_expiry_shrinks_toward_surface(self):
        import random

        from src.services.model_inputs import blend_expiry_smile, fit_vol_surface

        rng = random.Random(7)
        dense = [
            {"tenor": t, "points": self.make_points(t, 40, 0.01, rng)}
            for t in (0.1, 0.25, 0.5, 1.0)
        ]
        sparse_tenor = 2.5
        sparse_points = self.make_points(sparse_tenor, 5, 0.08, rng)  # few + noisy

        surface = fit_vol_surface(dense + [{"tenor": sparse_tenor, "points": sparse_points}])
        assert surface is not None

        from math import log

        s = log(sparse_tenor)
        prior_b = surface.b0 + surface.b1 * s
        prior_c = max(surface.c0 + surface.c1 * s, 0.0)
        _, b_blend, c_blend = blend_expiry_smile(sparse_points, prior_b, prior_c)

        import numpy as np

        # unpooled fit on the same 5 noisy points
        c_solo, b_solo, _ = np.polyfit(
            [z for z, _ in sparse_points], [y for _, y in sparse_points], 2
        )

        b_true, c_true = self.surface_truth(sparse_tenor)
        assert abs(b_blend - b_true) < abs(b_solo - b_true)
        assert abs(b_blend - b_true) < 0.03

    def test_dense_expiry_keeps_own_shape(self):
        import random

        from src.services.model_inputs import blend_expiry_smile

        rng = random.Random(3)
        tenor = 0.5
        pts = self.make_points(tenor, 60, 0.005, rng)
        # a deliberately wrong prior must barely bend a well-observed expiry:
        # the pull toward the prior must stay under 10% of the prior-truth gap
        prior_b, prior_c = +0.30, 0.50
        _, b, c = blend_expiry_smile(pts, prior_b=prior_b, prior_c=prior_c)
        b_true, c_true = self.surface_truth(tenor)
        assert abs(b - b_true) < 0.1 * abs(prior_b - b_true)
        assert abs(c - c_true) < 0.1 * abs(prior_c - c_true)

    def test_full_pipeline_zero_bias_per_expiry(self):
        """On real close data, the blended surface fit must be unbiased per
        expiry over the OTM quotes. (Uses a production-file fixture — the
        bundled sample_NVDA.csv has date-shifted, internally inconsistent
        quotes that the hygiene filters gut near the money.)"""
        from math import exp as _exp

        raw = pl.read_csv("tests/fixtures/closes-nvda-2026-01-29.csv")
        base = CBOEOptionsData(dataframe=raw, default_vol=0.25, use_remote_vol=False).get_data()
        df = apply_model_inputs(base, "surface", {}, list(DEFAULT_RATE_CURVE), 0.25)

        nv = df.filter((pl.col("Ticker") == "NVDA") & pl.col("MarketIV").is_not_null())
        for (_,), g in nv.group_by(["Expiry"]):
            first = g.row(0, named=True)
            fwd = float(first["Spot"]) * _exp((first["Rate"] - first["DividendYield"]) * first["T"])
            otm = g.filter(
                (((pl.col("Type") == "P") & (pl.col("Strike") <= fwd))
                 | ((pl.col("Type") == "C") & (pl.col("Strike") >= fwd)))
                & pl.col("VolFromSurface")
                & (pl.col("MarketIV") > 0.01)
                & (pl.col("MarketIV") < 2)
            )
            if otm.height < 10:
                continue
            # median residual: robust to junk quotes the fitter itself rejected
            # (|y| > 2 relative-vol cut and the 3σ outlier pass)
            import statistics

            bias = statistics.median(
                fv - miv for fv, miv in zip(otm["FittedVol"], otm["MarketIV"])
            )
            assert abs(bias) < 0.02, f"expiry fit biased by {bias:+.4f}"


class TestEdgeSanityRegression:
    def test_sample_file_edges_are_sane_with_surface(self):
        """Full pipeline on the bundled sample: median |%Overvalued| must stay
        small. Guards the valuation-date fix, the surface fit, and data hygiene —
        this metric was ~17x before those fixes."""
        raw = pl.read_csv("src/data/sample_NVDA.csv")
        loader = CBOEOptionsData(dataframe=raw, default_vol=0.25, use_remote_vol=False)
        df = apply_model_inputs(loader.get_data(), "surface", {}, list(DEFAULT_RATE_CURVE), 0.25)
        priced = add_probabilities(OptionsPrices(df, model="quantlib").price_options())

        liquid = priced.filter(pl.col("Last") >= 2.0)
        overs = [abs(x) for x in liquid["%Overvalued"].to_list() if x is not None]
        assert len(overs) > 500
        median = sorted(overs)[len(overs) // 2]
        assert median < 0.25, f"median |%Overvalued| {median:.3f} — edge model regressed"
