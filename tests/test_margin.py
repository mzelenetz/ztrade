"""Margin math: Reg T naked-short requirement and TIMS-style portfolio margin."""

import pytest

from src.services.margin_service import (
    CONTRACT_MULTIPLIER,
    PM_MIN_PER_SHORT_CONTRACT,
    bs_price,
    portfolio_margin_requirement,
    short_margin_requirement,
)


class TestRegT:
    def test_itm_call_20_percent_branch(self):
        # S=171.34, K=165 (ITM → OTM amount 0): 20% branch wins over the 10% floor.
        got = short_margin_requirement(spot=171.34, strike=165, opt_type="C", premium=35.69, qty=10)
        expected = (0.20 * 171.34 + 35.69) * 100 * 10
        assert got == pytest.approx(expected)

    def test_deep_otm_put_hits_10_percent_of_strike_floor(self):
        # S=171.34, K=120: OTM = 51.34, 20% branch goes negative → floor 10% of strike.
        got = short_margin_requirement(spot=171.34, strike=120, opt_type="P", premium=1.50, qty=5)
        expected = (0.10 * 120 + 1.50) * 100 * 5
        assert got == pytest.approx(expected)

    def test_slightly_otm_call_20_percent_branch_wins(self):
        # S=100, K=105: 20%·100 − 5 = 15 beats the 10%·100 = 10 floor.
        got = short_margin_requirement(spot=100, strike=105, opt_type="C", premium=2, qty=1)
        assert got == pytest.approx((15 + 2) * 100)

    def test_otm_call_10_percent_of_underlying_floor(self):
        # S=100, K=140: 20%·100 − 40 < 0 → floor 10% of underlying.
        got = short_margin_requirement(spot=100, strike=140, opt_type="C", premium=0.50, qty=2)
        assert got == pytest.approx((10 + 0.50) * 100 * 2)

    def test_scales_linearly_with_qty(self):
        one = short_margin_requirement(100, 100, "C", 5, 1)
        ten = short_margin_requirement(100, 100, "C", 5, 10)
        assert ten == pytest.approx(10 * one)


def leg(spot=100.0, strike=100.0, type_="C", qty=1, side=-1, vol=0.25, rate=0.05, dividend=0.0, tenor=0.5):
    return {
        "spot": spot, "strike": strike, "type": type_, "qty": qty, "side": side,
        "vol": vol, "rate": rate, "dividend": dividend, "tenor": tenor,
    }


class TestPortfolioMargin:
    def test_offsetting_legs_fall_back_to_minimum(self):
        # Long and short the identical option: zero risk → per-short-contract minimum.
        legs = [leg(side=1, qty=10), leg(side=-1, qty=10)]
        got = portfolio_margin_requirement(legs)
        assert got == pytest.approx(PM_MIN_PER_SHORT_CONTRACT * 10)

    def test_naked_short_call_worst_case_is_up_15_percent(self):
        l = leg(side=-1, qty=10, type_="C")
        base = bs_price(100.0, 100.0, 0.05, 0.0, 0.25, 0.5, "C")
        shocked = bs_price(115.0, 100.0, 0.05, 0.0, 0.25, 0.5, "C")
        expected = (shocked - base) * CONTRACT_MULTIPLIER * 10
        assert portfolio_margin_requirement([l]) == pytest.approx(expected)

    def test_naked_short_put_worst_case_is_down_15_percent(self):
        l = leg(side=-1, qty=10, type_="P")
        base = bs_price(100.0, 100.0, 0.05, 0.0, 0.25, 0.5, "P")
        shocked = bs_price(85.0, 100.0, 0.05, 0.0, 0.25, 0.5, "P")
        expected = (shocked - base) * CONTRACT_MULTIPLIER * 10
        assert portfolio_margin_requirement([l]) == pytest.approx(expected)

    def test_long_only_position_requirement_bounded_by_premium(self):
        # A long call can't lose more than its value; requirement ≤ current value, floor 0 shorts.
        l = leg(side=1, qty=5, type_="C")
        value = bs_price(100.0, 100.0, 0.05, 0.0, 0.25, 0.5, "C") * CONTRACT_MULTIPLIER * 5
        got = portfolio_margin_requirement([l])
        assert 0 <= got <= value + 1e-9

    def test_hedged_vertical_needs_less_than_naked_short(self):
        # Short 110 call hedged by long 100 call, same expiry: PM sees the offset.
        legs = [
            leg(strike=100.0, side=1, qty=10, type_="C"),
            leg(strike=110.0, side=-1, qty=10, type_="C"),
        ]
        pm = portfolio_margin_requirement(legs)
        reg_t = short_margin_requirement(
            spot=100.0, strike=110.0, opt_type="C",
            premium=bs_price(100.0, 110.0, 0.05, 0.0, 0.25, 0.5, "C"), qty=10,
        )
        assert pm < reg_t
        # And it can never exceed the strike width for a vertical.
        assert pm <= (110 - 100) * CONTRACT_MULTIPLIER * 10

    def test_requirement_never_negative(self):
        legs = [leg(side=1, qty=1)]  # long only, no shorts → floor is 0
        assert portfolio_margin_requirement(legs) >= 0


class TestBsPrice:
    def test_put_call_parity(self):
        from math import exp
        S, K, r, q, vol, T = 100.0, 105.0, 0.05, 0.01, 0.3, 1.0
        call = bs_price(S, K, r, q, vol, T, "C")
        put = bs_price(S, K, r, q, vol, T, "P")
        assert call - put == pytest.approx(S * exp(-q * T) - K * exp(-r * T), abs=1e-10)

    def test_zero_tenor_returns_intrinsic(self):
        assert bs_price(110, 100, 0.05, 0.0, 0.25, 0.0, "C") == pytest.approx(10.0)
        assert bs_price(110, 100, 0.05, 0.0, 0.25, 0.0, "P") == pytest.approx(0.0)
