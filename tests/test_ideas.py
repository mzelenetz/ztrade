"""Cross-ticker idea scoring: executable edge, confidence checks, diversity cap."""

from src.services.ideas_service import (
    MAX_IDEAS,
    MAX_PER_TICKER,
    assess_spread,
    build_ideas,
    leg_capacity,
)


def leg(bid=4.9, ask=5.1, mid=5.0, volume=500, vol_from_surface=True,
        open_interest=1000.0, bid_size=None, ask_size=None):
    return {
        "bid": bid, "ask": ask, "mid": mid, "volume": volume,
        "volFromSurface": vol_from_surface, "openInterest": open_interest,
        "bidSize": bid_size, "askSize": ask_size,
    }


def spread(net=1000.0, exec_edge=800.0, capital=10_000.0, buy_leg=None, sell_leg=None,
           buy_qty=10, sell_qty=10):
    return {
        "netEdgeDollars": net,
        "execEdgeDollars": exec_edge,
        "capitalEmployed": capital,
        "grossEdgeDollars": net + 50,
        "carryCost": 50.0,
        "buyQty": buy_qty,
        "sellQty": sell_qty,
        "buyLeg": buy_leg or leg(),
        "sellLeg": sell_leg or leg(),
    }


class TestAssess:
    def test_clean_spread_is_high_confidence(self):
        idea = assess_spread(spread())
        assert idea["confidence"] == "high"
        assert idea["flags"] == []
        assert idea["returnOnCapital"] == 0.1

    def test_negative_net_edge_disqualified(self):
        assert assess_spread(spread(net=-5.0)) is None

    def test_edge_dying_at_bid_ask_is_low(self):
        idea = assess_spread(spread(exec_edge=-100.0))
        assert idea["confidence"] == "low"
        assert any("survive" in f for f in idea["flags"])

    def test_wide_market_flagged(self):
        idea = assess_spread(spread(buy_leg=leg(bid=4.0, ask=6.0)))  # 40% wide
        assert any("wide" in f for f in idea["flags"])
        assert idea["confidence"] == "medium"

    def test_thin_volume_and_wing_vol_downgrade(self):
        idea = assess_spread(
            spread(
                buy_leg=leg(volume=3),
                sell_leg=leg(vol_from_surface=False),
            )
        )
        assert idea["confidence"] == "medium"
        assert len(idea["flags"]) == 2

    def test_three_soft_flags_is_low(self):
        idea = assess_spread(
            spread(
                buy_leg=leg(volume=3, vol_from_surface=False),
                sell_leg=leg(bid=4.0, ask=6.0),
            )
        )
        assert idea["confidence"] == "low"


class TestOpenInterestFilter:
    def test_thin_oi_leg_disqualifies(self):
        s = spread(sell_leg=leg(open_interest=12.0))
        assert assess_spread(s, min_open_interest=50) is None
        assert assess_spread(s, min_open_interest=0) is not None  # filter off

    def test_missing_oi_passes_the_hard_filter(self):
        # No OI data ≠ zero OI — don't silently drop; capacity flags handle it
        s = spread(buy_leg=leg(open_interest=None))
        assert assess_spread(s, min_open_interest=50) is not None


class TestFillability:
    def test_quote_size_is_the_direct_capacity(self):
        assert leg_capacity(leg(ask_size=7.0), "buy") == 7.0
        assert leg_capacity(leg(bid_size=3.0), "sell") == 3.0

    def test_proxy_capacity_from_volume_and_oi(self):
        # max(0.25·volume, 0.05·OI) = max(0.25·40, 0.05·400) = max(10, 20) = 20
        assert leg_capacity(leg(volume=40.0, open_interest=400.0), "buy") == 20.0
        assert leg_capacity(leg(volume=None, open_interest=None), "buy") is None

    def test_shortfall_flagged_and_severe_forces_low(self):
        # capacity 20, needs 25 → flagged but not severe
        idea = assess_spread(spread(buy_qty=25, buy_leg=leg(volume=40.0, open_interest=400.0)))
        assert any("fill" in f for f in idea["flags"])
        assert idea["confidence"] == "medium"

        # capacity 20, needs 50 → severe → low regardless of exec edge
        idea = assess_spread(spread(buy_qty=50, buy_leg=leg(volume=40.0, open_interest=400.0)))
        assert idea["confidence"] == "low"

    def test_displayed_size_overrides_generous_proxies(self):
        # Deep OI but the screen only shows 2 offered where 10 are needed
        idea = assess_spread(spread(buy_qty=10, buy_leg=leg(ask_size=2.0, volume=5000, open_interest=50000)))
        assert any("needs 10, capacity ~2" in f for f in idea["flags"])
        assert idea["confidence"] == "low"


class TestBuildIdeas:
    def test_per_ticker_cap_and_global_limit(self):
        by_ticker = {
            f"T{j}": [spread(net=1000 + i, exec_edge=900 + i) for i in range(10)]
            for j in range(5)
        }
        ideas = build_ideas(by_ticker)
        assert len(ideas) == MAX_IDEAS
        for t in {i["ticker"] for i in ideas}:
            assert sum(1 for i in ideas if i["ticker"] == t) <= MAX_PER_TICKER

    def test_confidence_outranks_edge(self):
        by_ticker = {
            "AAA": [spread(net=50_000, exec_edge=-1.0)],  # huge but dies at bid/ask
            "BBB": [spread(net=1_000, exec_edge=800.0)],  # modest but executable
        }
        ideas = build_ideas(by_ticker)
        assert ideas[0]["ticker"] == "BBB"
        assert ideas[0]["confidence"] == "high"

    def test_ranked_by_exec_edge_within_confidence(self):
        by_ticker = {"AAA": [spread(exec_edge=100.0), spread(exec_edge=900.0)]}
        ideas = build_ideas(by_ticker)
        assert ideas[0]["execEdgeDollars"] == 900.0
