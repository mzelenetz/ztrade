"""Cross-ticker idea scoring: executable edge, confidence checks, diversity cap."""

from src.services.ideas_service import MAX_IDEAS, MAX_PER_TICKER, assess_spread, build_ideas


def leg(bid=4.9, ask=5.1, mid=5.0, volume=500, vol_from_surface=True):
    return {"bid": bid, "ask": ask, "mid": mid, "volume": volume, "volFromSurface": vol_from_surface}


def spread(net=1000.0, exec_edge=800.0, capital=10_000.0, buy_leg=None, sell_leg=None):
    return {
        "netEdgeDollars": net,
        "execEdgeDollars": exec_edge,
        "capitalEmployed": capital,
        "grossEdgeDollars": net + 50,
        "carryCost": 50.0,
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
