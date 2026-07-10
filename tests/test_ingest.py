"""Ingestion: yfinance normalization, publish-gate validation, universe parsing,
and the round-trip contract with the app's loader."""

from datetime import date

import pandas as pd
import polars as pl
import pytest

from src.ingest.fetchers import SCHEMA_COLUMNS, YFinanceFetcher
from src.ingest.job import DEFAULT_TICKERS, run_ingest, validate_ticker_frame
from src.pricing_utils import CBOEOptionsData


def yf_side(strikes, lasts, bids, asks, ivs=None, vols=None, ois=None):
    n = len(strikes)
    return pd.DataFrame(
        {
            "strike": strikes,
            "lastPrice": lasts,
            "bid": bids,
            "ask": asks,
            "impliedVolatility": ivs or [0.3] * n,
            "volume": vols or [10] * n,
            "openInterest": ois or [100] * n,
        }
    )


class TestNormalize:
    def test_schema_and_values(self):
        pdf = yf_side([95.0, 100.0], [6.0, 3.0], [5.9, 2.9], [6.1, 3.1])
        out = YFinanceFetcher._normalize(
            pdf, "TST", date(2026, 7, 10), "2026-12-18", "C", 99.9, 100.1
        )
        assert out.columns == SCHEMA_COLUMNS
        row = out.to_dicts()[0]
        assert row["underlying_symbol"] == "TST"
        assert row["quote_date"] == "2026-07-10"
        assert row["expiration"] == "2026-12-18"
        assert row["option_type"] == "C"
        assert row["close"] == 6.0 and row["bid_1545"] == 5.9 and row["ask_1545"] == 6.1
        assert row["underlying_bid_1545"] == 99.9

    def test_priceless_rows_dropped_and_nulls_filled(self):
        pdf = yf_side([90.0, 95.0], [0.0, 5.0], [None, None], [None, None])
        out = YFinanceFetcher._normalize(
            pdf, "TST", date(2026, 7, 10), "2026-12-18", "P", 100.0, 100.0
        )
        assert out.height == 1  # the strike with no price signal at all is gone
        assert out["bid_1545"][0] == 0.0


def frame_for_validation(n_strikes=60, spot=100.0, scramble_parity=False) -> pl.DataFrame:
    rows = []
    for i in range(n_strikes):
        k = 70 + i
        call = max(spot - k, 0) + 2.0
        put = max(k - spot, 0) + 2.0
        if scramble_parity and i % 2:
            call += 30.0  # break C−P+K coherence on half the strikes
        for typ, px in (("C", call), ("P", put)):
            rows.append(
                {
                    "underlying_symbol": "TST",
                    "quote_date": "2026-07-10",
                    "expiration": "2026-08-21",
                    "strike": float(k),
                    "option_type": typ,
                    "close": px,
                    "bid_1545": px - 0.05,
                    "ask_1545": px + 0.05,
                    "underlying_bid_1545": spot,
                    "underlying_ask_1545": spot,
                    "implied_volatility_1545": 0.3,
                    "trade_volume": 10.0,
                    "open_interest": 100.0,
                }
            )
    return pl.DataFrame(rows)


class TestValidation:
    def test_clean_frame_passes(self):
        assert validate_ticker_frame(frame_for_validation(), "TST") is None

    def test_too_few_rows_rejected(self):
        assert "rows" in validate_ticker_frame(frame_for_validation(n_strikes=10), "TST")

    def test_incoherent_parity_rejected(self):
        reason = validate_ticker_frame(frame_for_validation(scramble_parity=True), "TST")
        assert reason is not None and "parity" in reason

    def test_all_tickers_failing_raises(self, monkeypatch, tmp_path):
        class FailingFetcher:
            def fetch(self, ticker, quote_date):
                raise RuntimeError("boom")

        monkeypatch.setenv("TICKERS", "AAA,BBB")
        monkeypatch.setattr("src.ingest.job.load_universe", lambda: ["AAA", "BBB"])
        with pytest.raises(RuntimeError, match="no publishable data"):
            run_ingest(
                quote_date=date(2026, 7, 10),  # a Friday
                dry_run_path=str(tmp_path / "out.csv"),
                fetcher=FailingFetcher(),
            )

    def test_weekend_skips(self, tmp_path):
        out = run_ingest(quote_date=date(2026, 7, 11), dry_run_path=str(tmp_path / "x.csv"))
        assert out is None

    def test_default_universe(self):
        assert "NVDA" in DEFAULT_TICKERS


class TestRoundTrip:
    def test_dry_run_output_feeds_the_app_loader(self, tmp_path):
        class FakeFetcher:
            def fetch(self, ticker, quote_date):
                return frame_for_validation()

        dest = run_ingest(
            quote_date=date(2026, 7, 10),
            dry_run_path=str(tmp_path / "closes.csv"),
            fetcher=FakeFetcher(),
        )
        raw = pl.read_csv(dest)
        df = CBOEOptionsData(dataframe=raw, default_vol=0.25).get_data()
        assert df.height > 0
        assert df["Spot"][0] == pytest.approx(100.0, abs=2.0)
        assert df["ValuationTime"][0].date() == date(2026, 7, 10)
        assert (df["T"] > 0).all()
