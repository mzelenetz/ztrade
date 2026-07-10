import contextlib
import os

import pandas as pd
import polars as pl
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import QuantLib as ql


cols = [
    "Ticker",            # e.g., AAPL
    "ValuationTime",     # ISO8601, e.g., 2025-09-11T15:30:00
    "Spot",              # Underlying spot price S
    "Type",              # 'C' for call, 'P' for put
    "Strike",            # K
    "Expiry",            # ISO date or datetime for option expiration
    "Rate",              # risk-free cont. comp. rate r (annualized, decimal)
    "DividendYield",     # continuous dividend yield q (annualized, decimal)
    "Vol30d",            # annualized vol sigma (decimal, e.g., 0.25)
    "ContractMultiplier",# usually 100
    "Bid",               # market bid
    "Ask",               # market ask
    "Mid"                # market mid you will compare to fair
]



def ql_black_scholes_price_and_delta(S, K, r, q, sigma, valuation_dt, expiry_dt, is_call):

    """
    Returns (price, delta).
    """
    day_counter = ql.Actual365Fixed()

    # Set evaluation date
    ql.Settings.instance().evaluationDate = valuation_dt

    spot = ql.QuoteHandle(ql.SimpleQuote(S))

    rf   = ql.YieldTermStructureHandle(
                ql.FlatForward(valuation_dt, r, day_counter))
    div  = ql.YieldTermStructureHandle(
                ql.FlatForward(valuation_dt, q, day_counter))
    vol  = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(valuation_dt, ql.NullCalendar(), sigma, day_counter))

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(expiry_dt)

    process = ql.BlackScholesMertonProcess(spot, div, rf, vol)
    option  = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return option.NPV(), option.delta()


def get_price_series(df, ticker_symbol, prefer_adj=True):
    """
    Return the price series (Adj Close preferred) for ONE symbol.
    """
    if isinstance(df.columns, pd.MultiIndex):
        cols = df.columns
        if prefer_adj and ('Adj Close', ticker_symbol) in cols:
            return df[('Adj Close', ticker_symbol)].rename(ticker_symbol)
        if ('Close', ticker_symbol) in cols:
            return df[('Close', ticker_symbol)].rename(ticker_symbol)
    else:
        # Single-symbol DF
        if prefer_adj and 'Adj Close' in df.columns:
            return df['Adj Close']
        if 'Close' in df.columns:
            return df['Close']

    raise KeyError(f"Could not find price columns for {ticker_symbol}")

def get_historical_volatility(ticker_symbols: List[str], as_of, window=30):
    as_of_dt = pd.to_datetime(as_of)
    start_date = as_of_dt - timedelta(days=window * 2) # buffer to ensure enough data
    df = yf.download(ticker_symbols, start=start_date, end=as_of_dt)
    # Storage
    vols = {}

    for sym in ticker_symbols:
        prices = get_price_series(df, sym, prefer_adj=True)

        # returns
        returns = prices.pct_change()

        # 30-day rolling vol, annualized
        trading_days = 252
        vol_series = returns.rolling(window).std() * np.sqrt(trading_days)

        vols[sym] = float(vol_series.iloc[-1])

    return vols

class CBOEOptionsData:
    def __init__(
        self,
        path: Optional[str] = None,
        date: str = "",
        symbols = [],
        default_vol: float = 0.25,
        use_remote_vol: bool = False,
        dataframe: Optional[pl.DataFrame] = None,
        as_of: Optional[str] = None,
    ):
        self.path = path
        self.date = as_of or date
        self.symbols = symbols
        self.default_vol = default_vol
        self.use_remote_vol = use_remote_vol
        self.dataframe = dataframe
    
    def _load_data(self) -> pl.DataFrame:
        if self.dataframe is not None:
            df = self.dataframe
        else:
            if not self.path:
                raise ValueError("A CSV path or DataFrame is required to load option data")
            df = pl.read_csv(self.path)
        if self.symbols:
            df = df.filter(pl.col("underlying_symbol").is_in(self.symbols))
        return df
    
    def _resolve_valuation_date(self, df: pl.DataFrame):
        """Valuation date priority: explicit as_of/date arg → the data's own
        quote_date → DATA_AS_OF_DATE env → today.

        Using today's date against prices captured on an earlier close date
        destroys the model's time value while leaving it in the market price,
        which manufactures enormous fake '%Overvalued' readings.
        """
        if self.date:
            return pd.to_datetime(self.date).date()

        if "quote_date" in df.columns:
            quote_date = df["quote_date"].max()
            if quote_date is not None:
                return pd.to_datetime(quote_date).date()

        env_date = os.getenv("DATA_AS_OF_DATE")
        if env_date:
            return pd.to_datetime(env_date).date()

        return datetime.today().date()

    def _prep_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # At the end of this step you'll have
        cols = [
            "Ticker",            # e.g., AAPL
            "ValuationTime",     # ISO8601, e.g., 2025-09-11T15:30:00
            "Spot",              # Underlying spot price S
            "Type",              # 'C' for call, 'P' for put
            "Strike",            # K
            "Expiry",            # ISO date or datetime for option expiration
            "Rate",              # risk-free cont. comp. rate r (annualized, decimal)
            "DividendYield",     # continuous dividend yield q (annualized, decimal)
            "ContractMultiplier",# usually 100
            "Bid",               # market bid
            "Ask",               # market ask
            "Mid",                # market mid you will compare to fair
            "Last",               # CLose price of the option
            "MarketIV",           # per-contract market implied vol (if provided)
            "HistVol30d",         # trailing 30d realized vol of the underlying (if provided)
            "Volume",             # contract trade volume (if provided)
        ]


        as_of_date = self._resolve_valuation_date(df)

        if "implied_volatility_1545" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("implied_volatility_1545"))
        if "hist_vol_30d" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("hist_vol_30d"))
        if "trade_volume" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("trade_volume"))

        df = (df.with_columns(
                pl.datetime(
                    year=as_of_date.year,
                    month=as_of_date.month,
                    day=as_of_date.day,
                    hour=16,
                    minute=0,
                    second=0,
                    time_zone="America/New_York",
            ).alias("ValuationTime"),
            ((pl.col("underlying_ask_1545") + pl.col("underlying_bid_1545"))/2).alias("underlying_mid_1545"),
            (
                pl.col("expiration").str.to_datetime(format="%Y-%m-%d", time_zone="America/New_York") + pl.duration(hours=16)
            ),#.dt.to_string("iso"),
            pl.lit(100).alias('ContractMultiplier'),
            pl.lit(.05).alias("temp_borrow_rate"),
            pl.lit(0.0).alias("temp_div"),
            ((pl.col('ask_1545') + pl.col('bid_1545'))/2).alias('mid_1545')
            )
            .rename({
                "underlying_symbol": "Ticker", 
                "underlying_mid_1545": "Spot",             
                "option_type": "Type",             
                "strike": "Strike",           
                "expiration": "Expiry",           
                "temp_borrow_rate": "Rate",             
                "temp_div": "DividendYield",    
                "ContractMultiplier": "ContractMultiplier",
                "bid_1545": "Bid",              
                "ask_1545": "Ask",              
                "mid_1545": "Mid",
                "close": "Last",
                "implied_volatility_1545": "MarketIV",
                "hist_vol_30d": "HistVol30d",
                "trade_volume": "Volume",
            })
            .filter(pl.col("Spot") > 0)
            .filter(pl.col("Expiry") > pl.col("ValuationTime"))
            .select(cols)
        )

        df = self._fix_spots_via_parity(df)

        # Data hygiene: drop quotes that violate no-arbitrage bounds
        # (option price below intrinsic or above its underlying/strike cap).
        # These are stale/corrupt prints on illiquid wings — untradeable, and
        # they otherwise show up as enormous fake "edge".
        intrinsic = (
            pl.when(pl.col("Type") == "C")
            .then((pl.col("Spot") - pl.col("Strike")).clip(lower_bound=0))
            .otherwise((pl.col("Strike") - pl.col("Spot")).clip(lower_bound=0))
        )
        upper_bound = (
            pl.when(pl.col("Type") == "C").then(pl.col("Spot")).otherwise(pl.col("Strike"))
        )
        # 5% tolerance + $0.05 absorbs the 15:45-spot vs 16:00-close timing gap.
        df = df.filter(
            (pl.col("Last") >= intrinsic * 0.95 - 0.05)
            & (pl.col("Last") <= upper_bound * 1.05)
        )
        return df
    
    def _fix_spots_via_parity(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace a ticker's quoted Spot when its own option prices contradict it.

        Put-call parity gives S ≈ C − P + K·e^(−rT) at each strike; the median
        across the nearest expiry's strikes is a robust implied spot. Some close
        files carry a stale/wrong underlying quote (seen: PLTR marked 175.85
        while its options priced a ~150.8 spot), which corrupts every FMV for
        that ticker. Only overrides on >2% disagreement so a noisy parity
        estimate never perturbs a good quote.
        """
        rate = 0.045  # rough discount for the parity leg; immaterial at these tenors

        implied_spots: dict[str, float] = {}
        for (ticker,), group in df.group_by(["Ticker"]):
            nearest = group["Expiry"].min()
            sub = group.filter(pl.col("Expiry") == nearest)
            t_years = (
                (nearest - sub["ValuationTime"][0]).total_seconds() / (365.0 * 24 * 3600)
            )

            calls = {r["Strike"]: r["Last"] for r in sub.filter(pl.col("Type") == "C").to_dicts()}
            puts = {r["Strike"]: r["Last"] for r in sub.filter(pl.col("Type") == "P").to_dicts()}
            common = [
                k
                for k in calls
                if k in puts
                and calls[k] is not None
                and puts[k] is not None
                and calls[k] > 0.05
                and puts[k] > 0.05
            ]
            if len(common) < 3:
                continue

            estimates = [calls[k] - puts[k] + k * np.exp(-rate * t_years) for k in common]
            med = float(np.median(estimates))
            # Trust parity only when the strikes agree with each other: a tight
            # spread of estimates means a real (possibly mis-marked) underlying;
            # wild dispersion means the option data itself is corrupt, and no
            # single implied spot exists — overriding would poison every price.
            q25, q75 = np.percentile(estimates, [25, 75])
            if med <= 0 or (q75 - q25) > 0.05 * med:
                continue
            implied_spots[ticker] = med

        def corrected(ticker: str, quoted: float) -> float:
            implied = implied_spots.get(ticker)
            if implied is not None and implied > 0 and abs(implied / quoted - 1) > 0.02:
                return implied
            return quoted

        return df.with_columns(
            pl.struct(["Ticker", "Spot"])
            .map_elements(lambda s: corrected(s["Ticker"], s["Spot"]), return_dtype=pl.Float64)
            .alias("Spot")
        )

    def _add_vols(self, df: pl.DataFrame, vols: dict) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("Ticker").replace(vols).cast(pl.Float64).alias("Vol30d")
        )
        return df
    
    def _add_durations(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            (
                (
                    pl.col("Expiry").dt.timestamp(time_unit="us")
                    - pl.col("ValuationTime").dt.timestamp(time_unit="us")
                )
                / (365.0 * 24 * 3600 * 1_000_000)
            ).alias("T")
        )
        return df
    
    def _get_vols(self, df: pl.DataFrame) -> dict:
        """Trailing 30d realized vol per ticker.

        Priority: the hist_vol_30d column stamped by the ingest job (no network,
        as-of the file's own date) → live yfinance lookup as of the valuation
        date → the static default. The result is both the flat-vol pricing
        assumption and the "30d Volatility" shown in the header.
        """
        ticker_symbols = df.select(pl.col("Ticker")).unique().to_series().to_list()

        vols: dict[str, float] = {}
        if "HistVol30d" in df.columns:
            for ticker in ticker_symbols:
                v = df.filter(pl.col("Ticker") == ticker)["HistVol30d"].drop_nulls()
                if v.len() and 0.01 < v[0] < 5.0:
                    vols[ticker] = float(v[0])

        missing = [t for t in ticker_symbols if t not in vols]
        if missing:
            try:
                as_of = self.date or df["ValuationTime"].max().date().isoformat()
                vols.update(get_historical_volatility(missing, as_of, window=30))
            except Exception:
                # Network or data fetch failures fall back to static vol so the UI remains usable.
                pass

        for t in ticker_symbols:
            vols.setdefault(t, self.default_vol)
        return vols

    def get_data(self) -> pl.DataFrame:
        df = self._load_data()
        df = self._prep_data(df)
        vols = self._get_vols(df)
        df = self._add_vols(df, vols)
        # HistVol30d resolves to the same per-ticker value (filled where the
        # file lacked it) so the API can report it regardless of vol mode.
        df = df.with_columns(
            pl.col("Ticker").replace(vols).cast(pl.Float64).alias("HistVol30d")
        )
        df = self._add_durations(df)
        return df
    

class OptionsPrices:
    def __init__(self, input_df: pl.DataFrame, model: str = "mzpricer"):
        self.input_data = input_df
        self.model = model

    @staticmethod
    def to_ql_date(py_date):
        return ql.Date(py_date.day, py_date.month, py_date.year)

    def _common_inputs(self):
        return {
            "S": self.input_data["Spot"].to_list(),
            "K": self.input_data["Strike"].to_list(),
            "r": self.input_data["Rate"].to_list(),
            "sigma": self.input_data["Vol30d"].to_list(),
            "q": self.input_data["DividendYield"].to_list(),
            "types": self.input_data["Type"].to_list(),
            "valuation_times": (
                self.input_data["ValuationTime"].to_pandas().dt.tz_localize(None).dt.date.tolist()
            ),
            "expiry_times": (
                self.input_data["Expiry"].to_pandas().dt.tz_localize(None).dt.date.tolist()
            ),
            "tenors": self.input_data["T"].to_list(),
        }

    def _price_with_quantlib(self) -> pl.DataFrame:
        data = self._common_inputs()
        ql_valuation_dates = [OptionsPrices.to_ql_date(d) for d in data["valuation_times"]]
        ql_expiry_dates = [OptionsPrices.to_ql_date(d) for d in data["expiry_times"]]

        prices = []
        deltas = []

        for s, k, rr, qq, vol, opt_type, vdt, edt in zip(
            data["S"],
            data["K"],
            data["r"],
            data["q"],
            data["sigma"],
            data["types"],
            ql_valuation_dates,
            ql_expiry_dates,
        ):
            is_call = opt_type == "C"
            p, d = ql_black_scholes_price_and_delta(
                S=s,
                K=k,
                r=rr,
                q=qq,
                sigma=vol,
                valuation_dt=vdt,
                expiry_dt=edt,
                is_call=is_call,
            )
            prices.append(p)
            deltas.append(d)

        return self._finalize_output(prices, {"Delta": deltas})

    @staticmethod
    @contextlib.contextmanager
    def _silence_stdout():
        """mzpricer's Rust code prints a debug line per option; at fd level so
        it also catches non-Python writes. 16k log lines per request crushes
        Cloud Run logging throughput."""
        saved = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            yield
        finally:
            os.dup2(saved, 1)
            os.close(devnull)
            os.close(saved)

    @staticmethod
    def _analytic_deltas(data: dict) -> list[float]:
        """Closed-form BSM deltas (e^(−qT)·N(±d1)). The binomial engine's
        bump-and-reprice greeks were ~10x the cost of pricing itself, and
        Delta is the only greek the app consumes — analytic is effectively
        instant and matches the QuantLib path's convention."""
        from math import erf, exp, log, sqrt

        deltas = []
        for s, k, t, r, q, sigma, typ in zip(
            data["S"], data["K"], data["tenors"], data["r"], data["q"], data["sigma"], data["types"]
        ):
            if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
                deltas.append(None)
                continue
            d1 = (log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
            nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
            call_delta = exp(-q * t) * nd1
            deltas.append(call_delta if typ == "C" else call_delta - exp(-q * t))
        return deltas

    def _price_with_mzpricer(self) -> pl.DataFrame:
        from mzpricer import option_price, TimeDuration, OptionType

        data = self._common_inputs()
        # TimeDuration(value, factor) is value/factor years, i.e. value is in days
        # when factor is 365 — convert the year-fraction tenors accordingly.
        tenors = [TimeDuration(t * 365.0, 365) for t in data["tenors"]]
        option_types = [OptionType.Call if t == "C" else OptionType.Put for t in data["types"]]

        # Deltas from the closed form, using the original spot and q.
        deltas = self._analytic_deltas(data)

        # mzpricer has no dividend-yield input; approximate continuous dividends by
        # pricing off the dividend-adjusted spot S·e^(−qT).
        from math import exp as _exp

        data["S"] = [
            s * _exp(-q * t) for s, q, t in zip(data["S"], data["q"], data["tenors"])
        ]

        with self._silence_stdout():
            prices, _ = option_price(
                data["S"],
                data["K"],
                tenors,
                data["r"],
                data["sigma"],
                option_types,
                # 200 binomial steps: within $0.01 of 500 steps (far below quote
                # noise) and much faster — tree size dominates request latency.
                precision=200,
            )

        return self._finalize_output(prices, {"Delta": deltas})

    def _finalize_output(self, prices, greeks: dict | None) -> pl.DataFrame:
        output = self.input_data.with_columns(pl.Series("FMV", prices))

        if greeks:
            for key, values in greeks.items():
                if values is None:
                    values = [float("nan")] * len(prices)
                output = output.with_columns(pl.Series(key, values))
        elif "Delta" not in output.columns:
            output = output.with_columns(pl.Series("Delta", [float("nan")] * len(prices)))

        return output.with_columns(
            pl.when(pl.col("FMV") >= 0.05)
            .then(pl.col("Last") / pl.col("FMV") - 1)
            .otherwise(None)
            .alias("%Overvalued")
        )

    def price_options(self) -> pl.DataFrame:
        if self.model == "quantlib":
            return self._price_with_quantlib()

        try:
            return self._price_with_mzpricer()
        except Exception:
            # Fallback to QuantLib if mzpricer is unavailable or errors
            return self._price_with_quantlib()

    
    def calls_puts_split(self, prices: pl.DataFrame) -> (pl.DataFrame, pl.DataFrame):
        calls_df = prices.filter(pl.col("Type") == 'C')
        puts_df = prices.filter(pl.col("Type") == 'P')
        return calls_df, puts_df
    
