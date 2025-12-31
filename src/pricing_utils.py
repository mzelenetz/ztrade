from mzpricer import option_greeks, option_price, StockPrice, TimeDuration, OptionType
import pandas as pd 
import polars as pl
import yfinance as yf
import pandas as pd
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
    def __init__(self, path: Optional[str] = None, date: str = "", symbols = [], default_vol: float = 0.25, use_remote_vol: bool = False, dataframe: Optional[pl.DataFrame] = None):
        self.path = path
        self.date = date
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
        ]


        df = (df.with_columns(
                pl.datetime(
                    year=2023,
                    month=8,
                    day=23,
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
            })
            .filter(pl.col("Spot") > 0)
            .select(cols)
        )
        return df
    
    def _add_vols(self, df: pl.DataFrame, vols: dict) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("Ticker").replace(vols).cast(pl.Float64).alias("Vol30d")
        )
        return df
    
    def _add_durations(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            (
                (pl.col("Expiry").dt.timestamp() - pl.col("ValuationTime").dt.timestamp()) / (365.0 * 24 * 3600)
            ).alias("T")
        )
        return df
    
    def _get_vols(self, df: pl.DataFrame) -> dict:
        ticker_symbols = df.select(pl.col("Ticker")).unique().to_series().to_list()
        if not self.use_remote_vol:
            return {sym: self.default_vol for sym in ticker_symbols}

        try:
            vols = get_historical_volatility(ticker_symbols, self.date, window=30)
            return vols
        except Exception:
            # Network or data fetch failures fall back to static vol so the UI remains usable.
            return {sym: self.default_vol for sym in ticker_symbols}
    
    def get_data(self) -> pl.DataFrame:
        df = self._load_data()
        df = self._prep_data(df)
        vols = self._get_vols(df)
        df = self._add_vols(df, vols)
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

    def _price_with_mzpricer(self) -> pl.DataFrame:
        data = self._common_inputs()
        tenors = [TimeDuration(t, 365) for t in data["tenors"]]
        option_types = [OptionType.Call if t == "C" else OptionType.Put for t in data["types"]]

        prices, _ = option_price(
            data["S"],
            data["K"],
            tenors,
            data["r"],
            data["sigma"],
            option_types,
            500,
        )

        greeks = option_greeks(
            data["S"],
            data["K"],
            tenors,
            data["r"],
            data["sigma"],
            option_types,
            500,
        )

        greek_columns = {}
        if isinstance(greeks, dict):
            greek_columns = {"Delta": greeks.get("delta") or greeks.get("Delta")}
        elif isinstance(greeks, (list, tuple)):
            greek_columns = {"Delta": greeks[0] if len(greeks) > 0 else None}

        return self._finalize_output(prices, greek_columns)

    def _finalize_output(self, prices, greeks: dict | None) -> pl.DataFrame:
        output = self.input_data.with_columns(pl.Series("FMV", prices))

        if greeks:
            for key, values in greeks.items():
                if values is None:
                    values = [float("nan")] * len(prices)
                output = output.with_columns(pl.Series(key, values))
        elif "Delta" not in output.columns:
            output = output.with_columns(pl.Series("Delta", [float("nan")] * len(prices)))

        return output.with_columns((pl.col("Last") / pl.col("FMV") - 1).alias("%Overvalued"))

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
    
