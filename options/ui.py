import os
from datetime import datetime

import polars as pl
import streamlit as st

from options.auth.users import UserRepository
from options.data_sources import load_from_env
from options.pricing_utils import CBOEOptionsData, OptionsPrices

st.set_page_config(layout="wide")


def compute_overvalued(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col(metric) / pl.col("FMV")) - 1.0).alias("%Overvalued")
    )


@st.cache_data
def load_data(pricing_model: str) -> pl.DataFrame:
    source = load_from_env()
    raw_df = source.load()

    as_of = os.getenv("DATA_AS_OF_DATE", datetime.today().strftime("%Y-%m-%d"))
    default_vol = float(os.getenv("DEFAULT_VOLATILITY", "0.25"))
    use_remote_vol = os.getenv("USE_REMOTE_VOL", "false").lower() == "true"

    loader = CBOEOptionsData(
        date=as_of,
        default_vol=default_vol,
        use_remote_vol=use_remote_vol,
        dataframe=raw_df,
    )

    df_opts = loader.get_data()
    try:
        prices = OptionsPrices(df_opts, model=pricing_model)
    except TypeError:
        # Fall back for environments where OptionsPrices does not yet accept a
        # model argument, while still honoring the user's selection when
        # possible.
        prices = OptionsPrices(df_opts)
        setattr(prices, "model", pricing_model)
    df = prices.price_options()
    return df


def login_gate(user_repo: UserRepository) -> bool:
    if "user" not in st.session_state:
        st.session_state.user = None

    st.sidebar.header("Login")

    if st.session_state.user:
        st.sidebar.success(f"Logged in as {st.session_state.user.username}")
        if st.sidebar.button("Log out"):
            st.session_state.user = None
            st.cache_data.clear()
        return st.session_state.user is not None

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Log in"):
        user = user_repo.authenticate(username=username, password=password)
        if user:
            st.session_state.user = user
            st.sidebar.success(f"Welcome {user.username}!")
            return True
        st.sidebar.error("Invalid credentials")

    return False


def bid_ask(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("Bid").cast(str) + " – " + pl.col("Ask").cast(str)).alias("BidAsk")
    )


def render_expiry_block(sdf: pl.DataFrame, metric_choice: str, call_delta_range: tuple[int, int]):
    sdf = compute_overvalued(sdf, metric_choice)

    sdf = (
        sdf.with_columns(
            pl.when(pl.col("Type") == "C")
            .then(pl.col("Delta"))
            .otherwise(-pl.col("Delta")).alias("CallDelta")
        )
        .filter(
            (pl.col("CallDelta") >= call_delta_range[0] / 100)
            & (pl.col("CallDelta") <= call_delta_range[1] / 100)
        )
    )

    calls = bid_ask(sdf.filter(pl.col("Type") == "C")).sort("Strike")
    puts = bid_ask(sdf.filter(pl.col("Type") == "P")).sort("Strike")

    calls_display = calls.select(
        [
            "Strike",
            pl.col("FMV").alias("Call FMV"),
            pl.col("Last").alias("Call Last"),
            pl.col("BidAsk").alias("Call Bid/Ask"),
            pl.col("Delta").alias("Call Delta"),
            pl.col("%Overvalued").alias("Call %Overvalued"),
        ]
    )

    puts_display = puts.select(
        [
            "Strike",
            pl.col("FMV").alias("Put FMV"),
            pl.col("Last").alias("Put Last"),
            pl.col("BidAsk").alias("Put Bid/Ask"),
            pl.col("Delta").alias("Put Delta"),
            pl.col("%Overvalued").alias("Put %Overvalued"),
        ]
    )

    combined = (
        calls_display.join(puts_display, on="Strike", how="outer")
        .sort("Strike")
        .fill_null("-")
    )

    st.dataframe(combined.to_pandas(), use_container_width=True)


def render_chain(df: pl.DataFrame, metric_choice: str, call_delta_range: tuple[int, int]):
    for (expiry,), sdf in df.sort("Expiry").group_by("Expiry"):
        with st.expander(expiry.strftime("%d-%b-%y"), expanded=True):
            render_expiry_block(sdf, metric_choice, call_delta_range)
        st.markdown("---")


def main():
    user_repo = UserRepository.from_env()
    if not login_gate(user_repo):
        st.info("Please log in to view the option chain.")
        st.stop()

    st.sidebar.header("Pricing")
    pricing_model = st.sidebar.selectbox(
        "Pricing library", ["mzpricer", "quantlib"], index=0, help="Choose which pricing engine to use for valuations."
    )

    df = load_data(pricing_model)

    ticker = st.sidebar.selectbox("Ticker", df["Ticker"].unique().to_list())
    metric_choice = st.sidebar.selectbox(
        "Overvaluation relative to:",
        ["Last", "Mid", "Bid", "Ask"],
        index=0,
    )

    call_delta_min, call_delta_max = st.sidebar.slider(
        "Call-Delta Range (0 = OTM, 100 = ITM)",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=1,
    )

    df = df.filter(pl.col("Ticker") == ticker)

    spot = float(df["Spot"][0])
    div = df["DividendYield"].mean()
    vol = float(df["Vol30d"][0])
    valuation_time = df["ValuationTime"].max()

    st.subheader(f"{ticker} – {valuation_time.date()}")
    col1, col2, col3 = st.columns(3)

    col1.metric("Last", f"{spot:.2f}")
    col2.metric("Dividend Yield", f"{div:.2f}")
    col3.metric("30d Volatility", f"{vol:.2f}")

    st.markdown("---")

    render_chain(df, metric_choice, (call_delta_min, call_delta_max))

    st.subheader("Positions / DR-CR Summary")
    positions_df = pl.DataFrame(
        {
            "Ticker": [ticker],
            "Expiry": ["2025-01-07"],
            "Strike": [104],
            "Call/Put": ["C"],
            "Qty": [10],
            "FMV": [6.325],
            "DR/CR": ["DR"],
        }
    )
    st.dataframe(positions_df)


if __name__ == "__main__":
    main()
