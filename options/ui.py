import os
from datetime import datetime

import polars as pl
import streamlit as st

from options.auth.users import UserRepository
from options.data_sources import GCSClosesDataSource, load_from_env
from options.pricing_utils import CBOEOptionsData, OptionsPrices

st.set_page_config(layout="wide")


def compute_overvalued(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col(metric) / pl.col("FMV")) - 1.0).alias("%Overvalued")
    )


@st.cache_data
def load_data(pricing_model: str, close_date: str | None) -> pl.DataFrame:
    source = load_from_env()
    if isinstance(source, GCSClosesDataSource):
        if close_date is None:
            raw_df = source.load()
        else:
            raw_df = source.load_for_date(datetime.strptime(close_date, "%Y-%m-%d").date())
    else:
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


def round_to_step(value: float, step: int) -> int:
    return max(step, int(round(value / step)) * step)


def format_contract(ticker: str, expiry: datetime, strike: float, opt_type: str) -> str:
    expiry_label = expiry.strftime("%b%y")
    opt_label = "c" if opt_type == "C" else "p"
    return f"{ticker} {expiry_label} {strike:g}{opt_label}"


def build_spreads(
    df: pl.DataFrame,
    metric_choice: str,
    call_delta_range: tuple[int, int],
    contract_step: int,
    max_contract_ratio: float,
    min_last_price: float,
    max_last_price: float,
    max_abs_net_delta: float,
    max_legs_per_side: int,
    max_results: int,
) -> list[dict[str, object]]:
    sdf = compute_overvalued(df, metric_choice)
    sdf = (
        sdf.with_columns(
            pl.when(pl.col("Type") == "C")
            .then(pl.col("Delta"))
            .otherwise(-pl.col("Delta"))
            .alias("CallDelta")
        )
        .filter(
            (pl.col("CallDelta") >= call_delta_range[0] / 100)
            & (pl.col("CallDelta") <= call_delta_range[1] / 100)
        )
        .with_columns(pl.col("Last").alias("LastPrice"))
        .filter(pl.col("LastPrice") > 0)
    )

    sdf = sdf.filter(
        (pl.col("LastPrice") >= min_last_price)
        & (pl.col("LastPrice") <= max_last_price)
    )

    if sdf.is_empty():
        return []

    sorted_legs = sdf.sort("%Overvalued")
    buys = sorted_legs.head(max_legs_per_side)
    sells = sorted_legs.tail(max_legs_per_side)

    buy_rows = buys.to_dicts()
    sell_rows = sells.to_dicts()
    spreads: list[dict[str, object]] = []

    for buy in buy_rows:
        buy_delta = float(buy["Delta"])
        if buy_delta == 0:
            continue
        for sell in sell_rows:
            if (
                buy["Expiry"] == sell["Expiry"]
                and buy["Strike"] == sell["Strike"]
                and buy["Type"] == sell["Type"]
            ):
                continue
            sell_delta = float(sell["Delta"])
            if sell_delta == 0:
                continue

            buy_qty = contract_step
            sell_qty = round_to_step(
                abs(buy_delta) / abs(sell_delta) * contract_step, contract_step
            )
            if sell_qty <= 0:
                continue

            contract_ratio = max(buy_qty, sell_qty) / min(buy_qty, sell_qty)
            if contract_ratio > max_contract_ratio:
                continue

            net_delta = buy_delta * buy_qty - sell_delta * sell_qty
            if abs(net_delta) > max_abs_net_delta:
                continue

            edge = float(sell["%Overvalued"]) - float(buy["%Overvalued"])
            if edge <= 0:
                continue

            buy_contract = format_contract(
                buy["Ticker"], buy["Expiry"], buy["Strike"], buy["Type"]
            )
            sell_contract = format_contract(
                sell["Ticker"], sell["Expiry"], sell["Strike"], sell["Type"]
            )
            spreads.append(
                {
                    "Buy": buy_contract,
                    "Sell": sell_contract,
                    "Buy Qty": buy_qty,
                    "Sell Qty": sell_qty,
                    "Net Delta": net_delta,
                    "Edge": edge,
                    "Details": f"BUY: {buy_qty} {buy_contract} | SELL: {sell_qty} {sell_contract}",
                }
            )

    spreads = sorted(spreads, key=lambda row: row["Edge"], reverse=True)
    return spreads[:max_results]


def main():
    user_repo = UserRepository.from_env()
    if not login_gate(user_repo):
        st.info("Please log in to view the option chain.")
        st.stop()

    source = load_from_env()
    close_date: str | None = None
    if isinstance(source, GCSClosesDataSource):
        available_dates = source.list_available_dates()
        if not available_dates:
            st.error("No closing files found in the configured GCS bucket.")
            st.stop()
        latest_date = available_dates[-1]
        selected_date = st.sidebar.date_input(
            "Close date",
            value=latest_date,
            min_value=available_dates[0],
            max_value=latest_date,
        )
        if selected_date not in available_dates:
            st.sidebar.warning(
                "Selected date is not available. Using the most recent close instead."
            )
            selected_date = latest_date
        close_date = selected_date.strftime("%Y-%m-%d")
        st.sidebar.caption(
            f"Available closes: {available_dates[0]:%Y-%m-%d} to {latest_date:%Y-%m-%d}"
        )

    pricing_model = st.sidebar.selectbox(
        "Pricing library", ["mzpricer", "quantlib"], index=0
    )

    df = load_data(pricing_model, close_date)

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
        value=(5, 95),
        step=1,
    )

    st.sidebar.subheader("Spread Filters")
    contract_step = st.sidebar.number_input(
        "Contracts per leg (multiple of 10)",
        min_value=10,
        value=10,
        step=10,
    )
    max_contract_ratio = st.sidebar.number_input(
        "Max contract ratio (leg imbalance)",
        min_value=1.0,
        value=3.0,
        step=0.5,
    )
    min_last_price = st.sidebar.number_input(
        "Min last price ($)",
        min_value=0.0,
        value=0.1,
        step=0.05,
    )
    max_last_price = st.sidebar.number_input(
        "Max last price ($)",
        min_value=0.0,
        value=1000.0,
        step=1.0,
    )
    max_abs_net_delta = st.sidebar.number_input(
        "Max |net delta|",
        min_value=0.0,
        value=5.0,
        step=0.5,
    )
    max_legs_per_side = st.sidebar.number_input(
        "Max legs per side",
        min_value=10,
        value=50,
        step=10,
    )
    max_results = st.sidebar.number_input(
        "Max spread ideas",
        min_value=10,
        value=50,
        step=10,
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

    chain_tab, spreads_tab = st.tabs(["Option Chain", "Spreads"])

    with chain_tab:
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

    with spreads_tab:
        spreads = build_spreads(
            df,
            metric_choice,
            (call_delta_min, call_delta_max),
            int(contract_step),
            max_contract_ratio,
            min_last_price,
            max_last_price,
            max_abs_net_delta,
            int(max_legs_per_side),
            int(max_results),
        )

        if not spreads:
            st.info("No spreads matched the current filters.")
        else:
            display_df = pl.DataFrame(spreads).with_columns(
                pl.col("Edge").cast(float).round(4),
                pl.col("Net Delta").cast(float).round(4),
            )
            st.dataframe(display_df.drop("Details").to_pandas(), use_container_width=True)

            spread_choices = {
                f"Idea {idx + 1}: {row['Buy']} / {row['Sell']}": row["Details"]
                for idx, row in enumerate(spreads)
            }
            selected = st.selectbox(
                "Select a spread to view construction details",
                list(spread_choices.keys()),
            )
            if selected:
                st.success(spread_choices[selected])


if __name__ == "__main__":
    main()
