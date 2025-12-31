import os
from datetime import datetime
from math import erf, log, sqrt

import polars as pl
import streamlit as st

from src.auth.users import UserRepository
from src.data_sources import GCSClosesDataSource, load_from_env
from src.pricing_utils import CBOEOptionsData, OptionsPrices


st.set_page_config(layout="wide")


def compute_overvalued(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    return df.with_columns(((pl.col(metric) / pl.col("FMV")) - 1.0).alias("%Overvalued"))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def add_probabilities(df: pl.DataFrame) -> pl.DataFrame:
    def prob_itm(row: dict) -> float | None:
        spot = float(row["Spot"])
        strike = float(row["Strike"])
        rate = float(row["Rate"])
        dividend = float(row["DividendYield"])
        vol = float(row["Vol30d"])
        tenor = float(row["T"])

        if spot <= 0 or strike <= 0 or vol <= 0 or tenor <= 0:
            return None

        d2 = (log(spot / strike) + (rate - dividend - 0.5 * vol**2) * tenor) / (
            vol * sqrt(tenor)
        )

        if row["Type"] == "C":
            return float(normal_cdf(d2))

        return float(normal_cdf(-d2))

    def prob_otm(row: dict) -> float | None:
        itm = prob_itm(row)
        if itm is None:
            return None
        return 1.0 - itm

    inputs = ["Spot", "Strike", "Rate", "DividendYield", "Vol30d", "T", "Type"]
    return df.with_columns(
        pl.struct(inputs).map_elements(prob_itm, return_dtype=pl.Float64).alias("Prob ITM"),
        pl.struct(inputs).map_elements(prob_otm, return_dtype=pl.Float64).alias("Prob OTM"),
    )


@st.cache_data
def load_data(pricing_model: str, close_date: str | None) -> pl.DataFrame:
    source = load_from_env()

    if isinstance(source, GCSClosesDataSource):
        raw_df = (
            source.load()
            if close_date is None
            else source.load_for_date(datetime.strptime(close_date, "%Y-%m-%d").date())
        )
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
        prices = OptionsPrices(df_opts)
        setattr(prices, "model", pricing_model)

    return add_probabilities(prices.price_options())


def login_gate(user_repo: UserRepository) -> bool:
    if "user" not in st.session_state:
        st.session_state.user = None

    st.sidebar.header("Login")

    if st.session_state.user:
        st.sidebar.success(f"Logged in as {st.session_state.user.username}")
        if st.sidebar.button("Log out"):
            st.session_state.user = None
            st.cache_data.clear()
        return True

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
    return df.with_columns((pl.col("Bid").cast(str) + " – " + pl.col("Ask").cast(str)).alias("BidAsk"))


def render_expiry_block(
    sdf: pl.DataFrame,
    metric_choice: str,
    call_delta_range,
    min_option_price: float,
    greek_columns: list[str],
):
    sdf = compute_overvalued(sdf, metric_choice)

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
            & (pl.col("Last") >= min_option_price)
        )
    )

    calls = bid_ask(sdf.filter(pl.col("Type") == "C")).sort("Strike")
    puts = bid_ask(sdf.filter(pl.col("Type") == "P")).sort("Strike")

    optional_columns = [col for col in greek_columns if col in sdf.columns]
    prob_columns = [col for col in ["Prob ITM", "Prob OTM"] if col in sdf.columns]

    combined = (
        calls.select(
            [
                "Strike",
                pl.col("FMV").alias("Call FMV"),
                pl.col("Last").alias("Call Last"),
                pl.col("BidAsk").alias("Call Bid/Ask"),
                pl.col("%Overvalued").alias("Call %Overvalued"),
                *[pl.col(col).alias(f"Call {col}") for col in optional_columns],
                *[pl.col(col).cast(pl.Float64).round(4).alias(f"Call {col}") for col in prob_columns],
            ]
        )
        .join(
            puts.select(
                [
                    "Strike",
                    pl.col("FMV").alias("Put FMV"),
                    pl.col("Last").alias("Put Last"),
                    pl.col("BidAsk").alias("Put Bid/Ask"),
                    pl.col("%Overvalued").alias("Put %Overvalued"),
                    *[pl.col(col).alias(f"Put {col}") for col in optional_columns],
                    *[pl.col(col).cast(pl.Float64).round(4).alias(f"Put {col}") for col in prob_columns],
                ]
            ),
            on="Strike",
            how="inner",
        )
        .sort("Strike")
        .fill_null("-")
    )

    if "Strike_right" in combined.columns:
        combined = combined.drop("Strike_right")

    st.dataframe(combined.to_pandas(), use_container_width=True)


def render_chain(df, metric_choice, call_delta_range, min_option_price, greek_columns):
    for (expiry,), sdf in df.sort("Expiry").group_by("Expiry"):
        with st.expander(expiry.strftime("%d-%b-%y"), expanded=True):
            render_expiry_block(sdf, metric_choice, call_delta_range, min_option_price, greek_columns)
        st.markdown("---")


def round_to_step(value: float, step: int) -> int:
    return max(step, int(round(value / step)) * step)


def format_contract(ticker, expiry, strike, opt_type):
    expiry_label = expiry.strftime("%b%y")
    opt_label = "c" if opt_type == "C" else "p"
    return f"{ticker} {expiry_label} {strike:g}{opt_label}"


def build_spreads(
    df,
    metric_choice,
    call_delta_range,
    contract_step,
    max_contract_ratio,
    max_straddle_ratio,
    min_option_price,
    max_last_price,
    max_abs_net_delta,
    max_legs_per_side,
    max_results,
):
    sdf = (
        compute_overvalued(df, metric_choice)
        .with_columns(
            pl.when(pl.col("Type") == "C")
            .then(pl.col("Delta"))
            .otherwise(-pl.col("Delta"))
            .alias("CallDelta")
        )
        .with_columns(pl.col("Last").alias("LastPrice"))
        .filter(
            (pl.col("CallDelta") >= call_delta_range[0] / 100)
            & (pl.col("CallDelta") <= call_delta_range[1] / 100)
            & (pl.col("LastPrice") >= min_option_price)
            & (pl.col("LastPrice") <= max_last_price)
        )
    )

    if sdf.is_empty():
        return []

    sorted_legs = sdf.sort("%Overvalued")
    buys = sorted_legs.head(max_legs_per_side)
    sells = sorted_legs.tail(max_legs_per_side)

    spreads = []

    for buy in buys.to_dicts():
        buy_delta = float(buy["Delta"])
        if buy_delta == 0:
            continue

        for sell in sells.to_dicts():
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
            sell_qty = round_to_step(abs(buy_delta) / abs(sell_delta) * contract_step, contract_step)

            if sell_qty <= 0:
                continue

            contract_ratio = max(buy_qty, sell_qty) / min(buy_qty, sell_qty)
            if contract_ratio > max_contract_ratio:
                continue

            is_straddle = (
                buy["Expiry"] == sell["Expiry"]
                and buy["Strike"] == sell["Strike"]
                and buy["Type"] != sell["Type"]
            )
            if is_straddle and contract_ratio > max_straddle_ratio:
                continue

            net_delta = buy_delta * buy_qty - sell_delta * sell_qty
            if abs(net_delta) > max_abs_net_delta:
                continue

            edge = float(sell["%Overvalued"]) - float(buy["%Overvalued"])
            if edge <= 0:
                continue

            spreads.append(
                {
                    "Buy": format_contract(buy["Ticker"], buy["Expiry"], buy["Strike"], buy["Type"]),
                    "Sell": format_contract(sell["Ticker"], sell["Expiry"], sell["Strike"], sell["Type"]),
                    "Buy Qty": buy_qty,
                    "Sell Qty": sell_qty,
                    "Net Delta": net_delta,
                    "Edge": edge,
                    "BuyKey": buy,
                    "SellKey": sell,
                }
            )

    return sorted(spreads, key=lambda r: r["Edge"], reverse=True)[:max_results]


def main():
    user_repo = UserRepository.from_env()
    if not login_gate(user_repo):
        st.info("Please log in to view the option chain.")
        st.stop()

    source = load_from_env()
    close_date = None

    if isinstance(source, GCSClosesDataSource):
        available_dates = source.list_available_dates()
        latest_date = available_dates[-1]

        selected_date = st.sidebar.date_input(
            "Close date",
            value=latest_date,
            min_value=available_dates[0],
            max_value=latest_date,
            key="close_date",
        )

        close_date = selected_date.strftime("%Y-%m-%d")

    pricing_model = st.sidebar.selectbox(
        "Pricing library",
        ["mzpricer", "quantlib"],
        index=0,
        key="pricing_model",
    )

    df = load_data(pricing_model, close_date)

    tickers = sorted(df["Ticker"].unique().to_list())

    if "ticker" not in st.session_state:
        st.session_state.ticker = tickers[0]

    if st.session_state.ticker not in tickers:
        st.session_state.ticker = tickers[0]

    st.session_state.ticker = st.sidebar.selectbox(
        "Ticker",
        tickers,
        index=tickers.index(st.session_state.ticker),
    )

    ticker = st.session_state.ticker

    df = df.filter(pl.col("Ticker") == ticker)

    metric_choice = st.sidebar.selectbox(
        "Overvaluation relative to:",
        ["Last", "Mid", "Bid", "Ask"],
        index=0,
        key="metric_choice",
    )

    call_delta_min, call_delta_max = st.sidebar.slider(
        "Call-Delta Range (0 = OTM, 100 = ITM)",
        min_value=0,
        max_value=100,
        value=(5, 95),
        step=1,
        key="delta_range",
    )

    greek_columns = ["Delta"] if "Delta" in df.columns else []

    st.sidebar.subheader("Spread Filters")

    contract_step = st.sidebar.number_input(
        "Contracts per leg",
        min_value=10,
        value=10,
        step=10,
        key="contract_step",
    )

    max_contract_ratio = st.sidebar.number_input(
        "Max contract ratio",
        min_value=1.0,
        value=2.5,
        step=0.1,
        key="contract_ratio",
    )

    max_straddle_ratio = st.sidebar.number_input(
        "Max straddle ratio",
        min_value=1.0,
        value=1.5,
        step=0.1,
        key="straddle_ratio",
    )

    min_option_price = st.sidebar.number_input(
        "Min option price",
        min_value=0.0,
        value=2.0,
        step=0.05,
        key="min_option_price",
    )

    max_last_price = st.sidebar.number_input(
        "Max last price",
        min_value=0.0,
        value=1000.0,
        step=1.0,
        key="max_last",
    )

    max_abs_net_delta = st.sidebar.number_input(
        "Max |net delta|",
        min_value=0.0,
        value=5.0,
        step=0.5,
        key="net_delta",
    )

    max_legs_per_side = st.sidebar.number_input(
        "Max legs per side",
        min_value=10,
        value=50,
        step=10,
        key="legs_side",
    )

    max_results = st.sidebar.number_input(
        "Max spread ideas",
        min_value=10,
        value=50,
        step=10,
        key="results",
    )

    spot = float(df["Spot"][0])
    div = df["DividendYield"].mean()
    vol = float(df["Vol30d"][0])
    valuation_time = df["ValuationTime"].max()

    st.subheader(f"{ticker} – {valuation_time.date()}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Last", f"{spot:.2f}")
    c2.metric("Dividend Yield", f"{div:.2f}")
    c3.metric("30d Volatility", f"{vol:.2f}")

    st.markdown("---")

    chain_tab, spreads_tab = st.tabs(["Option Chain", "Spreads"])

    with chain_tab:
        render_chain(df, metric_choice, (call_delta_min, call_delta_max), min_option_price, greek_columns)

    with spreads_tab:
        spreads = build_spreads(
            df,
            metric_choice,
            (call_delta_min, call_delta_max),
            int(contract_step),
            max_contract_ratio,
            max_straddle_ratio,
            min_option_price,
            max_last_price,
            max_abs_net_delta,
            int(max_legs_per_side),
            int(max_results),
        )

        if not spreads:
            st.info("No spreads matched the filters.")
            return

        display_pd = (
            pl.DataFrame(spreads)
            .drop(["BuyKey", "SellKey"])
            .with_columns(
                pl.col("Edge").cast(float).round(4),
                pl.col("Net Delta").cast(float).round(4),
            )
            .to_pandas()
            .reset_index(drop=True)
        )

        st.dataframe(
            display_pd,
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            key="spreads_table",
            hide_index=True,
        )

        selection = st.session_state.get("spreads_table", {}).get("selection", {})
        rows = selection.get("rows", [])

        if not rows:
            st.info("Select a spread to view details.")
            return

        idx = rows[0]

        if not (0 <= idx < len(spreads)):
            st.info("Selection changed after refresh.")
            return

        selected_spread = spreads[idx]

        def leg_details(leg_key):
            cols = [
                "Ticker",
                "Expiry",
                "Strike",
                "Type",
                "Last",
                "Bid",
                "Ask",
                "Mid",
                "FMV",
                "Prob ITM",
                "Prob OTM",
                *greek_columns,
            ]
            available = [c for c in cols if c in df.columns]

            leg_filter = (
                (pl.col("Ticker") == leg_key["Ticker"])
                & (pl.col("Expiry") == leg_key["Expiry"])
                & (pl.col("Strike") == leg_key["Strike"])
                & (pl.col("Type") == leg_key["Type"])
            )

            return df.filter(leg_filter).select(available)

        st.subheader("Leg 1 (Buy)")
        st.dataframe(leg_details(selected_spread["BuyKey"]).to_pandas(), use_container_width=True)
        st.subheader("Leg 2 (Sell)")
        st.dataframe(leg_details(selected_spread["SellKey"]).to_pandas(), use_container_width=True)


if __name__ == "__main__":
    main()
