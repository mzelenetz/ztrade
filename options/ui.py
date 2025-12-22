import streamlit as st
import polars as pl
from datetime import datetime
from pricing_utils import CBOEOptionsData, OptionsPrices

st.set_page_config(layout="wide")

def render_chain(df: pl.DataFrame):
    # Make sure Expiry is a string or datetime
    # Group by expiry
    for expiry, subdf in df.group_by("Expiry"):
        # Convert the sub-DataFrame to pandas for ease of display
        sdf = subdf.collect().to_pandas()

        with st.expander(f"Expiration: {expiry}", expanded=False):
            st.dataframe(sdf, use_container_width=True)

def compute_overvalued(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col(metric) / pl.col("FMV")) - 1.0).alias("%Overvalued")
    )
# -------------------------------------------------------------
# Load your option data
# -------------------------------------------------------------
@st.cache_data
def load_data():
    # replace this with your actual Polars loading
    opts = CBOEOptionsData("/Users/mz/Documents/projects/ztrade/UnderlyingOptionsEODCalcs_2023-08-25_cgi_or_historical.csv", "2023-08-25", [])
    df_opts = opts.get_data()
    prices = OptionsPrices(df_opts)
    df = prices.price_options() 
    return df

df = load_data()

ticker = st.sidebar.selectbox("Ticker", df["Ticker"].unique().to_list())
metric_choice = st.sidebar.selectbox(
    "Overvaluation relative to:",
    ["Last", "Mid", "Bid", "Ask"],
    index=0
)

call_delta_min, call_delta_max = st.sidebar.slider(
    "Call-Delta Range (0 = OTM, 100 = ITM)",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=1,
)

df = df.filter(pl.col("Ticker") == ticker)

# -------------------------------------------------------------
# Underlying summary box
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Split calls and puts
# -------------------------------------------------------------
# calls = df.filter(pl.col("Type") == "C")
# puts  = df.filter(pl.col("Type") == "P")

# expiries = sorted(df["Expiry"].unique())

# -------------------------------------------------------------
# Render each expiry as two side-by-side tables
# -------------------------------------------------------------
for (expiry,), sdf in df.sort("Expiry").group_by("Expiry"):
    sdf = compute_overvalued(sdf, metric_choice)

    with st.expander(expiry.strftime("%d-%b-%y"), expanded=True):
        sdf = (sdf
            .with_columns(
                pl.when(pl.col("Type") == "C")
                .then(pl.col("Delta"))          # calls:   +0 → +1
                .otherwise(-pl.col("Delta")).alias("CallDelta")
            )
            .filter(
                (pl.col("CallDelta") >= call_delta_min / 100) &
                (pl.col("CallDelta") <= call_delta_max / 100)
            ))
        
        # Split calls / puts *in Polars*
        calls = sdf.filter(pl.col("Type") == "C")
        puts  = sdf.filter(pl.col("Type") == "P")

        # Sort by strike
        calls_e = calls.sort("Strike")
        puts_e  = puts.sort("Strike")

        # Format bid–ask
        def bid_ask(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                (pl.col("Bid").cast(str) + " – " + pl.col("Ask").cast(str)).alias("BidAsk")
            )

        calls_e = bid_ask(calls_e)
        puts_e  = bid_ask(puts_e)

        # Compute %Overvalued = Last / FMV
        calls_e = calls_e.with_columns(((pl.col("Last") / pl.col("FMV"))).alias("%Overvalued"))
        puts_e  = puts_e.with_columns(((pl.col("Last") / pl.col("FMV"))).alias("%Overvalued"))

        # Layout: Calls on left, Puts on right
        colL, colR = st.columns(2)

        with colL:
            st.subheader("Calls")
            st.dataframe(
                calls_e.select([
                    "Strike", "FMV", "Last", "BidAsk", "Delta", "%Overvalued"
                ]).to_pandas(),
                use_container_width=True,
            )

        with colR:
            st.subheader("Puts")
            st.dataframe(
                puts_e.select([
                    "Strike", "FMV", "Last", "BidAsk", "Delta", "%Overvalued"
                ]).to_pandas(),
                use_container_width=True,
            )

    st.markdown("---")


# -------------------------------------------------------------
# Positions table (bottom block of your sheet)
# -------------------------------------------------------------
st.subheader("Positions / DR-CR Summary")

# You can load this from your blotter / df
positions_df = pl.DataFrame({
    "Ticker": [ticker],
    "Expiry": ["2025-01-07"],
    "Strike": [104],
    "Call/Put": ["C"],
    "Qty": [10],
    "FMV": [6.325],
    "DR/CR": ["DR"],
})

st.dataframe(positions_df)
