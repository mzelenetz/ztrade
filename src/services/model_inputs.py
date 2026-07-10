from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
import polars as pl

# (days, annualized rate) — editable in the UI; these are just starting values.
DEFAULT_RATE_CURVE: list[tuple[float, float]] = [
    (30, 0.0430),
    (91, 0.0432),
    (182, 0.0428),
    (365, 0.0415),
    (730, 0.0405),
    (1095, 0.0400),
]

VOL_CLAMP = (0.01, 3.0)
MAX_VALID_IV = 2.0  # quotes above 200% vol are treated as junk
MIN_SMILE_POINTS = 5
# Fit only within ±4 standard deviations of the forward (z = ln(K/F) / (σ_atm·√T)).
# This concentrates the fit on each expiry's actually-tradeable strike region —
# a 5-day smile lives within a few percent of spot; a 2-year one spans far more.
MAX_ABS_Z = 4.0


class SmileFit:
    """Quadratic smile iv(z) = a + b·z + c·z² in standardized moneyness
    z = ln(K/F) / (σ_atm·√T), valid on [z_lo, z_hi]; evaluation clamps z to
    that range (flat wings) so the parabola never extrapolates into nonsense."""

    def __init__(self, a: float, b: float, c: float, scale: float, z_lo: float, z_hi: float):
        self.a, self.b, self.c = a, b, c
        self.scale = scale  # σ_atm·√T
        self.z_lo, self.z_hi = z_lo, z_hi

    def _z(self, strike: float, forward: float) -> float:
        return log(strike / forward) / self.scale

    def __call__(self, strike: float, forward: float) -> float:
        z = min(max(self._z(strike, forward), self.z_lo), self.z_hi)
        iv = self.a + self.b * z + self.c * z**2
        return min(max(iv, VOL_CLAMP[0]), VOL_CLAMP[1])

    def in_range(self, strike: float, forward: float) -> bool:
        return self.z_lo <= self._z(strike, forward) <= self.z_hi


def interpolate_rate(curve: list[tuple[float, float]], t_years: float) -> float:
    """Linear interpolation of the (days, rate) curve at t_years, flat beyond the ends."""
    if not curve:
        raise ValueError("rate curve must have at least one point")

    days = t_years * 365.0
    points = sorted(curve)

    if days <= points[0][0]:
        return points[0][1]
    if days >= points[-1][0]:
        return points[-1][1]

    for (d0, r0), (d1, r1) in zip(points, points[1:]):
        if d0 <= days <= d1:
            w = (days - d0) / (d1 - d0)
            return r0 + w * (r1 - r0)

    return points[-1][1]  # unreachable, defensive


def _valid_iv(iv) -> bool:
    return iv is not None and VOL_CLAMP[0] < iv < MAX_VALID_IV


def implied_vol_from_price(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    div_yield: float,
    tenor: float,
    opt_type: str,
) -> float | None:
    """European BSM implied vol from a price, by bisection.

    The vendor's implied_volatility_1545 column embeds the vendor's own
    rate/dividend assumptions; when those differ from the market's actual
    forward, same-strike call and put IVs split into two parallel branches
    that no single smile can fit. Re-inverting the mid prices under our own
    (market-implied) carry collapses the branches back onto one curve.
    """
    from src.services.margin_service import bs_price

    if price is None or price <= 0 or spot <= 0 or strike <= 0 or tenor <= 0:
        return None

    lo, hi = VOL_CLAMP[0], VOL_CLAMP[1]
    p_lo = bs_price(spot, strike, rate, div_yield, lo, tenor, opt_type)
    p_hi = bs_price(spot, strike, rate, div_yield, hi, tenor, opt_type)
    if not (p_lo <= price <= p_hi):
        return None  # below intrinsic/carry floor or above the vol cap — junk quote

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(spot, strike, rate, div_yield, mid, tenor, opt_type) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return 0.5 * (lo + hi)


# Sanity band for market-implied dividend/borrow yield. Negative q is a real
# signal (hard-to-borrow premium), so allow moderately negative values.
IMPLIED_DIV_BAND = (-0.05, 0.15)
MIN_PARITY_PAIRS = 3
PARITY_MONEYNESS_BAND = 0.25  # |ln(K/S)| — near-ATM pairs where American≈European


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v = np.array(values)[order]
    w = np.array(weights)[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def _relative_spread(bid, ask, mid) -> float | None:
    """Quoted spread as a fraction of mid — the market-quality signal."""
    if bid is None or ask is None or mid is None or mid <= 0 or ask < bid or bid < 0:
        return None
    return (ask - bid) / mid


def fit_implied_forward(
    strikes: list[float],
    types: list[str],
    mids: list[float],
    spot: float,
    tenor: float,
    rate: float,
    bids: list | None = None,
    asks: list | None = None,
) -> float | None:
    """Market-implied forward for one expiry via put-call parity.

    Parity gives F = (C − P)·e^(rT) + K at every strike; the spread-weighted
    median across near-ATM pairs is the implied forward. Weighting each pair
    by the inverse of its legs' combined relative bid/ask spread lets tight
    markets dominate — wide, stale quotes (deep ITM legs especially) otherwise
    skew the forward and re-split same-strike call/put IVs.

    The discount rate deliberately comes from the user's curve, not from a
    parity regression slope: American puts' early-exercise premium biases a
    fitted r several points low, but that bias largely cancels in the forward,
    so the implied carry (r − q) — the thing that moves prices — is reliable.
    Implying clean rates requires European options (index boxes).
    """
    if tenor <= 0 or spot <= 0:
        return None

    n = len(strikes)
    bids = bids if bids is not None else [None] * n
    asks = asks if asks is not None else [None] * n

    calls: dict[float, tuple[float, float]] = {}
    puts: dict[float, tuple[float, float]] = {}
    for k, t, mid, bid, ask in zip(strikes, types, mids, bids, asks):
        if mid is None or mid <= 0.05 or k <= 0:
            continue
        if abs(log(k / spot)) > PARITY_MONEYNESS_BAND:
            continue
        rel = _relative_spread(bid, ask, mid)
        if rel is not None and rel > 0.5:
            continue  # effectively no market
        (calls if t == "C" else puts)[k] = (mid, rel if rel is not None else 0.05)

    common = sorted(set(calls) & set(puts))
    if len(common) < MIN_PARITY_PAIRS:
        return None

    growth = exp(rate * tenor)
    forwards, weights = [], []
    for k in common:
        (c_mid, c_rel), (p_mid, p_rel) = calls[k], puts[k]
        forwards.append((c_mid - p_mid) * growth + k)
        weights.append(1.0 / max(c_rel + p_rel, 0.005))

    forward = _weighted_median(forwards, weights)
    if forward <= 0:
        return None

    return forward


def fit_implied_dividend(
    strikes: list[float],
    types: list[str],
    mids: list[float],
    spot: float,
    tenor: float,
    rate: float,
    bids: list | None = None,
    asks: list | None = None,
) -> float | None:
    """Market-implied dividend/borrow yield: q = r − ln(F/S)/T from the parity forward."""
    forward = fit_implied_forward(strikes, types, mids, spot, tenor, rate, bids, asks)
    if forward is None:
        return None

    div_yield = rate - log(forward / spot) / tenor
    if not (IMPLIED_DIV_BAND[0] <= div_yield <= IMPLIED_DIV_BAND[1]):
        return None

    return float(div_yield)


def dividend_pv(
    schedule: list[tuple[str, float]],
    valuation,
    expiry,
    rate: float,
) -> float:
    """PV of scheduled dividends with valuation < ex-date ≤ expiry (escrowed model).

    `schedule` entries are (ISO date string, amount); `valuation`/`expiry` are dates.
    """
    from datetime import date as _date

    total = 0.0
    for ex_date_str, amount in schedule:
        ex_date = _date.fromisoformat(str(ex_date_str))
        if valuation < ex_date <= expiry:
            t_years = (ex_date - valuation).days / 365.0
            total += float(amount) * exp(-rate * t_years)
    return total


def escrowed_equivalent_yield(div_pv: float, spot: float, tenor: float) -> float | None:
    """The per-expiry continuous yield that reproduces the escrowed-dividend price:
    S·e^(−q·T) = S − PV  →  q = −ln(1 − PV/S)/T. Returns None when the schedule
    is implausible (PV ≥ 90% of spot) or T ≤ 0."""
    if tenor <= 0 or spot <= 0 or div_pv >= 0.9 * spot:
        return None
    if div_pv <= 0:
        return 0.0
    return -log(1.0 - div_pv / spot) / tenor


def fit_smile(
    strikes: list[float],
    ivs: list[float],
    types: list[str],
    forward: float,
    tenor: float,
) -> SmileFit | None:
    """Least-squares quadratic smile in standardized moneyness, fit from OTM
    quotes only (puts below the forward, calls above) — OTM IVs are far cleaner
    than deep-ITM ones. Returns None when there aren't enough valid points;
    callers fall back to the median IV. One outlier-rejection refit pass (>3σ)."""
    if tenor <= 0:
        return None

    otm = [
        (log(k / forward), iv)
        for k, iv, t in zip(strikes, ivs, types)
        if k > 0
        and _valid_iv(iv)
        and ((t == "P" and k <= forward) or (t == "C" and k >= forward))
    ]
    if len(otm) < MIN_SMILE_POINTS:
        return None

    # ATM vol estimate: median IV of the few quotes nearest the forward.
    otm.sort(key=lambda p: abs(p[0]))
    atm_iv = float(np.median([iv for _, iv in otm[:5]]))
    scale = atm_iv * sqrt(tenor)
    if scale <= 0:
        return None

    pairs = [(x / scale, iv) for x, iv in otm if abs(x / scale) <= MAX_ABS_Z]
    if len(pairs) < MIN_SMILE_POINTS:
        return None

    zs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])

    c, b, a = np.polyfit(zs, ys, 2)
    resid = ys - (a + b * zs + c * zs**2)
    sigma = resid.std()

    if sigma > 0:
        keep = np.abs(resid) <= 3 * sigma
        if MIN_SMILE_POINTS <= keep.sum() < len(pairs):
            zs, ys = zs[keep], ys[keep]
            c, b, a = np.polyfit(zs, ys, 2)

    return SmileFit(float(a), float(b), float(c), scale, float(zs.min()), float(zs.max()))


def _median_iv(ivs: list[float]) -> float | None:
    valid = [iv for iv in ivs if _valid_iv(iv)]
    if not valid:
        return None
    return float(np.median(valid))


MIN_SURFACE_EXPIRIES = 2
MIN_SURFACE_POINTS = 12


class VolSurface:
    """Global smile surface for one ticker, fit across all expiries at once.

    Shape model: y = σ/σ_atm(τ) − 1 = b(τ)·z + c(τ)·z², with
    z = ln(K/F)/(σ_atm·√τ), b(τ) = b0 + b1·ln τ, c(τ) = max(c0 + c1·ln τ, 0).

    Fitting each expiry independently lets the dense near-term strike grid look
    authoritative while sparse long-dated expiries produce noisy, tilted skews.
    Pooling every expiry into one 4-parameter weighted least squares — each
    expiry weighted equally regardless of its strike count — lets long tenors
    borrow the smile shape from the whole surface while ln τ terms let skew and
    curvature evolve smoothly with maturity.
    """

    def __init__(self, b0: float, b1: float, c0: float, c1: float):
        self.b0, self.b1, self.c0, self.c1 = b0, b1, c0, c1

    def relative_smile(self, z: float, tenor: float) -> float:
        s = log(tenor)
        b = self.b0 + self.b1 * s
        c = max(self.c0 + self.c1 * s, 0.0)  # never concave
        return b * z + c * z**2

    def vol(self, atm_iv: float, z: float, tenor: float) -> float:
        iv = atm_iv * (1.0 + self.relative_smile(z, tenor))
        return min(max(iv, VOL_CLAMP[0]), VOL_CLAMP[1])


def fit_vol_surface(expiry_points: list[dict]) -> VolSurface | None:
    """Weighted least squares over all expiries' (z, y) points.

    `expiry_points` entries: {"tenor": float, "points": [(z, y), ...]}.
    Each expiry gets total weight 1 (weight 1/n per point) so the near-term
    grid's strike count can't dominate the long end. One >3σ outlier-rejection
    refit pass, mirroring the per-expiry fitter.
    """
    rows, ys, ws = [], [], []
    n_expiries = 0
    for e in expiry_points:
        pts = [(p[0], p[1], p[2] if len(p) > 2 else 1.0) for p in e["points"]]
        tenor = e["tenor"]
        if not pts or tenor <= 0:
            continue
        n_expiries += 1
        s = log(tenor)
        # normalize quote-quality weights so each expiry contributes total 1
        w_total = sum(w for _, _, w in pts)
        for z, y, w in pts:
            rows.append([z, s * z, z * z, s * z * z])
            ys.append(y)
            ws.append(w / w_total)

    if n_expiries < MIN_SURFACE_EXPIRIES or len(rows) < MIN_SURFACE_POINTS:
        return None

    X = np.array(rows)
    Y = np.array(ys)
    sw = np.sqrt(np.array(ws))

    coeffs, *_ = np.linalg.lstsq(X * sw[:, None], Y * sw, rcond=None)
    resid = Y - X @ coeffs
    sigma = float(np.std(resid))
    if sigma > 0:
        keep = np.abs(resid) <= 3 * sigma
        if MIN_SURFACE_POINTS <= keep.sum() < len(rows):
            coeffs, *_ = np.linalg.lstsq(
                (X[keep]) * sw[keep][:, None], Y[keep] * sw[keep], rcond=None
            )

    b0, b1, c0, c1 = (float(v) for v in coeffs)
    return VolSurface(b0, b1, c0, c1)


# The global surface acts as a prior worth this many observations in each
# expiry's own fit: dense near-term expiries keep their own shape (their data
# swamps the prior), sparse long-dated ones are pulled toward the pooled
# surface instead of chasing a handful of noisy quotes.
SURFACE_PRIOR_WEIGHT = 10.0


def blend_expiry_smile(
    points: list[tuple[float, float]],
    prior_b: float,
    prior_c: float,
    kappa: float = SURFACE_PRIOR_WEIGHT,
) -> tuple[float, float, float]:
    """Per-expiry quadratic y = a + b·z + c·z², ridge-shrunk toward the global
    surface's (b, c) for this tenor (partial pooling), with quotes weighted by
    market tightness. The level `a` is free — ATM is well-observed at every
    tenor and must never be biased by the prior."""
    pts = [(p[0], p[1], p[2] if len(p) > 2 else 1.0) for p in points]
    X = np.array([[1.0, z, z * z] for z, _, _ in pts])
    Y = np.array([y for _, y, _ in pts])
    W = np.array([w for _, _, w in pts])
    W = W * (len(pts) / W.sum())  # mean weight 1 so kappa keeps its meaning

    def solve(Xm, Ym, Wm):
        Xw = Xm * Wm[:, None]
        A = Xw.T @ Xm + np.diag([0.0, kappa, kappa])
        rhs = Xw.T @ Ym + np.array([0.0, kappa * prior_b, kappa * prior_c])
        return np.linalg.solve(A, rhs)

    theta = solve(X, Y, W)
    resid = Y - X @ theta
    sigma = float(np.std(resid))
    if sigma > 0:
        keep = np.abs(resid) <= 3 * sigma
        if 3 <= keep.sum() < len(pts):
            theta = solve(X[keep], Y[keep], W[keep])

    a, b, c = theta
    return float(a), float(b), float(c)


def apply_model_inputs(
    df: pl.DataFrame,
    vol_mode: str,
    dividends: dict[str, float],
    rate_curve: list[tuple[float, float]],
    default_vol: float,
    carry_mode: str = "manual",
    dividend_schedule: dict[str, list[tuple[str, float]]] | None = None,
) -> pl.DataFrame:
    """Set Rate (from the curve) and DividendYield per (Ticker, Expiry) by
    priority — discrete schedule (escrowed model, as an exact equivalent
    yield) → market-implied parity forward (when carry_mode='implied') →
    manual per-ticker yield — then the pricing vol per vol_mode. Adds
    FittedVol (the vol the model used) and DivSource ('schedule' | 'implied'
    | 'manual')."""

    df = df.with_columns(
        pl.col("Ticker")
        .map_elements(lambda t: float(dividends.get(t, 0.0)), return_dtype=pl.Float64)
        .alias("DividendYield"),
        pl.col("T")
        .map_elements(lambda t: interpolate_rate(rate_curve, t), return_dtype=pl.Float64)
        .alias("Rate"),
        pl.lit("manual").alias("DivSource"),
    )

    df = _apply_dividends(df, carry_mode, dividend_schedule or {})
    df = _recompute_market_ivs(df)

    if vol_mode == "flat":
        df = df.with_columns(pl.lit(default_vol).alias("Vol30d"))
    elif vol_mode == "surface":
        df = _apply_surface_vols(df, default_vol)
    # historical: keep Vol30d as produced by CBOEOptionsData._get_vols

    if "VolFromSurface" not in df.columns:
        df = df.with_columns(pl.lit(False).alias("VolFromSurface"))

    return df.with_columns(pl.col("Vol30d").alias("FittedVol"))


def _apply_dividends(
    df: pl.DataFrame,
    carry_mode: str,
    dividend_schedule: dict[str, list[tuple[str, float]]],
) -> pl.DataFrame:
    """Per (Ticker, Expiry): schedule → implied → keep manual."""
    groups = []
    for (_, _), group in df.group_by(["Ticker", "Expiry"], maintain_order=True):
        first = group.row(0, named=True)
        ticker = first["Ticker"]
        tenor = float(first["T"])
        spot = float(first["Spot"])
        rate = float(first["Rate"])

        schedule = dividend_schedule.get(ticker) or []
        if schedule:
            valuation = first["ValuationTime"].date()
            expiry = first["Expiry"].date()
            pv = dividend_pv(schedule, valuation, expiry, rate)
            q_eff = escrowed_equivalent_yield(pv, spot, tenor)
            if q_eff is not None:
                groups.append(
                    group.with_columns(
                        pl.lit(q_eff).alias("DividendYield"),
                        pl.lit("schedule").alias("DivSource"),
                    )
                )
                continue

        if carry_mode == "implied":
            div_yield = fit_implied_dividend(
                strikes=group["Strike"].to_list(),
                types=group["Type"].to_list(),
                mids=group["Mid"].to_list(),
                spot=spot,
                tenor=tenor,
                rate=rate,
                bids=group["Bid"].to_list(),
                asks=group["Ask"].to_list(),
            )
            if div_yield is not None:
                groups.append(
                    group.with_columns(
                        pl.lit(div_yield).alias("DividendYield"),
                        pl.lit("implied").alias("DivSource"),
                    )
                )
                continue

        groups.append(group)

    return pl.concat(groups) if groups else df


def _recompute_market_ivs(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the vendor MarketIV with IVs inverted from mid quotes under our
    own carry (row Rate / DividendYield), so same-strike call and put IVs obey
    parity and lie on a single fittable smile."""

    def invert(row: dict) -> float | None:
        price = row["Mid"] if row["Mid"] is not None and row["Mid"] > 0 else row["Last"]
        return implied_vol_from_price(
            price=price,
            spot=row["Spot"],
            strike=row["Strike"],
            rate=row["Rate"],
            div_yield=row["DividendYield"],
            tenor=row["T"],
            opt_type=row["Type"],
        )

    return df.with_columns(
        pl.struct(["Mid", "Last", "Spot", "Strike", "Rate", "DividendYield", "T", "Type"])
        .map_elements(invert, return_dtype=pl.Float64)
        .alias("MarketIV")
    )


def _expiry_smile_inputs(group: pl.DataFrame) -> dict | None:
    """Per-expiry preliminaries for the surface fit: forward, ATM vol, and the
    OTM quotes as (z, y) points in standardized moneyness / relative vol."""
    first = group.row(0, named=True)
    tenor = float(first["T"])
    if tenor <= 0:
        return None

    forward = float(first["Spot"]) * exp(
        (float(first["Rate"]) - float(first["DividendYield"])) * tenor
    )

    otm = []
    for k, iv, t, bid, ask, mid in zip(
        group["Strike"], group["MarketIV"], group["Type"],
        group["Bid"], group["Ask"], group["Mid"],
    ):
        if k is None or k <= 0 or not _valid_iv(iv):
            continue
        if not ((t == "P" and k <= forward) or (t == "C" and k >= forward)):
            continue
        rel = _relative_spread(bid, ask, mid)
        # tighter markets get more say; a quote with no usable spread info
        # gets a middling weight, an untradeably wide one almost none
        weight = 1.0 / max(rel if rel is not None else 0.05, 0.005)
        otm.append((log(k / forward), iv, weight))
    if len(otm) < 3:
        return None

    otm.sort(key=lambda p: abs(p[0]))
    atm_iv = float(np.median([iv for _, iv, _ in otm[:5]]))
    scale = atm_iv * sqrt(tenor)
    if scale <= 0:
        return None

    points = [
        (x / scale, iv / atm_iv - 1.0, w)
        for x, iv, w in otm
        # |y| > 2 (an IV more than 3x / less than a third of ATM) is a junk
        # quote, not smile shape
        if abs(x / scale) <= MAX_ABS_Z and abs(iv / atm_iv - 1.0) <= 2.0
    ]
    if not points:
        return None

    zs = [z for z, _, _ in points]
    return {
        "tenor": tenor,
        "forward": forward,
        "atm_iv": atm_iv,
        "scale": scale,
        "points": points,
        "z_lo": min(zs),
        "z_hi": max(zs),
    }


def _apply_surface_vols(df: pl.DataFrame, default_vol: float) -> pl.DataFrame:
    """Vol per contract from a single global surface fit per ticker (smile shape
    pooled across expiries in standardized moneyness × ln-time), falling back to
    the independent per-expiry quadratic, then to the median IV."""
    out = []
    for (_,), tdf in df.group_by(["Ticker"], maintain_order=True):
        expiry_groups = [
            (group, _expiry_smile_inputs(group))
            for (_, _), group in tdf.group_by(["Ticker", "Expiry"], maintain_order=True)
        ]

        surface = fit_vol_surface(
            [
                {"tenor": info["tenor"], "points": info["points"]}
                for _, info in expiry_groups
                if info is not None
            ]
        )

        for group, info in expiry_groups:
            strikes = group["Strike"].to_list()
            ivs = group["MarketIV"].to_list()

            if info is not None and surface is not None:
                # Partial pooling: this expiry's own quadratic, shrunk toward the
                # global surface's skew/curvature at its tenor.
                s = log(info["tenor"])
                prior_b = surface.b0 + surface.b1 * s
                prior_c = max(surface.c0 + surface.c1 * s, 0.0)
                a, b, c = blend_expiry_smile(info["points"], prior_b, prior_c)

                def smile_vol(z: float) -> float:
                    iv = info["atm_iv"] * (1.0 + a + b * z + max(c, 0.0) * z * z)
                    return min(max(iv, VOL_CLAMP[0]), VOL_CLAMP[1])

                # Inside the expiry's own quoted z-range the surface is the
                # model's view; outside it the model has no opinion — use the
                # contract's own market IV so wings show ~zero edge instead of
                # extrapolation artifacts.
                vols, from_surface = [], []
                for k, iv in zip(strikes, ivs):
                    z = log(k / info["forward"]) / info["scale"] if k and k > 0 else None
                    if z is not None and info["z_lo"] <= z <= info["z_hi"]:
                        vols.append(smile_vol(z))
                        from_surface.append(True)
                    elif _valid_iv(iv):
                        vols.append(iv)
                        from_surface.append(False)
                    else:
                        z_clamped = min(max(z, info["z_lo"]), info["z_hi"]) if z is not None else 0.0
                        vols.append(smile_vol(z_clamped))
                        from_surface.append(False)
            else:
                # Per-expiry fallback (single-expiry tickers, failed global fit).
                first = group.row(0, named=True)
                tenor = float(first["T"])
                forward = float(first["Spot"]) * exp(
                    (float(first["Rate"]) - float(first["DividendYield"])) * tenor
                )
                types = group["Type"].to_list()
                smile = fit_smile(strikes, ivs, types, forward, tenor)
                if smile is not None:
                    vols, from_surface = [], []
                    for k, iv in zip(strikes, ivs):
                        if smile.in_range(k, forward):
                            vols.append(smile(k, forward))
                            from_surface.append(True)
                        elif _valid_iv(iv):
                            vols.append(iv)
                            from_surface.append(False)
                        else:
                            vols.append(smile(k, forward))
                            from_surface.append(False)
                else:
                    fallback = _median_iv(ivs) or default_vol
                    vols = [fallback] * len(strikes)
                    from_surface = [False] * len(strikes)

            out.append(
                group.with_columns(
                    pl.Series("Vol30d", vols, dtype=pl.Float64),
                    pl.Series("VolFromSurface", from_surface, dtype=pl.Boolean),
                )
            )

    return pl.concat(out) if out else df
