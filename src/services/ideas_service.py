from __future__ import annotations

# Cross-ticker spread recommendations.
#
# The Spreads tab ranks within one name by net edge at mid/last quotes. An
# *idea* has to clear a higher bar: the edge should survive realistic fills,
# the quotes behind it should be believable, and capital efficiency matters
# when comparing across names. Scoring is deliberately transparent — a few
# discrete checks, no opaque weights.

MAX_PER_TICKER = 5
MAX_IDEAS = 20

REL_SPREAD_SOFT_LIMIT = 0.08  # legs quoted wider than 8% of mid are suspect
MIN_LEG_VOLUME = 25           # fewer contracts traded → quote may be stale


def _leg_rel_spread(leg: dict) -> float | None:
    bid, ask, mid = leg.get("bid"), leg.get("ask"), leg.get("mid")
    if bid is None or ask is None or not mid or mid <= 0 or ask < bid:
        return None
    return (ask - bid) / mid


def assess_spread(spread: dict) -> dict | None:
    """Score one spread as an idea. Returns None when it can't qualify at all
    (no executable edge computable, or non-positive net edge)."""
    net = spread.get("netEdgeDollars")
    exec_edge = spread.get("execEdgeDollars")
    capital = spread.get("capitalEmployed") or 0.0
    if net is None or net <= 0:
        return None

    flags: list[str] = []

    if exec_edge is None:
        flags.append("no executable quotes on a leg")
    elif exec_edge <= 0:
        flags.append("edge does not survive bid/ask cross")

    for name, leg in (("buy", spread["buyLeg"]), ("sell", spread["sellLeg"])):
        rel = _leg_rel_spread(leg)
        if rel is None:
            flags.append(f"{name} leg missing quotes")
        elif rel > REL_SPREAD_SOFT_LIMIT:
            flags.append(f"{name} leg market {rel:.0%} wide")

        vol = leg.get("volume")
        if vol is not None and vol < MIN_LEG_VOLUME:
            flags.append(f"{name} leg volume {int(vol)}")

        if leg.get("volFromSurface") is False:
            flags.append(f"{name} leg vol not from fitted surface")

    # Confidence: does the edge survive execution, and are the quotes real?
    if exec_edge is not None and exec_edge > 0 and not flags:
        confidence = "high"
    elif exec_edge is not None and exec_edge > 0 and len(flags) <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        **spread,
        "confidence": confidence,
        "flags": flags,
        "returnOnCapital": (net / capital) if capital > 0 else None,
    }


def build_ideas(spreads_by_ticker: dict[str, list[dict]]) -> list[dict]:
    """Assess every ticker's spreads, keep the strongest few per name for
    diversity, and rank the survivors across names.

    Ranking is by executable edge (the conservative measure) with a
    confidence tiebreak; mid-only edges (no executable fill) sort last.
    """
    assessed: list[dict] = []
    for ticker, spreads in spreads_by_ticker.items():
        ideas = [a for s in spreads if (a := assess_spread(s)) is not None]
        conf_rank = {"high": 0, "medium": 1, "low": 2}
        ideas.sort(
            key=lambda x: (
                conf_rank[x["confidence"]],
                -(x["execEdgeDollars"] if x["execEdgeDollars"] is not None else -1e12),
            )
        )
        for idea in ideas[:MAX_PER_TICKER]:
            idea["ticker"] = ticker
        assessed.extend(ideas[:MAX_PER_TICKER])

    conf_rank = {"high": 0, "medium": 1, "low": 2}
    assessed.sort(
        key=lambda x: (
            conf_rank[x["confidence"]],
            -(x["execEdgeDollars"] if x["execEdgeDollars"] is not None else -1e12),
            -x["netEdgeDollars"],
        )
    )
    return assessed[:MAX_IDEAS]
