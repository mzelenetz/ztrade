import { Link } from "react-router-dom"
import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { ThemeToggle } from "@/components/ThemeToggle"

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground [&_strong]:text-foreground">
        {children}
      </CardContent>
    </Card>
  )
}

export function MethodologyPage() {
  return (
    <div className="min-h-svh bg-background">
      <header className="border-b">
        <div className="mx-auto flex max-w-[900px] items-center justify-between gap-4 px-6 py-3">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" asChild aria-label="Back to dashboard">
              <Link to="/">
                <ArrowLeft className="size-4" />
              </Link>
            </Button>
            <h1 className="text-lg font-semibold">Methodology</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-[900px] space-y-6 px-6 py-8">
        <Section title="What this tool does">
          <p>
            ZTrade compares the market price of every listed option against a model-based fair value
            (FMV), flags how rich or cheap each contract looks, and then searches for two-legged,
            roughly delta-neutral spreads that buy the cheap leg and sell the rich one. Every idea is
            ranked by the dollar edge left over <strong>after</strong> accounting for the capital the
            position ties up and what that capital costs to carry.
          </p>
        </Section>

        <Section title="Pricing models">
          <p>
            Two pricing engines are available from the sidebar; both produce a fair market value
            (FMV) and a delta for every contract:
          </p>
          <ul className="list-disc space-y-2 pl-5">
            <li>
              <strong>mzpricer</strong> (default) — an in-house pricer that values each option on a
              500-step binomial tree with <strong>American-style exercise</strong>, matching how
              listed US equity options actually trade. Its put values run slightly above European
              model values because they include the early-exercise premium.
            </li>
            <li>
              <strong>QuantLib</strong> — the open-source quantitative finance library, using its
              analytic Black-Scholes-Merton <strong>European</strong> pricer. Useful as an
              independent cross-check: calls (with no dividend) should match mzpricer almost
              exactly, and puts should differ only by a modest early-exercise premium. A larger
              disagreement is a red flag on that contract's inputs.
            </li>
          </ul>
          <p>The inputs both engines use for each contract:</p>
          <ul className="list-disc space-y-2 pl-5">
            <li>
              <strong>Valuation date</strong> — the data's own quote date (shown in the header),
              never today's date. Time to expiry and every FMV are measured from the moment the
              prices were captured.
            </li>
            <li>
              <strong>Spot</strong> — the midpoint of the underlying's bid/ask captured at 15:45 ET
              on the close date, cross-checked against the ticker's own option prices via put-call
              parity. If the quoted underlying disagrees with parity by more than 2% (a stale or
              corrupt mark), the parity-implied spot is used instead.
            </li>
            <li>
              <strong>Volatility</strong> — by default a <strong>fitted smile surface</strong>{" "}
              built from implied vols we compute ourselves by inverting the mid quotes under our
              own market-implied carry (the data vendor's IV column embeds its own rate/dividend
              assumptions, which splits same-strike call and put IVs into two branches no smile
              can fit). Out-of-the-money quotes only, in standardized moneyness so every expiry's
              tradeable region is comparable. The surface is fit{" "}
              <strong>jointly across all expiries</strong> with smooth time-dependence, then each
              expiry's smile is refined on its own quotes with shrinkage toward that global shape:
              densely quoted near-term expiries keep their own skew exactly, while sparsely quoted
              long-dated expiries inherit the pooled shape instead of chasing a handful of noisy
              quotes — so near-term skew can't distort (or be projected onto) the long end.
              "Edge" then means a contract is rich or cheap{" "}
              <strong>relative to its neighbors on the surface</strong> — a relative-value signal,
              not a claim about absolute mispricing. Outside an expiry's quoted strike range the
              model adopts the contract's own market IV (no opinion → no edge).
              <br />
              The alternative mode, <strong>"Surface anchored to 30d realized,"</strong> expresses
              a vol-reversion view: the same fitted surface is ratio-shifted so its ~30-day
              at-the-money vol equals the stock's trailing 30-day realized vol, keeping skew and
              term-structure shape intact. That prices the level view ("implied should revert to
              what the stock actually moves") without taking an accidental view on skew — a flat
              vol at all strikes would masquerade crash premium as edge. Two caveats: implied vol
              systematically exceeds realized (the variance risk premium), so this mode leans
              short-vol by construction; and the parallel shift overstates the effect on
              long-dated expiries, whose vols empirically move less than one-for-one with
              short-dated vol.
            </li>
            <li>
              <strong>Interest rates</strong> — interpolated from the editable rate curve in Model
              Inputs (1M–3Y tenor points), so long-dated and short-dated contracts each discount at
              an appropriate rate.
            </li>
            <li>
              <strong>Dividends</strong> — three treatments, in priority order (managed in the
              Dividends tab):
              <ul className="mt-2 list-[circle] space-y-2 pl-5">
                <li>
                  <strong>Discrete schedule (preferred).</strong> Enter each dividend as an amount
                  and ex-date. Pricing uses the escrowed model: each option's underlying is reduced
                  by the PV of dividends going ex <em>before that option's expiry</em> — an option
                  expiring before the ex-date gets no adjustment at all. (Implemented as an exact
                  per-expiry equivalent yield, so both pricing engines consume it without
                  approximation error at the European level.) One caveat: the ex-date's position{" "}
                  <em>inside</em> the binomial tree isn't modeled, which slightly understates the
                  early-exercise value of calls just before large dividends.
                </li>
                <li>
                  <strong>Market-implied (fallback).</strong> For tickers without a schedule,
                  put-call parity gives the option-implied forward per expiry
                  (F = (C−P)·e^(rT) + K, median near the money) and q = r − ln(F/S)/T. Captures
                  dividends and borrow costs together; negative q = hard-to-borrow premium. The
                  discount <em>rate</em> stays on your curve — American puts' early-exercise
                  premium biases a parity-fitted rate low, but the bias cancels in the forward.
                </li>
                <li>
                  <strong>Manual yield.</strong> A flat per-ticker yield, used when you switch off
                  implied mode and have no schedule.
                </li>
              </ul>
            </li>
          </ul>
          <p>
            Data hygiene: quotes violating basic no-arbitrage bounds (an option priced below
            intrinsic value or above its underlying/strike cap) are dropped — these are stale
            prints on illiquid strikes that would otherwise masquerade as enormous edge.
          </p>
        </Section>

        <Section title="Overvaluation and probabilities">
          <p>
            <strong>%Overvalued</strong> is simply market price ÷ FMV − 1, using whichever market
            price you select in the sidebar (Last, Mid, Bid, or Ask). +20% means the market is
            paying 20% over model value; −20% means the contract trades at a 20% discount to model.
          </p>
          <p>
            <strong>Prob ITM / Prob OTM</strong> are risk-neutral probabilities that the option
            finishes in or out of the money at expiry, computed from the same Black-Scholes inputs
            (they are the standard N(d₂) quantities). They are best used comparatively — as a
            moneyness gauge across strikes — rather than as literal real-world odds, since
            risk-neutral probabilities embed the flat-vol and zero-dividend assumptions above.
          </p>
        </Section>

        <Section title="Spread construction">
          <p>The Spreads tab searches for two-legged ideas as follows:</p>
          <ul className="list-disc space-y-2 pl-5">
            <li>
              All contracts are filtered by your call-delta band and price limits, then ranked by
              %Overvalued. The cheapest contracts become <strong>buy candidates</strong>; the richest
              become <strong>sell candidates</strong> (up to "max legs per side" of each).
            </li>
            <li>
              Every buy/sell pairing is sized for <strong>delta neutrality</strong>: one side is
              anchored at 10 contracts and the other side's quantity is set so the position's net
              delta is as close to zero as whole contracts allow. Pairings whose contract ratio,
              net delta, or straddle ratio exceed your limits are discarded.
            </li>
            <li>
              <strong>Edge (%)</strong> is the sell leg's %Overvalued minus the buy leg's
              %Overvalued — the total mispricing you're capturing per unit of premium. Only
              positive-edge pairs are kept.
            </li>
            <li>
              Ideas are ranked by <strong>Net Edge $</strong> (below), not by the raw percentage.
            </li>
          </ul>
          <p>
            Note the search is deliberately unconstrained about structure: a pairing can be a
            vertical, a calendar, a diagonal, or a call/put pair. It's a mispricing screen, not a
            strategy template.
          </p>
        </Section>

        <Section title="Cost of capital and margin">
          <p>
            Capturing edge isn't free: the short leg ties up margin and the long leg costs premium.
            The tool estimates that capital and charges it at your margin rate for the expected life
            of the position, so ideas are ranked by what's left after financing.
          </p>
          <ul className="list-disc space-y-2 pl-5">
            <li>
              <strong>Margin $</strong> — computed per the margin type you select in the sidebar:
              <ul className="mt-2 list-[circle] space-y-2 pl-5">
                <li>
                  <strong>Reg T</strong> — the short leg is margined as a naked short (Fidelity's
                  standard formula): the greater of 20% of the underlying minus the
                  out-of-the-money amount, or 10% of the underlying (calls) / 10% of the strike
                  (puts), plus the premium received — per contract, times quantity. The long leg is
                  not offset. This is the conservative ceiling.
                </li>
                <li>
                  <strong>Portfolio margin</strong> — risk-based, the way PM accounts are actually
                  margined (TIMS-style): both legs are repriced together at eleven underlying
                  scenarios from −15% to +15%, and the requirement is the worst-case loss of the
                  combined position, with a $37.50-per-short-contract minimum. Because the long leg
                  hedges the short under stress, PM requirements for spreads are usually far lower
                  than Reg T — meaning less capital, less carry, and better net edge.
                </li>
              </ul>
            </li>
            <li>
              <strong>Net Debit $</strong> — long premium paid minus short premium received. Negative
              means the position is put on for a net credit.
            </li>
            <li>
              <strong>Capital employed</strong> — under Reg T: the margin requirement excluding the
              premium-received portion (that part is funded by the sale itself), plus any net debit
              paid. Under portfolio margin: the full requirement plus any net debit (the PM
              requirement is a pure worst-case loss with no premium component). A net credit does
              not reduce the margin held in either mode.
            </li>
            <li>
              <strong>Carry $</strong> — capital employed × your margin rate × years until the
              nearer of the two legs' expiries.
            </li>
            <li>
              <strong>Gross Edge $</strong> — the dollar mispricing across both legs: (short price −
              short FMV) plus (long FMV − long price), each times 100 × quantity.
            </li>
            <li>
              <strong>Net Edge $</strong> — Gross Edge $ minus Carry $. This is the ranking metric.
            </li>
          </ul>
        </Section>

        <Section title="What to enter from Fidelity — and what can differ">
          <p>
            The only Fidelity-specific input is the <strong>margin rate</strong> in the sidebar.
            Fidelity's rate is tiered by your total margin debit balance, so enter the rate for your
            tier (Fidelity.com → Accounts → Margin; the default here is the base tier). Position
            size and everything else is computed from market data.
          </p>
          <p>Reasons your actual Fidelity requirement may differ from Margin $ shown here:</p>
          <ul className="list-disc space-y-2 pl-5">
            <li>
              <strong>Spread recognition (Reg T mode).</strong> In Reg T mode we conservatively
              margin the short leg as naked. If Fidelity's system pairs your legs as a recognized
              spread (e.g. a vertical, where the requirement is just the strike width), the real
              requirement can be far lower — the carry shown is then an overestimate and the true
              net edge is better. The portfolio-margin mode captures this offset naturally.
            </li>
            <li>
              <strong>House minimums and option levels.</strong> Fidelity applies per-contract house
              minimums and requires the appropriate option approval level (and minimum equity) for
              uncovered writing.
            </li>
            <li>
              <strong>Portfolio margin.</strong> If your account has PM, select "Portfolio margin"
              in the sidebar. Note our PM number stresses only this two-legged position in
              isolation; Fidelity stresses your whole portfolio, so concentrated accounts can see
              add-ons and offsetting positions elsewhere can lower it further. PM also requires
              account approval and minimum equity (typically $100k+ at Fidelity).
            </li>
            <li>
              <strong>What interest is actually charged on.</strong> Fidelity charges margin
              interest only on borrowed cash (a debit balance), not on requirement held against
              positions. Carry $ here treats all capital employed as financed at your rate — i.e. a
              cost-of-capital hurdle, which is the right way to rank trades, but it is not literally
              your interest bill unless you're fully financing the position.
            </li>
          </ul>
        </Section>

        <Section title="Dividends tab">
          <p>
            The Dividends tab holds the per-ticker schedule editor (add rows, project quarterly,
            or seed drafts from the market) and a chart of <strong>cumulative dividend PV by
            expiry</strong>: the market-implied points (from parity forwards) against your
            schedule's step-line. A jump in the implied points marks where the market prices a
            dividend — "Seed from market" converts those jumps into draft rows whose ex-dates you
            should correct against the declared schedule. When the two series track each other,
            your schedule agrees with the market; a gap flags either a stale schedule or a real
            borrow-cost signal.
          </p>
        </Section>

        <Section title="Vol Surface tab">
          <p>
            The Vol Surface tab shows, per expiry, every contract's market implied vol (calls and
            puts as points) against the fitted smile (line), with the spot marked. Points sitting
            off the line are the raw material of the Spreads screen — a point above the line is
            trading rich to its neighbors; below the line, cheap. A flat "median-IV fallback" badge
            means that expiry had too few clean quotes to fit a smile and the model used a single
            median vol instead — treat its edges with extra skepticism.
          </p>
        </Section>

        <Separator />
        <p className="pb-8 text-xs text-muted-foreground">
          This tool is a screening aid, not investment advice. Edge here is relative value against
          a fitted surface built from the same day's quotes — it can reflect liquidity, upcoming
          events, or borrow costs rather than free money. Verify pricing and margin with your
          broker before trading.
        </p>
      </main>
    </div>
  )
}
