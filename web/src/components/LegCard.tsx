import type { LegDetail } from "@/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

function Field({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p>{value ?? "–"}</p>
    </div>
  )
}

const num = (v: number | null | undefined, dp = 2) => (v != null ? v.toFixed(dp) : null)
const pct = (v: number | null | undefined, dp = 1) => (v != null ? `${(v * 100).toFixed(dp)}%` : null)
const int = (v: number | null | undefined) => (v != null ? Math.round(v).toLocaleString() : null)

export function LegCard({ title, leg }: { title: string; leg: LegDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {title}: {leg.ticker} {leg.expiry} {leg.strike}
          {leg.type === "C" ? "c" : "p"}
        </CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        <Field label="Last" value={num(leg.last)} />
        <Field
          label="Bid × size"
          value={leg.bid != null ? `${leg.bid.toFixed(2)}${leg.bidSize != null ? ` × ${int(leg.bidSize)}` : ""}` : null}
        />
        <Field
          label="Ask × size"
          value={leg.ask != null ? `${leg.ask.toFixed(2)}${leg.askSize != null ? ` × ${int(leg.askSize)}` : ""}` : null}
        />
        <Field label="Mid" value={num(leg.mid)} />
        <Field label="Model FMV" value={num(leg.fmv)} />
        <Field label="%Overvalued" value={pct(leg.overvalued)} />
        <Field label="Volume" value={int(leg.volume)} />
        <Field label="Open interest" value={int(leg.openInterest)} />
        <Field
          label="Market IV"
          value={pct(leg.marketIv)}
        />
        <Field
          label="Model vol"
          value={
            leg.modelVol != null
              ? `${(leg.modelVol * 100).toFixed(1)}%${leg.volFromSurface === false ? " (wing)" : ""}`
              : null
          }
        />
        <Field label="Prob ITM" value={pct(leg.probItm)} />
        <Field label="Prob OTM" value={pct(leg.probOtm)} />
        <Field label="Delta" value={num(leg.delta, 3)} />
        <Field label="Gamma" value={num(leg.gamma, 4)} />
        <Field label="Vega (1 pt)" value={num(leg.vega)} />
        <Field label="Theta (day)" value={num(leg.theta, 3)} />
      </CardContent>
    </Card>
  )
}
