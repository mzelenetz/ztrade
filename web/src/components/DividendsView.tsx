import { useMemo, useState } from "react"
import { Plus, Trash2 } from "lucide-react"
import type { DividendsResponse, DividendRow, Filters } from "@/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

const W = 860
const H = 300
const M = { top: 16, right: 20, bottom: 40, left: 56 }

// Same validated categorical palette as the vol surface chart.
const IMPLIED_COLOR = "text-[#2a78d6] dark:text-[#3987e5]"

function StaircaseChart({ data }: { data: DividendsResponse }) {
  const expiries = data.expiries.filter((e) => e.impliedDivPV !== null || e.scheduledDivPV > 0)

  const { xScale, yScale, ticksY, xTicks } = useMemo(() => {
    const dates = [data.valuationDate, ...expiries.map((e) => e.expiry)].map((d) =>
      new Date(d).getTime(),
    )
    const pvs = [
      0,
      ...expiries.map((e) => e.impliedDivPV ?? 0),
      ...expiries.map((e) => e.scheduledDivPV),
    ]
    const xMin = Math.min(...dates)
    const xMax = Math.max(...dates)
    const yMax = Math.max(...pvs, 0.1) * 1.15

    const xScale = (iso: string) =>
      M.left + ((new Date(iso).getTime() - xMin) / (xMax - xMin || 1)) * (W - M.left - M.right)
    const yScale = (v: number) => H - M.bottom - (v / yMax) * (H - M.top - M.bottom)

    const ticksY = Array.from({ length: 5 }, (_, i) => (yMax * i) / 4)
    const step = Math.max(1, Math.floor(expiries.length / 8))
    const xTicks = expiries.filter((_, i) => i % step === 0).map((e) => e.expiry)
    return { xScale, yScale, ticksY, xTicks }
  }, [data, expiries])

  const schedulePath = useMemo(() => {
    // Step function: PV holds until the next expiry point.
    let d = `M${xScale(data.valuationDate).toFixed(1)},${yScale(0).toFixed(1)}`
    let prev = 0
    for (const e of expiries) {
      const x = xScale(e.expiry)
      d += ` L${x.toFixed(1)},${yScale(prev).toFixed(1)} L${x.toFixed(1)},${yScale(e.scheduledDivPV).toFixed(1)}`
      prev = e.scheduledDivPV
    }
    return d
  }, [expiries, data.valuationDate, xScale, yScale])

  if (!expiries.length) {
    return <p className="text-sm text-muted-foreground">No dividend signal for this ticker.</p>
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Cumulative dividend PV by expiry">
      {ticksY.map((t) => (
        <g key={t}>
          <line x1={M.left} x2={W - M.right} y1={yScale(t)} y2={yScale(t)} className="stroke-border" strokeWidth="1" />
          <text x={M.left - 8} y={yScale(t) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
            ${t.toFixed(2)}
          </text>
        </g>
      ))}
      {xTicks.map((d) => (
        <text key={d} x={xScale(d)} y={H - M.bottom + 18} textAnchor="middle" className="fill-muted-foreground text-[11px]">
          {d.slice(5)}
        </text>
      ))}
      <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="fill-muted-foreground text-[11px]">
        Expiry
      </text>

      {/* schedule step-line */}
      <path d={schedulePath} fill="none" className="stroke-foreground" strokeWidth="2" />

      {/* implied PV points */}
      {expiries.map(
        (e) =>
          e.impliedDivPV !== null && (
            <circle
              key={e.expiry}
              cx={xScale(e.expiry)}
              cy={yScale(e.impliedDivPV)}
              r={4}
              fill="currentColor"
              className={IMPLIED_COLOR}
              stroke="var(--background)"
              strokeWidth="1.5"
            >
              <title>{`${e.expiry}: implied PV $${e.impliedDivPV.toFixed(2)}`}</title>
            </circle>
          ),
      )}
    </svg>
  )
}

interface Props {
  ticker: string
  filters: Filters
  data: DividendsResponse | undefined
  loading: boolean
  updateFilter: <K extends keyof Filters>(key: K, value: Filters[K]) => void
}

export function DividendsView({ ticker, filters, data, loading, updateFilter }: Props) {
  const rows: DividendRow[] = filters.dividendSchedule[ticker] ?? []
  const [seeded, setSeeded] = useState(false)

  const setRows = (next: DividendRow[]) =>
    updateFilter("dividendSchedule", { ...filters.dividendSchedule, [ticker]: next })

  const updateRow = (i: number, row: DividendRow) => setRows(rows.map((r, j) => (j === i ? row : r)))
  const removeRow = (i: number) => setRows(rows.filter((_, j) => j !== i))

  const addRow = () => {
    const last = rows[rows.length - 1]
    const nextDate = last
      ? new Date(new Date(last[0]).getTime() + 91 * 86400_000).toISOString().slice(0, 10)
      : (data?.valuationDate ?? new Date().toISOString().slice(0, 10))
    setRows([...rows, [nextDate, last?.[1] ?? 0.25]])
  }

  const projectQuarterly = () => {
    const last = rows[rows.length - 1]
    if (!last) return
    const extra: DividendRow[] = Array.from({ length: 4 }, (_, i) => [
      new Date(new Date(last[0]).getTime() + 91 * 86400_000 * (i + 1)).toISOString().slice(0, 10),
      last[1],
    ])
    setRows([...rows, ...extra])
  }

  const seedFromMarket = () => {
    if (!data?.seedSuggestions.length) return
    const drafts: DividendRow[] = data.seedSuggestions.map((s) => {
      const mid = new Date(
        (new Date(s.windowStart).getTime() + new Date(s.windowEnd).getTime()) / 2,
      )
      return [mid.toISOString().slice(0, 10), Math.round(s.amount * 100) / 100]
    })
    setRows(drafts)
    setSeeded(true)
  }

  if (loading) return <p className="text-sm text-muted-foreground">Loading dividends…</p>

  return (
    <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{ticker} dividend schedule</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {rows.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No scheduled dividends — pricing falls back to{" "}
                {filters.carryMode === "implied" ? "market-implied forwards" : "the manual yield"}.
              </p>
            )}
            {rows.map(([exDate, amount], i) => (
              <div key={i} className="flex items-center gap-2">
                <Input
                  type="date"
                  className="h-8"
                  value={exDate}
                  onChange={(e) => updateRow(i, [e.target.value, amount])}
                />
                <Input
                  type="number"
                  step={0.01}
                  min={0}
                  className="h-8 w-24"
                  value={amount}
                  onChange={(e) => updateRow(i, [exDate, Number(e.target.value)])}
                />
                <Button variant="ghost" size="icon-sm" onClick={() => removeRow(i)} aria-label="Remove row">
                  <Trash2 className="size-4" />
                </Button>
              </div>
            ))}
            <div className="flex flex-wrap gap-2 pt-1">
              <Button variant="outline" size="sm" onClick={addRow}>
                <Plus className="size-4" /> Add
              </Button>
              <Button variant="outline" size="sm" onClick={projectQuarterly} disabled={!rows.length}>
                Project quarterly ×4
              </Button>
              <Button variant="outline" size="sm" onClick={seedFromMarket} disabled={!data?.seedSuggestions.length}>
                Seed from market
              </Button>
            </div>
            {seeded && (
              <p className="text-xs text-muted-foreground">
                Seeded from implied forward jumps — ex-dates are window midpoints; correct them
                against the declared schedule.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Fallback (no schedule)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              <Label>Dividends &amp; borrow</Label>
              <Select
                value={filters.carryMode}
                onValueChange={(v) => updateFilter("carryMode", v as Filters["carryMode"])}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="implied">Market-implied (from forwards)</SelectItem>
                  <SelectItem value="manual">Manual yield</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Manual yield for {ticker} (% / yr)</Label>
              <Input
                type="number"
                step={0.05}
                min={0}
                className="h-8"
                value={((filters.dividends[ticker] ?? 0) * 100).toFixed(2)}
                onChange={(e) =>
                  updateFilter("dividends", {
                    ...filters.dividends,
                    [ticker]: Number(e.target.value) / 100,
                  })
                }
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle className="text-base">Cumulative dividend PV by expiry</CardTitle>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-full bg-[#2a78d6] dark:bg-[#3987e5]" />
              Market-implied
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-5 bg-foreground" /> Schedule
            </span>
            {rows.length > 0 && <Badge variant="outline">pricing: schedule</Badge>}
          </div>
        </CardHeader>
        <CardContent>
          {data ? <StaircaseChart data={data} /> : <p className="text-sm text-muted-foreground">No data.</p>}
        </CardContent>
      </Card>
    </div>
  )
}
