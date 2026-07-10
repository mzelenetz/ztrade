import { useMemo } from "react"
import type { Filters, RateCurvePoint } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

const TENOR_LABELS: Record<number, string> = {
  30: "1M",
  91: "3M",
  182: "6M",
  365: "1Y",
  730: "2Y",
  1095: "3Y",
}

function CurveSparkline({ curve }: { curve: RateCurvePoint[] }) {
  const path = useMemo(() => {
    if (curve.length < 2) return ""
    const xs = curve.map(([d]) => d)
    const ys = curve.map(([, r]) => r)
    const xMin = Math.min(...xs), xMax = Math.max(...xs)
    const yMin = Math.min(...ys) - 0.001, yMax = Math.max(...ys) + 0.001
    const W = 220, H = 48, pad = 4
    return curve
      .map(([d, r], i) => {
        const x = pad + ((d - xMin) / (xMax - xMin || 1)) * (W - 2 * pad)
        const y = H - pad - ((r - yMin) / (yMax - yMin || 1)) * (H - 2 * pad)
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`
      })
      .join(" ")
  }, [curve])

  return (
    <svg viewBox="0 0 220 48" className="h-12 w-full" role="img" aria-label="Rate curve shape">
      <path d={path} fill="none" className="stroke-foreground" strokeWidth="2" />
    </svg>
  )
}

interface Props {
  filters: Filters
  updateFilter: <K extends keyof Filters>(key: K, value: Filters[K]) => void
}

export function ModelInputsCard({ filters, updateFilter }: Props) {
  const sortedCurve = useMemo(
    () => [...filters.rateCurve].sort((a, b) => a[0] - b[0]),
    [filters.rateCurve],
  )

  const setCurvePoint = (days: number, ratePct: number) => {
    updateFilter(
      "rateCurve",
      filters.rateCurve.map(([d, r]) => (d === days ? [d, ratePct / 100] : [d, r])) as RateCurvePoint[],
    )
  }

  return (
    <Card>
      <CardContent className="space-y-4 pt-6">
        <p className="text-sm font-medium">Model inputs</p>

        <div className="space-y-2">
          <Label>Volatility model</Label>
          <Select
            value={filters.volMode}
            onValueChange={(v) => updateFilter("volMode", v as Filters["volMode"])}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="surface">Fitted surface (market IVs)</SelectItem>
              <SelectItem value="flat">Flat</SelectItem>
              <SelectItem value="historical">Historical 30d</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <p className="text-xs text-muted-foreground">
          Dividends are managed in the Dividends tab (schedule → market-implied → manual).
        </p>

        <div className="space-y-2">
          <Label>Rate curve (% / yr)</Label>
          <div className="grid grid-cols-3 gap-1.5">
            {sortedCurve.map(([days, rate]) => (
              <div key={days} className="space-y-0.5">
                <span className="text-xs text-muted-foreground">{TENOR_LABELS[days] ?? `${days}d`}</span>
                <Input
                  type="number"
                  step={0.05}
                  className="h-8 px-2 text-xs"
                  value={(rate * 100).toFixed(2)}
                  onChange={(e) => setCurvePoint(days, Number(e.target.value))}
                />
              </div>
            ))}
          </div>
          <CurveSparkline curve={sortedCurve} />
        </div>
      </CardContent>
    </Card>
  )
}
