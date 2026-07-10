import { useMemo, useState } from "react"
import type { VolSurfaceExpiry } from "@/types"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"

// Categorical palette validated for CVD + contrast (dataviz skill), per theme.
const SERIES = {
  call: { light: "#2a78d6", dark: "#3987e5", label: "Calls" },
  put: { light: "#eb6834", dark: "#d95926", label: "Puts" },
}

const W = 860
const H = 380
const M = { top: 16, right: 20, bottom: 40, left: 56 }

interface Hover {
  x: number
  y: number
  strike: number
  iv: number
  kind: string
}

function SmileChart({ expiry: raw }: { expiry: VolSurfaceExpiry }) {
  const [hover, setHover] = useState<Hover | null>(null)

  // Window the view around the fitted region (where the model has a view);
  // deep wings would squash the chart. Fall back to 0.5–2× spot if no fit.
  const expiry = useMemo(() => {
    let lo = raw.spot * 0.5
    let hi = raw.spot * 2.0
    if (raw.fitted.length >= 2) {
      const fitStrikes = raw.fitted.map((f) => f.strike)
      lo = Math.min(...fitStrikes) - raw.spot * 0.05
      hi = Math.max(...fitStrikes) + raw.spot * 0.05
    }
    return {
      ...raw,
      points: raw.points.filter((p) => p.strike >= lo && p.strike <= hi),
      fitted: raw.fitted.filter((f) => f.strike >= lo && f.strike <= hi),
    }
  }, [raw])

  const { xScale, yScale, ticksX, ticksY } = useMemo(() => {
    const strikes = [
      ...expiry.points.map((p) => p.strike),
      ...expiry.fitted.map((f) => f.strike),
    ]
    const ivs = [
      ...expiry.points.map((p) => p.marketIv),
      ...expiry.fitted.map((f) => f.iv),
    ]
    if (!strikes.length) {
      const id = (v: number) => v
      return { xScale: id, yScale: id, ticksX: [], ticksY: [] }
    }
    const xMin = Math.min(...strikes)
    const xMax = Math.max(...strikes)
    const yMin = Math.max(0, Math.min(...ivs) - 0.05)
    const yMax = Math.max(...ivs) + 0.05

    const xScale = (v: number) =>
      M.left + ((v - xMin) / (xMax - xMin || 1)) * (W - M.left - M.right)
    const yScale = (v: number) =>
      H - M.bottom - ((v - yMin) / (yMax - yMin || 1)) * (H - M.top - M.bottom)

    const tickCount = 8
    const ticksX = Array.from({ length: tickCount + 1 }, (_, i) => xMin + ((xMax - xMin) * i) / tickCount)
    const ticksY = Array.from({ length: 6 }, (_, i) => yMin + ((yMax - yMin) * i) / 5)

    return { xScale, yScale, ticksX, ticksY }
  }, [expiry])

  const fittedPath = useMemo(() => {
    const pts = [...expiry.fitted].sort((a, b) => a.strike - b.strike)
    return pts
      .map((f, i) => `${i === 0 ? "M" : "L"}${xScale(f.strike).toFixed(1)},${yScale(f.iv).toFixed(1)}`)
      .join(" ")
  }, [expiry, xScale, yScale])

  const spotX = xScale(expiry.spot)

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Vol smile for ${expiry.expiry}`}>
        {/* grid + axes (recessive) */}
        {ticksY.map((t) => (
          <g key={`y${t}`}>
            <line
              x1={M.left} x2={W - M.right} y1={yScale(t)} y2={yScale(t)}
              className="stroke-border" strokeWidth="1"
            />
            <text
              x={M.left - 8} y={yScale(t) + 4} textAnchor="end"
              className="fill-muted-foreground text-[11px]"
            >
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {ticksX.map((t) => (
          <text
            key={`x${t}`} x={xScale(t)} y={H - M.bottom + 18} textAnchor="middle"
            className="fill-muted-foreground text-[11px]"
          >
            {t.toFixed(0)}
          </text>
        ))}
        <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="fill-muted-foreground text-[11px]">
          Strike
        </text>

        {/* spot marker */}
        {spotX >= M.left && spotX <= W - M.right && (
          <>
            <line x1={spotX} x2={spotX} y1={M.top} y2={H - M.bottom} className="stroke-muted-foreground" strokeWidth="1" strokeDasharray="4 4" />
            <text x={spotX + 4} y={M.top + 10} className="fill-muted-foreground text-[11px]">
              spot {expiry.spot.toFixed(1)}
            </text>
          </>
        )}

        {/* fitted line */}
        <path d={fittedPath} fill="none" className="stroke-foreground" strokeWidth="2" />

        {/* market IV points */}
        {expiry.points.map((p, i) => {
          const c = p.type === "C" ? SERIES.call : SERIES.put
          return (
            <circle
              key={i}
              cx={xScale(p.strike)}
              cy={yScale(p.marketIv)}
              r={hover?.strike === p.strike && hover?.kind === c.label ? 6 : 4}
              fill="currentColor"
              className={p.type === "C" ? "text-[#2a78d6] dark:text-[#3987e5]" : "text-[#eb6834] dark:text-[#d95926]"}
              stroke="var(--background)"
              strokeWidth="1.5"
              onMouseEnter={() =>
                setHover({ x: xScale(p.strike), y: yScale(p.marketIv), strike: p.strike, iv: p.marketIv, kind: c.label })
              }
              onMouseLeave={() => setHover(null)}
            />
          )
        })}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded-md border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md"
          style={{
            left: `${(hover.x / W) * 100}%`,
            top: `${(hover.y / H) * 100}%`,
            transform: "translate(-50%, -130%)",
          }}
        >
          {hover.kind} · K {hover.strike} · IV {(hover.iv * 100).toFixed(1)}%
        </div>
      )}
    </div>
  )
}

export function VolSurfaceView({ expiries, loading }: { expiries: VolSurfaceExpiry[]; loading: boolean }) {
  const [selected, setSelected] = useState<string | null>(null)

  if (loading) return <p className="text-sm text-muted-foreground">Loading vol surface…</p>
  if (!expiries.length) return <p className="text-sm text-muted-foreground">No surface data.</p>

  const active = expiries.find((e) => e.expiry === selected) ?? expiries[0]

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Select value={active.expiry} onValueChange={setSelected}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {expiries.map((e) => (
              <SelectItem key={e.expiry} value={e.expiry}>
                {e.expiry}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="inline-block size-2.5 rounded-full bg-[#2a78d6] dark:bg-[#3987e5]" /> Calls (market IV)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block size-2.5 rounded-full bg-[#eb6834] dark:bg-[#d95926]" /> Puts (market IV)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-5 bg-foreground" /> Fitted smile
          </span>
        </div>

        <Badge variant="outline" className="font-normal">
          r {(active.rate * 100).toFixed(2)}% · q {(active.divYield * 100).toFixed(2)}%
          {active.divSource === "schedule"
            ? " (schedule)"
            : active.divSource === "implied"
              ? " (market-implied)"
              : " (manual)"}
        </Badge>

        {active.fallback && (
          <Badge variant="secondary">median-IV fallback (too few quotes to fit)</Badge>
        )}
      </div>

      <SmileChart expiry={active} />
    </div>
  )
}
