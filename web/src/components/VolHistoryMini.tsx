import { useEffect, useMemo, useState } from "react"
import type { VolHistoryPoint } from "@/types"

const MINI_W = 240
const MINI_H = 48
const BIG_W = 700
const BIG_H = 320
const BIG_M = { top: 16, right: 20, bottom: 32, left: 48 }

function buildPath(points: VolHistoryPoint[], xScale: (i: number) => number, yScale: (v: number) => number) {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(p.vol30d).toFixed(1)}`)
    .join(" ")
}

function MiniSparkline({ points }: { points: VolHistoryPoint[] }) {
  const { path, xScale, yScale, maxY } = useMemo(() => {
    const maxY = Math.max(...points.map((p) => p.vol30d)) * 1.1
    const xScale = (i: number) =>
      points.length > 1 ? (i / (points.length - 1)) * MINI_W : MINI_W / 2
    const yScale = (v: number) => MINI_H - (v / (maxY || 1)) * MINI_H
    return { path: buildPath(points, xScale, yScale), xScale, yScale, maxY }
  }, [points])

  const last = points[points.length - 1]

  return (
    <svg viewBox={`0 0 ${MINI_W} ${MINI_H}`} className="w-full" preserveAspectRatio="none" role="img" aria-label="30d realized vol, last 30 readings">
      <line x1={0} x2={MINI_W} y1={MINI_H - 0.5} y2={MINI_H - 0.5} className="stroke-border" strokeWidth="1" />
      <path d={path} fill="none" className="stroke-muted-foreground transition-colors group-hover:stroke-foreground" strokeWidth="1.5" />
      {last && (
        <circle
          cx={xScale(points.length - 1)}
          cy={yScale(last.vol30d)}
          r="2.5"
          className="fill-muted-foreground transition-colors group-hover:fill-foreground"
        />
      )}
      <title>{`0 – ${(maxY * 100).toFixed(0)}%`}</title>
    </svg>
  )
}

function BigChart({ points }: { points: VolHistoryPoint[] }) {
  const [hover, setHover] = useState<{ x: number; y: number; point: VolHistoryPoint } | null>(null)

  const { xScale, yScale, ticksY } = useMemo(() => {
    const maxY = Math.max(...points.map((p) => p.vol30d)) * 1.1
    const xScale = (i: number) =>
      BIG_M.left + (points.length > 1 ? (i / (points.length - 1)) * (BIG_W - BIG_M.left - BIG_M.right) : 0)
    const yScale = (v: number) =>
      BIG_H - BIG_M.bottom - (v / (maxY || 1)) * (BIG_H - BIG_M.top - BIG_M.bottom)
    const ticksY = Array.from({ length: 5 }, (_, i) => (maxY * i) / 4)
    return { xScale, yScale, maxY, ticksY }
  }, [points])

  const path = useMemo(() => buildPath(points, xScale, yScale), [points, xScale, yScale])

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${BIG_W} ${BIG_H}`} className="w-full" role="img" aria-label="30d realized vol history">
        {ticksY.map((t) => (
          <g key={t}>
            <line x1={BIG_M.left} x2={BIG_W - BIG_M.right} y1={yScale(t)} y2={yScale(t)} className="stroke-border" strokeWidth="1" />
            <text x={BIG_M.left - 8} y={yScale(t) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        <path d={path} fill="none" className="stroke-foreground" strokeWidth="2" />
        {points.map((p, i) => (
          <circle
            key={p.date}
            cx={xScale(i)}
            cy={yScale(p.vol30d)}
            r={hover?.point.date === p.date ? 5 : 3}
            fill="currentColor"
            className="text-foreground"
            stroke="var(--background)"
            strokeWidth="1"
            onMouseEnter={() => setHover({ x: xScale(i), y: yScale(p.vol30d), point: p })}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {points.length > 1 && points.map((p, i) => {
          if (i % Math.ceil(points.length / 8) !== 0 && i !== points.length - 1) return null
          return (
            <text key={`x${p.date}`} x={xScale(i)} y={BIG_H - BIG_M.bottom + 16} textAnchor="middle" className="fill-muted-foreground text-[10px]">
              {p.date.slice(5)}
            </text>
          )
        })}
      </svg>
      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded-md border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md"
          style={{ left: `${(hover.x / BIG_W) * 100}%`, top: `${(hover.y / BIG_H) * 100}%`, transform: "translate(-50%, -130%)" }}
        >
          {hover.point.date} · {(hover.point.vol30d * 100).toFixed(1)}%
        </div>
      )}
    </div>
  )
}

function VolHistoryModal({ points, onClose }: { points: VolHistoryPoint[]; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-xl border bg-card p-6 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <p className="text-sm font-medium">30d realized volatility — close-based</p>
            <p className="text-xs text-muted-foreground">
              Trailing 30-trading-day realized vol as stamped on each day's ingest. IVs from the fitted
              surface will be layered in once we have IV history.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md px-2 py-1 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <BigChart points={points} />

        <div className="mt-4 max-h-56 overflow-y-auto rounded-md border">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-muted text-xs text-muted-foreground">
              <tr>
                <th className="px-3 py-1.5 text-left font-medium">Date</th>
                <th className="px-3 py-1.5 text-right font-medium">30d RV (close)</th>
              </tr>
            </thead>
            <tbody>
              {[...points].reverse().map((p) => (
                <tr key={p.date} className="border-t">
                  <td className="px-3 py-1.5">{p.date}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{(p.vol30d * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export function VolHistoryMini({ points, loading }: { points: VolHistoryPoint[]; loading: boolean }) {
  const [open, setOpen] = useState(false)

  if (loading || points.length < 2) return null

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="group -m-1 mt-1 hidden w-full cursor-pointer rounded-md p-1 transition-colors hover:bg-accent/40 @[160px]:block"
        aria-label="Show 30d realized vol history"
      >
        <MiniSparkline points={points} />
      </button>
      {open && <VolHistoryModal points={points} onClose={() => setOpen(false)} />}
    </>
  )
}
