import { useMemo, useState } from "react"
import { AgGridReact } from "ag-grid-react"
import type { ColDef, ICellRendererParams, ValueFormatterParams } from "ag-grid-community"
import type { Idea } from "@/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LegCard } from "@/components/LegCard"
import { legLabel, STRUCTURE_LABELS } from "@/components/SpreadsView"

function dollars(value: number | null | undefined) {
  if (value === null || value === undefined) return "–"
  const rounded = Math.round(value)
  return rounded < 0 ? `-$${Math.abs(rounded).toLocaleString()}` : `$${rounded.toLocaleString()}`
}

function fmtDollars(params: ValueFormatterParams) {
  return typeof params.value === "number" ? dollars(params.value) : "–"
}

function ConfidenceBadge({ value }: { value: Idea["confidence"] }) {
  const variant = value === "high" ? "default" : value === "medium" ? "secondary" : "outline"
  return <Badge variant={variant}>{value}</Badge>
}

const columnDefs: ColDef<Idea>[] = [
  {
    field: "confidence",
    headerName: "Conf",
    width: 100,
    cellRenderer: (p: ICellRendererParams<Idea>) =>
      p.value ? <ConfidenceBadge value={p.value} /> : null,
  },
  { field: "ticker", headerName: "Ticker", width: 90 },
  {
    field: "structure",
    headerName: "Structure",
    width: 110,
    valueFormatter: (p) => STRUCTURE_LABELS[p.value as Idea["structure"]] ?? "-",
  },
  {
    colId: "leg1",
    headerName: "Leg 1",
    flex: 1,
    minWidth: 170,
    valueGetter: (p) => (p.data ? legLabel(p.data.leg1) : ""),
  },
  {
    colId: "leg2",
    headerName: "Leg 2",
    flex: 1,
    minWidth: 170,
    valueGetter: (p) => (p.data ? legLabel(p.data.leg2) : ""),
  },
  { field: "execEdgeDollars", headerName: "Exec Edge $", valueFormatter: fmtDollars, width: 120 },
  { field: "netEdgeDollars", headerName: "Net Edge $", valueFormatter: fmtDollars, width: 115 },
  { field: "capitalEmployed", headerName: "Capital $", valueFormatter: fmtDollars, width: 110 },
  {
    field: "returnOnCapital",
    headerName: "ROC",
    width: 90,
    valueFormatter: (p) => (typeof p.value === "number" ? `${(p.value * 100).toFixed(1)}%` : "–"),
  },
  {
    field: "flags",
    headerName: "Caveats",
    flex: 1,
    minWidth: 180,
    valueFormatter: (p) => (Array.isArray(p.value) && p.value.length ? p.value.join("; ") : "—"),
  },
]

export function IdeasView({ ideas, loading }: { ideas: Idea[]; loading: boolean }) {
  const [selected, setSelected] = useState<Idea | null>(null)
  const defaultColDef = useMemo(() => ({ resizable: true, sortable: true }), [])

  if (loading)
    return (
      <p className="text-sm text-muted-foreground">
        Scanning all tickers… first run prices the whole universe and can take a minute.
      </p>
    )
  if (!ideas.length)
    return <p className="text-sm text-muted-foreground">No qualifying ideas today.</p>

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Top {ideas.length} spreads across the universe, ranked by confidence then{" "}
        <strong className="text-foreground">executable edge</strong> — the profit left after
        filling buys at the ask, sells at the bid, and paying carry. Every structure pairs legs
        whose position deltas offset, isolating relative volatility rather than adding call/put
        delta exposure. High confidence = the edge survives realistic fills on tight, traded
        markets, priced off the fitted surface.
      </p>

      <div className="ag-theme-quartz w-full" style={{ height: 560 }}>
        <AgGridReact<Idea>
          rowData={ideas}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          rowSelection="single"
          onSelectionChanged={(e) => setSelected(e.api.getSelectedRows()[0] ?? null)}
        />
      </div>

      {selected && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {selected.ticker}: {legLabel(selected.leg1)}, {legLabel(selected.leg2)}
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
            <div>
              <p className="text-xs text-muted-foreground">Gross edge</p>
              <p>{dollars(selected.grossEdgeDollars)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Carry cost</p>
              <p>{dollars(selected.carryCost)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Margin requirement</p>
              <p>{dollars(selected.marginRequirement)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Net debit</p>
              <p>{dollars(selected.netDebit)}</p>
            </div>
            {selected.flags.length > 0 && (
              <div className="col-span-full">
                <p className="text-xs text-muted-foreground">Caveats</p>
                <ul className="list-disc pl-5">
                  {selected.flags.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {selected && (
        <div className="grid gap-4 lg:grid-cols-2">
          <LegCard
            title={`${selected.leg1.side === "buy" ? "Buy" : "Sell"} ${selected.leg1.qty}×`}
            leg={selected.leg1.detail}
          />
          <LegCard
            title={`${selected.leg2.side === "buy" ? "Buy" : "Sell"} ${selected.leg2.qty}×`}
            leg={selected.leg2.detail}
          />
        </div>
      )}
    </div>
  )
}
