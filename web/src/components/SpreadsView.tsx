import { useState } from "react"
import { AgGridReact } from "ag-grid-react"
import type { ColDef, ValueFormatterParams } from "ag-grid-community"
import { useTheme } from "next-themes"
import { gridThemeDark, gridThemeLight } from "@/lib/ag-grid-setup"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { Spread } from "@/types"
import { LegCard } from "@/components/LegCard"

function fmtNum(digits = 2) {
  return (params: ValueFormatterParams) =>
    typeof params.value === "number" ? params.value.toFixed(digits) : "-"
}

function fmtDollars(params: ValueFormatterParams) {
  return typeof params.value === "number" ? dollars(params.value) : "-"
}

const columnDefs: ColDef<Spread>[] = [
  { field: "buy", headerName: "Buy", flex: 1, minWidth: 140 },
  { field: "sell", headerName: "Sell", flex: 1, minWidth: 140 },
  { field: "buyQty", headerName: "Buy Qty", width: 90 },
  { field: "sellQty", headerName: "Sell Qty", width: 90 },
  { field: "netDelta", headerName: "Net Delta", valueFormatter: fmtNum(3), width: 100 },
  { field: "edge", headerName: "Edge", valueFormatter: fmtNum(4), width: 90 },
  { field: "marginRequirement", headerName: "Margin $", valueFormatter: fmtDollars, width: 110 },
  { field: "netDebit", headerName: "Net Debit $", valueFormatter: fmtDollars, width: 110 },
  { field: "carryCost", headerName: "Carry $", valueFormatter: fmtDollars, width: 100 },
  { field: "grossEdgeDollars", headerName: "Gross Edge $", valueFormatter: fmtDollars, width: 120 },
  { field: "netEdgeDollars", headerName: "Net Edge $", valueFormatter: fmtDollars, width: 120, sort: "desc" },
]

function dollars(value: number) {
  const rounded = Math.round(value)
  return rounded < 0 ? `-$${Math.abs(rounded).toLocaleString()}` : `$${rounded.toLocaleString()}`
}

function Field({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium">{value ?? "-"}</p>
    </div>
  )
}

export function SpreadsView({ spreads, loading }: { spreads: Spread[]; loading: boolean }) {
  const { resolvedTheme } = useTheme()
  const theme = resolvedTheme === "dark" ? gridThemeDark : gridThemeLight
  const [selected, setSelected] = useState<Spread | null>(null)

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading spreads…</p>
  }

  if (!spreads.length) {
    return <p className="text-sm text-muted-foreground">No spreads matched the current filters.</p>
  }

  return (
    <div className="space-y-4">
      <div style={{ height: Math.min(500, 42 + spreads.length * 36) }}>
        <AgGridReact
          theme={theme}
          rowData={spreads}
          columnDefs={columnDefs}
          defaultColDef={{ sortable: true, resizable: true }}
          rowSelection={{ mode: "singleRow" }}
          onRowClicked={(e) => setSelected(e.data ?? null)}
        />
      </div>

      {selected ? (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Position economics</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-5">
              <Field label="Margin requirement" value={dollars(selected.marginRequirement)} />
              <Field label="Net debit" value={dollars(selected.netDebit)} />
              <Field label="Carry cost" value={dollars(selected.carryCost)} />
              <Field label="Gross edge" value={dollars(selected.grossEdgeDollars)} />
              <Field label="Net edge" value={dollars(selected.netEdgeDollars)} />
            </CardContent>
          </Card>
          <div className="grid gap-4 lg:grid-cols-2">
            <LegCard title="Leg 1 (Buy)" leg={selected.buyLeg} />
            <LegCard title="Leg 2 (Sell)" leg={selected.sellLeg} />
          </div>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">Select a spread to view leg details.</p>
      )}
    </div>
  )
}
