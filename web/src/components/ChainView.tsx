import { useMemo } from "react"
import { AgGridReact } from "ag-grid-react"
import type { ColDef, ColGroupDef, ValueFormatterParams } from "ag-grid-community"
import { useTheme } from "next-themes"
import { gridThemeDark, gridThemeLight } from "@/lib/ag-grid-setup"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import type { ChainExpiry, ChainRow, LegSummary } from "@/types"

function fmtNum(digits = 2) {
  return (params: ValueFormatterParams) =>
    typeof params.value === "number" ? params.value.toFixed(digits) : "-"
}

function fmtPct(params: ValueFormatterParams) {
  return typeof params.value === "number" ? `${(params.value * 100).toFixed(1)}%` : "-"
}

function legColumns(side: "call" | "put"): ColGroupDef {
  const field = (key: keyof LegSummary) => `${side}.${key}`
  return {
    headerName: side === "call" ? "Call" : "Put",
    marryChildren: true,
    children: [
      { field: field("fmv"), headerName: "FMV", valueFormatter: fmtNum(2), width: 90 },
      { field: field("last"), headerName: "Last", valueFormatter: fmtNum(2), width: 90 },
      { field: field("bidAsk"), headerName: "Bid/Ask", width: 130 },
      { field: field("overvalued"), headerName: "%Over", valueFormatter: fmtPct, width: 90 },
      { field: field("delta"), headerName: "Delta", valueFormatter: fmtNum(3), width: 90 },
      { field: field("marketIv"), headerName: "Mkt IV", valueFormatter: fmtPct, width: 90 },
      { field: field("modelVol"), headerName: "Fit Vol", valueFormatter: fmtPct, width: 90 },
      { field: field("probItm"), headerName: "Prob ITM", valueFormatter: fmtPct, width: 100 },
      { field: field("probOtm"), headerName: "Prob OTM", valueFormatter: fmtPct, width: 100 },
    ] as ColDef[],
  }
}

const columnDefs: (ColDef | ColGroupDef)[] = [
  { field: "strike", headerName: "Strike", pinned: "left", width: 100, sort: "asc" },
  legColumns("call"),
  legColumns("put"),
]

function useGridTheme() {
  const { resolvedTheme } = useTheme()
  return resolvedTheme === "dark" ? gridThemeDark : gridThemeLight
}

function ExpiryGrid({ rows }: { rows: ChainRow[] }) {
  const theme = useGridTheme()
  const height = Math.min(600, 42 + rows.length * 36)

  return (
    <div style={{ height }}>
      <AgGridReact
        theme={theme}
        rowData={rows}
        columnDefs={columnDefs}
        defaultColDef={{ sortable: true, resizable: true }}
        animateRows={false}
      />
    </div>
  )
}

export function ChainView({ expiries, loading }: { expiries: ChainExpiry[]; loading: boolean }) {
  const defaultOpen = useMemo(() => expiries.map((e) => e.expiry), [expiries])

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading chain…</p>
  }

  if (!expiries.length) {
    return <p className="text-sm text-muted-foreground">No expiries matched the current filters.</p>
  }

  return (
    <Accordion type="multiple" defaultValue={defaultOpen} className="space-y-2">
      {expiries.map((expiry) => (
        <AccordionItem key={expiry.expiry} value={expiry.expiry} className="rounded-md border px-3">
          <AccordionTrigger className="font-semibold">{expiry.expiry}</AccordionTrigger>
          <AccordionContent>
            <ExpiryGrid rows={expiry.rows} />
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  )
}
