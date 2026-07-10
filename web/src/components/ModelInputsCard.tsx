import type { Filters } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface Props {
  filters: Filters
  updateFilter: <K extends keyof Filters>(key: K, value: Filters[K]) => void
}

export function ModelInputsCard({ filters, updateFilter }: Props) {
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
              <SelectItem value="realized_anchor">Surface anchored to 30d realized</SelectItem>
              <SelectItem value="flat">Flat 30d realized (naive)</SelectItem>
            </SelectContent>
          </Select>
          {filters.volMode === "realized_anchor" && (
            <p className="text-xs text-muted-foreground">
              The fitted surface shifted so its ~30d ATM vol equals the stock's trailing realized
              vol — a vol-reversion view that keeps skew and term shape. Expect a systematic
              short-vol lean: implied usually trades above realized.
            </p>
          )}
          {filters.volMode === "flat" && (
            <p className="text-xs text-muted-foreground">
              Every contract priced at the stock's trailing 30d realized vol — the naive
              baseline, for comparison against the surface modes. Its "edge" includes skew and
              term structure, which the market prices for a reason: wings and long-dated
              contracts will look mispriced even when they aren't.
            </p>
          )}
        </div>

        <p className="text-xs text-muted-foreground">
          Dividends are managed in the Dividends tab (schedule → market-implied → manual).
        </p>

      </CardContent>
    </Card>
  )
}
