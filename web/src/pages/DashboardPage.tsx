import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { BookOpen, LogOut, Settings } from "lucide-react"
import { fetchChain, fetchDividends, fetchIdeas, fetchMeta, fetchSpreads, fetchVolHistory, fetchVolSurface } from "@/lib/api"
import type { Filters } from "@/types"
import { useAuth } from "@/context/AuthContext"
import { ThemeToggle } from "@/components/ThemeToggle"
import { ChainView } from "@/components/ChainView"
import { SpreadsView } from "@/components/SpreadsView"
import { VolSurfaceView } from "@/components/VolSurfaceView"
import { DividendsView } from "@/components/DividendsView"
import { IdeasView } from "@/components/IdeasView"
import { ModelInputsCard } from "@/components/ModelInputsCard"
import { VolHistoryMini } from "@/components/VolHistoryMini"
import { Button } from "@/components/ui/button"
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
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DEFAULT_RATE_CURVE, loadModelSettings, saveModelSettings } from "@/lib/modelSettings"

const DEFAULT_FILTERS: Filters = {
  pricingModel: "mzpricer",
  closeDate: null,
  metric: "Last",
  deltaMin: 5,
  deltaMax: 95,
  minOptionPrice: 2.0,
  maxLastPrice: 1000.0,
  maxContractRatio: 2.5,
  maxStraddleRatio: 1.5,
  maxAbsNetDelta: 10.0,
  maxLegsPerSide: 50,
  maxResults: 50,
  minOpenInterest: 50,
  marginRatePct: 11.325,
  marginStyle: "reg_t",
  volMode: "surface",
  carryMode: "implied",
  dividends: {},
  dividendSchedule: {},
  rateCurve: DEFAULT_RATE_CURVE,
}

export function DashboardPage() {
  const { logout } = useAuth()
  const [filters, setFilters] = useState<Filters>(() => ({
    ...DEFAULT_FILTERS,
    ...loadModelSettings(),
  }))
  const [ticker, setTicker] = useState<string | null>(null)

  useEffect(() => {
    const { volMode, carryMode, dividends, dividendSchedule, rateCurve, marginRatePct, marginStyle } = filters
    saveModelSettings({ volMode, carryMode, dividends, dividendSchedule, rateCurve, marginRatePct, marginStyle })
  }, [filters])

  const metaQuery = useQuery({
    queryKey: ["meta", filters.pricingModel, filters.closeDate],
    queryFn: () => fetchMeta(filters.pricingModel, filters.closeDate),
  })

  const tickers = metaQuery.data?.tickers ?? []
  const availableDates = metaQuery.data?.availableDates ?? []

  useEffect(() => {
    if (tickers.length && (!ticker || !tickers.includes(ticker))) {
      setTicker(tickers[0])
    }
  }, [tickers, ticker])

  useEffect(() => {
    if (availableDates.length && !filters.closeDate) {
      setFilters((f) => ({ ...f, closeDate: availableDates[availableDates.length - 1] }))
    }
  }, [availableDates, filters.closeDate])

  const chainQuery = useQuery({
    queryKey: ["chain", ticker, filters],
    queryFn: () => fetchChain(ticker!, filters),
    enabled: Boolean(ticker),
  })

  const spreadsQuery = useQuery({
    queryKey: ["spreads", ticker, filters],
    queryFn: () => fetchSpreads(ticker!, filters),
    enabled: Boolean(ticker),
  })

  const volSurfaceQuery = useQuery({
    queryKey: [
      "volSurface", ticker, filters.pricingModel, filters.closeDate, filters.volMode,
      filters.carryMode, filters.dividends, filters.dividendSchedule, filters.rateCurve,
    ],
    queryFn: () => fetchVolSurface(ticker!, filters),
    enabled: Boolean(ticker),
  })

  const ideasQuery = useQuery({
    queryKey: ["ideas", filters],
    queryFn: () => fetchIdeas(filters),
    enabled: Boolean(ticker), // wait for meta so the universe is known
    staleTime: 5 * 60 * 1000, // scanning every ticker is expensive — don't refetch eagerly
  })

  const volHistoryQuery = useQuery({
    queryKey: ["volHistory", ticker, filters.closeDate],
    queryFn: () => fetchVolHistory(ticker!, filters.closeDate),
    enabled: Boolean(ticker),
  })

  const dividendsQuery = useQuery({
    queryKey: [
      "dividends", ticker, filters.pricingModel, filters.closeDate,
      filters.carryMode, filters.dividends, filters.dividendSchedule, filters.rateCurve,
    ],
    queryFn: () => fetchDividends(ticker!, filters),
    enabled: Boolean(ticker),
  })

  const updateFilter = <K extends keyof Filters>(key: K, value: Filters[K]) =>
    setFilters((f) => ({ ...f, [key]: value }))

  const stats = useMemo(() => {
    if (!chainQuery.data) return null
    return {
      spot: chainQuery.data.spot,
      dividendYield: chainQuery.data.dividendYield,
      vol30d: chainQuery.data.vol30d,
    }
  }, [chainQuery.data])

  return (
    <div className="min-h-svh bg-background">
      <header className="border-b">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-6 py-3">
          <h1 className="text-lg font-semibold">ZTrade</h1>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" asChild>
              <Link to="/methodology">
                <BookOpen className="size-4" />
                Methodology
              </Link>
            </Button>
            <Button variant="ghost" size="icon" asChild aria-label="Settings">
              <Link to="/settings">
                <Settings className="size-4" />
              </Link>
            </Button>
            <ThemeToggle />
            <Button variant="ghost" size="icon" onClick={logout} aria-label="Log out">
              <LogOut className="size-4" />
            </Button>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1600px] gap-6 px-6 py-6">
        <aside className="w-72 shrink-0 space-y-6">
          <Card>
            <CardContent className="space-y-4 pt-6">
              <div className="space-y-2">
                <Label>Ticker</Label>
                <Select value={ticker ?? undefined} onValueChange={setTicker}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select ticker" />
                  </SelectTrigger>
                  <SelectContent>
                    {tickers.map((t) => (
                      <SelectItem key={t} value={t}>
                        {t}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Pricing library</Label>
                <Select
                  value={filters.pricingModel}
                  onValueChange={(v) => updateFilter("pricingModel", v)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mzpricer">mzpricer</SelectItem>
                    <SelectItem value="quantlib">quantlib</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {availableDates.length > 0 && (
                <div className="space-y-2">
                  <Label>Close date</Label>
                  <Select
                    value={filters.closeDate ?? undefined}
                    onValueChange={(v) => updateFilter("closeDate", v)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {availableDates.map((d) => (
                        <SelectItem key={d} value={d}>
                          {d}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="space-y-2">
                <Label>Overvaluation relative to</Label>
                <Select
                  value={filters.metric}
                  onValueChange={(v) => updateFilter("metric", v as Filters["metric"])}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["Last", "Mid", "Bid", "Ask"].map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>
                  Call-delta range ({filters.deltaMin} – {filters.deltaMax})
                </Label>
                <Slider
                  min={0}
                  max={100}
                  step={1}
                  value={[filters.deltaMin, filters.deltaMax]}
                  onValueChange={([min, max]) => {
                    updateFilter("deltaMin", min)
                    updateFilter("deltaMax", max)
                  }}
                />
              </div>

              <div className="space-y-2">
                <Label>Min option price</Label>
                <Input
                  type="number"
                  step={0.05}
                  value={filters.minOptionPrice}
                  onChange={(e) => updateFilter("minOptionPrice", Number(e.target.value))}
                />
              </div>
            </CardContent>
          </Card>

          <ModelInputsCard filters={filters} updateFilter={updateFilter} />

          <Card>
            <CardContent className="space-y-4 pt-6">
              <p className="text-sm font-medium">Spread filters</p>

              <p className="text-xs text-muted-foreground">
                Margin type/rate and the rate curve live in Settings (gear icon).
              </p>

              <div className="space-y-2">
                <Label>Min open interest (ideas)</Label>
                <Input
                  type="number"
                  step={10}
                  min={0}
                  value={filters.minOpenInterest}
                  onChange={(e) => updateFilter("minOpenInterest", Number(e.target.value))}
                />
              </div>

              <div className="space-y-2">
                <Label>Max contract ratio</Label>
                <Input
                  type="number"
                  step={0.1}
                  value={filters.maxContractRatio}
                  onChange={(e) => updateFilter("maxContractRatio", Number(e.target.value))}
                />
              </div>

              <div className="space-y-2">
                <Label>Max straddle ratio</Label>
                <Input
                  type="number"
                  step={0.1}
                  value={filters.maxStraddleRatio}
                  onChange={(e) => updateFilter("maxStraddleRatio", Number(e.target.value))}
                />
              </div>

              <div className="space-y-2">
                <Label>Max last price</Label>
                <Input
                  type="number"
                  step={1}
                  value={filters.maxLastPrice}
                  onChange={(e) => updateFilter("maxLastPrice", Number(e.target.value))}
                />
              </div>

              <div className="space-y-2">
                <Label>Max |net delta|</Label>
                <Input
                  type="number"
                  step={0.5}
                  value={filters.maxAbsNetDelta}
                  onChange={(e) => updateFilter("maxAbsNetDelta", Number(e.target.value))}
                />
              </div>

              <div className="space-y-2">
                <Label>Max legs per side</Label>
                <Input
                  type="number"
                  step={10}
                  value={filters.maxLegsPerSide}
                  onChange={(e) => updateFilter("maxLegsPerSide", Number(e.target.value))}
                />
              </div>

              <div className="space-y-2">
                <Label>Max spread ideas</Label>
                <Input
                  type="number"
                  step={10}
                  value={filters.maxResults}
                  onChange={(e) => updateFilter("maxResults", Number(e.target.value))}
                />
              </div>
            </CardContent>
          </Card>
        </aside>

        <main className="min-w-0 flex-1 space-y-6">
          {ticker && stats && (
            <div className="grid grid-cols-3 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <p className="text-sm text-muted-foreground">
                    {ticker} — as of {chainQuery.data?.valuationDate ?? filters.closeDate ?? "…"}
                  </p>
                  <p className="text-2xl font-semibold">{stats.spot.toFixed(2)}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-sm text-muted-foreground">Dividend Yield</p>
                  <p className="text-2xl font-semibold">{stats.dividendYield.toFixed(2)}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="@container pt-6">
                  <p className="text-sm text-muted-foreground">30d Volatility</p>
                  <p className="text-2xl font-semibold">{stats.vol30d.toFixed(2)}</p>
                  <VolHistoryMini
                    points={volHistoryQuery.data?.points ?? []}
                    loading={volHistoryQuery.isLoading}
                  />
                </CardContent>
              </Card>
            </div>
          )}

          <Tabs defaultValue="chain">
            <TabsList>
              <TabsTrigger value="chain">Option Chain</TabsTrigger>
              <TabsTrigger value="spreads">Spreads</TabsTrigger>
              <TabsTrigger value="vol">Vol Surface</TabsTrigger>
              <TabsTrigger value="dividends">Dividends</TabsTrigger>
              <TabsTrigger value="ideas">Ideas</TabsTrigger>
            </TabsList>
            <TabsContent value="chain">
              <ChainView
                expiries={chainQuery.data?.expiries ?? []}
                loading={chainQuery.isLoading}
              />
            </TabsContent>
            <TabsContent value="spreads">
              <SpreadsView
                spreads={spreadsQuery.data?.spreads ?? []}
                loading={spreadsQuery.isLoading}
              />
            </TabsContent>
            <TabsContent value="vol">
              <VolSurfaceView
                expiries={volSurfaceQuery.data?.expiries ?? []}
                loading={volSurfaceQuery.isLoading}
              />
            </TabsContent>
            <TabsContent value="ideas">
              <IdeasView ideas={ideasQuery.data?.ideas ?? []} loading={ideasQuery.isLoading} />
            </TabsContent>
            <TabsContent value="dividends">
              {ticker && (
                <DividendsView
                  ticker={ticker}
                  filters={filters}
                  data={dividendsQuery.data}
                  loading={dividendsQuery.isLoading}
                  updateFilter={updateFilter}
                />
              )}
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  )
}
