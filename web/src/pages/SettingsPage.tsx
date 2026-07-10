import { useMemo, useState } from "react"
import { Link } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ArrowLeft, Play, Save } from "lucide-react"
import { fetchIngestSettings, runIngestJob, saveIngestTickers } from "@/lib/api"
import {
  DEFAULT_RATE_CURVE,
  loadModelSettings,
  saveModelSettings,
  type PersistedModelSettings,
} from "@/lib/modelSettings"
import type { RateCurvePoint } from "@/types"
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
import { Textarea } from "@/components/ui/textarea"
import { ThemeToggle } from "@/components/ThemeToggle"

const TENOR_LABELS: Record<number, string> = {
  30: "1M", 91: "3M", 182: "6M", 365: "1Y", 730: "2Y", 1095: "3Y",
}

function IngestionCard() {
  const queryClient = useQueryClient()
  const settings = useQuery({ queryKey: ["ingestSettings"], queryFn: fetchIngestSettings })
  const [draft, setDraft] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)

  const text = draft ?? settings.data?.tickers.join("\n") ?? ""

  const save = useMutation({
    mutationFn: () =>
      saveIngestTickers(text.split("\n").map((t) => t.trim()).filter(Boolean)),
    onSuccess: (d) => {
      setDraft(null)
      setStatus(`Saved ${d.tickers.length} symbols — they apply from the next ingest run.`)
      queryClient.invalidateQueries({ queryKey: ["ingestSettings"] })
    },
    onError: (e: Error) => setStatus(`Save failed: ${e.message}`),
  })

  const run = useMutation({
    mutationFn: runIngestJob,
    onSuccess: (d) =>
      setStatus(
        `Ingest started (${d.execution}). The new close file appears in the date picker in a few minutes.`,
      ),
    onError: (e: Error) => setStatus(`Could not start ingest: ${e.message}`),
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Ingestion</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Symbols fetched by the daily post-close job (weekdays 4:45pm ET)
          {settings.data?.latestCloseDate && (
            <> · latest close file: <strong className="text-foreground">{settings.data.latestCloseDate}</strong></>
          )}
        </p>

        <div className="space-y-2">
          <Label>Symbols (one per line)</Label>
          <Textarea
            rows={8}
            className="max-w-xs font-mono"
            value={text}
            onChange={(e) => setDraft(e.target.value)}
            disabled={settings.data?.editable === false}
          />
          {settings.data?.editable === false && (
            <p className="text-xs text-muted-foreground">
              Local dev uses a fixture file — the symbol list is editable in the deployed app.
            </p>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            onClick={() => save.mutate()}
            disabled={draft === null || save.isPending || settings.data?.editable === false}
          >
            <Save className="size-4" /> Save symbols
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => run.mutate()}
            disabled={run.isPending || settings.data?.editable === false}
          >
            <Play className="size-4" /> {run.isPending ? "Starting…" : "Run ingest now"}
          </Button>
        </div>

        {status && <p className="text-sm text-muted-foreground">{status}</p>}
      </CardContent>
    </Card>
  )
}

function ModelDefaultsCard() {
  const [settings, setSettings] = useState<PersistedModelSettings>(() => ({
    marginStyle: "reg_t",
    marginRatePct: 11.325,
    rateCurve: DEFAULT_RATE_CURVE,
    ...loadModelSettings(),
  }))

  const update = (patch: PersistedModelSettings) => {
    const next = { ...settings, ...patch }
    setSettings(next)
    saveModelSettings(patch)
  }

  const curve = useMemo(
    () => [...(settings.rateCurve ?? DEFAULT_RATE_CURVE)].sort((a, b) => a[0] - b[0]),
    [settings.rateCurve],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Model defaults</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Saved in this browser and applied to all pricing. Changes take effect when you return to
          the dashboard.
        </p>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label>Margin type</Label>
            <Select
              value={settings.marginStyle}
              onValueChange={(v) => update({ marginStyle: v as "reg_t" | "portfolio" })}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="reg_t">Reg T</SelectItem>
                <SelectItem value="portfolio">Portfolio margin</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Margin rate (% / yr)</Label>
            <Input
              type="number"
              step={0.125}
              min={0}
              value={settings.marginRatePct}
              onChange={(e) => update({ marginRatePct: Number(e.target.value) })}
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Rate curve (% / yr)</Label>
          <div className="grid max-w-md grid-cols-3 gap-2">
            {curve.map(([days, rate]) => (
              <div key={days} className="space-y-0.5">
                <span className="text-xs text-muted-foreground">{TENOR_LABELS[days] ?? `${days}d`}</span>
                <Input
                  type="number"
                  step={0.05}
                  className="h-8"
                  value={(rate * 100).toFixed(2)}
                  onChange={(e) =>
                    update({
                      rateCurve: curve.map(([d, r]) =>
                        d === days ? [d, Number(e.target.value) / 100] : [d, r],
                      ) as RateCurvePoint[],
                    })
                  }
                />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function SettingsPage() {
  return (
    <div className="min-h-svh bg-background">
      <header className="border-b">
        <div className="mx-auto flex max-w-[900px] items-center justify-between gap-4 px-6 py-3">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" asChild aria-label="Back to dashboard">
              <Link to="/">
                <ArrowLeft className="size-4" />
              </Link>
            </Button>
            <h1 className="text-lg font-semibold">Settings</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-[900px] space-y-6 px-6 py-8">
        <IngestionCard />
        <ModelDefaultsCard />
      </main>
    </div>
  )
}
