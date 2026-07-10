import type { Filters, RateCurvePoint } from "@/types"

export const MODEL_INPUTS_KEY = "ztrade_model_inputs"

export const DEFAULT_RATE_CURVE: RateCurvePoint[] = [
  [30, 0.043],
  [91, 0.0432],
  [182, 0.0428],
  [365, 0.0415],
  [730, 0.0405],
  [1095, 0.04],
]

export type PersistedModelSettings = Partial<
  Pick<
    Filters,
    | "volMode"
    | "carryMode"
    | "dividends"
    | "dividendSchedule"
    | "rateCurve"
    | "marginRatePct"
    | "marginStyle"
  >
>

export function loadModelSettings(): PersistedModelSettings {
  try {
    const raw = localStorage.getItem(MODEL_INPUTS_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    const saved: PersistedModelSettings = {
      // legacy "historical" mode = flat trailing realized vol
      volMode: parsed.volMode === "historical" ? "flat" : parsed.volMode,
      carryMode: parsed.carryMode,
      dividends: parsed.dividends,
      dividendSchedule: parsed.dividendSchedule,
      rateCurve: parsed.rateCurve,
      marginRatePct: parsed.marginRatePct,
      marginStyle: parsed.marginStyle,
    }
    return Object.fromEntries(
      Object.entries(saved).filter(([, v]) => v !== undefined),
    ) as PersistedModelSettings
  } catch {
    return {}
  }
}

export function saveModelSettings(settings: PersistedModelSettings) {
  localStorage.setItem(
    MODEL_INPUTS_KEY,
    JSON.stringify({ ...loadModelSettings(), ...settings }),
  )
}
