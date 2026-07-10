export interface LegSummary {
  fmv: number | null
  last: number | null
  bid: number | null
  ask: number | null
  mid: number | null
  bidAsk: string | null
  overvalued: number | null
  delta: number | null
  probItm: number | null
  probOtm: number | null
  marketIv: number | null
  modelVol: number | null
  volume: number | null
  volFromSurface: boolean | null
  openInterest: number | null
  bidSize: number | null
  askSize: number | null
  gamma: number | null
  vega: number | null
  theta: number | null
  rho: number | null
}

export interface LegDetail extends LegSummary {
  ticker: string
  expiry: string
  strike: number
  type: "C" | "P"
}

export interface ChainRow {
  strike: number
  call: LegSummary
  put: LegSummary
}

export interface ChainExpiry {
  expiry: string
  rows: ChainRow[]
}

export interface ChainResponse {
  ticker: string
  spot: number
  dividendYield: number
  vol30d: number
  valuationDate: string
  expiries: ChainExpiry[]
}

export interface VolSurfacePoint {
  strike: number
  marketIv: number
  type: "C" | "P"
}

export interface VolSurfaceExpiry {
  expiry: string
  spot: number
  points: VolSurfacePoint[]
  fitted: { strike: number; iv: number }[]
  fallback: boolean
  rate: number
  divYield: number
  divSource: "schedule" | "implied" | "manual"
}

export interface DividendExpiry {
  expiry: string
  tYears: number
  spot: number
  impliedForward: number | null
  impliedDivPV: number | null
  scheduledDivPV: number
  divSource: "schedule" | "implied" | "manual"
}

export interface SeedSuggestion {
  windowStart: string
  windowEnd: string
  amount: number
}

export interface DividendsResponse {
  ticker: string
  valuationDate: string
  expiries: DividendExpiry[]
  seedSuggestions: SeedSuggestion[]
}

export type DividendRow = [string, number] // [ex-date ISO, amount]

export interface VolSurfaceResponse {
  ticker: string
  expiries: VolSurfaceExpiry[]
}

export type RateCurvePoint = [number, number] // [days, rate]

export interface Spread {
  buy: string
  sell: string
  buyQty: number
  sellQty: number
  netDelta: number
  edge: number
  marginRequirement: number
  netDebit: number
  carryCost: number
  grossEdgeDollars: number
  netEdgeDollars: number
  execEdgeDollars: number | null
  capitalEmployed: number
  buyLeg: LegDetail
  sellLeg: LegDetail
}

export interface Idea extends Spread {
  ticker: string
  confidence: "high" | "medium" | "low"
  flags: string[]
  returnOnCapital: number | null
}

export interface IdeasResponse {
  ideas: Idea[]
}

export interface SpreadsResponse {
  spreads: Spread[]
}

export interface MetaResponse {
  dataSourceType: string
  availableDates: string[]
  tickers: string[]
}

export interface Filters {
  pricingModel: string
  closeDate: string | null
  metric: "Last" | "Mid" | "Bid" | "Ask"
  deltaMin: number
  deltaMax: number
  minOptionPrice: number
  maxLastPrice: number
  maxContractRatio: number
  maxStraddleRatio: number
  maxAbsNetDelta: number
  maxLegsPerSide: number
  maxResults: number
  marginRatePct: number
  marginStyle: "reg_t" | "portfolio"
  volMode: "surface" | "realized_anchor" | "flat"
  carryMode: "implied" | "manual"
  dividends: Record<string, number> // ticker → annual yield (decimal)
  dividendSchedule: Record<string, DividendRow[]> // ticker → [(ex-date, amount)]
  rateCurve: RateCurvePoint[]
}
