import axios from "axios"
import type {
  ChainResponse,
  DividendsResponse,
  Filters,
  MetaResponse,
  SpreadsResponse,
  VolSurfaceResponse,
} from "@/types"

const TOKEN_KEY = "ztrade_token"

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

export const api = axios.create({ baseURL: "/api" })

api.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearToken()
      if (!window.location.pathname.startsWith("/login")) {
        window.location.assign("/login")
      }
    }
    return Promise.reject(error)
  },
)

export async function login(username: string, password: string): Promise<string> {
  const { data } = await api.post<{ access_token: string }>("/auth/login", {
    username,
    password,
  })
  return data.access_token
}

export async function fetchMeta(pricingModel: string, closeDate: string | null) {
  const { data } = await api.get<MetaResponse>("/meta", {
    params: { pricing_model: pricingModel, close_date: closeDate ?? undefined },
  })
  return data
}

function modelInputParams(filters: Filters) {
  return {
    vol_mode: filters.volMode,
    carry_mode: filters.carryMode,
    dividends: JSON.stringify(filters.dividends),
    dividend_schedule: JSON.stringify(filters.dividendSchedule),
    rate_curve: JSON.stringify(filters.rateCurve),
  }
}

export async function fetchChain(ticker: string, filters: Filters) {
  const { data } = await api.get<ChainResponse>("/chain", {
    params: {
      ticker,
      pricing_model: filters.pricingModel,
      close_date: filters.closeDate ?? undefined,
      metric: filters.metric,
      delta_min: filters.deltaMin,
      delta_max: filters.deltaMax,
      min_price: filters.minOptionPrice,
      ...modelInputParams(filters),
    },
  })
  return data
}

export async function fetchDividends(ticker: string, filters: Filters) {
  const { data } = await api.get<DividendsResponse>("/dividends", {
    params: {
      ticker,
      pricing_model: filters.pricingModel,
      close_date: filters.closeDate ?? undefined,
      ...modelInputParams(filters),
    },
  })
  return data
}

export async function fetchVolSurface(ticker: string, filters: Filters) {
  const { data } = await api.get<VolSurfaceResponse>("/vol-surface", {
    params: {
      ticker,
      pricing_model: filters.pricingModel,
      close_date: filters.closeDate ?? undefined,
      ...modelInputParams(filters),
    },
  })
  return data
}

export async function fetchSpreads(ticker: string, filters: Filters) {
  const { data } = await api.get<SpreadsResponse>("/spreads", {
    params: {
      ticker,
      pricing_model: filters.pricingModel,
      close_date: filters.closeDate ?? undefined,
      metric: filters.metric,
      delta_min: filters.deltaMin,
      delta_max: filters.deltaMax,
      max_contract_ratio: filters.maxContractRatio,
      max_straddle_ratio: filters.maxStraddleRatio,
      min_option_price: filters.minOptionPrice,
      max_last_price: filters.maxLastPrice,
      max_abs_net_delta: filters.maxAbsNetDelta,
      max_legs_per_side: filters.maxLegsPerSide,
      max_results: filters.maxResults,
      margin_rate: filters.marginRatePct / 100,
      margin_style: filters.marginStyle,
      ...modelInputParams(filters),
    },
  })
  return data
}
