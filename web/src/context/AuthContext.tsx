import { createContext, useContext, useState, type ReactNode } from "react"
import {
  clearToken,
  getToken,
  requestMagicLink as apiRequestMagicLink,
  setToken,
  verifyMagicLink as apiVerifyMagicLink,
} from "@/lib/api"

interface AuthContextValue {
  isAuthenticated: boolean
  requestMagicLink: (email: string) => Promise<void>
  completeLogin: (token: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(() => Boolean(getToken()))

  async function requestMagicLink(email: string) {
    await apiRequestMagicLink(email)
  }

  async function completeLogin(token: string) {
    const accessToken = await apiVerifyMagicLink(token)
    setToken(accessToken)
    setIsAuthenticated(true)
  }

  function logout() {
    clearToken()
    setIsAuthenticated(false)
  }

  return (
    <AuthContext.Provider
      value={{ isAuthenticated, requestMagicLink, completeLogin, logout }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
