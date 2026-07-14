import { Navigate, Route, Routes } from "react-router-dom"
import { LoginPage } from "@/pages/LoginPage"
import { VerifyPage } from "@/pages/VerifyPage"
import { DashboardPage } from "@/pages/DashboardPage"
import { MethodologyPage } from "@/pages/MethodologyPage"
import { SettingsPage } from "@/pages/SettingsPage"
import { ProtectedRoute } from "@/components/ProtectedRoute"

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/auth/verify" element={<VerifyPage />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/methodology"
        element={
          <ProtectedRoute>
            <MethodologyPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <SettingsPage />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
