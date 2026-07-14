import { useEffect, useRef, useState } from "react"
import { Link, useNavigate, useSearchParams } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useAuth } from "@/context/AuthContext"

export function VerifyPage() {
  const { completeLogin } = useAuth()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const [failed, setFailed] = useState(false)
  // React StrictMode double-invokes effects in dev; guard so the single-use
  // token isn't spent twice.
  const ran = useRef(false)

  useEffect(() => {
    if (ran.current) return
    ran.current = true

    const token = params.get("token")
    if (!token) {
      setFailed(true)
      return
    }
    completeLogin(token)
      .then(() => navigate("/", { replace: true }))
      .catch(() => setFailed(true))
  }, [completeLogin, navigate, params])

  return (
    <div className="flex min-h-svh items-center justify-center bg-muted/30 p-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>ZTrade</CardTitle>
          <CardDescription>
            {failed ? "This sign-in link didn't work" : "Signing you in…"}
          </CardDescription>
        </CardHeader>
        {failed && (
          <CardContent className="flex flex-col gap-4">
            <p className="text-sm text-muted-foreground">
              The link may have expired or already been used. Request a fresh one.
            </p>
            <Button asChild>
              <Link to="/login">Back to sign in</Link>
            </Button>
          </CardContent>
        )}
      </Card>
    </div>
  )
}
