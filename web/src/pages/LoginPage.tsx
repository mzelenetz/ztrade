import { useState, type FormEvent } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useAuth } from "@/context/AuthContext"

export function LoginPage() {
  const { requestMagicLink } = useAuth()
  const [email, setEmail] = useState("")
  const [sent, setSent] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      await requestMagicLink(email)
      setSent(true)
    } catch {
      setError("Something went wrong. Please try again.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-svh items-center justify-center bg-muted/30 p-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>ZTrade</CardTitle>
          <CardDescription>
            {sent
              ? "Check your inbox for a sign-in link"
              : "Enter your email to get a sign-in link"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sent ? (
            <div className="flex flex-col gap-4">
              <p className="text-sm text-muted-foreground">
                If <span className="font-medium text-foreground">{email}</span> is
                authorized, a link is on its way. It expires in a few minutes.
              </p>
              <Button
                variant="outline"
                onClick={() => {
                  setSent(false)
                  setError(null)
                }}
              >
                Use a different email
              </Button>
            </div>
          ) : (
            <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
              <div className="flex flex-col gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoFocus
                  required
                />
              </div>
              {error && <p className="text-sm text-destructive">{error}</p>}
              <Button type="submit" disabled={submitting}>
                {submitting ? "Sending…" : "Send sign-in link"}
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
