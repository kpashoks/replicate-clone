"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2Icon, RefreshCwIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  getWanAnimateHealth,
  type WanAnimateHealth,
} from "@/lib/api";

/**
 * Health probe + restart-reminder for the self-hosted wan-animate Pod.
 *
 * The Pod has manual lifecycle: it costs $1.50-3/hr while running, so
 * the user routinely stops it. When they come back to use the self-hosted
 * character-swap model, they need to remember to:
 *   1. Resume the Pod on RunPod (or deploy a fresh one if terminated)
 *   2. Update .env if the Pod ID changed
 *   3. Start the inference server inside the Pod
 *   4. Restart uvicorn locally so the new URL loads
 *
 * Easy to forget any one step. This component:
 *   - Fires /api/wan-animate/health on mount
 *   - If status="up": small green badge, mostly invisible
 *   - If status="down" / "wrong_url" / "unconfigured": full reminder
 *     card with checklist + Re-check button
 *
 * Only render this when the user has actually picked the self-hosted
 * model. Atlas-hosted character-swap models don't need any of this.
 */
export function PodStatusBanner() {
  const [health, setHealth] = useState<WanAnimateHealth | null>(null);
  const [probing, setProbing] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const probe = useCallback(async () => {
    setProbing(true);
    setError(null);
    try {
      const h = await getWanAnimateHealth();
      setHealth(h);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setProbing(false);
    }
  }, []);

  useEffect(() => {
    probe();
  }, [probe]);

  if (probing && !health) {
    return (
      <div className="rounded-md border border-border bg-muted/40 p-2 text-xs text-muted-foreground">
        Checking Pod status…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
        Couldn&apos;t reach local backend to check Pod status: {error}
      </div>
    );
  }

  if (!health) return null;

  if (health.status === "up") {
    // Compact green confirmation. Don't take up much vertical space when
    // everything is fine.
    const torch = (health.debug_info?.["torch"] as string) || "";
    const cuda = health.debug_info?.["cuda_available"];
    return (
      <div className="flex items-center gap-2 rounded-md border border-green-500/40 bg-green-500/10 p-2 text-xs">
        <CheckCircle2Icon className="size-4 text-green-600 dark:text-green-400 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="font-medium text-green-700 dark:text-green-400">
            Pod is online
          </p>
          <p className="text-muted-foreground truncate font-mono">
            {health.endpoint}
            {torch && (
              <>
                {" "}
                · torch {torch} · cuda {cuda ? "✓" : "✗"}
              </>
            )}
          </p>
        </div>
        <Button
          variant="ghost"
          size="xs"
          onClick={probe}
          disabled={probing}
          title="Re-check"
        >
          <RefreshCwIcon className="size-3" />
        </Button>
      </div>
    );
  }

  // status is "down" | "wrong_url" | "unconfigured" -- show the full reminder
  return <PodDownReminder health={health} onRetry={probe} retrying={probing} />;
}

function PodDownReminder({
  health,
  onRetry,
  retrying,
}: {
  health: WanAnimateHealth;
  onRetry: () => void;
  retrying: boolean;
}) {
  const isUnconfigured = health.status === "unconfigured";
  const isWrongUrl = health.status === "wrong_url";

  return (
    <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm space-y-2">
      <div className="flex items-start gap-2">
        <span aria-hidden className="text-base leading-none">
          ⚠️
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-medium text-amber-700 dark:text-amber-400">
            {isUnconfigured
              ? "Pod URL not configured"
              : isWrongUrl
                ? "Reached a server but it isn't the wan-animate Pod"
                : "Pod is offline"}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {isUnconfigured ? (
              <>
                The <code>WAN_ANIMATE_ENDPOINT</code> env var is empty.
                Use any Atlas-hosted character-swap option instead, OR
                follow the steps below.
              </>
            ) : (
              <>
                Can&apos;t reach{" "}
                <code className="font-mono break-all">{health.endpoint}</code>.
                {" "}
                Atlas-hosted character-swap models work fine without
                this; otherwise restart the Pod with the steps below.
              </>
            )}
          </p>
        </div>
      </div>

      <details className="text-xs" open={!isUnconfigured}>
        <summary className="cursor-pointer text-amber-700 dark:text-amber-400 font-medium select-none">
          Resume / restart Pod (6 steps)
        </summary>
        <ol className="list-decimal pl-5 mt-2 space-y-1.5 text-muted-foreground">
          <li>
            Open{" "}
            <a
              href="https://runpod.io/console/pods"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-foreground"
            >
              runpod.io/console/pods
            </a>
            . If your Pod is listed and stopped, click{" "}
            <strong>Resume / Start</strong>. If it&apos;s gone, deploy a
            fresh one (see <code>NEXT-SESSION-COMMANDS.md</code> in the repo).
          </li>
          <li>
            Wait ~30 sec for boot. Check the Pod&apos;s ID in the
            dashboard — RunPod assigns a new one if you redeployed.
          </li>
          <li>
            If the Pod ID changed, edit <code>.env</code>:
            <pre className="mt-1 rounded bg-muted p-1.5 text-[10px] font-mono overflow-x-auto">
{`WAN_ANIMATE_ENDPOINT=https://<actual-pod-id>-8000.proxy.runpod.net`}
            </pre>
          </li>
          <li>
            In the Pod&apos;s Web Terminal, start the inference server:
            <pre className="mt-1 rounded bg-muted p-1.5 text-[10px] font-mono overflow-x-auto">
{`nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &`}
            </pre>
          </li>
          <li>
            Restart uvicorn locally so the new URL loads:
            <pre className="mt-1 rounded bg-muted p-1.5 text-[10px] font-mono overflow-x-auto">
{`# Ctrl+C in the uvicorn terminal, then:
uvicorn main:app --reload`}
            </pre>
          </li>
          <li>
            Click <strong>Re-check</strong> below. Once you see
            &quot;Pod is online&quot;, you can Run.
          </li>
        </ol>
      </details>

      <div className="flex items-center gap-2 pt-1">
        <Button
          variant="outline"
          size="sm"
          onClick={onRetry}
          disabled={retrying}
        >
          {retrying ? (
            <>
              <RefreshCwIcon className="size-3 mr-1.5 animate-spin" />
              Checking…
            </>
          ) : (
            <>
              <RefreshCwIcon className="size-3 mr-1.5" />
              Re-check
            </>
          )}
        </Button>
        {health.error && (
          <p className="text-[10px] text-muted-foreground font-mono truncate flex-1">
            {health.error}
          </p>
        )}
      </div>
    </div>
  );
}
