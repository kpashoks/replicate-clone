"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { EnhanceDiff } from "@/components/EnhanceDiff";

import { cancelJob, fileUrl, getJob, submitJob } from "@/lib/api";
import {
  TERMINAL_STATUSES,
  TEXT_TO_IMAGE_DEFAULTS,
  type Job,
  type TextToImageParams,
} from "@/lib/types";

const POLL_INTERVAL_MS = 2000;

export default function TextToImagePage() {
  const [params, setParams] = useState<TextToImageParams>(
    TEXT_TO_IMAGE_DEFAULTS,
  );
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [enhanceOpen, setEnhanceOpen] = useState(false);

  // Live-ticking elapsed seconds while job is running.
  const [elapsedSec, setElapsedSec] = useState(0);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimers = useCallback(() => {
    if (pollTimer.current) {
      clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
    if (tickTimer.current) {
      clearInterval(tickTimer.current);
      tickTimer.current = null;
    }
  }, []);

  // Poll loop: every POLL_INTERVAL_MS, fetch the latest job state. Stop when terminal.
  const pollJob = useCallback((jobId: string) => {
    const tick = async () => {
      try {
        const job = await getJob(jobId);
        setCurrentJob(job);
        if (TERMINAL_STATUSES.has(job.status)) {
          stopTimers();
          // On success, surface the actual seed that was used so the user can re-roll.
          if (job.status === "succeeded" && typeof job.params.seed === "number") {
            setParams((p) => ({ ...p, seed: job.params.seed as number }));
          }
          return;
        }
        pollTimer.current = setTimeout(tick, POLL_INTERVAL_MS);
      } catch (e) {
        setSubmitError(String(e));
        stopTimers();
      }
    };
    pollTimer.current = setTimeout(tick, POLL_INTERVAL_MS);
  }, [stopTimers]);

  // Tick elapsed-seconds counter while a non-terminal job exists.
  useEffect(() => {
    if (!currentJob) return;
    if (TERMINAL_STATUSES.has(currentJob.status)) return;

    const start = currentJob.created_at * 1000;
    setElapsedSec(Math.floor((Date.now() - start) / 1000));
    tickTimer.current = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => {
      if (tickTimer.current) {
        clearInterval(tickTimer.current);
        tickTimer.current = null;
      }
    };
  }, [currentJob]);

  // Cleanup on unmount.
  useEffect(() => stopTimers, [stopTimers]);

  const onSubmit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    stopTimers();
    try {
      const res = await submitJob("text-to-image", params);
      // Fetch the freshly-created job once for an immediate render, then start polling.
      const job = await getJob(res.job_id);
      setCurrentJob(job);
      pollJob(res.job_id);
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const onCancel = async () => {
    if (!currentJob) return;
    try {
      const job = await cancelJob(currentJob.id);
      setCurrentJob(job);
      stopTimers();
    } catch (e) {
      setSubmitError(String(e));
    }
  };

  const randomizeSeed = () => {
    setParams((p) => ({ ...p, seed: Math.floor(Math.random() * 0xffffffff) }));
  };

  const isRunning =
    currentJob && !TERMINAL_STATUSES.has(currentJob.status);
  const isSucceeded = currentJob?.status === "succeeded";
  const outputImageUrl =
    isSucceeded && currentJob.output_files[0]
      ? fileUrl(currentJob.output_files[0])
      : null;

  return (
    <main className="container mx-auto py-8 px-4 max-w-6xl">
      <nav className="mb-6 text-sm">
        <Link
          href="/"
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          ← All models
        </Link>
      </nav>

      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Text to Image</h1>
        <p className="text-muted-foreground mt-1">
          Generate an image from a text prompt using FLUX.1 [dev].
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ============================================================== */}
        {/* Left column: inputs                                            */}
        {/* ============================================================== */}
        <Card>
          <CardHeader>
            <CardTitle>Inputs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="prompt">Prompt</Label>
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  onClick={() => setEnhanceOpen(true)}
                  disabled={!params.prompt.trim()}
                  title="Rewrite this prompt with Qwen3-4B (local)"
                >
                  ✨ Enhance
                </Button>
              </div>
              <Textarea
                id="prompt"
                placeholder="e.g. a red panda eating ramen, cinematic, golden hour"
                rows={4}
                value={params.prompt}
                onChange={(e) =>
                  setParams((p) => ({ ...p, prompt: e.target.value }))
                }
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <NumberSlider
                label="Width"
                value={params.width}
                onChange={(v) => setParams((p) => ({ ...p, width: v }))}
                min={256}
                max={2048}
                step={64}
              />
              <NumberSlider
                label="Height"
                value={params.height}
                onChange={(v) => setParams((p) => ({ ...p, height: v }))}
                min={256}
                max={2048}
                step={64}
              />
            </div>

            <NumberSlider
              label="Steps"
              value={params.steps}
              onChange={(v) => setParams((p) => ({ ...p, steps: v }))}
              min={1}
              max={50}
              step={1}
              hint="Higher = more refined, slower. 20 is a good default."
            />

            <NumberSlider
              label="Guidance"
              value={params.guidance}
              onChange={(v) => setParams((p) => ({ ...p, guidance: v }))}
              min={0.5}
              max={10}
              step={0.1}
              hint="Lower = more creative, higher = more literal. 3.5 is FLUX-default."
              decimals={1}
            />

            <div className="space-y-2">
              <Label htmlFor="seed">
                Seed{" "}
                <span className="text-muted-foreground font-normal">
                  ({params.seed === -1 ? "random" : params.seed})
                </span>
              </Label>
              <div className="flex gap-2">
                <Input
                  id="seed"
                  type="number"
                  value={params.seed}
                  onChange={(e) =>
                    setParams((p) => ({
                      ...p,
                      seed: parseInt(e.target.value || "-1", 10),
                    }))
                  }
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={randomizeSeed}
                  title="Pick a random seed"
                >
                  🎲
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setParams((p) => ({ ...p, seed: -1 }))}
                  title="Reset to random-each-time"
                >
                  Reset
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                -1 = random each run. After a successful generation the
                actual seed is filled in here so you can re-roll the same image.
              </p>
            </div>

            <Button
              onClick={onSubmit}
              disabled={submitting || isRunning || !params.prompt.trim()}
              className="w-full"
              size="lg"
            >
              {submitting
                ? "Submitting…"
                : isRunning
                  ? "Running…"
                  : "Run"}
            </Button>

            {submitError && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {submitError}
              </div>
            )}
          </CardContent>
        </Card>

        {/* ============================================================== */}
        {/* Right column: status + output                                  */}
        {/* ============================================================== */}
        <Card>
          <CardHeader>
            <CardTitle>Output</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {!currentJob && (
              <div className="rounded-md border border-dashed border-muted-foreground/40 p-8 text-center text-sm text-muted-foreground">
                Submit a prompt to see the output here.
              </div>
            )}

            {currentJob && (
              <>
                <StatusPanel
                  job={currentJob}
                  elapsedSec={elapsedSec}
                  onCancel={onCancel}
                />

                {outputImageUrl && (
                  <div className="space-y-2">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={outputImageUrl}
                      alt="Generated output"
                      className="w-full rounded-md border border-border"
                    />
                    <div className="flex gap-2">
                      <a
                        href={outputImageUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={buttonVariants({
                          variant: "outline",
                          size: "sm",
                        })}
                      >
                        Open full size
                      </a>
                      <a
                        href={`${outputImageUrl}?download=1`}
                        className={buttonVariants({
                          variant: "outline",
                          size: "sm",
                        })}
                      >
                        Download
                      </a>
                    </div>
                  </div>
                )}

                {currentJob.status === "failed" && currentJob.error && (
                  <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs font-mono text-destructive whitespace-pre-wrap">
                    {currentJob.error}
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <EnhanceDiff
        open={enhanceOpen}
        onClose={() => setEnhanceOpen(false)}
        originalPrompt={params.prompt}
        targetModel="text-to-image"
        onAccept={(enhanced) =>
          setParams((p) => ({ ...p, prompt: enhanced }))
        }
      />
    </main>
  );
}

// =================================================================
// Sub-components (kept in this file since they're page-specific)
// =================================================================

function NumberSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  hint,
  decimals = 0,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  hint?: string;
  decimals?: number;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-baseline">
        <Label>{label}</Label>
        <span className="text-sm font-mono text-muted-foreground">
          {value.toFixed(decimals)}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
      />
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

function StatusPanel({
  job,
  elapsedSec,
  onCancel,
}: {
  job: Job;
  elapsedSec: number;
  onCancel: () => void;
}) {
  const isRunning = !TERMINAL_STATUSES.has(job.status);
  const elapsed =
    isRunning
      ? elapsedSec
      : Math.max(0, Math.floor(job.updated_at - job.created_at));

  const mm = Math.floor(elapsed / 60).toString().padStart(2, "0");
  const ss = (elapsed % 60).toString().padStart(2, "0");

  return (
    <div className="rounded-md border border-border p-3 space-y-1.5 text-sm">
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">Status</span>
        <StatusBadge status={job.status} />
      </div>
      {job.runpod_status && (
        <div className="flex justify-between gap-2">
          <span className="text-muted-foreground">RunPod</span>
          <span className="font-mono text-xs">{job.runpod_status}</span>
        </div>
      )}
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">Elapsed</span>
        <span className="font-mono">
          {mm}:{ss}
        </span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">Job ID</span>
        <span className="font-mono text-xs">{job.id}</span>
      </div>
      {isRunning && (
        <div className="pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onCancel}
            className="w-full"
          >
            Cancel
          </Button>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: Job["status"] }) {
  const styles: Record<Job["status"], string> = {
    queued: "bg-muted text-foreground",
    running: "bg-blue-500/15 text-blue-600 dark:text-blue-400",
    succeeded: "bg-green-500/15 text-green-600 dark:text-green-400",
    failed: "bg-red-500/15 text-red-600 dark:text-red-400",
    cancelled: "bg-yellow-500/15 text-yellow-600 dark:text-yellow-400",
  };
  return (
    <span
      className={`text-xs font-medium px-2 py-0.5 rounded-full ${styles[status]}`}
    >
      {status}
    </span>
  );
}
