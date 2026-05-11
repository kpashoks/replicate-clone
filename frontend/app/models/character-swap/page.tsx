"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { EnhanceDiff } from "@/components/EnhanceDiff";
import { ImageDropzone } from "@/components/ImageDropzone";
import { VideoDropzone } from "@/components/VideoDropzone";
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

import {
  cancelJob,
  fileUrl,
  getJob,
  submitJob,
  type UploadResponse,
} from "@/lib/api";
import {
  TERMINAL_STATUSES,
  type Job,
} from "@/lib/types";

const POLL_INTERVAL_MS = 3000; // slower poll for long-running video jobs

type CharacterSwapFormParams = {
  prompt: string;
  steps: number;
  fps: number;
  frames: number;
  seed: number;
};

const DEFAULTS: CharacterSwapFormParams = {
  prompt: "",
  steps: 20,
  fps: 16,
  frames: 81, // ~5 s at 16 fps
  seed: -1,
};

const VIDEO_EXTS = [".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"];

function isVideoUrl(url: string): boolean {
  const lower = url.toLowerCase();
  return VIDEO_EXTS.some((ext) => lower.endsWith(ext));
}

export default function CharacterSwapPage() {
  const [params, setParams] = useState<CharacterSwapFormParams>(DEFAULTS);
  const [video, setVideo] = useState<UploadResponse | null>(null);
  const [character, setCharacter] = useState<UploadResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [enhanceOpen, setEnhanceOpen] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);

  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimers = useCallback(() => {
    if (pollTimer.current) clearTimeout(pollTimer.current);
    if (tickTimer.current) clearInterval(tickTimer.current);
    pollTimer.current = null;
    tickTimer.current = null;
  }, []);

  const pollJob = useCallback(
    (jobId: string) => {
      const tick = async () => {
        try {
          const job = await getJob(jobId);
          setCurrentJob(job);
          if (TERMINAL_STATUSES.has(job.status)) {
            stopTimers();
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
    },
    [stopTimers],
  );

  useEffect(() => {
    if (!currentJob) return;
    if (TERMINAL_STATUSES.has(currentJob.status)) return;
    const start = currentJob.created_at * 1000;
    setElapsedSec(Math.floor((Date.now() - start) / 1000));
    tickTimer.current = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => {
      if (tickTimer.current) clearInterval(tickTimer.current);
    };
  }, [currentJob]);

  useEffect(() => stopTimers, [stopTimers]);

  const onSubmit = async () => {
    if (!video || !character) {
      setSubmitError("Please upload both a source video and a reference character image.");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    stopTimers();
    try {
      // Order matters: backend expects [video_id, character_id]
      const res = await submitJob("character-swap", params, [video.id, character.id]);
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

  const randomizeSeed = () =>
    setParams((p) => ({ ...p, seed: Math.floor(Math.random() * 0xffffffff) }));

  const isRunning = currentJob && !TERMINAL_STATUSES.has(currentJob.status);
  const isSucceeded = currentJob?.status === "succeeded";
  const outputUrl =
    isSucceeded && currentJob.output_files[0]
      ? fileUrl(currentJob.output_files[0])
      : null;
  const durationSec = (params.frames / params.fps).toFixed(1);

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
        <h1 className="text-3xl font-bold tracking-tight">Character Swap (Video)</h1>
        <p className="text-muted-foreground mt-1">
          Replace one character in a short video clip with a reference character
          image, using Wan 2.2 Animate. Wan transfers the source motion to the
          new character.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Inputs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <VideoDropzone
              onUploaded={setVideo}
              label="Source video (the motion to copy)"
              maxMB={16}
            />

            <ImageDropzone
              onUploaded={setCharacter}
              label="Reference character (the new face/body)"
            />

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="prompt">
                  Scene context{" "}
                  <span className="text-muted-foreground font-normal text-xs">
                    (optional)
                  </span>
                </Label>
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  onClick={() => setEnhanceOpen(true)}
                  disabled={!params.prompt.trim()}
                  title="Rewrite with Qwen3-4B"
                >
                  ✨ Enhance
                </Button>
              </div>
              <Textarea
                id="prompt"
                placeholder="e.g. dancing in a sunlit forest, cinematic"
                rows={2}
                value={params.prompt}
                onChange={(e) =>
                  setParams((p) => ({ ...p, prompt: e.target.value }))
                }
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between items-baseline">
                  <Label>FPS</Label>
                  <span className="text-sm font-mono text-muted-foreground">
                    {params.fps}
                  </span>
                </div>
                <Slider
                  value={[params.fps]}
                  min={8}
                  max={30}
                  step={1}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, fps: v }))
                  }
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-baseline">
                  <Label>Frames</Label>
                  <span className="text-sm font-mono text-muted-foreground">
                    {params.frames}
                  </span>
                </div>
                <Slider
                  value={[params.frames]}
                  min={17}
                  max={161}
                  step={8}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, frames: v }))
                  }
                />
              </div>
            </div>
            <p className="text-xs text-muted-foreground -mt-3">
              Output duration ≈ {durationSec} s
            </p>

            <div className="space-y-2">
              <div className="flex justify-between items-baseline">
                <Label>Steps</Label>
                <span className="text-sm font-mono text-muted-foreground">
                  {params.steps}
                </span>
              </div>
              <Slider
                value={[params.steps]}
                min={1}
                max={50}
                step={1}
                onValueChange={([v]) =>
                  setParams((p) => ({ ...p, steps: v }))
                }
              />
            </div>

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
                >
                  🎲
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setParams((p) => ({ ...p, seed: -1 }))}
                >
                  Reset
                </Button>
              </div>
            </div>

            <Button
              onClick={onSubmit}
              disabled={submitting || !!isRunning || !video || !character}
              className="w-full"
              size="lg"
            >
              {submitting
                ? "Submitting…"
                : isRunning
                  ? "Running… (this can take 10–15 min)"
                  : !video || !character
                    ? "Upload video + character first"
                    : "Run"}
            </Button>

            {submitError && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {submitError}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Output</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {!currentJob && (
              <div className="rounded-md border border-dashed border-muted-foreground/40 p-8 text-center text-sm text-muted-foreground">
                Upload a source video + a reference character, then click Run.
                Wan 2.2 Animate inference typically takes 8–15 min for a 5 s
                clip on an A6000.
              </div>
            )}

            {currentJob && (
              <>
                <StatusPanel
                  job={currentJob}
                  elapsedSec={elapsedSec}
                  onCancel={onCancel}
                />

                {outputUrl && (
                  <div className="space-y-2">
                    {isVideoUrl(outputUrl) ? (
                      <video
                        src={outputUrl}
                        controls
                        loop
                        className="w-full rounded-md border border-border"
                      />
                    ) : (
                      // Fallback: if backend ends up writing a PNG/sequence,
                      // show the first one. Useful while iterating on output handling.
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={outputUrl}
                        alt="Output"
                        className="w-full rounded-md border border-border"
                      />
                    )}
                    <div className="flex gap-2 flex-wrap">
                      <a
                        href={outputUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={buttonVariants({
                          variant: "outline",
                          size: "sm",
                        })}
                      >
                        Open in new tab
                      </a>
                      <a
                        href={outputUrl}
                        download
                        className={buttonVariants({
                          variant: "outline",
                          size: "sm",
                        })}
                      >
                        Download
                      </a>
                    </div>
                    {currentJob.output_files.length > 1 && (
                      <p className="text-xs text-muted-foreground">
                        {currentJob.output_files.length} output files total.
                        See <code>data/outputs/{currentJob.id}/</code>.
                      </p>
                    )}
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
        targetModel="character-swap"
        onAccept={(enhanced) =>
          setParams((p) => ({ ...p, prompt: enhanced }))
        }
      />
    </main>
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
  const elapsed = isRunning
    ? elapsedSec
    : Math.max(0, Math.floor(job.updated_at - job.created_at));
  const mm = Math.floor(elapsed / 60).toString().padStart(2, "0");
  const ss = (elapsed % 60).toString().padStart(2, "0");

  const styles: Record<Job["status"], string> = {
    queued: "bg-muted text-foreground",
    running: "bg-blue-500/15 text-blue-600 dark:text-blue-400",
    succeeded: "bg-green-500/15 text-green-600 dark:text-green-400",
    failed: "bg-red-500/15 text-red-600 dark:text-red-400",
    cancelled: "bg-yellow-500/15 text-yellow-600 dark:text-yellow-400",
  };

  return (
    <div className="rounded-md border border-border p-3 space-y-1.5 text-sm">
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">Status</span>
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded-full ${styles[job.status]}`}
        >
          {job.status}
        </span>
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
