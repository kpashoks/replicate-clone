"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { EnhanceDiff } from "@/components/EnhanceDiff";
import { ImageDropzone } from "@/components/ImageDropzone";
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
import { TERMINAL_STATUSES, type Job } from "@/lib/types";

const POLL_INTERVAL_MS = 2000;

type FormParams = {
  prompt: string;
  negative_prompt: string;
  steps: number;
  cfg: number;
  identity_strength: number;
  pose_strength: number;
  seed: number;
};

const DEFAULT_NEGATIVE =
  "low quality, worst quality, blurry, jpeg artifacts, watermark, signature, " +
  "text, deformed, distorted, mutated, extra fingers, missing fingers, " +
  "out of frame, cropped";

const DEFAULTS: FormParams = {
  prompt: "",
  negative_prompt: DEFAULT_NEGATIVE,
  steps: 25,
  cfg: 7.0,
  identity_strength: 0.7,
  pose_strength: 1.2,
  seed: -1,
};

export default function ImageCharSwapPage() {
  const [params, setParams] = useState<FormParams>(DEFAULTS);
  const [source, setSource] = useState<UploadResponse | null>(null);
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
            if (
              job.status === "succeeded" &&
              typeof job.params.seed === "number"
            ) {
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
    if (!source || !character) {
      setSubmitError("Please upload both a source image and a reference character image.");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    stopTimers();
    try {
      // Order matters: backend expects [source_id, character_id]
      const res = await submitJob("image-char-swap", params, [source.id, character.id]);
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
        <h1 className="text-3xl font-bold tracking-tight">
          Image Character Swap
        </h1>
        <p className="text-muted-foreground mt-1">
          Replace the person in a source image with a different reference
          character. Pose, framing, and composition are preserved via ControlNet
          OpenPose; identity comes from IP-Adapter on Juggernaut XL.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Inputs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <ImageDropzone
              onUploaded={setSource}
              label="Source image (the pose / scene to keep)"
            />

            <ImageDropzone
              onUploaded={setCharacter}
              label="Reference character (the identity to insert)"
            />

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="prompt">
                  Scene direction{" "}
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
                placeholder="e.g. cinematic, golden hour, photorealistic"
                rows={2}
                value={params.prompt}
                onChange={(e) =>
                  setParams((p) => ({ ...p, prompt: e.target.value }))
                }
              />
              <p className="text-xs text-muted-foreground">
                Identity comes from the character image — prompt only nudges
                style/lighting/mood.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="negative">Negative prompt</Label>
              <Textarea
                id="negative"
                rows={2}
                value={params.negative_prompt}
                onChange={(e) =>
                  setParams((p) => ({ ...p, negative_prompt: e.target.value }))
                }
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-baseline">
                <Label>Identity strength (IP-Adapter)</Label>
                <span className="text-sm font-mono text-muted-foreground">
                  {params.identity_strength.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[params.identity_strength]}
                min={0}
                max={2}
                step={0.05}
                onValueChange={([v]) =>
                  setParams((p) => ({ ...p, identity_strength: v }))
                }
              />
              <p className="text-xs text-muted-foreground">
                0.7 default (style-transfer mode). Lower = more freedom for ControlNet
                pose; higher (1.0–1.5) makes the result look more like the reference
                photo but at the cost of pose adherence.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-baseline">
                <Label>Pose strength (ControlNet)</Label>
                <span className="text-sm font-mono text-muted-foreground">
                  {params.pose_strength.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[params.pose_strength]}
                min={0}
                max={2}
                step={0.05}
                onValueChange={([v]) =>
                  setParams((p) => ({ ...p, pose_strength: v }))
                }
              />
              <p className="text-xs text-muted-foreground">
                1.2 default. Above 1.5 can over-constrain (stiff/uncanny poses);
                lower if proportions look off.
              </p>
            </div>

            <p className="text-xs text-muted-foreground">
              Output dimensions are auto-matched to your source image so the pose skeleton aligns correctly.
            </p>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between items-baseline">
                  <Label>Steps</Label>
                  <span className="text-sm font-mono text-muted-foreground">
                    {params.steps}
                  </span>
                </div>
                <Slider
                  value={[params.steps]}
                  min={10}
                  max={60}
                  step={1}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, steps: v }))
                  }
                />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-baseline">
                  <Label>CFG</Label>
                  <span className="text-sm font-mono text-muted-foreground">
                    {params.cfg.toFixed(1)}
                  </span>
                </div>
                <Slider
                  value={[params.cfg]}
                  min={1}
                  max={15}
                  step={0.5}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, cfg: v }))
                  }
                />
              </div>
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
              disabled={submitting || !!isRunning || !source || !character}
              className="w-full"
              size="lg"
            >
              {submitting
                ? "Submitting…"
                : isRunning
                  ? "Running…"
                  : !source || !character
                    ? "Upload both images first"
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
                Upload a source image + a reference character, optionally type a
                scene direction, and click Run. Expect ~60–90 s warm, ~3–4 min
                cold.
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
                      alt="Output"
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
