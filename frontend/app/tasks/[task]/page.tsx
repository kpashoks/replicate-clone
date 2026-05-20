"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EnhanceDiff } from "@/components/EnhanceDiff";
import { ImageDropzone } from "@/components/ImageDropzone";
import { ModelPicker } from "@/components/ModelPicker";
import { MultiImageDropzone } from "@/components/MultiImageDropzone";
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
  fetchModels,
  fileUrl,
  getJob,
  submitJob,
  type UploadResponse,
} from "@/lib/api";
import {
  TASK_META,
  TERMINAL_STATUSES,
  pickDefaultModel,
  type Job,
  type ModelEntry,
  type Task,
} from "@/lib/types";

const POLL_INTERVAL_MS = 2000;
const POLL_INTERVAL_VIDEO_MS = 3000;

const VIDEO_EXTS = [".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"];
function isVideoUrl(url: string): boolean {
  const lower = url.toLowerCase();
  return VIDEO_EXTS.some((ext) => lower.endsWith(ext));
}

type FormParams = {
  prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  fps: number;
  frames: number;
  seed: number;
  // LoRA passthrough (only used by Atlas models whose atlas_model_id
  // contains "lora"; the consolidated form keeps these fields hidden
  // otherwise so they don't clutter the UI).
  lora_url: string;
  lora_scale: number;
};

const DEFAULTS: FormParams = {
  prompt: "",
  width: 1024,
  height: 1024,
  steps: 20,
  guidance: 3.5,
  fps: 16,
  frames: 81,
  seed: -1,
  lora_url: "",
  lora_scale: 1.0,
};

/**
 * True when the selected Atlas model supports LoRA passthrough. Detected by
 * the substring "lora" in atlas_model_id (e.g. "black-forest-labs/flux-dev-lora").
 * Mirrors the gate in backend/jobs.py:_build_atlas_image_body which forwards
 * `loras: [...]` only for matching slugs to avoid 400s on non-LoRA models.
 */
function modelAcceptsLora(model: ModelEntry | null): boolean {
  if (!model) return false;
  const id = model.atlas_model_id?.toLowerCase() ?? "";
  return id.includes("lora");
}

/**
 * Curated FLUX LoRAs hosted on HuggingFace. Each is a one-click preset that
 * fills the lora_url input; users can still type any HF slug they want.
 * To add a new preset: drop in a row with a short user-facing label, the
 * full vendor/repo slug, and a one-line description that surfaces as the
 * chip's title (tooltip).
 */
type LoraPreset = {
  label: string;
  slug: string;
  description: string;
};

const LORA_PRESETS: LoraPreset[] = [
  {
    label: "Realism",
    slug: "strangerzonehf/Flux-Super-Realism-LoRA",
    description: "Hyperrealism / photoreal portraits. Atlas's own example.",
  },
  {
    label: "XLabs Realism",
    slug: "XLabs-AI/flux-RealismLora",
    description: "XLabs's realism aesthetic. Different vibe from Stranger Zone.",
  },
  {
    label: "Kodak Film",
    slug: "alvdansen/flux-koda",
    description: "Vintage Kodak film stock look. Warm tones, grain.",
  },
  {
    label: "Anime",
    slug: "prithivMLmods/Canopus-LoRA-Flux-Anime",
    description: "Anime / manga style. Strong line art.",
  },
  {
    label: "Tarot",
    slug: "multimodalart/flux-tarot-v1",
    description: "Tarot-card illustration style. Ornate borders, symbolism.",
  },
  {
    label: "Children's sketch",
    slug: "Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch",
    description: "Children's book illustration. Soft lines, watercolor-ish.",
  },
  {
    label: "90s anime",
    slug: "glif/90s-anime-art",
    description: "90s-era anime cel art. Limited palette, retro feel.",
  },
  {
    label: "Pixar",
    slug: "prithivMLmods/Canopus-Pixar-Art",
    description: "Pixar / 3D animation style. Stylized characters.",
  },
];

export default function TaskPage() {
  const router = useRouter();
  const params = useParams<{ task: string }>();
  const search = useSearchParams();
  const task = params.task as Task;
  const taskMeta = TASK_META[task];

  // ---- model registry / selection -------------------------------------
  const [allModels, setAllModels] = useState<ModelEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then(setAllModels)
      .catch((e) => setLoadError(String(e)));
  }, []);

  const taskModels = useMemo(
    () => (allModels || []).filter((m) => m.task === task),
    [allModels, task],
  );

  const queryModel = search.get("model") || "";
  const selectedModel = useMemo(() => {
    if (taskModels.length === 0) return null;
    const fromQuery = taskModels.find((m) => m.slug === queryModel);
    return fromQuery || pickDefaultModel(taskModels);
  }, [taskModels, queryModel]);

  const onSelectModel = useCallback(
    (slug: string) => {
      const sp = new URLSearchParams(search.toString());
      sp.set("model", slug);
      router.replace(`/tasks/${task}?${sp.toString()}`);
    },
    [router, search, task],
  );

  // Reset image uploads whenever the selected model changes — different models
  // have different ref-image conventions (slot count, semantic ordering), and
  // carrying uploads across would silently submit the wrong files.
  useEffect(() => {
    setImageUploads([null, null]);
    // Also clear LoRA fields so a slug typed for one model doesn't get
    // silently submitted against a different model later. The conditional
    // render already hides the inputs for non-LoRA models, but the form
    // state would persist invisibly without this.
    setForm((p) => ({ ...p, lora_url: "", lora_scale: 1.0 }));
  }, [selectedModel?.slug]);

  // ---- form state -----------------------------------------------------
  const [form, setForm] = useState<FormParams>(DEFAULTS);
  const [imageUploads, setImageUploads] = useState<(UploadResponse | null)[]>([
    null,
    null,
  ]);
  const [videoUpload, setVideoUpload] = useState<UploadResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [enhanceOpen, setEnhanceOpen] = useState(false);

  // ---- polling --------------------------------------------------------
  const [elapsedSec, setElapsedSec] = useState(0);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimers = useCallback(() => {
    if (pollTimer.current) clearTimeout(pollTimer.current);
    if (tickTimer.current) clearInterval(tickTimer.current);
    pollTimer.current = null;
    tickTimer.current = null;
  }, []);

  const pollInterval = task === "video-swap" ? POLL_INTERVAL_VIDEO_MS : POLL_INTERVAL_MS;

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
              setForm((p) => ({ ...p, seed: job.params.seed as number }));
            }
            return;
          }
          pollTimer.current = setTimeout(tick, pollInterval);
        } catch (e) {
          setSubmitError(String(e));
          stopTimers();
        }
      };
      pollTimer.current = setTimeout(tick, pollInterval);
    },
    [stopTimers, pollInterval],
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

  // ---- submission -----------------------------------------------------
  // Models with 3+ refs use the accumulating MultiImageDropzone; 1–2 use the
  // labeled-slots ImageDropzones (preserves semantic order for slugs like
  // image-char-swap where slot 0 = source, slot 1 = character).
  const maxRefs = selectedModel?.max_ref_images ?? 1;
  const useMultiUpload = task === "i2i" && maxRefs >= 3;
  const refSlots = useMultiUpload ? 0 : Math.min(maxRefs, 2);

  const requiredUploadsMet = (() => {
    if (task === "t2i") return true;
    if (task === "video-swap") return !!videoUpload && !!imageUploads[0];
    // i2i — at least one image must be uploaded (multi or labeled).
    return imageUploads.some((u) => u !== null);
  })();

  const onSubmit = async () => {
    if (!selectedModel) return;
    if (!selectedModel.available) {
      setSubmitError(
        "This model is marked as Coming soon. Wiring its provider into the backend is the next step (Phase 1b).",
      );
      return;
    }
    if (!requiredUploadsMet) {
      setSubmitError("Please upload the required inputs first.");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    stopTimers();
    try {
      const inputIds: string[] = [];
      if (task === "video-swap") {
        if (videoUpload) inputIds.push(videoUpload.id);
        if (imageUploads[0]) inputIds.push(imageUploads[0].id);
      } else if (task === "i2i") {
        for (const u of imageUploads) {
          if (u) inputIds.push(u.id);
        }
      }

      // Build params object trimmed to fields the selected slug expects.
      // Backend validates; sending extras gets ignored for most slugs.
      const paramsForSlug = buildParams(task, form);

      const res = await submitJob(selectedModel.slug, paramsForSlug, inputIds);
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
    setForm((p) => ({ ...p, seed: Math.floor(Math.random() * 0xffffffff) }));

  // ---- render helpers -------------------------------------------------
  if (!taskMeta) {
    return (
      <main className="container mx-auto py-12 px-4 max-w-3xl">
        <p className="text-destructive">Unknown task: {task}</p>
        <Link href="/" className="underline">Back to gallery</Link>
      </main>
    );
  }

  const isRunning = currentJob && !TERMINAL_STATUSES.has(currentJob.status);
  const isSucceeded = currentJob?.status === "succeeded";
  const outputPath = isSucceeded ? currentJob.output_files[0] : null;
  const outputUrl = outputPath ? fileUrl(outputPath) : null;
  const outputIsVideo = outputUrl ? isVideoUrl(outputUrl) : false;

  return (
    <main className="container mx-auto py-8 px-4 max-w-6xl">
      <nav className="mb-6 text-sm">
        <Link
          href="/"
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          ← All tasks
        </Link>
      </nav>

      <header className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">{taskMeta.label}</h1>
        <p className="text-muted-foreground mt-1">{taskMeta.description}</p>
      </header>

      {loadError && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive mb-6">
          Failed to load models: {loadError}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ===== Left: Picker + Inputs ====================================== */}
        <Card>
          <CardHeader>
            <CardTitle>Model & inputs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label>Model</Label>
              {selectedModel ? (
                <ModelPicker
                  models={taskModels}
                  selectedSlug={selectedModel.slug}
                  onSelect={onSelectModel}
                />
              ) : (
                <p className="text-sm text-muted-foreground">Loading…</p>
              )}
              {selectedModel && (
                <p className="text-xs text-muted-foreground">
                  {selectedModel.description}
                </p>
              )}
            </div>

            {/* ----- video upload (video-swap only) ----- */}
            {task === "video-swap" && (
              <VideoDropzone onUploaded={setVideoUpload} label="Source video" />
            )}

            {/* ----- image uploads (i2i + video-swap) ----- */}
            {task === "video-swap" && (
              <ImageDropzone
                label="Reference character"
                onUploaded={(u) =>
                  setImageUploads((arr) => {
                    const next = [...arr];
                    next[0] = u;
                    return next;
                  })
                }
              />
            )}

            {task === "i2i" && useMultiUpload && (
              <MultiImageDropzone
                max={maxRefs}
                label={`Reference images (up to ${maxRefs})`}
                onChange={(ups) => setImageUploads(ups)}
              />
            )}

            {task === "i2i" && !useMultiUpload && (
              <div className="space-y-3">
                {Array.from({ length: refSlots }).map((_, i) => (
                  <ImageDropzone
                    key={i}
                    label={
                      refSlots > 1
                        ? `Reference image ${i + 1}`
                        : "Source image"
                    }
                    onUploaded={(u) =>
                      setImageUploads((arr) => {
                        const next = [...arr];
                        next[i] = u;
                        return next;
                      })
                    }
                  />
                ))}
              </div>
            )}

            {/* ----- prompt ----- */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="prompt">
                  {task === "i2i" ? "Edit instruction" : "Prompt"}
                </Label>
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  onClick={() => setEnhanceOpen(true)}
                  disabled={!form.prompt.trim()}
                  title="Rewrite this prompt with Qwen3-4B (local)"
                >
                  ✨ Enhance
                </Button>
              </div>
              <Textarea
                id="prompt"
                placeholder={
                  task === "i2i"
                    ? "e.g. make it wear sunglasses; change the season to winter"
                    : "e.g. a red panda eating ramen, cinematic, golden hour"
                }
                rows={task === "t2i" ? 4 : 3}
                value={form.prompt}
                onChange={(e) =>
                  setForm((p) => ({ ...p, prompt: e.target.value }))
                }
              />
            </div>

            {/* ----- t2i size sliders ----- */}
            {task === "t2i" && (
              <div className="grid grid-cols-2 gap-4">
                <NumberSlider
                  label="Width"
                  value={form.width}
                  onChange={(v) => setForm((p) => ({ ...p, width: v }))}
                  min={256}
                  max={2048}
                  step={64}
                />
                <NumberSlider
                  label="Height"
                  value={form.height}
                  onChange={(v) => setForm((p) => ({ ...p, height: v }))}
                  min={256}
                  max={2048}
                  step={64}
                />
              </div>
            )}

            {/* ----- steps + guidance (t2i + i2i) ----- */}
            {(task === "t2i" || task === "i2i") && (
              <>
                <NumberSlider
                  label="Steps"
                  value={form.steps}
                  onChange={(v) => setForm((p) => ({ ...p, steps: v }))}
                  min={1}
                  max={50}
                  step={1}
                  hint="Higher = more refined, slower. 20 is a good default."
                />
                <NumberSlider
                  label="Guidance"
                  value={form.guidance}
                  onChange={(v) => setForm((p) => ({ ...p, guidance: v }))}
                  min={0.5}
                  max={10}
                  step={0.1}
                  hint={
                    task === "i2i"
                      ? "Kontext default 2.5. FLUX dev default 3.5."
                      : "Lower = more creative, higher = more literal. 3.5 is FLUX-default."
                  }
                  decimals={1}
                />
              </>
            )}

            {/* ----- video-swap fps + frames ----- */}
            {task === "video-swap" && (
              <div className="grid grid-cols-2 gap-4">
                <NumberSlider
                  label="FPS"
                  value={form.fps}
                  onChange={(v) => setForm((p) => ({ ...p, fps: v }))}
                  min={8}
                  max={30}
                  step={1}
                />
                <NumberSlider
                  label="Frames"
                  value={form.frames}
                  onChange={(v) => setForm((p) => ({ ...p, frames: v }))}
                  min={16}
                  max={161}
                  step={1}
                  hint={`~${(form.frames / form.fps).toFixed(1)}s`}
                />
              </div>
            )}

            {/* ----- seed (always) ----- */}
            <div className="space-y-2">
              <Label htmlFor="seed">
                Seed{" "}
                <span className="text-muted-foreground font-normal">
                  ({form.seed === -1 ? "random" : form.seed})
                </span>
              </Label>
              <div className="flex gap-2">
                <Input
                  id="seed"
                  type="number"
                  value={form.seed}
                  onChange={(e) =>
                    setForm((p) => ({
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
                  onClick={() => setForm((p) => ({ ...p, seed: -1 }))}
                >
                  Reset
                </Button>
              </div>
            </div>

            {/* ----- LoRA inputs (only for Atlas LoRA-capable models) ----- */}
            {modelAcceptsLora(selectedModel) && (
              <div className="space-y-3 rounded-md border border-border bg-muted/30 p-3">
                <div className="space-y-2">
                  <Label htmlFor="lora_url">
                    LoRA repo{" "}
                    <span className="text-muted-foreground font-normal">
                      (optional)
                    </span>
                  </Label>
                  <Input
                    id="lora_url"
                    type="text"
                    placeholder="e.g. strangerzonehf/Flux-Super-Realism-LoRA"
                    value={form.lora_url}
                    onChange={(e) =>
                      setForm((p) => ({ ...p, lora_url: e.target.value }))
                    }
                  />
                  <div className="space-y-1.5">
                    <p className="text-xs text-muted-foreground">
                      Quick picks (click to fill; you can also paste any HF
                      slug above):
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {LORA_PRESETS.map((preset) => {
                        const isActive =
                          form.lora_url.trim().toLowerCase() ===
                          preset.slug.toLowerCase();
                        return (
                          <button
                            key={preset.slug}
                            type="button"
                            onClick={() =>
                              setForm((p) => ({
                                ...p,
                                lora_url: preset.slug,
                              }))
                            }
                            title={`${preset.slug} — ${preset.description}`}
                            className={
                              isActive
                                ? "rounded-full border border-primary bg-primary/15 px-2.5 py-1 text-xs font-medium text-primary"
                                : "rounded-full border border-border bg-background px-2.5 py-1 text-xs text-muted-foreground hover:border-muted-foreground/40 hover:text-foreground transition-colors"
                            }
                          >
                            {preset.label}
                          </button>
                        );
                      })}
                      {form.lora_url.trim() && (
                        <button
                          type="button"
                          onClick={() =>
                            setForm((p) => ({ ...p, lora_url: "" }))
                          }
                          title="Clear LoRA"
                          className="rounded-full border border-border bg-background px-2.5 py-1 text-xs text-muted-foreground hover:border-destructive/60 hover:text-destructive transition-colors"
                        >
                          ✕ Clear
                        </button>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Atlas does NOT accept .safetensors URLs — must be an
                      HF repo slug (vendor/name).
                    </p>
                  </div>
                </div>
                <NumberSlider
                  label="LoRA strength"
                  value={form.lora_scale}
                  onChange={(v) => setForm((p) => ({ ...p, lora_scale: v }))}
                  min={0.0}
                  max={2.0}
                  step={0.05}
                  hint="1.0 default. 0.5-0.8 subtle, 1.0-1.5 strong, >1.5 often over-cooks."
                  decimals={2}
                />
              </div>
            )}

            <Button
              onClick={onSubmit}
              disabled={
                submitting ||
                !!isRunning ||
                !form.prompt.trim() ||
                !requiredUploadsMet ||
                !selectedModel?.available
              }
              className="w-full"
              size="lg"
            >
              {submitting
                ? "Submitting…"
                : isRunning
                  ? "Running…"
                  : !selectedModel?.available
                    ? "Coming soon"
                    : !requiredUploadsMet
                      ? "Upload required inputs"
                      : "Run"}
            </Button>

            {submitError && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {submitError}
              </div>
            )}
          </CardContent>
        </Card>

        {/* ===== Right: Status + Output ===================================== */}
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

                {outputUrl && !outputIsVideo && (
                  <div className="space-y-2">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={outputUrl}
                      alt="Generated output"
                      className="w-full rounded-md border border-border"
                    />
                    <OutputActions url={outputUrl} />
                  </div>
                )}

                {outputUrl && outputIsVideo && (
                  <div className="space-y-2">
                    <video
                      src={outputUrl}
                      controls
                      className="w-full rounded-md border border-border"
                    />
                    <OutputActions url={outputUrl} />
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

      {selectedModel && (
        <EnhanceDiff
          open={enhanceOpen}
          onClose={() => setEnhanceOpen(false)}
          originalPrompt={form.prompt}
          targetModel={selectedModel.slug}
          onAccept={(enhanced) => setForm((p) => ({ ...p, prompt: enhanced }))}
        />
      )}
    </main>
  );
}

// =====================================================================
// Param trimming
// =====================================================================
function buildParams(task: Task, f: FormParams): Record<string, unknown> {
  if (task === "t2i") {
    const body: Record<string, unknown> = {
      prompt: f.prompt,
      width: f.width,
      height: f.height,
      steps: f.steps,
      guidance: f.guidance,
      seed: f.seed,
    };
    // Only send LoRA fields when the user actually typed something. The
    // backend gates `loras` forwarding on the model slug (it drops the
    // fields silently for non-LoRA models), but keeping them out of the
    // request when unset means cleaner job records in data/jobs/<id>.json.
    const lora = f.lora_url.trim();
    if (lora) {
      body.lora_url = lora;
      body.lora_scale = f.lora_scale;
    }
    return body;
  }
  if (task === "i2i") {
    const body: Record<string, unknown> = {
      prompt: f.prompt,
      steps: f.steps,
      guidance: f.guidance,
      seed: f.seed,
    };
    // Same LoRA passthrough as t2i (used by atlas-flux-kontext-dev-lora).
    // Backend gates `loras` forwarding on the model slug, so it's safe to
    // send these for any i2i model; we only include them when the user
    // actually typed/picked something to keep job records tidy.
    const lora = f.lora_url.trim();
    if (lora) {
      body.lora_url = lora;
      body.lora_scale = f.lora_scale;
    }
    return body;
  }
  // video-swap
  return {
    prompt: f.prompt,
    steps: f.steps,
    fps: f.fps,
    frames: f.frames,
    seed: f.seed,
  };
}

// =====================================================================
// Sub-components
// =====================================================================
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
  const elapsed = isRunning
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

function OutputActions({ url }: { url: string }) {
  return (
    <div className="flex gap-2">
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className={buttonVariants({ variant: "outline", size: "sm" })}
      >
        Open full size
      </a>
      <a
        href={`${url}?download=1`}
        className={buttonVariants({ variant: "outline", size: "sm" })}
      >
        Download
      </a>
    </div>
  );
}
