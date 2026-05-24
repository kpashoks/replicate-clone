// Mirrors of backend Pydantic models. Keep in sync with backend/models_registry.py
// and backend/routes/generate.py.

export type OutputKind = "image" | "video";

export type Task = "t2i" | "i2i" | "video-swap" | "t2v" | "i2v" | "v2v";

export type Provider = "runpod" | "wan-animate-http" | "atlas";

export type SpeedBucket = "fast" | "medium" | "slow";

export type ModelEntry = {
  slug: string;
  label: string;
  description: string;
  workflow_file: string | null;
  atlas_model_id: string | null;
  output_kind: OutputKind;
  provider: Provider;
  accepts_image: boolean;
  accepts_video: boolean;
  stage: number;
  available: boolean;

  // Picker metadata (Phase 1)
  task: Task;
  nsfw: boolean;
  speed: SpeedBucket;
  best_for: string;
  price_per_image_usd: number | null;
  max_ref_images: number | null;
  min_image_dim: number | null;
  provider_label: string;
};

export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export type Job = {
  id: string;
  slug: string;
  status: JobStatus;
  params: Record<string, unknown>;
  runpod_request_id: string | null;
  runpod_status: string | null;
  output_files: string[];
  error: string | null;
  created_at: number;
  updated_at: number;
};

export type GenerateResponse = {
  job_id: string;
  status: JobStatus;
};

// Text-to-image input schema (matches TextToImageParams in routes/generate.py)
export type TextToImageParams = {
  prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
};

export const TEXT_TO_IMAGE_DEFAULTS: TextToImageParams = {
  prompt: "",
  width: 1024,
  height: 1024,
  steps: 20,
  guidance: 3.5,
  seed: -1, // -1 = random; backend fills in a real seed
};

export const TERMINAL_STATUSES: ReadonlySet<JobStatus> = new Set([
  "succeeded",
  "failed",
  "cancelled",
]);

// ---- Task-level UI helpers --------------------------------------------

export const TASK_META: Record<
  Task,
  { label: string; description: string; slug: Task }
> = {
  t2i: {
    slug: "t2i",
    label: "Text to Image",
    description: "Generate an image from a text prompt.",
  },
  i2i: {
    slug: "i2i",
    label: "Image Edit",
    description: "Edit or transform an uploaded image with a text prompt.",
  },
  "video-swap": {
    slug: "video-swap",
    label: "Character Swap (Video)",
    description: "Replace a character in a video with a reference image.",
  },
  t2v: {
    slug: "t2v",
    label: "Text to Video",
    description:
      "Generate a short video from a text prompt. Atlas-hosted models — Seedance, HappyHorse, Kling, Veo, Wan T2V.",
  },
  i2v: {
    slug: "i2v",
    label: "Image to Video",
    description:
      "Animate a still image into a short clip. Atlas-hosted — HappyHorse, Veo, Seedance, Kling, Vidu, plus NSFW Wan Spicy variants.",
  },
  v2v: {
    slug: "v2v",
    label: "Video to Video",
    description:
      "Edit or extend a source video with a text prompt. Atlas-hosted — HappyHorse Video Edit, Wan 2.7 Video Edit, Wan video-extend variants.",
  },
};

export function groupByTask(models: ModelEntry[]): Record<Task, ModelEntry[]> {
  const out: Record<Task, ModelEntry[]> = {
    t2i: [],
    i2i: [],
    "video-swap": [],
    t2v: [],
    i2v: [],
    v2v: [],
  };
  for (const m of models) out[m.task].push(m);
  return out;
}

/** Pick a sensible default model for a task: first SFW + available, else first available, else first. */
export function pickDefaultModel(
  models: ModelEntry[],
): ModelEntry | null {
  if (models.length === 0) return null;
  return (
    models.find((m) => !m.nsfw && m.available) ||
    models.find((m) => m.available) ||
    models[0]
  );
}

export function formatPrice(usd: number | null): string {
  if (usd === null) return "Self-hosted";
  if (usd < 0.01) return `$${usd.toFixed(3)}/img`;
  return `$${usd.toFixed(2)}/img`;
}

export function speedLabel(s: SpeedBucket): string {
  return s === "fast" ? "Fast" : s === "medium" ? "Medium" : "Slow";
}
