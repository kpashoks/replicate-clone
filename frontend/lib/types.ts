// Mirrors of backend Pydantic models. Keep in sync with backend/models_registry.py
// and backend/routes/generate.py.

export type OutputKind = "image" | "video";

export type ModelEntry = {
  slug: string;
  label: string;
  description: string;
  workflow_file: string;
  output_kind: OutputKind;
  accepts_image: boolean;
  accepts_video: boolean;
  stage: number;
  available: boolean;
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
