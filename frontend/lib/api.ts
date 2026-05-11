import type {
  GenerateResponse,
  Job,
  ModelEntry,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type EnhanceResponse = {
  original: string;
  enhanced: string;
  target_model: string;
};

export type UploadResponse = {
  id: string;
  url: string;
  name: string;
  size: number;
};

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = res.statusText;
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchModels(): Promise<ModelEntry[]> {
  const res = await fetch(`${API_BASE}/api/models`, { cache: "no-store" });
  return jsonOrThrow<ModelEntry[]>(res);
}

export async function submitJob(
  slug: string,
  params: Record<string, unknown>,
  inputIds: string[] = [],
): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE}/api/generate/${slug}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params, input_ids: inputIds }),
  });
  return jsonOrThrow<GenerateResponse>(res);
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
    cache: "no-store",
  });
  return jsonOrThrow<Job>(res);
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/uploads`, {
    method: "POST",
    body: fd,
  });
  return jsonOrThrow<UploadResponse>(res);
}

export async function enhancePrompt(
  prompt: string,
  targetModel: string,
): Promise<EnhanceResponse> {
  const res = await fetch(`${API_BASE}/api/prompts/enhance`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, target_model: targetModel }),
  });
  return jsonOrThrow<EnhanceResponse>(res);
}

export async function cancelJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, {
    method: "POST",
  });
  return jsonOrThrow<Job>(res);
}

/** Build an absolute URL for a server-served file path like "/api/files/outputs/<id>/<name>". */
export function fileUrl(path: string): string {
  if (!path.startsWith("/")) path = "/" + path;
  return `${API_BASE}${path}`;
}
