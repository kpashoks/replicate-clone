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

export async function uploadFile(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadResponse> {
  // Uses XMLHttpRequest instead of fetch so we can:
  //  (a) surface real upload progress in the UI
  //  (b) get clearer, distinguishable errors (network vs server)
  // fetch() throws a generic "TypeError: Failed to fetch" for both cases.
  return new Promise<UploadResponse>((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/uploads`, true);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse);
        } catch (parseErr) {
          reject(new Error(`Upload succeeded (HTTP ${xhr.status}) but response was not JSON: ${String(parseErr)}`));
        }
        return;
      }
      let detail = xhr.statusText || "(no status text)";
      try {
        const body = JSON.parse(xhr.responseText);
        if (body?.detail) detail = body.detail;
      } catch {
        // not JSON; leave detail as-is
      }
      reject(new Error(`Upload failed (HTTP ${xhr.status}): ${detail}`));
    });
    xhr.addEventListener("error", () => {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      reject(
        new Error(
          `Network error uploading "${file.name}" (${sizeMb} MB). ` +
            "Backend may be down, request body may have exceeded a limit, or a browser extension blocked the request. " +
            "Check DevTools Network tab for the failed request.",
        ),
      );
    });
    xhr.addEventListener("abort", () => {
      reject(new Error("Upload aborted."));
    });
    xhr.send(fd);
  });
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

export type SaveResponse = {
  saved_path: string;
  filename: string;
};

export async function saveJobOutput(
  jobId: string,
  folder: string,
  filename: string,
  outputIndex = 0,
): Promise<SaveResponse> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      folder,
      filename,
      output_index: outputIndex,
    }),
  });
  return jsonOrThrow<SaveResponse>(res);
}

export type AppSettings = {
  default_download_dir: string;
};

export async function getSettings(): Promise<AppSettings> {
  const res = await fetch(`${API_BASE}/api/settings`, { cache: "no-store" });
  return jsonOrThrow<AppSettings>(res);
}

/** Build an absolute URL for a server-served file path like "/api/files/outputs/<id>/<name>". */
export function fileUrl(path: string): string {
  if (!path.startsWith("/")) path = "/" + path;
  return `${API_BASE}${path}`;
}
