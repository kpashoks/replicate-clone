"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { uploadFile, fileUrl, type UploadResponse } from "@/lib/api";

type Props = {
  onUploaded: (upload: UploadResponse | null) => void;
  label?: string;
  accept?: string;
  /** Max upload size in MB (for client-side check before POST). */
  maxMB?: number;
  /** Pre-populate with an already-uploaded video (recipe load). Changing
   *  the id re-syncs. */
  initialUpload?: UploadResponse | null;
};

export function VideoDropzone({
  onUploaded,
  label = "Source video",
  accept = "video/mp4,video/quicktime,video/webm",
  maxMB = 32,
  initialUpload,
}: Props) {
  const [upload, setUpload] = useState<UploadResponse | null>(
    initialUpload ?? null,
  );

  // Re-sync when the parent hands us a restored upload (recipe load).
  // No onUploaded callback -- parent already set its own videoUpload state.
  useEffect(() => {
    if (initialUpload === undefined) return;
    setUpload(initialUpload ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialUpload?.id]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const file = Array.from(files)[0];
      if (!file) return;
      if (!file.type.startsWith("video/")) {
        setError("Please choose a video file (mp4, mov, webm).");
        return;
      }
      if (file.size > maxMB * 1024 * 1024) {
        setError(`File too large (${Math.round(file.size / 1024 / 1024)} MB). Max ${maxMB} MB.`);
        return;
      }
      setUploading(true);
      setError(null);
      setProgress(0);
      try {
        const res = await uploadFile(file, (pct) => setProgress(pct));
        setUpload(res);
        onUploaded(res);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setUpload(null);
        onUploaded(null);
      } finally {
        setUploading(false);
      }
    },
    [onUploaded, maxMB],
  );

  const clear = () => {
    setUpload(null);
    setError(null);
    onUploaded(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium leading-none">{label}</p>

      {upload ? (
        <div className="rounded-md border border-border p-2">
          <video
            src={fileUrl(upload.url)}
            controls
            className="w-full max-h-72 rounded"
          />
          <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
            <span className="truncate">
              {upload.name}{" "}
              <span className="opacity-60">
                ({Math.round(upload.size / 1024)} KB)
              </span>
            </span>
            <Button variant="outline" size="xs" onClick={clear}>
              Remove
            </Button>
          </div>
        </div>
      ) : (
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`block cursor-pointer rounded-md border-2 border-dashed p-8 text-center text-sm transition-colors ${
            isDragging
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/30 hover:border-muted-foreground/50"
          } ${uploading ? "opacity-50 cursor-wait" : ""}`}
        >
          <input
            ref={inputRef}
            type="file"
            accept={accept}
            className="hidden"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
            disabled={uploading}
          />
          {uploading ? (
            <span className="text-muted-foreground">
              Uploading{progress > 0 ? `… ${progress}%` : "…"}
            </span>
          ) : (
            <>
              <p className="font-medium">Drop a video here</p>
              <p className="text-muted-foreground text-xs mt-1">
                or click to browse · mp4, mov, webm · max {maxMB} MB
              </p>
              <p className="text-muted-foreground text-xs mt-1 opacity-70">
                Short clips work best (5–10 s, 480–720p)
              </p>
            </>
          )}
        </label>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
