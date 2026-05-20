"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  readImageDimensions,
  uploadFile,
  fileUrl,
  type UploadResponse,
} from "@/lib/api";

type Props = {
  /** Called once an upload completes; receives the upload id (or null on clear). */
  onUploaded: (upload: UploadResponse | null) => void;
  /** Optional label shown above the dropzone. */
  label?: string;
  /** Accept attribute for the file input. */
  accept?: string;
  /** Minimum px on the shortest side that the selected model accepts.
   *  When set, undersized uploads get a red border + warning, and
   *  hasInvalid bubbles up to the parent for Run-button gating.
   *  Undefined = no check. */
  minDim?: number;
  /** Bubbles up validation state when an upload's shortest side is below
   *  minDim. Parent uses this to disable Run. */
  onValidationChange?: (hasInvalid: boolean) => void;
};

export function ImageDropzone({
  onUploaded,
  label = "Source image",
  accept = "image/png,image/jpeg,image/webp",
  minDim,
  onValidationChange,
}: Props) {
  const [upload, setUpload] = useState<
    (UploadResponse & { width: number; height: number }) | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Bubble validation up the same way MultiImageDropzone does -- ref-wrapped
  // callback so inline arrow functions from the parent don't refire the
  // effect on every parent render.
  const onValidationChangeRef = useRef(onValidationChange);
  useEffect(() => {
    onValidationChangeRef.current = onValidationChange;
  });
  useEffect(() => {
    if (minDim == null || !upload) {
      onValidationChangeRef.current?.(false);
      return;
    }
    const shortSide = Math.min(upload.width, upload.height);
    const tooSmall = shortSide > 0 && shortSide < minDim;
    onValidationChangeRef.current?.(tooSmall);
  }, [upload, minDim]);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const file = Array.from(files)[0];
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        setError("Please choose an image file (PNG, JPEG, WebP).");
        return;
      }
      setUploading(true);
      setError(null);
      try {
        // Read dims + upload in parallel; dims are pure client-side and fast.
        const [res, dims] = await Promise.all([
          uploadFile(file),
          readImageDimensions(file).catch(() => ({ width: 0, height: 0 })),
        ]);
        const withDims = { ...res, width: dims.width, height: dims.height };
        setUpload(withDims);
        onUploaded(res);
      } catch (e) {
        setError(String(e));
        setUpload(null);
        onUploaded(null);
      } finally {
        setUploading(false);
      }
    },
    [onUploaded],
  );

  const clear = () => {
    setUpload(null);
    setError(null);
    onUploaded(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const shortSide = upload ? Math.min(upload.width, upload.height) : 0;
  const tooSmall = upload != null && minDim != null && shortSide > 0 && shortSide < minDim;
  const dimsKnown = upload != null && upload.width > 0 && upload.height > 0;

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium leading-none">{label}</p>

      {upload ? (
        <div className="space-y-2">
          <div
            className={
              tooSmall
                ? "rounded-md border-2 border-destructive p-2"
                : "rounded-md border border-border p-2"
            }
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={fileUrl(upload.url)}
              alt={upload.name}
              className="w-full max-h-72 object-contain rounded"
            />
            <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
              <span className="truncate">
                {upload.name}{" "}
                <span className="opacity-60">
                  ({dimsKnown ? `${upload.width}×${upload.height}, ` : ""}
                  {Math.round(upload.size / 1024)} KB)
                </span>
              </span>
              <Button variant="outline" size="xs" onClick={clear}>
                Remove
              </Button>
            </div>
            {tooSmall && (
              <p className="mt-1.5 text-xs text-destructive">
                Below this model&apos;s {minDim}px minimum (shortest side is{" "}
                {shortSide}px). Atlas will reject this image — replace it or
                pick a different model.
              </p>
            )}
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
            onChange={(e) =>
              e.target.files && handleFiles(e.target.files)
            }
            disabled={uploading}
          />
          {uploading ? (
            <span className="text-muted-foreground">Uploading…</span>
          ) : (
            <>
              <p className="font-medium">Drop an image here</p>
              <p className="text-muted-foreground text-xs mt-1">
                or click to browse · PNG, JPEG, WebP · max 8 MB
              </p>
            </>
          )}
        </label>
      )}

      {error && (
        <p className="text-xs text-destructive">{error}</p>
      )}
    </div>
  );
}
