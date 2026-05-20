"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { PlusIcon, XIcon } from "lucide-react";

import { Label } from "@/components/ui/label";
import {
  readImageDimensions,
  uploadFile,
  fileUrl,
  type UploadResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";

/** Upload with the image's natural pixel dimensions tracked alongside.
 *  Used for the per-tile dimension overlay and pre-flight min_image_dim
 *  validation. Dimensions are read locally from the File object in
 *  parallel with the upload — they're available the moment the tile
 *  renders, no extra round-trip. */
type UploadWithDims = UploadResponse & {
  width: number;
  height: number;
};

type Props = {
  /** Maximum number of references the selected model accepts. */
  max: number;
  /** Label above the tile grid. */
  label?: string;
  /** Accept attribute for the file input. */
  accept?: string;
  /** Emitted whenever the list of completed uploads changes (add or remove). */
  onChange: (uploads: UploadResponse[]) => void;
  /** Minimum px on the shortest side that the selected model accepts.
   *  Tiles smaller than this are marked invalid with a red border + tooltip,
   *  and hasInvalid bubbles up to the parent via onValidationChange so
   *  the Run button can be disabled. Undefined = no check. */
  minDim?: number;
  /** Called when the count of too-small uploads changes. Parent uses this
   *  to gate the Run button. */
  onValidationChange?: (hasInvalid: boolean) => void;
};

/**
 * Multi-image upload as an accumulating tile grid. Use for models that take
 * 3+ interchangeable reference images (Grok ≤ 8, GPT Image 2 Edit ≤ 10,
 * Nano Banana 2 ≤ 14, Qwen Edit Plus ≤ 3). For models with 1–2 ordered,
 * semantically-distinct slots (image-char-swap = source + character), keep
 * using the labeled <ImageDropzone>s.
 */
export function MultiImageDropzone({
  max,
  label = "Reference images",
  accept = "image/png,image/jpeg,image/webp",
  onChange,
  minDim,
  onValidationChange,
}: Props) {
  const [uploads, setUploads] = useState<UploadWithDims[]>([]);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Bubble validation state up to the parent whenever uploads change.
  // hasInvalid is true if minDim is set and any uploaded image has a
  // shortest-side dim below it. We use a ref+effect pattern (mirroring
  // the existing onChange handling above) so the parent can pass an
  // inline arrow function without triggering re-fires.
  const onValidationChangeRef = useRef(onValidationChange);
  useEffect(() => {
    onValidationChangeRef.current = onValidationChange;
  });
  useEffect(() => {
    if (minDim == null) {
      onValidationChangeRef.current?.(false);
      return;
    }
    const hasInvalid = uploads.some(
      (u) => Math.min(u.width, u.height) < minDim,
    );
    onValidationChangeRef.current?.(hasInvalid);
  }, [uploads, minDim]);

  // Mirror `uploads` up to the parent. The naive approach -- calling
  // onChange(next) inside the setUploads updater -- triggers React's
  // "Cannot update a component while rendering a different component"
  // warning under StrictMode, because state updater functions are invoked
  // twice and any setState call they make against the parent shows up as
  // happening during the child's render. Lifting the notification into an
  // effect makes it a side-effect of the commit, which is always safe.
  //
  // onChange is wrapped in a ref so an inline arrow `(ups) => setX(ups)`
  // from the parent (changing identity every render) doesn't cause the
  // effect to refire when only `uploads` is the relevant trigger.
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });
  const hasMountedRef = useRef(false);
  useEffect(() => {
    if (!hasMountedRef.current) {
      // Don't notify on the initial mount -- parent already has its own
      // initial state (e.g. [null, null]) and shouldn't be force-reset to
      // [] just because this dropzone exists.
      hasMountedRef.current = true;
      return;
    }
    onChangeRef.current(uploads);
  }, [uploads]);

  const remaining = Math.max(0, max - uploads.length - uploadingCount);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const arr = Array.from(files);
      if (!arr.length) return;

      const images = arr.filter((f) => f.type.startsWith("image/"));
      const skipped = arr.length - images.length;
      const accepted = images.slice(0, remaining);
      const overflow = images.length - accepted.length;

      const msgs: string[] = [];
      if (skipped) msgs.push(`${skipped} non-image file(s) skipped`);
      if (overflow) msgs.push(`${overflow} over the ${max}-image cap, skipped`);
      setError(msgs.join(" · ") || null);

      if (!accepted.length) return;

      setUploadingCount((n) => n + accepted.length);
      // Run upload and dimension-read in parallel per file. Failing either
      // means the file is unusable, so we treat it as a single failure.
      // readImageDimensions is purely client-side (reads the File via a
      // blob URL); no network round-trip, takes <100 ms for typical PNGs.
      const results = await Promise.allSettled(
        accepted.map(async (f) => {
          const [up, dims] = await Promise.all([
            uploadFile(f),
            readImageDimensions(f).catch(() => ({ width: 0, height: 0 })),
          ]);
          return { ...up, width: dims.width, height: dims.height } as UploadWithDims;
        }),
      );
      const successes: UploadWithDims[] = [];
      const failures: string[] = [];
      for (const r of results) {
        if (r.status === "fulfilled") successes.push(r.value);
        else failures.push(String(r.reason));
      }
      setUploadingCount((n) => n - accepted.length);
      if (failures.length) {
        setError(
          [error, `${failures.length} upload(s) failed: ${failures[0]}`]
            .filter(Boolean)
            .join(" · "),
        );
      }
      setUploads((prev) => [...prev, ...successes]);
      if (inputRef.current) inputRef.current.value = "";
    },
    // onChange intentionally omitted -- we forward it via the effect above,
    // not from inside this callback. Leaving it in the deps would force a
    // new handleFiles identity on every parent render (since parent passes
    // an inline arrow), invalidating drag handlers etc.
    [remaining, max, error],
  );

  const removeAt = (i: number) => {
    setUploads((prev) => prev.filter((_, j) => j !== i));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        <span className="text-xs text-muted-foreground">
          {uploads.length}/{max}
        </span>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (remaining > 0) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
        }}
        className={cn(
          "flex flex-wrap gap-2 rounded-md border-2 border-dashed p-3 min-h-[6.5rem] transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : uploads.length > 0
              ? "border-muted-foreground/20"
              : "border-muted-foreground/30 hover:border-muted-foreground/50",
        )}
      >
        {uploads.map((u, i) => {
          const shortSide = Math.min(u.width, u.height);
          const tooSmall = minDim != null && shortSide > 0 && shortSide < minDim;
          const dimsKnown = u.width > 0 && u.height > 0;
          return (
            <div
              key={u.id + i}
              className={cn(
                "relative size-20 overflow-hidden rounded-md border bg-muted group",
                tooSmall
                  ? "border-destructive border-2"
                  : "border-border",
              )}
              title={
                tooSmall
                  ? `${u.name} — ${u.width}×${u.height}\nBelow this model's ${minDim}px minimum (shortest side). Atlas will reject this image.`
                  : `${u.name} ${dimsKnown ? `(${u.width}×${u.height}, ${Math.round(u.size / 1024)} KB)` : `(${Math.round(u.size / 1024)} KB)`}`
              }
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={fileUrl(u.url)}
                alt={u.name}
                className="size-full object-cover"
              />
              {/* Dimension overlay along the bottom edge */}
              {dimsKnown && (
                <span
                  className={cn(
                    "absolute bottom-0 inset-x-0 px-1 py-0.5 text-[9px] font-mono leading-none text-center",
                    tooSmall
                      ? "bg-destructive/85 text-destructive-foreground"
                      : "bg-black/60 text-white",
                  )}
                >
                  {u.width}×{u.height}
                </span>
              )}
              <button
                type="button"
                onClick={() => removeAt(i)}
                className="absolute top-0.5 right-0.5 rounded-full bg-black/70 p-0.5 text-white opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
                title="Remove"
                aria-label={`Remove ${u.name}`}
              >
                <XIcon className="size-3" />
              </button>
            </div>
          );
        })}

        {Array.from({ length: uploadingCount }).map((_, i) => (
          <div
            key={`up-${i}`}
            className="size-20 rounded-md border border-border bg-muted flex items-center justify-center text-xs text-muted-foreground"
          >
            <span className="animate-pulse">…</span>
          </div>
        ))}

        {remaining > 0 && (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="size-20 rounded-md border border-dashed border-muted-foreground/40 hover:border-primary hover:bg-accent/40 flex flex-col items-center justify-center gap-1 text-xs text-muted-foreground transition-colors"
          >
            <PlusIcon className="size-5" />
            <span>Add</span>
          </button>
        )}

        {uploads.length === 0 && uploadingCount === 0 && (
          <p className="text-xs text-muted-foreground self-center pl-1">
            Drop images here, or click +. Up to {max}.
          </p>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
      </div>

      {error && <p className="text-xs text-destructive">{error}</p>}

      {/* Pre-flight validation summary: count of too-small uploads with a
          clear "what to do" message. Sits below the grid so it's visible
          before the user even clicks Run. */}
      {minDim != null &&
        (() => {
          const tooSmall = uploads.filter(
            (u) => Math.min(u.width, u.height) > 0 && Math.min(u.width, u.height) < minDim,
          );
          if (tooSmall.length === 0) return null;
          return (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
              <p className="font-medium">
                {tooSmall.length} image{tooSmall.length === 1 ? "" : "s"} below
                this model's {minDim}px minimum
              </p>
              <p className="mt-0.5 text-destructive/80">
                Remove or replace the red-bordered tile{tooSmall.length === 1 ? "" : "s"}.
                Upscale to at least {minDim} px on the shortest side, or pick a
                different model.
              </p>
            </div>
          );
        })()}
    </div>
  );
}
