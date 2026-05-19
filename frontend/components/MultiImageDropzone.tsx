"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { PlusIcon, XIcon } from "lucide-react";

import { Label } from "@/components/ui/label";
import { uploadFile, fileUrl, type UploadResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

type Props = {
  /** Maximum number of references the selected model accepts. */
  max: number;
  /** Label above the tile grid. */
  label?: string;
  /** Accept attribute for the file input. */
  accept?: string;
  /** Emitted whenever the list of completed uploads changes (add or remove). */
  onChange: (uploads: UploadResponse[]) => void;
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
}: Props) {
  const [uploads, setUploads] = useState<UploadResponse[]>([]);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

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
      const results = await Promise.allSettled(
        accepted.map((f) => uploadFile(f)),
      );
      const successes: UploadResponse[] = [];
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
        {uploads.map((u, i) => (
          <div
            key={u.id + i}
            className="relative size-20 overflow-hidden rounded-md border border-border bg-muted group"
            title={`${u.name} (${Math.round(u.size / 1024)} KB)`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={fileUrl(u.url)}
              alt={u.name}
              className="size-full object-cover"
            />
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
        ))}

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
    </div>
  );
}
