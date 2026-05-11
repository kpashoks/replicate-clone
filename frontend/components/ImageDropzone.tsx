"use client";

import { useCallback, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { uploadFile, fileUrl, type UploadResponse } from "@/lib/api";

type Props = {
  /** Called once an upload completes; receives the upload id (or null on clear). */
  onUploaded: (upload: UploadResponse | null) => void;
  /** Optional label shown above the dropzone. */
  label?: string;
  /** Accept attribute for the file input. */
  accept?: string;
};

export function ImageDropzone({
  onUploaded,
  label = "Source image",
  accept = "image/png,image/jpeg,image/webp",
}: Props) {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

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
        const res = await uploadFile(file);
        setUpload(res);
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

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium leading-none">{label}</p>

      {upload ? (
        <div className="space-y-2">
          <div className="rounded-md border border-border p-2">
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
                  ({Math.round(upload.size / 1024)} KB)
                </span>
              </span>
              <Button variant="outline" size="xs" onClick={clear}>
                Remove
              </Button>
            </div>
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
