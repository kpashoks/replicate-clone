"use client";

import { diffWords } from "diff";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { enhancePrompt } from "@/lib/api";

type Props = {
  open: boolean;
  onClose: () => void;
  originalPrompt: string;
  targetModel: string;
  onAccept: (enhanced: string) => void;
};

export function EnhanceDiff({
  open,
  onClose,
  originalPrompt,
  targetModel,
  onAccept,
}: Props) {
  const [loading, setLoading] = useState(false);
  const [enhanced, setEnhanced] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Trigger the enhance API call whenever the dialog is opened.
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setEnhanced(null);

    let cancelled = false;
    enhancePrompt(originalPrompt, targetModel)
      .then((res) => {
        if (cancelled) return;
        setEnhanced(res.enhanced);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e));
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, originalPrompt, targetModel]);

  const diffParts = enhanced ? diffWords(originalPrompt, enhanced) : [];

  return (
    <Dialog
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen) onClose();
      }}
    >
      <DialogContent className="!max-w-2xl">
        <DialogHeader>
          <DialogTitle>Enhanced prompt</DialogTitle>
          <DialogDescription>
            Qwen3-4B rewrote your prompt. Green = added, red = removed.
          </DialogDescription>
        </DialogHeader>

        {loading && (
          <div className="py-12 text-center text-sm text-muted-foreground">
            Calling local Qwen3-4B… first call after server start takes ~10–15 s.
          </div>
        )}

        {error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {enhanced && !loading && (
          <div className="space-y-4">
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Original
              </p>
              <p className="text-sm leading-relaxed rounded-md border border-border p-3 bg-muted/30">
                {originalPrompt}
              </p>
            </div>

            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Enhanced (word-level diff)
              </p>
              <p className="text-sm leading-relaxed rounded-md border border-border p-3 bg-muted/30">
                {diffParts.map((part, i) => {
                  if (part.added) {
                    return (
                      <span
                        key={i}
                        className="bg-green-500/20 text-green-700 dark:text-green-300 rounded px-0.5"
                      >
                        {part.value}
                      </span>
                    );
                  }
                  if (part.removed) {
                    return (
                      <span
                        key={i}
                        className="bg-red-500/20 text-red-700 dark:text-red-300 line-through rounded px-0.5"
                      >
                        {part.value}
                      </span>
                    );
                  }
                  return <span key={i}>{part.value}</span>;
                })}
              </p>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={loading}>
            Discard
          </Button>
          <Button
            onClick={() => {
              if (enhanced) {
                onAccept(enhanced);
                onClose();
              }
            }}
            disabled={loading || !enhanced}
          >
            Accept
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
