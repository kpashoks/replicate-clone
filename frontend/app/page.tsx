"use client";

import { useEffect, useState } from "react";
import { TaskCard } from "@/components/TaskCard";
import { fetchModels } from "@/lib/api";
import { groupByTask, type ModelEntry, type Task } from "@/lib/types";

const TASK_ORDER: Task[] = ["t2i", "i2i", "t2v", "i2v", "v2v", "video-swap"];

export default function Gallery() {
  const [models, setModels] = useState<ModelEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch((e) => setError(String(e)));
  }, []);

  const grouped = models ? groupByTask(models) : null;

  return (
    <main className="container mx-auto py-12 px-4 max-w-6xl">
      <header className="mb-10">
        <h1 className="text-4xl font-bold tracking-tight">replicate-local</h1>
        <p className="text-muted-foreground mt-2">
          Personal multimodal AI playground. Heavy inference runs on RunPod or
          Atlas Cloud; orchestration runs here.
        </p>
      </header>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive mb-6">
          <p className="font-medium mb-1">Backend unreachable.</p>
          <p className="opacity-80">{error}</p>
          <p className="opacity-80 mt-2">
            Make sure the FastAPI server is running at{" "}
            <code className="font-mono">
              {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
            </code>
            . Start it with:{" "}
            <code className="font-mono">
              cd backend &amp;&amp; uvicorn main:app --port 8000
            </code>
          </p>
        </div>
      )}

      {!models && !error && (
        <p className="text-muted-foreground">Loading models…</p>
      )}

      {grouped && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {TASK_ORDER.map((task) => {
            const ms = grouped[task];
            if (ms.length === 0) return null;
            return <TaskCard key={task} task={task} models={ms} />;
          })}
        </div>
      )}
    </main>
  );
}
