"use client";

import { useRef, useState } from "react";
import { InfoIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { uploadFile, type ParamSpec } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Sentinel value the backend recognizes: when a dynamic-form field value
 *  starts with this prefix, jobs.py resolves the upload id into an Atlas
 *  media URL at submit time. A literal URL the user pasted has no prefix
 *  and passes through untouched. */
const UPLOAD_SENTINEL = "upload://";

/**
 * Renders a form field per parameter that an Atlas model declares in its
 * OpenAPI schema. The widget type comes from Atlas's `x-ui-component` hint
 * (slider/select/textarea) combined with the JSON-Schema `type`. Values
 * live in the parent's state via the controlled-component pattern.
 *
 * The parent owns the form state and passes:
 *   - values: a Record<string, unknown> keyed by param name
 *   - onChange: callback fired with (name, newValue) on every change
 *
 * Specialized fields the dynamic form does NOT render (because the
 * specialized controls exist elsewhere on the task page):
 *   - prompt, seed (kept as the existing top-of-form / bottom-of-form
 *     controls because they have specialized UI: Enhance button, dice)
 *   - image, video, mask_image, reference_* (come from upload dropzones)
 *   - model, loras, lora_url, lora_scale (server-side / specialized)
 * Backend's atlas_schemas.py already filters these out before sending,
 * so we don't need to re-filter here -- if it shows up in `params`,
 * we render it.
 */
export function DynamicParamForm({
  params,
  values,
  onChange,
}: {
  params: ParamSpec[];
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
}) {
  if (params.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        This model has no additional parameters beyond the prompt.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {params.map((p) => {
        const current = values[p.name];
        return (
          <ParamField
            key={p.name}
            spec={p}
            value={current}
            onChange={(v) => onChange(p.name, v)}
          />
        );
      })}
    </div>
  );
}

/** Single field router: picks the widget based on (ui_component, type, enum). */
function ParamField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  // 0) secondary file-upload field (last_image, end_image, audio) -> dropzone
  if (spec.is_upload) {
    return <UploadField spec={spec} value={value} onChange={onChange} />;
  }

  // 1) enum (regardless of declared ui_component) renders as a button group
  //    for short lists or a select for long ones. Atlas's "select" hint
  //    triggers the same render path.
  if (spec.enum && spec.enum.length > 0) {
    return <EnumField spec={spec} value={value} onChange={onChange} />;
  }

  // 2) numeric slider when explicitly hinted OR when we have full bounds
  if (
    spec.ui_component === "slider" &&
    typeof spec.minimum === "number" &&
    typeof spec.maximum === "number"
  ) {
    return <SliderField spec={spec} value={value} onChange={onChange} />;
  }

  // 3) boolean -> toggle pair
  if (spec.type === "boolean") {
    return <BoolField spec={spec} value={value} onChange={onChange} />;
  }

  // 4) textarea when explicitly hinted OR for long-form free text
  if (spec.ui_component === "textarea") {
    return <TextareaField spec={spec} value={value} onChange={onChange} />;
  }

  // 5) numeric input fallback when no ui_component but type is numeric
  if (spec.type === "integer" || spec.type === "number") {
    return <NumericInputField spec={spec} value={value} onChange={onChange} />;
  }

  // 6) default: plain text input
  return <TextInputField spec={spec} value={value} onChange={onChange} />;
}

// =====================================================================
// Label + tooltip helper. Atlas's descriptions are typically 1-2 sentences
// of model-author copy. We surface them via the native `title` attribute
// on a small info icon, matching the user's "tooltip on hover/focus"
// choice. Title-attribute tooltips work on hover AND on focus (for
// keyboard users) and inherit OS accessibility behavior for free.
// =====================================================================
function FieldLabel({ spec }: { spec: ParamSpec }) {
  return (
    <div className="flex items-center gap-1.5">
      <Label className="text-sm">
        {spec.label}
        {spec.required && (
          <span className="text-destructive ml-0.5" aria-label="required">
            *
          </span>
        )}
      </Label>
      {spec.description && (
        <span
          className="cursor-help text-muted-foreground/70 hover:text-foreground"
          title={spec.description}
          aria-label={`Help: ${spec.description}`}
          tabIndex={0}
        >
          <InfoIcon className="size-3.5" />
        </span>
      )}
      {spec.default !== null && spec.default !== undefined && (
        <span className="ml-auto text-[10px] text-muted-foreground/70 font-mono">
          default: {String(spec.default)}
        </span>
      )}
    </div>
  );
}

// =====================================================================
// Widgets
// =====================================================================

function EnumField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current = value ?? spec.default;
  const choices = spec.enum || [];
  // For <= 5 choices, render as button group. Otherwise as a native select
  // to avoid a huge wall of buttons.
  if (choices.length <= 5) {
    return (
      <div className="space-y-1.5">
        <FieldLabel spec={spec} />
        <div className="flex flex-wrap gap-1.5">
          {choices.map((c) => {
            const isActive = String(c) === String(current);
            return (
              <Button
                key={String(c)}
                type="button"
                size="sm"
                variant={isActive ? "default" : "outline"}
                onClick={() => onChange(c)}
              >
                {String(c)}
              </Button>
            );
          })}
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-1.5">
      <FieldLabel spec={spec} />
      <select
        value={String(current ?? "")}
        onChange={(e) => {
          const raw = e.target.value;
          // Try to coerce back to original enum type. Atlas mixes ints + strings.
          const match = choices.find((c) => String(c) === raw);
          onChange(match ?? raw);
        }}
        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
      >
        {choices.map((c) => (
          <option key={String(c)} value={String(c)}>
            {String(c)}
          </option>
        ))}
      </select>
    </div>
  );
}

function SliderField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current =
    typeof value === "number"
      ? value
      : typeof spec.default === "number"
        ? spec.default
        : (spec.minimum ?? 0);
  const min = spec.minimum ?? 0;
  const max = spec.maximum ?? 1;
  const step = spec.step ?? (spec.type === "integer" ? 1 : 0.01);
  // Heuristic for display decimals: if step is integer or >=1, no decimals;
  // else use enough digits to capture the step precision.
  const decimals = Number.isInteger(step) ? 0 : Math.min(4, -Math.floor(Math.log10(step)));
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <FieldLabel spec={spec} />
        <span className="text-sm font-mono text-muted-foreground tabular-nums">
          {current.toFixed(decimals)}
        </span>
      </div>
      <Slider
        value={[current]}
        min={min}
        max={max}
        step={step}
        onValueChange={(raw) => {
          // base-ui Slider's onValueChange type widens to
          // `number | readonly number[]`. At runtime it's always the
          // array form when value is passed as an array, but TS can't
          // narrow. Handle both branches defensively.
          const v = Array.isArray(raw) ? raw[0] : raw;
          if (typeof v !== "number") return;
          onChange(spec.type === "integer" ? Math.round(v) : v);
        }}
      />
    </div>
  );
}

function BoolField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current =
    typeof value === "boolean"
      ? value
      : typeof spec.default === "boolean"
        ? spec.default
        : false;
  return (
    <div className="space-y-1.5">
      <FieldLabel spec={spec} />
      <div className="flex gap-2">
        <Button
          type="button"
          size="sm"
          variant={current ? "default" : "outline"}
          onClick={() => onChange(true)}
          className="flex-1"
        >
          On
        </Button>
        <Button
          type="button"
          size="sm"
          variant={!current ? "default" : "outline"}
          onClick={() => onChange(false)}
          className="flex-1"
        >
          Off
        </Button>
      </div>
    </div>
  );
}

function TextareaField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current = typeof value === "string" ? value : (spec.default as string) ?? "";
  return (
    <div className="space-y-1.5">
      <FieldLabel spec={spec} />
      <Textarea
        rows={3}
        value={current}
        onChange={(e) => onChange(e.target.value)}
        placeholder={String(spec.default ?? "")}
      />
    </div>
  );
}

function TextInputField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current = typeof value === "string" ? value : (spec.default as string) ?? "";
  return (
    <div className="space-y-1.5">
      <FieldLabel spec={spec} />
      <Input
        type="text"
        value={current}
        onChange={(e) => onChange(e.target.value)}
        placeholder={String(spec.default ?? "")}
      />
    </div>
  );
}

function NumericInputField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current =
    typeof value === "number"
      ? value
      : typeof spec.default === "number"
        ? spec.default
        : 0;
  return (
    <div className="space-y-1.5">
      <FieldLabel spec={spec} />
      <Input
        type="number"
        value={current}
        min={spec.minimum ?? undefined}
        max={spec.maximum ?? undefined}
        step={spec.step ?? (spec.type === "integer" ? 1 : "any")}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return;
          const parsed =
            spec.type === "integer" ? parseInt(raw, 10) : parseFloat(raw);
          if (!Number.isNaN(parsed)) onChange(parsed);
        }}
      />
    </div>
  );
}

/**
 * Secondary file-upload field (e.g. last_image, end_image, audio). Lets the
 * user EITHER upload a file (which we push to /api/uploads, then store as an
 * "upload://<id>" sentinel the backend resolves to an Atlas URL) OR paste a
 * URL directly (stored verbatim). Both are optional -- these fields enhance
 * the generation but aren't required.
 */
function UploadField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);

  const accept =
    spec.upload_kind === "audio"
      ? "audio/mpeg,audio/wav,audio/mp4,audio/ogg,audio/flac,audio/aac,.mp3,.wav,.m4a,.ogg,.flac,.aac"
      : spec.upload_kind === "video"
        ? "video/mp4,video/quicktime,video/webm"
        : "image/png,image/jpeg,image/webp";

  const current = typeof value === "string" ? value : "";
  const isUploaded = current.startsWith(UPLOAD_SENTINEL);
  const isUrl = current && !isUploaded;

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const res = await uploadFile(file);
      onChange(`${UPLOAD_SENTINEL}${res.id}`);
      setFilename(file.name);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setUploading(false);
    }
  };

  const clear = () => {
    onChange("");
    setFilename(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-1.5">
      <FieldLabel spec={spec} />

      {isUploaded ? (
        <div className="flex items-center justify-between rounded-md border border-green-500/40 bg-green-500/10 px-2.5 py-1.5 text-xs">
          <span className="truncate text-green-700 dark:text-green-400">
            ✓ Uploaded{filename ? `: ${filename}` : ""}
          </span>
          <Button type="button" variant="ghost" size="xs" onClick={clear}>
            Remove
          </Button>
        </div>
      ) : (
        <>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
            >
              {uploading
                ? "Uploading…"
                : `Upload ${spec.upload_kind || "file"}`}
            </Button>
            <input
              ref={inputRef}
              type="file"
              accept={accept}
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>
          {/* URL-paste fallback for users who already have a hosted URL. */}
          <Input
            type="text"
            placeholder="…or paste a URL"
            value={isUrl ? current : ""}
            onChange={(e) => onChange(e.target.value)}
          />
        </>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

// cn re-exported for callers that want it; not used internally above.
export { cn };
