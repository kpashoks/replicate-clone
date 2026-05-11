import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { ModelEntry } from "@/lib/types";

export function ModelCard({ model }: { model: ModelEntry }) {
  const disabled = !model.available;
  const kindBadge = model.output_kind === "video" ? "Video" : "Image";

  return (
    <Card className={disabled ? "opacity-60" : ""}>
      <CardHeader>
        <div className="flex items-center justify-between gap-2 mb-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">
            {kindBadge}
          </span>
          {disabled && (
            <span className="text-xs rounded-full bg-muted px-2 py-0.5 text-muted-foreground">
              Stage {model.stage}
            </span>
          )}
        </div>
        <CardTitle>{model.label}</CardTitle>
        <CardDescription>{model.description}</CardDescription>
      </CardHeader>
      <CardFooter>
        {disabled ? (
          <Button disabled variant="outline" className="w-full">
            Coming soon
          </Button>
        ) : (
          <Link
            href={`/models/${model.slug}`}
            className={buttonVariants({ className: "w-full" })}
          >
            Run
          </Link>
        )}
      </CardFooter>
    </Card>
  );
}
