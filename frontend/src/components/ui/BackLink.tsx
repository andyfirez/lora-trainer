import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/cn";

export interface BackLinkProps {
  href: string;
  "aria-label": string;
  className?: string;
  iconSize?: number;
}

export default function BackLink({ href, "aria-label": ariaLabel, className, iconSize = 16 }: BackLinkProps) {
  return (
    <Link
      href={href}
      aria-label={ariaLabel}
      className={cn(
        "p-2 rounded-lg border border-border text-muted hover:text-text hover:bg-white/5",
        className,
      )}
    >
      <ArrowLeft size={iconSize} />
    </Link>
  );
}
