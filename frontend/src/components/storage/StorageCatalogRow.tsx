"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";

interface StorageCatalogRowProps {
  href: string;
  icon: ReactNode;
  title: string;
  meta?: ReactNode;
  actions?: ReactNode;
}

export default function StorageCatalogRow({ href, icon, title, meta, actions }: StorageCatalogRowProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors group">
      <Link href={href} className="flex items-center gap-3 flex-1 min-w-0">
        <span className="shrink-0">{icon}</span>
        <span className="flex-1 min-w-0">
          <span className="block truncate font-medium text-text group-hover:text-accent transition-colors">
            {title}
          </span>
        </span>
        <ChevronRight size={16} className="text-muted shrink-0" />
      </Link>
      {meta ? <div className="hidden sm:block shrink-0 max-w-md">{meta}</div> : null}
      {actions ? <div className="shrink-0">{actions}</div> : null}
    </div>
  );
}
