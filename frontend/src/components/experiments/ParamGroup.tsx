"use client";

import type { ReactNode } from "react";

interface ParamGroupProps {
  title: string;
  children: ReactNode;
}

export default function ParamGroup({ title, children }: ParamGroupProps) {
  return (
    <div className="space-y-1">
      <div className="text-xs font-semibold text-muted uppercase tracking-wide pb-1">{title}</div>
      <div className="space-y-5">{children}</div>
    </div>
  );
}
