"use client";

import { Suspense } from "react";
import ExperimentsNewPage from "@/components/experiments/ExperimentsNewPage";

export default function ExperimentsNewRoutePage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-muted">Loading…</div>}>
      <ExperimentsNewPage />
    </Suspense>
  );
}
