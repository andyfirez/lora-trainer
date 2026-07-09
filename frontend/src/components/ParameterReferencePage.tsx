"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { TRAIN_PARAMETER_METADATA } from "@/lib/trainParameterMetadata";
import { TRAIN_SECTION_ORDER, groupBySection, parameterAnchor } from "@/lib/parameterUtils";

function ParameterEntry({ entry }: { entry: (typeof TRAIN_PARAMETER_METADATA)[number] }) {
  const anchor = parameterAnchor(entry.key);
  return (
    <article
      id={anchor}
      className="scroll-mt-24 rounded-lg border border-[var(--border)] bg-[var(--bg)] p-4 space-y-2"
    >
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-white">{entry.label}</h3>
        {entry.yamlOnly && (
          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-400">
            YAML only
          </span>
        )}
        {entry.deprecated && (
          <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-red-400">
            Deprecated
          </span>
        )}
      </div>
      <code className="block text-xs text-[var(--muted)] font-mono">{entry.key}</code>
      <p className="text-xs text-[var(--muted)] leading-relaxed">{entry.shortHint}</p>
      <p className="text-sm text-white/90 leading-relaxed">{entry.description}</p>
      <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 text-xs">
        {entry.defaultValue != null && (
          <div>
            <dt className="text-[var(--muted)]">Default</dt>
            <dd className="text-white font-mono mt-0.5">{entry.defaultValue}</dd>
          </div>
        )}
        {entry.constraints != null && (
          <div>
            <dt className="text-[var(--muted)]">Valid range</dt>
            <dd className="text-white font-mono mt-0.5">{entry.constraints}</dd>
          </div>
        )}
      </dl>
    </article>
  );
}

function SectionBlock({
  section,
  items,
  open,
  onToggle,
}: {
  section: string;
  items: (typeof TRAIN_PARAMETER_METADATA)[number][];
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <section className="bg-[var(--surface)] rounded-xl border border-[var(--border)] overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-white/5 transition-colors"
      >
        <div>
          <h2 className="text-base font-semibold text-white">{section}</h2>
          <p className="text-xs text-[var(--muted)] mt-0.5">{items.length} parameter(s)</p>
        </div>
        {open ? (
          <ChevronDown size={18} className="text-[var(--muted)] shrink-0" />
        ) : (
          <ChevronRight size={18} className="text-[var(--muted)] shrink-0" />
        )}
      </button>
      {open && (
        <div className="px-5 pb-5 space-y-3 border-t border-[var(--border)] pt-4">
          {items.map((entry) => (
            <ParameterEntry key={entry.key} entry={entry} />
          ))}
        </div>
      )}
    </section>
  );
}

function sectionForAnchor(
  sections: { section: string; items: (typeof TRAIN_PARAMETER_METADATA)[number][] }[],
  anchor: string,
): string | undefined {
  return sections.find((group) =>
    group.items.some((entry) => parameterAnchor(entry.key) === anchor),
  )?.section;
}

export default function ParameterReferencePage() {
  const sections = useMemo(
    () => groupBySection(TRAIN_PARAMETER_METADATA, TRAIN_SECTION_ORDER),
    [],
  );

  const [hash, setHash] = useState("");
  const [openSections, setOpenSections] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    sections.forEach((group, index) => {
      initial[group.section] = index < 3;
    });
    return initial;
  });

  const readHash = useCallback(() => {
    if (typeof window === "undefined") return;
    setHash(window.location.hash.slice(1));
  }, []);

  useEffect(() => {
    readHash();
    window.addEventListener("hashchange", readHash);
    return () => window.removeEventListener("hashchange", readHash);
  }, [readHash]);

  useEffect(() => {
    if (!hash) return;
    const targetSection = sectionForAnchor(sections, hash);
    if (targetSection) {
      setOpenSections((prev) => ({ ...prev, [targetSection]: true }));
    }
  }, [hash, sections]);

  useEffect(() => {
    if (!hash) return;
    const frame = requestAnimationFrame(() => {
      document.getElementById(hash)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return () => cancelAnimationFrame(frame);
  }, [hash, openSections]);

  const toggleSection = (section: string) => {
    setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Training Parameters</h1>
        <p className="text-[var(--muted)] mt-1">
          Reference for every LoRA training config field — what it does and how it affects quality,
          speed, and VRAM usage.
        </p>
      </div>
      <div className="space-y-4">
        {sections.map((group) => (
          <SectionBlock
            key={group.section}
            section={group.section}
            items={group.items}
            open={openSections[group.section] ?? false}
            onToggle={() => toggleSection(group.section)}
          />
        ))}
      </div>
    </div>
  );
}
