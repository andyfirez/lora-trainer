"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { TRAIN_PARAMETER_METADATA } from "@/lib/trainParameterMetadata";
import { TRAIN_SECTION_ORDER, groupBySection, parameterAnchor } from "@/lib/parameterUtils";
import type { ParameterMeta } from "@/lib/parameterUtils";

function ParameterRow({
  entry,
  expanded,
  onToggle,
}: {
  entry: ParameterMeta;
  expanded: boolean;
  onToggle: () => void;
}) {
  const anchor = parameterAnchor(entry.key);

  return (
    <div id={anchor} className="scroll-mt-24 border-b border-[var(--border)] last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/5 transition-colors"
      >
        <div className="flex-1 min-w-0 flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="text-sm font-medium text-white shrink-0">{entry.label}</span>
          {entry.yamlOnly && (
            <span className="rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-400">
              YAML only
            </span>
          )}
          {entry.deprecated && (
            <span className="rounded-full bg-red-500/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-red-400">
              Deprecated
            </span>
          )}
          <code className="text-xs text-[var(--muted)] font-mono truncate">{entry.key}</code>
        </div>
        <div className="hidden sm:flex items-center gap-4 shrink-0 text-xs text-[var(--muted)]">
          {entry.defaultValue != null && (
            <span className="font-mono text-white/80 max-w-[8rem] truncate" title={entry.defaultValue}>
              {entry.defaultValue}
            </span>
          )}
          {entry.constraints != null && (
            <span className="font-mono max-w-[10rem] truncate" title={entry.constraints}>
              {entry.constraints}
            </span>
          )}
        </div>
        {expanded ? (
          <ChevronDown size={14} className="text-[var(--muted)] shrink-0" />
        ) : (
          <ChevronRight size={14} className="text-[var(--muted)] shrink-0" />
        )}
      </button>
      {expanded && (
        <div className="px-4 pb-3 pt-0">
          <p className="text-sm text-white/90 leading-relaxed">{entry.description}</p>
          <dl className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-xs sm:hidden">
            {entry.defaultValue != null && (
              <div>
                <dt className="text-[var(--muted)] inline">Default: </dt>
                <dd className="text-white font-mono inline">{entry.defaultValue}</dd>
              </div>
            )}
            {entry.constraints != null && (
              <div>
                <dt className="text-[var(--muted)] inline">Range: </dt>
                <dd className="text-white font-mono inline">{entry.constraints}</dd>
              </div>
            )}
          </dl>
        </div>
      )}
    </div>
  );
}

function SectionBlock({
  section,
  items,
  open,
  onToggle,
  expandedRows,
  onToggleRow,
}: {
  section: string;
  items: ParameterMeta[];
  open: boolean;
  onToggle: () => void;
  expandedRows: Record<string, boolean>;
  onToggleRow: (key: string) => void;
}) {
  return (
    <section id={`section-${parameterAnchor(section)}`} className="bg-[var(--surface)] rounded-xl border border-[var(--border)] overflow-hidden scroll-mt-24">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-white/5 transition-colors"
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
        <div className="border-t border-[var(--border)]">
          {items.map((entry) => (
            <ParameterRow
              key={entry.key}
              entry={entry}
              expanded={expandedRows[entry.key] ?? false}
              onToggle={() => onToggleRow(entry.key)}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function sectionForAnchor(
  sections: { section: string; items: ParameterMeta[] }[],
  anchor: string,
): { section: string; key: string } | undefined {
  for (const group of sections) {
    const match = group.items.find((entry) => parameterAnchor(entry.key) === anchor);
    if (match) return { section: group.section, key: match.key };
  }
  return undefined;
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
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});

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
    const target = sectionForAnchor(sections, hash);
    if (target) {
      setOpenSections((prev) => ({ ...prev, [target.section]: true }));
      setExpandedRows((prev) => ({ ...prev, [target.key]: true }));
    }
  }, [hash, sections]);

  useEffect(() => {
    if (!hash) return;
    const frame = requestAnimationFrame(() => {
      document.getElementById(hash)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return () => cancelAnimationFrame(frame);
  }, [hash, openSections, expandedRows]);

  const toggleSection = (section: string) => {
    setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const toggleRow = (key: string) => {
    setExpandedRows((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const scrollToSection = (section: string) => {
    const sectionId = `section-${parameterAnchor(section)}`;
    setOpenSections((prev) => ({ ...prev, [section]: true }));
    requestAnimationFrame(() => {
      document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
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
      <nav className="flex flex-wrap gap-x-1 gap-y-1 text-xs text-[var(--muted)]">
        {sections.map((group, index) => (
          <span key={group.section} className="inline-flex items-center">
            {index > 0 && <span className="mx-1">·</span>}
            <button
              type="button"
              onClick={() => scrollToSection(group.section)}
              className="hover:text-white transition-colors"
            >
              {group.section}
            </button>
          </span>
        ))}
      </nav>
      <div className="space-y-3">
        {sections.map((group) => (
          <SectionBlock
            key={group.section}
            section={group.section}
            items={group.items}
            open={openSections[group.section] ?? false}
            onToggle={() => toggleSection(group.section)}
            expandedRows={expandedRows}
            onToggleRow={toggleRow}
          />
        ))}
      </div>
    </div>
  );
}
