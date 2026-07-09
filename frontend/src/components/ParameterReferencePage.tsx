"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { TRAIN_PARAMETER_METADATA } from "@/lib/trainParameterMetadata";
import { TRAIN_SECTION_ORDER, groupBySection, parameterAnchor } from "@/lib/parameterUtils";
import type { ParameterMeta } from "@/lib/parameterUtils";

function ParameterBadges({ entry }: { entry: ParameterMeta }) {
  return (
    <>
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
    </>
  );
}

function ParameterTableRow({ entry }: { entry: ParameterMeta }) {
  const anchor = parameterAnchor(entry.key);

  return (
    <tr id={anchor} className="scroll-mt-24 even:bg-white/[0.02] border-b border-[var(--border)] last:border-b-0">
      <td className="px-4 py-2.5 align-top">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium text-white">{entry.label}</span>
          <ParameterBadges entry={entry} />
        </div>
      </td>
      <td className="px-4 py-2.5 align-top">
        <code className="font-mono text-xs text-[var(--muted)] whitespace-nowrap">{entry.key}</code>
      </td>
      <td className="px-4 py-2.5 align-top">
        <span className="font-mono text-xs text-white/80 tabular-nums whitespace-nowrap">
          {entry.defaultValue ?? "—"}
        </span>
      </td>
      <td className="px-4 py-2.5 align-top">
        <span className="font-mono text-xs text-[var(--muted)] whitespace-nowrap">
          {entry.constraints ?? "—"}
        </span>
      </td>
      <td className="px-4 py-2.5 align-top min-w-[16rem]">
        <p className="text-sm text-white/70 leading-snug">{entry.description}</p>
      </td>
    </tr>
  );
}

function ParameterTable({ items }: { items: ParameterMeta[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--surface)]">
      <table className="w-full min-w-[48rem] text-left border-collapse">
        <thead className="sticky top-0 z-10 bg-[var(--surface)] border-b border-[var(--border)]">
          <tr>
            <th className="px-4 py-2.5 text-xs font-medium text-[var(--muted)] uppercase tracking-wide">
              Label
            </th>
            <th className="px-4 py-2.5 text-xs font-medium text-[var(--muted)] uppercase tracking-wide">
              Key
            </th>
            <th className="px-4 py-2.5 text-xs font-medium text-[var(--muted)] uppercase tracking-wide">
              Default
            </th>
            <th className="px-4 py-2.5 text-xs font-medium text-[var(--muted)] uppercase tracking-wide">
              Range
            </th>
            <th className="px-4 py-2.5 text-xs font-medium text-[var(--muted)] uppercase tracking-wide">
              Description
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((entry) => (
            <ParameterTableRow key={entry.key} entry={entry} />
          ))}
        </tbody>
      </table>
    </div>
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

  const [activeTab, setActiveTab] = useState<string>(TRAIN_SECTION_ORDER[0]);
  const [hash, setHash] = useState("");

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
      setActiveTab(target.section);
    }
  }, [hash, sections]);

  useEffect(() => {
    if (!hash) return;
    const frame = requestAnimationFrame(() => {
      document.getElementById(hash)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return () => cancelAnimationFrame(frame);
  }, [hash, activeTab]);

  const activeSection = sections.find((group) => group.section === activeTab);

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Training Parameters</h1>
        <p className="text-[var(--muted)] mt-1">
          Reference for every LoRA training config field — what it does and how it affects quality,
          speed, and VRAM usage.
        </p>
      </div>

      <div className="flex gap-1 border-b border-[var(--border)] overflow-x-auto">
        {sections.map((group) => (
          <button
            key={group.section}
            type="button"
            onClick={() => setActiveTab(group.section)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap shrink-0 ${
              activeTab === group.section
                ? "text-white border border-b-[var(--bg)] border-[var(--border)] bg-[var(--bg)] -mb-px"
                : "text-[var(--muted)] hover:text-white"
            }`}
          >
            {group.section}
          </button>
        ))}
      </div>

      {activeSection && (
        <div>
          <p className="text-xs text-[var(--muted)] mb-3">
            {activeSection.items.length} parameter(s)
          </p>
          <ParameterTable items={activeSection.items} />
        </div>
      )}
    </div>
  );
}
