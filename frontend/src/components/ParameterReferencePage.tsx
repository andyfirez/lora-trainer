"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { TRAIN_PARAMETER_METADATA } from "@/lib/trainParameterMetadata";
import { TRAIN_TAB_GROUPS, groupByTab, parameterAnchor } from "@/lib/parameterUtils";
import type { ParameterMeta } from "@/lib/parameterUtils";

function ParameterBadges({ entry }: { entry: ParameterMeta }) {
  return (
    <>
      {entry.yamlOnly && (
        <span className="rounded-full bg-warning-muted px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-warning">
          YAML only
        </span>
      )}
      {entry.deprecated && (
        <span className="rounded-full bg-error-muted px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-error">
          Deprecated
        </span>
      )}
    </>
  );
}

function ParameterCard({ entry, index }: { entry: ParameterMeta; index: number }) {
  const anchor = parameterAnchor(entry.key);
  const recommended = entry.recommendedValue ?? entry.defaultValue;

  return (
    <article
      id={anchor}
      className={`scroll-mt-24 rounded-lg border border-border px-4 py-3 space-y-2 ${
        index % 2 === 0 ? "bg-surface" : "bg-white/[0.02]"
      }`}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <h4 className="font-medium text-text">{entry.label}</h4>
        <ParameterBadges entry={entry} />
      </div>

      <p className="text-sm text-text-secondary leading-snug">{entry.description}</p>

      {recommended && (
        <p className="text-sm">
          <span className="text-text">Recommended: </span>
          <span className="font-mono text-accent">{recommended}</span>
        </p>
      )}

      {entry.valueOptions && entry.valueOptions.length > 0 && (
        <ul className="text-sm text-text-secondary space-y-1 list-disc list-inside">
          {entry.valueOptions.map((option) => (
            <li key={option.value}>
              <span className="font-mono text-text">{option.value}</span>
              {" — "}
              {option.description}
            </li>
          ))}
        </ul>
      )}

      {entry.rangeGuidance && entry.rangeGuidance.length > 0 && (
        <ul className="text-sm text-text-secondary space-y-1 list-disc list-inside">
          {entry.rangeGuidance.map((band) => (
            <li key={band.range}>
              <span className="font-mono text-text">{band.range}</span>
              {" — "}
              {band.description}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

function anchorForHash(
  tabGroups: ReturnType<typeof groupByTab>,
  anchor: string,
): { tab: string; key: string } | undefined {
  for (const group of tabGroups) {
    for (const section of group.sections) {
      const match = section.items.find((entry) => parameterAnchor(entry.key) === anchor);
      if (match) return { tab: group.tab, key: match.key };
    }
  }
  return undefined;
}

export default function ParameterReferencePage() {
  const tabGroups = useMemo(() => groupByTab(TRAIN_PARAMETER_METADATA), []);

  const [activeTab, setActiveTab] = useState<string>(TRAIN_TAB_GROUPS[0].tab);
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
    const target = anchorForHash(tabGroups, hash);
    if (target) {
      setActiveTab(target.tab);
    }
  }, [hash, tabGroups]);

  useEffect(() => {
    if (!hash) return;
    const frame = requestAnimationFrame(() => {
      document.getElementById(hash)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return () => cancelAnimationFrame(frame);
  }, [hash, activeTab]);

  const activeGroup = tabGroups.find((group) => group.tab === activeTab);

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-text font-display">Training Parameters</h1>
        <p className="text-muted mt-1">
          Reference for every LoRA training config field — what it does and how it affects quality,
          speed, and VRAM usage.
        </p>
      </div>

      <div className="flex flex-wrap gap-1 border-b border-border">
        {tabGroups.map((group) => (
          <button
            key={group.tab}
            type="button"
            onClick={() => setActiveTab(group.tab)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              activeTab === group.tab
                ? "text-text border border-b-bg border-border bg-bg -mb-px"
                : "text-muted hover:text-text"
            }`}
          >
            {group.label}
          </button>
        ))}
      </div>

      {activeGroup && (
        <div className="space-y-8">
          {activeGroup.sections.map((section) => (
            <section key={section.section} className="space-y-3">
              <h3 className="text-lg font-semibold text-text font-display">{section.section}</h3>
              <div className="space-y-3">
                {section.items.map((entry, index) => (
                  <ParameterCard key={entry.key} entry={entry} index={index} />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
