"use client";

import { useCallback, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import PromptBuilderActions from "@/components/prompt-builder/PromptBuilderActions";
import WorkspacePanel from "@/components/prompt-builder/WorkspacePanel";
import PageHeader from "@/components/ui/PageHeader";
import { cn } from "@/lib/cn";
import { promptBuilderApi } from "@/lib/api/promptBuilder";
import type { PromptPolarity, PromptSelectionMode } from "@/types/promptBuilder";

const POLARITIES: PromptPolarity[] = ["positive", "negative"];

export default function PromptBuilderPage() {
  const { mutate: globalMutate } = useSWRConfig();
  const [polarity, setPolarity] = useState<PromptPolarity>("positive");

  const { data: sections = [], mutate: mutateSections } = useSWR(
    "/prompt-sections",
    () => promptBuilderApi.listSections(),
  );

  const { data: positiveWorkspace, mutate: mutatePositiveWorkspace } = useSWR(
    "/prompt-workspace/positive",
    () => promptBuilderApi.getWorkspace("positive"),
  );
  const { data: negativeWorkspace, mutate: mutateNegativeWorkspace } = useSWR(
    "/prompt-workspace/negative",
    () => promptBuilderApi.getWorkspace("negative"),
  );

  const { data: savedTemplates = [], mutate: mutateSavedTemplates } = useSWR(
    `/prompt-templates?polarity=${polarity}`,
    () => promptBuilderApi.listTemplates(polarity),
  );

  const workspace = polarity === "positive" ? positiveWorkspace : negativeWorkspace;
  const mutateWorkspace = polarity === "positive" ? mutatePositiveWorkspace : mutateNegativeWorkspace;

  const refreshWorkspaces = useCallback(async () => {
    await Promise.all([
      globalMutate("/prompt-workspace/positive"),
      globalMutate("/prompt-workspace/negative"),
    ]);
  }, [globalMutate]);

  async function createSection(name: string): Promise<number | null> {
    const created = await promptBuilderApi.createSection(name);
    await mutateSections();
    return created.id;
  }

  async function createValue(sectionId: number, text: string): Promise<number | null> {
    const created = await promptBuilderApi.createSectionValue(sectionId, text);
    await mutateSections();
    await refreshWorkspaces();
    return created.id;
  }

  async function renameSection(sectionId: number, name: string) {
    await promptBuilderApi.updateSection(sectionId, name);
    await mutateSections();
    await refreshWorkspaces();
  }

  async function renameValue(sectionId: number, valueId: number, text: string) {
    await promptBuilderApi.updateSectionValue(sectionId, valueId, text);
    await mutateSections();
    await refreshWorkspaces();
  }

  async function saveTemplate(name: string) {
    await promptBuilderApi.saveWorkspaceAsTemplate(polarity, name);
    await mutateSavedTemplates();
  }

  async function loadTemplate(templateId: number) {
    await promptBuilderApi.loadTemplate(templateId);
    await mutateWorkspace();
    await refreshWorkspaces();
  }

  async function deleteTemplate(templateId: number) {
    await promptBuilderApi.deleteTemplate(templateId);
    await mutateSavedTemplates();
  }

  async function addWorkspaceSection(sectionId: number) {
    await promptBuilderApi.addWorkspaceSection(polarity, sectionId);
    await mutateWorkspace();
    await refreshWorkspaces();
  }

  async function reorderWorkspaceSections(order: number[]) {
    await promptBuilderApi.reorderWorkspaceSections(polarity, order);
    await mutateWorkspace();
    await refreshWorkspaces();
  }

  async function updateWorkspaceSection(
    linkId: number,
    body: {
      selection_mode?: PromptSelectionMode;
      selected_value_id?: number | null;
      selected_value_ids?: number[];
      clear_selected_value?: boolean;
      include_empty?: boolean;
    },
  ) {
    await promptBuilderApi.updateWorkspaceSection(polarity, linkId, body);
    await mutateWorkspace();
    await refreshWorkspaces();
  }

  async function removeWorkspaceSection(linkId: number) {
    await promptBuilderApi.removeWorkspaceSection(polarity, linkId);
    await mutateWorkspace();
    await refreshWorkspaces();
  }

  const composedPositive = positiveWorkspace?.composed_prompt ?? "";
  const composedNegative = negativeWorkspace?.composed_prompt ?? "";

  return (
    <div className="space-y-6 p-4 md:p-6 max-w-6xl mx-auto">
      <PageHeader
        title="Prompt Builder"
        description="Build wildcard-compatible prompts from reusable sections and send them to Sampling."
        actions={
          <PromptBuilderActions
            positivePrompt={composedPositive}
            negativePrompt={composedNegative}
          />
        }
      />

      <div className="flex gap-2">
        {POLARITIES.map((kind) => (
          <button
            key={kind}
            type="button"
            onClick={() => setPolarity(kind)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-sm font-medium capitalize border transition-colors",
              polarity === kind
                ? "border-accent bg-accent/15 text-text"
                : "border-border text-muted hover:text-text",
            )}
          >
            {kind}
          </button>
        ))}
      </div>

      <WorkspacePanel
        polarity={polarity}
        workspace={workspace ?? null}
        savedTemplates={savedTemplates}
        sections={sections}
        onSaveTemplate={saveTemplate}
        onLoadTemplate={loadTemplate}
        onDeleteTemplate={deleteTemplate}
        onAddSection={addWorkspaceSection}
        onCreateSection={createSection}
        onCreateValue={createValue}
        onRenameSection={renameSection}
        onRenameValue={renameValue}
        onReorder={reorderWorkspaceSections}
        onUpdateSection={updateWorkspaceSection}
        onRemoveSection={removeWorkspaceSection}
      />
    </div>
  );
}
