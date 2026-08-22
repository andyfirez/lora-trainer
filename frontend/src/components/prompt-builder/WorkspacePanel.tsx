"use client";

import { FolderOpen, Save } from "lucide-react";
import { useMemo, useState } from "react";
import { CREATE_OPTION } from "@/components/prompt-builder/constants";
import ComposedPromptPreview from "@/components/prompt-builder/ComposedPromptPreview";
import TemplateSectionList from "@/components/prompt-builder/TemplateSectionList";
import Button from "@/components/ui/Button";
import Input from "@/components/ui/Input";
import Modal, { ModalFooter } from "@/components/ui/Modal";
import { selectClassName } from "@/components/ui/Select";
import type {
  PromptPolarity,
  PromptSection,
  PromptSelectionMode,
  PromptTemplate,
  PromptTemplateDetail,
} from "@/types/promptBuilder";

interface WorkspacePanelProps {
  polarity: PromptPolarity;
  workspace: PromptTemplateDetail | null;
  savedTemplates: PromptTemplate[];
  sections: PromptSection[];
  onSaveTemplate: (name: string) => Promise<void>;
  onLoadTemplate: (templateId: number) => Promise<void>;
  onDeleteTemplate: (templateId: number) => Promise<void>;
  onAddSection: (sectionId: number) => Promise<void>;
  onCreateSection: (name: string) => Promise<number | null>;
  onCreateValue: (sectionId: number, text: string) => Promise<number | null>;
  onRenameSection: (sectionId: number, name: string) => Promise<void>;
  onRenameValue: (sectionId: number, valueId: number, text: string) => Promise<void>;
  onReorder: (order: number[]) => Promise<void>;
  onUpdateSection: (
    linkId: number,
    body: {
      selection_mode?: PromptSelectionMode;
      selected_value_id?: number | null;
      selected_value_ids?: number[];
      clear_selected_value?: boolean;
      include_empty?: boolean;
    },
  ) => Promise<void>;
  onRemoveSection: (linkId: number) => Promise<void>;
}

export default function WorkspacePanel({
  polarity,
  workspace,
  savedTemplates,
  sections,
  onSaveTemplate,
  onLoadTemplate,
  onDeleteTemplate,
  onAddSection,
  onCreateSection,
  onCreateValue,
  onRenameSection,
  onRenameValue,
  onReorder,
  onUpdateSection,
  onRemoveSection,
}: WorkspacePanelProps) {
  const [saveOpen, setSaveOpen] = useState(false);
  const [loadOpen, setLoadOpen] = useState(false);
  const [createSectionOpen, setCreateSectionOpen] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [newSectionName, setNewSectionName] = useState("");
  const [saving, setSaving] = useState(false);
  const [creatingSection, setCreatingSection] = useState(false);
  const [sectionPickKey, setSectionPickKey] = useState(0);

  const availableSections = useMemo(() => {
    if (!workspace) return sections;
    const linked = new Set(workspace.sections.map((section) => section.section_id));
    return sections.filter((section) => !linked.has(section.id));
  }, [sections, workspace]);

  async function handleSaveTemplate() {
    const name = saveName.trim();
    if (!name) return;
    setSaving(true);
    try {
      await onSaveTemplate(name);
      setSaveOpen(false);
      setSaveName("");
    } finally {
      setSaving(false);
    }
  }

  async function handleCreateSection() {
    const name = newSectionName.trim();
    if (!name) return;
    setCreatingSection(true);
    try {
      const sectionId = await onCreateSection(name);
      setCreateSectionOpen(false);
      setNewSectionName("");
      if (sectionId != null) {
        await onAddSection(sectionId);
      }
    } finally {
      setCreatingSection(false);
    }
  }

  function handleSectionPick(raw: string) {
    setSectionPickKey((key) => key + 1);
    if (raw === CREATE_OPTION) {
      setCreateSectionOpen(true);
      return;
    }
    if (!raw) return;
    void onAddSection(Number(raw));
  }

  return (
    <>
      <div className="rounded-xl border border-border bg-surface/30 p-4 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-text capitalize">{polarity} prompt</h2>
            <p className="text-xs text-muted mt-1">Build the current prompt and save or load presets.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" size="sm" onClick={() => setSaveOpen(true)}>
              <Save size={16} />
              Save as template
            </Button>
            <Button variant="secondary" size="sm" onClick={() => setLoadOpen(true)}>
              <FolderOpen size={16} />
              Load template
            </Button>
          </div>
        </div>

        {workspace ? (
          <div className="grid gap-4 lg:grid-cols-2 lg:items-start">
            <div className="lg:sticky lg:top-4">
              <ComposedPromptPreview label="Composed prompt" value={workspace.composed_prompt} />
            </div>

            <div className="space-y-4 min-w-0">
              <div>
                <label className="block text-xs font-medium text-text-secondary mb-1">Add section</label>
                <select
                  key={sectionPickKey}
                  className={selectClassName}
                  defaultValue=""
                  onChange={(event) => handleSectionPick(event.target.value)}
                >
                  <option value="" disabled>
                    Choose or create section…
                  </option>
                  {availableSections.map((section) => (
                    <option key={section.id} value={section.id}>
                      {section.name}
                    </option>
                  ))}
                  <option value={CREATE_OPTION}>+ Create new section…</option>
                </select>
              </div>

              <TemplateSectionList
                sections={workspace.sections}
                onReorder={(order) => void onReorder(order)}
                onUpdate={(linkId, body) => void onUpdateSection(linkId, body)}
                onRemove={(linkId) => void onRemoveSection(linkId)}
                onCreateValue={onCreateValue}
                onRenameSection={onRenameSection}
                onRenameValue={onRenameValue}
              />
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted rounded-lg border border-dashed border-border p-4">
            Loading workspace…
          </p>
        )}
      </div>

      <Modal
        open={saveOpen}
        onClose={() => {
          setSaveOpen(false);
          setSaveName("");
        }}
        title="Save as template"
        description={`Save the current ${polarity} prompt as a named preset.`}
        size="sm"
      >
        <Input
          label="Template name"
          placeholder="e.g. Portrait default"
          value={saveName}
          onChange={(event) => setSaveName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void handleSaveTemplate();
          }}
          autoFocus
        />
        <ModalFooter>
          <Button
            variant="secondary"
            onClick={() => {
              setSaveOpen(false);
              setSaveName("");
            }}
          >
            Cancel
          </Button>
          <Button onClick={() => void handleSaveTemplate()} disabled={!saveName.trim() || saving}>
            Save
          </Button>
        </ModalFooter>
      </Modal>

      <Modal
        open={loadOpen}
        onClose={() => setLoadOpen(false)}
        title="Load template"
        description={`Replace the current ${polarity} prompt with a saved preset.`}
        size="md"
      >
        {savedTemplates.length ? (
          <ul className="space-y-2 max-h-64 overflow-y-auto">
            {savedTemplates.map((template) => (
              <li
                key={template.id}
                className="flex items-center justify-between gap-3 rounded-lg border border-border px-3 py-2"
              >
                <span className="text-sm text-text truncate">{template.name}</span>
                <div className="flex gap-2 shrink-0">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      void onLoadTemplate(template.id).then(() => setLoadOpen(false));
                    }}
                  >
                    Load
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => void onDeleteTemplate(template.id)}
                  >
                    Delete
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted">No saved templates for {polarity} yet.</p>
        )}
        <ModalFooter>
          <Button variant="secondary" onClick={() => setLoadOpen(false)}>
            Close
          </Button>
        </ModalFooter>
      </Modal>

      <Modal
        open={createSectionOpen}
        onClose={() => {
          setCreateSectionOpen(false);
          setNewSectionName("");
        }}
        title="Create section"
        description="Create a new shared section and add it to the current prompt."
        size="sm"
      >
        <Input
          label="Section name"
          placeholder="e.g. Subject, Style, Quality"
          value={newSectionName}
          onChange={(event) => setNewSectionName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void handleCreateSection();
          }}
          autoFocus
        />
        <ModalFooter>
          <Button
            variant="secondary"
            onClick={() => {
              setCreateSectionOpen(false);
              setNewSectionName("");
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={() => void handleCreateSection()}
            disabled={!newSectionName.trim() || creatingSection}
          >
            Create & add
          </Button>
        </ModalFooter>
      </Modal>
    </>
  );
}
