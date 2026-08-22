"use client";

import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import SectionSelectionRow from "@/components/prompt-builder/SectionSelectionRow";
import type { PromptSelectionMode, PromptTemplateSection } from "@/types/promptBuilder";

interface TemplateSectionListProps {
  sections: PromptTemplateSection[];
  onReorder: (order: number[]) => void;
  onUpdate: (
    linkId: number,
    body: {
      selection_mode?: PromptSelectionMode;
      selected_value_id?: number | null;
      selected_value_ids?: number[];
      clear_selected_value?: boolean;
      include_empty?: boolean;
    },
  ) => void;
  onRemove: (linkId: number) => void;
  onCreateValue: (sectionId: number, text: string) => Promise<number | null>;
  onRenameSection: (sectionId: number, name: string) => Promise<void>;
  onRenameValue: (sectionId: number, valueId: number, text: string) => Promise<void>;
}

function SortableSectionRow({
  section,
  onUpdate,
  onRemove,
  onCreateValue,
  onRenameSection,
  onRenameValue,
}: {
  section: PromptTemplateSection;
  onUpdate: TemplateSectionListProps["onUpdate"];
  onRemove: TemplateSectionListProps["onRemove"];
  onCreateValue: TemplateSectionListProps["onCreateValue"];
  onRenameSection: TemplateSectionListProps["onRenameSection"];
  onRenameValue: TemplateSectionListProps["onRenameValue"];
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: section.id,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.6 : 1,
  };

  return (
    <div ref={setNodeRef} style={style}>
      <SectionSelectionRow
        section={section}
        onChange={(body) => onUpdate(section.id, body)}
        onRemove={() => onRemove(section.id)}
        onCreateValue={(text) => onCreateValue(section.section_id, text)}
        onRenameSection={(name) => onRenameSection(section.section_id, name)}
        onRenameValue={(valueId, text) => onRenameValue(section.section_id, valueId, text)}
        dragHandleProps={{ ...attributes, ...listeners }}
      />
    </div>
  );
}

export default function TemplateSectionList({
  sections,
  onReorder,
  onUpdate,
  onRemove,
  onCreateValue,
  onRenameSection,
  onRenameValue,
}: TemplateSectionListProps) {
  const sorted = sections.slice().sort((a, b) => a.sort_order - b.sort_order);
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const oldIndex = sorted.findIndex((section) => section.id === active.id);
    const newIndex = sorted.findIndex((section) => section.id === over.id);
    if (oldIndex < 0 || newIndex < 0) return;
    const next = [...sorted];
    const [moved] = next.splice(oldIndex, 1);
    next.splice(newIndex, 0, moved);
    onReorder(next.map((section) => section.id));
  }

  if (!sorted.length) {
    return (
      <p className="text-sm text-muted rounded-lg border border-dashed border-border p-4">
        No sections yet. Add one using the picker above.
      </p>
    );
  }

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
      <SortableContext items={sorted.map((section) => section.id)} strategy={verticalListSortingStrategy}>
        <div className="space-y-2">
          {sorted.map((section) => (
            <SortableSectionRow
              key={section.id}
              section={section}
              onUpdate={onUpdate}
              onRemove={onRemove}
              onCreateValue={onCreateValue}
              onRenameSection={onRenameSection}
              onRenameValue={onRenameValue}
            />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  );
}
