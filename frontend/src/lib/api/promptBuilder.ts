import { api } from "@/lib/api/client";
import type {
  PromptPolarity,
  PromptSection,
  PromptSectionValue,
  PromptSelectionMode,
  PromptTemplate,
  PromptTemplateDetail,
} from "@/types/promptBuilder";

export const promptBuilderApi = {
  listSections: () => api.get<PromptSection[]>("/prompt-sections/"),
  createSection: (name: string) => api.post<PromptSection>("/prompt-sections/", { name }),
  updateSection: (sectionId: number, name: string) =>
    api.patch<PromptSection>(`/prompt-sections/${sectionId}`, { name }),
  deleteSection: (sectionId: number) => api.delete(`/prompt-sections/${sectionId}`),

  createSectionValue: (sectionId: number, text: string) =>
    api.post<PromptSectionValue>(`/prompt-sections/${sectionId}/values/`, { text }),
  updateSectionValue: (sectionId: number, valueId: number, text: string) =>
    api.patch<PromptSectionValue>(`/prompt-sections/${sectionId}/values/${valueId}`, { text }),
  deleteSectionValue: (sectionId: number, valueId: number) =>
    api.delete(`/prompt-sections/${sectionId}/values/${valueId}`),

  getWorkspace: (polarity: PromptPolarity) =>
    api.get<PromptTemplateDetail>(`/prompt-workspace/${polarity}`),

  addWorkspaceSection: (polarity: PromptPolarity, sectionId: number) =>
    api.post<PromptTemplateDetail>(`/prompt-workspace/${polarity}/sections/`, { section_id: sectionId }),

  reorderWorkspaceSections: (polarity: PromptPolarity, order: number[]) =>
    api.patch<PromptTemplateDetail>(`/prompt-workspace/${polarity}/sections/reorder`, { order }),

  updateWorkspaceSection: (
    polarity: PromptPolarity,
    linkId: number,
    body: {
      selection_mode?: PromptSelectionMode;
      selected_value_id?: number | null;
      selected_value_ids?: number[];
      clear_selected_value?: boolean;
      include_empty?: boolean;
    },
  ) => api.patch<PromptTemplateDetail>(`/prompt-workspace/${polarity}/sections/${linkId}`, body),

  removeWorkspaceSection: (polarity: PromptPolarity, linkId: number) =>
    api.delete(`/prompt-workspace/${polarity}/sections/${linkId}`),

  saveWorkspaceAsTemplate: (polarity: PromptPolarity, name: string) =>
    api.post<PromptTemplate>(`/prompt-workspace/${polarity}/save`, { name }),

  listTemplates: (polarity?: PromptPolarity) => {
    const query = polarity ? `?polarity=${encodeURIComponent(polarity)}` : "";
    return api.get<PromptTemplate[]>(`/prompt-templates/${query}`);
  },

  loadTemplate: (templateId: number) =>
    api.post<PromptTemplateDetail>(`/prompt-templates/${templateId}/load`),

  deleteTemplate: (templateId: number) => api.delete(`/prompt-templates/${templateId}`),
};
