export type PromptPolarity = "positive" | "negative";
export type PromptSelectionMode = "fixed" | "random";

export interface PromptSectionValue {
  id: number;
  section_id: number;
  text: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface PromptSection {
  id: number;
  name: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
  values: PromptSectionValue[];
}

export interface PromptTemplate {
  id: number;
  name: string;
  polarity: PromptPolarity;
  created_at: string;
  updated_at: string;
}

export interface PromptTemplateSection {
  id: number;
  template_id: number;
  section_id: number;
  section_name: string;
  sort_order: number;
  selection_mode: PromptSelectionMode;
  selected_value_id: number | null;
  selected_value_ids: number[];
  include_empty: boolean;
  values: PromptSectionValue[];
}

export interface PromptTemplateDetail extends PromptTemplate {
  sections: PromptTemplateSection[];
  composed_prompt: string;
}
