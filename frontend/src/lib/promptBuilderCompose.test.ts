import assert from "node:assert/strict";
import test from "node:test";

import { composePrompt, composeSection, composeTemplatePrompt } from "./promptBuilderCompose.ts";
import type { PromptTemplateSection } from "@/types/promptBuilder";

test("composeSection random builds wildcard", () => {
  assert.equal(composeSection(["red", "blue"], "random", []), "{red|blue}");
});

test("composeSection fixed single value", () => {
  assert.equal(composeSection(["red", "blue"], "fixed", ["red"]), "red");
});

test("composeSection fixed multiple values join with comma", () => {
  assert.equal(composeSection(["red", "blue"], "fixed", ["red", "blue"]), "red, blue");
});

test("composeTemplatePrompt joins fixed multi-select and random sections", () => {
  const sections: PromptTemplateSection[] = [
    {
      id: 1,
      template_id: 1,
      section_id: 10,
      section_name: "Subject",
      sort_order: 0,
      selection_mode: "fixed",
      selected_value_id: 100,
      selected_value_ids: [100],
      include_empty: false,
      values: [{ id: 100, section_id: 10, text: "1girl", sort_order: 0, created_at: "", updated_at: "" }],
    },
    {
      id: 2,
      template_id: 1,
      section_id: 20,
      section_name: "Style",
      sort_order: 1,
      selection_mode: "fixed",
      selected_value_id: 200,
      selected_value_ids: [200, 201],
      include_empty: false,
      values: [
        { id: 200, section_id: 20, text: "anime", sort_order: 0, created_at: "", updated_at: "" },
        { id: 201, section_id: 20, text: "realistic", sort_order: 1, created_at: "", updated_at: "" },
      ],
    },
    {
      id: 3,
      template_id: 1,
      section_id: 30,
      section_name: "Lighting",
      sort_order: 2,
      selection_mode: "random",
      selected_value_id: 300,
      selected_value_ids: [300, 301],
      include_empty: false,
      values: [
        { id: 300, section_id: 30, text: "soft", sort_order: 0, created_at: "", updated_at: "" },
        { id: 301, section_id: 30, text: "dramatic", sort_order: 1, created_at: "", updated_at: "" },
      ],
    },
  ];
  assert.equal(composeTemplatePrompt(sections), "1girl, anime, realistic, {soft|dramatic}");
});

test("composeTemplatePrompt random include empty prepends empty wildcard option", () => {
  const sections: PromptTemplateSection[] = [
    {
      id: 1,
      template_id: 1,
      section_id: 30,
      section_name: "Lighting",
      sort_order: 0,
      selection_mode: "random",
      selected_value_id: 300,
      selected_value_ids: [300, 301],
      include_empty: true,
      values: [
        { id: 300, section_id: 30, text: "soft", sort_order: 0, created_at: "", updated_at: "" },
        { id: 301, section_id: 30, text: "dramatic", sort_order: 1, created_at: "", updated_at: "" },
      ],
    },
  ];
  assert.equal(composeTemplatePrompt(sections), "{|soft|dramatic}");
});

test("composeTemplatePrompt random with empty selection uses all values", () => {
  const sections: PromptTemplateSection[] = [
    {
      id: 1,
      template_id: 1,
      section_id: 30,
      section_name: "Lighting",
      sort_order: 0,
      selection_mode: "random",
      selected_value_id: null,
      selected_value_ids: [],
      include_empty: false,
      values: [
        { id: 300, section_id: 30, text: "soft", sort_order: 0, created_at: "", updated_at: "" },
        { id: 301, section_id: 30, text: "dramatic", sort_order: 1, created_at: "", updated_at: "" },
      ],
    },
  ];
  assert.equal(composeTemplatePrompt(sections), "{soft|dramatic}");
});

test("composePrompt skips empty segments", () => {
  assert.equal(composePrompt(["a", "", "b"]), "a, b");
});
