import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "../src/lib");
const source = readFileSync(join(root, "trainParameterMetadata.ts"), "utf8");
const arrayStart = source.indexOf("export const TRAIN_PARAMETER_METADATA");
const arrayBody = source.slice(source.indexOf("[", arrayStart), source.lastIndexOf("];") + 1);

const sections = [
  ["model", 5, 55],
  ["lora", 56, 104],
  ["trainingTargets", 105, 261],
  ["training", 262, 378],
  ["optimizer", 379, 559],
  ["data", 560, 734],
  ["optimization", 735, 778],
  ["performance", 779, 926],
  ["checkpointing", 927, 968],
  ["sampling", 969, 1167],
  ["logging", 1168, 1210],
];

const lines = arrayBody.split("\n");
const outDir = join(root, "metadata");
mkdirSync(outDir, { recursive: true });

for (const [file, start, end] of sections) {
  const chunk = lines.slice(start - 4, end - 3).join("\n").trim();
  const content = [
    'import type { ParameterMeta } from "../parameterUtils";',
    "",
    `export const ${file}ParameterMetadata: ParameterMeta[] = [`,
    chunk,
    "];",
    "",
  ].join("\n");
  writeFileSync(join(outDir, `${file}.ts`), content);
}

const imports = sections
  .map(([file]) => `import { ${file}ParameterMetadata } from "./${file}";`)
  .join("\n");
const spread = sections.map(([file]) => `  ...${file}ParameterMetadata,`).join("\n");
const index = [
  imports,
  'import { buildParameterLookup } from "../parameterUtils";',
  "",
  "export const TRAIN_PARAMETER_METADATA = [",
  spread,
  "];",
  "",
  "export const TRAIN_PARAMETER_LOOKUP = buildParameterLookup(TRAIN_PARAMETER_METADATA);",
  "",
  "export function trainHint(key: string): { hint?: string; hintAnchor?: string } {",
  "  const meta = TRAIN_PARAMETER_LOOKUP.get(key);",
  "  if (!meta || meta.showInlineHint === false) return {};",
  "  return { hint: meta.shortHint, hintAnchor: meta.key };",
  "}",
  "",
].join("\n");
writeFileSync(join(outDir, "index.ts"), index);

const shim = [
  "export {",
  "  TRAIN_PARAMETER_METADATA,",
  "  TRAIN_PARAMETER_LOOKUP,",
  "  trainHint,",
  '} from "./metadata";',
  "",
].join("\n");
writeFileSync(join(root, "trainParameterMetadata.ts"), shim);

console.log(`Split metadata into ${sections.length} section files`);
