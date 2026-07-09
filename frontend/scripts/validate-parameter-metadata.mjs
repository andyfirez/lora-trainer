/**
 * Validates trainParameterMetadata.ts structure and coverage.
 * Run: node scripts/validate-parameter-metadata.mjs
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const __dirname = dirname(fileURLToPath(import.meta.url));
const metadataPath = join(__dirname, "../src/lib/trainParameterMetadata.ts");
const source = readFileSync(metadataPath, "utf8");

// Extract keys from the metadata array (simple regex parse — no TS compile needed)
const keyMatches = [...source.matchAll(/key:\s*"([^"]+)"/g)].map((m) => m[1]);
const yamlOnlyCount = (source.match(/yamlOnly:\s*true/g) ?? []).length;

assert.ok(keyMatches.length >= 50, `Expected at least 50 parameters, found ${keyMatches.length}`);

const uniqueKeys = new Set(keyMatches);
assert.equal(uniqueKeys.size, keyMatches.length, "Duplicate parameter keys found");

const requiredFormKeys = [
  "base_model_name",
  "output_dir",
  "lora_name",
  "output_format",
  "lora_rank",
  "lora_alpha",
  "lora_dropout",
  "unet.train",
  "unet.weight_dtype",
  "text_encoder_1.train",
  "text_encoder_1.weight_dtype",
  "text_encoder_2.train",
  "text_encoder_2.weight_dtype",
  "clip_skip",
  "epochs",
  "batch_size",
  "gradient_accumulation_steps",
  "learning_rate",
  "lr_scheduler",
  "lr_warmup_steps",
  "min_snr_gamma",
  "noise_offset",
  "optimizer.type",
  "resolution",
  "enable_bucket",
  "mixed_precision",
  "seed",
  "gradient_checkpointing",
  "cache_latents",
  "cache_latents_to_disk",
  "cache_text_encoder_outputs",
  "cache_text_encoder_outputs_to_disk",
  "attention_mechanism",
  "tf32",
  "torch_compile",
  "num_dataloader_workers",
  "dataloader_pin_memory",
  "checkpointing_enabled",
  "save_every_n_epochs",
  "sampling_enabled",
  "sampling_config_id",
];

for (const key of requiredFormKeys) {
  assert.ok(uniqueKeys.has(key), `Missing metadata for form field: ${key}`);
}

const requiredYamlOnlyKeys = [
  "bucket_reso_steps",
  "min_bucket_reso",
  "max_bucket_reso",
  "bucket_no_upscale",
  "vae_dtype",
  "resume_from_checkpoint",
  "logging.use_ui_logger",
  "sample_prompts",
  "sample_negative_prompt",
  "sample_steps",
  "sample_cfg_scale",
  "sample_width",
  "sample_height",
  "sample_scheduler",
  "sample_vae_tiling",
  "sample_vae_fp32",
  "sample_offload_unet_before_decode",
  "post_training_sampling_config_id",
  "sample_after_training",
  "concepts.image_dir",
  "concepts.prepared_dir",
];

for (const key of requiredYamlOnlyKeys) {
  assert.ok(uniqueKeys.has(key), `Missing YAML-only metadata: ${key}`);
}

assert.ok(yamlOnlyCount >= 8, `Expected at least 8 yamlOnly entries, found ${yamlOnlyCount}`);

// Anchor slug uniqueness
function parameterAnchor(key) {
  return key.replace(/\./g, "-").replace(/_/g, "-");
}
const anchors = keyMatches.map(parameterAnchor);
const uniqueAnchors = new Set(anchors);
assert.equal(uniqueAnchors.size, anchors.length, "Duplicate parameter anchors found");

console.log(`OK: ${keyMatches.length} parameters validated (${yamlOnlyCount} YAML-only)`);
