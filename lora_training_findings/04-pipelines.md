# Пайплайны: lora-trainer vs Kohya

## lora-trainer (SDXL)

```
Dataset (.jpg + .txt)
  → preprocess: fit short side → center crop → square bake → .prepared/{resolution}/
  → latent cache: VAE encode → *_sdxl.npz
  → TE cache: CLIP-L/G penultimate → *_te.npz  (clip_skip=2 ≡ hidden_states[-2], max 77 tokens)
  → train loop (trainer.py):
      latents + noisy_latents
      add_time_ids = per-image [H,W,crop_top,crop_left,R,R]   # H/I fix (fitted+crop); ⚠️ регрессия → revert
      # baseline job 9/13: фикс. [R,R,0,0,R,R]
      prompt_embeds из cache или live encode
      UNet forward + MSE loss (optional min_snr, noise_offset)
      LoRA on UNet (attention + FF), TE frozen
      effective step ≈ (alpha/rank) × lr × grad
  → export .safetensors (Kohya-compatible metadata)
  → sampling: latent k-sample + VAE decode (832×1216 default in configs)
```

**Ключевые файлы:** `preprocess.py`, `latent_cache.py`, `te_cache.py`, `trainer.py`, `loss.py`, `latent_sampling/session.py`

**Ограничения:** нет bucketing; H/I fix (fitted+crop) даёт шум — revert перед P0; sampler/reForge всё ещё `(H,W,0,0,H,W)` на inference; TE cache valid по mtime image (не caption).

## Kohya sd-scripts (эталон Winx)

**Config:** `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/config_*.json`

```
Dataset (folder 10_* = 10 repeats)
  → bucketing: сохраняет AR, bucket_reso_steps=256
  → cache latents + TE (sdxl_cache_text_encoder_outputs)
  → train: MSE l2, enable_bucket, max_token_length=250
  → add_time_ids: per-image original_size + crop coords из bucket
  → clip_skip в UI — для SDXL train игнорируется (penultimate)
```

## Расхождения (значимые)

| | Kohya (Winx) | lora-trainer (job_9/13) |
|---|--------------|-------------------------|
| enable_bucket | ✅ | ❌ |
| add_time_ids (train) | per-image Kohya (source + virtual crop) | fitted+crop fix → **шум**; baseline `[R,R,0,0,R,R]` |
| effective LR `(α/rank)×lr` | 1.0 × 1e-3 = **1e-3** | 0.5 × 1e-4 = **5e-5** ❌ |
| max_token_length | 250 | 77 ❌ |
| rank / alpha | 16 / 16 (scale 1.0) | 32 / 16 (scale 0.5) ❌ |
| unet_lr | 1e-3 | 1e-4 ❌ |
| mixed_precision | bf16 (typical SDXL) | fp16 ❌ |
| optimizer | AdamW fp32 | AdamW8bit ❌ |
| epochs | 40 | 20 |
| repeats | 10 (folder) | 5 / 2 |
| loss | l2 | MSE ✅ |
| min_snr / noise_offset | 0 / 0 | 0 / 0 ✅ |
| LoRA targets | attn + FF | attn + FF ✅ |
| LoRA init | Kaiming A, zero B | ✅ |
| TE encoding SDXL | penultimate | penultimate ✅ |
| sample AR | 832×1216 | 832×1216 ✅ |

## add_time_ids

### Baseline (job 9/13)

Фикс. `[R,R,0,0,R,R]` — неверно для landscape, но единообразно с inference `(H,W,0,0,H,W)`. Результат: мыло, без likeness.

### H/I fix (отвергнуто, июнь 2026)

Реализация: fitted size + `_crop_box` offset per-image (`compute_add_time_ids_for_bake`). Пример landscape 1216×918 @ 1024: `(1024, 1356, 0, 166, 1024, 1024)`.

Bloom retrain с rank=16, alpha=16, lr=1e-3 → **шум** (хуже baseline). Причины: семантика ≠ Kohya `get_crop_ltrb`; train/infer mismatch по crop coords.

### Kohya (эталон)

Landscape source ~1216×918, bucket 1024×1024: `original_size` = source HW, crop = virtual `get_crop_ltrb`, не pixel offset после fit-short-side.

Hypothesis A (inference resolution) — отвергнута. H/I (training conditioning, fitted+crop) — отвергнута.

## OneTrainer

Тот же SDXL паттерн: TE `default_layer=-2`, bucketing в data loader, penultimate для обоих encoders. Детальный diff не делался — эталон для пользователя Kohya.

## Kohya reference run (работает)

- rank=16, alpha=16, unet_lr=1e-3, 40 ep, enable_bucket
- sample_prompt с trigger `Winx_Chimera_CFTS`
- Output: `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/`
