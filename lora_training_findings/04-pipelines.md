# Пайплайны: lora-trainer vs Kohya

## lora-trainer (SDXL) — июль 2026

```
Dataset (.jpg + .txt)
  → preprocess: assign_bucket → scale+crop → non-square bake → .prepared/{resolution}/
  → DB: bucket_width/height, scale_to_*, crop_x/y per image
  → latent cache: VAE encode → *_sdxl.npz (any size ÷8)
  → TE cache: CLIP-L/G penultimate → *_te.npz  (clip_skip=2, max 77 tokens)
  → train loop (trainer.py):
      UNet fp16 frozen; LoRA trainable fp32 (fix N ✅)
      GradScaler при fp16 mixed precision (fix N ✅)
      BucketBatchSampler if enable_bucket
      add_time_ids = per-image Kohya tuple (crop ≠ 0 для landscape)
      autocast(fp16) forward; MSE fp32; clip_grad_norm; AdamW8bit
  → export .safetensors
  → sampling: latent k-sample
      add_time_ids inference: median train bucket match if available (fix M ✅), else (H,W,0,0,H,W)
```

**Ключевые файлы:** `buckets.py`, `preprocess.py`, `concept_training_metadata.py`, `bucket_batch_sampler.py`, `dataset.py`, `latent_cache.py`, `te_cache.py`, `trainer.py`, `loss.py`, `latent_sampling/session.py`

**Ограничения:** reForge вне scope fix M; max_token_length=77. Fix N ✅, fix M inference ✅ (validation pending).

## Kohya sd-scripts (эталон Winx)

**Config:** `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/config_*.json`

```
Dataset (folder 10_* = 10 repeats)
  → bucketing: сохраняет AR, bucket_reso_steps=256
  → cache latents + TE (sdxl_cache_text_encoder_outputs)
  → train: MSE l2, enable_bucket, max_token_length=250
  → LoRA network weights fp32 (default; --full_fp16 опционально)
  → mixed_precision fp16 → autocast forward only
  → accelerator.backward(loss) → GradScaler при fp16
  → add_time_ids: per-image original_size + crop coords из bucket
```

## Расхождения (значимые)

| | Kohya (Winx) | lora-trainer (jul 2026 Bloom) |
|---|--------------|-------------------------------|
| enable_bucket | ✅ | ✅ |
| add_time_ids (train) | per-image Kohya | per-image Kohya ✅ |
| add_time_ids (infer) | aligned in pipeline | median train bucket (fix M) or fallback |
| effective LR | 1.0 × 1e-3 (типично) | 3e-4 constant (последний run) |
| LoRA weight dtype | fp32 | fp32 trainable ✅ (fix N) |
| backward | GradScaler | GradScaler ✅ (fix N) |
| optimizer | AdamW8bit | AdamW8bit ✅ |
| результат | likeness, без шума | ep1–3 ok; ep4+ — validate fix M |

## add_time_ids

### Baseline (job 9/13)

Фикс. `[R,R,0,0,R,R]` — неверно для landscape, но единообразно с inference `(H,W,0,0,H,W)`. Результат: мыло, без likeness.

### H/I fix fitted+crop (отвергнуто, июнь 2026)

Fitted size + `_crop_box`. Bloom + rank16/lr1e-3 → шум с ep1. **Не повторять.**

### Bucketing Kohya semantics (июль 2026)

`buckets.py`: source size + crop после scale-to-cover. Пример landscape 1216×918 → bucket 768×1024: `(918, 1216, 3, 0, 768, 1024)`.

Bloom retrain: ep1–3 ok при lr3e-4 constant; ep4+ шум. Root cause stack: **M** (infer add_time_ids). Fix N ✅. См. `09-lr-constant-post-run-jul2026.md`.

### fp16 stack N — fix ✅ (июль 2026)

LoRA fp32 + GradScaler. Частично подтверждено: убрал раннюю деградацию ep2. См. `07`, `08`.

### Train/infer add_time_ids M — fix inference ✅ (validation pending)

`resolve_reference_add_time_ids` + `SDXLSamplingSession.create(reference_add_time_ids=...)`. См. `10-fix-m-validation.md`.

### Collate bug L (fix, июль 2026)

Tuple в batch → transpose → mixed conditioning. Tensor в dataset → `(B,6)`. См. `06-add-time-ids-collate-bug.md`.

### Inference (latent sampling / reForge)

`session.py`: `(height, width, 0, 0, height, width)`. При bucketing train использует ненулевые crop — **mismatch (M, P0)**. Ep4+ шум сохраняется при lr3e-4 constant → не только eval artifact. Kohya без шума. См. `09-lr-constant-post-run-jul2026.md`.

## OneTrainer

Тот же SDXL паттерн: TE `default_layer=-2`, bucketing в data loader, penultimate для обоих encoders. Детальный diff не делался — эталон для пользователя Kohya.

## Kohya reference run (работает)

- rank=16, alpha=16, unet_lr=1e-3, 40 ep, enable_bucket
- sample_prompt с trigger `Winx_Chimera_CFTS`
- Output: `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/`

**Примечание:** Kohya fp16+AdamW8bit стабилен за счёт fp32 LoRA + GradScaler + **aligned add_time_ids**. lora-trainer: fix N ✅; **M открыта** — см. `09-lr-constant-post-run-jul2026.md`.
