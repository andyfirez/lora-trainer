# Kohya Comparison

Сравнение `lora-trainer` с эталонным Kohya config для Winx_Chimera_CFTS.

**Kohya config:** `D:/SD/LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/config_illustriousXL_v01_Winx_Chimera_CFTS_v1.json`

**lora-trainer config:** job id=4 snapshot / `Winx_Chimera_CFTS_config.yaml`

---

## Side-by-side

| Parameter | Kohya | lora-trainer (job_9) | Match? |
|-----------|-------|----------------------|--------|
| Base model | illustriousXL_v01.safetensors | illustriousXL_v01.safetensors | ✅ |
| SDXL | true | true (implicit) | ✅ |
| network_dim (rank) | 16 | 32 | ❌ |
| network_alpha | 16 | 16 | ✅ |
| unet_lr | 0.001 | 0.0001 | ❌ |
| lr_scheduler | cosine | cosine | ✅ |
| lr_warmup | 0 | 0 | ✅ |
| train_batch_size | 2 | 2 | ✅ |
| max_train_epochs | 40 | 20 | ❌ |
| optimizer | AdamW8bit | adamw_8bit | ✅ |
| loss_type | l2 | MSE | ✅ equivalent |
| min_snr_gamma | 0 | 0 | ✅ |
| noise_offset | 0 | 0 | ✅ |
| cache_latents | true | true | ✅ |
| cache_latents_to_disk | true | true | ✅ |
| sdxl_cache_text_encoder_outputs | true | true | ✅ |
| enable_bucket | **true** | **нет** | ❌ critical |
| bucket_reso_steps | 256 | — | ❌ |
| max_bucket_reso | 2048 | — | ❌ |
| min_bucket_reso | 512 | — | ❌ |
| max_resolution | 1024,1024 | 1024 (square only) | ❌ |
| clip_skip | **2** | **нет (always -2 layer)** | ❌ |
| max_token_length | 250 | tokenizer default (77) | ❌ |
| stop_text_encoder_training | 0 (TE frozen) | te train=false | ✅ |
| xformers | true | xformers | ✅ |
| sample size | 832×1216 | 832×1216 (sampler) | ✅ |
| sample_prompt | includes `Winx_Chimera_CFTS` trigger | sampling config id=5 без trigger | ❌ eval only |

---

## Dataset layout

| | Kohya | lora-trainer |
|---|-------|--------------|
| Path | `LoRA_training/Winx_Chimera_CFTS/dataset/v1/10_Winx_Chimera_CFTS/` | `datasets/Winx_Chimera/` |
| Repeats | 10 (folder prefix `10_`) | repeats=5 in config |
| Images | 13 | 13 |
| Captions | `.txt` sidecar | `.txt` sidecar |

---

## LoRA target scope

| | Kohya default | lora-trainer (after fix) |
|---|---------------|--------------------------|
| Attention (to_k/q/v/out) | ✅ | ✅ |
| FF (ff.net.0.proj, ff.net.2) | ✅ | ✅ (added) |
| TE layers | optional | optional (off in Winx) |

---

## Initialization

| | Kohya | lora-trainer (after fix) |
|---|-------|--------------------------|
| lora_A | Kaiming uniform | Kaiming uniform (`init_lora_weights=True`) |
| lora_B | zeros | zeros |

---

## Наиболее значимые расхождения (по impact)

1. **enable_bucket** — Kohya сохраняет aspect ratio; lora-trainer force square crop.
2. **clip_skip=2** — Kohya пропускает последний слой CLIP; lora-trainer использует `hidden_states[-2]` без configurable skip (может не совпадать semantically).
3. **Learning rate** — Kohya 1e-3 vs lora-trainer 1e-4 (но с repeats/epochs/bucketing сравнение не прямое).
4. **Rank** — Kohya 16 vs lora-trainer 32 (больше capacity → выше риск overfit на 13 images).

---

## loss_type: l2

В Kohya `loss_type: l2` — это mean squared error между prediction и target (noise или v-pred).

В lora-trainer:

```python
per_sample_loss = F.mse_loss(model_pred, target, reduction="none").mean(dim=(1,2,3))
```

**Вывод:** base loss идентичен. Расхождение не в loss type, а в preprocessing, conditioning и hyperparams.
