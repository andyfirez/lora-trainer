# Winx_Chimera_CFTS Case Study

Детальный разбор основного проблемного кейса.

---

## Paths

| Resource | Path |
|----------|------|
| Dataset (lora-trainer) | `D:/SD/datasets/Winx_Chimera` |
| Prepared images + cache | `D:/SD/datasets/Winx_Chimera/.prepared/1024/` |
| Kohya dataset | `D:/SD/LoRA_training/Winx_Chimera_CFTS/dataset/v1/10_Winx_Chimera_CFTS/` |
| LoRA output | `D:/SD/lora_output/Winx_Chimera_CFTS/` |
| Kohya output | `D:/SD/LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/` |
| SQLite DB | `D:/SD/lora-trainer/lora_trainer.db` |

---

## Training jobs (из SQLite `jobs`)

| Job ID | Type | Config | Date | Notes |
|--------|------|--------|------|-------|
| 1 | TRAINING | id=4 variant | 2026-06-28 | rank=16, repeats=1, **без trigger_words** |
| 3 | TRAINING | id=4 | 2026-06-28 | trigger добавлен; **TE cache создан** (13 encoded) |
| 5 | TRAINING | id=4 | 2026-06-28 | |
| 7 | TRAINING | id=4 | 2026-06-29 | gaussian init era |
| **9** | **TRAINING** | **id=4** | **2026-06-29** | **Последний полный прогон, Kohya init** |
| 10 | SAMPLING | id=5 | 2026-06-29 | epochs 1–20 |

### Config id=4 (training template)

```yaml
lora_name: Winx_Chimera_CFTS
lora_rank: 32
lora_alpha: 16
epochs: 20
batch_size: 2
learning_rate: 0.0001
lr_scheduler: cosine
repeats: 5
trigger_words: [Winx_Chimera_CFTS]
min_snr_gamma: 0
noise_offset: 0
cache_latents_to_disk: true
cache_text_encoder_outputs_to_disk: true
sampling_config_id: 5
```

### Config id=5 (sampling — проблемный для eval)

```yaml
name: Chimera
sample_prompts:
  - 1girl, blue eyes, blue hair, ...  # без Winx_Chimera_CFTS
output_dir: D:/SD/lora_output/Melanie_CFTS/samples
lora_paths: [Melanie_CFTS checkpoints]
sample_width: 832
sample_height: 1216
```

---

## job_9 timeline

```
14:04:22  Start: 20 epochs, 33 steps/epoch, 660 total
14:04:22  Latent cache: 0 encoded, 13 loaded from disk
14:04:25  TE cache: 0 encoded, 13 loaded from disk
14:08:28  epoch 1 checkpoint (avr_loss=0.5357)
14:12:43  epoch 2 checkpoint (avr_loss=0.5854)
14:16:59  epoch 3 checkpoint (avr_loss=0.5821)
...
15:29:45  epoch 20 complete (step 660/660)
```

---

## Dataset details

### Images (13)

```
00001-00000-00002-2498418305.jpg   1216×918
00002..00004                       1216×918
00005                              1136×1072
00006                              1040×1072
00007                              1216×925
00008                              1024×1072
00009                              976×992
00010                              1216×925
00011                              1152×1072
00012                              1216×1072
00013                              1216×915
```

### Caption examples

**Clean (00001):**
```
1girl, blue eyes, blue hair, long hair, lipstick, solo, makeup, eyelashes, smile, gloves, bare shoulders, dress, lips, red lips
```

**Noisy (00004):**
```
multiple girls, 2girls, dress, blue eyes, long hair, makeup, blue hair, clenched teeth, lipstick, weapon, teeth, helmet, veil, puffy sleeves, 1boy, parody, sword
```

**Training caption (с trigger):**
```
Winx_Chimera_CFTS, <raw caption tags>
```

---

## Cache files

В `.prepared/1024/` для каждого изображения:

- `{stem}.jpg` — baked 1024×1024
- `{stem}_sdxl.npz` — VAE latents
- `{stem}_te.npz` — prompt_embeds + pooled_prompt_embeds

TE npz keys: `prompt_embeds` (1, 77, 2048), `pooled_prompt_embeds` (1, 1280).

**Timestamps:** txt mtime ≈ 2026-06-28 18:10, TE npz mtime ≈ 18:58 (созданы в job_3).

---

## Crop centers (DB)

Примеры `crop_center_x` из `dataset_image_crops`:

| File | center_x | center_y |
|------|----------|----------|
| 00001 | 0.507 | 0.500 |
| 00004 | 0.446 | 0.500 |
| 00013 | 0.378 | 0.500 |

При landscape→square crop значительная часть кадра обрезается по бокам.

---

## Sampling results (job_10)

- Resolution: 832×1216
- Steps: 40, CFG: 7.5, euler_a
- LoRA: rank=32, alpha=16, te1=False, te2=False
- Checkpoints: epoch1 … epoch20

Пользователь сообщил: качество ухудшается с каждой эпохой; в reForge — аналогично.

---

## Kohya reference run

Из `config_illustriousXL_v01_Winx_Chimera_CFTS_v1.json`:

- rank=16, alpha=16
- unet_lr=0.001, 40 epochs
- enable_bucket=true
- clip_skip=2
- sample_prompt включает `Winx_Chimera_CFTS` trigger
- train_data_dir с folder `10_Winx_Chimera_CFTS` (10 repeats)

Kohya run давал **лучший результат** (эталон пользователя).

---

## Выводы по кейсу

1. **Train pipeline работает** (loss считается, checkpoint'ы сохраняются, LoRA загружается в reForge).
2. **Learned signal слабый/деградирующий** — не технический crash, а quality/correctness gap.
3. Наиболее вероятная комбинация причин:
   - square crop без bucketing на landscape frames;
   - overcooking (660 steps, repeats=5, rank=32);
   - шумные caption на 13-image dataset;
   - возможно suboptimal checkpoint selection (epoch 20 vs epoch 1–3).
