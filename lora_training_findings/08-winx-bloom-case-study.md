# Winx_Bloom_CFTS Case Study

Второй кейс расследования — другой персонаж, другой датасет, **тот же плохой результат**. Это ключевое доказательство **системной** проблемы пайплайна, а не артефактов маленького датасета Winx_Chimera.

---

## Paths

| Resource | Path |
|----------|------|
| Dataset | `D:/SD/datasets/Winx_Bloom` |
| Prepared + cache | `D:/SD/datasets/Winx_Bloom/.prepared/1024/` |
| LoRA output | `D:/SD/lora_output/Winx_Bloom_CFTS/` |
| Training log | `D:/SD/lora-trainer/logs/job_13.log` |
| Sampling log | `D:/SD/lora-trainer/logs/job_14.log` |
| Training config template | job_configs id=6 |
| Sampling config template | job_configs id=7 |

---

## Jobs

| Job ID | Type | Name | Status | Config |
|--------|------|------|--------|--------|
| 11 | TAGGING | Winx_Bloom auto-tag | COMPLETED | — |
| 12 | TRAINING | Chimera_CFTS | CANCELLED | id=4 (старый Chimera, не Bloom) |
| **13** | **TRAINING** | **Bloom_CFTS** | **COMPLETED** | **id=6** |
| **14** | **SAMPLING** | **Bloom_CFTS post-train sampling** | **COMPLETED** | **id=7** |

---

## Dataset

| Metric | Winx_Chimera | Winx_Bloom |
|--------|--------------|------------|
| Images | 13 | **80** |
| Captions | 13 | 80 |
| Target resolution | 1024 | 1024 |
| Non-square sources | ~13/13 | **76/80** |
| Trigger | `Winx_Chimera_CFTS` | `Winx_Bloom_CFTS` |

Типичные размеры исходников Bloom: `1216×918`, `1216×925`, `1216×1072`, `1216×936` и др.

Все bake'ятся в **1024×1024** без bucketing.

---

## Training config (job_13 / config id=6)

| Parameter | Value |
|-----------|-------|
| lora_rank | 32 |
| lora_alpha | 16 (**scale = alpha/rank = 0.5**) |
| learning_rate | 1e-4 |
| lr_scheduler | cosine |
| epochs | 20 |
| batch_size | 2 |
| repeats | **2** (мягче, чем Chimera repeats=5) |
| steps/epoch | **80** |
| total steps | **1600** |
| min_snr_gamma | 0 |
| noise_offset | 0 |
| cache | latents + TE, to disk |

### Cache (job_13)

```
Latent cache ready: 80 encoded, 0 loaded from disk
TE cache ready: 80 encoded, 0 loaded from disk
```

Кэши **свежие**, созданы в этом прогоне. Гипотеза stale cache **не применима**.

---

## Loss dynamics (job_13)

| Checkpoint | Step | avr_loss |
|------------|------|----------|
| epoch 1 | 80 | 0.5159 |
| epoch 2 | 160 | 0.5522 |
| epoch 10 | 800 | 0.4040 |
| epoch 20 | 1600 | 0.4119 |

Loss **снижается** к концу обучения (в отличие от Chimera, где avr_loss рос на эпохах 2–4). Визуально результат всё равно плохой → **низкий loss ≠ likeness**.

---

## LoRA weights ARE changing (training работает механически)

Анализ safetensors checkpoint'ов:

| Epoch | LoRA layers | alpha | up_mean (norm) | up_max |
|-------|-------------|-------|----------------|--------|
| 1 | 700 | 16 | 0.134 | 1.02 |
| 5 | 700 | 16 | 0.421 | 3.24 |
| 10 | 700 | 16 | 0.508 | 3.87 |
| 20 | 700 | 16 | 0.542 | 4.15 |

- Checkpoint'ы **не идентичны** (веса растут от epoch к epoch).
- 700 LoRA-модулей (UNet FF + attention targets).
- **LoRA обучается**, но не даёт нужного likeness → проблема в **корректности conditioning / preprocessing**, а не в «мёртвом» optimizer.

---

## Sampling (job_14)

- Resolution: **832×1216**
- Steps: 40, CFG: 7.5, euler_a
- LoRA: rank=32, alpha=16
- Samples: `D:/SD/lora_output/Winx_Bloom_CFTS/samples/Winx_Bloom_CFTS_epoch{1..20}_00.png`

### Sampling config id=7 — всё ещё некорректен для eval

```yaml
sample_prompts:
  - 1girl, solo, orange hair, ...  # Melanie, без Winx_Bloom_CFTS
output_dir: D:/SD/lora_output/Melanie_CFTS/samples
```

Встроенный sampler **не использует trigger Bloom**. Пользователь подтвердил, что в reForge результат тот же → eval config не root cause, но UI eval остаётся сломанным.

---

## Сравнение с Winx_Chimera

| Фактор | Chimera | Bloom | Вывод |
|--------|---------|-------|-------|
| Размер датасета | 13 | 80 | Bloom исключает «только маленький датасет» |
| Stale cache | loaded from disk | fresh encode | Bloom исключает stale cache |
| Repeats | 5 | 2 | Bloom мягче, результат тот же |
| Loss trend | рост avr_loss ep.2–4 | снижение к ep.20 | Разная динамика, одинаковый плохой результат |
| LoRA weights | растут | растут | Обучение идёт в обоих случаях |
| Square crop | да | да (76/80 non-square) | **Общий systemic factor** |
| add_time_ids train | 1024² | 1024² | **Общий systemic factor** |
| add_time_ids inference | 832×1216 | 832×1216 | **Train/inference mismatch** |
| lora_alpha/rank | 0.5 | 0.5 | **Слабее Kohya (1.0)** |

---

## Вывод по кейсу Bloom

1. Проблема **воспроизводится на другом датасете** → не специфична Winx_Chimera.
2. LoRA **реально обучается** (растут веса, падает loss) → не «optimizer/gradients dead».
3. Likeness **не появляется** → train signal не соответствует inference conditioning.
4. Наиболее вероятный systemic root cause: **фиксированные SDXL micro-conditioning (add_time_ids) + square crop без bucketing** при inference в portrait AR.
