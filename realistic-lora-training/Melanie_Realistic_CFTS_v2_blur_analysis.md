# Melanie_Realistic_CFTS_v2 — анализ «мыльных» результатов

Дата: 2026-07-20

Связанные документы:
- [Описание задачи](Melanie_CFTS_problem.md)
- [Анализ Noor_IL](Noor_IL_reference_analysis.md)
- [Проверенные гипотезы](Melanie_CFTS_hypotheses_tested.md)
- [Гипотезы на проверку](Melanie_CFTS_hypotheses_to_test.md)
- [Captioning guide](../docs/lora-captioning-guide.md)

---

## Итог

Прогон `Melanie_Realistic_CFTS_v2` технически завершился корректно, но его текущие samples недостаточны для вывода, что LoRA обучена неудачно:

- все checkpoints проверены одним seed и одним коротким промптом;
- генерация выполнялась с CFG 7.5 и пустым negative prompt;
- VAE декодировался в FP16 с tiling;
- отсутствует контрольное изображение базы без LoRA;
- sampling-параметры не совпадают с рекомендуемыми для Realism Illustrious v5.

Визуально epoch 1 уже сильно сглажен; epoch 5–10 немного резче, но кожа остаётся восковой, а мелкие черты лица теряются. Простая замена epoch 10 на epoch 3 не выглядит достаточным исправлением.

Сначала нужен контролируемый inference A/B-тест. Переобучение рекомендуется только если база без LoRA резкая, а подключение LoRA стабильно возвращает «мыло».

---

## Проверенный прогон

Конфиг:
`D:\SD\lora_output\Melanie_Realistic_CFTS_v2\Melanie_Realistic_CFTS_v2_config.yaml`

Результаты:
`D:\SD\lora_output\Melanie_Realistic_CFTS_v2\samples\`

| Параметр | Значение |
|---|---|
| База | `realismIllustriousBy_v50FP16.safetensors` |
| Rank / alpha | 16 / 16 |
| UNet | train, LR `1e-4` |
| TE1 / TE2 | train, LR `2e-5` каждый |
| Эпохи | 10 |
| Batch / repeats | 1 / 2 |
| Шагов | 1040 |
| Scheduler | cosine, warmup 0 |
| Min-SNR / noise offset | 5 / 0.01 |
| Precision | FP16 |
| Sampling | 832×1216, Euler a, 40 steps, CFG 7.5 |
| Sampling VAE | FP16, tiling включён |
| Negative prompt | пустой |

Обучение было остановлено на step 381 и затем корректно продолжено с полным state: optimizer, scheduler, epoch и global step восстановлены. Resume не считается причиной дефекта.

---

## Что воспроизведено из Noor_IL

- та же базовая модель Realism Illustrious v5;
- rank/alpha 16/16;
- batch size 1;
- обучение text encoder;
- Min-SNR gamma 5;
- noise offset 0.01;
- cosine scheduler без warmup;
- уменьшенный относительно старых Melanie-прогонов объём обучения.

## Что отличается от Noor_IL

| Параметр | Noor_IL | Melanie Realistic v2 |
|---|---:|---:|
| Уникальные изображения | 12 curated | 52 похожих кадра из видео |
| Опубликованный checkpoint | epoch 3, 468 steps | epoch 1–10, до 1040 steps |
| UNet LR | `5e-4` | `1e-4` |
| TE LR | `5e-5` | `2e-5` |
| Precision | BF16 | FP16 |
| Weight decay | 0.1 | 0.01 |
| Captions | одинаковые минимальные | `1girl` + переменные `shirt`/`mole` |
| Multires noise | 6 / 0.3 | нет поддержки |
| Flip / caption shuffle | включены | нет поддержки |
| Фактическое разрешение | 448–512 | примерно 704–896 × 768 |

Это не точное воспроизведение Noor_IL. Одновременно изменены база, captions, text encoder, loss-настройки и sampling, поэтому вклад каждого фактора пока не изолирован.

---

## Наиболее вероятные причины

### 1. Sampling-конфиг — высокая вероятность

Фактический промпт:

```text
Melanie_Realistic_CFTS, 1girl, solo, ponytail, suite, upper body
```

Проблемы:

- `suite` — опечатка; для костюма нужен `suit`;
- нет `looking at viewer`, `sharp focus`, `detailed eyes/skin`;
- пустой negative prompt;
- CFG 7.5 выше рекомендованного автором модели CFG 5;
- samples модели на Civitai часто используют quality embeddings, negative embeddings и ADetailer, которых в тесте нет.

Автор Realism Illustrious v5 рекомендует 27+ steps, CFG 5, clip skip 2 и DPM++ 2M SDE / Euler A. Разрешение 896×1152 указано как рекомендуемое, но модель заявлена для обычных SDXL-разрешений.

### 2. FP16 VAE и tiling — требуется A/B

Новый sampler использует `sample_vae_fp32: false` и `sample_vae_tiling: true`.

Старый Kohya-прогон на той же базе и размере 832×1216 создавал визуально более резкие samples при `sdxl_no_half_vae: true`. Сравнение не чистое, но делает FP32 VAE первым кандидатом для проверки.

Tiling обычно сильнее влияет на швы и локальные переходы, чем на общую резкость, поэтому его нельзя считать доказанной причиной без A/B.

### 3. Обучение обоих text encoder — средняя вероятность

В LoRA находятся UNet, TE1 и TE2 weights. Уже epoch 1 заметно смягчён, а дальнейшие эпохи не восстанавливают микродетали. Возможен слишком сильный сдвиг embedding триггера или обучение кинематографического сглаживания из однородного датасета.

### 4. Датасет и минимальные captions — средняя вероятность

52 кадра взяты из одного видео, имеют близкие ракурсы, свет и одежду. Упрощённые captions помогают запечь identity, но одновременно могут запечь:

- киношное сглаживание и компрессию;
- костюм и фон;
- ограниченный набор ракурсов;
- усреднённое лицо вместо устойчивых мелких черт.

### 5. Несовпадение aspect ratio — низкая/средняя вероятность

Training buckets в основном имеют высоту 768, а samples — 832×1216. Это может ухудшить композицию и generalization, но само по себе не объясняет глобальное «мыло»: базовая модель поддерживает портретные SDXL-разрешения и рекомендует 896×1152.

---

## Шаг 1 — обязательный A/B без переобучения

Создать отдельный sampling config `Melanie Realistic diagnostic`:

```yaml
base_model_name: D:/SD/SD_models/SD/realismIllustriousBy_v50FP16.safetensors
output_dir: output
sample_prompts:
  - realistic portrait photo of a woman, blonde ponytail, looking at viewer, sharp focus, detailed natural skin texture, detailed eyes, 85mm lens
sample_negative_prompt: blurry, soft focus, out of focus, plastic skin, waxy skin, airbrushed skin, low detail, low quality
sample_steps: 30
sample_cfg_scale: 5.0
sample_width: 896
sample_height: 1152
sample_scheduler: dpm++
sample_vae_tiling: false
sample_vae_fp32: true
sample_offload_unet_before_decode: true
attention_mechanism: xformers
mixed_precision: float16
vae_dtype: auto
tf32: false
```

Ограничение: `dpm++` в lora-trainer — ближайший доступный вариант DPM++, но не точное воспроизведение DPM++ 2M SDE Exponential из A1111.

### Матрица проверки

Использовать одинаковый seed во всех вариантах:

1. база без LoRA;
2. epoch 1, 3, 5 и 10;
3. LoRA weight 0.5 / 0.7 / 1.0;
4. дополнительный контроль: Euler A при тех же остальных настройках;
5. дополнительный контроль VAE: только один вариант с FP16 + tiling, чтобы подтвердить или исключить decoder.

Критерий:

- база тоже мыльная → сначала исправить sampling pipeline;
- база резкая, LoRA 0.5 резкая, 1.0 мыльная → подобрать вес/epoch, retrain не нужен;
- база резкая, любая эпоха и вес портят резкость → переходить к следующему прогону.

---

## Шаг 2 — рекомендуемый следующий training config

Цель следующего прогона: убрать влияние text encoder, сократить обучение и сохранить остальные важные параметры v2. Это практический improvement-run, а не строгий однофакторный эксперимент.

Перед запуском привести все `.txt` к одинаковому содержимому:

```text
woman, adult, person
```

Триггер `Melanie_Realistic_CFTS` trainer добавляет автоматически. Не добавлять `mole`, цвет волос, одежду, `blurry`, `realistic` и другие identity/noise-теги. Цена такого эксперимента — возможная привязка одежды и фона; её нужно проверять отдельными промптами.

```yaml
base_model_name: D:/SD/SD_models/SD/realismIllustriousBy_v50FP16.safetensors
output_dir: output
lora_name: Melanie_Realistic_CFTS_v3_unet
output_format: safetensors

lora_rank: 16
lora_alpha: 16.0
lora_dropout: 0.0

unet:
  train: true
  weight_dtype: float16
  learning_rate: 0.0001
text_encoder_1:
  train: false
  weight_dtype: float16
  learning_rate: 0.00002
text_encoder_2:
  train: false
  weight_dtype: float16
  learning_rate: 0.00002

epochs: 5
batch_size: 1
gradient_accumulation_steps: 1
lr_scheduler: cosine
lr_warmup_steps: 0

optimizer:
  type: adamw_8bit
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  relative_step: false
  scale_parameter: false
  warmup_init: false
  decouple: true
  use_bias_correction: true
  safeguard_warmup: true
  d0: 0.00001
  d_coef: 1.0

min_snr_gamma: 5.0
noise_offset: 0.01
clip_skip: 2

resolution: 1024
enable_bucket: true
bucket_reso_steps: 64
min_bucket_reso: 512
max_bucket_reso: 2048
bucket_no_upscale: true

concepts:
  - dataset_id: 5
    caption_extension: .txt
    trigger_words:
      - Melanie_Realistic_CFTS
    caption_suffix: ""
    repeats: 2

gradient_checkpointing: true
mixed_precision: float16
cache_latents: true
cache_latents_to_disk: true
cache_text_encoder_outputs: true
cache_text_encoder_outputs_to_disk: true
attention_mechanism: xformers
vae_dtype: auto
tf32: false
num_dataloader_workers: 0
dataloader_pin_memory: true
torch_compile: false

checkpointing_enabled: true
save_every_n_epochs: 1
sampling_enabled: true
sampling_config_id: REPLACE_WITH_DIAGNOSTIC_CONFIG_ID
sample_every_n_epochs: 1
sample_before_training: true

logging:
  use_ui_logger: true
  log_every: 1
```

Ожидаемый объём: 52 images × repeats 2 × 5 epochs = 520 optimizer steps.

### Почему не rank 32

Rank 16 уже достаточен для Noor_IL. Сначала нужно проверить sampling, VAE, captions и text encoder. Переход на rank 32 одновременно увеличит capacity и затруднит диагностику.

### Почему не UNet LR 5e-4

Это значение Noor_IL, но новый прогон должен по возможности изолировать влияние TE и captions относительно v2. После получения резкого результата можно отдельным коротким A/B сравнить `1e-4` и `3e-4/5e-4`.

---

## Шаг 3 — только если v3 остаётся мыльным

При резкой базе и мыльном UNet-only v3:

1. отобрать 12–20 наиболее резких и разнообразных кадров вместо всех 52;
2. исключить почти дублирующиеся video frames;
3. добавить профиль, 3/4, другой свет и одежду;
4. сравнить UNet LR `1e-4` против `3e-4`;
5. только затем проверить rank 32 / alpha 16;
6. для целевой 2D-задачи отдельно вернуться к `illustriousXL_v01` или `illustriousRealism`.

Не увеличивать число эпох, пока мыло присутствует уже на раннем checkpoint: текущие данные не указывают на недообучение.
