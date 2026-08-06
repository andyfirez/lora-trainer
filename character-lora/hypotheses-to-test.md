# Гипотезы на проверку

Датасет и состав изображений не меняем. Проверять по одной гипотезе. Все результаты смотреть на **`comix_v30.safetensors`**.

## 0. Текущая LoRA слаба, но нужен другой вес на Comix

Новое обучение не нужно, если чекпоинты ещё есть. Повторить сетку на Comix: seed 42, `832×1216`, эпохи 5/10/15/20, веса `0.5`, `0.7`, `0.9`, `1.0`, `1.2`.

Prompt: `Melanie_Comics_CFTS, flat colors, 1girl, portrait, looking at viewer`.

**Признак успеха:** при весе `0.9–1.2` лицо заметно ближе к Мелани, чем при `0.5–0.7`. Если нет — переходить к переобучению.

## 1. Слишком высокий LR и длинный warmup

Консервативный UNet-only запуск на том же датасете:

```yaml
base_model_name: illustriousXL_v01.safetensors
lora_rank: 16
lora_alpha: 16
lora_dropout: 0.0
unet:
  train: true
  learning_rate: 1.0e-04
text_encoder_1:
  train: false
text_encoder_2:
  train: false
epochs: 12
batch_size: 2
gradient_accumulation_steps: 1
lr_scheduler: cosine
lr_warmup_steps: 40
min_snr_gamma: 5.0
noise_offset: 0.0357
clip_skip: 2
resolution: 1024
enable_bucket: true
bucket_reso_steps: 64
min_bucket_reso: 512
max_bucket_reso: 2048
bucket_no_upscale: true
concepts:
- trigger_words:
  - Melanie_Comics_CFTS
  repeats: 5
cache_latents: true
cache_text_encoder_outputs: true
```

480 шагов. Проверка на Comix, веса `0.7`, `1.0`, `1.2`.

**Признак успеха:** лицо стабильнее на эпохах 6–12, чем у j115.

## 2. Триггер слабо переносится на Comix без обучения text encoders

От базы опыта 1 изменить только:

```yaml
lora_rank: 32
lora_alpha: 16
text_encoder_1:
  train: true
  learning_rate: 1.0e-05
text_encoder_2:
  train: true
  learning_rate: 1.0e-05
epochs: 10
cache_text_encoder_outputs: false
cache_text_encoder_outputs_to_disk: false
```

400 шагов. Проверка на Comix.

**Признак успеха:** персонаж вызывается одним триггером, меньше нужны теги волос и глаз.

## 3. Rank 16 не хватает для лица при переносе на Comix

От базы опыта 1 изменить только:

```yaml
lora_rank: 32
lora_alpha: 16
epochs: 12
```

TE оставить выключенными. Проверка на Comix.

**Признак успеха:** мелкие черты лица точнее. Если растут только артефакты стиля — rank 32 не подходит.

## 4. Нужна большая ёмкость для «усреднённого» лица из GPT-redraw

От базы опыта 1:

```yaml
lora_rank: 64
lora_alpha: 32
unet:
  learning_rate: 7.5e-05
epochs: 12
lr_warmup_steps: 40
min_snr_gamma: 5.0
noise_offset: 0.0357
```

480 шагов. Сравнить напрямую с опытом 1 на Comix.

**Признак успеха:** лицо узнаваемее без сильной деградации стиля Comix.

## 5. Подписи размывают лицо (только если можно править `.txt`)

Датасет по изображениям не трогаем. В каждый `.txt` добавить общий стиль и переменные детали кадра; не тегировать постоянные черты лица.

Пример: `western comic style, ink outlines, cel shading, halftone, portrait, looking at viewer, smile, striped shirt, simple background`.

Параметры — как в опыте 1 или 2. Проверка на Comix.

**Признак успеха:** лицо держится при смене одежды и фона в prompt.

## Рекомендуемый полный запуск

После коротких A/B — лучший вариант из опытов 1–4:

```yaml
base_model_name: illustriousXL_v01.safetensors
lora_rank: 32
lora_alpha: 16
lora_dropout: 0.0
unet:
  train: true
  learning_rate: 1.0e-04
text_encoder_1:
  train: true
  learning_rate: 1.0e-05
text_encoder_2:
  train: true
  learning_rate: 1.0e-05
epochs: 10
batch_size: 2
gradient_accumulation_steps: 1
lr_scheduler: cosine
lr_warmup_steps: 40
min_snr_gamma: 5.0
noise_offset: 0.0357
clip_skip: 2
resolution: 1024
enable_bucket: true
concepts:
- trigger_words:
  - Melanie_Comics_CFTS
  repeats: 5
cache_latents: true
cache_text_encoder_outputs: false
```

Сохранять каждую эпоху. На Comix смотреть эпохи 6, 8, 10 с весами `0.7`, `1.0`, `1.2`. Последняя эпоха не обязана быть лучшей.
