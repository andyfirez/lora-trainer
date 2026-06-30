# Problem and Symptoms

## Исходная проблема

Пользователь обучает character LoRA через `lora-trainer` и получает результат хуже, чем через Kohya_ss или OneTrainer.

**Обновление 2026-06-30:** добавлен второй кейс `Winx_Bloom_CFTS` (80 images) — **тот же плохий результат**. Проблема **не специфична одному датасету**.

## Кейсы

| LoRA | Dataset | Images | Job | Output |
|------|---------|--------|-----|--------|
| Winx_Chimera_CFTS | Winx_Chimera | 13 | job_9 | `D:/SD/lora_output/Winx_Chimera_CFTS/` |
| Winx_Bloom_CFTS | Winx_Bloom | 80 | job_13 | `D:/SD/lora_output/Winx_Bloom_CFTS/` |

## Наблюдаемое поведение

### Визуальные симптомы

- **«Мыло»** — изображение теряет резкость и детализацию по мере роста числа эпох.
- **Нет likeness** — сгенерированный персонаж не похож на референс из датасета.
- **Нет прогресса** — с каждой эпохой качество не улучшается, а ухудшается.

### Где воспроизводится

| Среда | Результат |
|-------|-----------|
| Встроенный sampler `lora-trainer` | Плохо |
| stable-diffusion-webui-reForge | **То же самое** (оба персонажа) |

Тестирование в reForge исключает гипотезу «проблема только в eval prompt / sampling config UI».

## Контекст датасета Winx_Chimera

- **Путь в lora-trainer DB:** `D:/SD/datasets/Winx_Chimera`
- **Изображений:** 13
- **Target resolution:** 1024
- **Prepared dir:** `D:/SD/datasets/Winx_Chimera/.prepared/1024/`
- **Trigger word:** `Winx_Chimera_CFTS`

### Размеры исходных изображений (не квадратные)

| Файл | Размер |
|------|--------|
| Большинство | 1216×918, 1216×925, 1216×915, 1216×1072 |
| Другие | 1136×1072, 1040×1072, 1024×1072, 976×992, 1152×1072 |

Все изображения **принудительно кропятся/бake'ятся в 1024×1024** (`src/services/datasets/preprocess.py`).

### Caption

- 13 уникальных `.txt` файлов (без дубликатов).
- Часть caption содержит **шумные/противоречивые теги**: `multiple girls`, `2girls`, `1boy`, `helmet`, `sword`, `parody` и т.п.
- Для character LoRA на 13 изображениях это может размывать identity-сигнал.

## Конфигурация Winx_Bloom (job_13)

| Параметр | Значение |
|----------|----------|
| epochs | 20 |
| batch_size | 2 |
| steps/epoch | 80 |
| total steps | 1600 |
| learning_rate | 1e-4 |
| lora_rank / alpha | 32 / 16 (scale 0.5) |
| repeats | 2 |
| min_snr_gamma / noise_offset | 0 / 0 |
| cache | 80 encoded fresh (latents + TE) |

### Loss (job_13)

| Epoch | avr_loss |
|-------|----------|
| 1 | 0.5159 |
| 10 | 0.4040 |
| 20 | 0.4119 |

Loss **снижается**, likeness **не появляется** → метрика loss не отражает качество.

---

## Конфигурация Winx_Chimera (job_9)

Из `logs/job_9.log` и snapshot конфига job id=4:

| Параметр | Значение |
|----------|----------|
| epochs | 20 |
| batch_size | 2 |
| steps/epoch | 33 |
| total steps | 660 |
| learning_rate | 1e-4 |
| lr_scheduler | cosine |
| lora_rank | 32 |
| lora_alpha | 16 |
| repeats | 5 |
| min_snr_gamma | 0 |
| noise_offset | 0 |
| text_encoder train | false |
| cache_latents | true (disk) |
| cache_text_encoder_outputs | true (disk) |

## Динамика loss (job_9)

| Checkpoint | Step | avr_loss (на конец эпохи) |
|------------|------|---------------------------|
| epoch 1 | 33 | 0.5357 |
| epoch 2 | 66 | 0.5854 |
| epoch 3 | 99 | 0.5821 |
| epoch 4 | 132 | 0.6517 |
| epoch 5 | 165 | 0.5597 |
| epoch 20 | 660 | 0.3835 (скользящее среднее; per-step loss нестабилен) |

**Паттерн:** после ранних шагов (~epoch 1) `avr_loss` не стабильно падает, а колеблется с тенденцией к росту на эпохах 2–4. Это согласуется с визуальной деградацией (overcooking), а не с «недообучением» в классическом смысле.
