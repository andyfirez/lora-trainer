# Noor_IL — анализ референсной LoRA

Дата: 2026-07-19

Связанные документы:
- [metadata_example.json](metadata_example.json) — исходные метаданные из safetensors
- [Melanie_CFTS_problem.md](Melanie_CFTS_problem.md) — описание задачи Melanie
- [Melanie_CFTS_hypotheses_tested.md](Melanie_CFTS_hypotheses_tested.md) — проверенные гипотезы
- [Melanie_CFTS_hypotheses_to_test.md](Melanie_CFTS_hypotheses_to_test.md) — план экспериментов

Конфиги Melanie для сравнения:
- `D:\SD\lora_output\Melanie_CFTS_v2\Melanie_CFTS_v2_config.yaml`
- `D:\SD\lora_output\Melanie_CFTS_v4\Melanie_CFTS_v4_config.yaml`

---

## Контекст

Найдена LoRA с стабильным реалистичным лицом на всех генерациях. Метаданные сохранены в `metadata_example.json` (название в metadata: **Noor_IL**). Ниже — сравнение с прогонами Melanie_CFTS v2 и v4 и выводы, какие отличия с наибольшей вероятностью влияют на identity.

**Главный вывод:** разница в результате объясняется не rank/alpha (везде 16/16), а комбинацией базовой модели, обучения text encoder, минимальных капшенов, ранней остановки, loss-настроек и фактического разрешения обучения.

---

## Референсная LoRA — ключевые факты

| Параметр | Значение |
|---|---|
| Название | Noor_IL |
| База | Realism Illustrious v5.0 FP16 (Civitai modelVersionId **2091367**) |
| Hash базы | `902f26484404f0e99a5bebde299f605c1c12c3bd21706d18ad36263979336121` |
| Rank / alpha | 16 / 16 |
| Уникальных изображений | 12 |
| Repeats | 13 |
| Эпох (план) | 10 |
| **Сохранённый checkpoint** | **epoch 3** (`ss_epoch: 3`, `ss_steps: 468`) |
| Max steps (план) | 1560 |
| Batch size | 1 |
| UNet LR | 5e-4 |
| Text encoder LR | 5e-5 |
| Scheduler | cosine, warmup 0 |
| Min-SNR gamma | 5 |
| Noise offset | 0.01 |
| Multires noise | iterations 6, discount 0.3 |
| Clip skip | 1 (Kohya metadata) |
| Mixed precision | bf16 |
| Weight decay | 0.1 |
| Триггер | `<illus_girl>` |
| Капшены | `<illus_girl>, woman, adult, person` на всех 12 кадрах |
| Аугментации | flip_aug, shuffle_caption (keep_tokens=1) |
| Фактические buckets | 448×448 (11 img), 512×448 (1 img) |

Ссылка на базу: [Realism Illustrious By Stable Yogi v5.0 FP16](https://civitai.com/models/974693/realism-illustrious-by-stable-yogi?modelVersionId=2091367)

---

## Объём обучения

| | Noor_IL (epoch 3) | Melanie v2/v4 (epoch 20) |
|---|---|---|
| Уникальных изображений | 12 | 52 |
| Repeats | 13 | 3 |
| Эпох | 3 (checkpoint) / 10 (план) | 20 |
| Batch size | 1 | 2 |
| Optimizer steps | **468** | **~1560** |
| Показов одного кадра | 39 | 60 |
| Всего sample presentations | 468 | 3120 |

Опубликованный файл — **ранний checkpoint**, не итог полного обучения. Это согласуется с H-04/H-07 по Melanie: оптимум часто в ранних эпохах.

---

## Сравнение с Melanie v2 и v4

### Базовая модель

| | Noor_IL | Melanie v2 | Melanie v4 |
|---|---|---|---|
| База | Realism Illustrious v5 | illustriousXL_v01 | perfectdeliberate_v10RL |
| Стиль | Реалистичный Illustrious | Чистый 2D / anime | Фотореализм |
| Совпадение с демо | Да (та же база) | Нет | Нет |

LoRA лучше всего работает на той базе, на которой обучалась. Сравнение «интернет-генераций на Realism Illustrious» с Melanie на другой базе — не чистое сравнение LoRA.

### Text encoder

| | Noor_IL | Melanie v2/v4 |
|---|---|---|
| TE обучение | Да (LR 5e-5) | Нет (`train: false`) |
| UNet LR | 5e-4 | v2: 3e-4, v4: 1e-4 |

Обучение TE усиливает привязку триггера к identity. В текущем lora-trainer TE можно включить, но **отдельного LR для TE нет** — один `learning_rate` на все trainable params.

### Капшены и датасет

**Noor_IL:** 4 одинаковых тега на всех кадрах — identity «вынуждена» жить в LoRA, а не в текстовых атрибутах.

**Melanie:** подробные WD14-капшены (одежда, фон, поза, `blonde hair`, `mole`, шумные теги). По текущему датасету:
- `blonde hair` — ~46 файлов
- `mole` — ~14 файлов
- шумные (`blurry`, `blood` и т.п.) — ~15 файлов
- `realistic` — не найден в текущих `.txt`

Риск Melanie: атрибуты (`mole`, hair color) учатся как управляемые теги, а не как часть identity-триггера.

### Гиперпараметры обучения

| Параметр | Noor_IL | v2 | v4 |
|---|---|---|---|
| LR (UNet) | 5e-4 | 3e-4 constant | 1e-4 cosine |
| LR warmup | 0 | 0 | 200 |
| Batch | 1 | 2 | 2 |
| min_snr_gamma | **5** | **0** | **0** |
| noise_offset | **0.01** | **0** | **0** |
| multires noise | **6 / 0.3** | нет | нет |
| weight_decay | 0.1 | 0.01 | 0.01 |
| clip_skip | 1 | 2 | 2 |
| Rank / alpha | 16/16 | 16/16 | 16/16 |

### Разрешение и bucketing

| | Noor_IL | Melanie v2/v4 |
|---|---|---|
| Target resolution | 512 (metadata) | 1024 |
| min_bucket_reso | 256 | 512 |
| max_bucket_reso | 1024 | 2048 |
| bucket_no_upscale | не указано (Kohya default: upscale возможен) | true |
| Фактические buckets | 448×448, 512×448 | ~842×783 (native, no upscale) |

Noor_IL обучался на ~0.2 MP; Melanie — на более высоком native resolution. Низкое разрешение может усиливать стабильность общей формы лица за счёт меньшего акцента на мелкие детали.

### Аугментации

| | Noor_IL | lora-trainer (Melanie) |
|---|---|---|
| flip_aug | true | нет |
| shuffle_caption | true (keep_tokens=1) | нет |
| random_crop | false | center crop (preprocess) |

Flip может помочь форме лица, но **вреден для асимметричных черт** (родинки Мелани). Копировать без A/B не рекомендуется.

---

## Что одинаково (маловероятные причины разницы)

- SDXL LoRA architecture
- Rank 16, alpha 16, dropout 0
- AdamW8bit, grad accumulation 1
- Gradient checkpointing, cache latents
- Без reg images, caption dropout, face crop aug
- Epsilon prediction, max grad norm 1

**Rank 16 у Noor_IL показывает, что для стабильного лица rank может быть достаточен** при правильной комбинации остальных факторов.

---

## Приоритет отличий по влиянию на identity

1. **База Realism Illustrious v5** + генерация на той же базе
2. **Обучение text encoder** (LR 5e-5)
3. **Минимальные одинаковые капшены** без описания черт лица
4. **Ранняя остановка** (epoch 3 vs 20)
5. **min_snr_gamma=5** + noise offset + multires noise
6. **Batch 1** + высокий UNet LR 5e-4
7. **Низкое фактическое разрешение** 448–512
8. **12 curated изображений** vs 52 кадра из одного видео
9. Caption shuffle, weight decay 0.1
10. bf16 vs fp16, flip aug, clip skip — вторично или неоднозначно

---

## Ограничения референса

Метаданные не доказывают «лучшую generalization». Возможны:

1. **Совпадение базы** — демо на Realism Illustrious v5
2. **Запекание** — одинаковые капшены + ранний overfit → лицо «на всех»
3. **Узкий тест** — portrait-промпты, фиксированные seed/weight/CFG

Проверка качества identity:
- без триггера;
- другая Illustrious-база;
- профиль, full body, другая причёска/одежда;
- multi-person prompt;
- веса 0.5 / 0.8 / 1.0, разные CFG и seeds.

Если лицо меняет всех женщин без trigger — это overfit, а не семантическая привязка.

---

## Что lora-trainer не воспроизводит из Kohya-референса

| Функция Noor_IL | Статус в lora-trainer |
|---|---|
| Отдельный `text_encoder_lr` / `unet_lr` | Нет — один `learning_rate` |
| multires noise | Нет |
| flip_aug | Нет |
| shuffle_caption | Нет |
| min_snr_gamma | Есть (default 5.0, у Melanie выставлен 0) |
| noise_offset | Есть (default 0.0357, у Melanie 0) |
| Обучение TE | Есть (`text_encoder_1/2.train: true`) |

---

## Рекомендуемый порядок экспериментов для Melanie

Не копировать референс слепо — задача photo→2D, асимметричные родинки, другой датасет.

```
1. Ранние checkpoints v2 (epoch 3–8) + подбор веса/CFG
2. min_snr_gamma=5, noise_offset=0.01
3. Упрощённые капшены: trigger + woman/person, без mole/hair/шума
4. Меньше эпох (8–10), repeats 2–3
5. batch_size=1 vs 2
6. База illustriousRealism / Realism Illustrious (промежуточный realistic-2D)
7. TE с отдельным LR (требует доработки тренера)
8. Rank 32 — только после пунктов 1–6
```

Flip aug и clip_skip=1 — только через отдельный A/B.

---

## Связь с существующими гипотезами Melanie

| Находка Noor_IL | Подтверждает / уточняет |
|---|---|
| Epoch 3 лучше epoch 20 | H-04, H-07 |
| Rank 16 достаточен | Опровергает H-05 как «rank 16 недостаточен» |
| Минимальные капшены | Усиливает H-T04, H-T01 |
| TE training | Новая гипотеза → H-T09 |
| Realism Illustrious база | Уточняет H-T06, H-03 |
| min_snr + noise | Новые параметры для v2.1 конфига |
