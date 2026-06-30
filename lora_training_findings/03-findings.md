# Technical Findings

## 0. Cross-dataset reproduction (Winx_Bloom, job_13) — ключевое обновление

**Дата:** 2026-06-30

| | Winx_Chimera (job_9) | Winx_Bloom (job_13) |
|---|---------------------|---------------------|
| Images | 13 | **80** |
| repeats | 5 | 2 |
| Cache | loaded from disk | **80 encoded fresh** |
| Loss trend | avr_loss ↑ ep.2–4 | avr_loss ↓ к ep.20 |
| LoRA weights | растут | **растут** (up_mean 0.13→0.54) |
| Visual result | плохо | **тот же плохой результат** |

**Вывод:** проблема **не** объясняется только маленьким датасетом, stale cache или «LoRA не учится». Обучение идёт, но likeness не формируется → **systemic conditioning/preprocessing bug**.

---

## 1. add_time_ids: train/inference mismatch (suspect → hypothesis A отвергнута)

**Файлы:** `src/trainer/sdxl/trainer.py`, `src/trainer/sdxl/latent_sampling/session.py`

При **обучении** micro-conditioning всегда квадрат 1024:

```python
# trainer.py
add_time_ids = [1024, 1024, 0, 0, 1024, 1024]
```

При **inference** (sampler, reForge-style 832×1216):

```python
# latent_sampling/session.py → _build_add_time_ids(height=1216, width=832)
add_time_ids = [1216, 832, 0, 0, 1216, 832]
```

UNet SDXL использует `time_ids` как часть conditioning. LoRA учится под одни micro-cond signals, а применяется при других → **character signal не переносится в inference**.

**Impact (теоретический):** affects все прогоны с portrait inference.

**Hypothesis A test (2026-06-30):** epoch5, seed=42, trigger prompt — inference 1024×1024 (train-matched add_time_ids) и 832×1216. **Оба без likeness, оба плохого качества** → гипотеза **отвергнута** как root cause. Mismatch AR не объясняет провал.

---

## 2. Отсутствие bucketing / multi-resolution training

**Файлы:** `src/services/datasets/preprocess.py`, `src/trainer/sdxl/trainer.py`

`lora-trainer` всегда готовит изображения как квадрат `resolution × resolution`:

```python
# preprocess.py
def prepared_dir_path(image_dir, resolution):
    return Path(image_dir) / ".prepared" / str(resolution)
```

В `trainer.py` SDXL `add_time_ids` всегда формируются из квадрата:

```python
add_time_ids = self._get_add_time_ids(
    original_size=(config.resolution, config.resolution),
    crops_coords_top_left=(0, 0),
    target_size=(config.resolution, config.resolution),
    ...
)
```

**Kohya** для того же датасета использует `enable_bucket: true`, `bucket_reso_steps: 256`, `max_resolution: 1024,1024`.

**Impact:** Bloom — **76/80** non-square исходников. Принудительный square crop искажает composition; при inference portrait AR (832×1216) train/inference gap максимален.

---

## 3. lora_alpha / rank = 0.5 (слабее Kohya)

Оба проблемных прогона: `lora_rank=32`, `lora_alpha=16` → PEFT scale **0.5**.

Kohya Winx config: `network_dim=16`, `network_alpha=16` → scale **1.0**.

LoRA weights растут (Bloom epoch20 up_max≈4.15), но effective contribution при inference ослаблена. Может давать «эффекта нет», но **не объясняет один** cross-dataset failure — скорее усиливает.

---

## 4. Агрессивные гиперпараметры (Chimera-specific, недостаточно для Bloom)

| Параметр | lora-trainer (job_9) | Kohya (Winx config) |
|----------|----------------------|---------------------|
| learning_rate | 1e-4 | 1e-3 (unet_lr) — но с bucketing и 40 epochs в другом режиме |
| repeats | 5 | через `10_Winx_Chimera_CFTS` folder naming (=10 в Kohya dataset) |
| epochs | 20 | 40 |
| rank | 32 | 16 |
| batch_size | 2 | 2 |

При 13 изображениях × repeats=5 × batch=2 → **33 steps/epoch**, **660 total steps**. Это много повторений одних и тех же latents для character LoRA.

**Impact:** для Chimera — overcooking вероятен. Bloom (80 img, repeats=2) с **тем же плохим результатом** → hyperparams alone **не root cause**.

---

## 5. Деградация loss (Chimera-only pattern)

По `logs/job_9.log`:

- Epoch 1: avr_loss ≈ 0.54 (минимум среди первых эпох).
- Epoch 2–4: avr_loss растёт до 0.58–0.65.
- Per-step loss сильно скачет (0.12–1.23).

**Impact:** для Chimera — смотреть ранние epoch. Bloom: loss **падает**, результат всё равно плохой → loss curve **не predictor** likeness.

---

## 6. LoRA обучается, checkpoint'ы не «пустые» (Bloom)

Анализ `Winx_Bloom_CFTS_epoch*.safetensors`:

- 700 LoRA layers, alpha=16
- up_mean norm: 0.134 (ep1) → 0.542 (ep20)
- Checkpoint'ы **различаются** по содержимому

Гипотеза «LoRA weights не обновляются / export сломан» — **отвергнута**.

---

## 7. Узкий LoRA scope (исправлено)

**Было:** только attention layers (`to_k`, `to_q`, `to_v`, `to_out.0`).

**Стало:** добавлены feed-forward (`ff.net.0.proj`, `ff.net.2`).

**Файл:** `src/trainer/sdxl/lora_targets.py`

Kohya по умолчанию тренирует более широкий набор слоёв UNet.

---

## 8. LoRA initialization (исправлено)

**Было:** PEFT default (`init_lora_weights="gaussian"` или False → gaussian).

**Стало:** `init_lora_weights=True` → Kaiming-uniform для `lora_A`, zeros для `lora_B` (Kohya-compatible).

**Файлы:** `src/trainer/sdxl/lora_peft.py`, `tests/trainer/test_lora_peft.py`

**Impact:** изменение dynamics обучения, особенно на малых датасетах. После внедрения пользователь перезапустил тренинг — **качество не улучшилось**, значит init не был единственной причиной.

---

## 9. Rank mismatch при sampling (исправлено)

**Симптом:** `The size of tensor a (32) must match the size of tensor b (16)` при inference.

**Причина:** `SamplingConfig` использовал defaults (`lora_rank=32`), а checkpoint был rank=16.

**Fix:** `infer_kohya_lora_metadata()` читает rank/alpha/TE flags из `.safetensors` и пересобирает PEFT stack.

**Файлы:** `src/trainer/sdxl/lora_export.py`, `src/sampler/sdxl/service.py`

---

## 10. Min-SNR и noise_offset (реализовано, но часто отключено)

**Файлы:** `src/trainer/sdxl/loss.py`, `src/trainer/config.py`

- `min_snr_weight()` — Min-SNR loss weighting.
- `apply_noise_offset()` — noise offset перед `add_noise`.

Defaults в config: `min_snr_gamma=5.0`, `noise_offset=0.0357`.

В проблемных прогонах Winx (job_9): **`min_snr_gamma=0`, `noise_offset=0`**.

**loss_type l2 в Kohya** эквивалентен MSE в lora-trainer — это **не** источник расхождения.

---

## 11. TE / latent disk cache — design limitation

**Файлы:** `src/trainer/sdxl/te_cache.py`, `src/trainer/sdxl/latent_cache.py`

Валидность disk cache проверяется только по `mtime` prepared image:

```python
def _disk_cache_valid(image_path, npz):
    return npz.is_file() and npz.stat().st_mtime >= image_path.stat().st_mtime
```

**Не учитывается:** изменение caption, trigger_words, resolution.

**Для текущего кейса:** пользователь подтвердил, что датасет не менялся → stale cache **не объясняет** текущий провал. Но это **баг/риск** на будущее при редактировании caption.

**job_9 (Chimera):** loaded from disk. **job_13 (Bloom):** 80 encoded fresh — stale cache **не применима** к Bloom.

---

## 12. Sampling config в UI (eval issue, не train)

Sampling config id=5 (Chimera) и id=7 (Bloom) — оба от Melanie, **без trigger** текущего персонажа:

- id=5: без `Winx_Chimera_CFTS`, output Melanie
- id=7: без `Winx_Bloom_CFTS`, output Melanie, orange hair prompt

**Impact:** встроенный sampler `lora-trainer` мог давать некорректную оценку. Пользователь тестировал в reForge — **это не главная причина**, но eval pipeline в UI требует отдельного sampling config на проект.

---

## 13. Отсутствующие параметры vs Kohya

| Kohya параметр | lora-trainer |
|----------------|--------------|
| enable_bucket | ❌ нет |
| max_token_length | ❌ нет (фикс. tokenizer max_length) |
| clip_skip | ✅ config есть, но **не влияет на SDXL** (см. §16) |
| caption_dropout | ❌ нет |
| shuffle_caption | ❌ нет |
| color_aug / flip_aug | ❌ нет |
| network_dim/alpha | ✅ lora_rank/lora_alpha |
| cache_latents / TE cache | ✅ |
| min_snr_gamma / noise_offset | ✅ (опционально) |
| FF + attention targets | ✅ (после fix) |
| Kohya LoRA init | ✅ (после fix) |

Наиболее критичный gap для Winx: **bucketing** (clip_skip — red herring, см. §16).

---

## 14. Caption quality (Chimera-specific, minor for Bloom)

Примеры шумных тегов в датасете:

- `00003`: `multiple girls, 2girls, applying makeup, mirror...`
- `00004`: `multiple girls, 2girls, weapon, helmet, sword, 1boy, parody...`
- `00008`: `multiple girls, 2girls, pink hair...`

Для character LoRA на 13 кадрах это создаёт **конфликтующий supervision signal**.

Для Chimera (13 img) — значимо. Bloom (80 img) — менее критично, результат всё равно плохой.

---

## 15. Crop centers

Crop centers хранятся в DB (`dataset_image_crops`), baked в `.prepared/1024/`. Центры варьируются (0.38–0.56 по X). При square crop из landscape-кадров часть лица/композиции может теряться.

---

## 16. clip_skip — red herring для SDXL (2026-06-30)

**Статус:** отвергнуто как contributing factor.

### Что было до добавления clip_skip

До commit `93c377c` код жёстко использовал penultimate layer:

```python
prompt_embeds_1 = enc1_out.hidden_states[-2]
prompt_embeds_2 = enc2_out.hidden_states[-2]
```

### Что даёт новый параметр

```python
# prompt_encoding.py
def select_clip_hidden_state(hidden_states, clip_skip):
    return hidden_states[-clip_skip]
```

Default `clip_skip=2` → `hidden_states[-2]` — **идентично старому поведению**. Качество не могло улучшиться.

### Kohya тоже не использует clip_skip для SDXL train

sd-scripts явно предупреждает при SDXL training:

> `clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません`

В Winx Kohya config `"clip_skip": 2` — **legacy UI field**, не активный train parameter для SDXL.

Документация sd-scripts: `--clip_skip=N` — «**Not typically used for SDXL**».

### Эталоны используют penultimate всегда

| Проект | SDXL TE encoding |
|--------|------------------|
| ai-toolkit `text_encode_xl` | `hidden_states[-2]` — comment: «always penultimate layer» |
| reForge `SDXLClipModel` | `clip_l` и `clip_g`: `layer_idx=-2` (фиксировано) |
| diffusers SDXL pipeline | penultimate для обоих encoders |
| lora-trainer (до и после) | `hidden_states[-2]` при default clip_skip=2 |

### Inference mismatch при clip_skip ≠ 2

Если поставить `clip_skip=1` в lora-trainer → `hidden_states[-1]` (last layer). reForge/reForge SDXL inference **всегда** `-2` → train/inference mismatch, качество только хуже.

### Вывод

- clip_skip — **SD1.5/SD2 concept**, не SDXL train gap.
- Сравнение с Kohya по `clip_skip: 2` было **misleading**.
- Реальные P0 остаются: **add_time_ids**, **bucketing**, **alpha=rank**.
