# Recommendations

Приоритизированные рекомендации (обновлено после Winx_Bloom, job_13).

---

## A. Немедленные эксперименты (без изменения кода)

### 1. Проверка alpha scale

Повторить короткий Bloom/Chimera прогон с **`lora_alpha = lora_rank`** (32/32 или 16/16), остальное без изменений.

Если likeness появится хотя бы слабо → alpha scale был significant contributor.

### 2. Inference при training resolution

Тест в reForge с **1024×1024** (как train), не 832×1216:

```
Winx_Bloom_CFTS, <character tags>
Size: 1024×1024
Steps: 40, CFG: 7.5
```

Если likeness улучшится vs 832×1216 → подтверждает **add_time_ids / AR mismatch**.

### 3. Eval prompt

```
Winx_Bloom_CFTS, <tags from dataset caption>
Negative: realistic, 3d
LoRA weight: 0.8–1.0
```

Смотреть epoch 1–5, не epoch 20.

### 4. Отдельный sampling config в UI

Создать config с trigger `Winx_Bloom_CFTS` / `Winx_Chimera_CFTS` — для regression test в built-in sampler (не влияет на reForge).

---

## B. Изменения в коде (приоритет после Bloom)

### P0 — Critical (единственные fixes, объясняющие оба кейса)

| Feature | Why |
|---------|-----|
| **Per-image / per-bucket add_time_ids при train** | Train `[1024,1024,0,0,1024,1024]` ≠ inference `[1216,832,0,0,1216,832]` |
| **Bucketing / multi-resolution** | 76/80 Bloom non-square; Kohya uses `enable_bucket` |
| **Default lora_alpha = lora_rank** | Текущий scale 0.5 vs Kohya 1.0 |

### P1 — High

| Feature | Why |
|---------|-----|
| clip_skip config (Kohya: 2) | Frozen TE + wrong CLIP layer |
| TE cache invalidation by caption hash | Future-proofing |
| Sampling config per LoRA project | UI eval broken (id=5, id=7 → Melanie) |

### P2 — Medium

| Feature | Why |
|---------|-----|
| min_snr_gamma / noise_offset defaults | Отключены в проблемных прогонах |
| max_token_length | Long captions truncated at 77 |
| Early checkpoint picker | Chimera loss rises ep.2–4 |

---

## C. Validation experiment (после P0 fixes)

1. Winx_Bloom dataset (80 img) — уже есть.
2. rank=16, alpha=16, bucketing on, correct add_time_ids.
3. 10 epochs, sample every 2.
4. Eval: reForge 832×1216 **и** 1024×1024 для сравнения.

**Success:** likeness visible by epoch 3–5 at intended inference AR.

---

## D. Что больше не нужно проверять

- ❌ «Удалить cache и переобучить» — Bloom уже с fresh cache, не помогло.
- ❌ «Добавить больше картинок» — 80 img достаточно для sanity check.
- ❌ «Поменять только init LoRA / FF targets» — уже сделано, не помогло.
- ❌ «Loss должен падать → всё OK» — Bloom loss падает, результат плохой.

---

## E. Open questions

1. Насколько inference at 1024×1024 улучшит Bloom без code changes? (quick user test)
2. Нужен ли TE LoRA при clip_skip=2?
3. Совместим ли Kohya export scale с reForge при alpha≠rank?
