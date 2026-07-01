# add_time_ids collate bug (batch_size > 1)

Техническая записка о баге SDXL micro-conditioning при батчинге.

## Симптом

- `batch_size >= 2`, bucketing включён
- Loss растёт или качество генерации деградирует в шум
- При `batch_size=1` проблема не воспроизводится

## Механизм

### Что возвращал dataset

```python
# ConceptDataset.__getitem__ (до fix)
return {
    "add_time_ids": (918, 1216, 3, 0, 768, 1024),  # tuple[int, ...]
    ...
}
```

### Что делает PyTorch default_collate

Для списка tuple'ов collate **транспонирует** батч:

```python
# pseudo: batch = [tuple_a, tuple_b]
# result = [tensor([a0, b0]), tensor([a1, b1]), ...]  # 6 tensors of shape (2,)
# NOT [tensor_a, tensor_b]  # 2 tensors of shape (6,)
```

### Что ожидал trainer (до fix)

`_stack_add_time_ids` строил `(6, batch_size)` вместо `(batch_size, 6)`.

### Что делает diffusers UNet

```python
time_embeds = self.add_time_proj(time_ids.flatten())
time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
```

При форме `(6, 2)` flatten даёт `[h0, h1, w0, w1, top0, top1, ...]` — **перемешивание** conditioning между сэмплами A и B.

## Fix

**Файлы:** `src/trainer/sdxl/dataset.py`, `src/trainer/sdxl/trainer.py`

```python
# dataset.py — в __getitem__
add_time_ids = torch.tensor(meta.add_time_ids, dtype=torch.float32)
```

`default_collate` распознаёт `torch.Tensor` и делает `torch.stack` → `(batch_size, 6)`.

**Тест:** `test_dataloader_collate_add_time_ids_without_cross_sample_mixing`

## Связь с прогоном Winx_Bloom_CFTS (июль 2026)

Fix применён до прогона (подтверждено пользователем). Деградация ep1→ep10 при стабильном loss указывает на **отдельную** проблему (LR/optimizer/precision), см. `05-bucketing-run-jul2026.md`.

Checkpoint'ы, обученные **до** fix при `batch_size>1`, не следует resume'ить.
