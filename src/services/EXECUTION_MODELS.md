# Backend execution models

LoRA Trainer uses three distinct ways to run work. They are intentionally separate: training and sampling need isolated GPU subprocesses with DB-backed lifecycle, autotagging is lightweight and in-process, and most API operations are synchronous request/response.

## 1. Subprocess runnable queue (`SubprocessRunnableWorker`)

**Location:** `src/services/worker/service.py`  
**Tables:** `loras`, `samplings` (shared runnable mixin: `QUEUED` → `RUNNING` → terminal)  
**Concurrency:** `TrainingSettings.max_concurrent_jobs` — limits parallel **runnable** subprocesses, not tagging or HTTP handlers.

On API startup the embedded worker polls the DB, dequeues runnable rows, and spawns child processes (`trainer.runner`, `sampler.runner`). Cancellation is cooperative: status flips to `CANCELLED`, the worker kills the process tree, handlers run `finalize()`.

This is **not** a generic job runner for every background task — only LoRA training and sampling configs use this path.

## 2. In-process tagging (`TaggingTaskManager`)

**Location:** `src/services/tagging/manager.py`  
**State:** in-memory dict keyed by `dataset_id` (no DB table, no subprocess)

WD14 autotagging runs inside the API process: a background asyncio task offloads blocking ONNX inference to a thread pool. Progress is polled via `/datasets/{id}/autotag/status`. State is lost on restart, which is acceptable for fire-and-forget caption generation.

Tagging does **not** go through the runnable queue or `SubprocessRunnableWorker`.

## 3. Synchronous API services

**Location:** `src/services/datasets/`, `src/services/loras/`, `src/services/sampling/`, etc.

CRUD, disk↔DB catalog sync, preprocess/crop/bake, and file browsing execute in the request handler (async I/O + CPU-bound PIL/hash work on the event loop thread). No queue, no subprocess.

---

## Naming notes

| Name | Meaning |
|------|---------|
| `SubprocessRunnableWorker` | Polls runnable tables and spawns training/sampling subprocesses |
| `max_concurrent_jobs` | Max parallel runnable subprocesses (loras + samplings combined) |
| `RunnableHandler` | Kind-specific command builder + finalize hook for one runnable type |
| `TaggingService` | Thin wrapper over the in-process autotag manager |

Legacy Alembic migrations may refer to removed `jobs` / `trained_loras` tables; current code uses `loras` and `samplings` only.
