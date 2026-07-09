# LoRA Trainer — Design System

## Subject grounding

LoRA Trainer is a local SDXL LoRA training workshop: GPU jobs, datasets, configs, and loss curves. The audience is ML practitioners who monitor long-running training jobs and need dense, scannable information without visual noise.

## Aesthetic

**Industrial studio / refined utilitarian** — a GPU workshop feel for power users. Dark, atmospheric surfaces with warm copper accents against deep charcoal. One signature element: subtle grain texture on surfaces. Everything else stays disciplined.

## Tech stack

Next.js 15, React 19, TypeScript, Tailwind CSS 3.4. No external UI component library.

## Typography

Single-family stack via `next/font/google` **Inter** (weights 400, 500, 600, 700). Both `--font-display` and `--font-body` map to Inter for a conventional, readable UI.

| Role | Family | Usage |
|------|--------|-------|
| Display | Inter (600–700) | Page titles, section headings, brand |
| Body | Inter (400–500) | UI copy, forms, tables, data |
| Mono | System `font-mono` or Inter + `tabular-nums` | Logs, paths, metrics |

### Type scale

- `text-xs` (12px) — labels, captions, badges
- `text-sm` (14px) — body, table cells, form inputs
- `text-base` (16px) — default body
- `text-lg` (18px) — section titles (`font-semibold`)
- `text-2xl` (24px) — page titles (`font-bold font-display`)

## Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg` | `#0c0d10` | Page background |
| `--surface` | `#14161b` | Cards, sidebar, panels |
| `--surface-raised` | `#1a1d24` | Elevated cards, modals |
| `--border` | `#2a2e38` | Borders, dividers |
| `--text` | `#e8e6e3` | Primary text |
| `--muted` | `#6b7280` | Secondary text, placeholders |
| `--accent` | `#d97706` | Training actions, primary CTA (copper) |
| `--accent-hover` | `#b45309` | Primary hover |
| `--sampling` | `#a78bfa` | Sampling job type accent |
| `--success` | `#22c55e` | Completed, ready states |
| `--warning` | `#eab308` | Queued, pending prep |
| `--error` | `#ef4444` | Failed, destructive |
| `--running` | `#3b82f6` | Active/running jobs |

## Signature element

Subtle SVG noise grain overlay (`.grain-overlay`) on the app shell — low opacity, fixed position, pointer-events none. Paired with a refined semantic status color system.

## Layout

- Collapsible sidebar: 224px expanded, 64px icon-only compact mode
- Mobile: drawer overlay below `lg` breakpoint
- Main content: responsive padding with max readable width on form pages

## Motion

- Dashboard stat cards: staggered `fade-up` entrance on load
- `prefers-reduced-motion`: all animations disabled
- No animation on frequently re-rendered job rows (polling pages)
