# LoRA Captioning — Tag Logic and Labeling Strategies

How captions work in character (and concept) LoRA training, and how to choose a labeling strategy.

---

## Core idea

Caption tags are not a human-readable file description — they **partition constants vs variables** during training.

> **What you tag and it varies across images** → controllable via prompt at inference.  
> **What you omit from the caption** → gets absorbed into the trigger word, becomes part of the concept.  
> **What you tag and it is identical everywhere** → duplicates the trigger, adds noise.

LoRA learns the association **full caption ↔ image**, not individual tags as on/off switches. A unique trigger word (e.g. `ohwx`, `sks person`) stands out in the embedding space, but every other tag still influences *which* visual features bind to the concept.

---

## Trigger word

Use a **unique, meaningless token** that does not collide with the base model vocabulary:

| Good | Bad |
|---|---|
| `ohwx`, `sks`, `zkw`, `mychar_x7` | `woman`, `girl`, `portrait`, a real name |

Convention for character LoRAs: `trigger, class` — e.g. `ohwx woman` or `ohwx, 1girl`.

The trigger must appear in **every** caption, with consistent spelling.

---

## Can you use a single tag for every image?

**Yes** — a valid strategy (DreamBooth identifier-only, “trigger only” workflows).

```
ohwx
ohwx
...
```

### Pros

- Strongest trigger-to-visual binding
- All untagged traits absorb into the trigger (face shape, palette, etc.)
- Minimal labeling effort

### Cons (critical with a diverse dataset)

If images **differ** (pose, clothing, background, lighting), the model learns:

```
ohwx = face + red dress + beach + arms crossed + ...
```

everything in one bundle. Result:

- concept blurs (averaging)
- “sticking” to specific training scenes
- you cannot prompt “same character, different outfit”

**Single tag works best** when images are homogeneous (same angle, lighting, outfit). With many varied frames, labeling separates **subject** from **scene**.

---

## Three labeling strategies

### A. Minimal — “bake everything into the trigger”

```
ohwx, 1girl
```

or just `ohwx`.

**When:** maximum identity lock-in; prompt flexibility is secondary.  
**Risk:** memorizes poses, outfits, and backgrounds from the dataset.

### B. Trigger + variable tags (standard for character LoRA)

```
ohwx, 1girl, ponytail, white shirt, black jacket, office, looking to the side
ohwx, 1girl, portrait, parted lips, indoors
```

**Logic:** elements that **change** across images → tag them → change via prompt at inference.  
Stable identity features (face shape, eye color, moles) → **omit** → absorbed into trigger.

### C. Trigger + explicit traits

```
ohwx, 1girl, blonde hair, blue eyes, mole on neck, office, suit
```

**Logic:** tagged traits must be **repeated in the prompt** at inference, or activation is weaker.  
Use when traits should be **prompt-controlled**, not baked in.

---

## Tag roles

| Tag type | Examples | Recommendation |
|---|---|---|
| Trigger | `ohwx` | Required, identical in every caption |
| Class | `1girl`, `woman`, `1boy` | Helps anchor what the LoRA modifies |
| Identity (optional) | `blue eyes`, `mole on neck` | Omit to bake in; tag to control via prompt |
| Variable scene | `office`, `suit`, `ponytail`, `outdoors` | Tag if you want to change them at inference |
| Style (booru) | `realistic`, `anime coloring` | Match your inference goal — see below |
| Auto-tagger noise | `blurry`, `absurdres`, wrong colors | Remove after WD14/BLIP |

---

## Training photos, generating illustrations

Common case: dataset images are **photographs**, target output is **2D/illustration** (Illustrious, Pony, etc.).

LoRA must extract identity from photo latents and apply it to the base model’s illustrative prior. Attributes (hair color, outfit) transfer easily; fine facial structure transfers with difficulty.

**Caption implication:** do not describe training images literally if the goal differs from inference. Tag for what you want **controllable** at generation time, not for what the JPEG happens to be.

---

## The `realistic` tag (Illustrious / booru models)

In booru-trained checkpoints, `realistic` means “illustration with realistic shading” — **not** “this is a photograph”.

| Training image | Caption tag |
|---|---|
| Actual photo | Pixels are photographic |
| WD14 output | `realistic` = semi-realistic 2D style token |

Training photos with `realistic` teaches an extra link “trigger + realistic = this render”, mixing identity and style.

**Keep `realistic` if:**
- target is semi-realistic 2D;
- you **always** include `realistic` in inference prompts.

**Remove `realistic` if:**
- target is pure anime/2D;
- you want the trigger to carry identity while the base model sets style.

Photo-level features are still learned from pixels without the tag. The goal is to avoid binding **style** to the trigger.

---

## What to remove after auto-captioning

WD14 and similar taggers often add:

- **Contradictions** — `brown hair` on some files, `blonde hair` on others for the same subject
- **Quality/meta tags** — `blurry`, `absurdres`, `highres`, duplicate `1girl`
- **Irrelevant scene tags** — unless you want them prompt-controllable
- **Tags that misdescribe the image** — common on real photos in booru vocab

Always review auto-tags manually or with a tag editor.

---

## Community practice

Sources: [Kohya train README](https://github.com/bmaltais/kohya_ss/blob/master/docs/train_README.md), [Civitai tagging guide](https://civitai.com/articles/29347/the-ultimate-lora-tagging-guide-from-dirty-image-to-perfect-model), [Scenario advanced captioning](https://help.scenario.com/articles/5782148871-advanced-captioning), [sandner.art](https://sandner.art/ai-for-designers-training-custom-lora-models/).

1. **Trigger in every caption**, same wording.
2. **Tag what you want to change** at inference.
3. **Do not tag what you want baked in** to the concept.
4. **Enable shuffle tags** in the trainer (`--shuffle_caption`) — learns meaning, not word order.
5. **Review auto-captions** — raw WD14/BLIP output is rarely sufficient for character LoRAs.
6. **Minimal captions** — valid for pure identity on homogeneous datasets.

---

## Choosing a strategy

| Dataset | Goal | Suggested approach |
|---|---|---|
| Homogeneous portraits | Lock identity | A: trigger only, or `trigger, 1girl` |
| Varied poses/outfits | Flexible character | B: trigger + variable tags per image |
| Need prompt control over hair/eyes | Controllable traits | C: tag those traits explicitly |
| Photos → 2D output | Illustrative likeness | B or minimal; drop `realistic`; match inference style |
| Style LoRA | Transfer aesthetic | Tag style elements; often fewer identity tags |

For a **diverse character dataset** (many poses, outfits, backgrounds), a practical middle ground:

```
# Identity-focused
ohwx, 1girl

# Scene control
ohwx, 1girl, ponytail, white shirt, black jacket, looking to the side
```

Avoid dumping every WD14 tag into every caption.

---

## A/B/C captioning experiment

Validate strategy on your data before a full run. Three short trainings (~5 epochs), identical hyperparameters:

| Variant | Captions |
|---|---|
| A | Trigger only |
| B | Trigger + class (`ohwx, 1girl`) |
| C | Trigger + class + per-image variable tags |

Compare:

- likeness / concept fidelity
- ability to change outfit, pose, or background via prompt
- memorization of specific training scenes

Pick the variant with the best trade-off for your use case.

---

## Quick checklist

- [ ] Unique trigger, same in every `.txt`
- [ ] Class token after trigger (`1girl`, `woman`, …)
- [ ] Variable elements tagged per image; stable identity traits omitted (or tagged deliberately)
- [ ] Auto-tagger noise and contradictions removed
- [ ] Style tags aligned with inference goal
- [ ] Shuffle tags enabled in training config
- [ ] Tested at multiple epochs and LoRA weights
