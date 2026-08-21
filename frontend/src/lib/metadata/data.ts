import type { ParameterMeta } from "../parameterUtils";

export const dataParameterMetadata: ParameterMeta[] = [
// Data
  {
    key: "resolution",
    label: "Resolution",
    section: "Data",
    shortHint: "Training image resolution in pixels. SDXL default is 1024.",
    description:
      "Target resolution for training images. SDXL is trained at 1024×1024; datasets should be preprocessed to match. With bucketing enabled, images are grouped into aspect-ratio buckets near this resolution.",
    defaultValue: "1024",
    constraints: "64–2048",
    showInlineHint: false,
    recommendedValue: "1024",
    rangeGuidance: [
      { range: "512–768", description: "Lower resolution; faster training, less detail for SDXL." },
      { range: "1024", description: "SDXL native resolution; recommended default." },
      { range: "1280–1536", description: "Higher detail; significantly more VRAM per image." },
    ],
  },
  {
    key: "enable_bucket",
    label: "Enable Bucketing",
    section: "Data",
    shortHint: "Group images by aspect ratio instead of forcing square crops.",
    description:
      "When enabled, images are assigned to resolution buckets preserving aspect ratio, reducing distortion from squashing non-square images. Dataset preprocessing must match this setting.",
    defaultValue: "false",
    showInlineHint: false,
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Preserve aspect ratios via resolution buckets." },
      { value: "false", description: "Square crop/resize all images to resolution." },
    ],
  },
  {
    key: "bucket_reso_steps",
    label: "Bucket Resolution Steps",
    section: "Data",
    shortHint: "Step size between bucket resolutions.",
    description:
      "Granularity of bucket sizes when aspect-ratio bucketing is enabled. Smaller steps give finer aspect-ratio matching but more buckets and potential batch inefficiency.",
    defaultValue: "64",
    constraints: "8–512",
    yamlOnly: true,
    recommendedValue: "64",
    rangeGuidance: [
      { range: "32", description: "Fine-grained buckets; better aspect-ratio matching, more buckets." },
      { range: "64", description: "Standard step size; good balance." },
      { range: "128", description: "Coarser buckets; fewer buckets, less precise aspect matching." },
    ],
  },
  {
    key: "min_bucket_reso",
    label: "Min Bucket Resolution",
    section: "Data",
    shortHint: "Smallest bucket edge length.",
    description:
      "Minimum bucket resolution edge when bucketing is enabled. Images smaller than this may be upscaled or filtered depending on bucket_no_upscale.",
    defaultValue: "512",
    constraints: "64–2048",
    yamlOnly: true,
    recommendedValue: "512",
    rangeGuidance: [
      { range: "256–512", description: "Filters very small images; reduces upscaling artifacts." },
      { range: "768", description: "Higher minimum; excludes low-res training images." },
    ],
  },
  {
    key: "max_bucket_reso",
    label: "Max Bucket Resolution",
    section: "Data",
    shortHint: "Largest bucket edge length.",
    description:
      "Maximum bucket resolution edge when bucketing is enabled. Caps VRAM usage for very large aspect-ratio images.",
    defaultValue: "2048",
    constraints: "64–2048",
    yamlOnly: true,
    recommendedValue: "2048",
    rangeGuidance: [
      { range: "1024", description: "Caps VRAM for large aspect-ratio images." },
      { range: "1536–2048", description: "Allows high-res buckets; needs more VRAM." },
    ],
  },
  {
    key: "bucket_no_upscale",
    label: "Bucket No Upscale",
    section: "Data",
    shortHint: "Prevent upscaling images to fit larger buckets.",
    description:
      "When true, images are not upscaled to fit a larger bucket — they stay at native resolution within bucket constraints. Reduces artifacts from upscaling small source images.",
    defaultValue: "true",
    yamlOnly: true,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Keep native resolution; no upscaling to fit buckets." },
      { value: "false", description: "Allow upscaling small images into larger buckets." },
    ],
  },
  {
    key: "concepts.dataset_id",
    label: "Dataset",
    section: "Data",
    shortHint: "Dataset providing images and captions for this concept.",
    description:
      "References a preprocessed dataset by ID. The dataset must match training resolution and bucketing settings. Images and captions are loaded from the dataset's prepared directory at training time.",
    recommendedValue: "your-dataset-id",
  },
  {
    key: "concepts.trigger_words",
    label: "Trigger Words",
    section: "Data",
    shortHint: "Tokens prepended to captions to activate the LoRA at inference.",
    description:
      "Comma-separated trigger words inserted into training captions and sample prompts. These tokens become associated with the learned concept — use unique, rare tokens (e.g. ohwx, sks) to avoid conflicts with base model vocabulary.",
    recommendedValue: "ohwx, unique token",
  },
  {
    key: "concepts.caption_extension",
    label: "Caption Extension",
    section: "Data",
    shortHint: "File extension for caption sidecar files.",
    description:
      "Extension of caption files alongside images (e.g. .txt for image.jpg → image.txt). Must match how captions were exported from your tagging workflow.",
    defaultValue: ".txt",
    recommendedValue: ".txt",
  },
  {
    key: "concepts.repeats",
    label: "Repeats",
    section: "Data",
    shortHint: "How many times each image in this concept is seen per epoch.",
    description:
      "Multiplies each image's contribution per epoch. Increase repeats for small datasets to give the model more exposure without adding epochs. Schema default is 3; the form defaults new concepts to 1.",
    defaultValue: "3",
    constraints: "≥ 1",
    recommendedValue: "3",
    rangeGuidance: [
      { range: "1", description: "Each image seen once per epoch; form default for new concepts." },
      { range: "3–5", description: "Common for small datasets (10–20 images)." },
      { range: "10+", description: "Heavy repetition for very small sets; watch for overfitting." },
    ],
  },
  {
    key: "concepts.caption_suffix",
    label: "Caption Suffix",
    section: "Data",
    shortHint: "Text appended to every caption in this concept.",
    description:
      "Optional suffix added to all captions for this concept after loading. Useful for adding consistent style tags or quality tokens across a dataset.",
    defaultValue: '""',
    yamlOnly: true,
    recommendedValue: '""',
  },
  {
    key: "concepts.image_dir",
    label: "Image Directory",
    section: "Data",
    shortHint: "Deprecated — resolved automatically from dataset_id at runtime.",
    description:
      "Legacy field for the raw image directory path. Modern configs use concepts.dataset_id instead; the trainer resolves image_dir from the dataset record when the job starts. Do not set manually in new configs.",
    yamlOnly: true,
    deprecated: true,
    recommendedValue: "do not set (use concepts.dataset_id)",
  },
  {
    key: "concepts.prepared_dir",
    label: "Prepared Directory",
    section: "Data",
    shortHint: "Deprecated — resolved automatically from dataset_id at runtime.",
    description:
      "Legacy field for the preprocessed dataset directory. Modern configs use concepts.dataset_id instead; the trainer resolves prepared_dir from the dataset record when the job starts. Do not set manually in new configs.",
    yamlOnly: true,
    deprecated: true,
    recommendedValue: "do not set (use concepts.dataset_id)",
  },
];
