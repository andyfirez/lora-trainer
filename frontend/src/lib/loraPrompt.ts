/** Append an A1111-style LoRA tag to a prompt string. */
export function appendLoraTag(prompt: string, name: string, weight: number): string {
  const tag = `<lora:${name}:${weight}>`;
  const trimmed = prompt.trim();
  if (!trimmed) return tag;
  return `${trimmed} ${tag}`;
}
