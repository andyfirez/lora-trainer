export interface SelectOption {
  value: string;
  label: string;
}

export const comfySamplerNameOptions: SelectOption[] = [
  { value: "euler", label: "Euler" },
  { value: "euler_ancestral", label: "Euler Ancestral" },
  { value: "dpmpp_2m", label: "DPM++ 2M" },
];

export const comfySchedulerOptions: SelectOption[] = [
  { value: "simple", label: "Simple" },
  { value: "karras", label: "Karras" },
];
