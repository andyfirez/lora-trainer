export interface SelectOption {
  value: string;
  label: string;
}

export const diffusersSchedulerOptions: SelectOption[] = [
  { value: "euler", label: "Euler" },
  { value: "euler_a", label: "Euler Ancestral" },
  { value: "ddim", label: "DDIM" },
  { value: "dpm++", label: "DPM++ (multistep)" },
];

export const reforgeSamplerOptions: SelectOption[] = [
  { value: "euler_a", label: "Euler Ancestral" },
  { value: "dpmpp_2m", label: "DPM++ 2M" },
];

export const schedulerModeOptions: SelectOption[] = [
  { value: "normal", label: "Normal" },
  { value: "karras", label: "Karras" },
];
