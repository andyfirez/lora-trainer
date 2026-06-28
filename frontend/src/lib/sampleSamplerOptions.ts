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
