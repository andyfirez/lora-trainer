export interface SelectOption {
  value: string;
  label: string;
  defaultScheduler?: string;
}

/** Bundled diffusers classes used by training YAML `sample_scheduler`. */
export const diffusersSchedulerOptions: SelectOption[] = [
  { value: "euler", label: "Euler" },
  { value: "euler_a", label: "Euler Ancestral" },
  { value: "ddim", label: "DDIM" },
  { value: "dpm++", label: "DPM++ (multistep)" },
];

export const samplerOptions: SelectOption[] = [
  { value: "euler", label: "euler", defaultScheduler: "normal" },
  { value: "euler_ancestral", label: "euler_ancestral", defaultScheduler: "normal" },
  { value: "dpmpp_2m", label: "dpmpp_2m", defaultScheduler: "karras" },
  { value: "ddim", label: "ddim", defaultScheduler: "ddim_uniform" },
  { value: "heun", label: "heun", defaultScheduler: "normal" },
  { value: "lms", label: "lms", defaultScheduler: "normal" },
  { value: "dpmpp_2s_ancestral", label: "dpmpp_2s_ancestral", defaultScheduler: "karras" },
  { value: "dpmpp_sde", label: "dpmpp_sde", defaultScheduler: "karras" },
];

export const schedulerOptions: SelectOption[] = [
  { value: "normal", label: "normal" },
  { value: "karras", label: "karras" },
  { value: "exponential", label: "exponential" },
  { value: "simple", label: "simple" },
  { value: "sgm_uniform", label: "sgm_uniform" },
  { value: "ddim_uniform", label: "ddim_uniform" },
  { value: "beta", label: "beta" },
  { value: "linear_quadratic", label: "linear_quadratic" },
  { value: "kl_optimal", label: "kl_optimal" },
];

export const LEGACY_BUNDLED_SCHEDULERS: Record<string, { sampler: string; scheduler: string }> = {
  euler: { sampler: "euler", scheduler: "normal" },
  euler_a: { sampler: "euler_ancestral", scheduler: "normal" },
  ddim: { sampler: "ddim", scheduler: "ddim_uniform" },
  "dpm++": { sampler: "dpmpp_2m", scheduler: "karras" },
};
