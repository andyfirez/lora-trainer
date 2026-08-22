"use client";

import { useEffect, useRef, type TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/cn";
import { inputClassName } from "@/components/ui/Input";

type AutoGrowTextareaProps = Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, "rows"> & {
  minRows?: number;
};

export default function AutoGrowTextarea({
  className,
  value,
  minRows = 2,
  onInput,
  ...props
}: AutoGrowTextareaProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  function fit() {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }

  useEffect(() => {
    fit();
  }, [value]);

  return (
    <textarea
      {...props}
      ref={ref}
      rows={minRows}
      value={value}
      onInput={(event) => {
        fit();
        onInput?.(event);
      }}
      className={cn(inputClassName, "min-h-[3.25rem] resize-none overflow-hidden", className)}
    />
  );
}
