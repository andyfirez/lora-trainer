import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, hint, className, id, ...props }, ref) => {
    const inputId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
    return (
      <div className="space-y-1">
        {label && (
          <label htmlFor={inputId} className="block text-caption text-text-muted">
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          className={cn(
            "w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text",
            "placeholder:text-text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30",
            "disabled:opacity-50",
            error && "border-error focus:border-error focus:ring-error/30",
            className,
          )}
          {...props}
        />
        {hint && !error && <p className="text-caption text-text-muted">{hint}</p>}
        {error && <p className="text-caption text-error">{error}</p>}
      </div>
    );
  },
);

Input.displayName = "Input";

export default Input;

export function Textarea({
  label,
  error,
  className,
  id,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement> & { label?: string; error?: string }) {
  const inputId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="space-y-1">
      {label && (
        <label htmlFor={inputId} className="block text-caption text-text-muted">
          {label}
        </label>
      )}
      <textarea
        id={inputId}
        className={cn(
          "w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text",
          "placeholder:text-text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30",
          "disabled:opacity-50 resize-y min-h-[80px]",
          error && "border-error",
          className,
        )}
        {...props}
      />
      {error && <p className="text-caption text-error">{error}</p>}
    </div>
  );
}

export const inputClass =
  "w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text placeholder:text-text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30";

export const labelClass = "block text-caption text-text-muted mb-1";

export const sectionClass =
  "rounded-xl border border-border bg-surface p-5 space-y-4";
