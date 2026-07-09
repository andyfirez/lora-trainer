import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export const inputClassName =
  "w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-text placeholder-muted focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/30 transition-colors";

export const labelClassName = "block text-xs font-medium text-muted mb-1";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, id, ...props }, ref) => {
    const inputId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
    return (
      <div>
        {label && (
          <label htmlFor={inputId} className={labelClassName}>
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          className={cn(inputClassName, error && "border-error focus:border-error focus:ring-error/30", className)}
          {...props}
        />
        {error && <p className="mt-1 text-xs text-error">{error}</p>}
      </div>
    );
  },
);
Input.displayName = "Input";

export default Input;
