import { forwardRef, type SelectHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
}

const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, error, className, id, children, ...props }, ref) => {
    const selectId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
    return (
      <div className="space-y-1">
        {label && (
          <label htmlFor={selectId} className="block text-caption text-text-muted">
            {label}
          </label>
        )}
        <select
          ref={ref}
          id={selectId}
          className={cn(
            "w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text",
            "focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30",
            "disabled:opacity-50",
            error && "border-error",
            className,
          )}
          {...props}
        >
          {children}
        </select>
        {error && <p className="text-caption text-error">{error}</p>}
      </div>
    );
  },
);

Select.displayName = "Select";

export default Select;

export function Checkbox({
  label,
  className,
  id,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & { label?: string }) {
  const checkboxId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <label htmlFor={checkboxId} className={cn("flex items-center gap-2 cursor-pointer", className)}>
      <input
        type="checkbox"
        id={checkboxId}
        className="h-4 w-4 rounded border-border bg-bg text-accent focus:ring-accent/30"
        {...props}
      />
      {label && <span className="text-sm text-text">{label}</span>}
    </label>
  );
}
