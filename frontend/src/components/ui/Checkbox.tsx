import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type"> {
  label?: string;
}

const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, label, id, ...props }, ref) => {
    const checkboxId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
    return (
      <label htmlFor={checkboxId} className="flex items-center gap-2 cursor-pointer">
        <input
          ref={ref}
          type="checkbox"
          id={checkboxId}
          className={cn("w-4 h-4 rounded border-border bg-bg accent-accent", className)}
          {...props}
        />
        {label && <span className="text-sm text-text">{label}</span>}
      </label>
    );
  },
);
Checkbox.displayName = "Checkbox";

export default Checkbox;
