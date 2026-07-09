import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/cn";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "icon";
type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  children: ReactNode;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-accent hover:bg-accent-hover text-bg font-medium shadow-sm",
  secondary:
    "border border-border bg-surface hover:bg-surface-raised text-text-muted hover:text-text",
  ghost:
    "text-text-muted hover:text-text hover:bg-white/5",
  danger:
    "text-error hover:text-error hover:bg-error-muted",
  icon:
    "p-1.5 text-text-muted hover:text-text hover:bg-white/10",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "px-3 py-1.5 text-xs rounded-md gap-1.5",
  md: "px-4 py-2 text-sm rounded-lg gap-2",
  lg: "px-5 py-2.5 text-sm rounded-lg gap-2",
};

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", size = "md", className, children, disabled, ...props }, ref) => {
    const isIcon = variant === "icon";
    return (
      <button
        ref={ref}
        disabled={disabled}
        className={cn(
          "inline-flex items-center justify-center transition-colors focus-ring disabled:opacity-50 disabled:cursor-not-allowed",
          isIcon ? variantClasses.icon : [variantClasses[variant], sizeClasses[size]],
          className,
        )}
        {...props}
      >
        {children}
      </button>
    );
  },
);

Button.displayName = "Button";

export default Button;
