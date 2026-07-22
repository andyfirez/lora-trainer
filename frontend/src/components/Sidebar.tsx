"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ListOrdered,
  Database,
  FileCog,
  Sparkles,
  BookOpen,
  PanelLeftClose,
  PanelLeft,
  Menu,
  X,
  ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/cn";

const NAV = [
  { href: "/jobs", label: "Jobs", icon: ListOrdered },
  { href: "/trainings", label: "Trainings", icon: FileCog },
  { href: "/loras", label: "LoRAs", icon: Sparkles },
  { href: "/sampling", label: "Sampling", icon: ImageIcon },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/parameters", label: "Parameters", icon: BookOpen },
];

const STORAGE_KEY = "sidebar-collapsed";

export default function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "true") setCollapsed(true);
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem(STORAGE_KEY, String(next));
      return next;
    });
  };

  const navContent = (compact: boolean) => (
    <nav className="flex-1 px-2 py-4 space-y-1">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            title={compact ? label : undefined}
            aria-label={compact ? label : undefined}
            className={cn(
              "flex items-center rounded-lg text-sm font-medium transition-colors",
              compact ? "justify-center px-2 py-2.5" : "gap-3 px-3 py-2",
              active
                ? "bg-accent text-white shadow-sm"
                : "text-muted hover:text-text hover:bg-white/5",
            )}
          >
            <Icon size={18} className="shrink-0" />
            {!compact && <span>{label}</span>}
          </Link>
        );
      })}
    </nav>
  );

  return (
    <>
      <button
        type="button"
        onClick={() => setMobileOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-40 p-2 rounded-lg bg-surface border border-border text-muted hover:text-text"
        aria-label="Open navigation"
      >
        <Menu size={20} />
      </button>

      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-black/60"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          "lg:hidden fixed inset-y-0 left-0 z-50 w-sidebar flex flex-col border-r border-border bg-surface shadow-lg transition-transform duration-200",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between px-4 py-4 border-b border-border">
          <span className="text-lg font-bold text-text font-display tracking-tight">LoRA Trainer</span>
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            className="p-1.5 rounded-lg hover:bg-white/5 text-muted hover:text-text"
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>
        {navContent(false)}
      </aside>

      <aside
        className={cn(
          "hidden lg:flex shrink-0 h-screen sticky top-0 flex-col border-r border-border bg-surface transition-[width] duration-200",
          collapsed ? "w-sidebar-collapsed" : "w-sidebar",
        )}
      >
        <div
          className={cn(
            "flex items-center border-b border-border",
            collapsed ? "justify-center px-2 py-4" : "justify-between px-4 py-4",
          )}
        >
          {!collapsed && (
            <span className="text-lg font-bold text-text font-display tracking-tight">LoRA Trainer</span>
          )}
          <button
            type="button"
            onClick={toggleCollapsed}
            className="p-1.5 rounded-lg hover:bg-white/5 text-muted hover:text-text"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <PanelLeft size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>
        {navContent(collapsed)}
      </aside>
    </>
  );
}
