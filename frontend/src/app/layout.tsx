import type { Metadata } from "next";
import { Syne, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const ibmPlexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "LoRA Trainer",
  description: "Local SDXL LoRA training with job queue",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${syne.variable} ${ibmPlexSans.variable}`}>
      <body className="font-body antialiased">
        <div className="app-atmosphere flex min-h-screen">
          <div className="grain-overlay" aria-hidden="true" />
          <Sidebar />
          <main className="flex-1 p-4 md:p-6 lg:p-8 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
