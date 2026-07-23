"use client";

import { useState } from "react";
import PageHeader from "@/components/ui/PageHeader";
import Tabs from "@/components/ui/Tabs";
import WorkerSettingsTab from "@/components/settings/WorkerSettingsTab";
import StorageSettingsTab from "@/components/settings/StorageSettingsTab";
import SystemInfoTab from "@/components/settings/SystemInfoTab";

const TABS = [
  { value: "worker", label: "Worker" },
  { value: "storage", label: "Storage" },
  { value: "system", label: "System" },
] as const;

type SettingsTab = (typeof TABS)[number]["value"];

export default function SettingsPage() {
  const [tab, setTab] = useState<SettingsTab>("worker");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Global application settings and system information"
      />
      <Tabs tabs={[...TABS]} value={tab} onChange={setTab} />
      {tab === "worker" ? <WorkerSettingsTab /> : tab === "storage" ? <StorageSettingsTab /> : <SystemInfoTab />}
    </div>
  );
}
