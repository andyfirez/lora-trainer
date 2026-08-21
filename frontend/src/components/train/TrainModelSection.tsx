"use client";

import PathInput from "@/components/PathInput";
import FormSection from "@/components/ui/FormSection";
import { trainHint } from "@/lib/trainParameterMetadata";
import { stripLoraVersionSuffix } from "@/lib/trainConfigSanitize";
import type { TrainConfigFormContext } from "@/hooks/useTrainConfigForm";
import { TrainSelectInput, TrainTextInput } from "@/components/train/TrainFormFields";

interface TrainModelSectionProps {
  form: TrainConfigFormContext;
}

export default function TrainModelSection({ form }: TrainModelSectionProps) {
  const { config, set } = form;

  return (
    <FormSection title="Model">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PathInput
          label="Base Model"
          value={(config.base_model_name as string) ?? ""}
          onChange={(value) => set("base_model_name", value)}
          placeholder="sdxl-base"
          kind="model"
          pickerTitle="Select base model"
          {...trainHint("base_model_name")}
        />
        <PathInput
          label="Output Folder"
          value={(config.output_dir as string) ?? ""}
          onChange={(value) => set("output_dir", value)}
          placeholder=""
          kind="directory"
          pickerTitle="Select output folder"
          {...trainHint("output_dir")}
        />
      </div>
      <TrainTextInput
        label="LoRA Name"
        value={stripLoraVersionSuffix((config.lora_name as string) ?? "")}
        onChange={(value) => set("lora_name", value)}
        placeholder="my_lora"
        paramKey="lora_name"
      />
      <p className="text-xs text-muted -mt-2">
        Each training run gets a unique output folder name automatically.
      </p>
      <TrainSelectInput
        label="Output Format"
        value={(config.output_format as string) ?? "safetensors"}
        onChange={(value) => set("output_format", value)}
        paramKey="output_format"
        options={[
          { value: "safetensors", label: "safetensors" },
          { value: "pt", label: "pt" },
        ]}
      />
    </FormSection>
  );
}
