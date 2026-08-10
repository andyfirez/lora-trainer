class LoraNotFoundError(Exception):
    def __init__(self, lora_id: int) -> None:
        self.lora_id = lora_id
        super().__init__(f"LoRA {lora_id} not found")


class LoraNameConflictError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"A LoRA named '{name}' already exists")


class LoraReproduceError(Exception):
    def __init__(self, lora_id: int) -> None:
        self.lora_id = lora_id
        super().__init__(f"LoRA {lora_id} has no config to reproduce")


class LoraCheckpointNotFoundError(Exception):
    def __init__(self, lora_id: int) -> None:
        self.lora_id = lora_id
        super().__init__(f"No checkpoint found for LoRA id={lora_id}")
