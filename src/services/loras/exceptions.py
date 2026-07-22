class TrainedLoraNotFoundError(Exception):
    def __init__(self, lora_id: int) -> None:
        self.lora_id = lora_id
        super().__init__(f"Trained LoRA {lora_id} not found")


class TrainedLoraAlreadyExistsError(Exception):
    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        super().__init__(f"Trained LoRA already exists for job {job_id}")
