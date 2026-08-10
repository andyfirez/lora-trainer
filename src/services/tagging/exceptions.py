class TaggingAlreadyRunningError(Exception):
    def __init__(self, dataset_id: int) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Autotag is already running for dataset id={dataset_id}")
