"""Handlers dispatching Runnable entities (Lora, Sampling) to subprocess runners."""

from src.db.repositories.runnable_queries import RunnableKind
from src.services.runnable.handlers.base import RunnableHandler
from src.services.runnable.handlers.lora import LoraHandler
from src.services.runnable.handlers.sampling import SamplingHandler

_HANDLERS: dict[RunnableKind, RunnableHandler] = {
    "lora": LoraHandler(),
    "sampling": SamplingHandler(),
}


def get_runnable_handler(kind: RunnableKind) -> RunnableHandler:
    return _HANDLERS[kind]
