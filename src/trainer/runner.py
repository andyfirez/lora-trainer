"""Training runner — CLI entry point spawned by the runnable worker.

Usage:
    python -m src.trainer.runner --lora-id <id>
"""

import argparse
import asyncio
import logging
import sys

from src.trainer.job_runner import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LoRA training run")
    parser.add_argument("--lora-id", type=int, required=True, help="Lora ID in the database")
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(run_training(args.lora_id))
        if exit_code != 0:
            sys.exit(exit_code)
    except SystemExit:
        raise
    except BaseException:
        logger.exception("Unhandled error in training runner for lora id=%d", args.lora_id)
        sys.exit(1)


if __name__ == "__main__":
    main()
