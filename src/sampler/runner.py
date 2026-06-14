"""Sampling runner — CLI entry point spawned by the queue worker."""

import argparse
import asyncio
import logging
import sys

from src.sampler.job_runner import run_sampling_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standalone LoRA sampling job")
    parser.add_argument("--job-id", type=int, required=True, help="Job ID in the database")
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(run_sampling_job(args.job_id))
        if exit_code != 0:
            sys.exit(exit_code)
    except SystemExit:
        raise
    except BaseException as exc:
        logger.exception("Sampling runner failed before run completed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
