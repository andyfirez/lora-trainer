"""Queue worker CLI — standalone debug entry point.

Run with:
    uv run python -m scripts.worker
"""

import asyncio
import logging
import sys

from src.db.session import run_migrations
from src.services.worker.service import QueueWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


async def _main() -> None:
    await run_migrations()
    worker = QueueWorker(echo_subprocess_output=True)
    await worker.start()
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await worker.stop()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
