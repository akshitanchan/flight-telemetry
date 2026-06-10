"""
systems/ingest/service.py
-------------------------
Poll loop that stitches together token management, HTTP polling, bronze
JSONL landing, normalization, deduplication, and the landing writer.

Pipeline per poll cycle
-----------------------
1. ``OpenSkyClient.fetch_states()``  → raw API dict
2. Append raw dict as one JSONL line to the bronze file (C4 shape:
   ``{"time": <unix_ts>, "states": [[...]]}``).
3. ``normalize_batch(snapshot_time, states)`` → landing-format dicts.
4. ``IdempotencyStore.check_and_mark(idem_key)`` → dedup.
5. ``LandingWriter.write(record)`` → silver-style JSONL landing.
6. Increment ``ingest_records_total`` per written record.

The bronze landing (step 2) is intentionally written BEFORE normalization
so that even if the process crashes mid-batch, the raw snapshot is durable
and can be replayed.

At-least-once polling + idempotent dedup → exactly-once landing.

Environment variables:
    OPENSKY_POLL_INTERVAL_S — seconds between poll cycles (default: 10)
    INGEST_BRONZE_PATH      — bronze JSONL output path
                              (default: data/interim/bronze_live.jsonl)
    INGEST_LANDING_PATH     — silver-style landing JSONL output path
                              (default: data/interim/landing_live.jsonl)
    INGEST_JOURNAL_PATH     — idempotency journal path
                              (default: data/interim/.idem_live_journal)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_S: float = 10.0


class IngestService:
    """Async poll loop for live OpenSky ingestion.

    Args:
        client: An :class:`~systems.ingest.client.OpenSkyClient` instance.
        bronze_path: Path for the raw bronze JSONL file (one line per API
                     response, exact ``{"time","states"}`` shape).
        landing_path: Path for the normalized/deduped silver-style landing
                      JSONL file (``LandingWriter`` output).
        journal_path: Optional idempotency journal path.  If ``None``, the
                      journal is in-memory only (lost on restart).
        poll_interval_s: Seconds between poll cycles.
        sleep_fn: Async sleep coroutine; defaults to ``asyncio.sleep``.
                  Inject a no-op for testing.
    """

    def __init__(
        self,
        client,  # OpenSkyClient; typed loosely to avoid circular import
        bronze_path: Path,
        landing_path: Path,
        journal_path: Path | None = None,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._client = client
        self._bronze_path = bronze_path
        self._landing_path = landing_path
        self._journal_path = journal_path
        self._poll_interval_s = poll_interval_s
        self._sleep_fn: Callable[[float], Awaitable[None]] = (
            sleep_fn if sleep_fn is not None else asyncio.sleep
        )

        # Initialized lazily in run()
        self._dedup = None
        self._writer = None
        self._bronze_file = None

        self._running = False
        self._poll_count = 0
        self._total_written = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self, max_polls: int | None = None) -> None:
        """Run the poll loop until cancelled or *max_polls* is reached.

        Args:
            max_polls: Stop after this many successful poll cycles.  Useful
                       for integration tests and bounded smoke runs.
                       ``None`` (default) loops forever.
        """
        from systems.replay.normalizer import normalize_batch
        from systems.replay.dedup import IdempotencyStore
        from systems.replay.writer import LandingWriter

        # Initialise components
        self._dedup = IdempotencyStore(journal_path=self._journal_path)
        self._bronze_path.parent.mkdir(parents=True, exist_ok=True)
        self._landing_path.parent.mkdir(parents=True, exist_ok=True)

        self._running = True
        logger.info(
            "IngestService starting; bronze=%s landing=%s poll_interval=%.1f s",
            self._bronze_path,
            self._landing_path,
            self._poll_interval_s,
        )

        with LandingWriter(output_path=self._landing_path) as writer:
            self._writer = writer
            with open(self._bronze_path, "a") as bronze_file:
                self._bronze_file = bronze_file

                async with self._client:
                    while self._running:
                        cycle_start = time.monotonic()

                        try:
                            written = await self._poll_once(
                                normalize_batch, bronze_file, writer
                            )
                            self._total_written += written
                        except asyncio.CancelledError:
                            logger.info("IngestService: poll cancelled, shutting down.")
                            break
                        except Exception as exc:
                            logger.error(
                                "Poll cycle %d failed: %s",
                                self._poll_count,
                                exc,
                                exc_info=True,
                            )

                        self._poll_count += 1
                        if max_polls is not None and self._poll_count >= max_polls:
                            logger.info(
                                "IngestService: reached max_polls=%d, stopping.",
                                max_polls,
                            )
                            break

                        # Respect poll interval minus cycle elapsed time.
                        elapsed = time.monotonic() - cycle_start
                        sleep_time = max(0.0, self._poll_interval_s - elapsed)
                        if sleep_time > 0:
                            await self._sleep_fn(sleep_time)

        # Persist dedup journal on clean exit.
        if self._dedup is not None:
            self._dedup.flush_journal()

        logger.info(
            "IngestService stopped after %d polls; %d records written total.",
            self._poll_count,
            self._total_written,
        )

    def stop(self) -> None:
        """Signal the loop to stop after the current cycle completes."""
        self._running = False
        logger.info("IngestService stop requested.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _poll_once(
        self,
        normalize_batch,
        bronze_file,
        writer,
    ) -> int:
        """Execute one poll cycle.

        1. Fetch states from OpenSky.
        2. Write raw snapshot to bronze JSONL.
        3. Normalize, dedup, write landing records.
        4. Return the number of records written this cycle.
        """
        from shared.obs.telemetry import ingest_records_total

        # 1. Fetch
        snapshot = await self._client.fetch_states()
        snapshot_time: int = snapshot.get("time") or int(time.time())
        states: list = snapshot.get("states") or []

        # 2. Bronze landing — write EXACT shape that read_snapshots() expects.
        bronze_line = json.dumps({"time": snapshot_time, "states": states})
        bronze_file.write(bronze_line + "\n")
        bronze_file.flush()

        logger.debug(
            "Poll %d: snapshot_time=%d vectors=%d",
            self._poll_count,
            snapshot_time,
            len(states),
        )

        # 3. Normalize
        records = normalize_batch(snapshot_time, states)

        # 4. Dedup + write
        written = 0
        for record in records:
            idem_key = record["idem_key"]
            if self._dedup.check_and_mark(idem_key):
                writer.write(record)
                ingest_records_total.inc()
                written += 1

        logger.info(
            "Poll %d: %d states → %d normalized → %d written (dedup skipped %d)",
            self._poll_count,
            len(states),
            len(records),
            written,
            len(records) - written,
        )
        return written


# ---------------------------------------------------------------------------
# Factory: build IngestService from environment
# ---------------------------------------------------------------------------

def ingest_service_from_env(client) -> "IngestService":
    """Build an :class:`IngestService` from environment variables.

    Args:
        client: A configured :class:`~systems.ingest.client.OpenSkyClient`.

    Returns:
        An :class:`IngestService` instance (no network calls made).
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    bronze_path = Path(
        os.environ.get(
            "INGEST_BRONZE_PATH",
            str(project_root / "data" / "interim" / "bronze_live.jsonl"),
        )
    )
    landing_path = Path(
        os.environ.get(
            "INGEST_LANDING_PATH",
            str(project_root / "data" / "interim" / "landing_live.jsonl"),
        )
    )
    journal_path = Path(
        os.environ.get(
            "INGEST_JOURNAL_PATH",
            str(project_root / "data" / "interim" / ".idem_live_journal"),
        )
    )
    poll_interval_s = float(
        os.environ.get("OPENSKY_POLL_INTERVAL_S", str(_DEFAULT_POLL_INTERVAL_S))
    )
    return IngestService(
        client=client,
        bronze_path=bronze_path,
        landing_path=landing_path,
        journal_path=journal_path,
        poll_interval_s=poll_interval_s,
    )
