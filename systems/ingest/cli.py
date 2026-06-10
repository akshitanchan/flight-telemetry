#!/usr/bin/env python3
"""
systems/ingest/cli.py
---------------------
Entrypoint for the real-time OpenSky ingestion service.

Usage:
    python -m systems.ingest.cli [OPTIONS]
    python -m systems.ingest.cli --help

Environment variables (all optional; sane defaults provided):
    OPENSKY_CLIENT_ID           OAuth2 client ID
    OPENSKY_CLIENT_SECRET       OAuth2 client secret
    OPENSKY_TOKEN_URL           OAuth2 token endpoint
    OPENSKY_BASE_URL            OpenSky API base URL
    OPENSKY_BBOX_LAMIN          Bounding-box south latitude  (default: 47.0)
    OPENSKY_BBOX_LAMAX          Bounding-box north latitude  (default: 55.0)
    OPENSKY_BBOX_LOMIN          Bounding-box west longitude  (default: 5.0)
    OPENSKY_BBOX_LOMAX          Bounding-box east longitude  (default: 15.0)
    OPENSKY_POLL_INTERVAL_S     Seconds between poll cycles  (default: 10)
    INGEST_BRONZE_PATH          Bronze JSONL output path
    INGEST_LANDING_PATH         Landing JSONL output path
    INGEST_JOURNAL_PATH         Idempotency journal path
    INGEST_METRICS_PORT         Prometheus /metrics port     (default: 8001)
    OTEL_EXPORTER_OTLP_ENDPOINT OTLP collector endpoint
    OTEL_SERVICE_NAME           OTel service name            (default: ingestion)
    LOG_LEVEL                   Logging level                (default: INFO)
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Resolve project root so the package is importable when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _configure_logging(level: str = "INFO") -> None:
    """Configure structured logging to stderr."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m systems.ingest.cli",
        description=(
            "Real-time ADS-B ingestion: polls OpenSky /states/all "
            "and lands records as bronze + normalized JSONL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.environ.get("INGEST_METRICS_PORT", "8001")),
        help="Port for the Prometheus /metrics HTTP server.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop after N poll cycles (useful for smoke tests and CI). "
            "Omit for continuous operation."
        ),
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Wire up and run the ingestion service."""
    # Import lazily so module-level import has zero side-effects.
    from shared.obs.telemetry import start_metrics_server, setup_tracing
    from systems.ingest.token import token_manager_from_env
    from systems.ingest.client import OpenSkyClient
    from systems.ingest.service import IngestService, ingest_service_from_env

    logger = logging.getLogger("ingest.cli")

    # --- Observability -------------------------------------------------------
    service_name = os.environ.get("OTEL_SERVICE_NAME", "ingestion")
    setup_tracing(service_name)
    start_metrics_server(port=args.metrics_port)
    logger.info("Observability configured (service=%s, metrics_port=%d).",
                service_name, args.metrics_port)

    # --- Build components (no network yet) -----------------------------------
    token_mgr = token_manager_from_env()
    client = OpenSkyClient(token_manager=token_mgr)
    service = ingest_service_from_env(client)

    # Override poll count from CLI arg if provided.
    if args.max_polls is not None:
        service._poll_count = 0  # reset; max_polls enforced inside run()

    # --- Graceful shutdown on SIGTERM/SIGINT ---------------------------------
    loop = asyncio.get_running_loop()

    def _handle_signal(signum, frame):
        logger.info("Signal %d received; stopping ingestion service.", signum)
        service.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # --- Run -----------------------------------------------------------------
    logger.info("Starting IngestService (max_polls=%s).", args.max_polls)
    await service.run(max_polls=args.max_polls)
    logger.info("IngestService exited cleanly.")


def main() -> None:
    """CLI entrypoint: parse args, configure logging, run async loop."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
