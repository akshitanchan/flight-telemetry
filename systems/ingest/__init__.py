"""
systems/ingest
--------------
Real-time ADS-B ingestion package.

Polls the OpenSky Network /states/all API over async HTTP using OAuth2
client-credentials auth, lands raw responses as bronze JSONL, then runs
the shared normalize → dedup → write path for at-least-once idempotent
landing.

Public surface:
    token.TokenManager   — OAuth2 client-credentials token lifecycle
    client.OpenSkyClient — async HTTP client for /states/all
    service.IngestService — poll loop wiring normalize/dedup/write
    cli.main             — CLI entrypoint (python -m systems.ingest.cli)

Importing this package opens NO network connections.  All network activity
is deferred to explicit calls on instantiated objects (lazy clients).
"""
