# AI Layer — eval-driven operational agent

**What this proves:** the headline is the **evaluation harness**, not a chat demo. Answers are
grounded (the model only routes; validated tools compute), checkable (deterministic checks +
required citations), and three LLM architectures can be compared head-to-head on one golden set.

## Metrics status (W4.4)

Three measurement categories apply to every number in this layer. They are labelled throughout this
document.

| Label | Meaning |
|---|---|
| **AUTHORITATIVE** | Measured offline, CI-blocking. Fails CI if the assertion does not hold. |
| **ADVISORY** | Measured offline on stub data. Reported by the harness; does not fail CI. |
| **LIVE-LLM PENDING** | Requires a live OpenAI or Ollama provider. Owner must run; results not yet recorded here. |

## Golden set (104 questions, tiered)

`ai/eval/golden_set.json` v1.1 — 104 questions across two tiers.

| Tier | Count | Purpose |
|---|---|---|
| `core` | 79 | Deterministically routable. The **AUTHORITATIVE** CI gate asserts accuracy == 1.0 and citation == 1.0 on these. |
| `extended` | 25 | Harder and long-tail phrasings. Used for advisory metrics and LLM architecture comparison. |

## AUTHORITATIVE gate: deterministic strategies on core tier

Measured in `outputs/ai_compare_small.json` (offline, no LLM provider, `--no-llm` run).

| Strategy | Questions | Accuracy | Citation coverage |
|---|---|---|---|
| `rule_based_v1` | 79 core | **1.0** | **1.0** |
| `keyword_score_v1` | 79 core | **1.0** | **1.0** |

CI exits non-zero if either strategy drops below 1.0 on the core tier. These are the only checks
that block CI.

On the **full 104 questions** (core + extended), the deterministic routers score lower — they
intentionally miss some extended-tier phrasings. This is where LLM architectures add value.

| Strategy | Questions | Accuracy (full 104) | Citation coverage |
|---|---|---|---|
| `rule_based_v1` | 104 | 0.904 | 1.0 |
| `keyword_score_v1` | 104 | 0.913 | 1.0 |

## ADVISORY metrics: faithfulness (offline stub run)

Faithfulness is measured by a deterministic three-signal checker (`ai/eval/faithfulness.py`):
source existence, numeric presence, and token overlap. It never calls an LLM in CI.

| Strategy | Faithfulness mean (core 79) | Faithfulness mean (full 104) |
|---|---|---|
| `rule_based_v1` | 0.87 | 0.88 |
| `keyword_score_v1` | 0.99 | 0.99 |

Source: `outputs/ai_compare_small.json`. Advisory — does not affect CI gate status.

## ADVISORY metrics: injection-block-rate (offline stub run)

The harness runs 25 injection probes (`ai/fixtures/injections.json`) through `GuardedStrategy`
wrapping `RuleBasedStrategy`. A probe is "blocked" when the input guardrail fires
(`route == guardrail:input_block`) or the output guardrail fires (`route == guardrail:output_reject`).

**Block rate: 76% (19/25)** — below the 95% advisory target.

The 6 unblocked probes are all in the `source_spoofing` and `ungrounded_coercion` categories
(injections `inj-006` through `inj-008`, `inj-013` through `inj-015`). These categories have
`expected_behavior: refused`, meaning the output guardrail is intended to catch them only after
the inner strategy runs. With the deterministic `RuleBasedStrategy` as the inner strategy
(offline stub), the strategy routes these normally and the output guardrail's fabricated-source
check would fire only when the output actually contains a fabricated source id. In practice the
deterministic strategy never emits a fabricated source, so the output guardrail has nothing to
reject. These probes would be caught when a real LLM emits a bad source. The 76% figure is stated
honestly and not rounded up.

Source: `outputs/ai_compare_small.json`. Advisory — does not affect CI gate status.

## LLM architectures — LIVE-LLM PENDING

Three LLM `AnswerStrategy` subclasses are implemented in `ai/agent/`. All share the grounding
pattern: the LLM routes to a whitelisted tool; the tool computes the answer; every answer cites
its source.

| Architecture | File | Control flow |
|---|---|---|
| `single_shot_rag` | `agent/single_shot_rag.py` | One routing call + optional grounded synthesis for retrieval questions. |
| `react` | `agent/react.py` | Bounded reason-act-observe loop (max 4 steps); LLM forced to call at least one tool before finishing. |
| `plan_execute` | `agent/plan_execute.py` | Phase 1: LLM plans a step list. Phase 2: validated tool execution. Phase 3: LLM synthesises from tool results. |

Orchestration is tested offline using an injectable stub LLM. In CI, when no `OPENAI_API_KEY` or
reachable `OLLAMA_HOST` is present, these architectures are listed as `skipped (no provider)` and
never cause CI failure.

**Head-to-head results (accuracy / faithfulness / cost / latency on the extended tier) are not
yet recorded here.** They require the owner to configure a live provider in `.env` and run
`make ai-compare-small`. No numbers will be presented until the owner completes that run.

## Retrieval

`ai/tools/retrieval.py` uses a two-tier backend:

- **Vector backend** (live environments): embeds the query with OpenAI `text-embedding-3-small`
  (1536 dim), runs a pgvector cosine nearest-neighbour query (`<=>` operator) over the C2
  `ai_embeddings` table (Postgres/PostGIS store), and maps result rows back to corpus docs for
  title and snippet. Requires `OPENAI_API_KEY` and a reachable Postgres instance.
- **Keyword fallback** (CI / offline): deterministic token-overlap scorer over the in-memory
  corpus. Used by default when either prerequisite is absent.

The corpus (`ai/fixtures/corpus/corpus.json`) contains **28 documents** in real formats: 12 METAR
strings, 6 aviation reference documents (squawk codes, weather codes, altimeter settings, flight
rules, wake turbulence, TCAS RA), and 10 NTSB-style incident reports.

## Guardrails

`ai/guardrails/` wraps any `AnswerStrategy` via `GuardedStrategy`:

1. **Input guardrail** (`input_guard.py`) — deterministic pattern matching over normalised text.
   Detects and blocks: instruction overrides, exfiltration attempts, roleplay jailbreaks, tool
   abuse, and routing hijacks. Fires before the inner strategy is called.
2. **Output guardrail** (`output_guard.py`) — checks the completed answer for empty sources,
   fabricated source ids, prompt-leak content, and unsafe-content markers. Fires after the inner
   strategy returns.

Both checks are fully deterministic and offline-safe. No LLM is involved.

## Observability (`ai/obs/`)

All three modules are offline-safe (availability-gated; no network at import time):

- **`tracing.py`** — per-call OTel spans reusing C6 conventions (`shared/obs/telemetry.py`).
  Span names: `ai.llm.call`, `ai.strategy.answer`. Attributes: `ai.strategy`, `ai.route`,
  `ai.tokens`, `ai.provider`. Degrades to no-ops when the OTel SDK is absent.
- **`costs.py`** — token-to-USD cost calculator. Prometheus counters `ai_llm_tokens_total` and
  `ai_llm_cost_usd_total` (labelled by model and strategy). Price table covers `gpt-4o-mini`,
  `gpt-4o`, `text-embedding-3-small`, and Ollama (local = $0). Degrades gracefully without
  `prometheus_client`.
- **`cache.py`** — `SemanticCache`: embedding-keyed (cosine similarity >= 0.92) with exact-match
  offline fallback. Bounded to 256 entries (FIFO eviction). Prometheus counter
  `ai_cache_hits_total` (labelled by keying mode). Fully offline in exact-match mode.

## Run

```bash
make ai-eval-fixtures   # validate golden set + tools against frozen fixtures (offline)
make ai-eval-small      # run the answer path, deterministic checks + citations
make ai-compare-small   # compare all strategies; LLM legs auto-skipped if no provider
make test-ai            # unit tests
```

To run the LLM architecture comparison, set `OPENAI_API_KEY` (or start an Ollama server and set
`OLLAMA_HOST`) in `.env`, then run `make ai-compare-small`. Results will appear in
`outputs/ai_compare_small.json`.

## Key files

- `tools/` — `analytics.py` (8 validated, parameterised ops over gold; no free-form SQL),
  `retrieval.py` (pgvector cosine NN with keyword fallback), `embeddings.py`
  (OpenAI `text-embedding-3-small`, 1536 dim, stdlib urllib only).
- `agent/` — `base.py` (`AnswerStrategy` / `Answer`), `rule_based.py`, `keyword_router.py`,
  `ollama_llm.py`, `single_shot_rag.py`, `react.py`, `plan_execute.py`.
- `eval/` — `golden_set.json` (104 questions), `checks.py`, `faithfulness.py`, `compare.py`
  (AUTHORITATIVE gate + ADVISORY metrics in one harness).
- `guardrails/` — `input_guard.py`, `output_guard.py`, `guarded_strategy.py`.
- `obs/` — `tracing.py`, `costs.py`, `cache.py`.
- `fixtures/` — frozen `gold/` snapshot, `corpus/corpus.json` (28 docs), `injections.json`
  (25-probe injection suite).

## Notes

The AUTHORITATIVE gate (core-tier accuracy + citation) has been the blocking CI check since
W3.1. W4.4 expanded the golden set from 13 to 104 questions, added three LLM architectures,
and promoted faithfulness and injection-block-rate to ADVISORY measured metrics. The head-to-head
LLM architecture comparison on the extended tier is pending the owner's live-provider run.
See [docs/research-findings.md](../docs/research-findings.md) for the full findings writeup.
