# AI Layer — eval-driven operational agent

**What this proves:** the headline is the **evaluation harness**, not a chat demo. Answers are
grounded (the model only routes; validated tools compute), checkable (deterministic checks +
required citations), and three answer strategies are compared head-to-head on one golden set.

## Metrics (measured, 13-question golden set)

| Strategy | Accuracy | Citation coverage | p50 latency | Tokens |
|---|---|---|---|---|
| `rule_based_v1` (deterministic) | **100%** | 100% | 0.01 ms | 0 |
| `keyword_score_v1` (deterministic) | **100%** | 100% | 0.01 ms | 0 |
| `ollama:qwen2.5-coder` (local 7B LLM) | 92.3% | 100% | 2746 ms | 4136 |

**Finding:** on this bounded operational set, deterministic structured-tool routing dominates —
the LLM lost one question while costing ~270,000× the latency and 4k tokens. The grounding pattern
("LLM routes, tools compute") keeps every strategy at 100% citation coverage. Full analysis in
[docs/research-findings.md](../docs/research-findings.md).

## Run

```bash
make ai-eval-fixtures   # validate golden set + tools against frozen fixtures (offline)
make ai-eval-small      # run the answer path, deterministic checks + citations
make ai-compare-small   # compare all strategies (Ollama leg auto-skipped if unavailable)
make test-ai            # unit tests
```

## Key files

- `tools/` — `analytics.py` (8 validated, parameterized ops over gold; no free-form SQL),
  `retrieval.py` (offline keyword search over the corpus).
- `agent/` — `base.py` (`AnswerStrategy`), `rule_based.py`, `keyword_router.py`, `ollama_llm.py`.
- `eval/` — `golden_set.json`, `checks.py`, `run_fixtures.py`, `run_answers.py`, `compare.py`.
- `fixtures/` — frozen `gold/` snapshot + `corpus/corpus.json` (METAR, squawk reference, synthetic reports).

## Notes

W3.1 onward kept the eval offline/no-LLM by design; the Ollama strategy is optional and
availability-gated. The golden set is small — expanding it with paraphrase/long-tail questions
(where routing is genuinely ambiguous) is a tracked stretch item. See ADR-0008.
