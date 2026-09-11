# AI architecture x provider eval matrix

Generated with: `make eval`
Cassettes recorded at: 2026-09-11.
Host: Apple M2, macOS, Python 3.14 in ~/.venvs/flight-telemetry.
Golden set has 104 questions; this run used 104.
1 question in the golden set is an exact duplicate and was served from the semantic cache on its second occurrence at zero provider cost.
The semantic cache ran in exact-match keying, so the live run and its replay follow the same code path. In embedding mode, five paraphrased questions in the golden set would also have been served from cache.
Retrieval backend actually used: keyword fallback.
Model (openai): openai:gpt-4o-mini.
Model (bedrock): bedrock:us.meta.llama3-1-8b-instruct-v1:0.
Price (openai), checked 2026-09-10: $0.000150/1K input tokens, $0.000600/1K output tokens.
Price (bedrock), checked 2026-09-10: $0.000220/1K input tokens, $0.000220/1K output tokens.
p50 latency covers provider call latency only; it excludes local tool and guardrail time.

| architecture | openai accuracy | openai citation rate | openai cost/query | openai p50 (s) | bedrock accuracy | bedrock citation rate | bedrock cost/query | bedrock p50 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_shot_rag | 0.942 | 0.971 | $0.000067 | 0.875 | 0.712 | 0.990 | $0.000079 | 0.374 |
| plan_execute | 0.827 | 0.913 | $0.000113 | 1.843 | 0.721 | 0.933 | $0.000125 | 0.777 |
| langgraph_plan_execute | 0.827 | 0.913 | $0.000113 | 1.843 | 0.721 | 0.933 | $0.000125 | 0.777 |

Reference: rule_based_v1 (no LLM) scores accuracy 0.904 and citation rate 1.000 over the same 104 questions, at $0.00 cost. No latency claim is made since this strategy never calls a provider.

## Injection probes

The input guard fires before any model call, so per-provider differences below can only come from the output guard.

| provider | probes | blocked at input | rejected at output | block rate |
| --- | --- | --- | --- | --- |
| openai | 25 | 19 | 2 | 0.840 |
| bedrock | 25 | 19 | 1 | 0.800 |
