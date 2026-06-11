"""Ask AI view — conversational question answering over aviation telemetry.

Routes the user's question through a C5 ``AnswerStrategy`` and renders:
  - the grounded ``answer_text``
  - an explicit citations panel (``sources``) showing which gold table or
    corpus document backed the answer

Default strategy: ``RuleBasedStrategy`` (deterministic, offline, no LLM keys).
LLM strategies (SingleShotRAG, ReAct, PlanExecute) appear in the selector only
when their ``is_available()`` check passes; otherwise they are listed with a
"(unavailable)" note and the selector falls back to deterministic.

This module MAY import streamlit.  Business logic that must be testable offline
lives in ``dashboard.agent_box`` (no streamlit import there).
"""

from __future__ import annotations

import streamlit as st

import dashboard.agent_box as agent_box

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
# Each entry: (display_label, builder_callable, is_llm)
# The builder is called at submission time only (lazy — never on import).


def _build_deterministic(question: str) -> agent_box.AnswerResult:
    return agent_box.answer_question(question)


def _make_llm_builder(strategy_cls):
    """Return a question -> AnswerResult closure for *strategy_cls*.

    Constructs the strategy with the default tools on each call.  Lightweight
    because LLM strategies build their LLM client lazily on the first answer()
    call.
    """
    def _builder(question: str) -> agent_box.AnswerResult:
        default = agent_box.build_default_strategy()
        llm_strategy = strategy_cls(
            analytics=default.analytics,
            retrieval=default.retrieval,
        )
        return agent_box.answer_question(question, strategy=llm_strategy)
    return _builder


def _build_strategy_options() -> list[dict]:
    """Return the list of selectable strategy options.

    Each dict has:
      - label      : human-readable name shown in the selector
      - key        : stable string key used for st.session_state
      - available  : bool
      - builder    : callable(question) -> AnswerResult
      - note       : extra note shown when unavailable
    """
    options: list[dict] = [
        {
            "label": "Deterministic (offline)",
            "key": "rule_based",
            "available": True,
            "builder": _build_deterministic,
            "note": "",
        }
    ]

    # Import LLM strategies — they are always importable (no network at import
    # time), but is_available() probes environment/network and may return False.
    try:
        from ai.agent.single_shot_rag import SingleShotRAGStrategy  # noqa: PLC0415
        from ai.agent.react import ReActStrategy  # noqa: PLC0415
        from ai.agent.plan_execute import PlanExecuteStrategy  # noqa: PLC0415

        llm_available = SingleShotRAGStrategy.is_available()

        for cls, label in [
            (SingleShotRAGStrategy, "Single-Shot RAG (LLM)"),
            (ReActStrategy, "ReAct (LLM)"),
            (PlanExecuteStrategy, "Plan-Execute (LLM)"),
        ]:
            options.append(
                {
                    "label": label,
                    "key": cls.name,
                    "available": llm_available,
                    "builder": _make_llm_builder(cls) if llm_available else None,
                    "note": "Requires OPENAI_API_KEY or OLLAMA_HOST" if not llm_available else "",
                }
            )
    except ImportError:
        pass  # ai.agent not importable in some minimal installs — silently skip

    return options


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Ask AI view."""
    st.header("Ask AI")
    st.caption(
        "Ask a natural-language question about the flight telemetry data. "
        "Every answer cites the source(s) it was derived from."
    )

    st.divider()

    # ---- Strategy selector ----
    strategy_options = _build_strategy_options()

    st.subheader("Strategy")
    strategy_labels = []
    for opt in strategy_options:
        if opt["available"]:
            strategy_labels.append(opt["label"])
        else:
            strategy_labels.append(f"{opt['label']} (unavailable)")

    selected_label = st.radio(
        "Answer strategy",
        strategy_labels,
        index=0,
        label_visibility="collapsed",
        help=(
            "Deterministic: fully offline, no LLM keys required.  "
            "LLM architectures require OPENAI_API_KEY or a running Ollama instance."
        ),
    )

    # Map selected label back to option dict
    selected_opt = strategy_options[0]  # safe default
    for opt in strategy_options:
        display = opt["label"] if opt["available"] else f"{opt['label']} (unavailable)"
        if display == selected_label:
            selected_opt = opt
            break

    if not selected_opt["available"]:
        st.info(
            f"**{selected_opt['label']}** is not available "
            f"({selected_opt['note']}). "
            "Falling back to the deterministic strategy."
        )
        selected_opt = strategy_options[0]  # deterministic fallback

    st.divider()

    # ---- Question input ----
    st.subheader("Your question")

    example_questions = [
        "How many emergency squawk events were there?",
        "Which aircraft squawked 7700?",
        "How many flights were tracked?",
        "What does METAR mean?",
        "How many active airports are in the dataset?",
        "What was the highest altitude flight?",
    ]

    with st.expander("Example questions", expanded=False):
        for q in example_questions:
            st.markdown(f"- {q}")

    question = st.text_input(
        "Ask a question about the telemetry data",
        placeholder="e.g. How many emergency squawk events were there?",
        key="ask_ai_question_input",
        label_visibility="collapsed",
    )

    submit = st.button("Ask", type="primary", disabled=not bool(question and question.strip()))

    # ---- Answer rendering ----
    if submit and question and question.strip():
        with st.spinner(f"Thinking with {selected_opt['label']}…"):
            try:
                result = selected_opt["builder"](question.strip())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Strategy raised an error: {exc}")
                st.info("Retrying with the deterministic fallback…")
                try:
                    result = _build_deterministic(question.strip())
                except Exception as fallback_exc:  # noqa: BLE001
                    st.error(f"Deterministic fallback also failed: {fallback_exc}")
                    return

        st.divider()

        # Answer body
        st.subheader("Answer")
        st.markdown(result.answer_text)

        # Citations
        st.subheader("Citations")
        if result.sources:
            for src in result.sources:
                st.markdown(f"- `{src}`")
        else:
            st.warning("No citations returned for this answer.")

        # Metadata expander
        with st.expander("Answer details", expanded=False):
            st.markdown(f"**Route:** `{result.route}`")
            st.markdown(f"**Strategy:** `{result.strategy}`")
            st.markdown(f"**Question:** {question!r}")
