#!/usr/bin/env python3
"""LLM provider resolution and wire formats shared by all answer strategies.

A :class:`Provider` is a callable ``(messages) -> (content, total_tokens)`` with
the exact signature the strategies already inject via their ``llm`` constructor
argument, plus bookkeeping (``name``, ``model``, ``label``, ``calls``) used for
cost/latency reporting. :func:`build` constructs one provider for a given name
("openai", "bedrock", "ollama") from environment variables, raising
``RuntimeError`` when required credentials are missing. :func:`build_default`
reproduces the strategies' historical zero-config resolution order (OpenAI if
configured, else a reachable Ollama server, else unavailable) without opening
any socket at import time.
"""

import json
import os
import time
import urllib.request


class Provider:
    """A callable LLM client plus per-call bookkeeping.

    Instances are returned by :func:`build`/:func:`build_default`; strategies
    call them as ``content, tokens = provider(messages)``.
    """

    def __init__(self, name, model, fn):
        self.name = name
        self.model = model
        self.label = f"{name}:{model}"
        self.calls = []
        self._fn = fn

    def __call__(self, messages):
        start = time.perf_counter()
        content, input_tokens, output_tokens = self._fn(messages)
        latency_s = time.perf_counter() - start
        self.calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": latency_s,
        })
        return content, input_tokens + output_tokens



def _openai_chat(messages, api_key, model, max_tokens, timeout=60):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _build_openai(max_tokens):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI provider requires OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _fn(messages):
        return _openai_chat(messages, api_key, model, max_tokens)

    return Provider("openai", model, _fn)



def _ollama_chat(messages, host, model, max_tokens, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    content = resp.get("message", {}).get("content", "")
    return content, resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)


def _build_ollama(max_tokens):
    from ai.agent.ollama_llm import _list_models, pick_model

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL") or pick_model(_list_models(host, timeout=2))
    if not model:
        raise RuntimeError(
            "Ollama provider requires OLLAMA_MODEL or a reachable OLLAMA_HOST with "
            "at least one installed model"
        )

    def _fn(messages):
        return _ollama_chat(messages, host, model, max_tokens)

    return Provider("ollama", model, _fn)



def _bedrock_messages(messages):
    # converse wants system text as its own argument, only user/assistant turns,
    # no two consecutive turns with the same role, and a user turn first
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]

    merged = []
    for m in turns:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append({"role": m["role"], "content": m["content"]})
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "(continue)"})

    system = [{"text": "\n".join(system_parts)}] if system_parts else []
    bedrock_messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]} for m in merged
    ]
    return system, bedrock_messages


def _bedrock_chat(client, model_id, messages, max_tokens):
    system, bedrock_messages = _bedrock_messages(messages)
    kwargs = {
        "modelId": model_id,
        "messages": bedrock_messages,
        "inferenceConfig": {"temperature": 0, "maxTokens": max_tokens},
    }
    if system:
        kwargs["system"] = system
    resp = client.converse(**kwargs)
    content = resp["output"]["message"]["content"][0]["text"]
    usage = resp["usage"]
    return content, usage["inputTokens"], usage["outputTokens"]


def _build_bedrock(max_tokens):
    missing = [
        var for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        if not os.environ.get(var)
    ]
    if missing:
        raise RuntimeError(f"Bedrock provider requires {', '.join(missing)}")

    import boto3
    from botocore.config import Config

    region = os.environ["AWS_REGION"]
    # llama 3.1 8b has no in-region endpoint in us-east-1, only the geo inference profile.
    default_model = "us.meta.llama3-1-8b-instruct-v1:0" if region.startswith("us-") \
        else "meta.llama3-1-8b-instruct-v1:0"
    model_id = os.environ.get("BEDROCK_MODEL_ID", default_model)

    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    def _fn(messages):
        return _bedrock_chat(client, model_id, messages, max_tokens)

    return Provider("bedrock", model_id, _fn)



_BUILDERS = {
    "openai": _build_openai,
    "bedrock": _build_bedrock,
    "ollama": _build_ollama,
}


def build(name, max_tokens=512):
    """Build a :class:`Provider` for ``name`` ("openai", "bedrock", "ollama").

    Raises ``RuntimeError`` naming the missing environment variables when
    required credentials are absent.
    """
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise RuntimeError(f"unknown provider {name!r}; choose one of {sorted(_BUILDERS)}")
    return builder(max_tokens)


def build_default(max_tokens=512):
    """Resolve the default provider from the environment.

    Order: OpenAI if OPENAI_API_KEY is set, else a reachable Ollama server with
    an installed model, else (None, None). No sockets are opened at import
    time; the Ollama probe uses a short timeout and only ever touches the
    configured host.
    """
    if os.environ.get("OPENAI_API_KEY", ""):
        provider = _build_openai(max_tokens)
        return provider, provider.label
    try:
        provider = _build_ollama(max_tokens)
        return provider, provider.label
    except Exception:
        return None, None
