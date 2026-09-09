#!/usr/bin/env python3
"""Offline unit tests for ai/providers.py (provider resolution and wire formats).

All tests run fully offline: urllib.request.urlopen and boto3.client are
mocked, socket.socket is patched to raise on every test so any accidental real
network attempt fails loudly, and os.environ is isolated per test via
patch.dict(..., clear=True).

Coverage:
  1. OpenAI adapter: request shape sent, content/usage/latency recorded.
  2. Bedrock adapter: request shape sent to a fake converse() client,
     content/usage/latency recorded.
  3. _bedrock_messages: system hoisting, same-role merging, assistant-first
     prepending, unknown-role dropping.
  4. Bedrock model id resolution: us. geo profile, bare id, BEDROCK_MODEL_ID
     override.
  5. Credential/config errors: missing env vars, unknown provider name,
     build_default() with nothing configured.
"""

import json
import os
import socket
import sys
import unittest
import unittest.mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai import providers


class _FakeHTTPResponse:
    """Minimal stand-in for the context-managed response urlopen() returns."""

    def __init__(self, body):
        self._data = json.dumps(body).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class NoSocketTestCase(unittest.TestCase):
    """Base class that fails any test attempting to open a real socket."""

    def setUp(self):
        super().setUp()
        patcher = unittest.mock.patch.object(
            socket, "socket",
            side_effect=AssertionError("test attempted to open a real socket"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)



class TestOpenAIAdapter(NoSocketTestCase):
    def test_returns_content_and_records_usage_and_latency(self):
        body = {
            "choices": [{"message": {"content": "the answer is 42"}}],
            "usage": {"prompt_tokens": 37, "completion_tokens": 9},
        }
        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["req"] = req
            return _FakeHTTPResponse(body)

        env = {"OPENAI_API_KEY": "sk-test-key", "OPENAI_MODEL": "gpt-4o-mini"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            provider = providers.build("openai", max_tokens=256)

        with unittest.mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            content, total_tokens = provider([{"role": "user", "content": "what is the answer?"}])

        self.assertEqual(content, "the answer is 42")
        self.assertEqual(total_tokens, 46)
        self.assertEqual(len(provider.calls), 1)
        call = provider.calls[0]
        self.assertEqual(call["input_tokens"], 37)
        self.assertEqual(call["output_tokens"], 9)
        self.assertGreaterEqual(call["latency_s"], 0)

        req = captured["req"]
        self.assertEqual(req.get_header("Authorization"), "Bearer sk-test-key")
        sent = json.loads(req.data)
        self.assertEqual(sent["model"], "gpt-4o-mini")
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(sent["max_tokens"], 256)



class TestBedrockAdapter(NoSocketTestCase):
    def test_returns_content_and_records_usage_and_latency(self):
        body = {
            "output": {"message": {"role": "assistant", "content": [{"text": "cleared for takeoff"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 51, "outputTokens": 14, "totalTokens": 65},
            "metrics": {"latencyMs": 812},
        }
        fake_client = unittest.mock.MagicMock()
        fake_client.converse.return_value = body

        env = {
            "AWS_ACCESS_KEY_ID": "AKIA-test",
            "AWS_SECRET_ACCESS_KEY": "secret-test",
            "AWS_REGION": "us-east-1",
        }
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            with unittest.mock.patch("boto3.client", return_value=fake_client) as mock_boto_client:
                provider = providers.build("bedrock", max_tokens=300)
                content, total_tokens = provider([
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "say hi"},
                ])

        self.assertEqual(content, "cleared for takeoff")
        self.assertEqual(total_tokens, 65)
        self.assertEqual(len(provider.calls), 1)
        call = provider.calls[0]
        self.assertEqual(call["input_tokens"], 51)
        self.assertEqual(call["output_tokens"], 14)
        self.assertGreaterEqual(call["latency_s"], 0)

        # no real client ever built
        mock_boto_client.assert_called_once()

        fake_client.converse.assert_called_once()
        _, kwargs = fake_client.converse.call_args
        self.assertEqual(kwargs["modelId"], "us.meta.llama3-1-8b-instruct-v1:0")
        self.assertEqual(kwargs["inferenceConfig"], {"temperature": 0, "maxTokens": 300})
        self.assertEqual(kwargs["system"], [{"text": "be concise"}])
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": [{"text": "say hi"}]}])



class TestBedrockMessageTranslation(NoSocketTestCase):
    def test_system_messages_hoisted_and_concatenated(self):
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "system", "content": "cite sources"},
            {"role": "user", "content": "hi"},
        ]
        system, turns = providers._bedrock_messages(messages)
        self.assertEqual(system, [{"text": "be terse\ncite sources"}])
        self.assertEqual(turns, [{"role": "user", "content": [{"text": "hi"}]}])

    def test_consecutive_same_role_turns_merged_with_newline(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        system, turns = providers._bedrock_messages(messages)
        self.assertEqual(system, [])
        self.assertEqual(turns, [
            {"role": "user", "content": [{"text": "a\nb"}]},
            {"role": "assistant", "content": [{"text": "c"}]},
        ])

    def test_assistant_first_conversation_gets_user_turn_prepended(self):
        messages = [{"role": "assistant", "content": "hi there"}]
        system, turns = providers._bedrock_messages(messages)
        self.assertEqual(system, [])
        self.assertEqual(turns, [
            {"role": "user", "content": [{"text": "(continue)"}]},
            {"role": "assistant", "content": [{"text": "hi there"}]},
        ])

    def test_unknown_roles_are_dropped(self):
        messages = [
            {"role": "tool", "content": "ignore me"},
            {"role": "user", "content": "hi"},
        ]
        system, turns = providers._bedrock_messages(messages)
        self.assertEqual(system, [])
        self.assertEqual(turns, [{"role": "user", "content": [{"text": "hi"}]}])



class TestBedrockModelIdResolution(NoSocketTestCase):
    def _build_with_region(self, region, model_id_override=None):
        env = {
            "AWS_ACCESS_KEY_ID": "AKIA-test",
            "AWS_SECRET_ACCESS_KEY": "secret-test",
            "AWS_REGION": region,
        }
        if model_id_override is not None:
            env["BEDROCK_MODEL_ID"] = model_id_override
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            with unittest.mock.patch("boto3.client", return_value=unittest.mock.MagicMock()):
                return providers.build("bedrock")

    def test_default_model_is_geo_profile_for_us_region(self):
        provider = self._build_with_region("us-east-1")
        self.assertEqual(provider.model, "us.meta.llama3-1-8b-instruct-v1:0")

    def test_default_model_is_bare_id_for_non_us_region(self):
        provider = self._build_with_region("eu-west-1")
        self.assertEqual(provider.model, "meta.llama3-1-8b-instruct-v1:0")

    def test_bedrock_model_id_env_overrides_default_in_us_region(self):
        provider = self._build_with_region("us-east-1", model_id_override="custom.model-x")
        self.assertEqual(provider.model, "custom.model-x")

    def test_bedrock_model_id_env_overrides_default_outside_us_region(self):
        provider = self._build_with_region("eu-west-1", model_id_override="custom.model-y")
        self.assertEqual(provider.model, "custom.model-y")



class TestCredentialErrors(NoSocketTestCase):
    def test_build_openai_without_api_key_names_the_missing_var(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                providers.build("openai")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_build_bedrock_without_aws_vars_names_all_of_them(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                providers.build("bedrock")
        message = str(ctx.exception)
        self.assertIn("AWS_ACCESS_KEY_ID", message)
        self.assertIn("AWS_SECRET_ACCESS_KEY", message)
        self.assertIn("AWS_REGION", message)

    def test_build_bedrock_names_only_the_vars_actually_missing(self):
        env = {"AWS_ACCESS_KEY_ID": "AKIA-test", "AWS_SECRET_ACCESS_KEY": "secret-test"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                providers.build("bedrock")
        message = str(ctx.exception)
        self.assertIn("AWS_REGION", message)
        self.assertNotIn("AWS_ACCESS_KEY_ID", message)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", message)

    def test_build_unknown_provider_names_the_valid_choices(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                providers.build("nope")
        message = str(ctx.exception)
        self.assertIn("bedrock", message)
        self.assertIn("ollama", message)
        self.assertIn("openai", message)

    def test_build_default_returns_none_when_nothing_configured(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with unittest.mock.patch("ai.agent.ollama_llm._list_models", return_value=[]):
                provider, label = providers.build_default()
        self.assertIsNone(provider)
        self.assertIsNone(label)


if __name__ == "__main__":
    unittest.main()
