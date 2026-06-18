"""Tests for LLM client factory and Gemini key resolution."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from src.llm import (
    GeminiLLM,
    MockLLM,
    _extract_openai_compatible_content,
    create_llm_client,
    resolve_gemini_api_key,
)


class LLMClientFactoryTest(unittest.TestCase):
    def test_create_mock_client(self) -> None:
        client = create_llm_client("mock")
        self.assertIsInstance(client, MockLLM)

    def test_mock_client_is_deterministic(self) -> None:
        client = create_llm_client("mock")
        first = client.generate("prompt")
        second = client.generate("prompt")
        self.assertEqual(first, second)

    def test_resolve_api_key_from_env(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            self.assertEqual(resolve_gemini_api_key(), "test-key")

    def test_resolve_api_key_prefers_explicit_value(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
            self.assertEqual(resolve_gemini_api_key("explicit-key"), "explicit-key")

    def test_resolve_api_key_raises_when_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_gemini_api_key()

    @patch("google.genai.Client")
    def test_gemini_client_returns_text(self, mock_client_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = "  Gemini answer  "
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = GeminiLLM(api_key="test-key")
        output = client.generate("prompt")

        self.assertEqual(output, "Gemini answer")
        mock_client.models.generate_content.assert_called_once()

    def test_extract_content_prefers_message_content(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                        "reasoning": "internal trace",
                    }
                }
            ]
        }
        self.assertEqual(_extract_openai_compatible_content(data), "final answer")

    def test_extract_content_falls_back_to_reasoning(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "trace with answer",
                    }
                }
            ]
        }
        self.assertEqual(
            _extract_openai_compatible_content(data),
            "trace with answer",
        )


if __name__ == "__main__":
    unittest.main()
