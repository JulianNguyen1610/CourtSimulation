"""LLM client interfaces, provider adapters, and deterministic test doubles."""

from __future__ import annotations

import json
import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from src.models import CaseProfile

LLMBackend = Literal["mock", "openai", "gemini", "local"]


class LLMClient(Protocol):
    """Minimal generation interface used by agents."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


@dataclass(frozen=True)
class LLMConfig:
    """Provider/model settings for one role or baseline method."""

    backend: LLMBackend = "mock"
    model: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 1024
    top_p: float = 0.95
    endpoint: str | None = None
    timeout: float | None = None


class MockLLM:
    """Deterministic mock LLM for local pipeline tests.

    The mock deliberately does not use the gold answer. It creates stable,
    evidence-grounded text so orchestration can be tested before API/model
    integration.
    """

    def generate(self, prompt: str) -> str:
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        if "Return valid JSON only" in prompt and '"charge"' in prompt:
            return (
                "{"
                f'"charge": "Trom cap tai san ({digest})", '
                '"articles": ["Dieu 173"], '
                '"sentence": "12 thang tu", '
                '"reasoning": "Mock LJP verdict dua tren tranh luan.", '
                '"confidence": 60, '
                '"cited_evidence_ids": ["cam-001"]'
                "}"
            )
        if "Return valid JSON only" in prompt and '"prediction"' in prompt:
            return (
                "{"
                f'"prediction": "Cần đối chiếu ngữ cảnh pháp luật ({digest})", '
                '"confidence": 55, '
                '"reasoning": "Mock judge dựa trên lập luận hai bên và chưa dùng đáp án vàng."'
                "}"
            )
        return (
            f"[mock-{digest}] Lập luận dựa trên ngữ cảnh được cung cấp; "
            "cần kiểm chứng bằng điều luật và bằng chứng truy xuất."
        )


class OpenAILLM:
    """OpenAI chat-completions client.

    The SDK import is intentionally lazy so offline unit tests do not need an
    OpenAI installation or API key.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = resolve_openai_api_key(api_key)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self._client = None

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the `openai` package. "
                "Install project requirements before using backend='openai'."
            ) from exc

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            top_p=self.top_p,
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
        raise RuntimeError("OpenAI returned an empty response.")


class GeminiLLM:
    """Gemini API client backed by the unified Google Gen AI SDK."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = api_key or resolve_gemini_api_key()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self._client = None

    def generate(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                top_p=self.top_p,
            ),
        )
        text = getattr(response, "text", None)
        if text:
            return text.strip()
        raise RuntimeError("Gemini returned an empty response.")


class LocalLLM:
    """OpenAI-compatible local HTTP endpoint client.

    Expected endpoint example: http://localhost:11434/v1/chat/completions
    API key is optional and can be supplied with LOCAL_LLM_API_KEY.
    """

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float = 0.95,
        timeout: float | None = None,
    ) -> None:
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "local-model")
        self.endpoint = endpoint or os.getenv("LOCAL_LLM_ENDPOINT")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.timeout = float(
            timeout if timeout is not None else os.getenv("LOCAL_LLM_TIMEOUT", "600")
        )
        if not self.endpoint:
            raise ValueError(
                "Local LLM endpoint not found. Set LOCAL_LLM_ENDPOINT or pass endpoint."
            )

    def generate(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "top_p": self.top_p,
        }
        reasoning_effort = os.getenv("LOCAL_LLM_REASONING_EFFORT", "none")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = _extract_openai_compatible_content(data)
        if content:
            return content.strip()
        raise RuntimeError("Local LLM endpoint returned an empty response.")


def _extract_openai_compatible_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            # Ollama thinking models (e.g. qwen3.5) may leave content empty and
            # place the trace in reasoning/thinking when thinking is enabled.
            for field in ("reasoning", "thinking"):
                value = message.get(field)
                if isinstance(value, str) and value.strip():
                    return value
        text = choices[0].get("text")
        if isinstance(text, str) and text.strip():
            return text
    text = data.get("text") or data.get("response")
    return text if isinstance(text, str) else ""


def resolve_openai_api_key(explicit_key: str | None = None) -> str:
    """Resolve OpenAI API key from explicit value or environment variables."""

    if explicit_key:
        return explicit_key
    value = os.getenv("OPENAI_API_KEY")
    if value:
        return value
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")


def resolve_gemini_api_key(explicit_key: str | None = None) -> str:
    """Resolve Gemini API key from explicit value or environment variables."""

    if explicit_key:
        return explicit_key
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    raise ValueError(
        "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
        "or pass --api-key."
    )


def create_llm_client(
    backend: LLMBackend = "mock",
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    top_p: float = 0.95,
    endpoint: str | None = None,
    timeout: float | None = None,
) -> LLMClient:
    """Create an LLM client for tests, hosted APIs, or local endpoints."""

    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAILLM(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
        )
    if backend == "gemini":
        return GeminiLLM(
            model=model,
            api_key=resolve_gemini_api_key(api_key),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
        )
    if backend == "local":
        return LocalLLM(
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported LLM backend: {backend}")


def create_llm_from_config(config: LLMConfig) -> LLMClient:
    """Create an LLM client from structured config."""

    return create_llm_client(
        backend=config.backend,
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        top_p=config.top_p,
        endpoint=config.endpoint,
        timeout=config.timeout,
    )


def create_role_llm_clients(role_configs: dict[str, LLMConfig]) -> dict[str, LLMClient]:
    """Create one client per configured role or baseline method."""

    return {role: create_llm_from_config(config) for role, config in role_configs.items()}


def llm_config_from_mapping(
    mapping: dict[str, Any] | None,
    *,
    default_backend: LLMBackend = "mock",
    role: str | None = None,
) -> LLMConfig:
    """Build LLMConfig from YAML/JSON-style mapping with env overrides."""

    mapping = mapping or {}
    env_prefix = f"{role.upper()}_" if role else ""
    backend = str(
        os.getenv(f"{env_prefix}LLM_BACKEND")
        or mapping.get("backend")
        or mapping.get("provider")
        or default_backend
    )
    if backend not in ("mock", "openai", "gemini", "local"):
        raise ValueError(f"Unsupported LLM backend in config: {backend}")

    model = os.getenv(f"{env_prefix}LLM_MODEL") or os.getenv("LLM_MODEL") or mapping.get("model")
    temperature = float(
        os.getenv(f"{env_prefix}LLM_TEMPERATURE")
        or os.getenv("LLM_TEMPERATURE")
        or mapping.get("temperature", 0.2)
    )
    max_output_tokens = int(
        os.getenv(f"{env_prefix}LLM_MAX_OUTPUT_TOKENS")
        or os.getenv("LLM_MAX_OUTPUT_TOKENS")
        or mapping.get("max_output_tokens", 1024)
    )
    top_p = float(
        os.getenv(f"{env_prefix}LLM_TOP_P")
        or os.getenv("LLM_TOP_P")
        or mapping.get("top_p", 0.95)
    )
    endpoint = os.getenv(f"{env_prefix}LLM_ENDPOINT") or os.getenv("LOCAL_LLM_ENDPOINT") or mapping.get("endpoint")
    timeout_value = os.getenv(f"{env_prefix}LLM_TIMEOUT") or os.getenv("LOCAL_LLM_TIMEOUT") or mapping.get("timeout")
    return LLMConfig(
        backend=backend,  # type: ignore[arg-type]
        model=str(model) if model else None,
        api_key=None,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        endpoint=str(endpoint) if endpoint else None,
        timeout=float(timeout_value) if timeout_value is not None else None,
    )


def is_mock_llm(client: LLMClient) -> bool:
    """Return true for deterministic test doubles."""

    return isinstance(client, MockLLM)


def extract_candidate_from_context(case: CaseProfile) -> str:
    """Extract a simple non-gold candidate answer from context for mock runs."""

    # Prefer short legal duration/amount spans often used in ViLQA answers.
    patterns = [
        r"\b\d{1,2}\s*năm\b",
        r"\b\d{1,2}\s*tháng\b",
        r"\b\d{1,3}\s*tuổi\b",
        r"\b\d{1,3}(?:[.,]\d{3})*\s*đồng\b",
    ]
    lowered_context = case.context.lower()
    for pattern in patterns:
        matches = re.findall(pattern, lowered_context, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1])
    return "Không xác định"
