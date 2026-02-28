from __future__ import annotations

import json
import os
from urllib import error as urlerror
from urllib import request as urlrequest
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class PolicyJSON:
    """
    Canonical JSON schema used to define executable threshold policies.
    """

    rationale: str
    low_stock_threshold: float
    high_stock_threshold: float
    low_harvest_frac: float
    mid_harvest_frac: float
    high_harvest_frac: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "low_stock_threshold": float(self.low_stock_threshold),
            "high_stock_threshold": float(self.high_stock_threshold),
            "low_harvest_frac": float(self.low_harvest_frac),
            "mid_harvest_frac": float(self.mid_harvest_frac),
            "high_harvest_frac": float(self.high_harvest_frac),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PolicyJSON":
        required = [
            "rationale",
            "low_stock_threshold",
            "high_stock_threshold",
            "low_harvest_frac",
            "mid_harvest_frac",
            "high_harvest_frac",
        ]
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"Missing policy keys: {missing}")
        return cls(
            rationale=str(raw["rationale"]),
            low_stock_threshold=float(raw["low_stock_threshold"]),
            high_stock_threshold=float(raw["high_stock_threshold"]),
            low_harvest_frac=float(raw["low_harvest_frac"]),
            mid_harvest_frac=float(raw["mid_harvest_frac"]),
            high_harvest_frac=float(raw["high_harvest_frac"]),
        )


def clamp_policy(policy: PolicyJSON, stock_max: float) -> PolicyJSON:
    low_stock_threshold = max(0.0, min(stock_max - 1.0, policy.low_stock_threshold))
    high_stock_threshold = max(low_stock_threshold + 1.0, min(stock_max, policy.high_stock_threshold))
    return PolicyJSON(
        rationale=policy.rationale,
        low_stock_threshold=low_stock_threshold,
        high_stock_threshold=high_stock_threshold,
        low_harvest_frac=max(0.0, min(1.0, policy.low_harvest_frac)),
        mid_harvest_frac=max(0.0, min(1.0, policy.mid_harvest_frac)),
        high_harvest_frac=max(0.0, min(1.0, policy.high_harvest_frac)),
    )


def build_policy_prompt(
    parent_policy: PolicyJSON,
    parent_fitness: float,
    stock_max: float,
    adversarial_pressure: float,
) -> str:
    """
    Prompt template for turning strategy selection context into a JSON policy proposal.
    """
    target_style = "aggressive invader" if adversarial_pressure >= 0.5 else "balanced contender"
    schema = {
        "rationale": "short explanation (<=25 words)",
        "low_stock_threshold": "float in [0, stock_max]",
        "high_stock_threshold": "float in [low_stock_threshold, stock_max]",
        "low_harvest_frac": "float in [0,1]",
        "mid_harvest_frac": "float in [0,1]",
        "high_harvest_frac": "float in [0,1]",
    }
    prompt = (
        "You are generating a fishery strategy as strict JSON.\n"
        f"Parent fitness: {parent_fitness:.4f}\n"
        f"Stock max: {stock_max:.4f}\n"
        f"Target style: {target_style}\n"
        "Parent policy:\n"
        f"{json.dumps(parent_policy.to_dict(), indent=2)}\n\n"
        "Output exactly one JSON object using this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "No markdown. No extra keys."
    )
    return prompt


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Parse the first JSON object found in a string. Supports raw JSON or fenced text.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in model output.")
    return json.loads(text[start : end + 1])


class PolicyLLMClient(Protocol):
    """
    Minimal interface so this module does not hard-bind to a specific provider SDK.
    """

    def complete(self, prompt: str) -> str: ...


class NullPolicyLLMClient:
    """
    Deterministic offline stub used when no external LLM is configured.
    It mirrors the parent policy, enabling adapter-path testing without API calls.
    """

    def complete(self, prompt: str) -> str:
        del prompt
        return json.dumps(
            {
                "rationale": "offline stub mirrors parent policy",
                "low_stock_threshold": 40.0,
                "high_stock_threshold": 130.0,
                "low_harvest_frac": 0.08,
                "mid_harvest_frac": 0.45,
                "high_harvest_frac": 0.75,
            }
        )


class OpenAIResponsesPolicyLLMClient:
    """
    Live OpenAI Responses API client implementing PolicyLLMClient.
    Expects an API key via argument or environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 45.0,
        temperature: float = 0.8,
        max_output_tokens: int = 300,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = float(max(1.0, timeout_s))
        self.temperature = float(max(0.0, temperature))
        self.max_output_tokens = int(max(64, max_output_tokens))

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not provided. Set env var OPENAI_API_KEY or pass api_key."
            )

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url=f"{self.base_url}/responses",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"OpenAI API HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {e}") from e

        data = json.loads(raw)
        text = self._extract_text(data)
        if not text:
            raise RuntimeError("OpenAI response contained no text output.")
        return text

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = data.get("output", [])
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            if chunks:
                return "\n".join(chunks)

        choices = data.get("choices", [])
        if isinstance(choices, list) and choices:
            try:
                text = choices[0]["message"]["content"]
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception:
                pass
        return ""


class OllamaPolicyLLMClient:
    """
    Local Ollama client implementing PolicyLLMClient (no paid API key required).
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        base_url: str | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.8,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout_s = float(max(1.0, timeout_s))
        self.temperature = float(max(0.0, temperature))

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url=f"{self.base_url}/api/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Ollama API HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(
                f"Ollama request failed (is Ollama running at {self.base_url}?): {e}"
            ) from e

        data = json.loads(raw)
        text = data.get("response", "")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama response contained no text.")
        return text.strip()


class FileReplayPolicyLLMClient:
    """
    Offline deterministic client that replays JSON outputs from disk.
    Accepted file formats:
    - JSON object
    - JSON list of objects
    - JSONL with one object per line
    """

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        if not raw_text:
            raise ValueError(f"Policy replay file is empty: {path}")
        self._items = self._load_items(raw_text)
        self._index = 0

    @staticmethod
    def _load_items(raw_text: str) -> list[str]:
        try:
            loaded = json.loads(raw_text)
            if isinstance(loaded, dict):
                return [json.dumps(loaded)]
            if isinstance(loaded, list):
                return [json.dumps(item) for item in loaded]
        except Exception:
            pass
        items = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not items:
            raise ValueError("Could not parse replay items from file.")
        return items

    def complete(self, prompt: str) -> str:
        del prompt
        item = self._items[self._index % len(self._items)]
        self._index += 1
        return item
