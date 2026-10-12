"""Clients for model families not served by Ollama Cloud (L5: studies/L5_llm_families/protocol.md).

OpenRouterClient speaks the OpenAI-compatible chat API at https://openrouter.ai/api/v1. The key is read from the
environment variable OPENROUTER_API_KEY and is never printed, logged or written to a file.
Same interface as fishery_sim.llm_actor.OllamaClient: chat(messages, seed=None, json_mode=True) -> Reply.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from experiments.oversight import run_l2_llm_agents as l2
from fishery_sim import llm_actor as L

# What each closed model accepts, from OpenRouter's public model list (12 Oct 2026; L5 protocol "Sampling deviations").
SUPPORTS = {
    "anthropic/claude-sonnet-5.5": dict(temperature=False, seed=False),
    "openai/gpt-6.1-sol": dict(temperature=False, seed=True),
    "google/gemini-3.8-flash": dict(temperature=True, seed=True),
}


class MissingKey(RuntimeError):
    pass


class OpenRouterClient:
    url = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model, temperature=0.7, timeout=300, api_key=None, url=None):
        self.model, self.temperature, self.timeout = model, temperature, timeout
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise MissingKey("OPENROUTER_API_KEY is not set; closed families cannot run (L5 protocol)")
        self.url = url or self.url
        self.supports = SUPPORTS.get(model, dict(temperature=True, seed=True))

    def chat(self, messages, seed=None, json_mode=True):
        body = {"model": self.model, "messages": messages, "usage": {"include": True}}
        if self.supports["temperature"]:
            body["temperature"] = self.temperature
        if seed is not None and self.supports["seed"]:
            body["seed"] = int(seed) % (2 ** 31)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        t0, last = time.time(), None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                if "error" in data:  # OpenRouter can return an error object with HTTP 200
                    code = int(data["error"].get("code") or 0)
                    if code in (402, 429):
                        raise l2.QuotaStop(f"OpenRouter {code}: {str(data['error'].get('message'))[:200]}")
                    last = RuntimeError(str(data["error"])[:200])
                else:
                    break
            except urllib.error.HTTPError as exc:
                text = exc.read().decode(errors="ignore")[:200]
                if exc.code in (402, 429):  # no credit left, or rate limited: stop cleanly and resume later
                    raise l2.QuotaStop(f"HTTP {exc.code}: {text}")
                if exc.code in (400, 401, 403, 404):
                    raise RuntimeError(f"HTTP {exc.code}: {text}") from None
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            time.sleep(5 * (attempt + 1))
        else:
            raise l2.Transient(f"OpenRouter request failed after retries: {last}")
        msg = (data.get("choices") or [{}])[0].get("message", {})
        usage = data.get("usage") or {}
        reply = L.Reply(content=msg.get("content") or "", prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0), seconds=time.time() - t0,
                        model=data.get("model", self.model))
        reply.cost = float(usage.get("cost") or 0.0)  # dollars, as reported by OpenRouter
        return reply


def make_client(model):
    """Ollama cloud names end in ':cloud' or '-cloud'; anything with a '/' goes to OpenRouter."""
    return OpenRouterClient(model) if "/" in model else l2.Client(model)
