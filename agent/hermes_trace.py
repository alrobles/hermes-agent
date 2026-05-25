"""Lightweight per-request trace accumulator for ``hermes_trace`` telemetry.

Opt-in via ``"hermes": {"trace": true}`` in the request body.  The gateway
attaches the serialized trace to the response (non-streaming JSON field or
``event: hermes.trace`` SSE event before ``[DONE]``).

Design constraints:
  - No PII: tool *names* only, never arguments or results.
  - Bounded: ``llm_calls`` and ``tool_calls`` capped at ``MAX_ENTRIES``.
  - Zero overhead when tracing is off (the object is never created).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MAX_ENTRIES = 50


@dataclass
class LLMCallTrace:
    """One upstream LLM API call."""
    iteration: int = 0
    upstream: str = ""
    model: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: Optional[int] = None
    total_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "iter": self.iteration,
            "upstream": self.upstream,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "total_ms": self.total_ms,
        }
        if self.ttft_ms is not None:
            d["ttft_ms"] = self.ttft_ms
        return d


@dataclass
class ToolCallTrace:
    """One tool invocation."""
    name: str = ""
    duration_ms: int = 0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }


@dataclass
class HermesTrace:
    """Accumulates telemetry for one ``run_conversation`` turn."""
    model: str = ""
    _llm_calls: List[LLMCallTrace] = field(default_factory=list)
    _tool_calls: List[ToolCallTrace] = field(default_factory=list)
    _truncated: bool = False
    _turn_start: float = field(default_factory=time.monotonic)
    system_prompt_tokens: int = 0
    session_id: Optional[str] = None

    # ---- recording helpers ----

    def record_llm_call(
        self,
        iteration: int,
        upstream: str,
        model: str,
        prompt_tokens: int = 0,
        cached_tokens: int = 0,
        completion_tokens: int = 0,
        ttft_ms: Optional[int] = None,
        total_ms: int = 0,
    ) -> None:
        if len(self._llm_calls) >= MAX_ENTRIES:
            self._truncated = True
            return
        self._llm_calls.append(LLMCallTrace(
            iteration=iteration,
            upstream=upstream,
            model=model,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
        ))

    def record_tool_call(
        self,
        name: str,
        duration_ms: int = 0,
        status: str = "ok",
    ) -> None:
        if len(self._tool_calls) >= MAX_ENTRIES:
            self._truncated = True
            return
        self._tool_calls.append(ToolCallTrace(
            name=name,
            duration_ms=duration_ms,
            status=status,
        ))

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        total_ms = int((time.monotonic() - self._turn_start) * 1000)
        result: Dict[str, Any] = {
            "model": self.model,
            "agent_loop": {
                "iterations": max(
                    (c.iteration for c in self._llm_calls), default=0
                ),
                "llm_calls": [c.to_dict() for c in self._llm_calls],
                "tool_calls": [c.to_dict() for c in self._tool_calls],
                "total_ms": total_ms,
            },
            "gateway": {},
        }
        if self._truncated:
            result["agent_loop"]["truncated"] = True
        if self.system_prompt_tokens:
            result["gateway"]["system_prompt_tokens"] = self.system_prompt_tokens
        if self.session_id:
            result["gateway"]["session_id"] = self.session_id
        return result
