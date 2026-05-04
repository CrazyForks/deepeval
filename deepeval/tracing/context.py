from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional
from contextvars import ContextVar

from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
    Trace,
)
from deepeval.test_case.llm_test_case import ToolCall, LLMTestCase
from deepeval.prompt.prompt import Prompt


class SpanContext:
    def __init__(self):
        self.current_span: ContextVar[Optional[BaseSpan]] = ContextVar(
            "current_span", default=None
        )

    def get(self):
        return self.current_span.get()

    def set(self, value):
        return self.current_span.set(value)

    def reset(self, value):
        return self.current_span.reset(value)

    def drop(self):
        span = self.current_span.get()
        if span:
            span.drop = True


class TraceContext:
    def __init__(self):
        self.current_trace: ContextVar[Optional[Trace]] = ContextVar(
            "current_trace", default=None
        )

    def get(self):
        return self.current_trace.get()

    def set(self, value):
        return self.current_trace.set(value)

    def reset(self, value):
        return self.current_trace.reset(value)

    def drop(self):
        trace = self.current_trace.get()
        if trace:
            trace.drop = True


current_span_context = SpanContext()
current_trace_context = TraceContext()


def update_current_span(
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    test_case: Optional[LLMTestCase] = None,
    metric_collection: Optional[str] = None,
):
    current_span = current_span_context.get()
    if not current_span:
        return
    if test_case:

        current_span.input = test_case.input
        current_span.output = test_case.actual_output
        current_span.expected_output = test_case.expected_output
        current_span.retrieval_context = test_case.retrieval_context
        current_span.context = test_case.context
        current_span.tools_called = test_case.tools_called
        current_span.expected_tools = test_case.expected_tools
    if metadata:
        current_span.metadata = metadata
    if input:
        current_span.input = input
    if output:
        current_span.output = output
    if retrieval_context:
        current_span.retrieval_context = retrieval_context
    if context:
        current_span.context = context
    if expected_output:
        current_span.expected_output = expected_output
    if tools_called:
        current_span.tools_called = tools_called
    if expected_tools:
        current_span.expected_tools = expected_tools
    if name:
        current_span.name = name
    if metric_collection:
        current_span.metric_collection = metric_collection


def update_current_trace(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    test_case: Optional[LLMTestCase] = None,
    confident_api_key: Optional[str] = None,
    test_case_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    metric_collection: Optional[str] = None,
):
    current_trace = current_trace_context.get()
    if not current_trace:
        return
    if test_case:
        current_trace.input = test_case.input
        current_trace.output = test_case.actual_output
        current_trace.expected_output = test_case.expected_output
        current_trace.retrieval_context = test_case.retrieval_context
        current_trace.context = test_case.context
        current_trace.tools_called = test_case.tools_called
        current_trace.expected_tools = test_case.expected_tools
    if name:
        current_trace.name = name
    if tags:
        current_trace.tags = tags
    if metadata:
        current_trace.metadata = metadata
    if thread_id:
        current_trace.thread_id = thread_id
    if user_id:
        current_trace.user_id = user_id
    if input:
        current_trace.input = input
    if output:
        current_trace.output = output
    if retrieval_context:
        current_trace.retrieval_context = retrieval_context
    if context:
        current_trace.context = context
    if expected_output:
        current_trace.expected_output = expected_output
    if tools_called:
        current_trace.tools_called = tools_called
    if expected_tools:
        current_trace.expected_tools = expected_tools
    if confident_api_key:
        current_trace.confident_api_key = confident_api_key
    if test_case_id:
        current_trace.test_case_id = test_case_id
    if turn_id:
        current_trace.turn_id = turn_id
    if metric_collection:
        current_trace.metric_collection = metric_collection


def update_llm_span(
    model: Optional[str] = None,
    input_token_count: Optional[float] = None,
    output_token_count: Optional[float] = None,
    cost_per_input_token: Optional[float] = None,
    cost_per_output_token: Optional[float] = None,
    token_intervals: Optional[Dict[float, str]] = None,
    prompt: Optional[Prompt] = None,
):
    current_span = current_span_context.get()
    if not current_span or not isinstance(current_span, LlmSpan):
        return
    if model:
        current_span.model = model
    if input_token_count:
        current_span.input_token_count = input_token_count
    if output_token_count:
        current_span.output_token_count = output_token_count
    if cost_per_input_token:
        current_span.cost_per_input_token = cost_per_input_token
    if cost_per_output_token:
        current_span.cost_per_output_token = cost_per_output_token
    if token_intervals:
        current_span.token_intervals = token_intervals
    if prompt:
        current_span.prompt = prompt
        # Updating on span as well
        current_span.prompt_alias = prompt.alias
        current_span.prompt_commit_hash = prompt.hash
        current_span.prompt_label = prompt.label
        current_span.prompt_version = prompt.version


def update_agent_span(
    available_tools: Optional[List[str]] = None,
    agent_handoffs: Optional[List[str]] = None,
):
    """Mutate the active ``AgentSpan`` with agent-specific fields.

    Type-specific counterpart to ``update_current_span(...)``: only
    handles fields unique to ``AgentSpan``. Generic fields (name,
    metadata, metric_collection, input/output, ...) still go through
    ``update_current_span(...)``. No-op if the current span isn't an
    ``AgentSpan``.
    """
    current_span = current_span_context.get()
    if not current_span or not isinstance(current_span, AgentSpan):
        return
    if available_tools is not None:
        current_span.available_tools = available_tools
    if agent_handoffs is not None:
        current_span.agent_handoffs = agent_handoffs


def update_tool_span(
    description: Optional[str] = None,
):
    """Mutate the active ``ToolSpan`` with tool-specific fields.

    Type-specific counterpart to ``update_current_span(...)``: only
    handles fields unique to ``ToolSpan``. ``ToolSpan.name`` is set at
    span creation; use ``update_current_span(name=...)`` to rename
    after the fact. No-op if the current span isn't a ``ToolSpan``.
    """
    current_span = current_span_context.get()
    if not current_span or not isinstance(current_span, ToolSpan):
        return
    if description is not None:
        current_span.description = description


def update_retriever_span(
    embedder: Optional[str] = None,
    top_k: Optional[int] = None,
    chunk_size: Optional[int] = None,
):
    current_span = current_span_context.get()
    if not current_span or not isinstance(current_span, RetrieverSpan):
        return
    if embedder:
        current_span.embedder = embedder
    if top_k:
        current_span.top_k = top_k
    if chunk_size:
        current_span.chunk_size = chunk_size


# ---------------------------------------------------------------------------
# next_*_span: declarative defaults for the NEXT span of a given type.
#
# Counterpart to ``update_current_*_span(...)`` for spans without a
# user-code seam — i.e. spans the user never executes code inside, so
# ``update_current_*_span`` from "their" body isn't reachable. The
# canonical case is an integration-emitted agent / LLM span where the
# only callsite the user owns is the one wrapping the framework call.
#
# Semantics:
#   - One-shot: the dict is consumed by the FIRST span of the matching
#     type that the consumer (typically an integration's OTel processor)
#     creates inside the active scope. Subsequent spans see an empty slot.
#   - Per-type isolation: each ``next_*_span`` writes to its own
#     ``ContextVar``, so stacking ``with next_agent_span(...),
#     next_llm_span(...):`` is safe and unambiguous.
#   - One-stop kwargs: each helper accepts BASE fields (everything
#     ``update_current_span`` takes) AND its type-specific fields in a
#     single call. Diverges intentionally from the
#     ``update_*_span`` family (which is decomposed) — see commit msg.
#   - Consumer responsibility: integrations call ``_pop_pending_*(...)``
#     when classifying a fresh span and apply the dict to the placeholder
#     they push onto ``current_span_context``. If no integration is
#     listening the dict is silently discarded on ``with`` exit.
# ---------------------------------------------------------------------------


_pending_next_span: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "pending_next_span", default=None
)
_pending_next_agent_span: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "pending_next_agent_span", default=None
)
_pending_next_llm_span: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "pending_next_llm_span", default=None
)
_pending_next_tool_span: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "pending_next_tool_span", default=None
)
_pending_next_retriever_span: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "pending_next_retriever_span", default=None
)


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Strip keys whose value is None — keeps the pending dict tight so
    consumers don't have to re-check every kwarg they passed through."""
    return {k: v for k, v in d.items() if v is not None}


# --- base: applies to the next span of ANY type ----------------------------


@contextmanager
def next_span(
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    test_case: Optional[LLMTestCase] = None,
    metric_collection: Optional[str] = None,
) -> Iterator[None]:
    """Set base-span defaults for the next span of any type.

    Mirrors ``update_current_span(...)`` kwargs. Use when the type of
    the upcoming span doesn't matter or isn't known. For a typed match,
    use ``next_agent_span`` / ``next_llm_span`` / ``next_tool_span`` /
    ``next_retriever_span``.
    """
    payload = _drop_none(
        {
            "input": input,
            "output": output,
            "retrieval_context": retrieval_context,
            "context": context,
            "expected_output": expected_output,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "metadata": metadata,
            "name": name,
            "test_case": test_case,
            "metric_collection": metric_collection,
        }
    )
    token = _pending_next_span.set(payload)
    try:
        yield
    finally:
        _pending_next_span.reset(token)


# --- agent: base + agent-specific (one-stop) -------------------------------


@contextmanager
def next_agent_span(
    available_tools: Optional[List[str]] = None,
    agent_handoffs: Optional[List[str]] = None,
    # base fields (mirror update_current_span)
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    test_case: Optional[LLMTestCase] = None,
    metric_collection: Optional[str] = None,
) -> Iterator[None]:
    """Set defaults for the next ``AgentSpan``. One-stop: accepts
    agent-specific fields (``available_tools``, ``agent_handoffs``) AND
    the same base fields ``update_current_span(...)`` takes."""
    payload = _drop_none(
        {
            "available_tools": available_tools,
            "agent_handoffs": agent_handoffs,
            "input": input,
            "output": output,
            "retrieval_context": retrieval_context,
            "context": context,
            "expected_output": expected_output,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "metadata": metadata,
            "name": name,
            "test_case": test_case,
            "metric_collection": metric_collection,
        }
    )
    token = _pending_next_agent_span.set(payload)
    try:
        yield
    finally:
        _pending_next_agent_span.reset(token)


# --- llm: base + llm-specific (one-stop) -----------------------------------


@contextmanager
def next_llm_span(
    model: Optional[str] = None,
    input_token_count: Optional[float] = None,
    output_token_count: Optional[float] = None,
    cost_per_input_token: Optional[float] = None,
    cost_per_output_token: Optional[float] = None,
    token_intervals: Optional[Dict[float, str]] = None,
    prompt: Optional[Prompt] = None,
    # base fields
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    test_case: Optional[LLMTestCase] = None,
    metric_collection: Optional[str] = None,
) -> Iterator[None]:
    """Set defaults for the next ``LlmSpan``. One-stop: accepts
    LLM-specific fields (``model``, token counts, ``prompt``, ...) AND
    the same base fields ``update_current_span(...)`` takes."""
    payload = _drop_none(
        {
            "model": model,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "cost_per_input_token": cost_per_input_token,
            "cost_per_output_token": cost_per_output_token,
            "token_intervals": token_intervals,
            "prompt": prompt,
            "input": input,
            "output": output,
            "retrieval_context": retrieval_context,
            "context": context,
            "expected_output": expected_output,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "metadata": metadata,
            "name": name,
            "test_case": test_case,
            "metric_collection": metric_collection,
        }
    )
    token = _pending_next_llm_span.set(payload)
    try:
        yield
    finally:
        _pending_next_llm_span.reset(token)


# --- tool: base + tool-specific (one-stop) ---------------------------------


@contextmanager
def next_tool_span(
    description: Optional[str] = None,
    # base fields
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    test_case: Optional[LLMTestCase] = None,
    metric_collection: Optional[str] = None,
) -> Iterator[None]:
    """Set defaults for the next ``ToolSpan``. One-stop: accepts
    tool-specific fields (``description``) AND the same base fields
    ``update_current_span(...)`` takes."""
    payload = _drop_none(
        {
            "description": description,
            "input": input,
            "output": output,
            "retrieval_context": retrieval_context,
            "context": context,
            "expected_output": expected_output,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "metadata": metadata,
            "name": name,
            "test_case": test_case,
            "metric_collection": metric_collection,
        }
    )
    token = _pending_next_tool_span.set(payload)
    try:
        yield
    finally:
        _pending_next_tool_span.reset(token)


# --- retriever: base + retriever-specific (one-stop) -----------------------


@contextmanager
def next_retriever_span(
    embedder: Optional[str] = None,
    top_k: Optional[int] = None,
    chunk_size: Optional[int] = None,
    # base fields
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    test_case: Optional[LLMTestCase] = None,
    metric_collection: Optional[str] = None,
) -> Iterator[None]:
    """Set defaults for the next ``RetrieverSpan``. One-stop: accepts
    retriever-specific fields (``embedder``, ``top_k``, ``chunk_size``)
    AND the same base fields ``update_current_span(...)`` takes."""
    payload = _drop_none(
        {
            "embedder": embedder,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "input": input,
            "output": output,
            "retrieval_context": retrieval_context,
            "context": context,
            "expected_output": expected_output,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "metadata": metadata,
            "name": name,
            "test_case": test_case,
            "metric_collection": metric_collection,
        }
    )
    token = _pending_next_retriever_span.set(payload)
    try:
        yield
    finally:
        _pending_next_retriever_span.reset(token)


# ---------------------------------------------------------------------------
# Consumer-facing pop helpers.
#
# Integrations (e.g. ``deepeval.integrations.pydantic_ai.SpanInterceptor``)
# call these the moment they classify a fresh span and BEFORE they push the
# placeholder onto ``current_span_context``. The pop is one-shot: the slot
# is reset to None for the rest of the active ``with`` scope.
#
# ``pop_pending_for(span_type)`` returns the merged dict of base + typed
# defaults — base values are overwritten by the typed slot's values when
# both are present, matching "more specific wins".
# ---------------------------------------------------------------------------


_TYPED_SLOTS = {
    "agent": _pending_next_agent_span,
    "llm": _pending_next_llm_span,
    "tool": _pending_next_tool_span,
    "retriever": _pending_next_retriever_span,
}


def pop_pending_for(span_type: Optional[str]) -> Dict[str, Any]:
    """One-shot consume the pending-defaults dict for ``span_type``.

    Returns a merged dict {**base_slot, **typed_slot}. Typed values win
    on overlap. Slots that are popped are reset to ``None`` for the
    remainder of the active scope (until the surrounding ``with`` exits
    and restores the prior token).

    ``span_type`` may be one of ``"agent" | "llm" | "tool" |
    "retriever"`` or ``None`` to consume only the base slot.
    """
    merged: Dict[str, Any] = {}

    base_payload = _pending_next_span.get()
    if base_payload:
        merged.update(base_payload)
        _pending_next_span.set(None)

    if span_type and span_type in _TYPED_SLOTS:
        slot = _TYPED_SLOTS[span_type]
        typed_payload = slot.get()
        if typed_payload:
            merged.update(typed_payload)
            slot.set(None)

    return merged


def apply_pending_to_span(span: BaseSpan, payload: Dict[str, Any]) -> None:
    """Apply a popped pending-defaults dict to ``span`` in-place.

    Mirrors ``update_current_span(...)`` semantics for the BASE keys —
    notably the ``test_case`` unpacking path, which writes the
    LLMTestCase's fields onto the span and overrides any individual
    field set in the same payload. Typed kwargs (``available_tools``,
    ``model``, ``embedder``, ``description``, etc.) are setattr'd
    directly when the span is the matching subclass; mismatches are
    silently dropped (e.g. ``model`` on a ``ToolSpan``).

    Used by integrations after pushing a fresh placeholder onto
    ``current_span_context`` so that ``next_*_span(...)`` defaults land
    on the placeholder before user code or downstream serialization sees
    it.
    """
    if not payload:
        return

    test_case = payload.get("test_case")
    if test_case is not None:
        span.input = test_case.input
        span.output = test_case.actual_output
        span.expected_output = test_case.expected_output
        span.retrieval_context = test_case.retrieval_context
        span.context = test_case.context
        span.tools_called = test_case.tools_called
        span.expected_tools = test_case.expected_tools

    for key, value in payload.items():
        if key == "test_case" or value is None:
            continue
        # Only setattr keys the span actually declares — guards against
        # cross-type leakage (e.g. ``embedder`` landing on an LlmSpan).
        if not hasattr(span, key):
            continue
        try:
            setattr(span, key, value)
        except Exception:
            # Pydantic validation errors / locked fields → skip silently.
            continue
