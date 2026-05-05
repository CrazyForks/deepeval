"""AgentCore × deepeval OTel SpanInterceptor.

Translates spans emitted by AWS Bedrock AgentCore / Strands (and any
OpenLLMetry-traceloop instrumented framework that flows through the
same TracerProvider) into ``confident.*`` OTel attributes that
``ConfidentSpanExporter`` rebuilds into deepeval ``BaseSpan``s.

Mirrors the Pydantic AI POC pattern (see
``deepeval/integrations/pydantic_ai/instrumentator.py``):

  - ``current_span_context`` is populated with a ``BaseSpan`` placeholder
    for the OTel span's lifetime, so ``update_current_span(...)`` from
    inside a Strands ``@tool`` body lands on that span. At ``on_end``
    the placeholder's mutated fields are serialized back into
    ``confident.span.*`` OTel attributes.
  - ``current_trace_context`` is populated with an implicit ``Trace``
    placeholder for bare callers (no enclosing ``@observe`` /
    ``with trace(...)``) so ``update_current_trace(...)`` from inside a
    tool body has somewhere to write. The placeholder is tagged
    ``is_otel_implicit=True`` so ``ContextAwareSpanProcessor`` keeps
    routing to OTLP for those callers.
  - ``next_*_span(...)`` payloads are consumed at ``on_start`` via
    ``pop_pending_for(span_type)`` + ``apply_pending_to_span(...)`` so
    ``with next_agent_span(metric_collection=..., metrics=[...]):``
    around an AgentCore invoke lands on the agent span.
  - Trace-level attrs are resolved FRESH at ``on_end`` from
    ``current_trace_context`` (mutations during the run are seen) with
    settings as fallback.
  - ``BaseMetric`` instances staged via ``next_*_span(metrics=[...])``
    are stashed via ``stash_pending_metrics`` so the exporter can
    re-attach them after rebuilding the span (gated on
    ``trace_manager.is_evaluating`` to keep the registry tight).

Framework-specific extraction (Strands ``gen_ai.*`` events, Traceloop
attributes, AWS Bedrock body parsing) stays in this module — those
fields capture data the framework writes, not values the user mutates,
so they don't go through the placeholder serializer.
"""

from __future__ import annotations

import contextvars
import json
import logging
from time import perf_counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    current_trace_context,
    pop_pending_for,
)
from deepeval.tracing.otel.utils import (
    stash_pending_metrics,
    to_hex_string,
)
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    Trace,
    TraceSpanStatus,
    ToolCall,
)

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    from opentelemetry.sdk.trace import (
        ReadableSpan as _ReadableSpan,
        SpanProcessor as _SpanProcessor,
    )

    dependency_installed = True
except ImportError as e:
    dependency_installed = False

    if settings.DEEPEVAL_VERBOSE_MODE:
        logger.warning(
            "Optional tracing dependency not installed: %s",
            getattr(e, "name", repr(e)),
            stacklevel=2,
        )

    class _SpanProcessor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_start(self, span: Any, parent_context: Any) -> None:
            pass

        def on_end(self, span: Any) -> None:
            pass

    class _ReadableSpan:
        pass


def is_dependency_installed() -> bool:
    if not dependency_installed:
        raise ImportError(
            "Dependencies are not installed. Please install them with "
            "`pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`."
        )
    return True


if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
else:
    SpanProcessor = _SpanProcessor
    ReadableSpan = _ReadableSpan


init_clock_bridge()


# ---------------------------------------------------------------------------
# Span classification — kept from the prior implementation. AgentCore +
# Strands emit a mix of ``gen_ai.*`` (OTel GenAI semconv), Traceloop /
# OpenLLMetry attrs, and span-name heuristics. None of this depends on
# settings; it inspects the raw OTel span only.
# ---------------------------------------------------------------------------


_AGENT_OP_NAMES = {"invoke_agent", "create_agent"}

# gen_ai.operation.name values that indicate an LLM-call span
_LLM_OP_NAMES = {
    "chat",
    "generate_content",
    "invoke_model",
    "text_completion",
    "embeddings",
}

# gen_ai.operation.name values that indicate a tool span
_TOOL_OP_NAMES = {"execute_tool"}

# traceloop.span.kind values → confident span type
_TRACELOOP_KIND_MAP = {
    "workflow": "agent",
    "agent": "agent",
    "task": "tool",
    "tool": "tool",
    "retriever": "retriever",
    "llm": "llm",
}


def _get_attr(span, *keys: str) -> Optional[str]:
    attrs = span.attributes or {}
    for k in keys:
        v = attrs.get(k)
        if v:
            return str(v)
    return None


def _classify_span(span) -> Optional[str]:
    attrs = span.attributes or {}
    span_name_lower = (span.name or "").lower()

    # 1. Explicit gen_ai.operation.name (Strands + generic OTel GenAI)
    op_name = attrs.get("gen_ai.operation.name", "")
    if op_name in _AGENT_OP_NAMES:
        return "agent"
    if op_name in _LLM_OP_NAMES:
        return "llm"
    if op_name in _TOOL_OP_NAMES:
        return "tool"

    # 2. OpenLLMetry / traceloop conventions (LangChain, LangGraph, CrewAI)
    traceloop_kind = attrs.get("traceloop.span.kind", "")
    if traceloop_kind in _TRACELOOP_KIND_MAP:
        return _TRACELOOP_KIND_MAP[traceloop_kind]

    # 3. Presence of canonical tool/agent attributes
    if attrs.get("gen_ai.tool.name") or attrs.get("gen_ai.tool.call.id"):
        return "tool"
    if attrs.get("gen_ai.agent.name") or attrs.get("gen_ai.agent.id"):
        return "agent"

    # 4. Heuristic span-name matching (last resort)
    if any(kw in span_name_lower for kw in ("invoke_agent", "agent")):
        return "agent"
    if any(kw in span_name_lower for kw in ("execute_tool", ".tool")):
        return "tool"
    if any(kw in span_name_lower for kw in ("retriev", "memory", "datastore")):
        return "retriever"
    if any(
        kw in span_name_lower
        for kw in ("llm", "chat", "invoke_model", "generate")
    ):
        return "llm"

    return None


def _get_agent_name(span) -> Optional[str]:
    """Extract the most descriptive agent name available."""
    return (
        _get_attr(
            span,
            "gen_ai.agent.name",
            "traceloop.entity.name",
            "traceloop.workflow.name",
        )
        or span.name
        or None
    )


def _get_tool_name(span) -> Optional[str]:
    """Extract the tool name from a tool span."""
    return (
        _get_attr(
            span,
            "gen_ai.tool.name",
            "traceloop.entity.name",
        )
        or span.name
        or None
    )


# ---------------------------------------------------------------------------
# Content / I/O extraction helpers.
#
# These walk Strands-style ``gen_ai.*`` events and Traceloop-style
# attributes to pull out user-facing input/output text and tool calls.
# They produce data we want surfaced as ``confident.span.*`` /
# ``confident.trace.*`` attrs at on_end — capturing what the FRAMEWORK
# wrote, distinct from what the user mutated via update_current_span.
# ---------------------------------------------------------------------------


def _parse_genai_content(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return str(raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return first.get("text") or first.get("content") or str(first)
            return str(first)
        if isinstance(data, dict):
            return data.get("text") or data.get("content") or str(data)
        return str(data)
    except (json.JSONDecodeError, TypeError):
        return raw


def _extract_messages(span) -> tuple[Optional[str], Optional[str]]:
    input_text: Optional[str] = None
    output_text: Optional[str] = None

    # 1. Extract from Events (Strands / strict OTel GenAI)
    for event in getattr(span, "events", []):
        event_name = event.name or ""
        event_attrs = event.attributes or {}

        if event_name == "gen_ai.user.message":
            input_text = _parse_genai_content(event_attrs.get("content"))
        elif event_name in ("gen_ai.choice", "gen_ai.assistant.message"):
            output_text = _parse_genai_content(
                event_attrs.get("message") or event_attrs.get("content")
            )
        elif event_name == "gen_ai.system.message":
            if not input_text:
                input_text = _parse_genai_content(event_attrs.get("content"))
        elif event_name in (
            "gen_ai.client.inference.operation.details",
            "agent.invocation",
            "tool.invocation",
        ):
            body_raw = event_attrs.get("body") or event_attrs.get("event.body")
            if body_raw:
                try:
                    body = (
                        json.loads(body_raw)
                        if isinstance(body_raw, str)
                        else body_raw
                    )
                    if not input_text and "input" in body:
                        msgs = body["input"].get("messages", [])
                        if msgs:
                            input_text = _parse_genai_content(
                                msgs[-1].get("content")
                                if isinstance(msgs[-1], dict)
                                else msgs[-1]
                            )
                    if not output_text and "output" in body:
                        msgs = body["output"].get("messages", [])
                        if msgs:
                            output_text = _parse_genai_content(
                                msgs[-1].get("content")
                                if isinstance(msgs[-1], dict)
                                else msgs[-1]
                            )
                except Exception:
                    pass

    # 2. Fall back to attributes (LangChain, CrewAI, Traceloop)
    if not input_text:
        raw = _get_attr(
            span,
            "gen_ai.user.message",
            "gen_ai.input.messages",
            "gen_ai.prompt",
            "traceloop.entity.input",
            "crewai.task.description",
        )
        if raw:
            input_text = _parse_genai_content(raw)

    if not output_text:
        raw = _get_attr(
            span,
            "gen_ai.choice",
            "gen_ai.output.messages",
            "gen_ai.completion",
            "traceloop.entity.output",
        )
        if raw:
            output_text = _parse_genai_content(raw)

    return input_text, output_text


def _extract_tool_calls(span) -> List[ToolCall]:
    tools: List[ToolCall] = []

    # 1. Extract from events (Strands / strict OTel)
    for event in getattr(span, "events", []):
        event_attrs = event.attributes or {}
        event_name = event.name or ""

        if event_name in ("gen_ai.tool.call", "tool_call", "execute_tool"):
            try:
                name = (
                    event_attrs.get("gen_ai.tool.name")
                    or event_attrs.get("name")
                    or "unknown_tool"
                )
                args_raw = (
                    event_attrs.get("gen_ai.tool.call.arguments")
                    or event_attrs.get("gen_ai.tool.arguments")
                    or event_attrs.get("input")
                    or "{}"
                )
                input_params = (
                    json.loads(args_raw)
                    if isinstance(args_raw, str)
                    else args_raw
                )
                tools.append(
                    ToolCall(name=str(name), input_parameters=input_params)
                )
            except Exception as exc:
                logger.debug("Failed to parse tool call event: %s", exc)

    # 2. Extract from attributes (LangChain / CrewAI / Traceloop)
    attrs = span.attributes or {}

    tool_calls_raw = (
        attrs.get("gen_ai.tool.calls")
        or attrs.get("traceloop.tool_calls")
        or attrs.get("llm.tool_calls")
    )

    if tool_calls_raw:
        try:
            calls = (
                json.loads(tool_calls_raw)
                if isinstance(tool_calls_raw, str)
                else tool_calls_raw
            )
            if isinstance(calls, list):
                for call in calls:
                    # Traceloop/OpenLLMetry often nests these under a "function" key
                    name = (
                        call.get("name")
                        or call.get("function", {}).get("name")
                        or "unknown_tool"
                    )
                    args = (
                        call.get("arguments")
                        or call.get("function", {}).get("arguments")
                        or "{}"
                    )

                    input_params = (
                        json.loads(args) if isinstance(args, str) else args
                    )
                    tools.append(
                        ToolCall(name=str(name), input_parameters=input_params)
                    )
        except Exception as exc:
            logger.debug("Failed to parse tool call attributes: %s", exc)

    return tools


def _extract_tool_call_from_tool_span(span) -> Optional[ToolCall]:
    tool_name = _get_tool_name(span)
    if not tool_name:
        return None

    attrs = span.attributes or {}
    args_raw = (
        attrs.get("gen_ai.tool.call.arguments")
        or attrs.get("traceloop.entity.input")
        or "{}"
    )
    try:
        input_params = (
            json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        )
    except Exception:
        input_params = {}

    return ToolCall(name=tool_name, input_parameters=input_params)


# ---------------------------------------------------------------------------
# Settings — trace-only kwargs, mirroring DeepEvalInstrumentationSettings.
#
# Span-level configuration is intentionally NOT here. Set per-call
# defaults via ``with next_agent_span(...)`` / ``with next_llm_span(...)``
# / ``with next_tool_span(...)`` before invoking the agent, or mutate
# the live placeholder via ``update_current_span(...)`` from inside a
# Strands ``@tool`` body. See deepeval/integrations/README.md for the
# full migration table.
# ---------------------------------------------------------------------------


class AgentCoreInstrumentationSettings:
    """Trace-level defaults for AgentCore instrumentation.

    All kwargs are optional. Trace fields are stamped onto every trace
    produced through ``instrument_agentcore(...)`` (resolved at every
    span's ``on_end`` so runtime mutations via ``update_current_trace``
    win).

    A Confident AI ``api_key`` is fully optional. When omitted (and
    ``CONFIDENT_API_KEY`` isn't in the environment), the OTel pipeline
    still runs locally but no auth header is attached to the OTLP
    exporter, so the Confident AI backend rejects the upload. Wire a
    key when you actually want spans to land in Confident AI.
    """

    # All known span-level kwargs that used to live on the top-level
    # signature. Removed in the migration; raise ``TypeError`` so callers
    # see exactly what to do. Mirrors
    # ``test_span_related_kwargs_are_removed_from_settings`` in the
    # Pydantic AI test suite.
    _REMOVED_KWARGS = (
        "is_test_mode",
        "agent_metric_collection",
        "llm_metric_collection",
        "tool_metric_collection_map",
        "trace_metric_collection",
        "agent_metrics",
        "confident_prompt",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        metric_collection: Optional[str] = None,
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        environment: Optional[str] = None,
        **removed_kwargs: Any,
    ):
        is_dependency_installed()

        # Reject removed span-level kwargs explicitly so callers get a
        # crisp error pointing at the migration. We accept ``**removed_kwargs``
        # only to produce a TypeError with a helpful message; otherwise
        # Python's default would already raise on unknown kwargs.
        if removed_kwargs:
            offending = ", ".join(sorted(removed_kwargs))
            raise TypeError(
                f"AgentCoreInstrumentationSettings: unexpected keyword "
                f"argument(s) {offending}. Span-level kwargs were removed "
                "in the OTel POC migration. Use ``with next_agent_span(...)`` "
                "/ ``with next_llm_span(...)`` / ``with next_tool_span(...)`` "
                "before invoking the agent, or "
                "``update_current_span(...)`` from inside a Strands @tool "
                "body. See deepeval/integrations/README.md for details."
            )

        if trace_manager.environment is not None:
            _env = trace_manager.environment
        elif environment is not None:
            _env = environment
        elif settings.CONFIDENT_TRACE_ENVIRONMENT is not None:
            _env = settings.CONFIDENT_TRACE_ENVIRONMENT
        else:
            _env = "development"

        if _env not in ("production", "staging", "development", "testing"):
            _env = "development"
        self.environment = _env

        self.api_key = api_key
        self.name = name
        self.thread_id = thread_id
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.metric_collection = metric_collection
        self.test_case_id = test_case_id
        self.turn_id = turn_id


# ---------------------------------------------------------------------------
# Span interceptor.
#
# Two responsibilities:
#   1. Push a ``BaseSpan`` placeholder onto ``current_span_context`` for
#      every OTel span's lifetime. ``update_current_span(...)`` /
#      ``update_*_span(...)`` calls from inside a Strands ``@tool`` body
#      mutate this placeholder; at on_end the mutated fields are
#      serialized back into ``confident.span.*`` OTel attrs.
#   2. Translate AgentCore / Strands / Traceloop framework-emitted
#      attributes into ``confident.*`` attrs (input/output text, tool
#      calls, model name, token counts) — distinct from user mutations.
#
# Plus the smaller bridges from the Pydantic AI POC:
#   - Implicit ``Trace(is_otel_implicit=True)`` push for bare callers so
#     ``update_current_trace(...)`` from a tool body works.
#   - ``confident.span.parent_uuid`` stamp for OTel roots inside an
#     enclosing deepeval span (so ``@observe(type="agent") -> agent(...)``
#     stitches into a single trace).
#   - ``next_*_span(...)`` payload consumption + ``stash_pending_metrics``
#     gate for component-level evals.
# ---------------------------------------------------------------------------


class AgentCoreSpanInterceptor(SpanProcessor):

    def __init__(self, settings_instance: AgentCoreInstrumentationSettings):
        self.settings = settings_instance
        # Per-OTel-span state, keyed by OTel span_id. Two spans never
        # share an id within a process so this is safe across threads /
        # asyncio tasks.
        self._tokens: Dict[int, contextvars.Token] = {}
        self._placeholders: Dict[int, BaseSpan] = {}
        # Per-OTel-root-span state for the implicit trace placeholder we
        # push when there's no enclosing context.
        self._trace_tokens: Dict[int, contextvars.Token] = {}
        self._trace_placeholders: Dict[int, Trace] = {}

    # ------------------------------------------------------------------
    # on_start
    # ------------------------------------------------------------------

    def on_start(self, span, parent_context):
        # Push implicit Trace for bare callers BEFORE classification, so
        # the implicit context is in place if anything downstream reads
        # ``current_trace_context`` (parity with Pydantic AI's order).
        self._maybe_push_implicit_trace_context(span)

        # Bridge OTel root spans to an enclosing deepeval span (e.g.
        # ``@observe(type="agent")`` wrapping an agentcore invoke) by
        # stamping ``confident.span.parent_uuid``. Only fires for OTel
        # roots; child OTel spans keep their native parent.
        self._maybe_bridge_otel_root_to_deepeval_parent(span)

        # Classify the span using AgentCore's existing heuristics. This
        # is the source of truth for ``confident.span.type`` and decides
        # which placeholder subclass we push.
        span_type = _classify_span(span)
        if span_type:
            try:
                span.set_attribute("confident.span.type", span_type)
            except Exception:
                pass

        # Span-type-specific name discovery (kept from the prior
        # implementation; agent / tool name attrs land at on_start
        # because the placeholder type depends on them).
        if span_type == "agent":
            agent_name = _get_agent_name(span)
            if agent_name:
                try:
                    span.set_attribute("confident.span.name", agent_name)
                except Exception:
                    pass
        elif span_type == "tool":
            tool_name = _get_tool_name(span)
            if tool_name:
                try:
                    span.set_attribute("confident.span.name", tool_name)
                except Exception:
                    pass

        # Push BaseSpan placeholder so ``update_current_span(...)``
        # from inside a Strands ``@tool`` body lands somewhere.
        self._push_span_context(span, span_type)

    # ------------------------------------------------------------------
    # on_end
    # ------------------------------------------------------------------

    def on_end(self, span):
        sid = span.get_span_context().span_id

        # Snapshot trace context FRESH at on_end so the latest
        # ``update_current_trace(...)`` values land on this OTel span.
        try:
            self._serialize_trace_context_to_otel_attrs(span)
        except Exception as exc:
            logger.debug(
                "Failed to serialize trace context for span_id=%s: %s",
                sid,
                exc,
            )

        # Pop placeholder + reset contextvar token; serialize user-mutated
        # fields onto ``confident.span.*`` attrs.
        placeholder = self._placeholders.pop(sid, None)
        token = self._tokens.pop(sid, None)
        if token is not None:
            try:
                current_span_context.reset(token)
            except Exception as exc:
                logger.debug(
                    "Failed to reset current_span_context for span_id=%s: %s",
                    sid,
                    exc,
                )
        if placeholder is not None:
            try:
                self._serialize_placeholder_to_otel_attrs(placeholder, span)
            except Exception as exc:
                logger.debug(
                    "Failed to serialize span placeholder for span_id=%s: %s",
                    sid,
                    exc,
                )
            try:
                if placeholder.metrics and trace_manager.is_evaluating:
                    stash_pending_metrics(
                        to_hex_string(sid, 16), placeholder.metrics
                    )
            except Exception as exc:
                logger.debug(
                    "Failed to stash pending metrics for span_id=%s: %s",
                    sid,
                    exc,
                )

        # Framework extraction: write Strands / Traceloop / GenAI attrs
        # onto ``confident.*`` so the exporter rebuilds richer spans.
        # These are FRAMEWORK fields (events written by the agent loop),
        # not user-mutable, so they live alongside the placeholder
        # serialization rather than inside it.
        try:
            self._serialize_framework_attrs(span)
        except Exception as exc:
            logger.debug(
                "Failed to serialize framework attrs for span_id=%s: %s",
                sid,
                exc,
            )

        # Pop implicit Trace placeholder if we pushed one for this span.
        # MUST run after trace-context serialization above so the
        # implicit placeholder's mutations land on this root's attrs.
        self._maybe_pop_implicit_trace_context(span)

    # ------------------------------------------------------------------
    # Placeholder push / pop on current_span_context
    # ------------------------------------------------------------------

    def _push_span_context(self, span, span_type: Optional[str]) -> None:
        """Create a placeholder ``BaseSpan`` (or ``AgentSpan``) and push.

        Mirrors ``deepeval.integrations.pydantic_ai.SpanInterceptor._push_span_context``.
        Consumes any ``next_*_span(...)`` defaults via ``pop_pending_for``
        + ``apply_pending_to_span`` BEFORE the push so the placeholder
        the user code sees has the staged values.
        """
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            start_time = (
                peb.epoch_nanos_to_perf_seconds(span.start_time)
                if span.start_time
                else perf_counter()
            )
            kwargs: Dict[str, Any] = dict(
                uuid=to_hex_string(sid, 16),
                trace_uuid=to_hex_string(tid, 32),
                status=TraceSpanStatus.IN_PROGRESS,
                start_time=start_time,
            )
            if span_type == "agent":
                # Pull the on_start-stamped name (set above from
                # _get_agent_name) so the placeholder's name matches the
                # OTel attr — saves a duplicate lookup.
                attrs = span.attributes or {}
                placeholder = AgentSpan(
                    name=(
                        attrs.get("confident.span.name")
                        or _get_agent_name(span)
                        or "agent"
                    ),
                    **kwargs,
                )
            else:
                placeholder = BaseSpan(**kwargs)

            pending = pop_pending_for(span_type)
            if pending:
                apply_pending_to_span(placeholder, pending)

            token = current_span_context.set(placeholder)
            self._tokens[sid] = token
            self._placeholders[sid] = placeholder
        except Exception as exc:
            logger.debug(
                "Failed to push current_span_context placeholder: %s", exc
            )

    def _maybe_push_implicit_trace_context(self, span) -> None:
        """Push an implicit ``Trace`` placeholder for bare callers.

        Symmetric to ``_push_span_context`` but at the trace level.
        Only fires for OTel root spans AND only when no caller-pushed
        trace context is active. Tagged ``is_otel_implicit=True`` so
        ``ContextAwareSpanProcessor`` keeps routing to OTLP.
        """
        if current_trace_context.get() is not None:
            return
        if getattr(span, "parent", None) is not None:
            return
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            start_time = (
                peb.epoch_nanos_to_perf_seconds(span.start_time)
                if span.start_time
                else perf_counter()
            )
            implicit = Trace(
                uuid=to_hex_string(tid, 32),
                root_spans=[],
                status=TraceSpanStatus.IN_PROGRESS,
                start_time=start_time,
                is_otel_implicit=True,
            )
            token = current_trace_context.set(implicit)
            self._trace_tokens[sid] = token
            self._trace_placeholders[sid] = implicit
        except Exception as exc:
            logger.debug(
                "Failed to push implicit current_trace_context: %s", exc
            )

    def _maybe_bridge_otel_root_to_deepeval_parent(self, span) -> None:
        """Re-parent OTel roots onto an enclosing deepeval span.

        When ``@observe(type="agent")`` wraps an agentcore invoke, the
        deepeval span lives in ``current_span_context`` but isn't an
        OTel span — so the framework's OTel root has no native parent.
        Stamping ``confident.span.parent_uuid`` lets the exporter
        re-parent the OTel root onto the deepeval span, producing a
        single trace tree instead of two siblings.
        """
        if getattr(span, "parent", None) is not None:
            return
        parent_span = current_span_context.get()
        if parent_span is None:
            return
        parent_uuid = getattr(parent_span, "uuid", None)
        if not parent_uuid:
            return
        try:
            self._set_attr_post_end(
                span, "confident.span.parent_uuid", parent_uuid
            )
        except Exception as exc:
            logger.debug(
                "Failed to bridge OTel root span to deepeval parent "
                "(parent_uuid=%s): %s",
                parent_uuid,
                exc,
            )

    def _maybe_pop_implicit_trace_context(self, span) -> None:
        try:
            sid = span.get_span_context().span_id
        except Exception:
            return
        token = self._trace_tokens.pop(sid, None)
        self._trace_placeholders.pop(sid, None)
        if token is None:
            return
        try:
            current_trace_context.reset(token)
        except Exception as exc:
            logger.debug(
                "Failed to reset implicit current_trace_context for "
                "span_id=%s: %s",
                sid,
                exc,
            )

    # ------------------------------------------------------------------
    # Attribute writers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_attr_post_end(span, key: str, value: Any) -> None:
        """Write an attribute onto a span that may already have ended.

        ``Span.set_attribute`` becomes a silent no-op once
        ``Span.end()`` has been called. Mirrors the Pydantic AI POC
        helper of the same name — writes directly through the
        underlying ``BoundedAttributes`` mapping (which is
        ``immutable=False`` while the span is being processed) so
        downstream processors / exporters see the value.
        """
        try:
            attrs = getattr(span, "_attributes", None)
            if attrs is not None:
                attrs[key] = value
                return
        except Exception as exc:
            logger.debug(
                "Direct _attributes write failed for %s; "
                "falling back to set_attribute (may be dropped): %s",
                key,
                exc,
            )
        try:
            span.set_attribute(key, value)
        except Exception as exc:
            logger.debug("set_attribute fallback failed for %s: %s", key, exc)

    @classmethod
    def _serialize_placeholder_to_otel_attrs(
        cls, placeholder: BaseSpan, span
    ) -> None:
        """Mirror update_current_span writes onto confident.span.* attrs.

        Only writes attrs the user actively set on the placeholder.
        Existing attrs already populated by ``on_start`` (e.g.
        ``confident.span.name`` from the discovered agent name) are not
        overwritten by empty placeholder fields.
        """
        existing = span.attributes or {}

        if placeholder.metadata:
            cls._set_attr_post_end(
                span,
                "confident.span.metadata",
                json.dumps(placeholder.metadata, default=str),
            )
        if placeholder.input is not None:
            cls._set_attr_post_end(
                span,
                "confident.span.input",
                json.dumps(placeholder.input, default=str),
            )
        if placeholder.output is not None:
            cls._set_attr_post_end(
                span,
                "confident.span.output",
                json.dumps(placeholder.output, default=str),
            )
        if placeholder.metric_collection:
            cls._set_attr_post_end(
                span,
                "confident.span.metric_collection",
                placeholder.metric_collection,
            )
        if placeholder.retrieval_context:
            cls._set_attr_post_end(
                span,
                "confident.span.retrieval_context",
                json.dumps(placeholder.retrieval_context),
            )
        if placeholder.context:
            cls._set_attr_post_end(
                span,
                "confident.span.context",
                json.dumps(placeholder.context),
            )
        if placeholder.expected_output:
            cls._set_attr_post_end(
                span,
                "confident.span.expected_output",
                placeholder.expected_output,
            )
        if placeholder.name and not existing.get("confident.span.name"):
            cls._set_attr_post_end(
                span, "confident.span.name", placeholder.name
            )

    def _serialize_trace_context_to_otel_attrs(self, span) -> None:
        """Resolve trace-level attrs FRESH and write to ``confident.trace.*``.

        Reads ``current_trace_context.get()`` (so
        ``update_current_trace(...)`` mutations land on every OTel
        span's attrs) with ``self.settings.*`` as fallback. Metadata
        merges settings as base + runtime context on top.
        """
        trace_ctx = current_trace_context.get()

        _name = (trace_ctx.name if trace_ctx else None) or self.settings.name
        _thread_id = (
            trace_ctx.thread_id if trace_ctx else None
        ) or self.settings.thread_id
        _user_id = (
            trace_ctx.user_id if trace_ctx else None
        ) or self.settings.user_id
        _tags = (trace_ctx.tags if trace_ctx else None) or self.settings.tags
        _test_case_id = (
            trace_ctx.test_case_id if trace_ctx else None
        ) or self.settings.test_case_id
        _turn_id = (
            trace_ctx.turn_id if trace_ctx else None
        ) or self.settings.turn_id
        _trace_metric_collection = (
            trace_ctx.metric_collection if trace_ctx else None
        ) or self.settings.metric_collection
        _metadata = {
            **(self.settings.metadata or {}),
            **((trace_ctx.metadata or {}) if trace_ctx else {}),
        }

        if _name:
            self._set_attr_post_end(span, "confident.trace.name", _name)
        if _thread_id:
            self._set_attr_post_end(
                span, "confident.trace.thread_id", _thread_id
            )
        if _user_id:
            self._set_attr_post_end(span, "confident.trace.user_id", _user_id)
        if _tags:
            self._set_attr_post_end(span, "confident.trace.tags", _tags)
        if _metadata:
            self._set_attr_post_end(
                span, "confident.trace.metadata", json.dumps(_metadata)
            )
        if _trace_metric_collection:
            self._set_attr_post_end(
                span,
                "confident.trace.metric_collection",
                _trace_metric_collection,
            )
        if _test_case_id:
            self._set_attr_post_end(
                span, "confident.trace.test_case_id", _test_case_id
            )
        if _turn_id:
            self._set_attr_post_end(span, "confident.trace.turn_id", _turn_id)
        if self.settings.environment:
            self._set_attr_post_end(
                span,
                "confident.trace.environment",
                self.settings.environment,
            )

        # Mirror inputs/outputs onto the trace if Strands wrote them on
        # the agent root span (existing behavior — derived from
        # framework attrs, not user mutation).
        if not (span.attributes or {}).get("confident.trace.thread_id"):
            session_id = (span.attributes or {}).get("session.id")
            if session_id:
                self._set_attr_post_end(
                    span, "confident.trace.thread_id", session_id
                )

    def _serialize_framework_attrs(self, span) -> None:
        """Translate Strands / Traceloop / GenAI attrs into ``confident.*``.

        These fields are written by the framework (events on the OTel
        span, raw attrs from the underlying SDK), not by user code, so
        they don't go through the placeholder serializer. We still
        prefer the placeholder's value if the user mutated it via
        ``update_current_span(...)`` — the placeholder serializer
        already wrote those, so we use ``setdefault`` semantics here.
        """
        attrs = span.attributes or {}
        span_type = attrs.get("confident.span.type") or _classify_span(span)
        if span_type and "confident.span.type" not in attrs:
            self._set_attr_post_end(span, "confident.span.type", span_type)

        input_text, output_text = _extract_messages(span)

        if input_text and "confident.span.input" not in attrs:
            self._set_attr_post_end(span, "confident.span.input", input_text)
            if span_type == "agent":
                self._set_attr_post_end(
                    span, "confident.trace.input", input_text
                )

        if output_text and "confident.span.output" not in attrs:
            self._set_attr_post_end(span, "confident.span.output", output_text)
            if span_type == "agent":
                self._set_attr_post_end(
                    span, "confident.trace.output", output_text
                )

        input_tokens = attrs.get("gen_ai.usage.input_tokens") or attrs.get(
            "gen_ai.usage.prompt_tokens"
        )
        output_tokens = attrs.get("gen_ai.usage.output_tokens") or attrs.get(
            "gen_ai.usage.completion_tokens"
        )
        if input_tokens is not None:
            self._set_attr_post_end(
                span, "confident.llm.input_token_count", int(input_tokens)
            )
        if output_tokens is not None:
            self._set_attr_post_end(
                span, "confident.llm.output_token_count", int(output_tokens)
            )

        model = _get_attr(
            span,
            "gen_ai.response.model",
            "gen_ai.request.model",
        )
        if model:
            self._set_attr_post_end(span, "confident.llm.model", model)

        tools_called: List[ToolCall] = []

        if span_type == "agent":
            tools_called = _extract_tool_calls(span)

            tool_defs_raw = attrs.get("gen_ai.tool.definitions") or attrs.get(
                "gen_ai.agent.tools"
            )
            if tool_defs_raw:
                self._set_attr_post_end(
                    span,
                    "confident.agent.tool_definitions",
                    str(tool_defs_raw),
                )

        elif span_type == "tool":
            tc = _extract_tool_call_from_tool_span(span)
            if tc:
                tools_called = [tc]

                if tc.input_parameters and "confident.span.input" not in attrs:
                    self._set_attr_post_end(
                        span,
                        "confident.span.input",
                        json.dumps(tc.input_parameters),
                    )

            if "confident.span.output" not in attrs:
                raw_output = _get_attr(
                    span, "traceloop.entity.output", "gen_ai.tool.output"
                )
                if raw_output:
                    self._set_attr_post_end(
                        span, "confident.span.output", raw_output
                    )

        if tools_called:
            self._set_attr_post_end(
                span,
                "confident.span.tools_called",
                [t.model_dump_json() for t in tools_called],
            )

        if span_type == "agent" and "confident.span.name" not in attrs:
            agent_name = _get_agent_name(span)
            if agent_name:
                self._set_attr_post_end(span, "confident.span.name", agent_name)
