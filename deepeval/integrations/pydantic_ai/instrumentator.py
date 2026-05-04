from __future__ import annotations

import contextvars
import json
import logging
from time import perf_counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
)
from deepeval.tracing.trace_context import (
    current_agent_context,
    current_llm_context,
)
from deepeval.tracing.otel.context_aware_processor import (
    ContextAwareSpanProcessor,
)
from deepeval.tracing.otel.utils import (
    to_hex_string,
)
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    TraceSpanStatus,
)

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    # Optional dependencies
    from opentelemetry.sdk.trace import (
        ReadableSpan as _ReadableSpan,
        SpanProcessor as _SpanProcessor,
        TracerProvider,
    )
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.trace import set_tracer_provider
    from pydantic_ai.models.instrumented import (
        InstrumentationSettings as _BaseInstrumentationSettings,
    )

    dependency_installed = True
except ImportError as e:
    dependency_installed = False

    # Preserve previous behavior: only log when verbose mode is enabled.
    if settings.DEEPEVAL_VERBOSE_MODE:
        if isinstance(e, ModuleNotFoundError):
            logger.warning(
                "Optional tracing dependency not installed: %s",
                getattr(e, "name", repr(e)),
                stacklevel=2,
            )
        else:
            logger.warning(
                "Optional tracing import failed: %s",
                e,
                stacklevel=2,
            )

    # Dummy fallbacks so imports and class definitions don't crash when
    # optional deps are missing. Actual use is still guarded by
    # is_dependency_installed().
    class _BaseInstrumentationSettings:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

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
            "Dependencies are not installed. Please install it with "
            "`pip install pydantic-ai opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http`."
        )
    return True


if TYPE_CHECKING:
    # For type checkers, use real types
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
    from pydantic_ai.models.instrumented import InstrumentationSettings
else:
    # At runtime we always have something to subclass / annotate with
    InstrumentationSettings = _BaseInstrumentationSettings
    SpanProcessor = _SpanProcessor
    ReadableSpan = _ReadableSpan

# Routing + OTLP endpoint live in ContextAwareSpanProcessor now.
init_clock_bridge()  # initialize clock bridge for perf_counter() to epoch_nanos conversion


class ConfidentInstrumentationSettings(InstrumentationSettings):

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        metric_collection: Optional[str] = None,
        confident_prompt: Optional[Prompt] = None,
        llm_metric_collection: Optional[str] = None,
        agent_metric_collection: Optional[str] = None,
        tool_metric_collection_map: Optional[dict] = None,
        trace_metric_collection: Optional[str] = None,
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        agent_metrics: Optional[List[BaseMetric]] = None,
    ):
        is_dependency_installed()

        if trace_manager.environment is not None:
            _environment = trace_manager.environment
        elif settings.CONFIDENT_TRACE_ENVIRONMENT is not None:
            _environment = settings.CONFIDENT_TRACE_ENVIRONMENT
        else:
            _environment = "development"
        if _environment and _environment in [
            "production",
            "staging",
            "development",
            "testing",
        ]:
            self.environment = _environment

        self.tool_metric_collection_map = tool_metric_collection_map or {}
        self.name = name
        self.thread_id = thread_id
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.metric_collection = metric_collection
        self.confident_prompt = confident_prompt
        self.llm_metric_collection = llm_metric_collection
        self.agent_metric_collection = agent_metric_collection
        self.trace_metric_collection = trace_metric_collection
        self.test_case_id = test_case_id
        self.turn_id = turn_id
        self.agent_metrics = agent_metrics

        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError("CONFIDENT_API_KEY is not set")

        trace_provider = TracerProvider()

        # Per-span attribute writes (thread/user/tags/metric_collection lookups
        # against the live deepeval contexts) happen here.
        span_interceptor = SpanInterceptor(self)
        trace_provider.add_span_processor(span_interceptor)

        # Single processor handles both transports: REST (via
        # ConfidentSpanExporter -> trace_manager) when a deepeval trace
        # context is active or an evaluation is running, OTLP otherwise.
        trace_provider.add_span_processor(
            ContextAwareSpanProcessor(api_key=api_key)
        )

        try:
            set_tracer_provider(trace_provider)
        except Exception as e:
            # Handle case where provider is already set (optional warning)
            logger.warning(f"Could not set global tracer provider: {e}")

        super().__init__(tracer_provider=trace_provider)


class SpanInterceptor(SpanProcessor):
    """Translate Pydantic AI OTel spans into deepeval ``confident.*`` attrs.

    Trace-level attrs (``confident.trace.*``) are resolved per-span as a
    union of the live ``current_trace_context`` (set anywhere via
    ``update_current_trace(...)``) and the ``ConfidentInstrumentationSettings``
    defaults — context wins, settings fall back. The same applies to
    agent / LLM ``metric_collection`` via ``current_agent_context`` /
    ``current_llm_context``.

    Span-level attrs (``confident.span.*``) are populated from a per-OTel-span
    ``BaseSpan`` placeholder pushed onto ``current_span_context`` for the span's
    lifetime. This is what makes ``update_current_span(metadata=..., name=...,
    input=..., output=..., metric_collection=..., ...)`` work from anywhere in
    the call stack — including from inside ``@agent.tool_plain`` functions —
    just like Langfuse's SDK. At ``on_end`` the placeholder's mutated fields
    are serialized back into ``confident.span.*`` OTel attributes so the
    exporter (REST or OTLP) picks them up.
    """

    LLM_OPERATION_NAMES = {"chat", "generate_content", "text_completion"}

    def __init__(self, settings_instance: ConfidentInstrumentationSettings):
        self.settings = settings_instance
        # Per-OTel-span state, keyed by span_id. Two spans never share an id
        # within a process so this is safe across threads / asyncio tasks.
        self._tokens: Dict[int, contextvars.Token] = {}
        self._placeholders: Dict[int, BaseSpan] = {}

    def on_start(self, span, parent_context):
        # NOTE: we deliberately do NOT mutate ``trace_ctx.uuid`` to match the
        # OTel trace_id here. Doing so would desync ``trace.uuid`` from its
        # ``trace_manager.active_traces`` dict key, causing the exporter to
        # cache-miss on lookup and spawn a phantom duplicate trace.
        # ``ConfidentSpanExporter`` re-keys incoming OTel spans to the active
        # context's real trace_uuid when a deepeval trace is in scope.

        # Trace-level + span-level user-mutable attrs (everything that
        # ``update_current_trace(...)`` / ``update_current_span(...)`` can
        # change) are written at ``on_end`` instead of here, so the OTel span
        # captures the LATEST values rather than a stale on_start snapshot.
        # See ``_serialize_trace_context_to_otel_attrs`` and
        # ``_serialize_placeholder_to_otel_attrs``.

        # ----- on_start writes only the things that won't change later -----
        # ``confident_prompt`` is bound at ``ConfidentInstrumentationSettings``
        # construction time and is span-level metadata, so it's safe to set
        # here while the span is still mutable (and avoids one more pass at
        # on_end through the post-end attr writer).
        if self.settings.confident_prompt:
            span.set_attribute(
                "confident.span.prompt_alias",
                self.settings.confident_prompt.alias,
            )
            span.set_attribute(
                "confident.span.prompt_commit_hash",
                self.settings.confident_prompt.hash,
            )
            if self.settings.confident_prompt.version:
                span.set_attribute(
                    "confident.span.prompt_label",
                    self.settings.confident_prompt.label,
                )
                span.set_attribute(
                    "confident.span.prompt_version",
                    self.settings.confident_prompt.version,
                )

        # ----- per-span classification + per-span metric_collection -----
        # Span classification (agent / llm / tool) needs to happen at
        # on_start because ``_push_span_context`` reads the assigned
        # ``confident.span.type`` to decide whether to create an ``AgentSpan``
        # vs a ``BaseSpan`` placeholder. The metric_collection lookups read
        # ``current_agent_context`` / ``current_llm_context`` here as well so
        # the placeholder carries the right metric_collection from the
        # outset (the user can still override it via update_current_span).
        agent_ctx = current_agent_context.get()
        llm_ctx = current_llm_context.get()
        operation_name = span.attributes.get("gen_ai.operation.name")
        agent_name = (
            span.attributes.get("gen_ai.agent.name")
            or span.attributes.get("pydantic_ai.agent.name")
            or span.attributes.get("agent_name")
        )

        if agent_name and self._is_agent_span(operation_name):
            self._add_agent_span(span, agent_name, agent_ctx)

        if operation_name in self.LLM_OPERATION_NAMES:
            # Explicitly classify model request spans as LLM spans so they are
            # not mislabeled as agent spans when gen_ai.agent.name is present.
            span.set_attribute("confident.span.type", "llm")
            _llm_metric_collection = (
                (llm_ctx.metric_collection if llm_ctx else None)
                or self.settings.llm_metric_collection
            )
            if _llm_metric_collection:
                span.set_attribute(
                    "confident.span.metric_collection",
                    _llm_metric_collection,
                )

        tool_name = span.attributes.get("gen_ai.tool.name")
        if tool_name:
            tool_metric_collection = (
                self.settings.tool_metric_collection_map.get(tool_name)
            )
            if tool_metric_collection:
                span.set_attribute(
                    "confident.span.metric_collection",
                    str(tool_metric_collection),
                )

        # ----- push BaseSpan placeholder so update_current_span works -----
        self._push_span_context(span, agent_name, operation_name)

    def on_end(self, span):
        sid = span.get_span_context().span_id

        # ----- snapshot trace context FRESH at on_end -----
        # Resolved here (not at on_start) so the latest update_current_trace
        # values land on the OTel span. Uses the post-end attr writer because
        # the SDK has already set ``_end_time`` by the time on_end fires,
        # which makes ``span.set_attribute`` a silent no-op.
        try:
            self._serialize_trace_context_to_otel_attrs(span)
        except Exception as exc:
            logger.debug(
                "Failed to serialize trace context for span_id=%s: %s",
                sid,
                exc,
            )

        # ----- pop current_span_context and serialize user mutations -----
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

        # ----- catch any agent spans that weren't classified at on_start -----
        already_processed = span.attributes.get("confident.span.type") in {
            "agent",
            "llm",
            "tool",
        }
        if not already_processed:
            operation_name = span.attributes.get("gen_ai.operation.name")
            agent_name = (
                span.attributes.get("gen_ai.agent.name")
                or span.attributes.get("pydantic_ai.agent.name")
                or span.attributes.get("agent_name")
            )
            if agent_name and self._is_agent_span(operation_name):
                self._add_agent_span(span, agent_name)

    def _push_span_context(
        self,
        span,
        agent_name: Optional[str],
        operation_name: Optional[str],
    ) -> None:
        """Create a placeholder BaseSpan and push it onto current_span_context.

        The placeholder is only used as a write target for
        ``update_current_span(...)``. Its fields are serialized back into
        ``confident.span.*`` OTel attributes at ``on_end``. The actual span
        objects shipped to Confident AI are still constructed by the exporter.
        """
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            span_type = span.attributes.get("confident.span.type")
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
                placeholder = AgentSpan(
                    name=(
                        span.attributes.get("confident.span.name")
                        or agent_name
                        or "agent"
                    ),
                    **kwargs,
                )
            else:
                placeholder = BaseSpan(**kwargs)
            token = current_span_context.set(placeholder)
            self._tokens[sid] = token
            self._placeholders[sid] = placeholder
        except Exception as exc:
            logger.debug(
                "Failed to push current_span_context placeholder: %s", exc
            )

    @staticmethod
    def _set_attr_post_end(span, key: str, value: Any) -> None:
        """Write an attribute onto a span that may already have ended.

        ``Span.set_attribute`` becomes a silent no-op once ``Span.end()`` has
        been called (the SDK guards on ``self._end_time is not None`` and just
        logs a warning), and the SDK invokes ``on_end`` AFTER setting
        ``_end_time`` — so the obvious ``span.set_attribute(...)`` from inside
        ``SpanInterceptor.on_end`` never lands.

        However the live span constructs its ``_attributes`` as a
        ``BoundedAttributes`` with ``immutable=False`` and passes that same
        dict by reference into ``_readable_span()`` (the ReadableSpan passed to
        all processors). Writing through the mapping's ``__setitem__``
        bypasses the ended-span guard while still respecting the bounded-size
        limits. SpanProcessors fire in registration order, so writes from
        ``SpanInterceptor.on_end`` are visible to ``ConfidentSpanExporter``
        downstream.

        We fall back to ``span.set_attribute`` if the private API ever
        disappears — that path will warn-and-drop, but at least it won't
        crash.
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
            logger.debug(
                "set_attribute fallback failed for %s: %s", key, exc
            )

    @classmethod
    def _serialize_placeholder_to_otel_attrs(
        cls, placeholder: BaseSpan, span
    ) -> None:
        """Mirror update_current_span writes onto confident.span.* attrs.

        Only writes attrs the user actively set on the placeholder. Existing
        attrs already populated by ``on_start`` (e.g. ``confident.span.name``
        when the agent name was discovered, or ``confident.span.metric_collection``
        from settings) are not overwritten by empty placeholder fields.
        """
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
        if placeholder.name and not span.attributes.get(
            "confident.span.name"
        ):
            cls._set_attr_post_end(
                span, "confident.span.name", placeholder.name
            )

    def _serialize_trace_context_to_otel_attrs(self, span) -> None:
        """Resolve trace-level attrs FRESH and write to ``confident.trace.*``.

        Reads from ``current_trace_context`` (so ``update_current_trace(...)``
        from anywhere in the call stack lands on every OTel span) with
        ``ConfidentInstrumentationSettings`` defaults as fallback. Metadata
        merges settings as base + runtime context on top.

        Called at ``on_end`` (not ``on_start``) so the latest values are
        captured rather than a stale snapshot. Goes through
        ``_set_attr_post_end`` so it works after the SDK has finalized the
        span's ``_end_time``.
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
            (trace_ctx.metric_collection if trace_ctx else None)
            or self.settings.trace_metric_collection
            or self.settings.metric_collection
        )
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
            self._set_attr_post_end(
                span, "confident.trace.user_id", _user_id
            )
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
            self._set_attr_post_end(
                span, "confident.trace.turn_id", _turn_id
            )
        if self.settings.environment:
            self._set_attr_post_end(
                span,
                "confident.trace.environment",
                self.settings.environment,
            )

    def _add_agent_span(self, span, name, agent_ctx=None):
        # Uses the post-end-safe writer because this is called from BOTH
        # ``on_start`` (where set_attribute would also work) and ``on_end``
        # (where it wouldn't, since the SDK has already set ``_end_time``).
        # ``_set_attr_post_end`` writes through the underlying mutable
        # ``_attributes`` mapping in either case.
        self._set_attr_post_end(span, "confident.span.type", "agent")
        self._set_attr_post_end(span, "confident.span.name", name)
        _agent_metric_collection = (
            (agent_ctx.metric_collection if agent_ctx else None)
            or self.settings.agent_metric_collection
        )
        if _agent_metric_collection:
            self._set_attr_post_end(
                span,
                "confident.span.metric_collection",
                _agent_metric_collection,
            )

    def _is_agent_span(self, operation_name: Optional[str]) -> bool:
        return operation_name == "invoke_agent"
