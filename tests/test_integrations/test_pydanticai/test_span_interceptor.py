"""Unit tests for ``SpanInterceptor`` (Pydantic AI OTel integration).

Covers:
  - Trace-level reads from ``current_trace_context`` for ``thread_id``,
    ``name``, ``user_id``, ``tags``, ``metadata``, ``test_case_id``,
    ``turn_id``, and trace-level ``metric_collection``.
  - Span-level reads of per-span ``metric_collection`` from
    ``current_agent_context`` / ``current_llm_context``.
  - Span-context push/pop: ``current_span_context`` is set to a placeholder
    ``BaseSpan`` for the OTel span's lifetime so ``update_current_span(...)``
    works from anywhere in the call stack, and the placeholder's mutations
    are serialized back into ``confident.span.*`` OTel attributes at
    ``on_end``.
  - ``ContextAwareSpanProcessor`` routing logic (REST when a deepeval trace
    context is active or an evaluation is running, OTLP otherwise).
"""

import json
from itertools import count
from unittest.mock import MagicMock, patch

import pytest

from deepeval.integrations.pydantic_ai.instrumentator import SpanInterceptor
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    update_current_span,
    update_current_trace,
)
from deepeval.tracing.otel.context_aware_processor import (
    ContextAwareSpanProcessor,
)
from deepeval.tracing.trace_context import (
    AgentSpanContext,
    LlmSpanContext,
    current_agent_context,
    current_llm_context,
    trace,
)


_span_id_counter = count(start=1)
_trace_id_counter = count(start=1)


def _make_mock_span(operation_name=None, agent_name=None, tool_name=None):
    """Mock OTel span that records ``set_attribute`` calls.

    Mirrors the real OTel SDK's invariant that ``Span.attributes`` is a view
    over the same underlying ``_attributes`` mapping — so writes via either
    ``set_attribute(...)`` or direct ``_attributes[...] = ...`` (used by
    ``SpanInterceptor._set_attr_post_end`` to bypass the ended-span guard)
    are observable via ``span.attributes.get(...)``.
    """
    span = MagicMock()
    backing: dict = {}
    span._attributes = backing
    span.attributes = backing
    if operation_name:
        backing["gen_ai.operation.name"] = operation_name
    if agent_name:
        backing["gen_ai.agent.name"] = agent_name
    if tool_name:
        backing["gen_ai.tool.name"] = tool_name
    span.set_attribute.side_effect = lambda k, v: backing.__setitem__(k, v)
    span.get_span_context.return_value = MagicMock(
        trace_id=next(_trace_id_counter),
        span_id=next(_span_id_counter),
    )
    span.parent = None
    span.start_time = None  # forces _push_span_context to use perf_counter()
    return span


def _make_settings(**kwargs):
    """Return a minimal mock ``ConfidentInstrumentationSettings``.

    Only the attributes ``SpanInterceptor`` reads are populated. Anything not
    provided defaults to ``None`` so the context-vs-settings precedence logic
    is exercised cleanly.
    """
    settings = MagicMock(spec=[])  # spec=[] disallows auto-attrs
    settings.thread_id = kwargs.get("thread_id")
    settings.name = kwargs.get("name")
    settings.metadata = kwargs.get("metadata")
    settings.user_id = kwargs.get("user_id")
    settings.tags = kwargs.get("tags")
    settings.test_case_id = kwargs.get("test_case_id")
    settings.turn_id = kwargs.get("turn_id")
    settings.metric_collection = kwargs.get("metric_collection")
    settings.trace_metric_collection = kwargs.get("trace_metric_collection")
    settings.environment = kwargs.get("environment")
    settings.confident_prompt = kwargs.get("confident_prompt")
    settings.llm_metric_collection = kwargs.get("llm_metric_collection")
    settings.agent_metric_collection = kwargs.get("agent_metric_collection")
    settings.tool_metric_collection_map = kwargs.get(
        "tool_metric_collection_map", {}
    )
    settings.agent_metrics = kwargs.get("agent_metrics")
    return settings


# ---------------------------------------------------------------------------
# Trace-context reads (existing fields)
# ---------------------------------------------------------------------------


class TestSpanInterceptorTraceContextReads:
    def test_uses_settings_when_no_trace_context(self):
        """Falls back to settings when current_trace_context is None."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings(
                thread_id="settings-thread",
                name="settings-name",
                metadata={"source": "settings"},
            )
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)
            interceptor.on_end(span)

            assert (
                span.attributes.get("confident.trace.thread_id")
                == "settings-thread"
            )
            assert (
                span.attributes.get("confident.trace.name") == "settings-name"
            )
            assert json.loads(span.attributes["confident.trace.metadata"]) == {
                "source": "settings"
            }
        finally:
            current_trace_context.reset(token)

    def test_prefers_trace_context_over_settings_for_scalars(self):
        """thread_id and name from current_trace_context override settings."""
        settings = _make_settings(
            thread_id="settings-thread",
            name="settings-name",
            metadata={"settings_key": "settings_val"},
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(
            thread_id="ctx-thread",
            name="ctx-name",
            metadata={"ctx_key": "ctx_val"},
        ):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.thread_id") == "ctx-thread"
        assert span.attributes.get("confident.trace.name") == "ctx-name"

    def test_metadata_is_merged_with_context_winning(self):
        """metadata from settings + current_trace_context merge; context wins."""
        settings = _make_settings(
            metadata={"base_key": "base_val", "shared_key": "from_settings"},
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(metadata={"ctx_key": "ctx_val", "shared_key": "from_ctx"}):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        result = json.loads(span.attributes["confident.trace.metadata"])
        assert result["base_key"] == "base_val"
        assert result["ctx_key"] == "ctx_val"
        assert result["shared_key"] == "from_ctx"

    def test_no_attributes_set_when_all_none(self):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)
            interceptor.on_end(span)

            assert "confident.trace.thread_id" not in span.attributes
            assert "confident.trace.name" not in span.attributes
            assert "confident.trace.metadata" not in span.attributes
            assert "confident.trace.user_id" not in span.attributes
            assert "confident.trace.tags" not in span.attributes
        finally:
            current_trace_context.reset(token)


# ---------------------------------------------------------------------------
# Trace-context reads (new in Phase 2)
# ---------------------------------------------------------------------------


class TestSpanInterceptorNewTraceContextReads:
    def test_user_id_from_trace_context_overrides_settings(self):
        settings = _make_settings(user_id="settings-user")
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(user_id="ctx-user"):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.user_id") == "ctx-user"

    def test_tags_from_trace_context_overrides_settings(self):
        settings = _make_settings(tags=["settings-tag"])
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(tags=["ctx-tag-1", "ctx-tag-2"]):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert list(span.attributes.get("confident.trace.tags")) == [
            "ctx-tag-1",
            "ctx-tag-2",
        ]

    def test_test_case_id_and_turn_id_from_trace_context_override_settings(
        self,
    ):
        settings = _make_settings(
            test_case_id="settings-tc",
            turn_id="settings-turn",
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace():
            update_current_trace(test_case_id="ctx-tc", turn_id="ctx-turn")
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.test_case_id") == "ctx-tc"
        assert span.attributes.get("confident.trace.turn_id") == "ctx-turn"

    def test_trace_metric_collection_resolution_order(self):
        """trace_ctx.metric_collection > settings.trace_metric_collection > settings.metric_collection."""
        settings = _make_settings(
            metric_collection="settings-mc",
            trace_metric_collection="settings-trace-mc",
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(metric_collection="ctx-mc"):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert (
            span.attributes.get("confident.trace.metric_collection") == "ctx-mc"
        )

    def test_update_current_trace_after_on_start_lands_on_otel_attrs(self):
        """Trace attrs are snapshotted FRESH at on_end, not on_start.

        Regression guard for the trace-attrs-at-on_start asymmetry: if a
        downstream caller mutates the active trace via ``update_current_trace``
        AFTER the OTel span's ``on_start`` has fired (e.g. from inside an
        ``@agent.tool_plain`` body or any nested helper), the new values
        must still land on this span's ``confident.trace.*`` OTel attributes
        when ``on_end`` runs.
        """
        settings = _make_settings(name="settings-name", user_id="settings-user")
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(name="initial-name"):
            interceptor.on_start(span, None)

            update_current_trace(
                name="updated-name",
                user_id="updated-user",
                tags=["after-update"],
                metadata={"phase": "post-start"},
            )

            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.name") == "updated-name"
        assert span.attributes.get("confident.trace.user_id") == "updated-user"
        assert list(span.attributes.get("confident.trace.tags")) == [
            "after-update"
        ]
        assert json.loads(span.attributes["confident.trace.metadata"]) == {
            "phase": "post-start"
        }

    def test_trace_metric_collection_falls_back_to_settings(self):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings(
                metric_collection="settings-mc",
                trace_metric_collection="settings-trace-mc",
            )
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)
            interceptor.on_end(span)

            assert (
                span.attributes.get("confident.trace.metric_collection")
                == "settings-trace-mc"
            )
        finally:
            current_trace_context.reset(token)


# ---------------------------------------------------------------------------
# Span-context reads for per-span metric_collection
# ---------------------------------------------------------------------------


class TestSpanInterceptorSpanContextMetricCollection:
    def test_llm_metric_collection_from_llm_context_overrides_settings(self):
        settings = _make_settings(llm_metric_collection="settings-llm-mc")
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span(operation_name="chat")

        token = current_llm_context.set(
            LlmSpanContext(metric_collection="ctx-llm-mc")
        )
        try:
            interceptor.on_start(span, None)
        finally:
            current_llm_context.reset(token)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "ctx-llm-mc"
        )

    def test_agent_metric_collection_from_agent_context_overrides_settings(
        self,
    ):
        settings = _make_settings(agent_metric_collection="settings-agent-mc")
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span(
            operation_name="invoke_agent", agent_name="my-agent"
        )

        token = current_agent_context.set(
            AgentSpanContext(metric_collection="ctx-agent-mc")
        )
        try:
            interceptor.on_start(span, None)
        finally:
            current_agent_context.reset(token)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "ctx-agent-mc"
        )

    def test_tool_metric_collection_still_uses_settings_map(self):
        """Tool spans use ``tool_metric_collection_map`` only; no per-context override."""
        settings = _make_settings(
            tool_metric_collection_map={"my_tool": "settings-tool-mc"},
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span(tool_name="my_tool")

        interceptor.on_start(span, None)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "settings-tool-mc"
        )


# ---------------------------------------------------------------------------
# Span-context push/pop: enables update_current_span(...) from anywhere
# ---------------------------------------------------------------------------


class TestSpanInterceptorSpanContextPushPop:
    def test_current_span_context_set_during_span_lifetime(self):
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        # Outside the span, current_span_context.get() may be None or a stale
        # sentinel; we only assert about the *change* introduced by on_start.
        before = current_span_context.get()
        interceptor.on_start(span, None)
        during = current_span_context.get()

        assert during is not None
        assert during is not before

        interceptor.on_end(span)
        after = current_span_context.get()
        assert after is before

    def test_update_current_span_metadata_lands_in_otel_attrs(self):
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        interceptor.on_start(span, None)
        update_current_span(
            metadata={"weather_source": "mock", "city": "Paris"},
            input={"query": "Weather?"},
            output="Sunny",
        )
        interceptor.on_end(span)

        assert span.attributes.get("confident.span.metadata") is not None
        assert json.loads(span.attributes["confident.span.metadata"]) == {
            "weather_source": "mock",
            "city": "Paris",
        }
        assert json.loads(span.attributes["confident.span.input"]) == {
            "query": "Weather?"
        }
        assert json.loads(span.attributes["confident.span.output"]) == "Sunny"

    def test_update_current_span_metric_collection_lands_in_otel_attrs(self):
        """update_current_span(metric_collection=...) overwrites placeholder."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        interceptor.on_start(span, None)
        update_current_span(metric_collection="runtime-collection")
        interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "runtime-collection"
        )

    def test_nested_spans_lifo_pop_restores_parent_placeholder(self):
        """Inner span's on_end restores the outer span's placeholder."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        outer = _make_mock_span()
        inner = _make_mock_span()

        interceptor.on_start(outer, None)
        outer_placeholder = current_span_context.get()

        interceptor.on_start(inner, None)
        inner_placeholder = current_span_context.get()
        assert inner_placeholder is not outer_placeholder

        interceptor.on_end(inner)
        assert current_span_context.get() is outer_placeholder

        interceptor.on_end(outer)


# ---------------------------------------------------------------------------
# ContextAwareSpanProcessor routing
# ---------------------------------------------------------------------------


class _FakeSpan:
    """Minimal stand-in for an OTel span with a stable identity."""


class TestContextAwareSpanProcessorRouting:
    @staticmethod
    def _make_processor():
        """Bypass ``__init__`` so the test doesn't depend on the OTLP exporter
        package being installed locally — we only care about routing logic.
        """
        processor = ContextAwareSpanProcessor.__new__(
            ContextAwareSpanProcessor
        )
        processor._api_key = "test-key"
        processor._rest_processor = MagicMock()
        processor._otlp_processor = MagicMock()
        return processor, processor._rest_processor, processor._otlp_processor

    def test_routes_to_rest_when_trace_context_active(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        with trace():
            processor.on_end(span)

        rest.on_end.assert_called_once_with(span)
        otlp.on_end.assert_not_called()

    def test_routes_to_otlp_when_no_context(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        token = current_trace_context.set(None)
        try:
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = False
                processor.on_end(span)
        finally:
            current_trace_context.reset(token)

        otlp.on_end.assert_called_once_with(span)
        rest.on_end.assert_not_called()

    def test_routes_to_rest_when_evaluating(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        token = current_trace_context.set(None)
        try:
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = True
                processor.on_end(span)
        finally:
            current_trace_context.reset(token)

        rest.on_end.assert_called_once_with(span)
        otlp.on_end.assert_not_called()

    def test_on_start_forwarded_to_both(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        processor.on_start(span, None)

        rest.on_start.assert_called_once_with(span, None)
        otlp.on_start.assert_called_once_with(span, None)

    def test_shutdown_and_force_flush_forwarded_to_both(self):
        processor, rest, otlp = self._make_processor()

        rest.force_flush.return_value = True
        otlp.force_flush.return_value = True

        assert processor.force_flush(timeout_millis=5000) is True
        rest.force_flush.assert_called_once_with(5000)
        otlp.force_flush.assert_called_once_with(5000)

        processor.shutdown()
        rest.shutdown.assert_called_once_with()
        otlp.shutdown.assert_called_once_with()


# ---------------------------------------------------------------------------
# Pytest signal: is_test_mode is gone for good.
# ---------------------------------------------------------------------------


def test_is_test_mode_kwarg_is_removed_from_settings():
    """Phase 2 hard-removed the kwarg. Calling with it must raise TypeError."""
    from deepeval.integrations.pydantic_ai.instrumentator import (
        ConfidentInstrumentationSettings,
    )

    with pytest.raises(TypeError):
        ConfidentInstrumentationSettings(api_key="dummy", is_test_mode=False)
