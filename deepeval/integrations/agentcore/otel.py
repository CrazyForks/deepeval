"""``instrument_agentcore(...)`` — wire AgentCore spans into deepeval.

Mirrors the Pydantic AI POC pattern: builds a single ``TracerProvider``
that runs the ``AgentCoreSpanInterceptor`` (translates Strands /
Traceloop / GenAI attrs and pushes placeholders for
``update_current_*``) followed by ``ContextAwareSpanProcessor`` (routes
each finished span to REST when a deepeval trace context is active or
an evaluation is running, OTLP otherwise).

Span-level configuration (per-call ``metric_collection``, ``metrics``,
``prompt``, etc.) is NOT a kwarg here. Use ``with next_agent_span(...)``
/ ``with next_llm_span(...)`` / ``with next_tool_span(...)`` before
invoking the agent, or ``update_current_span(...)`` from inside a
Strands ``@tool`` body. See ``deepeval/integrations/README.md`` for
the migration table.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.telemetry import capture_tracing_integration

logger = logging.getLogger(__name__)
settings = get_settings()


try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    _opentelemetry_installed = True
except ImportError:
    _opentelemetry_installed = False


def _require_opentelemetry() -> None:
    if not _opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. "
            "Install it with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )


# Removed kwargs — preserved here for crisp error reporting in case
# callers haven't migrated yet. Mirror of
# ``AgentCoreInstrumentationSettings._REMOVED_KWARGS``.
_REMOVED_INSTRUMENT_KWARGS = (
    "is_test_mode",
    "agent_metric_collection",
    "llm_metric_collection",
    "tool_metric_collection_map",
    "trace_metric_collection",
    "agent_metrics",
    "confident_prompt",
)


def instrument_agentcore(
    api_key: Optional[str] = None,
    name: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    environment: Optional[str] = None,
    metric_collection: Optional[str] = None,
    test_case_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    **removed_kwargs,
) -> None:
    """Attach Confident AI / deepeval telemetry to AWS Bedrock AgentCore.

    All kwargs are optional and trace-level. Span-level fields
    (``metric_collection`` per agent / LLM / tool, ``metrics``,
    ``prompt``) belong on per-call ``with next_*_span(...)`` blocks or
    ``update_current_span(...)`` calls — see the integration README.

    Routing is decided per-span by ``ContextAwareSpanProcessor``:
    REST when a deepeval trace context is active (``@observe`` /
    ``with trace(...)``) or ``trace_manager.is_evaluating`` is True;
    OTLP otherwise.
    """
    if removed_kwargs:
        offending = ", ".join(sorted(removed_kwargs))
        raise TypeError(
            f"instrument_agentcore: unexpected keyword argument(s) "
            f"{offending}. Span-level kwargs were removed in the OTel "
            "POC migration. Use ``with next_agent_span(...)`` / "
            "``with next_llm_span(...)`` / ``with next_tool_span(...)`` "
            "before invoking the agent, or "
            "``update_current_span(...)`` from inside a Strands @tool "
            "body. See deepeval/integrations/README.md for details."
        )

    with capture_tracing_integration("agentcore"):
        _require_opentelemetry()

        if not api_key:
            api_key = get_confident_api_key()

        # Imported here so the OTel-dependent module load can fail
        # cleanly inside ``_require_opentelemetry`` above before the
        # interceptor pulls in the rest of the OTel SDK.
        from deepeval.tracing.otel.context_aware_processor import (
            ContextAwareSpanProcessor,
        )

        from .instrumentator import (
            AgentCoreInstrumentationSettings,
            AgentCoreSpanInterceptor,
        )

        agentcore_settings = AgentCoreInstrumentationSettings(
            api_key=api_key,
            name=name,
            thread_id=thread_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            environment=environment,
            metric_collection=metric_collection,
            test_case_id=test_case_id,
            turn_id=turn_id,
        )

        # Reuse the active TracerProvider when a real one is already
        # registered (so multiple ``instrument_*`` calls layer cleanly);
        # otherwise create one and set it globally.
        current_provider = trace.get_tracer_provider()
        if type(current_provider).__name__ in (
            "ProxyTracerProvider",
            "NoOpTracerProvider",
        ):
            tracer_provider = TracerProvider()
            try:
                trace.set_tracer_provider(tracer_provider)
                logger.debug("Created and registered a new TracerProvider.")
            except Exception as exc:
                logger.warning("Could not set global tracer provider: %s", exc)
            current_provider = trace.get_tracer_provider()

        if not hasattr(current_provider, "add_span_processor"):
            logger.warning(
                "The active TracerProvider (%s) does not support "
                "add_span_processor. AgentCore telemetry cannot be attached.",
                type(current_provider).__name__,
            )
            return

        # Order matters: the interceptor mutates ``confident.*`` attrs
        # and pushes / pops placeholders; ``ContextAwareSpanProcessor``
        # routes the final span. Since OTel runs SpanProcessors in
        # registration order on on_start AND on_end, the interceptor's
        # writes are visible to the routing processor's exporters.
        current_provider.add_span_processor(
            AgentCoreSpanInterceptor(agentcore_settings)
        )
        current_provider.add_span_processor(
            ContextAwareSpanProcessor(api_key=api_key)
        )

        logger.info(
            "Confident AI AgentCore telemetry attached (env=%s).",
            agentcore_settings.environment,
        )
