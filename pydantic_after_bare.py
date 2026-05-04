"""pydantic_after_bare.py — bare ``agent.run`` (no @observe / with trace).

Validates that ``update_current_trace(...)`` AND ``update_current_span(...)``
both work even when the caller does NOT wrap in ``@observe`` or
``with trace(...)``. This is the symmetric pair to the existing
``pydantic_after.py``: same dynamic-context ergonomics, no enclosing
deepeval trace context.

What this proves:

  - ``SpanInterceptor`` pushes an implicit ``Trace`` placeholder onto
    ``current_trace_context`` at the OTel root span's on_start so that
    ``update_current_trace(...)`` from anywhere in the call stack has a
    target to mutate.
  - The placeholder is tagged ``is_otel_implicit=True`` so
    ``ContextAwareSpanProcessor`` keeps routing to OTLP — bare callers
    get OTLP behavior, NOT REST.
  - Mutations to the implicit placeholder are picked up automatically by
    ``_serialize_trace_context_to_otel_attrs`` at every ``on_end``, so
    every OTel span (root + children) ships with the latest values.
  - ``update_current_span(...)`` from inside ``@agent.tool_plain``
    continues to land on the tool span (proven by ``pydantic_after.py``;
    re-verified here for completeness).

Requirements:
  - ``CONFIDENT_API_KEY`` in env (or ``deepeval login``)
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import os
import uuid
from pathlib import Path

from pydantic_ai import Agent

from deepeval.tracing import (
    update_current_span,
    update_current_trace,
)
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


# Unique per-script-run id so every trace produced by this run can be
# grouped in the dashboard via metadata.run_id.
RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


settings = ConfidentInstrumentationSettings()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise. One short sentence.",
    instrument=settings,
)


@agent.tool_plain
def get_weather(city: str) -> str:
    # Span-level mutation from inside a tool — works without @observe
    # because SpanInterceptor.on_start pushes a BaseSpan placeholder
    # onto current_span_context for every OTel span.
    update_current_span(
        metadata={"weather_source": "mock", "city": city},
    )
    # Trace-level mutation from inside a tool — works without @observe
    # because SpanInterceptor.on_start pushes an implicit Trace
    # placeholder onto current_trace_context at the root OTel span.
    # Without that implicit placeholder, this call would silently no-op.
    update_current_trace(
        name="pydantic-ai-bare-validation",
        user_id="user-bare",
        tags=["after-rewrite", "bare"],
        metadata={
            "source": "tool",
            "run_id": RUN_ID,
            "city_in_tool": city,
            "trace_mutated_from_tool": True,
        },
    )
    return f"{city}: sunny, 24C"


def main() -> None:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise SystemExit("CONFIDENT_API_KEY is not set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    # No @observe, no `with trace(...)`. Bare agent.run_sync — the
    # implicit Trace placeholder push happens inside SpanInterceptor's
    # on_start when the root OTel span opens.
    output_1 = agent.run_sync("What's the weather in Paris?").output
    # Second query verifies isolation across sequential bare calls
    # (each call gets its own root span and its own implicit Trace).
    output_2 = agent.run_sync("What's the weather in Tokyo?").output

    print(output_1)
    print(output_2)
    print(
        f"\nAll traces from this run share metadata.run_id = '{RUN_ID}' "
        "(stamped by the tool via update_current_trace).\n"
        "Open the Confident AI dashboard. Even though the script never "
        "used @observe or `with trace(...)`, each trace will show:\n"
        "  - user_id = user-bare\n"
        "  - tags = [after-rewrite, bare]\n"
        "  - metadata.trace_mutated_from_tool = True\n"
        "  - metadata.city_in_tool = Paris (or Tokyo for the 2nd call)\n"
        "  - the get_weather tool span will carry "
        "metadata.weather_source=mock and metadata.city=<city>"
    )


if __name__ == "__main__":
    main()
