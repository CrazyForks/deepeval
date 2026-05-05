"""pydantic_after.py — post-Phase-2 Pydantic AI integration syntax.

Run AFTER Phase 2 lands to confirm the new behavior on Confident AI.

Post-rewrite ergonomics (Langfuse-style):

  - ``DeepEvalInstrumentationSettings`` takes only essentials (api_key from
    env). All trace and span metadata is set at runtime via
    ``update_current_trace(...)`` and ``update_current_span(...)``.
  - Both helpers can be called from ANYWHERE in the call stack, including
    deeply nested helpers and ``@agent.tool_plain`` functions, and the values
    flow into the active OTel trace / span attributes.
  - When wrapped in ``@observe`` (or any active ``current_trace_context``),
    spans are routed via REST through ``trace_manager`` — no OTLP duplication,
    no double-posting.

Run BEFORE Phase 2: the script imports and runs, but the trace on Confident AI
will be missing user_id, tags, request_id, and the per-tool span metadata —
because today's ``SpanInterceptor`` only reads name/thread_id/metadata from
``current_trace_context`` and never pushes the OTel span onto
``current_span_context``.

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
    observe,
    update_current_span,
    update_current_trace,
)
from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings


# Unique per-script-run id so every trace produced by this run can be
# grouped in the dashboard via metadata.run_id (more reliable than
# filtering by timestamp).
RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


settings = DeepEvalInstrumentationSettings()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise. One short sentence.",
    instrument=settings,
)


@agent.tool_plain
def get_weather(city: str) -> str:
    update_current_span(
        metadata={"weather_source": "mock", "city": city},
    )
    return f"{city}: sunny, 24C"


@observe(type="agent")
def answer(query: str, request_id: str) -> str:
    update_current_trace(
        name="pydantic-ai-validation",
        thread_id="thread-123",
        user_id="user-abc",
        tags=["after-rewrite"],
        metadata={
            "source": "runtime",
            "request_id": request_id,
            "run_id": RUN_ID,
        },
    )
    return agent.run_sync(query).output


def main() -> None:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise SystemExit("CONFIDENT_API_KEY is not set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    output = answer("What's the weather in Paris?", request_id="req-001")
    print(output)
    print(
        f"\nAll traces from this run share metadata.run_id = '{RUN_ID}'.\n"
        "Open the Confident AI dashboard. The trace will show the "
        "runtime metadata (user_id=user-abc, tags=[after-rewrite], "
        "metadata.request_id=req-001) AND the get_weather tool span will "
        "carry metadata.weather_source=mock and metadata.city=Paris."
    )


if __name__ == "__main__":
    main()
