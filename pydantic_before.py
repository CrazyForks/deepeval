"""pydantic_before.py — today's Pydantic AI integration syntax.

Run BEFORE Phase 2 lands to capture today's behavior on Confident AI.
After Phase 2, this script will crash on `is_test_mode=False` with
``TypeError`` at import — that's the trip-wire confirming the rewrite ran.

Limitations of today's syntax (this script demonstrates them honestly):

  - Trace metadata (name, thread_id, user_id, metadata, tags) is FROZEN at
    ``DeepEvalInstrumentationSettings(...)`` instantiation time.
  - ``update_current_span(metadata=...)`` from inside ``@agent.tool_plain`` is a
    SILENT NO-OP — ``SpanInterceptor.on_start`` never pushes the OTel span onto
    ``current_span_context`` today, so there is no current span to mutate.

Both of the above are exactly what Phase 2's rewrite fixes.

Requirements:
  - ``CONFIDENT_API_KEY`` in env (or ``deepeval login``)
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import os

from pydantic_ai import Agent

from deepeval.tracing import update_current_span
from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings


settings = DeepEvalInstrumentationSettings(
    name="pydantic-ai-validation",
    thread_id="thread-123",
    user_id="user-abc",
    metadata={"source": "instrument-time"},
    tags=["before-rewrite"],
    is_test_mode=False,
)

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


def main() -> None:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise SystemExit("CONFIDENT_API_KEY is not set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    result = agent.run_sync("What's the weather in Paris?")
    print(result.output)
    print(
        "\nOpen the Confident AI dashboard. The trace will show the "
        "instrument-time metadata (user_id=user-abc, tags=[before-rewrite]). "
        "The get_weather tool span will NOT have weather_source/city metadata "
        "because update_current_span(...) is a no-op for OTel spans today."
    )


if __name__ == "__main__":
    main()
