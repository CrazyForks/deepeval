"""pydantic_after_concurrent.py — concurrent async tasks validation.

Three concurrent ``await agent.run(...)`` calls fired off via
``asyncio.gather(...)``. Each runs inside its own ``@observe`` scope and
sets its own ``user_id`` / ``request_id``.

Validates that ``ContextVar`` isolation between asyncio tasks works: the
trace and span context for task A doesn't bleed into task B even though
all three are racing through the same ``SpanInterceptor`` /
``ContextAwareSpanProcessor`` instance. ``asyncio.gather`` wraps each
coroutine in a task at gather time; tasks inherit the parent context as
a SNAPSHOT (Python 3.7+), so subsequent ``current_trace_context.set(...)``
inside ``@observe`` only affects that one task's view.

How to verify isolation in the dashboard
----------------------------------------
The tool function actively READS the current trace's ``request_id`` from
``current_trace_context`` and stamps it onto the tool span's metadata as
``request_id_from_trace_ctx``. This means each trace must show:

    trace.metadata.request_id == get_weather span metadata.request_id_from_trace_ctx

If those two values don't match within a single trace, ContextVar
isolation has regressed — task A's tool body saw task B's trace context.

We do NOT rely on visually matching trace name → tool span ``city``,
because that's correlated by the LLM's choice (which is usually right
even if isolation is broken — the bug would slip through).

Requirements:
  - ``CONFIDENT_API_KEY`` in env
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import asyncio
import os

from pydantic_ai import Agent

from deepeval.tracing import (
    observe,
    update_current_span,
    update_current_trace,
)
from deepeval.tracing.context import current_trace_context
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


settings = ConfidentInstrumentationSettings()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise. One short sentence.",
    instrument=settings,
)


@agent.tool_plain
def get_weather(city: str) -> str:
    # Active isolation check: read the current trace's request_id at tool
    # invocation time and write it onto the span. If contextvars leak
    # across asyncio tasks, this will be the WRONG request_id for the
    # trace this span ends up in.
    trace = current_trace_context.get()
    request_id_from_trace_ctx = None
    if trace and trace.metadata:
        request_id_from_trace_ctx = trace.metadata.get("request_id")

    update_current_span(
        metadata={
            "weather_source": "mock",
            "city": city,
            "request_id_from_trace_ctx": request_id_from_trace_ctx,
        },
    )
    return f"{city}: sunny, 24C"


@observe(type="agent")
async def answer(query: str, user_id: str, request_id: str) -> str:
    update_current_trace(
        name=f"pydantic-ai-concurrent-{user_id}",
        user_id=user_id,
        tags=["concurrent", "after-rewrite"],
        metadata={"request_id": request_id},
    )
    result = await agent.run(query)
    return result.output


async def main() -> None:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise SystemExit("CONFIDENT_API_KEY is not set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    queries = [
        ("What's the weather in Paris?", "user-paris", "req-paris-001"),
        ("What's the weather in Tokyo?", "user-tokyo", "req-tokyo-002"),
        ("What's the weather in Rio?", "user-rio", "req-rio-003"),
    ]

    results = await asyncio.gather(
        *(answer(q, uid, rid) for q, uid, rid in queries)
    )

    for (q, uid, rid), out in zip(queries, results):
        print(f"[{uid} | {rid}] {out}")

    print(
        "\n========================================================\n"
        "  ISOLATION CHECK — what to verify in the dashboard\n"
        "========================================================\n"
        "You should see 3 traces:\n"
        "  - pydantic-ai-concurrent-user-paris\n"
        "  - pydantic-ai-concurrent-user-tokyo\n"
        "  - pydantic-ai-concurrent-user-rio\n"
        "\n"
        "For EACH trace, click into it, find the get_weather tool span,\n"
        "and verify these two values MATCH:\n"
        "\n"
        "    trace metadata.request_id\n"
        "        ==\n"
        "    get_weather span metadata.request_id_from_trace_ctx\n"
        "\n"
        "Expected pairs:\n"
        "  user-paris trace → both should be 'req-paris-001'\n"
        "  user-tokyo trace → both should be 'req-tokyo-002'\n"
        "  user-rio   trace → both should be 'req-rio-003'\n"
        "\n"
        "If any trace shows a request_id_from_trace_ctx that DOESN'T\n"
        "match its trace's request_id, asyncio task ContextVar isolation\n"
        "has regressed — the tool body saw a different task's trace ctx.\n"
        "========================================================"
    )


if __name__ == "__main__":
    asyncio.run(main())
