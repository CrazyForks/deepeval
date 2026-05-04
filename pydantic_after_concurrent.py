"""pydantic_after_concurrent.py — concurrent async tasks, BARE agent.run.

Three concurrent ``await agent.run(...)`` calls fired off via
``asyncio.gather(...)``. NO ``@observe`` / ``with trace(...)`` wrapper —
every trace ships through the implicit ``Trace`` placeholder that
``SpanInterceptor`` pushes onto ``current_trace_context`` at the root
OTel span's on_start.

This validates two things at once:

  1. ``update_current_trace(...)`` works from inside a tool body even
     without ``@observe``, in concurrent asyncio tasks (proves the
     implicit placeholder is per-task and never leaks across tasks).
  2. The deepeval span placeholder + trace placeholder pair propagates
     correctly across pydantic-ai's anyio thread bridge for sync tools.

How to verify isolation in the dashboard
----------------------------------------
A separate ``_request_ctx`` ContextVar carries per-request data
(``user_id`` / ``request_id``) set in the coroutine BEFORE
``agent.run(...)``. The tool body reads it and:

  - calls ``update_current_trace(...)`` to stamp the trace with the
    request_id (lands on the implicit placeholder)
  - calls ``update_current_span(...)`` to stamp ``request_id_from_request_ctx``
    onto the tool span

Each trace must show:

    trace.metadata.request_id == get_weather span.metadata.request_id_from_request_ctx

If those don't match, ``_request_ctx`` leaked across asyncio tasks. Across
all 3 traces, the request_ids must be distinct (no cross-task leakage of
either ``_request_ctx`` OR the implicit Trace placeholder).

Requirements:
  - ``CONFIDENT_API_KEY`` in env
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import asyncio
import contextvars
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from pydantic_ai import Agent

from deepeval.tracing import (
    update_current_span,
    update_current_trace,
)
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


# Unique per-script-run id so all 3 traces produced by this run can be
# grouped in the dashboard via metadata.run_id.
RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


# Per-request ContextVar carrying the data the tool body needs to stamp
# onto the implicit trace + span. Set in the coroutine before agent.run;
# read inside the tool. This is the "user app's own contextvar" — what
# we're proving is that BOTH this user contextvar AND deepeval's
# implicit Trace placeholder maintain per-asyncio-task isolation.
_request_ctx: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "_request_ctx", default={}
)


settings = ConfidentInstrumentationSettings()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise. One short sentence.",
    instrument=settings,
)


@agent.tool_plain
def get_weather(city: str) -> str:
    req = _request_ctx.get()
    user_id = req.get("user_id")
    request_id = req.get("request_id")

    # Span-level mutation: works without @observe because
    # SpanInterceptor pushes a BaseSpan placeholder at on_start.
    update_current_span(
        metadata={
            "weather_source": "mock",
            "city": city,
            "request_id_from_request_ctx": request_id,
        },
    )
    # Trace-level mutation: works without @observe because
    # SpanInterceptor pushes an implicit Trace placeholder at the root
    # OTel span's on_start. THIS is what's new.
    update_current_trace(
        name=f"pydantic-ai-bare-concurrent-{user_id}",
        user_id=user_id,
        tags=["bare", "concurrent", "after-rewrite"],
        metadata={"request_id": request_id, "run_id": RUN_ID},
    )
    return f"{city}: sunny, 24C"


async def answer(query: str, user_id: str, request_id: str) -> str:
    # Set the per-request data into THIS task's contextvar copy. Each
    # asyncio.gather task gets its own context snapshot (Python 3.7+),
    # so this never leaks into the other tasks.
    _request_ctx.set({"user_id": user_id, "request_id": request_id})
    # Bare agent.run — no @observe, no `with trace(...)`. The trace
    # context is established by SpanInterceptor's implicit placeholder
    # push at the root OTel span's on_start.
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
        f"All 3 traces from this run share metadata.run_id = '{RUN_ID}'\n"
        "(filter by it in the dashboard to scope to just this run).\n\n"
        "You should see 3 traces (NONE wrapped in @observe):\n"
        "  - pydantic-ai-bare-concurrent-user-paris\n"
        "  - pydantic-ai-bare-concurrent-user-tokyo\n"
        "  - pydantic-ai-bare-concurrent-user-rio\n"
        "\n"
        "For EACH trace, click into it, find the get_weather tool span,\n"
        "and verify these two values MATCH:\n"
        "\n"
        "    trace metadata.request_id\n"
        "        ==\n"
        "    get_weather span metadata.request_id_from_request_ctx\n"
        "\n"
        "Expected pairs:\n"
        "  user-paris trace → both should be 'req-paris-001'\n"
        "  user-tokyo trace → both should be 'req-tokyo-002'\n"
        "  user-rio   trace → both should be 'req-rio-003'\n"
        "\n"
        "If any trace shows mismatched values, either:\n"
        "  - _request_ctx leaked across asyncio tasks (user contextvar bug), OR\n"
        "  - the implicit Trace placeholder leaked across tasks\n"
        "    (deepeval bug — SpanInterceptor's per-root push is broken).\n"
        "\n"
        "Across all 3 traces, ALL request_ids and user_ids must be distinct.\n"
        "========================================================"
    )


if __name__ == "__main__":
    asyncio.run(main())
