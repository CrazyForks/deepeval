"""pydantic_after_async.py — pure async path validation.

Same goals as ``pydantic_after.py`` but uses ``await agent.run(...)`` with
an ``async def`` tool function — i.e. the **pure async** path with NO
thread bridge anywhere.

Why this matters: pydantic-ai's ``@agent.tool_plain`` accepts either a
sync or async function (per its docstring: "Can decorate a sync or async
functions."). The two paths exercise different ContextVar propagation
semantics:

  - **sync tool body inside async agent.run** — tool body is scheduled via
    ``anyio.to_thread.run_sync(...)``, which explicitly captures and runs
    the callable inside ``ctx.run(...)``. ContextVars cross the thread
    boundary because anyio carries them.
  - **async tool body inside async agent.run (this file)** — tool body
    runs directly in the event loop, in the SAME asyncio task as the agent
    run. No thread bridge. ContextVar inheritance is just normal Python
    task semantics.

Validates that ``update_current_trace(...)`` and ``update_current_span(...)``
both land via the pure async path. (The anyio-bridge case is covered by
``pydantic_after_concurrent.py``, which uses sync tools under
``await agent.run`` across multiple concurrent tasks.)

Requirements:
  - ``CONFIDENT_API_KEY`` in env
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import asyncio
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
# grouped in the dashboard via metadata.run_id.
RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


settings = DeepEvalInstrumentationSettings()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise. One short sentence.",
    instrument=settings,
)


@agent.tool_plain
async def get_weather(city: str) -> str:
    # Simulate a real async tool: yield to the event loop, then write
    # span metadata via update_current_span. Runs in the same asyncio
    # task as the @observe wrapper — no thread bridge.
    await asyncio.sleep(0)
    update_current_span(
        metadata={"weather_source": "mock-async", "city": city},
    )
    return f"{city}: sunny, 24C"


@observe(type="agent")
async def answer(query: str, request_id: str) -> str:
    update_current_trace(
        name="pydantic-ai-async-validation",
        thread_id="thread-async-123",
        user_id="user-async",
        tags=["async-pure", "after-rewrite"],
        metadata={
            "source": "runtime-async",
            "request_id": request_id,
            "run_id": RUN_ID,
        },
    )
    result = await agent.run(query)
    return result.output


async def main() -> None:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise SystemExit("CONFIDENT_API_KEY is not set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    output = await answer(
        "What's the weather in Paris?", request_id="req-async-001"
    )
    print(output)
    print(
        f"\nAll traces from this run share metadata.run_id = '{RUN_ID}'.\n"
        "Open the Confident AI dashboard. The trace should show:\n"
        "  - name: pydantic-ai-async-validation\n"
        "  - user_id=user-async, thread_id=thread-async-123\n"
        "  - tags=[async-pure, after-rewrite]\n"
        "  - metadata.request_id=req-async-001\n"
        "  - get_weather (async) tool span: "
        "metadata.weather_source=mock-async, metadata.city=Paris\n"
        "Single REST POST. No OTLP duplication. No thread bridge in this "
        "path — pure async ContextVar inheritance."
    )


if __name__ == "__main__":
    asyncio.run(main())
