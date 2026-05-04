"""pydantic_after_threads.py — multi-thread validation.

Three concurrent ``agent.run_sync(...)`` calls submitted to a
``ThreadPoolExecutor``. Each worker thread starts its own ``@observe`` scope
and sets its own trace metadata + tool span metadata.

Validates that:

  - ``ContextVar`` state in the main thread does NOT bleed into worker
    threads (Python's ``ThreadPoolExecutor.submit`` does not copy
    contextvars by default — each worker starts with the empty default).
  - Each worker independently establishes its own ``current_trace_context``
    via ``@observe``, so spans created during ``agent.run_sync`` are
    correctly attributed to the right per-thread trace.
  - ``trace_manager``'s shared ``active_traces`` / ``active_spans`` /
    worker queue is thread-safe under concurrent posts (multiple workers
    finishing at roughly the same time enqueue independent traces).
  - ``current_trace_context`` propagates correctly through pydantic-ai's
    internal thread bridge: ``agent.run_sync`` starts an asyncio event
    loop on the calling worker, then dispatches sync tool bodies via
    ``anyio.to_thread.run_sync(...)`` onto an *anyio* worker thread.
    anyio captures the current ``contextvars.Context`` and runs the tool
    inside ``ctx.run(...)`` — so the tool body, despite running in an
    "AnyIO worker thread" rather than ``agent-worker_X``, must still see
    THIS worker's trace context.

Why we don't compare ``thread_name``
------------------------------------
The trace ``thread_name`` is the ``agent-worker_X`` that ran ``@observe``,
but the tool span ``thread_name`` is always ``"AnyIO worker thread"``
(pydantic-ai's internal anyio bridge). They will NEVER match in any run,
correct or buggy — so thread name comparison is not a useful isolation
check. Instead, we have the tool body actively READ
``current_trace_context.get().metadata['request_id']`` and stamp it onto
the span as ``request_id_from_trace_ctx``. The real check is:

    trace.metadata.request_id == get_weather span metadata.request_id_from_trace_ctx

If those two match within a single trace, contextvar propagation worked
correctly across the worker → anyio thread bridge for THAT trace. Across
all 3 traces, the request_ids must remain distinct (no cross-worker
leakage).

NOTE: If you ever need to inherit the main thread's ``ContextVar`` state
into a worker thread, wrap the submitted callable in
``contextvars.copy_context().run(fn, *args)`` — this script deliberately
does NOT do that, so each worker establishes its own context from scratch
via ``@observe``.

Requirements:
  - ``CONFIDENT_API_KEY`` in env
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

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
    # Active isolation check: read the current trace's request_id from
    # contextvars at tool invocation time and stamp it onto the span. This
    # tool body runs in pydantic-ai's "AnyIO worker thread" (NOT the
    # agent-worker_X thread that's driving the run), so this validates
    # that anyio carried the agent-worker_X thread's contextvars across
    # the thread boundary correctly.
    trace = current_trace_context.get()
    request_id_from_trace_ctx = None
    if trace and trace.metadata:
        request_id_from_trace_ctx = trace.metadata.get("request_id")

    update_current_span(
        metadata={
            "weather_source": "mock",
            "city": city,
            "request_id_from_trace_ctx": request_id_from_trace_ctx,
            "tool_thread_name": threading.current_thread().name,
        },
    )
    return f"{city}: sunny, 24C"


@observe(type="agent")
def answer(query: str, user_id: str, request_id: str) -> str:
    update_current_trace(
        name=f"pydantic-ai-threads-{user_id}",
        user_id=user_id,
        tags=["threads", "after-rewrite"],
        metadata={
            "request_id": request_id,
            "observe_thread_name": threading.current_thread().name,
        },
    )
    return agent.run_sync(query).output


def main() -> None:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise SystemExit("CONFIDENT_API_KEY is not set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    queries = [
        ("What's the weather in Paris?", "user-paris", "req-paris-101"),
        ("What's the weather in Tokyo?", "user-tokyo", "req-tokyo-102"),
        ("What's the weather in Rio?", "user-rio", "req-rio-103"),
    ]

    with ThreadPoolExecutor(
        max_workers=3, thread_name_prefix="agent-worker"
    ) as pool:
        futures = [
            pool.submit(answer, q, uid, rid) for q, uid, rid in queries
        ]
        for fut in futures:
            print(fut.result())

    print(
        "\n========================================================\n"
        "  ISOLATION CHECK — what to verify in the dashboard\n"
        "========================================================\n"
        "You should see 3 traces:\n"
        "  - pydantic-ai-threads-user-paris\n"
        "  - pydantic-ai-threads-user-tokyo\n"
        "  - pydantic-ai-threads-user-rio\n"
        "\n"
        "For EACH trace, click into it, find the get_weather tool span,\n"
        "and verify these two values MATCH:\n"
        "\n"
        "    trace metadata.request_id\n"
        "        ==\n"
        "    get_weather span metadata.request_id_from_trace_ctx\n"
        "\n"
        "Expected pairs:\n"
        "  user-paris trace → both should be 'req-paris-101'\n"
        "  user-tokyo trace → both should be 'req-tokyo-102'\n"
        "  user-rio   trace → both should be 'req-rio-103'\n"
        "\n"
        "If the request_ids match within each trace, contextvars\n"
        "propagated correctly through pydantic-ai's anyio thread bridge\n"
        "AND there was no cross-worker leak.\n"
        "\n"
        "FYI on thread names (these will NOT match — that's expected):\n"
        "  - trace metadata.observe_thread_name = 'agent-worker_X'\n"
        "      (one of your ThreadPoolExecutor workers)\n"
        "  - tool span metadata.tool_thread_name = 'AnyIO worker thread'\n"
        "      (pydantic-ai's internal anyio worker)\n"
        "These differ because pydantic-ai dispatches sync tool bodies\n"
        "through anyio.to_thread.run_sync — not a bug.\n"
        "========================================================"
    )


if __name__ == "__main__":
    main()
