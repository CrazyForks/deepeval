"""pydantic_after_threads.py — multi-thread validation, BARE agent.run.

Three concurrent ``agent.run_sync(...)`` calls submitted to a
``ThreadPoolExecutor``. NO ``@observe`` / ``with trace(...)`` wrapper —
every trace ships through the implicit ``Trace`` placeholder that
``SpanInterceptor`` pushes onto ``current_trace_context`` at the root
OTel span's on_start.

This validates several things at once:

  - ``ContextVar`` state in the main thread does NOT bleed into worker
    threads (Python's ``ThreadPoolExecutor.submit`` does not copy
    contextvars by default — each worker starts with the empty default).
  - Each worker's ``agent.run_sync(...)`` call independently triggers an
    implicit ``Trace`` placeholder push inside ``SpanInterceptor.on_start``,
    so spans created during that worker's run are correctly attributed
    to its own per-thread trace.
  - ``trace_manager``'s shared ``active_traces`` / ``active_spans`` /
    worker queue is thread-safe under concurrent posts (multiple workers
    finishing at roughly the same time enqueue independent traces).
  - Both the user's ``_request_ctx`` AND deepeval's implicit Trace
    placeholder propagate correctly through pydantic-ai's internal
    thread bridge: ``agent.run_sync`` starts an asyncio event loop on
    the calling worker, then dispatches sync tool bodies via
    ``anyio.to_thread.run_sync(...)`` onto an *anyio* worker thread.
    anyio captures the current ``contextvars.Context`` and runs the
    tool inside ``ctx.run(...)`` — so the tool body, despite running
    on "AnyIO worker thread" rather than ``agent-worker_X``, must still
    see THIS worker's request_ctx + implicit trace placeholder.

How to verify isolation in the dashboard
----------------------------------------
A separate ``_request_ctx`` ContextVar carries per-request data
(``user_id`` / ``request_id``) set in the worker function BEFORE
``agent.run_sync(...)``. The tool body reads it and:

  - calls ``update_current_trace(...)`` to stamp the trace with the
    request_id (lands on the implicit placeholder)
  - calls ``update_current_span(...)`` to stamp ``request_id_from_request_ctx``
    onto the tool span

Each trace must show:

    trace.metadata.request_id == get_weather span.metadata.request_id_from_request_ctx

If those don't match, ``_request_ctx`` leaked across worker threads OR
across the worker → AnyIO thread bridge. Across all 3 traces, the
request_ids must be distinct (no cross-worker leakage of either
``_request_ctx`` OR the implicit Trace placeholder).

Why we don't compare ``thread_name``
------------------------------------
The trace runs in ``agent-worker_X`` but the tool span runs on
``"AnyIO worker thread"`` (pydantic-ai's internal anyio bridge). They
will NEVER match in any run, correct or buggy — so thread name
comparison is not a useful isolation check. We do still record both
thread names purely as informational metadata.

NOTE: If you ever need to inherit the main thread's ``ContextVar`` state
into a worker thread, wrap the submitted callable in
``contextvars.copy_context().run(fn, *args)`` — this script deliberately
does NOT do that, so each worker establishes its own contextvars from
scratch (and the implicit Trace placeholder is pushed independently
inside each worker's agent.run_sync call).

Requirements:
  - ``CONFIDENT_API_KEY`` in env
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import contextvars
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from pydantic_ai import Agent

from deepeval.tracing import (
    update_current_span,
    update_current_trace,
)
from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings


# Unique per-script-run id so all 3 traces produced by this run can be
# grouped in the dashboard via metadata.run_id.
RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


# Per-request ContextVar carrying the data the tool body needs to stamp
# onto the implicit trace + span. Set in the worker function before
# agent.run_sync; read inside the tool (which runs on AnyIO worker
# thread but inherits the calling worker's context via anyio).
_request_ctx: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "_request_ctx", default={}
)


settings = DeepEvalInstrumentationSettings()

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
            "tool_thread_name": threading.current_thread().name,
        },
    )
    # Trace-level mutation: works without @observe because
    # SpanInterceptor pushes an implicit Trace placeholder at the root
    # OTel span's on_start. THIS is what's new.
    update_current_trace(
        name=f"pydantic-ai-bare-threads-{user_id}",
        user_id=user_id,
        tags=["bare", "threads", "after-rewrite"],
        metadata={
            "request_id": request_id,
            "run_id": RUN_ID,
            "worker_thread_name": req.get("worker_thread_name"),
        },
    )
    return f"{city}: sunny, 24C"


def answer(query: str, user_id: str, request_id: str) -> str:
    # Set the per-request data into THIS worker thread's contextvars.
    # ThreadPoolExecutor.submit doesn't copy contextvars from the main
    # thread, so each worker starts with the empty default and stamps
    # its own values here. The AnyIO worker thread that runs the tool
    # body inherits these via anyio's ctx.run(...).
    _request_ctx.set(
        {
            "user_id": user_id,
            "request_id": request_id,
            "worker_thread_name": threading.current_thread().name,
        }
    )
    # Bare agent.run_sync — no @observe, no `with trace(...)`. The
    # implicit Trace placeholder gets pushed inside SpanInterceptor's
    # on_start when the root OTel span opens within this worker.
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
        futures = [pool.submit(answer, q, uid, rid) for q, uid, rid in queries]
        for fut in futures:
            print(fut.result())

    print(
        "\n========================================================\n"
        "  ISOLATION CHECK — what to verify in the dashboard\n"
        "========================================================\n"
        f"All 3 traces from this run share metadata.run_id = '{RUN_ID}'\n"
        "(filter by it in the dashboard to scope to just this run).\n\n"
        "You should see 3 traces (NONE wrapped in @observe):\n"
        "  - pydantic-ai-bare-threads-user-paris\n"
        "  - pydantic-ai-bare-threads-user-tokyo\n"
        "  - pydantic-ai-bare-threads-user-rio\n"
        "\n"
        "For EACH trace, click into it, find the get_weather tool span,\n"
        "and verify these two values MATCH:\n"
        "\n"
        "    trace metadata.request_id\n"
        "        ==\n"
        "    get_weather span metadata.request_id_from_request_ctx\n"
        "\n"
        "Expected pairs:\n"
        "  user-paris trace → both should be 'req-paris-101'\n"
        "  user-tokyo trace → both should be 'req-tokyo-102'\n"
        "  user-rio   trace → both should be 'req-rio-103'\n"
        "\n"
        "If the request_ids match within each trace, contextvars\n"
        "(both _request_ctx AND deepeval's implicit Trace placeholder)\n"
        "propagated correctly through pydantic-ai's anyio thread bridge\n"
        "AND there was no cross-worker leak.\n"
        "\n"
        "FYI on thread names (these will NOT match — that's expected):\n"
        "  - trace metadata.worker_thread_name = 'agent-worker_X'\n"
        "      (one of your ThreadPoolExecutor workers)\n"
        "  - tool span metadata.tool_thread_name = 'AnyIO worker thread'\n"
        "      (pydantic-ai's internal anyio worker)\n"
        "These differ because pydantic-ai dispatches sync tool bodies\n"
        "through anyio.to_thread.run_sync — not a bug.\n"
        "========================================================"
    )


if __name__ == "__main__":
    main()
