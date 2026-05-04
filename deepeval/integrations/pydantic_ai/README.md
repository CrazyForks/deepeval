# DeepEval × Pydantic AI integration

End-to-end reference for running [Pydantic AI](https://ai.pydantic.dev/)
agents with DeepEval / Confident AI tracing. Covers the public API,
every supported usage mode, all edge cases we know about, and the
exact rules for how trace and span attributes are resolved.

This document is for engineers extending or debugging the integration.
For a 5-minute getting-started guide, see the
[Confident AI Pydantic AI docs](https://www.confident-ai.com/docs/integrations/frameworks/pydanticai).

---

## Table of contents

- [Architecture in 60 seconds](#architecture-in-60-seconds)
- [Public API surface](#public-api-surface)
- [The three execution modes](#the-three-execution-modes)
- [Push vs pull configuration](#push-vs-pull-configuration)
- [What lives where (configuration matrix)](#what-lives-where-configuration-matrix)
- [Configuring traces](#configuring-traces)
- [Configuring spans](#configuring-spans)
- [Resolution & precedence](#resolution--precedence)
- [Routing: REST vs OTLP](#routing-rest-vs-otlp)
- [Concurrency: asyncio, threads, sub-contexts](#concurrency-asyncio-threads-sub-contexts)
- [Edge cases and pitfalls](#edge-cases-and-pitfalls)
- [Application patterns](#application-patterns)
- [Field reference](#field-reference)
- [Validation scripts](#validation-scripts)

---

## Architecture in 60 seconds

```
                       ┌─────────────────────────────────────────┐
   user code           │  Agent(instrument=ConfidentInstrumenta- │
                       │     tionSettings(...))                  │
                       │  agent.run_sync("...")                  │
                       └──────────────────┬──────────────────────┘
                                          │ pydantic-ai opens OTel spans
                                          ▼
                       ┌─────────────────────────────────────────┐
                       │  TracerProvider                         │
                       │   • SpanInterceptor (this integration)  │
                       │   • ContextAwareSpanProcessor           │
                       └──────────────────┬──────────────────────┘
                                          │
              ┌───────────────────────────┴────────────────────────────┐
              │                                                        │
              ▼                                                        ▼
   ┌─────────────────────┐                                ┌────────────────────────┐
   │ SpanInterceptor      │   reads/writes              │ ContextAwareSpanProcessor│
   │   • classify span    │   ───────────►              │   • routes to REST when  │
   │   • push placeholder │                              │     a deepeval trace    │
   │     onto             │   placeholder                │     context is active   │
   │     current_span_ctx │   visible to                 │   • routes to OTLP      │
   │   • consume          │   user code                  │     otherwise           │
   │     next_*_span      │                              └────────────────────────┘
   │     payloads         │
   │   • on_end:          │
   │     serialize back   │
   │     to confident.*   │
   │     OTel attrs       │
   └─────────────────────┘
```

`ConfidentInstrumentationSettings` does the wiring (`TracerProvider`
creation, processor registration, global-tracer-provider set,
forwarding to pydantic-ai's `Agent(instrument=...)`). It also carries
trace-level defaults.

`SpanInterceptor` is a custom OTel `SpanProcessor`. It runs
synchronously inline with span open/close; it does **not** export. Its
job is to:

1. Classify the OTel span type (`agent` / `llm` / `tool` / other) by
   reading pydantic-ai's `gen_ai.*` attributes.
2. Push a `BaseSpan` / `AgentSpan` placeholder onto
   `current_span_context`, and an implicit `Trace` placeholder onto
   `current_trace_context` for bare callers (root span only).
3. Consume any `next_*_span(...)` defaults the user staged.
4. At `on_end`, serialize all user-mutated fields (from
   `update_current_*` and the `next_*` payload) back into
   `confident.span.*` / `confident.trace.*` OTel attributes so the
   exporter — REST or OTLP — picks them up.

`ContextAwareSpanProcessor` is the routing decision. It does not look
at attributes; it decides REST vs OTLP based on whether a deepeval
trace context is active and whether the current trace is "implicit"
(more in [Routing](#routing-rest-vs-otlp)).

---

## Public API surface

```python
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings
```

That's the only symbol exported by this module. Everything else is the
generic deepeval tracing API:

```python
from deepeval.tracing import (
    # active-context mutation (push-style, requires a live context)
    update_current_trace,
    update_current_span,
    update_llm_span,
    update_retriever_span,
    update_agent_span,
    update_tool_span,

    # deferred staging (pull-style, no live context needed)
    next_span,
    next_agent_span,
    next_llm_span,
    next_tool_span,
    next_retriever_span,

    # explicit context entry
    trace,        # `with trace(...)` context manager
    observe,      # @observe decorator
)
```

---

## The three execution modes

Every Pydantic AI agent invocation runs in one of three modes,
distinguished by what (if anything) wraps the call.

### Mode 1: Bare `agent.run` / `agent.run_sync`

```python
agent = Agent("openai:gpt-4o-mini", instrument=ConfidentInstrumentationSettings())
result = agent.run_sync("hello")
```

- **No** enclosing `@observe` or `with trace(...)`.
- The user has not pushed any deepeval trace context.
- `SpanInterceptor.on_start` for the OTel root span pushes an
  *implicit* `Trace` placeholder onto `current_trace_context`, tagged
  `is_otel_implicit=True`. This placeholder exists only so that
  `update_current_trace(...)` from inside a tool body has something to
  mutate; the value flows back to the OTel attributes via the standard
  on_end serialization.
- Routing: **OTLP**. The `is_otel_implicit=True` tag tells
  `ContextAwareSpanProcessor` to ignore this trace context for routing
  purposes — bare callers are opted in to OTLP.
- The implicit placeholder is popped at the root span's on_end. Outside
  the call, `current_trace_context` is back to `None`.

### Mode 2: `with trace(...)` wrapper

```python
with trace(name="my-trace", user_id="u1"):
    agent.run_sync("hello")
```

- The user pushes their own real `Trace` (via the
  `with trace(...)` context manager).
- `is_otel_implicit=False` on this Trace.
- `SpanInterceptor` sees a non-None `current_trace_context` at on_start
  and skips the implicit-placeholder push (it doesn't clobber the
  user's trace).
- Routing: **REST**. A non-implicit trace context tells
  `ContextAwareSpanProcessor` to ship via deepeval's REST API.

### Mode 3: `@observe` decorator

```python
@observe(name="my-handler")
def handle(query: str) -> str:
    return agent.run_sync(query).output
```

- Equivalent to Mode 2 from the integration's perspective: pushes a
  real `Trace` onto `current_trace_context`, REST routing.
- Adds an outer deepeval-managed span around the agent invocation, so
  the trace tree shows: `handle → agent → llm → tool`.

---

## Push vs pull configuration

This is the most important distinction in the API. Once you internalize
it, every "why doesn't my call do anything" question answers itself.

### Push: `update_current_*`

- **Mutates the active context.** Reads `current_trace_context.get()`
  / `current_span_context.get()`, writes to that object.
- **No-op if there's no active context.** Guarded by
  `if not current_trace: return`.
- Reachable only from inside something that opened a trace/span:
  `@observe` / `with trace(...)` / inside a tool body during
  `agent.run`.

### Pull: `next_*_span(...)`

- **Stages a payload in a contextvar slot.** Doesn't read any active
  context; doesn't need one to exist.
- **Consumed at the next span open.** `SpanInterceptor.on_start` calls
  `pop_pending_for(span_type)` and applies the payload to the
  placeholder it's about to push. One-shot per slot.
- Works regardless of mode (bare / `with trace` / `@observe`).

### Declarative: `ConfidentInstrumentationSettings(...)`

- **Trace-level defaults baked into the agent.** Resolved at every
  span's `on_end` as a fallback under the active trace context.
- Per-agent. Different agents can have different settings.

### When to use which

| Goal | Bare mode | `with trace(...)` / `@observe` |
|---|---|---|
| Trace `name`/`user_id`/`tags` known at agent-construction | `ConfidentInstrumentationSettings(name=..., ...)` | same |
| Trace fields known per call, before agent.run | use `with trace(...)` instead — there is no `next_trace(...)` | `with trace(name=...)` |
| Trace fields known *during* agent.run (e.g. derived from tool result) | `update_current_trace(...)` from inside a tool body | `update_current_trace(...)` from anywhere inside |
| Agent-span fields per call, before agent.run | `with next_agent_span(...)` | `with next_agent_span(...)` |
| LLM-span fields per call, before agent.run | `with next_llm_span(...)` | `with next_llm_span(...)` |
| Tool-span fields | `update_current_span(...)` / `update_tool_span(...)` from inside the tool body | same |

---

## What lives where (configuration matrix)

| Layer | Where you can write to it from |
|---|---|
| **Trace** | (a) `ConfidentInstrumentationSettings(...)` defaults <br> (b) `with trace(...)` kwargs <br> (c) `update_current_trace(...)` from inside an active trace context (any tool body, `@observe` body, or `with trace(...)` body) |
| **Agent span** | (a) `with next_agent_span(...)` BEFORE `agent.run` <br> (b) NOT reachable from inside the agent — pydantic-ai owns the agent span body |
| **LLM span** | (a) `with next_llm_span(...)` BEFORE `agent.run` <br> (b) NOT reachable from inside — pydantic-ai opens it around the model call |
| **Tool span** | (a) `update_current_span(...)` / `update_tool_span(...)` from INSIDE the `@agent.tool_plain` function body <br> (b) `with next_tool_span(...)` BEFORE `agent.run` (one-shot, hits the FIRST tool span only) |
| **Retriever span** | (a) `update_current_span(...)` / `update_retriever_span(...)` from INSIDE the retriever function body <br> (b) `with next_retriever_span(...)` BEFORE `agent.run` |

The asymmetry between agent/LLM and tool/retriever is structural:
**user code never runs inside agent or LLM spans** in pydantic-ai
(they wrap the model call and the agent loop, which are framework
internals). User code DOES run inside tool/retriever bodies.

---

## Configuring traces

### Settings defaults (most common)

```python
settings = ConfidentInstrumentationSettings(
    name="my-app",
    user_id="user-123",
    thread_id="thread-456",
    tags=["prod"],
    metadata={"env": "production"},
    metric_collection="prod-metrics",
    test_case_id="tc-001",
    turn_id="turn-9",
)
agent = Agent(..., instrument=settings)
```

Every trace produced by this agent ships with these values, unless
overridden.

### Per-call override via `with trace(...)`

```python
with trace(name="per-call-name", metadata={"variant": "B"}):
    agent.run_sync("...")
```

Switches routing to REST. Wins over settings for any field it touches.

### Per-call override from inside a tool body

```python
@agent.tool_plain
def lookup(city: str) -> str:
    update_current_trace(metadata={"resolved_city": city})
    return ...
```

Works in all three modes (bare / `with trace` / `@observe`). In bare
mode, mutates the implicit placeholder.

### What you CAN'T do

- ❌ `update_current_trace(...)` BEFORE `agent.run_sync` in bare mode —
  no implicit placeholder exists yet, no-op.
- ❌ `update_current_trace(...)` between two `agent.run_sync` calls in
  bare mode — placeholder was popped after the first call returned.
- ❌ `next_trace(...)` — it doesn't exist. Use `with trace(...)` instead.
  The trace is the OTel root span; it's not "deferred-staged" the same
  way span attrs are.

---

## Configuring spans

### Tool spans (the easy one)

```python
@agent.tool_plain
def get_weather(city: str) -> str:
    update_current_span(
        metadata={"weather_source": "mock", "city": city},
        metric_collection="weather-tool-evals",
    )
    return f"{city}: sunny"
```

User code IS the tool span body. `update_current_span` from inside hits
the tool span placeholder.

### Agent spans (require staging)

```python
with next_agent_span(metric_collection="orchestrator_v1", metrics=[...]):
    result = agent.run_sync("...")
```

`next_agent_span` is the **only** mechanism for agent-span fields,
because user code never runs inside an agent span. The wrapper stages a
payload; `SpanInterceptor.on_start` consumes it for the next OTel
agent-type span and applies it to the placeholder.

### LLM spans (require staging)

Same shape:

```python
with next_llm_span(model="gpt-4o-mini", prompt=my_prompt):
    agent.run_sync("...")
```

LLM spans (`gen_ai.operation.name` ∈ {`chat`, `generate_content`,
`text_completion`}) are framework internals. The only seam is staging.

### Stacking

```python
with next_agent_span(metric_collection="A"), next_llm_span(model="gpt-4o-mini"):
    agent.run_sync("...")
```

Each typed slot is independent. Agent span gets agent values, LLM span
gets LLM values, no cross-contamination.

### Combining staging + body mutation

```python
with next_agent_span(metric_collection="agent_metrics"):
    agent.run_sync("Use the get_weather tool for Tokyo")
    # During the run:
    #   - agent span starts → pops next_agent_span payload → metric_collection=agent_metrics
    #   - LLM span starts/ends
    #   - tool span starts → tool body runs → update_current_span(metadata={...}) hits tool placeholder
    #   - tool span ends, LLM span ends, agent span ends
```

`next_*_span` and `update_current_*` write to different placeholders,
so they never conflict.

---

## Resolution & precedence

Both trace and span attribute resolution follow the same shape:

### Trace-level (every span emits these as `confident.trace.*`)

At every span's `on_end`, `SpanInterceptor._serialize_trace_context_to_otel_attrs`:

```
final_value = current_trace_context.get().<field>   if present
              else self.settings.<field>             if present
              else <not written>
```

For `metadata`, base + context are merged (settings as base layer,
context dict overlaid on top, key by key).

### Span-level (each span emits its own `confident.span.*`)

For agent spans:

```
placeholder = AgentSpan(name=<from gen_ai.agent.name>, …, status=IN_PROGRESS, …)
              # then at construction time:
apply_pending_to_span(placeholder, pop_pending_for("agent"))
              # (consumes next_agent_span + base next_span slots)

# inside the agent's lifetime:
update_current_span(…)   # mutates `placeholder` further
update_agent_span(…)     # type-specific mutator
update_current_trace(…)  # mutates the trace, not this span

# at on_end:
serialize_placeholder(placeholder, span)   # writes confident.span.*
```

So the precedence is **last-write-wins** on a single placeholder:

```
next_agent_span sets the floor at on_start
   → update_current_span / update_agent_span overrides during the span's life
      → on_end serialization captures the final state
```

### `metadata` specifics

- **Trace metadata**: settings + context dict-merge (per-key context wins).
- **Span metadata**: last assignment wins. `next_agent_span(metadata={"a":1})`
  followed by `update_current_span(metadata={"b":2})` from a tool body
  results in `{"b": 2}`, NOT `{"a": 1, "b": 2}`. This matches
  `update_current_span`'s historical "assign, don't merge" semantics.

---

## Routing: REST vs OTLP

`ContextAwareSpanProcessor._should_route_to_rest()` decides per span:

| Active deepeval context | Trace.is_otel_implicit | Routing |
|---|---|---|
| None | — | **OTLP** |
| Real (`with trace`, `@observe`) | False | **REST** |
| Implicit (bare `agent.run`) | True | **OTLP** |

Why "implicit" goes OTLP: the bare caller didn't ask for REST
behavior. The implicit placeholder is purely a write target for
`update_current_trace(...)`; promoting it to REST would silently
change user-visible behavior.

`SpanInterceptor` does NOT decide routing. It just produces
`confident.*` attributes; both transports read the same attributes.

---

## Concurrency: asyncio, threads, sub-contexts

Everything in this integration is built on `contextvars.ContextVar`,
which means:

### Asyncio tasks
Each `asyncio.create_task(...)` snapshots the parent's context. Mutations
via `ContextVar.set(...)` from inside a task do NOT propagate back to
the parent. This applies to `update_current_*` (which doesn't
re-`set`, it mutates the placeholder object — fine) and to
`next_*_span` slot draining.

### Threads
`concurrent.futures.ThreadPoolExecutor` workers do NOT inherit
contextvars from the submitting thread by default; you have to wrap
with `contextvars.copy_context()`. Pydantic AI uses `anyio.to_thread.run_sync`
for tool functions, which DOES propagate the context. So tool bodies
running in worker threads still see the implicit placeholder pushed in
the main thread.

### The "sub-context drain" subtlety

`Agent.run_sync(...)` calls `asyncio.run(...)` internally, which
creates a new asyncio context that inherits a *snapshot* of the
parent's contextvars.

A naive design that consumed `next_*_span` slots via
`ContextVar.set(slot, None)` would set the slot to `None` only inside
the snapshot — invisible to the outer `with` block. A second
`agent.run_sync` would then re-consume the same value.

Solution (already implemented): `next_*_span` stores a `_PendingSlot`
*wrapper* in the contextvar. The consumer drains via
`slot.payload = None` (mutation on the shared object), not
`ContextVar.set(None)`. Both contexts see the mutation because they
inherit the same wrapper reference.

Regression test: `test_drain_visible_across_asyncio_sub_context` in
`tests/test_integrations/test_pydanticai/test_span_interceptor.py`.

### Concurrent agent.run

Multiple `agent.run` calls in different asyncio tasks each get their
own implicit `Trace` placeholder (one per OTel root span, isolated by
contextvar inheritance per task). No cross-task leakage. See
`pydantic_after_concurrent.py` for a runnable validation.

Same applies to thread-based concurrency (`pydantic_after_threads.py`).

---

## Edge cases and pitfalls

### `update_current_*` BEFORE `agent.run_sync` (bare mode)

```python
# WRONG: no current trace exists yet → silent no-op
update_current_trace(name="X")
agent.run_sync("...")
```

The implicit placeholder is pushed inside `agent.run_sync` (at root
span on_start). Before the call, `current_trace_context.get()` returns
`None`. `update_current_trace` returns early.

**Fix**: use `ConfidentInstrumentationSettings(name="X")` for static
defaults, `with trace(name="X")` for per-call (REST mode), or
`update_current_trace` from inside a tool body.

### `update_current_*` BETWEEN two `agent.run_sync` calls (bare mode)

```python
agent.run_sync("first")
update_current_trace(name="X")  # silent no-op — first call's context already popped
agent.run_sync("second")
```

The implicit placeholder is popped at the root span's on_end, before
`agent.run_sync` returns. Same fix as above.

### `next_*_span` with multiple `agent.run_sync` in one block

```python
with next_agent_span(metric_collection="A"):
    agent.run_sync("first")    # gets A
    agent.run_sync("second")   # gets nothing (one-shot)
```

This is intentional and matches the literal name "next". If you want
sticky semantics, wrap each call individually:

```python
for q in queries:
    with next_agent_span(metric_collection="A"):
        agent.run_sync(q)
```

### Nested `next_agent_span(...)`

```python
with next_agent_span(metric_collection="outer"):
    with next_agent_span(metric_collection="inner"):
        agent.run_sync("...")  # gets inner
    agent.run_sync("...")      # gets outer (token reset restored it)
```

Inner pushes a new `_PendingSlot` (different object). On exit, the
contextvar is reset to the outer's slot. Outer's payload was untouched.

### Empty `with next_agent_span():`

If no kwargs are passed, `_drop_none(...)` produces an empty dict.
`pop_pending_for` short-circuits on empty dicts (`if base_slot.payload`
is False), so no fields land on the placeholder. Effectively a no-op.

### `next_*_span` in a scope where no consumer fires

```python
with next_agent_span(metric_collection="leaked"):
    pass  # no agent.run_sync inside → payload is just discarded on exit
```

No leak. `_PendingSlot` is bound to the contextvar; on `with` exit the
token is reset and the wrapper is discarded with the prior value
restored.

### Type mismatch on `apply_pending_to_span`

`next_llm_span(model="gpt-4")` runs but the next OTel span is somehow
classified as a tool, not LLM. `pop_pending_for("tool")` doesn't drain
the LLM slot, so the LLM kwargs sit there. They'll get consumed by the
NEXT LLM span — possibly in a later iteration. If the `with` exits
first, they're discarded.

### `available_tools` / `agent_handoffs` not visible in OTel attrs

The placeholder serializer (`_serialize_placeholder_to_otel_attrs`)
writes a fixed list of fields back to `confident.span.*`. Some
agent-specific fields (`available_tools`, `agent_handoffs`) are
present on the `AgentSpan` placeholder but not currently serialized.
Mutating them via `next_agent_span(available_tools=[...])` updates the
placeholder but won't surface in the trace JSON without an exporter
update. This is a known gap; flagged for future work.

### Span name collision

`next_agent_span(name="custom")` writes `placeholder.name = "custom"`,
but `_serialize_placeholder_to_otel_attrs` skips writing
`confident.span.name` if it's already set — and `_add_agent_span` sets
it at `on_start` from `gen_ai.agent.name`. Net effect: `name` from
`next_agent_span` does NOT override the pydantic-ai-derived agent
name. To rename a span, set the agent's `name` at `Agent(name="...")`.

### `metric_collection` precedence

For traces:
```
update_current_trace(metric_collection=…)   # wins if set during run
   > ConfidentInstrumentationSettings(metric_collection=…)
   > <not stamped>
```

For spans:
```
update_current_span(metric_collection=…) from inside the span body
   > next_*_span(metric_collection=…) BEFORE the span starts
   > <not stamped>
```

(`metric_collection` is intentionally NOT a kwarg of typed
`update_*_span` helpers — it's a base-span field, reachable through
`update_current_span`.)

---

## Application patterns

### Pattern 1: Single agent, static config

```python
settings = ConfidentInstrumentationSettings(
    name="my-bot",
    metric_collection="prod-metrics",
    metadata={"env": "prod"},
)
agent = Agent("openai:gpt-4o-mini", instrument=settings, name="my_bot")

agent.run_sync("hello")
```

All traces from this agent ship identical metadata. Routing: OTLP.

### Pattern 2: Per-call attribution

```python
with trace(user_id="u1", thread_id="thread-1"):
    agent.run_sync("hello")
```

Each call attributes to a different user/thread. Routing: REST.

### Pattern 3: Orchestrator → sub-agents

```python
orchestrator = Agent("openai:gpt-4o-mini", instrument=settings_a, name="orchestrator")
sub_agent = Agent("openai:gpt-4o-mini", instrument=settings_b, name="sub_agent")

@orchestrator.tool_plain
def delegate(query: str) -> str:
    with next_agent_span(metric_collection="sub_metrics_v1"):
        return sub_agent.run_sync(query).output

with next_agent_span(metric_collection="orchestrator_metrics_v1"):
    orchestrator.run_sync("...")
```

Each agent invocation gets its own metric_collection, configured
declaratively at the callsite. Each `with next_agent_span(...)`
consumes exactly one agent-root span (the one inside it).

### Pattern 4: Tool-driven trace metadata enrichment

```python
@agent.tool_plain
def lookup_user(user_id: str) -> dict:
    user = db.fetch(user_id)
    update_current_trace(
        user_id=user.id,
        metadata={"plan": user.plan, "country": user.country},
    )
    return user.profile

with trace():    # use REST routing so metadata is searchable in the dashboard
    agent.run_sync("get my profile")
```

The trace doesn't know who the user is until the lookup tool runs.
`update_current_trace` from inside the tool body enriches it
mid-flight.

### Pattern 5: Online evals via metric_collection

```python
@agent.tool_plain
def special_tool(query: str) -> str:
    update_current_span(metric_collection="tool-evals-v1")
    return ...

with next_agent_span(metric_collection="agent-evals-v1"):
    agent.run_sync(
        "use special_tool",
    )
```

Each span layer points to a different metric collection in Confident
AI, so different evals run on each.

### Pattern 6: Concurrent runs (asyncio)

```python
import asyncio

async def query(prompt: str, user_id: str):
    with trace(user_id=user_id):
        return (await agent.run(prompt)).output

results = await asyncio.gather(
    query("p1", "u1"),
    query("p2", "u2"),
    query("p3", "u3"),
)
```

Each task gets its own contextvar copy; per-task `with trace(...)` is
isolated. No cross-task attribute leakage. See
`pydantic_after_concurrent.py`.

### Pattern 7: Concurrent runs (threads)

```python
from concurrent.futures import ThreadPoolExecutor

def query(prompt: str, user_id: str):
    with trace(user_id=user_id):
        return agent.run_sync(prompt).output

with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(lambda p: query(*p), [("p1", "u1"), ("p2", "u2")]))
```

`ThreadPoolExecutor` does not propagate contextvars unless wrapped in
`contextvars.copy_context()`, but each worker's `with trace(...)`
establishes its own context, so isolation holds. See
`pydantic_after_threads.py`.

---

## Field reference

### `ConfidentInstrumentationSettings(...)`

All optional. All trace-level (no per-span configuration).

| Kwarg | Type | Description |
|---|---|---|
| `api_key` | `str` | Confident AI API key. Falls back to `CONFIDENT_API_KEY` env / `deepeval login`. |
| `name` | `str` | Default trace name. |
| `thread_id` | `str` | Default thread id. |
| `user_id` | `str` | Default user id. |
| `metadata` | `dict` | Default trace metadata (merged base under runtime overlay). |
| `tags` | `list[str]` | Default trace tags. |
| `metric_collection` | `str` | Default trace metric_collection. |
| `test_case_id` | `str` | Default test_case_id. |
| `turn_id` | `str` | Default turn_id. |

Removed in the refactor (will raise `TypeError`):
`is_test_mode`, `confident_prompt`, `trace_metric_collection`,
`agent_metric_collection`, `llm_metric_collection`,
`tool_metric_collection_map`, `agent_metrics`. See
`test_span_related_kwargs_are_removed_from_settings` for the
authoritative list.

### `next_span(...)`, `next_agent_span(...)`, `next_llm_span(...)`, `next_tool_span(...)`, `next_retriever_span(...)`

Each typed helper accepts:
- The same **base** kwargs `update_current_span(...)` accepts (`input`,
  `output`, `metadata`, `name`, `metric_collection`, `metrics`,
  `test_case`, etc.).
- Plus its **type-specific** kwargs (e.g. `next_llm_span(model=...)`,
  `next_agent_span(available_tools=...)`).

Stacking different typed helpers in one `with` is supported and safe.

### `update_current_*_span` family

- `update_current_span(...)` — base fields only.
- `update_llm_span(...)` — LLM-specific only (`model`, token counts,
  `prompt`).
- `update_retriever_span(...)` — retriever-specific only (`embedder`,
  `top_k`, `chunk_size`).
- `update_agent_span(...)` — agent-specific only (`available_tools`,
  `agent_handoffs`).
- `update_tool_span(...)` — tool-specific only (`description`).

For combined base + type-specific in one call, call both:

```python
update_current_span(metadata={...}, metric_collection="…")
update_llm_span(model="gpt-4o-mini")
```

(or use `next_llm_span(...)` which accepts both at once, if you're
configuring before the span opens).

---

## Validation scripts

Runnable end-to-end checks at the repo root:

| Script | What it validates |
|---|---|
| `pydantic_after.py` | `update_current_trace` / `update_current_span` from inside tool bodies (the canonical `@observe` flow). |
| `pydantic_after_bare.py` | Same dynamics work with bare `agent.run` (implicit Trace placeholder). |
| `pydantic_after_concurrent.py` | ContextVar isolation across `asyncio.gather`. |
| `pydantic_after_threads.py` | ContextVar propagation across `ThreadPoolExecutor` + `anyio.to_thread.run_sync`. |
| `pydantic_after_next_span.py` | All four `next_agent_span` / `next_llm_span` scenarios: simple, stacked, one-shot, nested. |

Each script needs `CONFIDENT_API_KEY` and `OPENAI_API_KEY` in env and
prints the expected dashboard outcome at the end so you can spot-check
on Confident AI.

---

## Test suite

Unit tests (no LLM calls): `tests/test_integrations/test_pydanticai/test_span_interceptor.py`.

Integration tests (real LLMs, schema-asserted):
`tests/test_integrations/test_pydanticai/test_sync.py` and `test_async.py`.

Schemas in `tests/test_integrations/test_pydanticai/schemas/` are
generated via `GENERATE_SCHEMAS=true pytest ...` and asserted in normal
mode.
