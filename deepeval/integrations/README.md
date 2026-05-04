# `deepeval.integrations`

Contributor reference for the framework integrations. Each integration plugs deepeval's tracing / evaluation into a third-party framework using one of four mechanisms.

> Note: `deepeval.openai`, `deepeval.anthropic`, and `deepeval.openai_agents` live at the top level of the `deepeval` package, not under this folder. They're listed here so the matrix is complete.

## Integration matrix

| Integration | Mode | Entry point | Transport | Source |
|---|---|---|---|---|
| OpenAI | Native client wrapper | `from deepeval.openai import OpenAI` | REST | `deepeval/openai/` |
| Anthropic | Native client wrapper | `from deepeval.anthropic import Anthropic` | REST | `deepeval/anthropic/` |
| LangChain | Callback handler | `CallbackHandler()` | REST | `deepeval/integrations/langchain/` |
| LangGraph | Callback handler (LangChain's) | `CallbackHandler()` | REST | `deepeval/integrations/langchain/` |
| LlamaIndex | Event handler | `instrument_llama_index()` | REST | `deepeval/integrations/llama_index/` |
| CrewAI | Event listener + wrapper classes | `instrument_crewai()` | REST | `deepeval/integrations/crewai/` |
| Hugging Face | Trainer callback | `DeepEvalHuggingFaceCallback(...)` | REST | `deepeval/integrations/hugging_face/` |
| OpenAI Agents | Trace processor + agent wrapper | `add_trace_processor(DeepEvalTracingProcessor())` | REST | `deepeval/openai_agents/` |
| AgentCore | OpenTelemetry | `instrument_agentcore()` | OTLP | `deepeval/integrations/agentcore/` |
| Google ADK | OpenTelemetry (via OpenInference) | `instrument_google_adk()` | OTLP | `deepeval/integrations/google_adk/` |
| Pydantic AI | OpenTelemetry | `ConfidentInstrumentationSettings(...)` | OTLP | `deepeval/integrations/pydantic_ai/` |

## Mode reference

- **Native client wrapper** — drop-in replacement for the vendor SDK's client class (e.g. `deepeval.openai.OpenAI` instead of `openai.OpenAI`). Spans are built directly via `trace_manager`. Lowest friction, but only covers calls that go through that client.
- **Callback handler / event listener** — registers with the framework's own callback or event API (LangChain `BaseCallbackHandler`, LlamaIndex `BaseEventHandler`, CrewAI `BaseEventListener`, etc.). Spans are built directly via `trace_manager`. Covers all calls the framework dispatches through that surface — no need to swap clients.
- **Trace processor** — for frameworks that already have their own tracing pipeline (OpenAI Agents SDK), we plug into it as a processor and translate events into deepeval spans.
- **OpenTelemetry** — registers an OTel `SpanProcessor` against the global `TracerProvider`. The framework (or a community-maintained instrumentor like `openinference-instrumentation-google-adk`) emits OTel spans; deepeval translates them into Confident span attributes and ships them via OTLP.

## Transport reference

- **REST** — `trace_manager` posts the full trace to `api.confident-ai.com/v1/traces` once per trace.
- **OTLP** — `BatchSpanProcessor` flushes OTel spans to `otel.confident-ai.com/v1/traces` on a timer / queue threshold.

## OpenInference (shared OTel backend)

All three OTel-mode integrations sit on top of `deepeval/integrations/openinference/`, which sets up the `TracerProvider`, registers the `OpenInferenceSpanInterceptor` (translates OpenInference / gen_ai semconv attributes into `confident.span.*`), and wires the OTLP exporter. It is also exposed at the top level as `deepeval.instrument(...)` so users can pair it with any OpenInference instrumentor directly:

```python
import deepeval
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

deepeval.instrument(name="my-app", environment="development")
GoogleADKInstrumentor().instrument()
```

`instrument_google_adk(...)` is just a convenience wrapper that calls `GoogleADKInstrumentor().instrument()` then `deepeval.instrument(...)` for you.

## Mixing OTel-mode with `@observe`

When an OTel-mode integration runs inside an active `@observe` / `with trace(...)` context, the OTel span interceptor synchronizes the trace UUID (`current_trace_context.uuid = OTel trace_id`) so both transports land on the same trace server-side. This means a mixed trace currently produces one REST POST + one or more OTLP POSTs that the backend reconciles by UUID. See internal notes for the proposed single-transport refactor that would route OTel spans through `ConfidentSpanExporter` (REST) when a deepeval trace is active.
