"""pydantic_after_next_span.py — ``next_*_span(...)`` validation.

Validates the new ``with next_agent_span(...)`` / ``with next_llm_span(...)``
context managers against a real pydantic-ai ``agent.run_sync(...)`` call.
Companion to ``pydantic_after_bare.py`` — same bare-call posture, but
focusing on the OUTSIDE-IN configuration mechanism that fills the gap
``update_current_span(...)`` can't reach.

Why this exists:

  - Pydantic AI agent spans (and LLM spans) have NO user-code seam — the
    user never gets to write code "as the agent" / "as the LLM". So
    ``update_current_span(...)`` from inside a tool body lands on the
    TOOL span only, never on the enclosing agent / LLM span.
  - ``with next_*_span(...)`` is the only mechanism that lets a user
    stamp agent / LLM span fields per call, declaratively, from the
    outside.

What this script proves:

  1. ``with next_agent_span(metric_collection=..., metrics=..., metadata=...)``
     wrapping ``agent.run_sync(...)`` lands on the AGENT span (not the
     trace, not the tool span).
  2. Stacking ``with next_agent_span(...), next_llm_span(...)`` is safe:
     each typed slot is independent. Agent gets agent values, LLM gets
     LLM values, no cross-talk.
  3. One-shot consumption: a SECOND ``agent.run_sync(...)`` inside the
     same ``with`` block produces an agent span with NO defaults from
     the wrapper (the slot was drained by the first call).
  4. Nested ``with next_agent_span(...)`` blocks: the inner overrides
     for its scope, the outer is restored on exit. Useful for
     orchestrator → sub-agent patterns where each agent invocation
     wants its own metric_collection.

Requirements:
  - ``CONFIDENT_API_KEY`` in env (or ``deepeval login``)
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai``
"""

import os
import uuid
from pathlib import Path

from pydantic_ai import Agent

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import (
    next_agent_span,
    next_llm_span,
)
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


# Unique per-script-run id so every trace produced by this run can be
# grouped in the dashboard via metadata.run_id.
RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


settings = ConfidentInstrumentationSettings(
    name="pydantic-ai-next-span-validation",
    tags=["after-rewrite", "next-span"],
    metadata={"run_id": RUN_ID},
)

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise. One short sentence.",
    instrument=settings,
    name="weather_agent",
)


@agent.tool_plain
def get_weather(city: str) -> str:
    return f"{city}: sunny, 24C"


def scenario_1_simple_next_agent_span() -> str:
    """``with next_agent_span(...)`` wrapping a bare ``agent.run_sync``.

    Expected on dashboard:
      - agent span has metric_collection=scenario-1-agent
      - agent span has metadata.scenario=1
      - agent span has metrics=[AnswerRelevancy] (best-effort visualization)
    """
    metric = AnswerRelevancyMetric(threshold=0.5)
    with next_agent_span(
        metric_collection="scenario-1-agent",
        metrics=[metric],
        metadata={"scenario": 1, "purpose": "simple_next_agent_span"},
    ):
        return agent.run_sync("What's the weather in Paris?").output


def scenario_2_stacked_typed_slots() -> str:
    """Stack ``next_agent_span(...)`` AND ``next_llm_span(...)`` in one
    ``with`` statement. Each writes to its own slot; the integration
    pops them independently when classifying agent vs LLM spans.

    Expected on dashboard:
      - agent span has metric_collection=scenario-2-agent
      - LLM span has metric_collection=scenario-2-llm
      - LLM span has metadata.role=primary_model_call
    """
    with next_agent_span(
        metric_collection="scenario-2-agent",
        metadata={"scenario": 2, "layer": "agent"},
    ), next_llm_span(
        metric_collection="scenario-2-llm",
        metadata={"scenario": 2, "layer": "llm", "role": "primary_model_call"},
    ):
        return agent.run_sync("What's the weather in Tokyo?").output


def scenario_3_one_shot_consumption() -> tuple[str, str]:
    """Two ``agent.run_sync(...)`` calls inside ONE
    ``with next_agent_span(...)`` block.

    Expected on dashboard:
      - First call's agent span has metric_collection=scenario-3-only-first
      - Second call's agent span has NO metric_collection (slot drained
        by the first call's on_start consumption — the "next" in
        ``next_agent_span`` is literal: only the next one).
    """
    with next_agent_span(
        metric_collection="scenario-3-only-first",
        metadata={"scenario": 3, "consumption": "one_shot"},
    ):
        first = agent.run_sync("What's the weather in Berlin?").output
        second = agent.run_sync("What's the weather in Madrid?").output
    return first, second


def scenario_4_nested_overrides() -> tuple[str, str]:
    """Nested ``with next_agent_span(...)`` blocks. Inner overrides for
    its scope; outer is restored on exit so a subsequent agent.run uses
    the outer's values.

    Expected on dashboard:
      - Inner call (Sydney): metric_collection=scenario-4-INNER
      - Outer call (Lima): metric_collection=scenario-4-OUTER
        (the inner ``with`` exited, the outer's slot was restored, and
         the outer-scope agent.run consumes it.)
    """
    with next_agent_span(
        metric_collection="scenario-4-OUTER",
        metadata={"scenario": 4, "layer": "outer"},
    ):
        with next_agent_span(
            metric_collection="scenario-4-INNER",
            metadata={"scenario": 4, "layer": "inner"},
        ):
            inner_output = agent.run_sync(
                "What's the weather in Sydney?"
            ).output

        # Inner ``with`` has exited; outer slot is restored. The next
        # agent.run consumes the outer-scope value.
        outer_output = agent.run_sync("What's the weather in Lima?").output

    return inner_output, outer_output


def main() -> None:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise SystemExit("CONFIDENT_API_KEY is not set.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    print("Scenario 1: simple next_agent_span")
    print("  ->", scenario_1_simple_next_agent_span())

    print("\nScenario 2: stacked next_agent_span + next_llm_span")
    print("  ->", scenario_2_stacked_typed_slots())

    print("\nScenario 3: one-shot consumption (2 runs, 1 wrapper)")
    s3_first, s3_second = scenario_3_one_shot_consumption()
    print("  first  ->", s3_first)
    print("  second ->", s3_second)

    print("\nScenario 4: nested wrappers (inner overrides, outer restored)")
    s4_inner, s4_outer = scenario_4_nested_overrides()
    print("  inner ->", s4_inner)
    print("  outer ->", s4_outer)

    print(
        f"\nAll traces from this run share metadata.run_id = '{RUN_ID}' "
        "(stamped via ConfidentInstrumentationSettings.metadata).\n"
        "Open the Confident AI dashboard and filter by that run_id. You "
        "should see SIX traces (1 + 1 + 2 + 2). Per scenario:\n"
        "  - Scenario 1: agent span has metric_collection=scenario-1-agent\n"
        "  - Scenario 2: agent span has metric_collection=scenario-2-agent\n"
        "                LLM span has metric_collection=scenario-2-llm\n"
        "  - Scenario 3: first agent span has scenario-3-only-first\n"
        "                second agent span has NO metric_collection\n"
        "  - Scenario 4: inner agent span has scenario-4-INNER\n"
        "                outer agent span has scenario-4-OUTER"
    )


if __name__ == "__main__":
    main()
