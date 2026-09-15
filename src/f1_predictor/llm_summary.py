"""Optional LLM reasoning layer: turn SHAP values into a natural-language race summary.

Requires ANTHROPIC_API_KEY (see .env.example). If it's unset, `summarize_prediction`
returns None so callers (the API, the CLI) can skip this feature gracefully instead
of crashing — it's a differentiator on top of the core ML pipeline, not a dependency
of it.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "claude-sonnet-5"

PROMPT_TEMPLATE = """You are an F1 analyst. A machine learning model predicted the following \
race result for {race_name}:

Predicted P1: {p1_driver} ({p1_time:.2f}s)
Predicted P2: {p2_driver} ({p2_time:.2f}s)
Predicted P3: {p3_driver} ({p3_time:.2f}s)

The model's top feature contributions (SHAP values, positive = predicts a slower race \
time) for the predicted winner, {p1_driver}, were:
{shap_lines}

In 3-4 sentences, explain in plain language why the model favors {p1_driver}, referencing \
the two or three biggest contributing factors. Be concrete and avoid hedging."""


def _format_shap_lines(shap_values: dict[str, float], top_n: int = 4) -> str:
    ranked = sorted(shap_values.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    return "\n".join(f"  - {name}: {value:+.3f}" for name, value in ranked)


def summarize_prediction(
    race_name: str,
    podium: list[tuple[str, float]],
    winner_shap_values: dict[str, float],
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Generate a natural-language explanation of the predicted podium using Claude.

    Returns None if ANTHROPIC_API_KEY isn't configured.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = PROMPT_TEMPLATE.format(
        race_name=race_name,
        p1_driver=podium[0][0],
        p1_time=podium[0][1],
        p2_driver=podium[1][0],
        p2_time=podium[1][1],
        p3_driver=podium[2][0],
        p3_time=podium[2][1],
        shap_lines=_format_shap_lines(winner_shap_values),
    )

    response = client.messages.create(model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}])
    return response.content[0].text
