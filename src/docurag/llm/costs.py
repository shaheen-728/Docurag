from google.genai import types

# USD per million tokens: input, output, cached input.
# Standard text-generation rates checked on 2026-11-20.
GEMINI_PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
}


def estimate_gemini_cost(
    model: str,
    usage: types.GenerateContentResponseUsageMetadata | None,
    *,
    completed: bool,
) -> float | None:
    """Estimate standard text-generation cost, excluding tools and storage."""
    prices = GEMINI_PRICES.get(model.removeprefix("models/"))

    if prices is None or usage is None or not completed:
        return None

    input_tokens = usage.prompt_token_count
    answer_tokens = usage.candidates_token_count
    total_tokens = usage.total_token_count

    if input_tokens is None or answer_tokens is None or total_tokens is None:
        return None

    cached_tokens = usage.cached_content_token_count or 0

    # For our text-only calls without tools, the remainder includes
    # answer and thinking tokens. Do not add thinking tokens again.
    output_tokens = total_tokens - input_tokens

    if (
        input_tokens < 0
        or answer_tokens < 0
        or not 0 <= cached_tokens <= input_tokens
        or output_tokens < answer_tokens
    ):
        return None

    input_rate, output_rate, cached_rate = prices

    cost = (
        (input_tokens - cached_tokens) * input_rate
        + cached_tokens * cached_rate
        + output_tokens * output_rate
    ) / 1_000_000

    return round(cost, 10)
