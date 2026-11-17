from docurag.core.config import Settings
from docurag.llm.base import LLMProvider
from docurag.llm.gemini import GeminiProvider
from docurag.llm.openai_provider import OpenAIProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings)

    return OpenAIProvider(settings)
