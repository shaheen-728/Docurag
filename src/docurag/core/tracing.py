import logging
from functools import lru_cache

from langfuse import Langfuse

from docurag.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_tracing_client() -> Langfuse | None:
    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    secret_key = settings.langfuse_secret_key.get_secret_value()

    if not settings.langfuse_public_key or not secret_key:
        logger.warning("tracing_disabled reason=missing_credentials")
        return None

    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=secret_key,
            base_url=settings.langfuse_base_url,
        )
    except Exception:
        logger.exception("tracing_initialization_failed")
        return None
