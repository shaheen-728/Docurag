from redis.backoff import NoBackoff
from redis.retry import Retry
from slowapi import Limiter
from slowapi.util import get_remote_address

from docurag.core.config import get_settings

settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,
    storage_options={
        "socket_connect_timeout": 0.2,
        "socket_timeout": 0.2,
        "retry": Retry(NoBackoff(), 0),
    },
    key_prefix=f"docurag:{settings.postgres_db}:rate-limit",
    strategy="fixed-window",
    headers_enabled=False,
    in_memory_fallback_enabled=True,
)
