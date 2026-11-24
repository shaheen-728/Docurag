import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from docurag.core.config import get_settings
from docurag.core.logging import configure_logging
from docurag.core.rate_limit import limiter
from docurag.core.tracing import get_tracing_client
from docurag.core.tracing_middleware import RequestTracingMiddleware
from docurag.llm.accounting import ChatAccountingMiddleware
from docurag.routers import chat, health, search

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    client = get_tracing_client()

    try:
        yield
    finally:
        if client is not None:
            await asyncio.to_thread(client.shutdown)


app = FastAPI(
    title="DocuRAG",
    description="Document question answering using RAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

app.add_middleware(RequestTracingMiddleware)
app.add_middleware(ChatAccountingMiddleware)

app.include_router(health.router)
if get_settings().documents_enabled:
    from docurag.routers import documents

    app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
