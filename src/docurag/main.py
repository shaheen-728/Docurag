from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from docurag.core.logging import configure_logging
from docurag.core.rate_limit import limiter
from docurag.llm.accounting import ChatAccountingMiddleware
from docurag.routers import chat, documents, health, search

configure_logging()

app = FastAPI(
    title="DocuRAG",
    description="Document question answering using RAG",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

app.add_middleware(ChatAccountingMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
