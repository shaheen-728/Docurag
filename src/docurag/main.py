from fastapi import FastAPI

from docurag.core.logging import configure_logging
from docurag.llm.accounting import ChatAccountingMiddleware
from docurag.routers import chat, documents, health, search

configure_logging()

app = FastAPI(
    title="DocuRAG",
    description="Document question answering using RAG",
    version="0.1.0",
)

app.add_middleware(ChatAccountingMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
