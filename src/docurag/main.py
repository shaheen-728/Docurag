import logging

from fastapi import FastAPI

from docurag.llm.accounting import ChatAccountingMiddleware
from docurag.routers import chat, documents, health, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Enable application INFO logs explicitly.
logging.getLogger("docurag").setLevel(logging.INFO)

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
