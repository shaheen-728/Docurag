from fastapi import FastAPI

from docurag.routers import chat, documents, health, search

app = FastAPI(
    title="DocuRAG",
    description="Document question answering using RAG",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
