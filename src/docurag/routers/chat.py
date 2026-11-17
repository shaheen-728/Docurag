from fastapi import APIRouter, HTTPException, status

router = APIRouter(tags=["Chat"])


@router.post("/chat", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def chat_with_documents() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document chat is not implemented yet.",
    )
