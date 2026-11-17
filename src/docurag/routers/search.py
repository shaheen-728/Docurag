from fastapi import APIRouter, HTTPException, status

router = APIRouter(tags=["Search"])


@router.post("/search", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def search_documents() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document search is not implemented yet.",
    )
