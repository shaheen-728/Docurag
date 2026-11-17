from fastapi import APIRouter, HTTPException, status

router = APIRouter(tags=["Documents"])


@router.post("/documents", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def upload_document() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document upload is not implemented yet.",
    )
