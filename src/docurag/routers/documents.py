import asyncio
import hashlib
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from kombu.exceptions import OperationalError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from docurag.core.config import get_settings
from docurag.db.models import Document
from docurag.db.session import get_session
from docurag.schemas.documents import (
    DocumentStatus,
    DocumentUploadResponse,
)
from docurag.worker.tasks import ingest_document

router = APIRouter(tags=["Documents"])

SessionDependency = Annotated[AsyncSession, Depends(get_session)]

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
}


def save_source(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Never overwrite an existing source file.
    with path.open("xb") as destination:
        destination.write(content)


def remove_source(path: Path) -> None:
    path.unlink(missing_ok=True)


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    response: Response,
    session: SessionDependency,
) -> DocumentUploadResponse:
    settings = get_settings()

    # Strip directories from both Windows-style and Unix-style filenames.
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(filename).suffix.lower()

    try:
        if not filename or len(filename) > 255:
            raise HTTPException(
                status_code=400,
                detail="Provide a filename of at most 255 characters.",
            )

        if extension not in CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Supported formats: PDF, Markdown, HTML and TXT.",
            )

        # Read at most the configured limit plus one byte.
        content = await file.read(settings.max_upload_bytes + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail="The uploaded file exceeds the size limit.",
            )
    finally:
        await file.close()

    content_hash = hashlib.sha256(content).hexdigest()

    document_id = uuid4()
    storage_path = (
        Path(settings.upload_directory).resolve() / f"{document_id}{extension}"
    )

    source_created = False

    try:
        async with session.begin():
            existing = await session.scalar(
                select(Document).where(Document.content_hash == content_hash)
            )

            if existing is not None:
                response.status_code = status.HTTP_200_OK

                return DocumentUploadResponse(
                    **DocumentStatus.model_validate(existing).model_dump(),
                    duplicate=True,
                )

            # Keep blocking filesystem writes off the async event loop.
            # Mark before writing so partial writes are cleaned up on error.
            source_created = True
            await asyncio.to_thread(save_source, storage_path, content)

            statement = (
                insert(Document)
                .values(
                    id=document_id,
                    filename=filename,
                    content_type=CONTENT_TYPES[extension],
                    content_hash=content_hash,
                    storage_path=str(storage_path),
                    status="queued",
                    chunk_count=0,
                )
                .on_conflict_do_nothing(constraint="uq_documents_content_hash")
                .returning(Document)
            )

            document = await session.scalar(statement)
            duplicate = document is None

            if duplicate:
                # Another concurrent request inserted the same hash first.
                await asyncio.to_thread(remove_source, storage_path)
                source_created = False

                document = await session.scalar(
                    select(Document).where(Document.content_hash == content_hash)
                )

                if document is None:
                    raise RuntimeError("Duplicate document lookup failed.")

                response.status_code = status.HTTP_200_OK

            result = DocumentUploadResponse(
                **DocumentStatus.model_validate(document).model_dump(),
                duplicate=duplicate,
            )

    except Exception:
        if source_created:
            await asyncio.to_thread(remove_source, storage_path)
        raise

    # Publish only after the database transaction commits.
    if not duplicate:
        try:
            await asyncio.to_thread(
                ingest_document.delay,
                str(result.id),
            )
        except OperationalError as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "Document saved, but task publication failed. "
                        "Submit it again using the task command."
                    ),
                    "document_id": str(result.id),
                },
            ) from error

    return result


@router.get(
    "/documents/{document_id}",
    response_model=DocumentStatus,
)
async def get_document(
    document_id: UUID,
    session: SessionDependency,
) -> DocumentStatus:
    document = await session.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return DocumentStatus.model_validate(document)
