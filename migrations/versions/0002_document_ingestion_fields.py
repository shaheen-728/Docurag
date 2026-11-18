"""Add document ingestion tracking."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )

    op.add_column(
        "documents",
        sa.Column("storage_path", sa.Text(), nullable=True),
    )

    op.add_column(
        "documents",
        sa.Column(
            "status",
            sa.String(20),
            server_default=sa.text("'queued'"),
            nullable=False,
        ),
    )

    op.add_column(
        "documents",
        sa.Column("error_message", sa.Text(), nullable=True),
    )

    op.add_column(
        "documents",
        sa.Column(
            "chunk_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )

    op.add_column(
        "documents",
        sa.Column("processing_seconds", sa.Float(), nullable=True),
    )

    # Older rows have no saved source path or content hash.
    # Preserve them, but don't falsely report them as queued tasks.
    op.execute(
        """
        UPDATE documents
        SET status = 'failed',
            error_message =
                'Legacy record: upload the original file to ingest it.'
        WHERE content_hash IS NULL
        """
    )

    op.create_unique_constraint(
        "uq_documents_content_hash",
        "documents",
        ["content_hash"],
    )

    op.create_check_constraint(
        "ck_documents_status",
        "documents",
        "status IN ('queued', 'processing', 'completed', 'failed')",
    )

    op.create_check_constraint(
        "ck_documents_chunk_count",
        "documents",
        "chunk_count >= 0",
    )

    op.create_check_constraint(
        "ck_documents_processing_seconds",
        "documents",
        "processing_seconds >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_documents_processing_seconds",
        "documents",
        type_="check",
    )

    op.drop_constraint(
        "ck_documents_chunk_count",
        "documents",
        type_="check",
    )

    op.drop_constraint(
        "ck_documents_status",
        "documents",
        type_="check",
    )

    op.drop_constraint(
        "uq_documents_content_hash",
        "documents",
        type_="unique",
    )

    op.drop_column("documents", "processing_seconds")
    op.drop_column("documents", "chunk_count")
    op.drop_column("documents", "error_message")
    op.drop_column("documents", "status")
    op.drop_column("documents", "storage_path")
    op.drop_column("documents", "content_hash")
