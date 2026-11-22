"""Add a corpus revision for retrieval cache invalidation."""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE corpus_revision (
            id integer PRIMARY KEY CHECK (id = 1),
            revision bigint NOT NULL DEFAULT 0
        )
        """
    )

    op.execute(
        """
        INSERT INTO corpus_revision (id, revision)
        VALUES (1, 0)
        """
    )

    op.execute(
        """
        CREATE FUNCTION bump_corpus_revision()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            UPDATE corpus_revision
            SET revision = revision + 1
            WHERE id = 1;

            RETURN NULL;
        END;
        $$
        """
    )

    for table in ("documents", "chunks"):
        op.execute(
            f"""
            CREATE TRIGGER {table}_corpus_revision
            AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE
            ON {table}
            FOR EACH STATEMENT
            EXECUTE FUNCTION bump_corpus_revision()
            """
        )


def downgrade() -> None:
    for table in ("chunks", "documents"):
        op.execute(
            f"""
            DROP TRIGGER {table}_corpus_revision
            ON {table}
            """
        )

    op.execute("DROP FUNCTION bump_corpus_revision()")
    op.execute("DROP TABLE corpus_revision")
