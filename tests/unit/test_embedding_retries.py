import pytest
from google.genai.errors import ClientError, ServerError

from docurag.ingestion.embeddings import is_retryable


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ClientError(
                429,
                {
                    "error": {
                        "message": (
                            "Embed content requests per day limit exceeded. "
                            "EmbedContentRequestsPerDayPerUserPerProjectPerModel"
                        ),
                    }
                },
            ),
            False,
        ),
        (
            ClientError(
                429,
                {"error": {"message": "Requests per minute exceeded."}},
            ),
            True,
        ),
        (
            ServerError(
                503,
                {"error": {"message": "Temporary high demand."}},
            ),
            True,
        ),
        (
            ClientError(
                400,
                {"error": {"message": "Invalid request."}},
            ),
            False,
        ),
    ],
)
def test_embedding_retry_classification(error, expected):
    assert is_retryable(error) is expected
