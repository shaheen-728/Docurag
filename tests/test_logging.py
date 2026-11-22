import json
import logging

from docurag.core.logging import JSONFormatter
from docurag.llm.accounting import (
    RequestAccounting,
    current_accounting,
)


def test_json_logging_preserves_message_fields_and_request_id():
    token = current_accounting.set(RequestAccounting(request_id="test-request"))

    try:
        record = logging.LogRecord(
            name="docurag.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Processed %s",
            args=("document",),
            exc_info=None,
        )
        record.event_data = {
            "event": "document_processed",
            "chunk_count": 7,
            "level": "incorrect-overwrite",
        }

        payload = json.loads(JSONFormatter().format(record))

        assert payload["message"] == "Processed document"
        assert payload["level"] == "INFO"
        assert payload["request_id"] == "test-request"
        assert payload["event"] == "document_processed"
        assert payload["chunk_count"] == 7

    finally:
        current_accounting.reset(token)


def test_exception_remains_in_one_json_line():
    try:
        raise ValueError("Example failure")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="docurag.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="Operation failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    formatted = JSONFormatter().format(record)
    payload = json.loads(formatted)

    assert len(formatted.splitlines()) == 1
    assert "ValueError: Example failure" in payload["exception"]
