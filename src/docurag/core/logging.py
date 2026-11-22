import json
import logging
from datetime import UTC, datetime
from logging.config import dictConfig

from docurag.llm.accounting import current_accounting


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        accounting = current_accounting.get()

        if accounting is not None:
            payload["request_id"] = accounting.request_id

        event_data = getattr(record, "event_data", None)

        if isinstance(event_data, dict):
            # Keep formatter-owned fields protected.
            for key, value in event_data.items():
                if key not in {
                    "timestamp",
                    "level",
                    "logger",
                    "message",
                    "exception",
                }:
                    payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


def configure_logging() -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JSONFormatter,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "json",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "INFO",
            },
            "loggers": {
                "docurag": {
                    "handlers": [],
                    "level": "INFO",
                    "propagate": True,
                },
                "uvicorn": {
                    "handlers": [],
                    "level": "INFO",
                    "propagate": True,
                },
                "uvicorn.error": {
                    "handlers": [],
                    "level": "INFO",
                    "propagate": True,
                },
                "uvicorn.access": {
                    "handlers": [],
                    "level": "INFO",
                    "propagate": True,
                },
            },
        }
    )
