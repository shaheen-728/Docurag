import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


@dataclass
class RequestAccounting:
    request_id: str
    calls: list[dict] = field(default_factory=list)


current_accounting: ContextVar[RequestAccounting | None] = ContextVar(
    "current_accounting",
    default=None,
)


def record_generation(
    *,
    provider: str,
    model: str,
    operation: str,
    input_tokens: int | None,
    total_tokens: int | None,
    estimated_cost_usd: float | None,
    completed: bool,
) -> None:
    accounting = current_accounting.get()

    if accounting is None:
        return

    accounting.calls.append(
        {
            "provider": provider,
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "completed": completed,
        }
    )


def record_embedding(
    *,
    model: str,
    task: str,
    text_count: int,
    latency_ms: float,
    completed: bool,
) -> None:
    accounting = current_accounting.get()

    if accounting is None:
        return

    accounting.calls.append(
        {
            "provider": "gemini",
            "model": model,
            "operation": "embed",
            "task": task,
            "text_count": text_count,
            "latency_ms": round(latency_ms, 3),
            "input_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": None,
            "usage_source": "not_returned_by_provider",
            "completed": completed,
        }
    )


class ChatAccountingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope["path"] != "/chat"
            or scope["method"] != "POST"
        ):
            await self.app(scope, receive, send)
            return

        accounting = RequestAccounting(request_id=str(uuid4()))
        context_token = current_accounting.set(accounting)
        started = perf_counter()
        http_status = None
        response_finished = False

        async def tracked_send(message: Message) -> None:
            nonlocal http_status, response_finished

            if message["type"] == "http.response.start":
                http_status = message["status"]
                message = dict(message)
                message["headers"] = [
                    *message.get("headers", []),
                    (
                        b"x-request-id",
                        accounting.request_id.encode("ascii"),
                    ),
                ]

            await send(message)

            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_finished = True

        try:
            await self.app(scope, receive, tracked_send)

        finally:
            try:
                all_calls = accounting.calls

                calls = [call for call in all_calls if call["operation"] != "embed"]
                embedding_calls = [
                    call for call in all_calls if call["operation"] == "embed"
                ]
                costs_known = all(
                    call["estimated_cost_usd"] is not None for call in calls
                )
                tokens_known = all(
                    call["completed"] and call["total_tokens"] is not None
                    for call in calls
                )
                known_cost = round(
                    sum(
                        call["estimated_cost_usd"]
                        for call in calls
                        if call["estimated_cost_usd"] is not None
                    ),
                    10,
                )

                summary = {
                    "event": "chat_accounting",
                    "request_id": accounting.request_id,
                    "http_status": http_status,
                    "response_finished": response_finished,
                    "latency_ms": round(
                        (perf_counter() - started) * 1000,
                        3,
                    ),
                    "accounting_scope": "gemini_generation_and_embeddings",
                    "embedding_calls": len(embedding_calls),
                    "embedding_total_tokens": (None if embedding_calls else 0),
                    "embedding_cost_usd": (None if embedding_calls else 0.0),
                    "api_cost_usd": (
                        known_cost if costs_known and not embedding_calls else None
                    ),
                    "generation_calls": len(calls),
                    "generation_total_tokens": (
                        sum(call["total_tokens"] for call in calls)
                        if tokens_known
                        else None
                    ),
                    "generation_tokens_complete": tokens_known,
                    "known_generation_cost_usd": known_cost,
                    "generation_cost_usd": (known_cost if costs_known else None),
                    "generation_cost_complete": costs_known,
                    "request_cost_complete": False,
                    "excluded_costs": [
                        "hosting",
                        "database",
                    ],
                    "unavailable_usage": (
                        ["embedding_tokens_and_cost"] if embedding_calls else []
                    ),
                    "calls": all_calls,
                }

                logger.info(
                    "chat_accounting",
                    extra={"event_data": summary},
                )
            finally:
                current_accounting.reset(context_token)
