import asyncio

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from docurag.core.tracing import get_tracing_client
from docurag.llm.accounting import current_accounting


class RequestTracingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http" or scope["path"] not in {
            "/chat",
            "/search",
        }:
            await self.app(scope, receive, send)
            return

        client = get_tracing_client()

        if client is None:
            await self.app(scope, receive, send)
            return

        accounting = current_accounting.get()
        metadata = {
            "method": scope["method"],
            "path": scope["path"],
        }

        if accounting is not None:
            metadata["request_id"] = accounting.request_id

        context = client.start_as_current_observation(
            name=f"docurag{scope['path']}",
            as_type="span",
            metadata=metadata,
        )

        http_status = None
        response_finished = False
        outcome = "exception"

        async def tracked_send(message: Message) -> None:
            nonlocal http_status, response_finished

            if message["type"] == "http.response.start":
                http_status = message["status"]

            await send(message)

            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_finished = True

        with context as observation:
            try:
                await self.app(scope, receive, tracked_send)
                outcome = "returned"
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                observation.update(
                    output={
                        "http_status": http_status,
                        "response_finished": response_finished,
                        "transport_outcome": outcome,
                    }
                )
