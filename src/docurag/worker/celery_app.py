from celery import Celery
from celery.signals import setup_logging

from docurag.core.config import get_settings
from docurag.core.logging import configure_logging


@setup_logging.connect
def configure_worker_logging(**kwargs: object) -> None:
    configure_logging()


settings = get_settings()

celery_app = Celery(
    "docurag",
    broker=settings.redis_url,
    include=["docurag.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    task_default_queue="ingestion",
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_connection_timeout=5,
    task_publish_retry=False,
    broker_transport_options={
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
        "visibility_timeout": 7200,
    },
)
