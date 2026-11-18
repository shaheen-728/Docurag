from celery import Celery

from docurag.core.config import get_settings

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
