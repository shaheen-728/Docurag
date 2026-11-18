import argparse
import json
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://kubernetes.io/docs/concepts/"

DOCUMENT_PATHS = [
    "overview/",
    "overview/components/",
    "overview/working-with-objects/",
    "overview/working-with-objects/names/",
    "overview/working-with-objects/namespaces/",
    "overview/working-with-objects/labels/",
    "overview/working-with-objects/annotations/",
    "overview/working-with-objects/field-selectors/",
    "overview/working-with-objects/owners-dependents/",
    "architecture/nodes/",
    "architecture/controller/",
    "architecture/garbage-collection/",
    "containers/",
    "containers/images/",
    "containers/container-lifecycle-hooks/",
    "workloads/pods/",
    "workloads/pods/pod-lifecycle/",
    "workloads/pods/init-containers/",
    "workloads/pods/sidecar-containers/",
    "workloads/controllers/deployment/",
    "workloads/controllers/replicaset/",
    "workloads/controllers/statefulset/",
    "workloads/controllers/daemonset/",
    "workloads/controllers/job/",
    "workloads/controllers/cron-jobs/",
    "services-networking/service/",
    "services-networking/dns-pod-service/",
    "services-networking/ingress/",
    "services-networking/network-policies/",
    "services-networking/endpoint-slices/",
    "configuration/configmap/",
    "configuration/secret/",
    "storage/volumes/",
    "storage/persistent-volumes/",
    "storage/storage-classes/",
]

CORPUS_DIRECTORY = Path("data/corpus/kubernetes")
REPORT_PATH = Path("data/reports/kubernetes-ingestion.json")


def download_document(
    client: httpx.Client,
    document_path: str,
) -> tuple[Path, str]:
    source_url = BASE_URL + document_path

    filename = document_path.strip("/").replace("/", "--") + ".html"
    destination = CORPUS_DIRECTORY / filename

    # Reuse the same snapshot on subsequent runs.
    if destination.exists():
        return destination, source_url

    response = client.get(source_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Exclude the site's navigation and sidebar from the corpus.
    article = soup.select_one(".td-content")

    if article is None:
        raise ValueError(f"Article content not found: {source_url}")

    for element in article.select("script, style, nav, .td-feedback, .td-page-meta"):
        element.decompose()

    if len(article.get_text(" ", strip=True)) < 100:
        raise ValueError(f"Insufficient article text: {source_url}")

    snapshot = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="UTF-8"></head>'
        f"<body>{article}</body></html>"
    )

    destination.write_text(snapshot, encoding="utf-8")

    # Modest spacing between requests to the documentation website.
    time.sleep(1)

    return destination, source_url


def upload_document(
    client: httpx.Client,
    api_url: str,
    path: Path,
) -> dict:
    with path.open("rb") as source:
        response = client.post(
            f"{api_url}/documents",
            files={
                "file": (
                    path.name,
                    source,
                    "text/html",
                )
            },
        )

    response.raise_for_status()
    return response.json()


def wait_for_document(
    client: httpx.Client,
    api_url: str,
    document_id: str,
    timeout_seconds: int,
) -> dict:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        response = client.get(f"{api_url}/documents/{document_id}")
        response.raise_for_status()

        document = response.json()

        if document["status"] in {"completed", "failed"}:
            return document

        time.sleep(2)

    raise TimeoutError(
        f"Document {document_id} did not finish within "
        f"{timeout_seconds} seconds. The worker may still be processing it."
    )


def save_report(records: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if not 1 <= args.limit <= len(DOCUMENT_PATHS):
        parser.error("--limit must be between 1 and 35.")

    if args.timeout < 1:
        parser.error("--timeout must be positive.")

    api_url = args.api_url.rstrip("/")
    CORPUS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    run_started = time.perf_counter()

    with httpx.Client(
        timeout=60,
        follow_redirects=True,
    ) as client:
        health = client.get(f"{api_url}/health")
        health.raise_for_status()

        for index, document_path in enumerate(
            DOCUMENT_PATHS[: args.limit],
            start=1,
        ):
            started = time.perf_counter()

            record = {
                "source_url": BASE_URL + document_path,
                "filename": None,
                "document_id": None,
                "status": "started",
            }

            print(
                f"\n[{index}/{args.limit}] {document_path}",
                flush=True,
            )

            try:
                path, source_url = download_document(
                    client,
                    document_path,
                )

                record.update(
                    filename=path.name,
                    source_url=source_url,
                )

                upload = upload_document(client, api_url, path)

                record.update(
                    document_id=upload["id"],
                    duplicate=upload["duplicate"],
                )

                document = wait_for_document(
                    client,
                    api_url,
                    upload["id"],
                    args.timeout,
                )

                record.update(
                    status=document["status"],
                    chunk_count=document["chunk_count"],
                    processing_seconds=document["processing_seconds"],
                    error_message=document["error_message"],
                )

            except Exception as error:
                record.update(
                    status="check_required",
                    error_message=f"{type(error).__name__}: {error}",
                )

            record["wall_seconds"] = round(
                time.perf_counter() - started,
                3,
            )

            records.append(record)
            save_report(records)

            print(
                f"Status: {record['status']} | "
                f"Chunks: {record.get('chunk_count', 0)} | "
                f"Worker seconds: {record.get('processing_seconds')}",
                flush=True,
            )

            if record["status"] != "completed":
                print(
                    "Stopped. Inspect the report and worker logs before continuing.",
                    flush=True,
                )
                break

    # Count each database document only once.
    completed = {
        record["document_id"]: record
        for record in records
        if record["status"] == "completed"
    }

    total_chunks = sum(record["chunk_count"] for record in completed.values())

    total_processing = sum(
        record["processing_seconds"] or 0 for record in completed.values()
    )

    print("\nSummary")
    print("Selected documents:", args.limit)
    print("Attempted documents:", len(records))
    print("Unique completed documents:", len(completed))
    print("Total stored chunks:", total_chunks)
    print("Sum of recorded worker seconds:", round(total_processing, 3))
    print(
        "Current run wall seconds:",
        round(time.perf_counter() - run_started, 3),
    )
    print("Report:", REPORT_PATH)

    if len(completed) != args.limit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
