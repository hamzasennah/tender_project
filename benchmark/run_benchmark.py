import json
import os
import sys
import time
from pathlib import Path


# --------------------------------------------------
# Django setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient


if "testserver" not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS.append("testserver")


# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------

DOCUMENT_ID = int(os.getenv("BENCHMARK_DOCUMENT_ID", "12"))
BENCHMARK_USERNAME = os.getenv("BENCHMARK_USERNAME", "Hamza")
TOP_K = int(os.getenv("BENCHMARK_TOP_K", "5"))
DELAY_SECONDS = float(os.getenv("BENCHMARK_DELAY_SECONDS", "3"))
MAX_ATTEMPTS = max(1, int(os.getenv("BENCHMARK_MAX_RETRIES", "3")))
RETRY_BACKOFF_SECONDS = float(os.getenv("BENCHMARK_RETRY_BACKOFF_SECONDS", "5"))

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results_final.json"
FALLBACK_ANSWER = "Information not found in the provided document."
TRANSIENT_HTTP_STATUSES = {429, 503}

APPROACHES = (
    {
        "name": "Enhanced RAG",
        "endpoint": f"/api/documents/{DOCUMENT_ID}/ask/",
        "metadata_key": "rag_metadata",
    },
    {
        "name": "RAPTOR v3",
        "endpoint": f"/api/documents/{DOCUMENT_ID}/raptor/ask/",
        "metadata_key": "raptor_metadata",
    },
    {
        "name": "Prompt Engineering v2",
        "endpoint": f"/api/documents/{DOCUMENT_ID}/prompt-engineering/ask/",
        "metadata_key": "prompt_engineering_metadata",
    },
)


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_authenticated_client():
    User = get_user_model()
    try:
        user = User.objects.get(username=BENCHMARK_USERNAME)
    except User.DoesNotExist as exc:
        raise RuntimeError(
            f"User '{BENCHMARK_USERNAME}' was not found."
        ) from exc

    client = APIClient()
    client.force_authenticate(user=user)
    return client


def json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def response_payload(response):
    if hasattr(response, "data"):
        return json_safe(response.data)

    try:
        return response.json()
    except Exception:
        return {
            "raw_response": response.content.decode("utf-8", errors="replace"),
        }


def error_code(payload):
    return payload.get("error_code") if isinstance(payload, dict) else None


def error_message(payload):
    return payload.get("error_message") if isinstance(payload, dict) else None


def provider_incident_results_file():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).parent / f"results_provider_incident_{timestamp}.json"


def ask_endpoint_once(client, endpoint, question):
    return client.post(
        endpoint,
        {
            "question": question,
            "top_k": TOP_K,
        },
        format="json",
    )


def ask_endpoint(client, endpoint, question):
    started_at = time.perf_counter()
    attempts = []
    response = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = ask_endpoint_once(client, endpoint, question)
        payload = response_payload(response)
        attempts.append(
            {
                "attempt": attempt,
                "http_status": response.status_code,
                "error_code": error_code(payload),
                "error_message": error_message(payload),
            }
        )

        if response.status_code not in TRANSIENT_HTTP_STATUSES:
            break
        if attempt >= MAX_ATTEMPTS:
            break

        delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        print(
            "    transient provider error "
            f"HTTP {response.status_code}; retrying in {delay} s "
            f"(attempt {attempt + 1}/{MAX_ATTEMPTS})"
        )
        time.sleep(delay)

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return response, elapsed_ms, attempts


def fallback_detected(answer):
    return (answer or "").strip() == FALLBACK_ANSWER


def compact_sources(sources):
    compact = []
    for source in sources or []:
        compact.append(
            {
                "source_number": source.get("source_number"),
                "chunk_id": source.get("chunk_id"),
                "node_id": source.get("node_id"),
                "level": source.get("level"),
                "score": source.get("score"),
                "document_id": source.get("document_id"),
                "chunk_index": source.get("chunk_index"),
                "page_number": source.get("page_number"),
                "section_title": source.get("section_title"),
                "matched_subintents": source.get("matched_subintents"),
            }
        )
    return compact


def enhanced_rag_useful_metadata(metadata, sources):
    context = metadata.get("context", {}) or {}
    retrieval = metadata.get("retrieval", {}) or {}

    return {
        "retrieval_strategy": retrieval.get("search_type"),
        "retrieval": retrieval,
        "query_decomposition": metadata.get("query_decomposition"),
        "multi_query_metadata": metadata.get("multi_query_retrieval"),
        "context_char_count": context.get("context_char_count"),
        "context_truncated": context.get("context_truncated"),
        "included_count": context.get("included_count"),
        "sources": compact_sources(sources),
        "chunks": [
            source.get("chunk_id")
            for source in sources or []
            if source.get("chunk_id") is not None
        ],
    }


def raptor_useful_metadata(metadata, sources):
    return {
        "retrieval_strategy": metadata.get("retrieval_strategy"),
        "multi_intent": metadata.get("multi_intent"),
        "subintents": metadata.get("subintents"),
        "coverage": metadata.get("coverage"),
        "selected_nodes_per_level": metadata.get("selected_nodes_per_level"),
        "context_char_count": metadata.get("context_char_count"),
        "context_truncated": metadata.get("context_truncated"),
        "sources": compact_sources(sources),
        "nodes": [
            {
                "node_id": source.get("node_id"),
                "level": source.get("level"),
                "score": source.get("score"),
                "matched_subintents": source.get("matched_subintents"),
            }
            for source in sources or []
        ],
        "levels": [
            source.get("level")
            for source in sources or []
            if source.get("level") is not None
        ],
    }


def prompt_engineering_useful_metadata(metadata):
    return {
        "original_char_count": metadata.get("original_char_count"),
        "context_char_count": metadata.get("context_char_count"),
        "original_token_count": metadata.get("original_token_count"),
        "context_token_count": metadata.get("context_token_count"),
        "estimated_original_token_count": metadata.get(
            "estimated_original_token_count"
        ),
        "estimated_context_token_count": metadata.get(
            "estimated_context_token_count"
        ),
        "token_count_measured": metadata.get("token_count_measured"),
        "safe_input_token_limit": metadata.get("safe_input_token_limit"),
        "truncated": metadata.get("truncated"),
        "full_document_used": metadata.get("full_document_used"),
    }


def useful_metadata_for(approach_name, metadata, sources):
    if approach_name == "Enhanced RAG":
        return enhanced_rag_useful_metadata(metadata, sources)
    if approach_name == "RAPTOR v3":
        return raptor_useful_metadata(metadata, sources)
    if approach_name == "Prompt Engineering v2":
        return prompt_engineering_useful_metadata(metadata)
    return {}


def collect_result(client, item, approach):
    response, elapsed_ms, attempt_details = ask_endpoint(
        client,
        approach["endpoint"],
        item["question"],
    )
    payload = response_payload(response)
    answer = payload.get("answer") if isinstance(payload, dict) else None
    sources = payload.get("sources", []) if isinstance(payload, dict) else []
    metadata = (
        payload.get(approach["metadata_key"], {})
        if isinstance(payload, dict)
        else {}
    )

    result = {
        "question_id": item["id"],
        "question_type": item["type"],
        "question": item["question"],
        "expected": item.get("expected"),
        "approach": approach["name"],
        "endpoint": approach["endpoint"],
        "http_status": response.status_code,
        "final_http_status": response.status_code,
        "answer": answer,
        "elapsed_ms": elapsed_ms,
        "attempts": len(attempt_details),
        "attempt_details": attempt_details,
        "error_code": error_code(payload),
        "error_message": error_message(payload),
        "fallback_detected": fallback_detected(answer),
        "metadata": metadata,
        "useful_metadata": useful_metadata_for(
            approach["name"],
            metadata,
            sources,
        ),
        "sources": sources,
    }

    if response.status_code != 200:
        result["error"] = payload

    return result


def print_result_summary(result):
    useful = result["useful_metadata"]
    print(
        f"  [{result['approach']}] HTTP {result['http_status']} "
        f"- {result['elapsed_ms']} ms - attempts: {result['attempts']}"
    )
    print(f"    fallback_detected: {result['fallback_detected']}")

    if result["answer"]:
        one_line_answer = " ".join(str(result["answer"]).split())
        print(f"    answer: {one_line_answer[:220]}")

    if result["approach"] == "Enhanced RAG":
        print(f"    retrieval_strategy: {useful.get('retrieval_strategy')}")
        print(f"    context_char_count: {useful.get('context_char_count')}")
        print(f"    chunks: {useful.get('chunks')}")
    elif result["approach"] == "RAPTOR v3":
        print(f"    retrieval_strategy: {useful.get('retrieval_strategy')}")
        print(f"    multi_intent: {useful.get('multi_intent')}")
        print(
            "    selected_nodes_per_level: "
            f"{useful.get('selected_nodes_per_level')}"
        )
        print(f"    context_char_count: {useful.get('context_char_count')}")
    elif result["approach"] == "Prompt Engineering v2":
        print(f"    full_document_used: {useful.get('full_document_used')}")
        print(f"    truncated: {useful.get('truncated')}")
        print(f"    context_char_count: {useful.get('context_char_count')}")
        print(
            "    token_count: "
            f"{useful.get('context_token_count') or useful.get('estimated_context_token_count')}"
        )

    if result.get("error"):
        print(f"    error: {result['error']}")


def write_results(payload):
    result_file = (
        RESULTS_FILE
        if payload["benchmark"]["valid_final_benchmark"]
        else provider_incident_results_file()
    )
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return result_file


def status_counts(results):
    counts = {}
    for result in results:
        status = str(result["http_status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def main():
    questions = load_questions()
    client = get_authenticated_client()
    started_at = time.perf_counter()
    results = []

    print("=" * 80)
    print("FINAL BENCHMARK - Enhanced RAG / RAPTOR v3 / Prompt Engineering v2")
    print(f"Document: {DOCUMENT_ID}")
    print(f"User: {BENCHMARK_USERNAME}")
    print(f"Questions: {len(questions)}")
    print(f"Results file on full success: {RESULTS_FILE}")
    print(f"Delay between calls: {DELAY_SECONDS} s")
    print(f"Max attempts for 429/503: {MAX_ATTEMPTS}")
    print("=" * 80)

    total_expected_calls = len(questions) * len(APPROACHES)
    call_index = 0

    for item in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION {item['id']} - {item['type']}")
        print("=" * 80)
        print(item["question"])
        if item.get("expected"):
            print(f"Expected: {item['expected']}")

        for approach in APPROACHES:
            result = collect_result(client, item, approach)
            results.append(result)
            call_index += 1
            print_result_summary(result)
            if DELAY_SECONDS > 0 and call_index < total_expected_calls:
                time.sleep(DELAY_SECONDS)

    total_elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    success_count = sum(1 for result in results if result["http_status"] == 200)
    total_count = len(results)
    counts = status_counts(results)
    retry_count = sum(max(0, result["attempts"] - 1) for result in results)
    other_error_count = sum(
        1
        for result in results
        if result["http_status"] not in {200, 429, 503}
    )
    valid_final_benchmark = success_count == total_count

    output = {
        "benchmark": {
            "document_id": DOCUMENT_ID,
            "username": BENCHMARK_USERNAME,
            "questions_file": str(QUESTIONS_FILE),
            "result_count": total_count,
            "success_count": success_count,
            "status_counts": counts,
            "retry_count": retry_count,
            "valid_final_benchmark": valid_final_benchmark,
            "duration_ms": total_elapsed_ms,
            "duration_seconds": round(total_elapsed_ms / 1000, 2),
            "delay_seconds": DELAY_SECONDS,
            "max_attempts_for_429_503": MAX_ATTEMPTS,
            "retry_backoff_seconds": RETRY_BACKOFF_SECONDS,
            "approaches": [approach["name"] for approach in APPROACHES],
        },
        "results": results,
    }
    result_file = write_results(output)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"HTTP 200: {success_count}/{total_count}")
    print(f"HTTP 429: {counts.get('429', 0)}")
    print(f"HTTP 503: {counts.get('503', 0)}")
    print(f"Other errors: {other_error_count}")
    print(f"Retries: {retry_count}")
    print(f"Total duration: {total_elapsed_ms} ms")
    print(f"Saved results: {result_file}")
    print(f"Valid final benchmark: {valid_final_benchmark}")
    if not valid_final_benchmark:
        print(
            "This run is a provider-incident record, not a scientific final "
            "benchmark."
        )

    failed_results = [
        result
        for result in results
        if result["http_status"] != 200
    ]
    if failed_results:
        print("Errors:")
        for result in failed_results:
            print(
                "  "
                f"Q{result['question_id']} - {result['approach']} "
                f"HTTP {result['http_status']} "
                f"{result.get('error_code') or ''}"
            )
    else:
        print("Errors: none")


if __name__ == "__main__":
    main()
