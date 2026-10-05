from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from documents.models import Document
from extraction.models import ChunkEmbedding, RaptorIndex, TextChunk


def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return redirect("login")


def _max_upload_size_mb():
    max_bytes = getattr(settings, "DOCUMENTS_MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
    return round(max_bytes / (1024 * 1024), 1)


def _safe_related(instance, related_name):
    try:
        return getattr(instance, related_name)
    except ObjectDoesNotExist:
        return None


def _greeting():
    hour = timezone.localtime().hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def _document_summary(document):
    extraction = _safe_related(document, "text_extraction")
    raptor_index = _safe_related(document, "raptor_index")
    chunk_count = TextChunk.objects.filter(document=document).count()
    embedding_count = ChunkEmbedding.objects.filter(chunk__document=document).count()
    business_status = _business_status(document, extraction, embedding_count)

    return {
        "document": document,
        "extraction": extraction,
        "raptor_index": raptor_index,
        "page_count": extraction.page_count if extraction else None,
        "character_count": extraction.character_count if extraction else 0,
        "chunk_count": chunk_count,
        "embedding_count": embedding_count,
        "business_status": business_status,
        "status_label": business_status["label"],
        "status_tone": business_status["tone"],
        "is_ready": business_status["key"] == "ready",
        "is_failed": business_status["key"] == "failed",
        "pipeline_status": _pipeline_status(extraction, chunk_count, embedding_count, raptor_index),
    }


def _business_status(document, extraction, embedding_count):
    if document.status == Document.Status.FAILED:
        return {"key": "failed", "label": "Failed", "tone": "danger"}
    if extraction and extraction.status == "failed":
        return {"key": "failed", "label": "Failed", "tone": "danger"}
    if embedding_count > 0:
        return {"key": "ready", "label": "Ready", "tone": "success"}
    return {"key": "preparing", "label": "Preparing", "tone": "pending"}


def _pipeline_status(extraction, chunk_count, embedding_count, raptor_index):
    if raptor_index and raptor_index.status == RaptorIndex.Status.COMPLETED:
        return "RAPTOR ready"
    if embedding_count:
        return "Embeddings ready"
    if chunk_count:
        return "Chunks ready"
    if extraction and extraction.status == "completed":
        return "Text extracted"
    if extraction and extraction.status == "failed":
        return "Extraction failed"
    return "Awaiting extraction"


def _document_queryset(user):
    return (
        Document.objects.filter(owner=user)
        .select_related("text_extraction", "raptor_index")
        .order_by("-created_at")
    )


@login_required
def dashboard(request):
    documents = _document_queryset(request.user)
    document_summaries = [_document_summary(document) for document in documents[:8]]

    return render(
        request,
        "dashboard/index.html",
        {
            "active_nav": "documents",
            "page_title": "Your documents",
            "greeting": _greeting(),
            "documents": document_summaries,
            "max_upload_size_mb": _max_upload_size_mb(),
        },
    )


@login_required
def document_library(request):
    documents = [_document_summary(document) for document in _document_queryset(request.user)]
    return render(
        request,
        "documents/list.html",
        {
            "active_nav": "documents",
            "page_title": "Documents",
            "documents": documents,
            "max_upload_size_mb": _max_upload_size_mb(),
        },
    )


@login_required
def document_workspace(request, document_id):
    document = get_object_or_404(_document_queryset(request.user), pk=document_id)
    summary = _document_summary(document)

    extraction = summary["extraction"]
    raptor_index = summary["raptor_index"]
    extraction_completed = bool(extraction and extraction.status == "completed" and extraction.has_text)
    chunks_ready = summary["chunk_count"] > 0
    embeddings_ready = summary["embedding_count"] > 0
    raptor_ready = bool(raptor_index and raptor_index.status == RaptorIndex.Status.COMPLETED)

    pipeline = [
        {
            "key": "document",
            "label": "Document",
            "description": "PDF stored and scoped to your account.",
            "status": document.status,
            "state": "done",
        },
        {
            "key": "extraction",
            "label": "Extraction",
            "description": "Native text extraction with OCR fallback when needed.",
            "status": extraction.status if extraction else "not_started",
            "state": "done" if extraction_completed else "failed" if extraction and extraction.status == "failed" else "ready",
            "action": "Extract text" if not extraction_completed else "",
            "endpoint_name": "document-text-extraction",
        },
        {
            "key": "chunks",
            "label": "Chunks",
            "description": "Semantic chunks prepared for retrieval.",
            "status": f"{summary['chunk_count']} chunks",
            "state": "done" if chunks_ready else "ready" if extraction_completed else "locked",
            "action": "Generate chunks" if extraction_completed and not chunks_ready else "",
            "endpoint_name": "document-text-chunks",
        },
        {
            "key": "embeddings",
            "label": "Embeddings",
            "description": "Vector representations stored in pgvector.",
            "status": f"{summary['embedding_count']} embeddings",
            "state": "done" if embeddings_ready else "ready" if chunks_ready else "locked",
            "action": "Generate embeddings" if chunks_ready and not embeddings_ready else "",
            "endpoint_name": "document-chunk-embeddings",
        },
        {
            "key": "raptor",
            "label": "RAPTOR Index",
            "description": "Hierarchical summaries and retrieval graph.",
            "status": raptor_index.status if raptor_index else "not_started",
            "state": "done" if raptor_ready else "failed" if raptor_index and raptor_index.status == "failed" else "ready" if embeddings_ready else "locked",
            "action": "Build RAPTOR" if embeddings_ready and not raptor_ready else "",
            "endpoint_name": "document-raptor-index",
        },
    ]

    return render(
        request,
        "documents/detail.html",
        {
            "active_nav": "documents",
            "page_title": document.original_filename,
            "summary": summary,
            "document": document,
            "pipeline": pipeline,
            "extraction_ready": extraction_completed,
            "chunks_ready": chunks_ready,
            "embeddings_ready": embeddings_ready,
            "raptor_ready": raptor_ready,
            "business_status": summary["business_status"],
            "can_ask": summary["is_ready"],
        },
    )
