import logging
import math
import time
from dataclasses import dataclass

from django.conf import settings
from django.db import DatabaseError, transaction
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from rest_framework import status

from extraction.models import ChunkEmbedding, TextChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_PROVIDER = "gemini"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_MAX_CHUNKS_PER_RUN = 200
DEFAULT_EMBEDDING_RETRY_COUNT = 2
DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS = 0.25
DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECONDS = 30


class EmbeddingServiceError(Exception):
    def __init__(
        self,
        code,
        public_message,
        internal_detail="",
        response_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        retryable=False,
    ):
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.internal_detail = internal_detail
        self.response_status = response_status
        self.retryable = retryable


class EmbeddingProviderError(EmbeddingServiceError):
    pass


@dataclass(frozen=True)
class EmbeddingProvider:
    provider_name: str
    model: str
    dimension: int

    def embed_texts(self, texts):
        raise NotImplementedError


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, api_key, model, dimension, timeout_seconds):
        super().__init__(
            provider_name="gemini",
            model=model,
            dimension=dimension,
        )
        if not api_key:
            raise EmbeddingProviderError(
                "missing_api_key",
                "Embedding provider API key is not configured.",
                response_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )

    def embed_texts(self, texts):
        contents = [
            types.Content(role="user", parts=[types.Part(text=text)])
            for text in texts
        ]
        config = types.EmbedContentConfig(output_dimensionality=self.dimension)

        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as exc:
            code = getattr(exc, "code", None)
            error_code = _client_error_code(code)
            raise EmbeddingProviderError(
                error_code,
                _public_message_for_provider_error(error_code),
                exc.__class__.__name__,
                _response_status_for_provider_error(error_code),
                retryable=(code == 429),
            ) from exc
        except genai_errors.ServerError as exc:
            raise EmbeddingProviderError(
                "embedding_service_error",
                "Embedding provider service failed.",
                exc.__class__.__name__,
                status.HTTP_503_SERVICE_UNAVAILABLE,
                retryable=True,
            ) from exc
        except genai_errors.APIError as exc:
            raise EmbeddingProviderError(
                "embedding_provider_error",
                "Embedding provider request failed.",
                exc.__class__.__name__,
                status.HTTP_502_BAD_GATEWAY,
                retryable=False,
            ) from exc
        except TimeoutError as exc:
            raise EmbeddingProviderError(
                "embedding_timeout",
                "Embedding provider request timed out.",
                exc.__class__.__name__,
                status.HTTP_503_SERVICE_UNAVAILABLE,
                retryable=True,
            ) from exc
        except OSError as exc:
            raise EmbeddingProviderError(
                "embedding_network_error",
                "Embedding provider network request failed.",
                exc.__class__.__name__,
                status.HTTP_503_SERVICE_UNAVAILABLE,
                retryable=True,
            ) from exc

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None or len(embeddings) != len(texts):
            raise EmbeddingProviderError(
                "malformed_embedding_response",
                "Embedding provider returned an invalid response.",
                "embedding_count_mismatch",
                status.HTTP_502_BAD_GATEWAY,
            )

        return [list(embedding.values or []) for embedding in embeddings]


def _client_error_code(code):
    if code in {401, 403}:
        return "embedding_authentication_failed"
    if code == 429:
        return "embedding_rate_limited"
    if code == 400:
        return "embedding_invalid_request"
    if code == 404:
        return "embedding_model_not_found"
    return "embedding_provider_error"


def _public_message_for_provider_error(error_code):
    messages = {
        "embedding_authentication_failed": "Embedding provider authentication failed.",
        "embedding_rate_limited": "Embedding provider rate limit was reached.",
        "embedding_invalid_request": "Embedding provider rejected the request.",
        "embedding_model_not_found": "Configured embedding model was not found.",
    }
    return messages.get(error_code, "Embedding provider request failed.")


def _response_status_for_provider_error(error_code):
    statuses = {
        "embedding_authentication_failed": status.HTTP_502_BAD_GATEWAY,
        "embedding_rate_limited": status.HTTP_429_TOO_MANY_REQUESTS,
        "embedding_invalid_request": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "embedding_model_not_found": status.HTTP_422_UNPROCESSABLE_ENTITY,
    }
    return statuses.get(error_code, status.HTTP_502_BAD_GATEWAY)


def _get_positive_int_setting(name, default):
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _get_nonnegative_float_setting(name, default):
    try:
        value = float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(0.0, value)


def get_embedding_provider_name():
    return getattr(settings, "EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)


def get_embedding_model():
    return getattr(settings, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_dimension():
    return _get_positive_int_setting(
        "EMBEDDING_DIMENSION",
        DEFAULT_EMBEDDING_DIMENSION,
    )


def get_embedding_batch_size():
    return _get_positive_int_setting(
        "EMBEDDING_BATCH_SIZE",
        DEFAULT_EMBEDDING_BATCH_SIZE,
    )


def get_embedding_max_chunks_per_run():
    return _get_positive_int_setting(
        "EMBEDDING_MAX_CHUNKS_PER_RUN",
        DEFAULT_EMBEDDING_MAX_CHUNKS_PER_RUN,
    )


def get_embedding_retry_count():
    try:
        value = int(getattr(settings, "EMBEDDING_RETRY_COUNT", DEFAULT_EMBEDDING_RETRY_COUNT))
    except (TypeError, ValueError):
        value = DEFAULT_EMBEDDING_RETRY_COUNT
    return max(0, value)


def get_embedding_retry_backoff_seconds():
    return _get_nonnegative_float_setting(
        "EMBEDDING_RETRY_BACKOFF_SECONDS",
        DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS,
    )


def get_embedding_request_timeout_seconds():
    return _get_positive_int_setting(
        "EMBEDDING_REQUEST_TIMEOUT_SECONDS",
        DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECONDS,
    )


def get_embedding_api_key():
    return getattr(settings, "GEMINI_API_KEY", "") or ""


def get_embedding_provider():
    provider_name = get_embedding_provider_name()
    if provider_name != "gemini":
        raise EmbeddingServiceError(
            "unsupported_embedding_provider",
            "Configured embedding provider is not supported.",
            response_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return GeminiEmbeddingProvider(
        api_key=get_embedding_api_key(),
        model=get_embedding_model(),
        dimension=get_embedding_dimension(),
        timeout_seconds=get_embedding_request_timeout_seconds(),
    )


def _embedding_limits_metadata():
    return {
        "batch_size": get_embedding_batch_size(),
        "max_chunks_per_run": get_embedding_max_chunks_per_run(),
        "retry_count": get_embedding_retry_count(),
        "retry_backoff_seconds": get_embedding_retry_backoff_seconds(),
        "request_timeout_seconds": get_embedding_request_timeout_seconds(),
    }


def _validate_embedding_configuration(provider):
    if provider.provider_name != "gemini":
        raise EmbeddingServiceError(
            "unsupported_embedding_provider",
            "Configured embedding provider is not supported.",
            response_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    if not provider.model:
        raise EmbeddingServiceError(
            "missing_embedding_model",
            "Embedding model is not configured.",
            response_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    if provider.dimension != get_embedding_dimension():
        raise EmbeddingServiceError(
            "invalid_embedding_dimension_configuration",
            "Embedding dimension configuration is invalid.",
            response_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _valid_vector(vector, dimension):
    if vector is None:
        return False
    if not isinstance(vector, list):
        try:
            vector = list(vector)
        except TypeError:
            return False
    if len(vector) != dimension:
        return False
    for value in vector:
        if not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def validate_embedding_vector(vector, dimension):
    if not _valid_vector(vector, dimension):
        raise EmbeddingServiceError(
            "invalid_embedding_vector",
            "Embedding provider returned an invalid vector.",
            response_status=status.HTTP_502_BAD_GATEWAY,
        )
    return [float(value) for value in vector]


def _is_embedding_current(embedding, chunk, provider):
    return (
        embedding is not None
        and embedding.status == ChunkEmbedding.Status.COMPLETED
        and embedding.provider == provider.provider_name
        and embedding.model == provider.model
        and embedding.dimension == provider.dimension
        and embedding.chunk_sha256 == chunk.text_sha256
        and _valid_vector(embedding.embedding_vector, provider.dimension)
    )


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _embed_with_retries(provider, texts):
    attempts = get_embedding_retry_count() + 1
    for attempt in range(attempts):
        try:
            return provider.embed_texts(texts)
        except EmbeddingProviderError as exc:
            if not exc.retryable or attempt >= attempts - 1:
                raise
            time.sleep(get_embedding_retry_backoff_seconds() * (2**attempt))

    raise EmbeddingProviderError(
        "embedding_provider_error",
        "Embedding provider request failed.",
        response_status=status.HTTP_502_BAD_GATEWAY,
    )


def _embedding_metadata(provider):
    return {
        "provider": provider.provider_name,
        "model": provider.model,
        "dimension": provider.dimension,
        "normalization": "provider_default",
        "normalization_note": (
            "No application-side normalization applied; Gemini returns normalized "
            "reduced-dimension embeddings for this model."
        ),
    }


def _save_embedding(chunk, provider, vector):
    ChunkEmbedding.objects.update_or_create(
        chunk=chunk,
        defaults={
            "provider": provider.provider_name,
            "model": provider.model,
            "dimension": provider.dimension,
            "vector": vector,
            "embedding_vector": vector,
            "chunk_sha256": chunk.text_sha256,
            "status": ChunkEmbedding.Status.COMPLETED,
            "error_code": "",
            "error_message": "",
            "metadata": _embedding_metadata(provider),
        },
    )


def _mark_batch_failed(chunks, provider, exc):
    for chunk in chunks:
        ChunkEmbedding.objects.update_or_create(
            chunk=chunk,
            defaults={
                "provider": provider.provider_name,
                "model": provider.model,
                "dimension": provider.dimension,
                "vector": [],
                "embedding_vector": None,
                "chunk_sha256": chunk.text_sha256,
                "status": ChunkEmbedding.Status.FAILED,
                "error_code": exc.code,
                "error_message": exc.public_message,
                "metadata": _embedding_metadata(provider),
            },
        )


def _document_chunks(document):
    return list(
        TextChunk.objects.filter(document=document)
        .select_related("document", "extraction_result", "embedding")
        .order_by("chunk_index")
    )


def _summary_payload(document, chunks, embeddings, metadata):
    return {
        "document_id": document.id,
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embeddings": embeddings,
        "embedding_metadata": metadata,
    }


def get_document_embeddings_status(document):
    chunks = _document_chunks(document)
    embeddings = list(
        ChunkEmbedding.objects.filter(chunk__in=chunks)
        .select_related("chunk")
        .order_by("chunk__chunk_index")
    )
    return _summary_payload(
        document,
        chunks,
        embeddings,
        {
            "provider": get_embedding_provider_name(),
            "model": get_embedding_model(),
            "dimension": get_embedding_dimension(),
            "limits": _embedding_limits_metadata(),
        },
    )


def embed_document_chunks(document, provider=None):
    provider = provider or get_embedding_provider()
    _validate_embedding_configuration(provider)

    chunks = _document_chunks(document)
    if not chunks:
        raise EmbeddingServiceError(
            "chunks_not_found",
            "Persisted text chunks were not found for this document.",
            response_status=status.HTTP_404_NOT_FOUND,
        )

    if len(chunks) > get_embedding_max_chunks_per_run():
        raise EmbeddingServiceError(
            "too_many_chunks_for_embedding",
            (
                "Document has more chunks than the configured embedding run limit "
                f"of {get_embedding_max_chunks_per_run()}."
            ),
        )

    existing_by_chunk_id = {
        embedding.chunk_id: embedding
        for embedding in ChunkEmbedding.objects.filter(chunk__in=chunks)
    }
    chunks_to_embed = [
        chunk
        for chunk in chunks
        if not _is_embedding_current(existing_by_chunk_id.get(chunk.id), chunk, provider)
    ]
    reused_count = len(chunks) - len(chunks_to_embed)
    generated_count = 0

    for batch in _batched(chunks_to_embed, get_embedding_batch_size()):
        try:
            vectors = _embed_with_retries(provider, [chunk.text for chunk in batch])
            if len(vectors) != len(batch):
                raise EmbeddingProviderError(
                    "partial_embedding_batch_failure",
                    "Embedding provider returned a partial batch response.",
                    "batch_vector_count_mismatch",
                    status.HTTP_502_BAD_GATEWAY,
                )
            clean_vectors = [
                validate_embedding_vector(vector, provider.dimension)
                for vector in vectors
            ]
        except EmbeddingProviderError as exc:
            with transaction.atomic():
                _mark_batch_failed(batch, provider, exc)
            logger.warning(
                "Chunk embedding batch failed",
                extra={
                    "document_id": document.id,
                    "provider": provider.provider_name,
                    "model": provider.model,
                    "error_code": exc.code,
                },
            )
            raise
        except EmbeddingServiceError as exc:
            with transaction.atomic():
                _mark_batch_failed(batch, provider, exc)
            logger.warning(
                "Chunk embedding validation failed",
                extra={
                    "document_id": document.id,
                    "provider": provider.provider_name,
                    "model": provider.model,
                    "error_code": exc.code,
                },
            )
            raise

        try:
            with transaction.atomic():
                for chunk, vector in zip(batch, clean_vectors):
                    _save_embedding(chunk, provider, vector)
                    generated_count += 1
        except DatabaseError as exc:
            raise EmbeddingServiceError(
                "embedding_storage_failed",
                "Chunk embeddings could not be stored safely.",
                exc.__class__.__name__,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    embeddings = list(
        ChunkEmbedding.objects.filter(chunk__in=chunks)
        .select_related("chunk")
        .order_by("chunk__chunk_index")
    )
    metadata = {
        "provider": provider.provider_name,
        "model": provider.model,
        "dimension": provider.dimension,
        "generated_count": generated_count,
        "reused_count": reused_count,
        "limits": _embedding_limits_metadata(),
        "normalization": "provider_default",
    }

    logger.info(
        "Chunk embeddings completed",
        extra={
            "document_id": document.id,
            "provider": provider.provider_name,
            "model": provider.model,
            "dimension": provider.dimension,
            "generated_count": generated_count,
            "reused_count": reused_count,
        },
    )
    return _summary_payload(document, chunks, embeddings, metadata)
