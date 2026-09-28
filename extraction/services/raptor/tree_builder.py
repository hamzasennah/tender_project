import hashlib
import logging
import time

from django.conf import settings
from django.db import DatabaseError, transaction
from rest_framework import status

from extraction.models import (
    ChunkEmbedding,
    RaptorIndex,
    RaptorNode,
    RaptorNodeChild,
    TextChunk,
)
from extraction.services.embeddings import (
    EmbeddingProviderError,
    EmbeddingServiceError,
    get_embedding_dimension,
    get_embedding_model,
    get_embedding_provider,
    get_embedding_provider_name,
    validate_embedding_vector,
)
from extraction.services.rag import RAGError, get_llm_provider
from extraction.services.raptor.clustering import cluster_embeddings
from extraction.services.raptor.summarization import summarize_cluster
from extraction.services.raptor.types import RaptorBuildItem

logger = logging.getLogger(__name__)

RAPTOR_BUILD_VERSION = "raptor-hierarchical-v1"
DEFAULT_RAPTOR_ENABLED = True
DEFAULT_RAPTOR_MAX_LEVELS = 4
DEFAULT_RAPTOR_MAX_CLUSTERS = 20
DEFAULT_RAPTOR_SOFT_CLUSTER_THRESHOLD = 0.1
DEFAULT_RAPTOR_RANDOM_STATE = 42
DEFAULT_RAPTOR_MAX_CHUNKS = 300
DEFAULT_RAPTOR_MAX_NODES = 1000
DEFAULT_RAPTOR_MAX_SUMMARY_CONTEXT_CHARS = 12000
DEFAULT_RAPTOR_MAX_SUMMARY_CHARS = 2000


class RaptorError(Exception):
    def __init__(
        self,
        code,
        public_message,
        internal_detail="",
        response_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.internal_detail = internal_detail
        self.response_status = response_status


def _get_bool_setting(name, default):
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def get_raptor_enabled():
    return _get_bool_setting("RAPTOR_ENABLED", DEFAULT_RAPTOR_ENABLED)


def get_raptor_max_levels():
    return _get_positive_int_setting("RAPTOR_MAX_LEVELS", DEFAULT_RAPTOR_MAX_LEVELS)


def get_raptor_max_clusters():
    return _get_positive_int_setting("RAPTOR_MAX_CLUSTERS", DEFAULT_RAPTOR_MAX_CLUSTERS)


def get_raptor_soft_cluster_threshold():
    return _get_nonnegative_float_setting(
        "RAPTOR_SOFT_CLUSTER_THRESHOLD",
        DEFAULT_RAPTOR_SOFT_CLUSTER_THRESHOLD,
    )


def get_raptor_random_state():
    return _get_positive_int_setting("RAPTOR_RANDOM_STATE", DEFAULT_RAPTOR_RANDOM_STATE)


def get_raptor_max_chunks():
    return _get_positive_int_setting("RAPTOR_MAX_CHUNKS", DEFAULT_RAPTOR_MAX_CHUNKS)


def get_raptor_max_nodes():
    return _get_positive_int_setting("RAPTOR_MAX_NODES", DEFAULT_RAPTOR_MAX_NODES)


def get_raptor_max_summary_context_chars():
    return _get_positive_int_setting(
        "RAPTOR_MAX_SUMMARY_CONTEXT_CHARS",
        DEFAULT_RAPTOR_MAX_SUMMARY_CONTEXT_CHARS,
    )


def get_raptor_max_summary_chars():
    return _get_positive_int_setting(
        "RAPTOR_MAX_SUMMARY_CHARS",
        DEFAULT_RAPTOR_MAX_SUMMARY_CHARS,
    )


def _limits_metadata():
    return {
        "enabled": get_raptor_enabled(),
        "max_levels": get_raptor_max_levels(),
        "max_clusters": get_raptor_max_clusters(),
        "soft_cluster_threshold": get_raptor_soft_cluster_threshold(),
        "random_state": get_raptor_random_state(),
        "max_chunks": get_raptor_max_chunks(),
        "max_nodes": get_raptor_max_nodes(),
        "max_summary_context_chars": get_raptor_max_summary_context_chars(),
        "max_summary_chars": get_raptor_max_summary_chars(),
    }


def _status_payload(document, index=None):
    if index is None:
        try:
            index = document.raptor_index
        except RaptorIndex.DoesNotExist:
            return {
                "document_id": document.id,
                "status": "not_built",
                "levels": 0,
                "leaf_count": 0,
                "summary_node_count": 0,
                "total_node_count": 0,
                "nodes_per_level": {},
                "clusters_per_level": {},
                "raptor_metadata": {
                    "build_version": RAPTOR_BUILD_VERSION,
                    "limits": _limits_metadata(),
                },
            }

    return {
        "document_id": document.id,
        "status": index.status,
        "levels": index.level_count,
        "leaf_count": index.leaf_count,
        "summary_node_count": index.summary_node_count,
        "total_node_count": index.total_node_count,
        "nodes_per_level": index.nodes_per_level,
        "clusters_per_level": index.clusters_per_level,
        "error_code": index.error_code,
        "error_message": index.error_message,
        "raptor_metadata": {
            "build_version": index.build_version or RAPTOR_BUILD_VERSION,
            "provider": index.provider,
            "embedding_model": index.embedding_model,
            "embedding_dimension": index.embedding_dimension,
            "metadata": index.metadata,
            "limits": _limits_metadata(),
        },
    }


def get_raptor_index_status(document):
    return _status_payload(document)


def _sha256_text(text):
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _document_chunks(document):
    return list(
        TextChunk.objects.filter(document=document)
        .select_related("extraction_result", "embedding")
        .order_by("chunk_index", "id")
    )


def _embedding_for_chunk(chunk, provider):
    embedding = getattr(chunk, "embedding", None)
    if (
        embedding is None
        or embedding.status != ChunkEmbedding.Status.COMPLETED
        or embedding.provider != provider.provider_name
        or embedding.model != provider.model
        or embedding.dimension != provider.dimension
        or embedding.chunk_sha256 != chunk.text_sha256
        or embedding.embedding_vector is None
    ):
        return None
    try:
        return validate_embedding_vector(list(embedding.embedding_vector), provider.dimension)
    except EmbeddingServiceError:
        return None


def _prepare_index(document, provider):
    with transaction.atomic():
        index, _created = RaptorIndex.objects.select_for_update().get_or_create(
            document=document,
            defaults={"status": RaptorIndex.Status.PENDING},
        )
        index.nodes.all().delete()
        index.status = RaptorIndex.Status.PROCESSING
        index.build_version = RAPTOR_BUILD_VERSION
        index.provider = provider.provider_name
        index.embedding_model = provider.model
        index.embedding_dimension = provider.dimension
        index.level_count = 0
        index.leaf_count = 0
        index.summary_node_count = 0
        index.total_node_count = 0
        index.nodes_per_level = {}
        index.clusters_per_level = {}
        index.metadata = {"started_at_monotonic": time.monotonic()}
        index.error_code = ""
        index.error_message = ""
        index.internal_error_detail = ""
        index.save()
    return index


def _mark_failed(index, exc):
    with transaction.atomic():
        locked_index = RaptorIndex.objects.select_for_update().get(pk=index.pk)
        locked_index.nodes.all().delete()
        locked_index.status = RaptorIndex.Status.FAILED
        locked_index.error_code = exc.code
        locked_index.error_message = exc.public_message
        locked_index.internal_error_detail = exc.internal_detail
        locked_index.save()


def _complete_index(index, metadata):
    with transaction.atomic():
        locked_index = RaptorIndex.objects.select_for_update().get(pk=index.pk)
        locked_index.status = RaptorIndex.Status.COMPLETED
        locked_index.level_count = metadata["level_count"]
        locked_index.leaf_count = metadata["leaf_count"]
        locked_index.summary_node_count = metadata["summary_node_count"]
        locked_index.total_node_count = metadata["total_node_count"]
        locked_index.nodes_per_level = metadata["nodes_per_level"]
        locked_index.clusters_per_level = metadata["clusters_per_level"]
        locked_index.metadata = metadata["metadata"]
        locked_index.error_code = ""
        locked_index.error_message = ""
        locked_index.internal_error_detail = ""
        locked_index.save()
        return locked_index


def _create_leaf_nodes(index, document, chunks, vectors):
    items = []
    for node_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        node = RaptorNode.objects.create(
            index=index,
            document=document,
            source_chunk=chunk,
            level=0,
            node_index=node_index,
            text="",
            text_sha256=chunk.text_sha256,
            embedding_vector=None,
            metadata={
                "source": "text_chunk",
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
            },
        )
        items.append(
            RaptorBuildItem(
                node_id=node.id,
                text=chunk.text,
                vector=vector,
                level=0,
            )
        )
    return items


def _embed_summaries(embedding_provider, summaries):
    try:
        vectors = embedding_provider.embed_texts(summaries)
    except EmbeddingServiceError:
        raise
    if len(vectors) != len(summaries):
        raise RaptorError(
            "malformed_summary_embedding_response",
            "Embedding provider returned an invalid RAPTOR summary embedding response.",
            "summary_embedding_count_mismatch",
            status.HTTP_502_BAD_GATEWAY,
        )
    return [
        validate_embedding_vector(vector, embedding_provider.dimension)
        for vector in vectors
    ]


def _create_summary_level(
    *,
    index,
    document,
    current_items,
    clusters,
    level,
    llm_provider,
    embedding_provider,
):
    summaries = []
    child_groups = []
    for cluster in clusters:
        members = [current_items[item_index] for item_index in cluster.member_indices]
        summary = summarize_cluster(
            llm_provider,
            members,
            max_context_chars=get_raptor_max_summary_context_chars(),
            max_summary_chars=get_raptor_max_summary_chars(),
        )
        if not summary:
            raise RaptorError(
                "empty_raptor_summary",
                "RAPTOR summarization produced an empty summary.",
                response_status=status.HTTP_502_BAD_GATEWAY,
            )
        summaries.append(summary)
        child_groups.append((cluster, members))

    summary_vectors = _embed_summaries(embedding_provider, summaries)
    next_items = []
    for node_index, (summary, vector, child_group) in enumerate(
        zip(summaries, summary_vectors, child_groups)
    ):
        cluster, members = child_group
        node = RaptorNode.objects.create(
            index=index,
            document=document,
            source_chunk=None,
            level=level,
            node_index=node_index,
            text=summary,
            text_sha256=_sha256_text(summary),
            embedding_vector=vector,
            metadata={
                "source": "cluster_summary",
                "cluster_id": cluster.cluster_id,
                "child_count": len(members),
                "summary_chars": len(summary),
            },
        )
        for child_rank, member in enumerate(members):
            RaptorNodeChild.objects.create(
                parent=node,
                child_id=member.node_id,
                child_rank=child_rank,
                metadata={"cluster_id": cluster.cluster_id},
            )
        next_items.append(
            RaptorBuildItem(
                node_id=node.id,
                text=summary,
                vector=vector,
                level=level,
            )
        )
    return next_items


def _build_tree(index, document, embedding_provider, llm_provider):
    chunks = _document_chunks(document)
    if not chunks:
        raise RaptorError(
            "chunks_not_found",
            "Persisted text chunks were not found for this document.",
            response_status=status.HTTP_404_NOT_FOUND,
        )
    if len(chunks) > get_raptor_max_chunks():
        raise RaptorError(
            "too_many_chunks_for_raptor",
            (
                "Document has more chunks than the configured RAPTOR build limit "
                f"of {get_raptor_max_chunks()}."
            ),
        )

    vectors = []
    missing_chunk_ids = []
    for chunk in chunks:
        vector = _embedding_for_chunk(chunk, embedding_provider)
        if vector is None:
            missing_chunk_ids.append(chunk.id)
        else:
            vectors.append(vector)
    if missing_chunk_ids:
        raise RaptorError(
            "embeddings_not_found",
            "Completed current chunk embeddings are required before building RAPTOR.",
            f"missing_chunk_ids={missing_chunk_ids[:10]}",
            status.HTTP_404_NOT_FOUND,
        )

    logger.info(
        "RAPTOR build started",
        extra={"document_id": document.id, "chunk_count": len(chunks)},
    )

    with transaction.atomic():
        current_items = _create_leaf_nodes(index, document, chunks, vectors)

    nodes_per_level = {"0": len(current_items)}
    clusters_per_level = {}
    summary_node_count = 0
    level = 0

    while len(current_items) > 1 and level + 1 < get_raptor_max_levels():
        clusters = cluster_embeddings(
            [item.vector for item in current_items],
            max_clusters=get_raptor_max_clusters(),
            soft_cluster_threshold=get_raptor_soft_cluster_threshold(),
            random_state=get_raptor_random_state(),
        )
        if not clusters:
            break
        next_level = level + 1
        logger.info(
            "RAPTOR level clustering completed",
            extra={
                "document_id": document.id,
                "level": next_level,
                "cluster_count": len(clusters),
            },
        )

        with transaction.atomic():
            next_items = _create_summary_level(
                index=index,
                document=document,
                current_items=current_items,
                clusters=clusters,
                level=next_level,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
            )

        if len(next_items) >= len(current_items):
            break
        total_nodes_so_far = RaptorNode.objects.filter(index=index).count()
        if total_nodes_so_far > get_raptor_max_nodes():
            raise RaptorError(
                "too_many_raptor_nodes",
                (
                    "RAPTOR build exceeded the configured node limit "
                    f"of {get_raptor_max_nodes()}."
                ),
            )

        clusters_per_level[str(next_level)] = len(clusters)
        nodes_per_level[str(next_level)] = len(next_items)
        summary_node_count += len(next_items)
        current_items = next_items
        level = next_level

    total_node_count = RaptorNode.objects.filter(index=index).count()
    metadata = {
        "level_count": len(nodes_per_level),
        "leaf_count": len(chunks),
        "summary_node_count": summary_node_count,
        "total_node_count": total_node_count,
        "nodes_per_level": nodes_per_level,
        "clusters_per_level": clusters_per_level,
        "metadata": {
            "build_version": RAPTOR_BUILD_VERSION,
            "clustering": "umap_gmm_bic_soft_clustering",
            "reduction": "umap",
            "random_state": get_raptor_random_state(),
            "provider": embedding_provider.provider_name,
            "embedding_model": embedding_provider.model,
            "embedding_dimension": embedding_provider.dimension,
        },
    }

    logger.info(
        "RAPTOR build completed",
        extra={
            "document_id": document.id,
            "level_count": metadata["level_count"],
            "total_node_count": total_node_count,
        },
    )
    return metadata


def _coerce_raptor_error(exc):
    if isinstance(exc, RaptorError):
        return exc
    if isinstance(exc, RAGError):
        return RaptorError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        )
    if isinstance(exc, EmbeddingProviderError):
        return RaptorError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        )
    if isinstance(exc, EmbeddingServiceError):
        return RaptorError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        )
    if isinstance(exc, TimeoutError) or exc.__class__.__name__.lower().endswith("timeout"):
        return RaptorError(
            "raptor_provider_timeout",
            "RAPTOR provider request timed out.",
            exc.__class__.__name__,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return RaptorError(
        "raptor_build_failed",
        "RAPTOR index could not be built safely.",
        exc.__class__.__name__,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def build_raptor_tree(document, *, embedding_provider=None, llm_provider=None):
    if not get_raptor_enabled():
        raise RaptorError(
            "raptor_disabled",
            "RAPTOR indexing is disabled.",
            response_status=status.HTTP_403_FORBIDDEN,
        )

    try:
        embedding_provider = embedding_provider or get_embedding_provider()
        llm_provider = llm_provider or get_llm_provider()
    except (EmbeddingServiceError, RAGError) as exc:
        raise _coerce_raptor_error(exc) from exc

    index = _prepare_index(document, embedding_provider)
    try:
        metadata = _build_tree(index, document, embedding_provider, llm_provider)
        completed_index = _complete_index(index, metadata)
        return _status_payload(document, completed_index)
    except Exception as exc:
        raptor_error = _coerce_raptor_error(exc)
        logger.warning(
            "RAPTOR build failed",
            extra={"document_id": document.id, "error_code": raptor_error.code},
        )
        _mark_failed(index, raptor_error)
        raise raptor_error from exc
