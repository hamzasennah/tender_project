import logging
import re
import unicodedata
from dataclasses import replace

from django.conf import settings
from django.db import DatabaseError
from pgvector.django import CosineDistance
from rest_framework import status

from extraction.models import ChunkEmbedding, RaptorIndex, RaptorNode, RaptorNodeChild
from extraction.services.embeddings import (
    EmbeddingServiceError,
    get_embedding_provider,
    validate_embedding_vector,
)
from extraction.services.raptor.intents import (
    detect_raptor_intents,
    get_raptor_max_subintents,
    get_raptor_multi_intent_enabled,
)
from extraction.services.raptor.tree_builder import RaptorError
from extraction.services.raptor.types import RaptorRetrievedNode

logger = logging.getLogger(__name__)

DEFAULT_RAPTOR_RETRIEVAL_TOP_K = 6
DEFAULT_RAPTOR_MAX_TOP_K = 20
DEFAULT_RAPTOR_MAX_QUESTION_CHARS = 1000
DEFAULT_RAPTOR_CHILD_EXPANSION_LIMIT = 2
DEFAULT_RAPTOR_FACTUAL_LEAF_CANDIDATES = 8
DEFAULT_RAPTOR_RETRIEVAL_STRATEGY = "top_down"
DEFAULT_RAPTOR_TOP_DOWN_BEAM_WIDTH = 3
DEFAULT_RAPTOR_TOP_DOWN_CHILDREN_PER_NODE = 4
DEFAULT_RAPTOR_TOP_DOWN_MAX_DEPTH = 4
MIN_DIVERSITY_SCORE = 0.05
DIVERSITY_RELATIVE_SCORE_FLOOR = 0.75
DIVERSITY_ABSOLUTE_SCORE_MARGIN = 0.25
CHILD_RELATIVE_SCORE_MARGIN = 0.25
FACTUAL_VALUE_BONUS = 0.09
FACTUAL_TOKEN_BONUS = 0.012
FACTUAL_MAX_TOKEN_BONUS = 0.072
FACTUAL_PROXIMITY_BONUS = 0.06
MIN_SUBINTENT_COVERAGE_SCORE = 0.35
MULTI_INTENT_MATCH_BONUS = 0.015

VALUE_PATTERNS = {
    "date": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+"
        r"(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|"
        r"octobre|novembre|decembre)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    "time": re.compile(r"\b\d{1,2}\s*(?:h|heure|heures)(?:\s*\d{2})?\b", re.IGNORECASE),
    "percentage": re.compile(r"\b\d+(?:[,.]\d+)?\s*%"),
    "amount": re.compile(
        r"\b(?:\d[\d\s.,]*(?:mad|dh|dhs|dirhams?|eur|euros?|\$)|"
        r"(?:mad|dh|dhs|eur|\$)\s*\d[\d\s.,]*)\b",
        re.IGNORECASE,
    ),
    "duration": re.compile(
        r"\b\d(?:[\d\s.,]*\d)?\s*(?:jours?|mois|ans?|annees?|heures?)\b",
        re.IGNORECASE,
    ),
}
QUERY_VALUE_TERMS = {
    "date": {"date", "delai", "limite", "depot", "echeance", "ouverture", "quand"},
    "time": {"heure", "horaire", "date", "limite", "ouverture"},
    "duration": {"duree", "dure", "validite", "validit", "periode", "delai", "jours"},
    "percentage": {"pourcentage", "taux", "pourcent", "montant", "garantie"},
    "amount": {"montant", "prix", "cout", "valeur", "garantie"},
}
TOKEN_STOP_WORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "la",
    "le",
    "les",
    "pour",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "quoi",
    "sont",
    "sur",
    "the",
    "what",
    "when",
    "where",
}


def _get_positive_int_setting(name, default):
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def get_raptor_retrieval_top_k():
    return _get_positive_int_setting(
        "RAPTOR_RETRIEVAL_TOP_K",
        DEFAULT_RAPTOR_RETRIEVAL_TOP_K,
    )


def get_raptor_max_top_k():
    return _get_positive_int_setting("RAPTOR_MAX_TOP_K", DEFAULT_RAPTOR_MAX_TOP_K)


def get_raptor_max_question_chars():
    return _get_positive_int_setting(
        "RAPTOR_MAX_QUESTION_CHARS",
        DEFAULT_RAPTOR_MAX_QUESTION_CHARS,
    )


def get_raptor_child_expansion_limit():
    return _get_positive_int_setting(
        "RAPTOR_CHILD_EXPANSION_LIMIT",
        DEFAULT_RAPTOR_CHILD_EXPANSION_LIMIT,
    )


def get_raptor_factual_leaf_candidates():
    return _get_positive_int_setting(
        "RAPTOR_FACTUAL_LEAF_CANDIDATES",
        DEFAULT_RAPTOR_FACTUAL_LEAF_CANDIDATES,
    )


def get_raptor_retrieval_strategy():
    value = str(
        getattr(
            settings,
            "RAPTOR_RETRIEVAL_STRATEGY",
            DEFAULT_RAPTOR_RETRIEVAL_STRATEGY,
        )
        or ""
    ).strip().lower()
    if value in {"flattened", "top_down"}:
        return value
    return DEFAULT_RAPTOR_RETRIEVAL_STRATEGY


def get_raptor_top_down_beam_width():
    return _get_positive_int_setting(
        "RAPTOR_TOP_DOWN_BEAM_WIDTH",
        DEFAULT_RAPTOR_TOP_DOWN_BEAM_WIDTH,
    )


def get_raptor_top_down_children_per_node():
    return _get_positive_int_setting(
        "RAPTOR_TOP_DOWN_CHILDREN_PER_NODE",
        DEFAULT_RAPTOR_TOP_DOWN_CHILDREN_PER_NODE,
    )


def get_raptor_top_down_max_depth():
    return _get_positive_int_setting(
        "RAPTOR_TOP_DOWN_MAX_DEPTH",
        DEFAULT_RAPTOR_TOP_DOWN_MAX_DEPTH,
    )


def _clean_question(question):
    normalized = str(question or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or ord(character) >= 32
    )
    return " ".join(normalized.split()).strip()


def _fold_text(value):
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    without_replacement = without_marks.replace("�", "")
    return without_replacement.lower()


def _tokens(value):
    return {
        token
        for token in re.findall(r"[0-9a-z]+", _fold_text(value))
        if len(token) >= 3 and token not in TOKEN_STOP_WORDS
    }


def _token_list(value):
    return [
        token
        for token in re.findall(r"[0-9a-z]+", _fold_text(value))
        if len(token) >= 3 and token not in TOKEN_STOP_WORDS
    ]


def _loose_token_match(left, right):
    if left == right:
        return True
    if len(left) >= 5 and len(right) >= 5:
        return left.startswith(right[:5]) or right.startswith(left[:5])
    return False


def _loose_token_overlap(query_tokens, text_tokens):
    overlap = set()
    for query_token in query_tokens:
        for text_token in text_tokens:
            if _loose_token_match(query_token, text_token):
                overlap.add(query_token)
                break
    return overlap


def _query_token_proximity_bonus(query_tokens, text):
    text_tokens = _token_list(text)
    if len(query_tokens) < 2 or len(text_tokens) < 2:
        return 0.0

    positions_by_query_token = {}
    for query_token in query_tokens:
        positions = [
            index
            for index, text_token in enumerate(text_tokens)
            if _loose_token_match(query_token, text_token)
        ]
        if positions:
            positions_by_query_token[query_token] = positions

    matched_query_tokens = list(positions_by_query_token)
    for left_index, left_token in enumerate(matched_query_tokens):
        for right_token in matched_query_tokens[left_index + 1 :]:
            if any(
                abs(left_position - right_position) <= 6
                for left_position in positions_by_query_token[left_token]
                for right_position in positions_by_query_token[right_token]
            ):
                return FACTUAL_PROXIMITY_BONUS
    return 0.0


def _query_value_intents(query):
    query_tokens = _tokens(query)
    intents = {
        value_type
        for value_type, terms in QUERY_VALUE_TERMS.items()
        if query_tokens & terms
    }
    return intents, query_tokens


def _value_types_in_text(text):
    folded_text = _fold_text(text)
    return {
        value_type
        for value_type, pattern in VALUE_PATTERNS.items()
        if pattern.search(folded_text)
    }


def _factual_leaf_score(query_intents, query_tokens, text):
    if not query_intents:
        return 0.0, {}

    text_tokens = _tokens(text)
    overlap = _loose_token_overlap(query_tokens, text_tokens)
    value_types = _value_types_in_text(text)
    matching_value_types = sorted(query_intents & value_types)
    if "date" in query_intents and "time" in value_types:
        matching_value_types.append("time")
    matching_value_types = sorted(set(matching_value_types))

    if not matching_value_types:
        return 0.0, {
            "matched_query_tokens": sorted(overlap),
            "matching_value_types": [],
        }

    token_bonus = min(len(overlap) * FACTUAL_TOKEN_BONUS, FACTUAL_MAX_TOKEN_BONUS)
    proximity_bonus = _query_token_proximity_bonus(query_tokens, text)
    value_bonus = min(len(matching_value_types), 2) * FACTUAL_VALUE_BONUS
    return token_bonus + proximity_bonus + value_bonus, {
        "matched_query_tokens": sorted(overlap),
        "matching_value_types": matching_value_types,
    }


def validate_raptor_question(question):
    normalized = _clean_question(question)
    if not normalized:
        raise RaptorError("empty_question", "Question must not be empty.")
    if len(normalized) > get_raptor_max_question_chars():
        raise RaptorError(
            "question_too_long",
            (
                "Question exceeds the configured RAPTOR maximum length of "
                f"{get_raptor_max_question_chars()} characters."
            ),
        )
    return normalized


def validate_raptor_top_k(top_k):
    if top_k in {None, ""}:
        return get_raptor_retrieval_top_k()
    try:
        value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise RaptorError("invalid_top_k", "top_k must be a positive integer.") from exc
    if value < 1 or value > get_raptor_max_top_k():
        raise RaptorError(
            "invalid_top_k",
            f"top_k must be between 1 and {get_raptor_max_top_k()}.",
        )
    return value


def _query_embedding(provider, question):
    try:
        vectors = provider.embed_texts([question])
    except EmbeddingServiceError as exc:
        raise RaptorError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        ) from exc

    if len(vectors) != 1:
        raise RaptorError(
            "malformed_query_embedding_response",
            "Embedding provider returned an invalid RAPTOR query embedding response.",
            response_status=status.HTTP_502_BAD_GATEWAY,
        )
    try:
        return validate_embedding_vector(vectors[0], provider.dimension)
    except EmbeddingServiceError as exc:
        raise RaptorError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        ) from exc


def _get_completed_index(document):
    try:
        index = RaptorIndex.objects.get(document=document)
    except RaptorIndex.DoesNotExist as exc:
        raise RaptorError(
            "raptor_index_not_found",
            "Completed RAPTOR index was not found for this document.",
            response_status=status.HTTP_404_NOT_FOUND,
        ) from exc

    if index.status != RaptorIndex.Status.COMPLETED:
        raise RaptorError(
            "raptor_index_not_ready",
            "RAPTOR index must be completed before asking RAPTOR questions.",
            f"status={index.status}",
            status.HTTP_409_CONFLICT,
        )
    return index


def _ensure_index_provider_matches(index, provider):
    if (
        index.provider != provider.provider_name
        or index.embedding_model != provider.model
        or index.embedding_dimension != provider.dimension
    ):
        raise RaptorError(
            "raptor_embedding_model_mismatch",
            "RAPTOR index embeddings do not match the configured embedding provider.",
            (
                f"index_provider={index.provider}; index_model={index.embedding_model}; "
                f"index_dimension={index.embedding_dimension}"
            ),
            status.HTTP_409_CONFLICT,
        )


def _candidate_limit_per_level(limit):
    return min(get_raptor_max_top_k(), max(limit * 2, limit + 2))


def _levels_available(index):
    try:
        return list(
            RaptorNode.objects.filter(index=index)
            .order_by("level")
            .values_list("level", flat=True)
            .distinct()
        )
    except DatabaseError as exc:
        raise RaptorError(
            "raptor_retrieval_failed",
            "RAPTOR retrieval could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


def _node_text(node):
    if node.level == 0 and node.source_chunk_id:
        return node.source_chunk.text
    return node.text


def _node_type(node):
    return "leaf" if node.level == 0 else "summary"


def _retrieved_node(node):
    distance = float(node.cosine_distance)
    chunk = node.source_chunk if node.source_chunk_id else None
    return RaptorRetrievedNode(
        node_id=node.id,
        level=node.level,
        node_index=node.node_index,
        node_type=_node_type(node),
        text=_node_text(node),
        cosine_distance=distance,
        similarity_score=1.0 - distance,
        chunk_id=chunk.id if chunk else None,
        chunk_index=chunk.chunk_index if chunk else None,
    )


def _level_candidates(index, level, query_vector, provider, limit):
    base_queryset = RaptorNode.objects.filter(
        index=index,
        document=index.document,
        level=level,
    ).select_related("source_chunk", "source_chunk__embedding")

    if level == 0:
        queryset = (
            base_queryset.filter(
                source_chunk__isnull=False,
                source_chunk__embedding__status=ChunkEmbedding.Status.COMPLETED,
                source_chunk__embedding__provider=provider.provider_name,
                source_chunk__embedding__model=provider.model,
                source_chunk__embedding__dimension=provider.dimension,
                source_chunk__embedding__embedding_vector__isnull=False,
            )
            .annotate(
                cosine_distance=CosineDistance(
                    "source_chunk__embedding__embedding_vector",
                    query_vector,
                )
            )
            .order_by("cosine_distance", "node_index", "id")[:limit]
        )
    else:
        queryset = (
            base_queryset.filter(embedding_vector__isnull=False)
            .annotate(cosine_distance=CosineDistance("embedding_vector", query_vector))
            .order_by("cosine_distance", "node_index", "id")[:limit]
        )

    try:
        return [_retrieved_node(node) for node in queryset]
    except DatabaseError as exc:
        raise RaptorError(
            "raptor_retrieval_failed",
            "RAPTOR retrieval could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


def _factual_leaf_candidates(index, question, query_vector, provider):
    query_intents, query_tokens = _query_value_intents(question)
    if not query_intents:
        return []

    queryset = (
        RaptorNode.objects.filter(
            index=index,
            document=index.document,
            level=0,
            source_chunk__isnull=False,
            source_chunk__embedding__status=ChunkEmbedding.Status.COMPLETED,
            source_chunk__embedding__provider=provider.provider_name,
            source_chunk__embedding__model=provider.model,
            source_chunk__embedding__dimension=provider.dimension,
            source_chunk__embedding__embedding_vector__isnull=False,
        )
        .select_related("source_chunk", "source_chunk__embedding")
        .annotate(
            cosine_distance=CosineDistance(
                "source_chunk__embedding__embedding_vector",
                query_vector,
            )
        )
    )

    try:
        nodes = list(queryset)
    except DatabaseError as exc:
        raise RaptorError(
            "raptor_retrieval_failed",
            "RAPTOR retrieval could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    candidates = []
    for node in nodes:
        factual_score, signals = _factual_leaf_score(
            query_intents,
            query_tokens,
            node.source_chunk.text,
        )
        if factual_score <= 0:
            continue
        result = _retrieved_node(node)
        candidates.append(
            replace(
                result,
                retrieval_score=result.similarity_score + factual_score,
                selection_reason="factual_leaf",
            )
        )

    candidates.sort(key=_result_sort_key)
    return candidates[: get_raptor_factual_leaf_candidates()]


def _merge_level_candidates(existing_candidates, additional_candidates):
    by_node_id = {candidate.node_id: candidate for candidate in existing_candidates}
    for candidate in additional_candidates:
        existing = by_node_id.get(candidate.node_id)
        if existing is None or _result_sort_key(candidate) < _result_sort_key(existing):
            by_node_id[candidate.node_id] = candidate
    merged = list(by_node_id.values())
    merged.sort(key=_result_sort_key)
    return merged


def _merge_candidate_into_levels(candidates_by_level, candidate):
    level_candidates = candidates_by_level.setdefault(candidate.level, [])
    candidates_by_level[candidate.level] = _merge_level_candidates(
        level_candidates,
        [candidate],
    )


def _selection_score(result):
    return (
        result.retrieval_score
        if result.retrieval_score is not None
        else result.similarity_score
    )


def _result_sort_key(result):
    selection_score = _selection_score(result)
    return (
        -selection_score,
        -result.similarity_score,
        result.cosine_distance,
        result.level,
        result.node_index,
        result.node_id,
    )


def _diversity_threshold(best_score):
    return max(
        MIN_DIVERSITY_SCORE,
        best_score * DIVERSITY_RELATIVE_SCORE_FLOOR,
        best_score - DIVERSITY_ABSOLUTE_SCORE_MARGIN,
    )


def _select_diverse_candidates(candidates_by_level, limit):
    all_candidates = [
        candidate
        for candidates in candidates_by_level.values()
        for candidate in candidates
    ]
    all_candidates.sort(key=_result_sort_key)
    if not all_candidates:
        return [], set(), []

    selected = []
    selected_ids = set()
    best_score = all_candidates[0].similarity_score
    threshold = _diversity_threshold(best_score)
    diversity_slots = min(len(candidates_by_level), max(1, (limit + 1) // 2))

    level_heads = []
    for level, candidates in candidates_by_level.items():
        if not candidates:
            continue
        level_heads.append(candidates[0])
    level_heads.sort(key=_result_sort_key)

    for candidate in level_heads:
        if len(selected) >= min(limit, diversity_slots):
            break
        if candidate.similarity_score < threshold:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.node_id)

    return selected, selected_ids, all_candidates


def _child_candidates_for_parent(
    child_ids,
    query_vector,
    provider,
    query_intents,
    query_tokens,
):
    if not child_ids:
        return []

    leaf_queryset = (
        RaptorNode.objects.filter(
            id__in=child_ids,
            level=0,
            source_chunk__isnull=False,
            source_chunk__embedding__status=ChunkEmbedding.Status.COMPLETED,
            source_chunk__embedding__provider=provider.provider_name,
            source_chunk__embedding__model=provider.model,
            source_chunk__embedding__dimension=provider.dimension,
            source_chunk__embedding__embedding_vector__isnull=False,
        )
        .select_related("source_chunk", "source_chunk__embedding")
        .annotate(
            cosine_distance=CosineDistance(
                "source_chunk__embedding__embedding_vector",
                query_vector,
            )
        )
    )
    summary_queryset = (
        RaptorNode.objects.filter(
            id__in=child_ids,
            level__gt=0,
            embedding_vector__isnull=False,
        )
        .select_related("source_chunk")
        .annotate(cosine_distance=CosineDistance("embedding_vector", query_vector))
    )

    try:
        nodes = list(leaf_queryset) + list(summary_queryset)
    except DatabaseError:
        logger.warning("RAPTOR child expansion scoring failed")
        return []

    candidates = []
    for node in nodes:
        result = _retrieved_node(node)
        if node.level == 0:
            factual_score, _signals = _factual_leaf_score(
                query_intents,
                query_tokens,
                node.source_chunk.text,
            )
            if factual_score > 0:
                result = replace(
                    result,
                    retrieval_score=result.similarity_score + factual_score,
                    selection_reason="parent_child_factual_leaf",
                )
        candidates.append(result)

    candidates.sort(key=_result_sort_key)
    return candidates


def _root_node_ids(index, levels):
    try:
        parented_child_ids = list(
            RaptorNodeChild.objects.filter(parent__index=index)
            .values_list("child_id", flat=True)
            .distinct()
        )
        roots = list(
            RaptorNode.objects.filter(index=index, document=index.document)
            .exclude(id__in=parented_child_ids)
            .order_by("-level", "node_index", "id")
            .values_list("id", flat=True)
        )
    except DatabaseError as exc:
        raise RaptorError(
            "raptor_retrieval_failed",
            "RAPTOR retrieval could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    if roots:
        return roots

    top_level = max(levels)
    try:
        return list(
            RaptorNode.objects.filter(
                index=index,
                document=index.document,
                level=top_level,
            )
            .order_by("node_index", "id")
            .values_list("id", flat=True)
        )
    except DatabaseError as exc:
        raise RaptorError(
            "raptor_retrieval_failed",
            "RAPTOR retrieval could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


def _score_node_ids(
    index,
    node_ids,
    query_vector,
    provider,
    query_intents,
    query_tokens,
    *,
    selection_reason,
):
    if not node_ids:
        return []

    leaf_queryset = (
        RaptorNode.objects.filter(
            index=index,
            document=index.document,
            id__in=node_ids,
            level=0,
            source_chunk__isnull=False,
            source_chunk__embedding__status=ChunkEmbedding.Status.COMPLETED,
            source_chunk__embedding__provider=provider.provider_name,
            source_chunk__embedding__model=provider.model,
            source_chunk__embedding__dimension=provider.dimension,
            source_chunk__embedding__embedding_vector__isnull=False,
        )
        .select_related("source_chunk", "source_chunk__embedding")
        .annotate(
            cosine_distance=CosineDistance(
                "source_chunk__embedding__embedding_vector",
                query_vector,
            )
        )
    )
    summary_queryset = (
        RaptorNode.objects.filter(
            index=index,
            document=index.document,
            id__in=node_ids,
            level__gt=0,
            embedding_vector__isnull=False,
        )
        .select_related("source_chunk")
        .annotate(cosine_distance=CosineDistance("embedding_vector", query_vector))
    )

    try:
        nodes = list(leaf_queryset) + list(summary_queryset)
    except DatabaseError as exc:
        raise RaptorError(
            "raptor_retrieval_failed",
            "RAPTOR retrieval could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    candidates = []
    for node in nodes:
        result = _retrieved_node(node)
        reason = selection_reason
        if node.level == 0:
            factual_score, _signals = _factual_leaf_score(
                query_intents,
                query_tokens,
                node.source_chunk.text,
            )
            if factual_score > 0:
                result = replace(
                    result,
                    retrieval_score=result.similarity_score + factual_score,
                )
                reason = f"{selection_reason}_factual_leaf"
        candidates.append(replace(result, selection_reason=reason))

    candidates.sort(key=_result_sort_key)
    return candidates


def _child_ids_by_parent(parent_ids):
    if not parent_ids:
        return {}
    try:
        links = list(
            RaptorNodeChild.objects.filter(parent_id__in=parent_ids)
            .order_by("parent_id", "child_rank", "child_id")
        )
    except DatabaseError as exc:
        raise RaptorError(
            "raptor_retrieval_failed",
            "RAPTOR retrieval could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    child_ids = {}
    for link in links:
        child_ids.setdefault(link.parent_id, []).append(link.child_id)
    return child_ids


def _retrieve_top_down_candidates(index, levels, question, query_vector, provider):
    query_intents, query_tokens = _query_value_intents(question)
    beam_width = get_raptor_top_down_beam_width()
    children_per_node = get_raptor_top_down_children_per_node()
    max_depth = get_raptor_top_down_max_depth()
    root_ids = _root_node_ids(index, levels)
    root_candidates = _score_node_ids(
        index,
        root_ids,
        query_vector,
        provider,
        query_intents,
        query_tokens,
        selection_reason="top_down_root",
    )
    candidates_by_level = {}
    selected_ids = set()
    nodes_examined = len(root_candidates)
    depth_reached = 0

    frontier = root_candidates[:beam_width]
    for candidate in frontier:
        selected_ids.add(candidate.node_id)
        _merge_candidate_into_levels(candidates_by_level, candidate)

    for depth in range(1, max_depth + 1):
        parent_ids = [
            candidate.node_id
            for candidate in frontier
            if candidate.node_type == "summary"
        ]
        if not parent_ids:
            break

        children_by_parent = _child_ids_by_parent(parent_ids)
        next_frontier = []
        for parent in frontier:
            child_ids = children_by_parent.get(parent.node_id, [])
            child_candidates = _score_node_ids(
                index,
                child_ids,
                query_vector,
                provider,
                query_intents,
                query_tokens,
                selection_reason="top_down_child",
            )
            nodes_examined += len(child_candidates)
            for child in child_candidates[:children_per_node]:
                child = replace(child, expanded_from_parent_id=parent.node_id)
                if child.node_id not in selected_ids:
                    selected_ids.add(child.node_id)
                    _merge_candidate_into_levels(candidates_by_level, child)
                next_frontier.append(child)

        if not next_frontier:
            break
        next_frontier.sort(key=_result_sort_key)
        frontier = next_frontier[:beam_width]
        depth_reached = depth

    metadata = {
        "root_count": len(root_ids),
        "nodes_examined": nodes_examined,
        "depth_reached": depth_reached,
        "beam_width": beam_width,
        "children_per_node": children_per_node,
        "max_depth": max_depth,
    }
    return candidates_by_level, metadata


def _expand_selected_children(
    index,
    selected,
    selected_ids,
    limit,
    query_vector,
    provider,
    question,
):
    child_limit = get_raptor_child_expansion_limit()
    if child_limit <= 0 or len(selected) >= limit:
        return

    summary_node_ids = [
        result.node_id
        for result in selected
        if result.node_type == "summary"
    ]
    if not summary_node_ids:
        return

    query_intents, query_tokens = _query_value_intents(question)
    child_links = (
        RaptorNodeChild.objects.filter(parent_id__in=summary_node_ids)
        .select_related("parent", "child")
        .order_by("parent_id", "child_rank", "child_id")
    )
    child_ids_by_parent = {}
    try:
        for link in child_links:
            child_ids_by_parent.setdefault(link.parent_id, []).append(link.child_id)
    except DatabaseError:
        logger.warning(
            "RAPTOR child expansion lookup failed",
            extra={"raptor_index_id": index.id},
        )
        return

    for parent in list(selected):
        if len(selected) >= limit:
            break
        child_candidates = _child_candidates_for_parent(
            child_ids_by_parent.get(parent.node_id, []),
            query_vector,
            provider,
            query_intents,
            query_tokens,
        )
        included_for_parent = 0
        for child in child_candidates:
            if child.node_id in selected_ids:
                continue
            if len(selected) >= limit or included_for_parent >= child_limit:
                break
            child_score = _selection_score(child)
            parent_score = _selection_score(parent)
            if child_score < parent_score - CHILD_RELATIVE_SCORE_MARGIN:
                continue
            selected.append(
                replace(child, expanded_from_parent_id=parent.node_id)
            )
            selected_ids.add(child.node_id)
            included_for_parent += 1


def _include_factual_leaf_candidates(selected, selected_ids, all_candidates, limit):
    for candidate in all_candidates:
        if len(selected) >= limit:
            break
        if candidate.node_id in selected_ids:
            continue
        if candidate.level != 0 or candidate.selection_reason != "factual_leaf":
            continue
        selected.append(candidate)
        selected_ids.add(candidate.node_id)


def _finalize_selection(
    index,
    candidates_by_level,
    limit,
    query_vector,
    provider,
    question,
    *,
    expand_children=True,
):
    selected, selected_ids, all_candidates = _select_diverse_candidates(
        candidates_by_level,
        limit,
    )
    _include_factual_leaf_candidates(
        selected,
        selected_ids,
        all_candidates,
        limit,
    )
    if expand_children:
        _expand_selected_children(
            index,
            selected,
            selected_ids,
            limit,
            query_vector,
            provider,
            question,
        )

    for candidate in all_candidates:
        if len(selected) >= limit:
            break
        if candidate.node_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.node_id)

    selected.sort(key=_result_sort_key)
    return selected[:limit]


def _selected_nodes_per_level(selected):
    counts = {}
    for result in selected:
        key = str(result.level)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _retrieve_single_intent(document, index, question, limit, provider, levels):
    candidate_limit = _candidate_limit_per_level(limit)
    query_vector = _query_embedding(provider, question)
    retrieval_strategy = get_raptor_retrieval_strategy()
    top_down_metadata = None
    if retrieval_strategy == "top_down":
        candidates_by_level, top_down_metadata = _retrieve_top_down_candidates(
            index,
            levels,
            question,
            query_vector,
            provider,
        )
    else:
        candidates_by_level = {
            level: _level_candidates(index, level, query_vector, provider, candidate_limit)
            for level in levels
        }

    if 0 in candidates_by_level:
        factual_candidates = _factual_leaf_candidates(
            index,
            question,
            query_vector,
            provider,
        )
        candidates_by_level[0] = _merge_level_candidates(
            candidates_by_level[0],
            factual_candidates,
        )
    else:
        factual_candidates = _factual_leaf_candidates(
            index,
            question,
            query_vector,
            provider,
        )
        if factual_candidates:
            candidates_by_level[0] = factual_candidates

    selected = _finalize_selection(
        index,
        candidates_by_level,
        limit,
        query_vector,
        provider,
        question,
        expand_children=(retrieval_strategy != "top_down"),
    )
    selected_counts = _selected_nodes_per_level(selected)

    return {
        "document_id": document.id,
        "index_id": index.id,
        "question": question,
        "query_length": len(question),
        "top_k": limit,
        "result_count": len(selected),
        "results": selected,
        "retrieval_metadata": {
            "retrieval_strategy": (
                "hierarchical_top_down_vector_retrieval"
                if retrieval_strategy == "top_down"
                else "hierarchical_vector_retrieval"
            ),
            "metric": "cosine",
            "ranking": (
                "top_down_beam_then_hierarchical_diversity"
                if retrieval_strategy == "top_down"
                else "per_level_pgvector_then_hierarchical_diversity"
            ),
            "base_retrieval_strategy": retrieval_strategy,
            "levels_available": levels,
            "levels_searched": [
                level
                for level, candidates in candidates_by_level.items()
                if candidates
            ],
            "candidate_count_per_level": {
                str(level): len(candidates)
                for level, candidates in candidates_by_level.items()
            },
            "selected_nodes_per_level": selected_counts,
            "candidate_limit_per_level": candidate_limit,
            "child_expansion_limit": get_raptor_child_expansion_limit(),
            "factual_leaf_candidates": get_raptor_factual_leaf_candidates(),
            "top_down": top_down_metadata,
            "provider": provider.provider_name,
            "model": provider.model,
            "dimension": provider.dimension,
        },
    }


def _multi_intent_record_sort_key(record):
    best = record["best_result"]
    match_bonus = max(0, len(record["matched_subintents"]) - 1) * MULTI_INTENT_MATCH_BONUS
    return (
        -(record["best_score"] + match_bonus),
        -best.similarity_score,
        best.cosine_distance,
        best.level,
        best.node_index,
        best.node_id,
    )


def _multi_intent_result_from_record(record):
    best = record["best_result"]
    matched_subintents = tuple(sorted(record["matched_subintents"]))
    match_bonus = max(0, len(matched_subintents) - 1) * MULTI_INTENT_MATCH_BONUS
    retrieval_score = max(
        _selection_score(best),
        record["best_score"] + match_bonus,
    )
    return replace(
        best,
        retrieval_score=retrieval_score,
        matched_subintents=matched_subintents,
    )


def _merge_subintent_records(subintent_payloads):
    records = {}
    for subintent_index, payload in enumerate(subintent_payloads, start=1):
        for result in payload.get("results", []):
            score = _selection_score(result)
            record = records.setdefault(
                result.node_id,
                {
                    "best_result": result,
                    "best_score": score,
                    "matched_subintents": set(),
                    "scores_by_subintent": {},
                },
            )
            previous_score = record["scores_by_subintent"].get(subintent_index)
            if previous_score is None or score > previous_score:
                record["scores_by_subintent"][subintent_index] = score
            if score >= MIN_SUBINTENT_COVERAGE_SCORE:
                record["matched_subintents"].add(subintent_index)
            if _result_sort_key(result) < _result_sort_key(record["best_result"]):
                record["best_result"] = result
            if score > record["best_score"]:
                record["best_score"] = score
    return records


def _select_multi_intent_results(records, subintent_count, limit):
    selected_records = []
    selected_ids = set()
    coverage = {
        f"subintent_{index}": False
        for index in range(1, subintent_count + 1)
    }

    for subintent_index in range(1, subintent_count + 1):
        eligible = [
            record
            for node_id, record in records.items()
            if record["scores_by_subintent"].get(subintent_index, 0.0)
            >= MIN_SUBINTENT_COVERAGE_SCORE
        ]
        if not eligible:
            continue
        eligible.sort(
            key=lambda record: (
                -record["scores_by_subintent"].get(subintent_index, 0.0),
                _multi_intent_record_sort_key(record),
            )
        )
        coverage[f"subintent_{subintent_index}"] = True
        if eligible[0]["best_result"].node_id in selected_ids:
            continue
        if len(selected_records) >= limit:
            coverage[f"subintent_{subintent_index}"] = False
            continue
        selected_records.append(eligible[0])
        selected_ids.add(eligible[0]["best_result"].node_id)

    remaining_records = sorted(records.values(), key=_multi_intent_record_sort_key)
    for record in remaining_records:
        if len(selected_records) >= limit:
            break
        node_id = record["best_result"].node_id
        if node_id in selected_ids:
            continue
        selected_records.append(record)
        selected_ids.add(node_id)

    selected = [
        _multi_intent_result_from_record(record)
        for record in selected_records[:limit]
    ]
    selected_node_ids = {result.node_id for result in selected}
    for subintent_index in range(1, subintent_count + 1):
        coverage[f"subintent_{subintent_index}"] = any(
            subintent_index in result.matched_subintents
            for result in selected
            if result.node_id in selected_node_ids
        )
    return selected, coverage


def _sum_candidate_counts(subintent_payloads):
    counts = {}
    for payload in subintent_payloads:
        metadata = payload.get("retrieval_metadata", {})
        for level, count in metadata.get("candidate_count_per_level", {}).items():
            counts[str(level)] = counts.get(str(level), 0) + count
    return counts


def _union_metadata_levels(subintent_payloads, key):
    levels = set()
    for payload in subintent_payloads:
        metadata = payload.get("retrieval_metadata", {})
        levels.update(metadata.get(key, []))
    return sorted(levels)


def _subintent_retrieval_metadata(intent_plan, subintent_payloads):
    return [
        {
            "subintent_index": index,
            "subintent": subintent,
            "result_count": payload.get("result_count", 0),
            "levels_searched": payload.get("retrieval_metadata", {}).get(
                "levels_searched",
                [],
            ),
            "selected_nodes_per_level": payload.get("retrieval_metadata", {}).get(
                "selected_nodes_per_level",
                {},
            ),
            "top_down": payload.get("retrieval_metadata", {}).get("top_down"),
        }
        for index, (subintent, payload) in enumerate(
            zip(intent_plan.subintents, subintent_payloads),
            start=1,
        )
    ]


def _sum_top_down_nodes_examined(subintent_payloads):
    total = 0
    found = False
    for payload in subintent_payloads:
        metadata = payload.get("retrieval_metadata", {}).get("top_down")
        if not metadata:
            continue
        found = True
        total += metadata.get("nodes_examined", 0)
    return total if found else None


def _retrieve_multi_intent(document, index, question, limit, provider, levels, intent_plan):
    subintent_payloads = [
        _retrieve_single_intent(
            document,
            index,
            subintent,
            limit,
            provider,
            levels,
        )
        for subintent in intent_plan.subintents
    ]
    records = _merge_subintent_records(subintent_payloads)
    selected, coverage = _select_multi_intent_results(
        records,
        len(intent_plan.subintents),
        limit,
    )
    retrieval_strategy = get_raptor_retrieval_strategy()
    top_down_nodes_examined = _sum_top_down_nodes_examined(subintent_payloads)

    return {
        "document_id": document.id,
        "index_id": index.id,
        "question": question,
        "query_length": len(question),
        "top_k": limit,
        "result_count": len(selected),
        "results": selected,
        "retrieval_metadata": {
            "retrieval_strategy": (
                "hierarchical_multi_intent_top_down_vector_retrieval"
                if retrieval_strategy == "top_down"
                else "hierarchical_multi_intent_vector_retrieval"
            ),
            "metric": "cosine",
            "ranking": "best_relevant_node_per_subintent_then_global_score",
            "base_retrieval_strategy": retrieval_strategy,
            "levels_available": levels,
            "levels_searched": _union_metadata_levels(
                subintent_payloads,
                "levels_searched",
            ),
            "candidate_count_per_level": _sum_candidate_counts(subintent_payloads),
            "selected_nodes_per_level": _selected_nodes_per_level(selected),
            "candidate_limit_per_level": _candidate_limit_per_level(limit),
            "child_expansion_limit": get_raptor_child_expansion_limit(),
            "factual_leaf_candidates": get_raptor_factual_leaf_candidates(),
            "top_down": (
                {"nodes_examined": top_down_nodes_examined}
                if top_down_nodes_examined is not None
                else None
            ),
            "provider": provider.provider_name,
            "model": provider.model,
            "dimension": provider.dimension,
            "multi_intent": True,
            "subintents": list(intent_plan.subintents),
            "subintent_count": len(intent_plan.subintents),
            "coverage": coverage,
            "intent_detection_strategy": intent_plan.strategy,
            "fusion_method": "best_relevant_node_per_subintent_then_global_score",
            "subintent_retrieval": _subintent_retrieval_metadata(
                intent_plan,
                subintent_payloads,
            ),
            "subintent_min_coverage_score": MIN_SUBINTENT_COVERAGE_SCORE,
        },
    }


def _provider_or_error(provider):
    try:
        return provider or get_embedding_provider()
    except EmbeddingServiceError as exc:
        raise RaptorError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        ) from exc


def retrieve_raptor_nodes(document, question, top_k=None, *, provider=None):
    normalized_question = validate_raptor_question(question)
    limit = validate_raptor_top_k(top_k)
    index = _get_completed_index(document)
    levels = _levels_available(index)
    if not levels:
        raise RaptorError(
            "raptor_index_empty",
            "Completed RAPTOR index does not contain any nodes.",
            response_status=status.HTTP_404_NOT_FOUND,
        )

    provider = _provider_or_error(provider)
    _ensure_index_provider_matches(index, provider)
    intent_plan = detect_raptor_intents(normalized_question)
    if intent_plan.multi_intent:
        return _retrieve_multi_intent(
            document,
            index,
            normalized_question,
            limit,
            provider,
            levels,
            intent_plan,
        )

    payload = _retrieve_single_intent(
        document,
        index,
        normalized_question,
        limit,
        provider,
        levels,
    )
    payload["retrieval_metadata"].update(
        {
            "multi_intent": False,
            "subintents": [normalized_question],
            "subintent_count": 1,
            "coverage": {"subintent_1": bool(payload["results"])},
            "intent_detection_strategy": intent_plan.strategy,
        }
    )
    return payload


def raptor_retrieval_limits_metadata():
    return {
        "default_top_k": get_raptor_retrieval_top_k(),
        "max_top_k": get_raptor_max_top_k(),
        "max_question_chars": get_raptor_max_question_chars(),
        "child_expansion_limit": get_raptor_child_expansion_limit(),
        "factual_leaf_candidates": get_raptor_factual_leaf_candidates(),
        "retrieval_strategy": get_raptor_retrieval_strategy(),
        "top_down_beam_width": get_raptor_top_down_beam_width(),
        "top_down_children_per_node": get_raptor_top_down_children_per_node(),
        "top_down_max_depth": get_raptor_top_down_max_depth(),
        "multi_intent_enabled": get_raptor_multi_intent_enabled(),
        "max_subintents": get_raptor_max_subintents(),
    }
