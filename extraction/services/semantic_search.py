import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from django.conf import settings
from django.db import DatabaseError, connection
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from pgvector.django import CosineDistance
from rest_framework import status

from extraction.models import ChunkEmbedding, TextChunk
from extraction.services.embeddings import (
    EmbeddingServiceError,
    get_embedding_dimension,
    get_embedding_model,
    get_embedding_provider,
    get_embedding_provider_name,
    validate_embedding_vector,
)

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_TOP_K = 5
DEFAULT_SEARCH_MAX_TOP_K = 20
DEFAULT_SEARCH_MAX_QUERY_CHARS = 1000
DEFAULT_HYBRID_SEARCH_ENABLED = True
DEFAULT_HYBRID_VECTOR_CANDIDATES = 20
DEFAULT_HYBRID_LEXICAL_CANDIDATES = 20
DEFAULT_HYBRID_RRF_K = 60
DEFAULT_HYBRID_LEXICAL_CONFIG = "french"
DEFAULT_HYBRID_QUERY_EXPANSION_ENABLED = True
DEFAULT_HYBRID_RERANK_ENABLED = True
DEFAULT_HYBRID_RERANK_EXACT_TERM_WEIGHT = 0.004
DEFAULT_HYBRID_RERANK_VALUE_PATTERN_WEIGHT = 0.018
DEFAULT_HYBRID_RERANK_PHRASE_WEIGHT = 0.012
DEFAULT_HYBRID_RERANK_STRUCTURE_WEIGHT = 0.006
DEFAULT_HYBRID_RERANK_CONCEPT_MATCH_WEIGHT = 0.024
DEFAULT_HYBRID_RERANK_CONCEPT_MISMATCH_WEIGHT = 0.014
DEFAULT_HYBRID_FUZZY_ENABLED = True
DEFAULT_HYBRID_FUZZY_CANDIDATES = 8
DEFAULT_HYBRID_FUZZY_SCAN_LIMIT = 200
DEFAULT_HYBRID_FUZZY_MIN_TERM_LENGTH = 5
DEFAULT_HYBRID_FUZZY_TERM_LIMIT = 8
DEFAULT_HYBRID_FUZZY_SCORE_THRESHOLD = 0.86
BASE_LEXICAL_QUERY_TERM_LIMIT = 16
EXPANDED_LEXICAL_QUERY_TERM_LIMIT = 24
LEXICAL_SEARCH_TERM_LIMIT = 32
FUZZY_CHUNK_TOKEN_LIMIT = 160
FALLBACK_LEXICAL_CONFIG = "simple"
LEXICAL_QUERY_STOP_WORDS = {
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
    "which",
}
QUERY_SYNONYMS = {
    "duree": ("periode", "delai"),
    "periode": ("duree", "delai"),
    "delai": ("duree", "periode"),
    "date": ("delai", "echeance", "remise", "depot"),
    "limite": ("delai", "echeance"),
    "echeance": ("date", "limite", "delai"),
    "remise": ("depot", "date", "limite"),
    "depot": ("remise", "date", "limite"),
    "garantie": ("caution", "offre"),
    "caution": ("garantie", "offre"),
    "numero": ("reference", "marche", "contrat"),
    "marche": ("reference", "numero", "contrat"),
    "reference": ("numero", "marche", "contrat"),
    "contrat": ("numero", "reference", "marche"),
    "montant": ("valeur", "prix", "cout"),
    "valeur": ("montant", "prix", "cout"),
    "prix": ("montant", "valeur", "cout"),
    "cout": ("montant", "valeur", "prix"),
    "pourcentage": ("taux", "pourcent"),
    "taux": ("pourcentage", "pourcent"),
    "exige": ("exigee", "requise", "requis", "obligatoire"),
    "exigee": ("exige", "requise", "requis", "obligatoire"),
    "requis": ("requise", "exige", "exigee", "obligatoire"),
    "requise": ("requis", "exige", "exigee", "obligatoire"),
    "obligatoire": ("exige", "exigee", "requis", "requise"),
    "offre": ("soumission",),
    "soumission": ("offre",),
}
QUERY_TERM_VARIANTS = {
    "duree": ("durée",),
    "periode": ("période",),
    "delai": ("délai",),
    "validite": ("validité",),
    "echeance": ("échéance",),
    "depot": ("dépôt",),
    "numero": ("numéro",),
    "reference": ("référence",),
    "cout": ("coût",),
    "pieces": ("pièces",),
    "quantite": ("quantité",),
    "exige": ("exigé",),
    "exigee": ("exigée",),
}
BUSINESS_CONCEPT_PATTERNS = {
    "bid_guarantee": (
        re.compile(
            r"\b(?:garantie|caution)\b.{0,80}\b(?:offres?|soumission)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:offres?|soumission)\b.{0,80}\b(?:garantie|caution)\b",
            re.IGNORECASE,
        ),
    ),
    "performance_guarantee": (
        re.compile(
            r"\bgarantie\b.{0,40}\b(?:bonne\s+execution|execution)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:bonne\s+execution|execution)\b.{0,40}\bgarantie\b",
            re.IGNORECASE,
        ),
    ),
    "technical_warranty": (
        re.compile(
            r"\bgarantie\b.{0,60}\b(?:fournitures?|technique|contractuelle)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:fournitures?|technique|contractuelle)\b.{0,60}\bgarantie\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bperiode\s+de\s+garantie\b", re.IGNORECASE),
    ),
    "insurance": (
        re.compile(r"\bassurance\b", re.IGNORECASE),
    ),
}
DATE_MONTHS_PATTERN = (
    "janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|"
    "septembre|octobre|novembre|decembre|décembre"
)
VALUE_PATTERNS = {
    "date": re.compile(
        rf"\b(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{1,2}}\s+(?:{DATE_MONTHS_PATTERN})\s+\d{{4}})\b",
        re.IGNORECASE,
    ),
    "time": re.compile(
        r"\b(?:\d{1,2}\s*(?:h|heure|heures)(?:\s*\d{2})?|\d{1,2}:\d{2})\b",
        re.IGNORECASE,
    ),
    "percentage": re.compile(r"\b\d+(?:[,.]\d+)?\s*%(?=\s|$|[.,;:])"),
    "amount": re.compile(
        r"\b(?:\d[\d\s.,]*(?:mad|dh|dhs|dirhams?|eur|euros?|€|\$)|(?:mad|dh|dhs|eur|€|\$)\s*\d[\d\s.,]*)\b",
        re.IGNORECASE,
    ),
    "market_reference": re.compile(
        r"\b[A-Z0-9]{2,}(?:[/-][A-Z0-9]{2,}){2,}\b",
        re.IGNORECASE,
    ),
    "duration": re.compile(
        r"\b\d+(?:[,.]\d+)?\s*(?:jours?|mois|ans?|annees?|années?|heures?)\b",
        re.IGNORECASE,
    ),
    "quantity": re.compile(
        r"\b\d+(?:[,.]\d+)?\s*(?:unites?|unités?|pages?|lots?|exemplaires?|pieces?|pièces?)\b",
        re.IGNORECASE,
    ),
}
STRUCTURED_MARKER_PATTERN = re.compile(
    r"\b(?:is|clause|article|section|numero|numéro|n[°o]|ref|réf|reference|référence)\s*[-.:/]?\s*[A-Z0-9]+(?:[./-][A-Z0-9]+)*\b",
    re.IGNORECASE,
)
TENDER_PHRASE_PATTERNS = [
    re.compile(r"\b(?:date|delai|délai)\s+limite\b.{0,80}\b(?:depot|dépôt)\b.{0,80}\boffres?\b", re.IGNORECASE),
    re.compile(r"\b(?:depot|dépôt)\b.{0,80}\boffres?\b.{0,80}\b(?:date|delai|délai)\s+limite\b", re.IGNORECASE),
]
VALUE_INTENT_TERMS = {
    "date": {"date", "delai", "deadline", "echeance", "limite", "depot", "ouverture"},
    "time": {"heure", "horaire", "delai", "date", "limite", "ouverture"},
    "percentage": {"pourcentage", "taux", "percent", "retenue"},
    "amount": {"montant", "budget", "prix", "caution", "garantie", "devise"},
    "market_reference": {"marche", "reference", "numero", "avis", "appel", "offres"},
    "duration": {"duree", "delai", "jours", "mois", "ans"},
    "quantity": {"quantite", "nombre", "lots", "exemplaires", "pieces"},
}


class SemanticSearchError(Exception):
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


@dataclass(frozen=True)
class SemanticSearchResult:
    chunk_id: int
    document_id: int
    extraction_result_id: int
    embedding_id: int
    chunk_index: int
    text: str
    character_start: int
    character_end: int
    text_length: int
    text_sha256: str
    chunk_metadata: dict
    cosine_distance: float
    similarity_score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    rerank_signals: dict | None = None


@dataclass(frozen=True)
class LexicalQueryPlan:
    base_terms: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    query_terms: tuple[str, ...]
    query_term_groups: tuple[tuple[str, ...], ...]
    expansion_enabled: bool


def _get_positive_int_setting(name, default):
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _get_bool_setting(name, default):
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_search_default_top_k():
    return _get_positive_int_setting("SEMANTIC_SEARCH_DEFAULT_TOP_K", DEFAULT_SEARCH_TOP_K)


def get_search_max_top_k():
    return _get_positive_int_setting("SEMANTIC_SEARCH_MAX_TOP_K", DEFAULT_SEARCH_MAX_TOP_K)


def get_search_max_query_chars():
    return _get_positive_int_setting(
        "SEMANTIC_SEARCH_MAX_QUERY_CHARS",
        DEFAULT_SEARCH_MAX_QUERY_CHARS,
    )


def get_hybrid_search_enabled():
    return _get_bool_setting("HYBRID_SEARCH_ENABLED", DEFAULT_HYBRID_SEARCH_ENABLED)


def get_hybrid_vector_candidates():
    return _get_positive_int_setting(
        "HYBRID_VECTOR_CANDIDATES",
        DEFAULT_HYBRID_VECTOR_CANDIDATES,
    )


def get_hybrid_lexical_candidates():
    return _get_positive_int_setting(
        "HYBRID_LEXICAL_CANDIDATES",
        DEFAULT_HYBRID_LEXICAL_CANDIDATES,
    )


def get_hybrid_rrf_k():
    return _get_positive_int_setting("HYBRID_RRF_K", DEFAULT_HYBRID_RRF_K)


def get_hybrid_lexical_config():
    return getattr(settings, "HYBRID_LEXICAL_CONFIG", DEFAULT_HYBRID_LEXICAL_CONFIG)


def get_hybrid_query_expansion_enabled():
    return _get_bool_setting(
        "HYBRID_QUERY_EXPANSION_ENABLED",
        DEFAULT_HYBRID_QUERY_EXPANSION_ENABLED,
    )


def get_hybrid_rerank_enabled():
    return _get_bool_setting("HYBRID_RERANK_ENABLED", DEFAULT_HYBRID_RERANK_ENABLED)


def _get_nonnegative_float_setting(name, default):
    try:
        value = float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(0.0, value)


def get_hybrid_rerank_exact_term_weight():
    return _get_nonnegative_float_setting(
        "HYBRID_RERANK_EXACT_TERM_WEIGHT",
        DEFAULT_HYBRID_RERANK_EXACT_TERM_WEIGHT,
    )


def get_hybrid_rerank_value_pattern_weight():
    return _get_nonnegative_float_setting(
        "HYBRID_RERANK_VALUE_PATTERN_WEIGHT",
        DEFAULT_HYBRID_RERANK_VALUE_PATTERN_WEIGHT,
    )


def get_hybrid_rerank_phrase_weight():
    return _get_nonnegative_float_setting(
        "HYBRID_RERANK_PHRASE_WEIGHT",
        DEFAULT_HYBRID_RERANK_PHRASE_WEIGHT,
    )


def get_hybrid_rerank_structure_weight():
    return _get_nonnegative_float_setting(
        "HYBRID_RERANK_STRUCTURE_WEIGHT",
        DEFAULT_HYBRID_RERANK_STRUCTURE_WEIGHT,
    )


def get_hybrid_rerank_concept_match_weight():
    return _get_nonnegative_float_setting(
        "HYBRID_RERANK_CONCEPT_MATCH_WEIGHT",
        DEFAULT_HYBRID_RERANK_CONCEPT_MATCH_WEIGHT,
    )


def get_hybrid_rerank_concept_mismatch_weight():
    return _get_nonnegative_float_setting(
        "HYBRID_RERANK_CONCEPT_MISMATCH_WEIGHT",
        DEFAULT_HYBRID_RERANK_CONCEPT_MISMATCH_WEIGHT,
    )


def get_hybrid_fuzzy_enabled():
    return _get_bool_setting("HYBRID_FUZZY_ENABLED", DEFAULT_HYBRID_FUZZY_ENABLED)


def get_hybrid_fuzzy_candidates():
    return _get_positive_int_setting(
        "HYBRID_FUZZY_CANDIDATES",
        DEFAULT_HYBRID_FUZZY_CANDIDATES,
    )


def get_hybrid_fuzzy_scan_limit():
    return _get_positive_int_setting(
        "HYBRID_FUZZY_SCAN_LIMIT",
        DEFAULT_HYBRID_FUZZY_SCAN_LIMIT,
    )


def get_hybrid_fuzzy_min_term_length():
    return _get_positive_int_setting(
        "HYBRID_FUZZY_MIN_TERM_LENGTH",
        DEFAULT_HYBRID_FUZZY_MIN_TERM_LENGTH,
    )


def get_hybrid_fuzzy_term_limit():
    return _get_positive_int_setting(
        "HYBRID_FUZZY_TERM_LIMIT",
        DEFAULT_HYBRID_FUZZY_TERM_LIMIT,
    )


def get_hybrid_fuzzy_score_threshold():
    value = _get_nonnegative_float_setting(
        "HYBRID_FUZZY_SCORE_THRESHOLD",
        DEFAULT_HYBRID_FUZZY_SCORE_THRESHOLD,
    )
    return min(value, 1.0)


def clean_query_text(query):
    normalized = str(query or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or ord(character) >= 32
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def validate_search_query(query):
    normalized = clean_query_text(query)
    if not normalized:
        raise SemanticSearchError(
            "empty_query",
            "Search query must not be empty.",
        )
    if len(normalized) > get_search_max_query_chars():
        raise SemanticSearchError(
            "query_too_long",
            (
                "Search query exceeds the configured maximum length of "
                f"{get_search_max_query_chars()} characters."
            ),
        )
    return normalized


def validate_top_k(top_k):
    if top_k in {None, ""}:
        return get_search_default_top_k()
    try:
        value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise SemanticSearchError(
            "invalid_top_k",
            "top_k must be a positive integer.",
        ) from exc
    if value < 1 or value > get_search_max_top_k():
        raise SemanticSearchError(
            "invalid_top_k",
            f"top_k must be between 1 and {get_search_max_top_k()}.",
        )
    return value


def _query_embedding(provider, query):
    try:
        vectors = provider.embed_texts([query])
    except EmbeddingServiceError as exc:
        raise SemanticSearchError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        ) from exc

    if len(vectors) != 1:
        raise SemanticSearchError(
            "malformed_query_embedding_response",
            "Embedding provider returned an invalid query embedding response.",
            response_status=status.HTTP_502_BAD_GATEWAY,
        )

    try:
        return validate_embedding_vector(vectors[0], provider.dimension)
    except EmbeddingServiceError as exc:
        raise SemanticSearchError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        ) from exc


def _provider_metadata(provider):
    if provider is not None:
        return provider.provider_name, provider.model, provider.dimension
    return get_embedding_provider_name(), get_embedding_model(), get_embedding_dimension()


def _get_search_provider(provider):
    if provider is not None:
        return provider
    try:
        return get_embedding_provider()
    except EmbeddingServiceError as exc:
        raise SemanticSearchError(
            exc.code,
            exc.public_message,
            exc.internal_detail,
            exc.response_status,
        ) from exc


def _search_limits_metadata():
    return {
        "default_top_k": get_search_default_top_k(),
        "max_top_k": get_search_max_top_k(),
        "max_query_chars": get_search_max_query_chars(),
        "hybrid_search_enabled": get_hybrid_search_enabled(),
        "hybrid_vector_candidates": get_hybrid_vector_candidates(),
        "hybrid_lexical_candidates": get_hybrid_lexical_candidates(),
        "hybrid_rrf_k": get_hybrid_rrf_k(),
        "hybrid_lexical_config": get_hybrid_lexical_config(),
        "hybrid_query_expansion_enabled": get_hybrid_query_expansion_enabled(),
        "hybrid_rerank_enabled": get_hybrid_rerank_enabled(),
        "hybrid_rerank_exact_term_weight": get_hybrid_rerank_exact_term_weight(),
        "hybrid_rerank_value_pattern_weight": get_hybrid_rerank_value_pattern_weight(),
        "hybrid_rerank_phrase_weight": get_hybrid_rerank_phrase_weight(),
        "hybrid_rerank_structure_weight": get_hybrid_rerank_structure_weight(),
        "hybrid_rerank_concept_match_weight": get_hybrid_rerank_concept_match_weight(),
        "hybrid_rerank_concept_mismatch_weight": get_hybrid_rerank_concept_mismatch_weight(),
        "hybrid_fuzzy_enabled": get_hybrid_fuzzy_enabled(),
        "hybrid_fuzzy_candidates": get_hybrid_fuzzy_candidates(),
        "hybrid_fuzzy_scan_limit": get_hybrid_fuzzy_scan_limit(),
        "hybrid_fuzzy_min_term_length": get_hybrid_fuzzy_min_term_length(),
        "hybrid_fuzzy_term_limit": get_hybrid_fuzzy_term_limit(),
        "hybrid_fuzzy_score_threshold": get_hybrid_fuzzy_score_threshold(),
    }


def _result_payload(result):
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "extraction_result_id": result.extraction_result_id,
        "embedding_id": result.embedding_id,
        "chunk_index": result.chunk_index,
        "text": result.text,
        "character_start": result.character_start,
        "character_end": result.character_end,
        "text_length": result.text_length,
        "text_sha256": result.text_sha256,
        "chunk_metadata": result.chunk_metadata,
        "cosine_distance": result.cosine_distance,
        "similarity_score": result.similarity_score,
        "vector_rank": result.vector_rank,
        "lexical_rank": result.lexical_rank,
        "lexical_score": result.lexical_score,
        "hybrid_score": result.hybrid_score,
        "rerank_score": result.rerank_score,
        "final_score": result.final_score,
        "rerank_signals": result.rerank_signals,
    }


def _current_embedding_queryset(document, provider_name, model, dimension):
    return ChunkEmbedding.objects.filter(
        chunk__document=document,
        chunk__document__owner=document.owner,
        status=ChunkEmbedding.Status.COMPLETED,
        provider=provider_name,
        model=model,
        dimension=dimension,
        embedding_vector__isnull=False,
    )


def _ensure_document_has_chunks(document):
    try:
        chunks_exist = TextChunk.objects.filter(document=document).exists()
    except DatabaseError as exc:
        logger.warning(
            "Semantic search chunk existence query failed",
            extra={"document_id": document.id, "error_code": "semantic_search_failed"},
        )
        raise SemanticSearchError(
            "semantic_search_failed",
            "Semantic search could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
    if not chunks_exist:
        raise SemanticSearchError(
            "chunks_not_found",
            "Persisted text chunks were not found for this document.",
            response_status=status.HTTP_404_NOT_FOUND,
        )


def _ensure_embeddings_exist(base_queryset, document):
    try:
        embeddings_exist = base_queryset.exists()
    except DatabaseError as exc:
        logger.warning(
            "Semantic search embedding existence query failed",
            extra={"document_id": document.id, "error_code": "semantic_search_failed"},
        )
        raise SemanticSearchError(
            "semantic_search_failed",
            "Semantic search could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
    if not embeddings_exist:
        raise SemanticSearchError(
            "embeddings_not_found",
            "Completed chunk embeddings were not found for this document.",
            response_status=status.HTTP_404_NOT_FOUND,
        )


def _text_search_config_available(config):
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "select exists(select 1 from pg_catalog.pg_ts_config where cfgname = %s)",
                [config],
            )
            return bool(cursor.fetchone()[0])
    except DatabaseError:
        logger.warning(
            "PostgreSQL text search config lookup failed",
            extra={"text_search_config": config},
        )
        return False


def _resolve_lexical_config():
    configured = get_hybrid_lexical_config() or FALLBACK_LEXICAL_CONFIG
    if _text_search_config_available(configured):
        return configured
    return FALLBACK_LEXICAL_CONFIG


def _fold_text(value):
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_marks.lower()


def _tokenize_folded(value):
    return re.findall(r"[0-9a-z]+", _fold_text(value))


def _fuzzy_query_terms(query_plan):
    terms = []
    seen = set()
    min_length = get_hybrid_fuzzy_min_term_length()
    for term in query_plan.base_terms:
        if len(term) < min_length or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= get_hybrid_fuzzy_term_limit():
            break
    return tuple(terms)


def _fuzzy_term_similarity(query_term, candidate_term):
    if query_term == candidate_term:
        return 1.0
    if not query_term or not candidate_term:
        return 0.0
    if abs(len(query_term) - len(candidate_term)) > max(2, len(query_term) // 3):
        return 0.0
    return SequenceMatcher(None, query_term, candidate_term).ratio()


def _append_query_term(terms, seen, term):
    normalized = _fold_text(term).strip()
    if len(normalized) < 2 or normalized in LEXICAL_QUERY_STOP_WORDS:
        return False
    if normalized in seen:
        return False
    seen.add(normalized)
    terms.append(normalized)
    return True


def _base_lexical_query_terms(query):
    tokens = _tokenize_folded(query)
    terms = []
    seen = set()
    for token in tokens:
        _append_query_term(terms, seen, token)
        if len(terms) >= BASE_LEXICAL_QUERY_TERM_LIMIT:
            break
    return tuple(terms)


def _term_search_variants(term):
    variants = [term]
    for variant in QUERY_TERM_VARIANTS.get(term, ()):
        normalized_variant = str(variant or "").strip().lower()
        if not normalized_variant:
            continue
        if normalized_variant not in variants:
            variants.append(normalized_variant)
    return tuple(variants)


def _search_groups_from_canonical_groups(canonical_groups):
    search_groups = []
    seen = set()

    for canonical_group in canonical_groups:
        search_group = []
        for term in canonical_group:
            for variant in _term_search_variants(term):
                if variant in search_group:
                    continue
                if variant not in seen and len(seen) >= LEXICAL_SEARCH_TERM_LIMIT:
                    continue
                search_group.append(variant)
                seen.add(variant)
        if not search_group and canonical_group:
            search_group.append(canonical_group[0])
        search_groups.append(tuple(search_group))

    return tuple(search_groups)


def _expanded_lexical_query_terms_and_groups(base_terms):
    terms = list(base_terms)
    expanded_terms = []
    seen = set(base_terms)
    canonical_groups = []

    for term in base_terms:
        group = [term]
        for synonym in QUERY_SYNONYMS.get(term, ()):
            normalized = _fold_text(synonym).strip()
            if len(normalized) < 2 or normalized in LEXICAL_QUERY_STOP_WORDS:
                continue
            if normalized not in seen and len(terms) >= EXPANDED_LEXICAL_QUERY_TERM_LIMIT:
                continue
            if normalized not in group:
                group.append(normalized)
            if normalized in seen:
                continue
            if _append_query_term(terms, seen, normalized):
                expanded_terms.append(normalized)
        canonical_groups.append(tuple(group))

    return tuple(expanded_terms), tuple(terms), _search_groups_from_canonical_groups(
        canonical_groups
    )


def _lexical_query_plan(query):
    base_terms = _base_lexical_query_terms(query)
    if not get_hybrid_query_expansion_enabled():
        return LexicalQueryPlan(
            base_terms=base_terms,
            expanded_terms=(),
            query_terms=base_terms,
            query_term_groups=_search_groups_from_canonical_groups(
                tuple((term,) for term in base_terms)
            ),
            expansion_enabled=False,
        )

    expanded_terms, query_terms, query_term_groups = _expanded_lexical_query_terms_and_groups(
        base_terms
    )
    return LexicalQueryPlan(
        base_terms=base_terms,
        expanded_terms=expanded_terms,
        query_terms=query_terms,
        query_term_groups=query_term_groups,
        expansion_enabled=True,
    )


def _query_value_intents(query_terms):
    terms = set(query_terms)
    return {
        value_type
        for value_type, intent_terms in VALUE_INTENT_TERMS.items()
        if terms & intent_terms
    }


def _max_query_terms_in_window(chunk_terms, query_terms, window_size=12):
    query_term_set = set(query_terms)
    if not chunk_terms or not query_term_set:
        return 0
    best = 0
    for start in range(len(chunk_terms)):
        window_terms = set(chunk_terms[start : start + window_size])
        best = max(best, len(window_terms & query_term_set))
    return best


def _value_pattern_signals(text):
    return {
        value_type: bool(pattern.search(text))
        for value_type, pattern in VALUE_PATTERNS.items()
    }


def _business_concept_text(value):
    folded = _fold_text(value)
    folded = re.sub(r"[’‘`´]", "'", folded)
    folded = re.sub(r"\s+", " ", folded)
    return folded.strip()


def _business_concepts_in_text(value):
    concept_text = _business_concept_text(value)
    return tuple(
        concept
        for concept, patterns in BUSINESS_CONCEPT_PATTERNS.items()
        if any(pattern.search(concept_text) for pattern in patterns)
    )


def _expected_business_concept(query_terms, query_text):
    concepts = _business_concepts_in_text(query_text)
    if len(concepts) == 1:
        return concepts[0]
    if len(concepts) > 1:
        return None

    terms = set(query_terms)
    if {"garantie", "offre"} <= terms or {"caution", "offre"} <= terms:
        return "bid_guarantee"
    if "garantie" in terms and ("execution" in terms or "bonne" in terms):
        return "performance_guarantee"
    if "assurance" in terms:
        return "insurance"
    if "garantie" in terms and (
        "fournitures" in terms
        or "fourniture" in terms
        or "technique" in terms
        or "periode" in terms
    ):
        return "technical_warranty"
    return None


def _phrase_match(text, chunk_terms, query_terms):
    if any(pattern.search(text) for pattern in TENDER_PHRASE_PATTERNS):
        return True
    if not query_terms:
        return False
    close_terms = _max_query_terms_in_window(
        chunk_terms,
        query_terms,
        window_size=10,
    )
    return close_terms >= min(3, len(query_terms))


def _rerank_signals(query_terms, text, query_text=""):
    folded_text = _fold_text(text)
    chunk_terms = _tokenize_folded(text)
    query_term_set = set(query_terms)
    matched_terms = sorted(set(chunk_terms) & query_term_set)
    close_terms = _max_query_terms_in_window(chunk_terms, query_terms)
    value_patterns = _value_pattern_signals(text)
    expected_concept = _expected_business_concept(query_terms, query_text)
    candidate_concepts = _business_concepts_in_text(text)
    concept_match = bool(expected_concept and expected_concept in candidate_concepts)
    concept_mismatch = bool(
        expected_concept
        and candidate_concepts
        and expected_concept not in candidate_concepts
    )
    value_types = sorted(
        value_type for value_type, is_present in value_patterns.items() if is_present
    )
    intents = _query_value_intents(query_terms)
    matching_value_types = sorted(
        value_type
        for value_type in value_types
        if value_type in intents
        or (
            value_type == "time"
            and "date" in intents
        )
    )
    if concept_mismatch:
        matching_value_types = [
            value_type
            for value_type in matching_value_types
            if value_type != "percentage"
        ]
    if not matching_value_types and value_types and not concept_mismatch:
        matching_value_types = value_types[:1]

    phrase_match = _phrase_match(folded_text, chunk_terms, query_terms)
    structured_marker = bool(STRUCTURED_MARKER_PATTERN.search(text))
    value_bonus_count = min(len(matching_value_types), 2)

    term_bonus = len(matched_terms) * get_hybrid_rerank_exact_term_weight()
    value_bonus = value_bonus_count * get_hybrid_rerank_value_pattern_weight()
    phrase_bonus = get_hybrid_rerank_phrase_weight() if phrase_match else 0.0
    structure_bonus = (
        get_hybrid_rerank_structure_weight() if structured_marker else 0.0
    )
    concept_match_bonus = (
        get_hybrid_rerank_concept_match_weight() if concept_match else 0.0
    )
    concept_mismatch_penalty = (
        get_hybrid_rerank_concept_mismatch_weight() if concept_mismatch else 0.0
    )
    rerank_score = (
        term_bonus
        + value_bonus
        + phrase_bonus
        + structure_bonus
        + concept_match_bonus
        - concept_mismatch_penalty
    )

    return rerank_score, {
        "matched_terms": len(matched_terms),
        "matched_query_terms": matched_terms,
        "close_matched_terms": close_terms,
        "contains_date": value_patterns["date"],
        "contains_time": value_patterns["time"],
        "contains_percentage": value_patterns["percentage"],
        "contains_amount": value_patterns["amount"],
        "contains_market_reference": value_patterns["market_reference"],
        "contains_duration": value_patterns["duration"],
        "contains_quantity": value_patterns["quantity"],
        "matching_value_types": matching_value_types,
        "phrase_match": phrase_match,
        "structured_marker": structured_marker,
        "expected_concept": expected_concept,
        "candidate_concept": candidate_concepts[0] if candidate_concepts else None,
        "candidate_concepts": list(candidate_concepts),
        "concept_match": concept_match,
        "concept_mismatch": concept_mismatch,
    }


def _apply_deterministic_reranking(results, normalized_query):
    reranked_results = []
    if not get_hybrid_rerank_enabled():
        for result in results:
            reranked_results.append(
                replace(
                    result,
                    rerank_score=0.0,
                    final_score=result.hybrid_score,
                    rerank_signals={},
                )
            )
        return reranked_results

    query_terms = _base_lexical_query_terms(normalized_query)
    for result in results:
        rerank_score, signals = _rerank_signals(
            query_terms,
            result.text,
            normalized_query,
        )
        reranked_results.append(
            replace(
                result,
                rerank_score=rerank_score,
                final_score=(result.hybrid_score or 0.0) + rerank_score,
                rerank_signals=signals,
            )
        )
    return reranked_results


def _search_query_from_terms(terms, normalized_query, config):
    if not terms:
        return SearchQuery(normalized_query, config=config, search_type="websearch")

    query = SearchQuery(terms[0], config=config, search_type="plain")
    for term in terms[1:]:
        query = query | SearchQuery(term, config=config, search_type="plain")
    return query


def _lexical_search_query(normalized_query, config, query_plan):
    if not query_plan.query_term_groups:
        return SearchQuery(normalized_query, config=config, search_type="websearch")

    query = _search_query_from_terms(
        query_plan.query_term_groups[0],
        normalized_query,
        config,
    )
    for term_group in query_plan.query_term_groups[1:]:
        query = query & _search_query_from_terms(term_group, normalized_query, config)
    return query


def _lexical_candidates(document, normalized_query, provider, limit):
    config = _resolve_lexical_config()
    query_plan = _lexical_query_plan(normalized_query)
    vector = SearchVector("text", config=config)
    query = _lexical_search_query(normalized_query, config, query_plan)
    base_query = _search_query_from_terms(
        query_plan.base_terms,
        normalized_query,
        config,
    )

    queryset = (
        TextChunk.objects.filter(
            document=document,
            document__owner=document.owner,
            embedding__status=ChunkEmbedding.Status.COMPLETED,
            embedding__provider=provider.provider_name,
            embedding__model=provider.model,
            embedding__dimension=provider.dimension,
            embedding__embedding_vector__isnull=False,
        )
        .annotate(search_vector=vector)
        .annotate(lexical_score=SearchRank("search_vector", query, cover_density=True))
        .annotate(
            base_lexical_score=SearchRank("search_vector", base_query, cover_density=True)
        )
        .filter(search_vector=query)
        .order_by("-base_lexical_score", "-lexical_score", "chunk_index", "id")[:limit]
    )

    try:
        chunks = list(queryset)
    except DatabaseError as exc:
        logger.warning(
            "Semantic lexical search failed",
            extra={"document_id": document.id, "error_code": "semantic_search_failed"},
        )
        raise SemanticSearchError(
            "semantic_search_failed",
            "Semantic search could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    return chunks, config, query_plan


def _best_fuzzy_chunk_score(query_terms, chunk_text):
    chunk_terms = _tokenize_folded(chunk_text)[:FUZZY_CHUNK_TOKEN_LIMIT]
    if not query_terms or not chunk_terms:
        return 0.0, ()

    matched_terms = []
    scores = []
    threshold = get_hybrid_fuzzy_score_threshold()
    for query_term in query_terms:
        best_score = max(
            _fuzzy_term_similarity(query_term, chunk_term)
            for chunk_term in chunk_terms
        )
        if best_score >= threshold:
            matched_terms.append(query_term)
            scores.append(best_score)

    if not scores:
        return 0.0, ()
    return sum(scores) / len(query_terms), tuple(matched_terms)


def _fuzzy_candidates(document, query_plan, provider, existing_chunk_ids):
    metadata = {
        "enabled": get_hybrid_fuzzy_enabled(),
        "library": "python_stdlib_difflib",
        "score_threshold": get_hybrid_fuzzy_score_threshold(),
        "min_term_length": get_hybrid_fuzzy_min_term_length(),
        "term_limit": get_hybrid_fuzzy_term_limit(),
        "candidate_limit": get_hybrid_fuzzy_candidates(),
        "scan_limit": get_hybrid_fuzzy_scan_limit(),
        "chunk_token_limit": FUZZY_CHUNK_TOKEN_LIMIT,
        "query_terms": [],
        "scanned_chunk_count": 0,
        "candidate_count": 0,
    }
    if not get_hybrid_fuzzy_enabled():
        return [], {}, {}, metadata

    query_terms = _fuzzy_query_terms(query_plan)
    metadata["query_terms"] = list(query_terms)
    if not query_terms:
        return [], {}, {}, metadata

    queryset = (
        TextChunk.objects.filter(
            document=document,
            document__owner=document.owner,
            embedding__status=ChunkEmbedding.Status.COMPLETED,
            embedding__provider=provider.provider_name,
            embedding__model=provider.model,
            embedding__dimension=provider.dimension,
            embedding__embedding_vector__isnull=False,
        )
        .exclude(id__in=existing_chunk_ids)
        .order_by("chunk_index", "id")[: get_hybrid_fuzzy_scan_limit()]
    )

    try:
        scanned_chunks = list(queryset)
    except DatabaseError as exc:
        logger.warning(
            "Semantic fuzzy candidate scan failed",
            extra={"document_id": document.id, "error_code": "semantic_search_failed"},
        )
        raise SemanticSearchError(
            "semantic_search_failed",
            "Semantic search could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    metadata["scanned_chunk_count"] = len(scanned_chunks)
    scored_chunks = []
    score_by_chunk_id = {}
    signals_by_chunk_id = {}
    for chunk in scanned_chunks:
        score, matched_terms = _best_fuzzy_chunk_score(query_terms, chunk.text)
        if not score:
            continue
        scored_chunks.append((score, -len(matched_terms), chunk.chunk_index, chunk.id, chunk))
        score_by_chunk_id[chunk.id] = score
        signals_by_chunk_id[chunk.id] = {
            "matched_terms": list(matched_terms),
            "score": score,
        }

    scored_chunks.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    selected = [
        item[-1]
        for item in scored_chunks[: get_hybrid_fuzzy_candidates()]
    ]
    metadata["candidate_count"] = len(selected)
    metadata["candidate_chunk_indexes"] = [chunk.chunk_index for chunk in selected]
    return selected, score_by_chunk_id, signals_by_chunk_id, metadata


def _rrf_score(vector_rank, lexical_rank, rrf_k):
    score = 0.0
    if vector_rank is not None:
        score += 1.0 / (rrf_k + vector_rank)
    if lexical_rank is not None:
        score += 1.0 / (rrf_k + lexical_rank)
    return score


def _result_from_embedding(
    embedding,
    *,
    vector_rank=None,
    lexical_rank=None,
    lexical_score=None,
    hybrid_score=None,
):
    distance = float(embedding.cosine_distance)
    chunk = embedding.chunk
    return SemanticSearchResult(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        extraction_result_id=chunk.extraction_result_id,
        embedding_id=embedding.id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        character_start=chunk.character_start,
        character_end=chunk.character_end,
        text_length=chunk.text_length,
        text_sha256=chunk.text_sha256,
        chunk_metadata=chunk.metadata,
        cosine_distance=distance,
        similarity_score=1.0 - distance,
        vector_rank=vector_rank,
        lexical_rank=lexical_rank,
        lexical_score=lexical_score,
        hybrid_score=hybrid_score,
    )


def _exact_vector_search_payload(document, normalized_query, limit, provider, base_queryset):
    query_vector = _query_embedding(provider, normalized_query)

    try:
        embeddings = list(
            base_queryset.select_related(
                "chunk",
                "chunk__document",
                "chunk__extraction_result",
            )
            .annotate(cosine_distance=CosineDistance("embedding_vector", query_vector))
            .order_by("cosine_distance", "chunk__chunk_index", "id")[:limit]
        )
    except DatabaseError as exc:
        logger.warning(
            "Semantic search database query failed",
            extra={"document_id": document.id, "error_code": "semantic_search_failed"},
        )
        raise SemanticSearchError(
            "semantic_search_failed",
            "Semantic search could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    results = [
        _result_from_embedding(embedding, vector_rank=index, hybrid_score=None)
        for index, embedding in enumerate(embeddings, start=1)
    ]

    return {
        "document_id": document.id,
        "query_length": len(normalized_query),
        "top_k": limit,
        "result_count": len(results),
        "results": [_result_payload(result) for result in results],
        "search_metadata": {
            "provider": provider.provider_name,
            "model": provider.model,
            "dimension": provider.dimension,
            "metric": "cosine",
            "ranking": "ascending_cosine_distance",
            "similarity_score": "1 - cosine_distance",
            "search_type": "exact_pgvector",
            "limits": _search_limits_metadata(),
        },
    }


def _hybrid_search_payload(document, normalized_query, limit, provider, base_queryset):
    query_vector = _query_embedding(provider, normalized_query)
    vector_limit = get_hybrid_vector_candidates()
    lexical_limit = get_hybrid_lexical_candidates()
    rrf_k = get_hybrid_rrf_k()

    try:
        vector_embeddings = list(
            base_queryset.select_related("chunk")
            .annotate(cosine_distance=CosineDistance("embedding_vector", query_vector))
            .order_by("cosine_distance", "chunk__chunk_index", "id")[:vector_limit]
        )
    except DatabaseError as exc:
        logger.warning(
            "Semantic vector candidate query failed",
            extra={"document_id": document.id, "error_code": "semantic_search_failed"},
        )
        raise SemanticSearchError(
            "semantic_search_failed",
            "Semantic search could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    lexical_chunks, lexical_config, lexical_query_plan = _lexical_candidates(
        document,
        normalized_query,
        provider,
        lexical_limit,
    )
    vector_rank_by_chunk_id = {
        embedding.chunk_id: index
        for index, embedding in enumerate(vector_embeddings, start=1)
    }
    lexical_rank_by_chunk_id = {
        chunk.id: index
        for index, chunk in enumerate(lexical_chunks, start=1)
    }
    lexical_score_by_chunk_id = {
        chunk.id: float(chunk.lexical_score)
        for chunk in lexical_chunks
    }
    fuzzy_chunks, fuzzy_score_by_chunk_id, fuzzy_signals_by_chunk_id, fuzzy_metadata = (
        _fuzzy_candidates(
            document,
            lexical_query_plan,
            provider,
            set(lexical_rank_by_chunk_id),
        )
    )
    combined_lexical_chunks = list(lexical_chunks)
    for chunk in fuzzy_chunks:
        if chunk.id in lexical_rank_by_chunk_id:
            continue
        lexical_rank_by_chunk_id[chunk.id] = len(combined_lexical_chunks) + 1
        lexical_score_by_chunk_id[chunk.id] = fuzzy_score_by_chunk_id.get(chunk.id)
        combined_lexical_chunks.append(chunk)
    candidate_chunk_ids = set(vector_rank_by_chunk_id) | set(lexical_rank_by_chunk_id)

    try:
        candidate_embeddings = list(
            base_queryset.filter(chunk_id__in=candidate_chunk_ids)
            .select_related(
                "chunk",
                "chunk__document",
                "chunk__extraction_result",
            )
            .annotate(cosine_distance=CosineDistance("embedding_vector", query_vector))
        )
    except DatabaseError as exc:
        logger.warning(
            "Semantic hybrid candidate scoring failed",
            extra={"document_id": document.id, "error_code": "semantic_search_failed"},
        )
        raise SemanticSearchError(
            "semantic_search_failed",
            "Semantic search could not be completed safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    results = []
    for embedding in candidate_embeddings:
        vector_rank = vector_rank_by_chunk_id.get(embedding.chunk_id)
        lexical_rank = lexical_rank_by_chunk_id.get(embedding.chunk_id)
        hybrid_score = _rrf_score(vector_rank, lexical_rank, rrf_k)
        results.append(
            _result_from_embedding(
                embedding,
                vector_rank=vector_rank,
                lexical_rank=lexical_rank,
                lexical_score=lexical_score_by_chunk_id.get(embedding.chunk_id),
                hybrid_score=hybrid_score,
            )
        )

    results = _apply_deterministic_reranking(results, normalized_query)
    results.sort(
        key=lambda result: (
            -(result.final_score or 0.0),
            -(result.hybrid_score or 0.0),
            result.lexical_rank if result.lexical_rank is not None else 1_000_000,
            result.vector_rank if result.vector_rank is not None else 1_000_000,
            result.chunk_index,
            result.chunk_id,
        )
    )
    final_results = results[:limit]

    return {
        "document_id": document.id,
        "query_length": len(normalized_query),
        "top_k": limit,
        "result_count": len(final_results),
        "results": [_result_payload(result) for result in final_results],
        "search_metadata": {
            "provider": provider.provider_name,
            "model": provider.model,
            "dimension": provider.dimension,
            "metric": "hybrid_rrf",
            "ranking": "descending_final_score",
            "similarity_score": "1 - cosine_distance",
            "search_type": "hybrid_pgvector_postgres_fts",
            "fusion_method": "rrf",
            "reranking": "deterministic_tender_value_signals",
            "rerank_enabled": get_hybrid_rerank_enabled(),
            "rrf_k": rrf_k,
            "vector_candidate_count": len(vector_embeddings),
            "lexical_candidate_count": len(combined_lexical_chunks),
            "postgres_fts_candidate_count": len(lexical_chunks),
            "fuzzy_candidate_count": len(fuzzy_chunks),
            "requested_vector_candidates": vector_limit,
            "requested_lexical_candidates": lexical_limit,
            "lexical_config": lexical_config,
            "lexical_search": "postgres_full_text_search",
            "fuzzy_matching": fuzzy_metadata,
            "fuzzy_candidate_signals": {
                str(chunk_id): signals
                for chunk_id, signals in fuzzy_signals_by_chunk_id.items()
            },
            "query_expansion": lexical_query_plan.expansion_enabled,
            "expanded_terms": list(lexical_query_plan.expanded_terms),
            "lexical_query_terms": list(lexical_query_plan.query_terms),
            "lexical_query_term_groups": [
                list(group) for group in lexical_query_plan.query_term_groups
            ],
            "limits": _search_limits_metadata(),
        },
    }


def semantic_search_document(document, query, top_k=None, provider=None):
    normalized_query = validate_search_query(query)
    limit = validate_top_k(top_k)
    provider_name, model, dimension = _provider_metadata(provider)

    _ensure_document_has_chunks(document)
    base_queryset = _current_embedding_queryset(document, provider_name, model, dimension)
    _ensure_embeddings_exist(base_queryset, document)

    provider = _get_search_provider(provider)
    if get_hybrid_search_enabled():
        payload = _hybrid_search_payload(
            document,
            normalized_query,
            limit,
            provider,
            base_queryset,
        )
    else:
        payload = _exact_vector_search_payload(
            document,
            normalized_query,
            limit,
            provider,
            base_queryset,
        )

    logger.info(
        "Semantic search completed",
        extra={
            "document_id": document.id,
            "provider": provider.provider_name,
            "model": provider.model,
            "top_k": limit,
            "result_count": payload["result_count"],
            "search_type": payload["search_metadata"]["search_type"],
        },
    )
    return payload


def semantic_search_settings_metadata():
    return {
        "provider": get_embedding_provider_name(),
        "model": get_embedding_model(),
        "dimension": get_embedding_dimension(),
        "limits": _search_limits_metadata(),
    }
