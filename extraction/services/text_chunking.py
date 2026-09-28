import hashlib
import logging
import re
from dataclasses import dataclass

from django.conf import settings
from django.db import DatabaseError, transaction
from rest_framework import status

from extraction.models import TextChunk, TextExtractionResult

logger = logging.getLogger(__name__)

DEFAULT_TARGET_CHARS = 1200
DEFAULT_MAX_CHARS = 1600
DEFAULT_OVERLAP_CHARS = 200
DEFAULT_MIN_CHARS = 100
DEFAULT_MAX_CHUNKS = 1000
DEFAULT_MAX_INPUT_CHARS = 1_000_000
CHUNKING_VERSION = "character-semantic-v1"


class TextChunkingError(Exception):
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
class TextSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ChunkBuildResult:
    chunks: list
    metadata: dict


def _get_positive_int_setting(name, default):
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def get_target_chunk_chars():
    return _get_positive_int_setting(
        "TEXT_CHUNKING_TARGET_CHARS",
        DEFAULT_TARGET_CHARS,
    )


def get_max_chunk_chars():
    return _get_positive_int_setting("TEXT_CHUNKING_MAX_CHARS", DEFAULT_MAX_CHARS)


def get_overlap_chars():
    try:
        value = int(getattr(settings, "TEXT_CHUNKING_OVERLAP_CHARS", DEFAULT_OVERLAP_CHARS))
    except (TypeError, ValueError):
        value = DEFAULT_OVERLAP_CHARS
    return max(0, value)


def get_min_chunk_chars():
    return _get_positive_int_setting("TEXT_CHUNKING_MIN_CHARS", DEFAULT_MIN_CHARS)


def get_max_chunks():
    return _get_positive_int_setting("TEXT_CHUNKING_MAX_CHUNKS", DEFAULT_MAX_CHUNKS)


def get_max_input_chars():
    return _get_positive_int_setting(
        "TEXT_CHUNKING_MAX_INPUT_CHARS",
        getattr(settings, "EXTRACTION_MAX_EXTRACTED_TEXT_CHARS", DEFAULT_MAX_INPUT_CHARS),
    )


def _chunking_limits_metadata():
    return {
        "target_chars": get_target_chunk_chars(),
        "max_chars": get_max_chunk_chars(),
        "overlap_chars": get_overlap_chars(),
        "min_chars": get_min_chunk_chars(),
        "max_chunks": get_max_chunks(),
        "max_input_chars": get_max_input_chars(),
    }


def _validate_chunking_settings():
    max_chars = get_max_chunk_chars()
    overlap_chars = get_overlap_chars()
    if overlap_chars >= max_chars:
        raise TextChunkingError(
            "invalid_chunking_configuration",
            "Text chunking configuration is invalid.",
            "TEXT_CHUNKING_OVERLAP_CHARS must be smaller than TEXT_CHUNKING_MAX_CHARS",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def clean_chunking_text(text):
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or ord(character) >= 32
    )
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return TextSpan(text=text[start:end], start=start, end=end)


def _split_hard(text, start, end, max_chars):
    spans = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + max_chars, end)
        if chunk_end < end:
            whitespace_split = max(
                text.rfind(" ", cursor, chunk_end),
                text.rfind("\n", cursor, chunk_end),
                text.rfind("\t", cursor, chunk_end),
            )
            if whitespace_split > cursor:
                chunk_end = whitespace_split

        span = _trim_span(text, cursor, chunk_end)
        if span is not None:
            spans.append(span)

        cursor = max(chunk_end, cursor + 1)
        while cursor < end and text[cursor].isspace():
            cursor += 1

    return spans


def _split_paragraph_sentences(text, paragraph, max_chars):
    sentence_boundary = re.compile(r"(?<=[.!?])\s+")
    spans = []
    sentence_start = paragraph.start

    for match in sentence_boundary.finditer(paragraph.text):
        sentence_end = paragraph.start + match.start()
        span = _trim_span(text, sentence_start, sentence_end)
        if span is not None:
            if len(span.text) > max_chars:
                spans.extend(_split_hard(text, span.start, span.end, max_chars))
            else:
                spans.append(span)
        sentence_start = paragraph.start + match.end()

    span = _trim_span(text, sentence_start, paragraph.end)
    if span is not None:
        if len(span.text) > max_chars:
            spans.extend(_split_hard(text, span.start, span.end, max_chars))
        else:
            spans.append(span)

    return spans


def _semantic_spans(text, max_non_overlap_chars):
    paragraph_pattern = re.compile(r"\S.*?(?=\n\s*\n|\Z)", re.DOTALL)
    spans = []

    for match in paragraph_pattern.finditer(text):
        paragraph = _trim_span(text, match.start(), match.end())
        if paragraph is None:
            continue

        if len(paragraph.text) <= max_non_overlap_chars:
            spans.append(paragraph)
            continue

        spans.extend(_split_paragraph_sentences(text, paragraph, max_non_overlap_chars))

    if not spans and text:
        spans.extend(_split_hard(text, 0, len(text), max_non_overlap_chars))

    return spans


def _join_spans(text, start, end):
    return clean_chunking_text(text[start:end])


def _build_base_chunks(text, spans):
    target_chars = get_target_chunk_chars()
    min_chars = get_min_chunk_chars()
    max_non_overlap_chars = get_max_chunk_chars() - get_overlap_chars()

    chunks = []
    current_start = None
    current_end = None

    for span in spans:
        if current_start is None:
            current_start = span.start
            current_end = span.end
            continue

        proposed_text = _join_spans(text, current_start, span.end)
        current_text = _join_spans(text, current_start, current_end)

        if (
            len(proposed_text) <= target_chars
            or (
                len(current_text) < min_chars
                and len(proposed_text) <= max_non_overlap_chars
            )
        ):
            current_end = span.end
            continue

        chunks.append(TextSpan(current_text, current_start, current_end))
        current_start = span.start
        current_end = span.end

    if current_start is not None:
        current_text = _join_spans(text, current_start, current_end)
        chunks.append(TextSpan(current_text, current_start, current_end))

    return chunks


def _overlap_start_for_chunk(text, previous_chunk, current_chunk):
    overlap_chars = get_overlap_chars()
    if overlap_chars <= 0:
        return current_chunk.start

    overlap_start = max(previous_chunk.end - overlap_chars, previous_chunk.start)
    while overlap_start > previous_chunk.start and not text[overlap_start - 1].isspace():
        overlap_start -= 1

    return min(overlap_start, current_chunk.start)


def _apply_overlap(text, base_chunks):
    if not base_chunks:
        return []

    chunks = [base_chunks[0]]
    for index, base_chunk in enumerate(base_chunks[1:], start=1):
        overlap_start = _overlap_start_for_chunk(text, base_chunks[index - 1], base_chunk)
        chunk_text = _join_spans(text, overlap_start, base_chunk.end)
        chunks.append(TextSpan(chunk_text, overlap_start, base_chunk.end))

    return chunks


def _estimated_token_count(text):
    # Rough English/French-oriented estimate. Future embedding models may tokenize
    # differently, so this is metadata only, not a hard token budget.
    return max(1, round(len(text) / 4)) if text else 0


def build_text_chunks(text):
    _validate_chunking_settings()
    normalized_text = clean_chunking_text(text)
    input_length = len(normalized_text)

    if input_length > get_max_input_chars():
        raise TextChunkingError(
            "chunking_input_too_large",
            "Extracted text exceeds the configured chunking input limit.",
        )

    if not normalized_text:
        return ChunkBuildResult(
            chunks=[],
            metadata={
                "chunking_version": CHUNKING_VERSION,
                "input_length": 0,
                "chunk_count": 0,
                "warnings": ["empty_extracted_text"],
                "limits": _chunking_limits_metadata(),
            },
        )

    max_non_overlap_chars = get_max_chunk_chars() - get_overlap_chars()
    spans = _semantic_spans(normalized_text, max_non_overlap_chars)
    base_chunks = _build_base_chunks(normalized_text, spans)
    chunks = _apply_overlap(normalized_text, base_chunks)

    if len(chunks) > get_max_chunks():
        raise TextChunkingError(
            "too_many_chunks",
            f"Chunking produced more than the configured limit of {get_max_chunks()} chunks.",
        )

    for chunk in chunks:
        if len(chunk.text) > get_max_chunk_chars():
            raise TextChunkingError(
                "chunk_too_large",
                "Chunking produced a chunk larger than the configured maximum size.",
            )

    return ChunkBuildResult(
        chunks=chunks,
        metadata={
            "chunking_version": CHUNKING_VERSION,
            "input_length": input_length,
            "chunk_count": len(chunks),
            "warnings": [],
            "limits": _chunking_limits_metadata(),
            "token_count_kind": "estimated",
            "token_count_note": "Estimated as character_count / 4; embedding tokenizer may differ.",
        },
    )


def _get_completed_extraction_result(document):
    try:
        extraction_result = TextExtractionResult.objects.get(document=document)
    except TextExtractionResult.DoesNotExist as exc:
        raise TextChunkingError(
            "extraction_not_found",
            "Completed text extraction result was not found for this document.",
            response_status=status.HTTP_404_NOT_FOUND,
        ) from exc

    if extraction_result.status != TextExtractionResult.Status.COMPLETED:
        raise TextChunkingError(
            "extraction_not_completed",
            "Only completed text extraction results can be chunked.",
        )

    return extraction_result


def _chunk_checksum(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_model(document, extraction_result, chunk_index, chunk, total_chunks):
    return TextChunk(
        document=document,
        extraction_result=extraction_result,
        chunk_index=chunk_index,
        text=chunk.text,
        character_start=chunk.start,
        character_end=chunk.end,
        text_length=len(chunk.text),
        text_sha256=_chunk_checksum(chunk.text),
        metadata={
            "chunking_version": CHUNKING_VERSION,
            "estimated_tokens": _estimated_token_count(chunk.text),
            "overlap_chars": get_overlap_chars() if chunk_index else 0,
            "is_first_chunk": chunk_index == 0,
            "is_last_chunk": chunk_index == total_chunks - 1,
        },
    )


def _payload(document, extraction_result, chunks, metadata):
    return {
        "document_id": document.id,
        "extraction_result_id": extraction_result.id,
        "chunk_count": len(chunks),
        "chunks": list(chunks),
        "chunking_metadata": metadata,
    }


def get_document_text_chunks(document):
    extraction_result = _get_completed_extraction_result(document)
    chunks = TextChunk.objects.filter(
        document=document,
        extraction_result=extraction_result,
    ).order_by("chunk_index")
    return _payload(
        document,
        extraction_result,
        chunks,
        {
            "chunking_version": CHUNKING_VERSION,
            "chunk_count": chunks.count(),
            "limits": _chunking_limits_metadata(),
        },
    )


def chunk_document_text(document):
    extraction_result = _get_completed_extraction_result(document)
    build_result = build_text_chunks(extraction_result.extracted_text)

    try:
        with transaction.atomic():
            TextChunk.objects.filter(extraction_result=extraction_result).delete()
            chunk_models = [
                _chunk_model(
                    document,
                    extraction_result,
                    chunk_index,
                    chunk,
                    len(build_result.chunks),
                )
                for chunk_index, chunk in enumerate(build_result.chunks)
            ]
            if chunk_models:
                TextChunk.objects.bulk_create(chunk_models)
    except DatabaseError as exc:
        logger.warning(
            "Text chunk replacement failed",
            extra={
                "document_id": document.id,
                "extraction_result_id": extraction_result.id,
                "error_code": "chunk_storage_failed",
            },
        )
        raise TextChunkingError(
            "chunk_storage_failed",
            "Text chunks could not be stored safely.",
            exc.__class__.__name__,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    chunks = TextChunk.objects.filter(
        document=document,
        extraction_result=extraction_result,
    ).order_by("chunk_index")

    logger.info(
        "Text chunking completed",
        extra={
            "document_id": document.id,
            "extraction_result_id": extraction_result.id,
            "chunk_count": len(build_result.chunks),
        },
    )

    return _payload(document, extraction_result, chunks, build_result.metadata)
