import hashlib
import logging
import re
from dataclasses import dataclass
from io import BytesIO

from django.conf import settings
from pdf2image import convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFPopplerTimeoutError,
    PDFSyntaxError,
)
from PIL import Image
from pypdf import PdfReader
from pypdf.errors import (
    EmptyFileError,
    FileNotDecryptedError,
    PdfReadError,
    PdfStreamError,
    PyPdfError,
    WrongPasswordError,
)
import pytesseract
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from extraction.models import TextExtractionResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAGES = 100
DEFAULT_MAX_EXTRACTED_TEXT_CHARS = 1_000_000
DEFAULT_MAX_PAGE_CONTENT_BYTES = 5 * 1024 * 1024
DEFAULT_NATIVE_SAMPLE_PAGES = 5
DEFAULT_USABLE_TEXT_MIN_CHARS = 12
DEFAULT_USABLE_TEXT_MIN_ALNUM_CHARS = 8
DEFAULT_NATIVE_CLASSIFICATION_MIN_TOTAL_CHARS = 100
DEFAULT_NATIVE_CLASSIFICATION_MIN_AVERAGE_CHARS = 30
DEFAULT_OCR_DPI = 200
DEFAULT_OCR_MAX_IMAGE_PIXELS = 20_000_000
DEFAULT_OCR_TIMEOUT_SECONDS = 30
DEFAULT_OCR_MAX_TEXT_CHARS_PER_PAGE = 100_000

SOURCE_NATIVE = "native"
SOURCE_OCR = "ocr"
SOURCE_MIXED = "mixed"
SOURCE_NONE = "none"


class TextExtractionError(Exception):
    def __init__(self, code, public_message, internal_detail="", metadata=None):
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.internal_detail = internal_detail
        self.metadata = metadata or {}


@dataclass(frozen=True)
class TextUsability:
    normalized_character_count: int
    alphanumeric_character_count: int
    usable: bool


@dataclass(frozen=True)
class PageNativeText:
    page_number: int
    text: str
    content_bytes: int
    usability: TextUsability


@dataclass(frozen=True)
class PdfClassification:
    probable_source: str
    sampled_pages: int
    sampled_usable_pages: int
    sampled_usable_characters: int
    sampled_average_usable_characters: float


@dataclass(frozen=True)
class ExtractedPdfText:
    text: str
    page_count: int
    pages_processed: int
    has_text: bool
    metadata: dict


def _get_positive_int_setting(name, default):
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def get_max_pages():
    return _get_positive_int_setting("EXTRACTION_MAX_PDF_PAGES", DEFAULT_MAX_PAGES)


def get_max_extracted_text_chars():
    return _get_positive_int_setting(
        "EXTRACTION_MAX_EXTRACTED_TEXT_CHARS",
        DEFAULT_MAX_EXTRACTED_TEXT_CHARS,
    )


def get_max_page_content_bytes():
    return _get_positive_int_setting(
        "EXTRACTION_MAX_PAGE_CONTENT_BYTES",
        DEFAULT_MAX_PAGE_CONTENT_BYTES,
    )


def get_native_sample_pages():
    return _get_positive_int_setting(
        "EXTRACTION_NATIVE_SAMPLE_PAGES",
        DEFAULT_NATIVE_SAMPLE_PAGES,
    )


def get_usable_text_min_chars():
    return _get_positive_int_setting(
        "EXTRACTION_USABLE_TEXT_MIN_CHARS",
        DEFAULT_USABLE_TEXT_MIN_CHARS,
    )


def get_usable_text_min_alnum_chars():
    return _get_positive_int_setting(
        "EXTRACTION_USABLE_TEXT_MIN_ALNUM_CHARS",
        DEFAULT_USABLE_TEXT_MIN_ALNUM_CHARS,
    )


def get_native_classification_min_total_chars():
    return _get_positive_int_setting(
        "EXTRACTION_NATIVE_CLASSIFICATION_MIN_TOTAL_CHARS",
        DEFAULT_NATIVE_CLASSIFICATION_MIN_TOTAL_CHARS,
    )


def get_native_classification_min_average_chars():
    return _get_positive_int_setting(
        "EXTRACTION_NATIVE_CLASSIFICATION_MIN_AVERAGE_CHARS",
        DEFAULT_NATIVE_CLASSIFICATION_MIN_AVERAGE_CHARS,
    )


def get_ocr_dpi():
    return _get_positive_int_setting("EXTRACTION_OCR_DPI", DEFAULT_OCR_DPI)


def get_ocr_max_pages():
    return _get_positive_int_setting("EXTRACTION_OCR_MAX_PDF_PAGES", get_max_pages())


def get_ocr_max_image_pixels():
    return _get_positive_int_setting(
        "EXTRACTION_OCR_MAX_IMAGE_PIXELS",
        DEFAULT_OCR_MAX_IMAGE_PIXELS,
    )


def get_ocr_timeout_seconds():
    return _get_positive_int_setting(
        "EXTRACTION_OCR_TIMEOUT_SECONDS",
        DEFAULT_OCR_TIMEOUT_SECONDS,
    )


def get_ocr_max_text_chars_per_page():
    return _get_positive_int_setting(
        "EXTRACTION_OCR_MAX_TEXT_CHARS_PER_PAGE",
        DEFAULT_OCR_MAX_TEXT_CHARS_PER_PAGE,
    )


def get_poppler_path():
    return getattr(settings, "POPPLER_PATH", "") or None


def get_tesseract_cmd():
    return getattr(settings, "TESSERACT_CMD", "") or ""


def clean_extracted_text(text):
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


def measure_text_usability(text):
    cleaned_text = clean_extracted_text(text)
    normalized_character_count = sum(
        1 for character in cleaned_text if not character.isspace()
    )
    alphanumeric_character_count = sum(
        1 for character in cleaned_text if character.isalnum()
    )
    return TextUsability(
        normalized_character_count=normalized_character_count,
        alphanumeric_character_count=alphanumeric_character_count,
        usable=(
            normalized_character_count >= get_usable_text_min_chars()
            and alphanumeric_character_count >= get_usable_text_min_alnum_chars()
        ),
    )


def is_usable_text(text):
    return measure_text_usability(text).usable


def get_page_content_size(page):
    contents = page.get_contents()
    if contents is None:
        return 0
    data = contents.get_data()
    return len(data)


def _read_document_bytes(document):
    try:
        document.file.open("rb")
        return document.file.read()
    except OSError as exc:
        raise TextExtractionError(
            "file_unavailable",
            "Document file is temporarily unavailable.",
            exc.__class__.__name__,
        ) from exc
    finally:
        try:
            document.file.close()
        except OSError:
            logger.debug(
                "Could not close document file after extraction",
                extra={"document_id": document.id},
            )


def _extract_native_page_text(page, page_number):
    content_size = get_page_content_size(page)
    if content_size > get_max_page_content_bytes():
        raise TextExtractionError(
            "page_content_too_large",
            "A PDF page content stream exceeds the configured extraction limit.",
            f"page={page_number}; content_bytes={content_size}",
        )

    page_text = clean_extracted_text(page.extract_text() or "")
    return PageNativeText(
        page_number=page_number,
        text=page_text,
        content_bytes=content_size,
        usability=measure_text_usability(page_text),
    )


def _sample_native_pages(reader):
    sampled_page_count = min(len(reader.pages), get_native_sample_pages())
    native_page_cache = {}

    for page_number in range(1, sampled_page_count + 1):
        native_page_cache[page_number] = _extract_native_page_text(
            reader.pages[page_number - 1],
            page_number,
        )

    return native_page_cache


def _classify_pdf_from_sample(native_page_cache):
    sampled_pages = len(native_page_cache)
    usable_pages = [
        page_text
        for page_text in native_page_cache.values()
        if page_text.usability.usable
    ]
    sampled_usable_characters = sum(
        page_text.usability.normalized_character_count
        for page_text in usable_pages
    )
    sampled_average_usable_characters = (
        sampled_usable_characters / sampled_pages if sampled_pages else 0
    )
    probable_source = (
        "probably_native"
        if (
            sampled_usable_characters >= get_native_classification_min_total_chars()
            or sampled_average_usable_characters
            >= get_native_classification_min_average_chars()
        )
        else "probably_scanned"
    )

    return PdfClassification(
        probable_source=probable_source,
        sampled_pages=sampled_pages,
        sampled_usable_pages=len(usable_pages),
        sampled_usable_characters=sampled_usable_characters,
        sampled_average_usable_characters=round(sampled_average_usable_characters, 2),
    )


def _configure_tesseract():
    tesseract_cmd = get_tesseract_cmd()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def _render_page_for_ocr(pdf_bytes, page_number):
    try:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=get_ocr_dpi(),
            first_page=page_number,
            last_page=page_number,
            fmt="png",
            thread_count=1,
            poppler_path=get_poppler_path(),
            grayscale=True,
            strict=True,
            timeout=get_ocr_timeout_seconds(),
        )
    except PDFInfoNotInstalledError as exc:
        raise TextExtractionError(
            "poppler_unavailable",
            "PDF rendering service is unavailable.",
            exc.__class__.__name__,
            metadata={"ocr_status": "failed", "ocr_error_code": "poppler_unavailable"},
        ) from exc
    except PDFPopplerTimeoutError as exc:
        raise TextExtractionError(
            "pdf_render_timeout",
            "PDF page rendering timed out.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={"ocr_status": "failed", "ocr_error_code": "pdf_render_timeout"},
        ) from exc
    except PDFSyntaxError as exc:
        raise TextExtractionError(
            "pdf_rendering_failed",
            "PDF page rendering failed.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={"ocr_status": "failed", "ocr_error_code": "pdf_rendering_failed"},
        ) from exc
    except PDFPageCountError as exc:
        raise TextExtractionError(
            "pdf_rendering_failed",
            "PDF page rendering failed.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={"ocr_status": "failed", "ocr_error_code": "pdf_rendering_failed"},
        ) from exc
    except (OSError, ValueError) as exc:
        raise TextExtractionError(
            "pdf_rendering_failed",
            "PDF page rendering failed.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={"ocr_status": "failed", "ocr_error_code": "pdf_rendering_failed"},
        ) from exc

    if not images:
        raise TextExtractionError(
            "pdf_rendering_failed",
            "PDF page rendering failed.",
            f"page={page_number}; no_image_returned",
            metadata={"ocr_status": "failed", "ocr_error_code": "pdf_rendering_failed"},
        )

    return images[0]


def _validate_rendered_image(image, page_number):
    try:
        width, height = image.size
    except (AttributeError, ValueError) as exc:
        raise TextExtractionError(
            "malformed_rendered_image",
            "Rendered PDF page image could not be processed.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={
                "ocr_status": "failed",
                "ocr_error_code": "malformed_rendered_image",
            },
        ) from exc

    pixel_count = width * height
    if pixel_count > get_ocr_max_image_pixels():
        raise TextExtractionError(
            "rendered_image_too_large",
            "Rendered PDF page image exceeds the configured OCR size limit.",
            f"page={page_number}; pixels={pixel_count}",
            metadata={
                "ocr_status": "failed",
                "ocr_error_code": "rendered_image_too_large",
            },
        )


def _ocr_page_text(pdf_bytes, page_number):
    _configure_tesseract()
    image = _render_page_for_ocr(pdf_bytes, page_number)
    ocr_image = None

    try:
        _validate_rendered_image(image, page_number)
        ocr_image = image if image.mode == "L" else image.convert("L")
        raw_text = pytesseract.image_to_string(
            ocr_image,
            timeout=get_ocr_timeout_seconds(),
        )
    except TesseractNotFoundError as exc:
        raise TextExtractionError(
            "tesseract_unavailable",
            "OCR engine is unavailable.",
            exc.__class__.__name__,
            metadata={
                "ocr_status": "failed",
                "ocr_error_code": "tesseract_unavailable",
            },
        ) from exc
    except TesseractError as exc:
        raise TextExtractionError(
            "ocr_engine_error",
            "OCR engine failed while processing the document.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={"ocr_status": "failed", "ocr_error_code": "ocr_engine_error"},
        ) from exc
    except RuntimeError as exc:
        raise TextExtractionError(
            "ocr_engine_timeout",
            "OCR engine timed out while processing the document.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={"ocr_status": "failed", "ocr_error_code": "ocr_engine_timeout"},
        ) from exc
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise TextExtractionError(
            "ocr_image_error",
            "Rendered PDF page image could not be processed.",
            f"page={page_number}; {exc.__class__.__name__}",
            metadata={"ocr_status": "failed", "ocr_error_code": "ocr_image_error"},
        ) from exc
    finally:
        if ocr_image is not None and ocr_image is not image:
            ocr_image.close()
        image.close()

    page_text = clean_extracted_text(raw_text)
    if len(page_text) > get_ocr_max_text_chars_per_page():
        raise TextExtractionError(
            "ocr_text_too_large",
            "OCR output exceeds the configured per-page text limit.",
            f"page={page_number}; characters={len(page_text)}",
            metadata={"ocr_status": "failed", "ocr_error_code": "ocr_text_too_large"},
        )

    return page_text


def _build_empty_metadata(warnings):
    return {
        "parser": "pypdf",
        "method": "native_with_ocr_fallback",
        "ocr_status": "not_used",
        "source": SOURCE_NATIVE,
        "classification": {
            "strategy": "first_pages_native_text_heuristic",
            "probable_source": "probably_native",
            "sampled_pages": 0,
            "sampled_usable_pages": 0,
            "sampled_usable_characters": 0,
            "sampled_average_usable_characters": 0,
        },
        "warnings": warnings,
        "limits": _limits_metadata(),
    }


def _source_from_counts(native_pages, ocr_pages):
    if native_pages and ocr_pages:
        return SOURCE_MIXED
    if ocr_pages:
        return SOURCE_OCR
    return SOURCE_NATIVE


def _page_metadata(page_number, source, native_usability, ocr_usability=None):
    metadata = {
        "page": page_number,
        "source": source,
        "native_usable": native_usability.usable,
        "native_normalized_characters": native_usability.normalized_character_count,
        "native_alphanumeric_characters": native_usability.alphanumeric_character_count,
    }
    if ocr_usability is not None:
        metadata.update(
            {
                "ocr_usable": ocr_usability.usable,
                "ocr_normalized_characters": (
                    ocr_usability.normalized_character_count
                ),
                "ocr_alphanumeric_characters": (
                    ocr_usability.alphanumeric_character_count
                ),
            }
        )
    return metadata


def extract_text_from_document_file(document):
    max_pages = get_max_pages()
    max_text_chars = get_max_extracted_text_chars()
    pdf_bytes = _read_document_bytes(document)

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=True)

        if reader.is_encrypted:
            raise TextExtractionError(
                "encrypted_pdf",
                "Password-protected or encrypted PDFs are not supported.",
            )

        page_count = len(reader.pages)
        if page_count > max_pages:
            raise TextExtractionError(
                "too_many_pages",
                f"PDF exceeds the maximum extraction limit of {max_pages} pages.",
            )

        if page_count == 0:
            return ExtractedPdfText(
                text="",
                page_count=0,
                pages_processed=0,
                has_text=False,
                metadata=_build_empty_metadata(["empty_pdf"]),
            )

        native_page_cache = _sample_native_pages(reader)
        classification = _classify_pdf_from_sample(native_page_cache)

        page_texts = []
        page_metadata = []
        warnings = []
        total_characters = 0
        pages_processed = 0
        native_pages = 0
        ocr_pages = 0
        ocr_empty_pages = 0
        native_unusable_pages = 0

        for page_number, page in enumerate(reader.pages, start=1):
            native_page = native_page_cache.get(page_number)
            if native_page is None:
                native_page = _extract_native_page_text(page, page_number)

            pages_processed += 1
            selected_text = ""

            if native_page.usability.usable:
                selected_text = native_page.text
                native_pages += 1
                page_metadata.append(
                    _page_metadata(
                        page_number,
                        SOURCE_NATIVE,
                        native_page.usability,
                    )
                )
            else:
                native_unusable_pages += 1
                if page_count > get_ocr_max_pages():
                    raise TextExtractionError(
                        "too_many_ocr_pages",
                        (
                            "PDF exceeds the maximum OCR extraction limit of "
                            f"{get_ocr_max_pages()} pages."
                        ),
                        metadata={
                            "ocr_status": "blocked",
                            "ocr_error_code": "too_many_ocr_pages",
                        },
                    )

                ocr_text = _ocr_page_text(pdf_bytes, page_number)
                ocr_pages += 1
                ocr_usability = measure_text_usability(ocr_text)

                if ocr_usability.usable:
                    selected_text = ocr_text
                    page_source = SOURCE_OCR
                else:
                    ocr_empty_pages += 1
                    page_source = SOURCE_NONE

                page_metadata.append(
                    _page_metadata(
                        page_number,
                        page_source,
                        native_page.usability,
                        ocr_usability,
                    )
                )

            if not selected_text:
                continue

            total_characters += len(selected_text)
            if total_characters > max_text_chars:
                raise TextExtractionError(
                    "extracted_text_too_large",
                    "Extracted text exceeds the configured maximum character limit.",
                )

            page_texts.append(selected_text)

        extracted_text = clean_extracted_text("\n\n".join(page_texts))

        if not extracted_text:
            warnings.append("no_extractable_text")
        if ocr_empty_pages:
            warnings.append("ocr_returned_no_usable_text")

        extraction_source = _source_from_counts(native_pages, ocr_pages)
        metadata = {
            "parser": "pypdf",
            "method": "native_with_ocr_fallback",
            "ocr_status": "used" if ocr_pages else "not_used",
            "ocr_engine": "tesseract" if ocr_pages else "",
            "source": extraction_source,
            "classification": {
                "strategy": "first_pages_native_text_heuristic",
                "probable_source": classification.probable_source,
                "sampled_pages": classification.sampled_pages,
                "sampled_usable_pages": classification.sampled_usable_pages,
                "sampled_usable_characters": (
                    classification.sampled_usable_characters
                ),
                "sampled_average_usable_characters": (
                    classification.sampled_average_usable_characters
                ),
                "native_thresholds": {
                    "page_min_normalized_characters": get_usable_text_min_chars(),
                    "page_min_alphanumeric_characters": (
                        get_usable_text_min_alnum_chars()
                    ),
                    "sample_min_total_usable_characters": (
                        get_native_classification_min_total_chars()
                    ),
                    "sample_min_average_usable_characters": (
                        get_native_classification_min_average_chars()
                    ),
                },
            },
            "page_sources": page_metadata,
            "page_source_counts": {
                "native": native_pages,
                "ocr": ocr_pages,
                "native_unusable": native_unusable_pages,
                "ocr_empty": ocr_empty_pages,
            },
            "warnings": warnings,
            "limits": _limits_metadata(),
        }

        return ExtractedPdfText(
            text=extracted_text,
            page_count=page_count,
            pages_processed=pages_processed,
            has_text=bool(extracted_text),
            metadata=metadata,
        )
    except TextExtractionError:
        raise
    except (EmptyFileError, FileNotDecryptedError, WrongPasswordError) as exc:
        raise TextExtractionError(
            "encrypted_or_empty_pdf",
            "PDF is empty, encrypted, or cannot be decrypted.",
            exc.__class__.__name__,
        ) from exc
    except (PdfReadError, PdfStreamError) as exc:
        raise TextExtractionError(
            "malformed_pdf",
            "PDF could not be parsed safely.",
            exc.__class__.__name__,
        ) from exc
    except PyPdfError as exc:
        raise TextExtractionError(
            "pdf_parser_error",
            "PDF text extraction failed.",
            exc.__class__.__name__,
        ) from exc
    except Exception as exc:
        raise TextExtractionError(
            "unexpected_parser_error",
            "PDF text extraction failed.",
            exc.__class__.__name__,
        ) from exc


def extract_document_text(document):
    result, _ = TextExtractionResult.objects.update_or_create(
        document=document,
        defaults={
            "status": TextExtractionResult.Status.PROCESSING,
            "extracted_text": "",
            "has_text": False,
            "page_count": 0,
            "pages_processed": 0,
            "character_count": 0,
            "text_sha256": "",
            "extraction_metadata": {
                "parser": "pypdf",
                "method": "native_with_ocr_fallback",
                "ocr_status": "not_used",
                "source": SOURCE_NATIVE,
                "limits": _limits_metadata(),
            },
            "error_code": "",
            "error_message": "",
            "internal_error_detail": "",
        },
    )

    logger.info(
        "PDF text extraction started",
        extra={"document_id": document.id, "owner_id": document.owner_id},
    )

    try:
        extracted = extract_text_from_document_file(document)
    except TextExtractionError as exc:
        result.status = TextExtractionResult.Status.FAILED
        result.error_code = exc.code
        result.error_message = exc.public_message
        result.internal_error_detail = exc.internal_detail
        result.extraction_metadata = {
            "parser": "pypdf",
            "method": "native_with_ocr_fallback",
            "ocr_status": exc.metadata.get("ocr_status", "not_used"),
            "source": "failed",
            "limits": _limits_metadata(),
            **exc.metadata,
        }
        result.save(
            update_fields=[
                "status",
                "error_code",
                "error_message",
                "internal_error_detail",
                "extraction_metadata",
                "updated_at",
            ]
        )
        logger.warning(
            "PDF text extraction failed",
            extra={
                "document_id": document.id,
                "owner_id": document.owner_id,
                "error_code": exc.code,
            },
        )
        return result

    result.status = TextExtractionResult.Status.COMPLETED
    result.extracted_text = extracted.text
    result.has_text = extracted.has_text
    result.page_count = extracted.page_count
    result.pages_processed = extracted.pages_processed
    result.character_count = len(extracted.text)
    result.text_sha256 = (
        hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
        if extracted.text
        else ""
    )
    result.extraction_metadata = extracted.metadata
    result.error_code = ""
    result.error_message = ""
    result.internal_error_detail = ""
    result.save(
        update_fields=[
            "status",
            "extracted_text",
            "has_text",
            "page_count",
            "pages_processed",
            "character_count",
            "text_sha256",
            "extraction_metadata",
            "error_code",
            "error_message",
            "internal_error_detail",
            "updated_at",
        ]
    )
    logger.info(
        "PDF text extraction completed",
        extra={
            "document_id": document.id,
            "owner_id": document.owner_id,
            "page_count": result.page_count,
            "character_count": result.character_count,
            "has_text": result.has_text,
            "source": result.extraction_metadata.get("source"),
            "ocr_pages": result.extraction_metadata.get(
                "page_source_counts",
                {},
            ).get("ocr", 0),
        },
    )
    return result


def _limits_metadata():
    return {
        "max_pages": get_max_pages(),
        "max_extracted_text_chars": get_max_extracted_text_chars(),
        "max_page_content_bytes": get_max_page_content_bytes(),
        "native_sample_pages": get_native_sample_pages(),
        "usable_text_min_chars": get_usable_text_min_chars(),
        "usable_text_min_alnum_chars": get_usable_text_min_alnum_chars(),
        "native_classification_min_total_chars": (
            get_native_classification_min_total_chars()
        ),
        "native_classification_min_average_chars": (
            get_native_classification_min_average_chars()
        ),
        "ocr_max_pages": get_ocr_max_pages(),
        "ocr_dpi": get_ocr_dpi(),
        "ocr_max_image_pixels": get_ocr_max_image_pixels(),
        "ocr_timeout_seconds": get_ocr_timeout_seconds(),
        "ocr_max_text_chars_per_page": get_ocr_max_text_chars_per_page(),
    }
