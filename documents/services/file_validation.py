import logging
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

from django.conf import settings
from django.utils.text import get_valid_filename
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

PDF_EXTENSION = ".pdf"
PDF_HEADER = b"%PDF-"
DEFAULT_MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_ALLOWED_CONTENT_TYPES = {"application/pdf"}


class DocumentUploadValidationError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class ValidatedDocumentUpload:
    original_filename: str
    stored_filename: str
    mime_type: str
    file_size: int


def get_max_upload_size():
    return getattr(
        settings,
        "DOCUMENTS_MAX_UPLOAD_SIZE_BYTES",
        DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    )


def get_allowed_content_types():
    return set(
        getattr(
            settings,
            "DOCUMENTS_ALLOWED_UPLOAD_CONTENT_TYPES",
            DEFAULT_ALLOWED_CONTENT_TYPES,
        )
    )


def sanitize_original_filename(filename):
    raw_filename = str(filename or "")
    basename = raw_filename.replace("\\", "/").split("/")[-1]
    basename = PurePath(basename).name
    safe_name = get_valid_filename(basename).strip(" ._")

    if not safe_name:
        raise DocumentUploadValidationError(
            "A valid PDF filename is required.",
            "invalid_filename",
        )

    if PurePath(safe_name).suffix.lower() != PDF_EXTENSION:
        raise DocumentUploadValidationError(
            "Only PDF files are supported.",
            "unsupported_extension",
        )

    if len(safe_name) > 255:
        stem = PurePath(safe_name).stem[: 255 - len(PDF_EXTENSION)]
        safe_name = f"{stem.rstrip(' ._')}{PDF_EXTENSION}"

    return safe_name


def build_stored_filename(original_filename):
    extension = PurePath(original_filename).suffix.lower()
    if extension != PDF_EXTENSION:
        raise DocumentUploadValidationError(
            "Only PDF files are supported.",
            "unsupported_extension",
        )

    return f"{uuid.uuid4().hex}{PDF_EXTENSION}"


def validate_pdf_structure(content):
    try:
        reader = PdfReader(BytesIO(content), strict=False)
    except (PdfReadError, OSError, ValueError, TypeError) as exc:
        logger.warning("Rejected malformed PDF upload: %s", exc)
        raise DocumentUploadValidationError(
            "Uploaded PDF appears to be incomplete or malformed.",
            "invalid_pdf_structure",
        ) from exc

    if reader.is_encrypted:
        raise DocumentUploadValidationError(
            "Password-protected or encrypted PDFs are not supported.",
            "encrypted_pdf",
        )

    try:
        if "/Root" not in reader.trailer:
            raise PdfReadError("missing document catalog root")
        len(reader.pages)
    except (PdfReadError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.warning("Rejected malformed PDF upload: %s", exc)
        raise DocumentUploadValidationError(
            "Uploaded PDF appears to be incomplete or malformed.",
            "invalid_pdf_structure",
        ) from exc


def validate_pdf_upload(uploaded_file):
    original_filename = sanitize_original_filename(uploaded_file.name)
    stored_filename = build_stored_filename(original_filename)
    file_size = getattr(uploaded_file, "size", 0) or 0
    max_size = get_max_upload_size()

    if file_size <= 0:
        raise DocumentUploadValidationError(
            "Uploaded files must not be empty.",
            "empty_file",
        )

    if file_size > max_size:
        raise DocumentUploadValidationError(
            f"Uploaded PDF exceeds the maximum allowed size of {max_size} bytes.",
            "file_too_large",
        )

    mime_type = (getattr(uploaded_file, "content_type", "") or "").lower()
    allowed_content_types = get_allowed_content_types()
    if mime_type not in allowed_content_types:
        raise DocumentUploadValidationError(
            "Uploaded file must declare a PDF content type.",
            "unsupported_content_type",
        )

    current_position = uploaded_file.tell() if hasattr(uploaded_file, "tell") else None
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
    except (OSError, ValueError) as exc:
        logger.warning("Rejected unreadable document upload: %s", exc)
        raise DocumentUploadValidationError(
            "Uploaded file could not be read safely.",
            "unreadable_file",
        ) from exc
    finally:
        try:
            uploaded_file.seek(0 if current_position is None else current_position)
        except (OSError, ValueError):
            logger.debug("Could not restore uploaded file pointer after validation.")

    if not content.startswith(PDF_HEADER):
        raise DocumentUploadValidationError(
            "Uploaded file is not a valid PDF.",
            "invalid_pdf_header",
        )

    validate_pdf_structure(content)

    try:
        uploaded_file.seek(0)
    except (OSError, ValueError):
        logger.debug("Could not reset uploaded file pointer before storage.")

    return ValidatedDocumentUpload(
        original_filename=original_filename,
        stored_filename=stored_filename,
        mime_type=mime_type,
        file_size=file_size,
    )
