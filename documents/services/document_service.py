import logging

from django.db import transaction

from documents.models import Document

logger = logging.getLogger(__name__)


@transaction.atomic
def create_document(*, owner, uploaded_file, validated_upload):
    document = Document.objects.create(
        owner=owner,
        file=uploaded_file,
        original_filename=validated_upload.original_filename,
        stored_filename=validated_upload.stored_filename,
        mime_type=validated_upload.mime_type,
        file_size=validated_upload.file_size,
        status=Document.Status.UPLOADED,
    )

    logger.info(
        "Document upload accepted",
        extra={
            "document_id": document.id,
            "owner_id": owner.id,
            "file_size": document.file_size,
            "mime_type": document.mime_type,
        },
    )
    return document


@transaction.atomic
def delete_document(document):
    file_name = document.file.name
    storage = document.file.storage
    document_id = document.id
    owner_id = document.owner_id

    document.delete()

    if file_name:
        def delete_file():
            try:
                storage.delete(file_name)
            except OSError:
                logger.exception(
                    "Document file deletion failed",
                    extra={"document_id": document_id, "owner_id": owner_id},
                )

        transaction.on_commit(delete_file)

    logger.info(
        "Document deleted",
        extra={"document_id": document_id, "owner_id": owner_id},
    )
