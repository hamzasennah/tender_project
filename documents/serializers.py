import logging

from rest_framework import serializers
from rest_framework.reverse import reverse

from documents.models import Document
from documents.services.document_service import create_document
from documents.services.file_validation import (
    DocumentUploadValidationError,
    validate_pdf_upload,
)

logger = logging.getLogger(__name__)


class DocumentReadSerializer(serializers.ModelSerializer):
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id",
            "original_filename",
            "mime_type",
            "file_size",
            "status",
            "processing_metadata",
            "error_code",
            "error_message",
            "created_at",
            "updated_at",
            "download_url",
        ]
        read_only_fields = fields

    def get_download_url(self, obj):
        request = self.context.get("request")
        return reverse("document-download", args=[obj.pk], request=request)


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField(write_only=True)

    def validate_file(self, uploaded_file):
        request = self.context.get("request")
        try:
            validated_upload = validate_pdf_upload(uploaded_file)
        except DocumentUploadValidationError as exc:
            logger.warning(
                "Document upload rejected",
                extra={
                    "owner_id": getattr(getattr(request, "user", None), "id", None),
                    "reason": exc.code,
                    "declared_content_type": getattr(uploaded_file, "content_type", ""),
                    "file_size": getattr(uploaded_file, "size", None),
                },
            )
            raise serializers.ValidationError(exc.message, code=exc.code) from exc

        self._validated_upload = validated_upload
        return uploaded_file

    def create(self, validated_data):
        request = self.context["request"]
        return create_document(
            owner=request.user,
            uploaded_file=validated_data["file"],
            validated_upload=self._validated_upload,
        )
