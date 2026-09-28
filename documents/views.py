import logging

from django.http import FileResponse
from rest_framework import status, viewsets
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from rest_framework.decorators import action
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from documents.models import Document
from documents.permissions import IsDocumentOwner
from documents.serializers import DocumentReadSerializer, DocumentUploadSerializer
from documents.services.document_service import delete_document

logger = logging.getLogger(__name__)


class DocumentFileUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Document file is temporarily unavailable."
    default_code = "document_file_unavailable"


class DocumentViewSet(viewsets.ModelViewSet):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated, IsDocumentOwner]
    parser_classes = [MultiPartParser, FormParser]
    http_method_names = ["get", "post", "delete", "head", "options"]

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user).order_by("-created_at")

    def get_serializer_class(self):
        if self.action == "create":
            return DocumentUploadSerializer
        return DocumentReadSerializer

    def create(self, request, *args, **kwargs):
        if len(request.FILES) != 1 or "file" not in request.FILES:
            logger.warning(
                "Document upload rejected",
                extra={
                    "owner_id": request.user.id,
                    "reason": "invalid_file_count",
                    "file_count": len(request.FILES),
                },
            )
            raise ValidationError({"file": "Upload exactly one PDF file."})

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        document = serializer.save()
        read_serializer = DocumentReadSerializer(
            document,
            context=self.get_serializer_context(),
        )
        headers = self.get_success_headers(read_serializer.data)
        return Response(
            read_serializer.data,
            status=status.HTTP_201_CREATED,
            headers=headers,
        )

    def perform_destroy(self, instance):
        delete_document(instance)

    @action(detail=True, methods=["get"], url_path="download")
    def download(self, request, pk=None):
        document = self.get_object()
        try:
            file_handle = document.file.open("rb")
        except OSError as exc:
            logger.exception(
                "Document file could not be opened",
                extra={"document_id": document.id, "owner_id": request.user.id},
            )
            raise DocumentFileUnavailable() from exc

        response = FileResponse(
            file_handle,
            as_attachment=True,
            filename=document.original_filename,
            content_type=document.mime_type,
        )
        response["X-Content-Type-Options"] = "nosniff"
        return response
