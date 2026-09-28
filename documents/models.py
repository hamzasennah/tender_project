import posixpath
import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q


def document_upload_path(instance, filename):
    stored_filename = instance.stored_filename or f"{uuid.uuid4().hex}.pdf"
    owner_segment = f"user_{instance.owner_id}" if instance.owner_id else "unassigned"
    return posixpath.join("documents", owner_segment, stored_filename)


class Document(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="documents",
        db_index=True,
    )
    file = models.FileField(upload_to=document_upload_path, max_length=512)
    original_filename = models.CharField(max_length=255)
    stored_filename = models.CharField(max_length=128, editable=False, db_index=True)
    mime_type = models.CharField(max_length=100)
    file_size = models.PositiveBigIntegerField()
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.UPLOADED,
        db_index=True,
    )
    processing_metadata = models.JSONField(default=dict, blank=True)
    error_code = models.CharField(max_length=64, blank=True)
    error_message = models.CharField(max_length=255, blank=True)
    internal_error_detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner", "-created_at"], name="documents_owner_created_idx"),
            models.Index(fields=["owner", "status"], name="documents_owner_status_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=Q(file_size__gt=0),
                name="documents_file_size_positive",
            ),
            models.CheckConstraint(
                condition=Q(
                    status__in=[
                        "uploaded",
                        "processing",
                        "completed",
                        "failed",
                    ]
                ),
                name="documents_valid_status",
            ),
        ]

    def __str__(self):
        return self.original_filename
