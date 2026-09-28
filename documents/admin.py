from django.contrib import admin

from documents.models import Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "original_filename",
        "owner",
        "status",
        "mime_type",
        "file_size",
        "created_at",
    )
    list_filter = ("status", "mime_type", "created_at")
    search_fields = (
        "original_filename",
        "stored_filename",
        "owner__username",
        "owner__email",
    )
    readonly_fields = (
        "owner",
        "file",
        "original_filename",
        "stored_filename",
        "mime_type",
        "file_size",
        "status",
        "processing_metadata",
        "error_code",
        "error_message",
        "internal_error_detail",
        "created_at",
        "updated_at",
    )
    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("owner")

    def has_add_permission(self, request):
        return False
