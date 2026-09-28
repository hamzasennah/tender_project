from django.contrib import admin

from extraction.models import ChunkEmbedding, TextChunk, TextExtractionResult


@admin.register(TextExtractionResult)
class TextExtractionResultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "document",
        "status",
        "has_text",
        "page_count",
        "character_count",
        "updated_at",
    )
    list_filter = ("status", "has_text", "created_at", "updated_at")
    search_fields = (
        "document__original_filename",
        "document__owner__username",
        "document__owner__email",
        "error_code",
    )
    readonly_fields = (
        "document",
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
        "created_at",
        "updated_at",
    )
    ordering = ("-updated_at",)

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("document", "document__owner")

    def has_add_permission(self, request):
        return False


@admin.register(TextChunk)
class TextChunkAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "document",
        "extraction_result",
        "chunk_index",
        "text_length",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = (
        "document__original_filename",
        "document__owner__username",
        "document__owner__email",
        "text_sha256",
    )
    readonly_fields = (
        "document",
        "extraction_result",
        "chunk_index",
        "text",
        "character_start",
        "character_end",
        "text_length",
        "text_sha256",
        "metadata",
        "created_at",
    )
    ordering = ("document", "chunk_index")

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related("document", "document__owner", "extraction_result")
        )

    def has_add_permission(self, request):
        return False


@admin.register(ChunkEmbedding)
class ChunkEmbeddingAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "chunk",
        "provider",
        "model",
        "dimension",
        "status",
        "updated_at",
    )
    list_filter = ("provider", "model", "status", "created_at", "updated_at")
    search_fields = (
        "chunk__document__original_filename",
        "chunk__document__owner__username",
        "chunk_sha256",
        "error_code",
    )
    readonly_fields = (
        "chunk",
        "provider",
        "model",
        "dimension",
        "chunk_sha256",
        "status",
        "error_code",
        "error_message",
        "metadata",
        "created_at",
        "updated_at",
    )
    exclude = ("vector", "embedding_vector")
    ordering = ("chunk__document", "chunk__chunk_index")

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related("chunk", "chunk__document", "chunk__document__owner")
        )

    def has_add_permission(self, request):
        return False
