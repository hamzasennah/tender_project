from rest_framework import serializers

from extraction.models import ChunkEmbedding, TextChunk, TextExtractionResult


class TextExtractionResultSerializer(serializers.ModelSerializer):
    document_id = serializers.IntegerField(source="document.id", read_only=True)

    class Meta:
        model = TextExtractionResult
        fields = [
            "id",
            "document_id",
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
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


class TextChunkSerializer(serializers.ModelSerializer):
    document_id = serializers.IntegerField(source="document.id", read_only=True)
    extraction_result_id = serializers.IntegerField(
        source="extraction_result.id",
        read_only=True,
    )

    class Meta:
        model = TextChunk
        fields = [
            "id",
            "document_id",
            "extraction_result_id",
            "chunk_index",
            "text",
            "character_start",
            "character_end",
            "text_length",
            "text_sha256",
            "metadata",
            "created_at",
        ]
        read_only_fields = fields


class ChunkEmbeddingStatusSerializer(serializers.ModelSerializer):
    chunk_id = serializers.IntegerField(source="chunk.id", read_only=True)
    chunk_index = serializers.IntegerField(source="chunk.chunk_index", read_only=True)

    class Meta:
        model = ChunkEmbedding
        fields = [
            "id",
            "chunk_id",
            "chunk_index",
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
        ]
        read_only_fields = fields
