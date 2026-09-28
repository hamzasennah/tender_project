from django.db import models
from django.db.models import Q
from pgvector.django import VectorField

from documents.models import Document


class TextExtractionResult(models.Model):
    class Status(models.TextChoices):
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    document = models.OneToOneField(
        Document,
        on_delete=models.CASCADE,
        related_name="text_extraction",
    )
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.PROCESSING,
        db_index=True,
    )
    extracted_text = models.TextField(blank=True)
    has_text = models.BooleanField(default=False)
    page_count = models.PositiveIntegerField(default=0)
    pages_processed = models.PositiveIntegerField(default=0)
    character_count = models.PositiveIntegerField(default=0)
    text_sha256 = models.CharField(max_length=64, blank=True)
    extraction_metadata = models.JSONField(default=dict, blank=True)
    error_code = models.CharField(max_length=64, blank=True)
    error_message = models.CharField(max_length=255, blank=True)
    internal_error_detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["status", "-updated_at"], name="text_ext_status_updated_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=Q(
                    status__in=[
                        "processing",
                        "completed",
                        "failed",
                    ]
                ),
                name="text_ext_valid_status",
            ),
            models.CheckConstraint(
                condition=Q(character_count__gte=0),
                name="text_ext_character_count_nonnegative",
            ),
            models.CheckConstraint(
                condition=Q(page_count__gte=0),
                name="text_ext_page_count_nonnegative",
            ),
            models.CheckConstraint(
                condition=Q(pages_processed__gte=0),
                name="text_ext_pages_processed_nonnegative",
            ),
        ]

    def __str__(self):
        return f"Text extraction for document {self.document_id}"


class TextChunk(models.Model):
    extraction_result = models.ForeignKey(
        TextExtractionResult,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="text_chunks",
    )
    chunk_index = models.PositiveIntegerField()
    text = models.TextField()
    character_start = models.PositiveIntegerField()
    character_end = models.PositiveIntegerField()
    text_length = models.PositiveIntegerField()
    text_sha256 = models.CharField(max_length=64)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["chunk_index"]
        indexes = [
            models.Index(
                fields=["document", "chunk_index"],
                name="text_chunk_document_idx",
            ),
            models.Index(
                fields=["extraction_result", "chunk_index"],
                name="text_chunk_extraction_idx",
            ),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["extraction_result", "chunk_index"],
                name="text_chunk_unique_index_per_extraction",
            ),
            models.CheckConstraint(
                condition=Q(character_end__gte=models.F("character_start")),
                name="text_chunk_valid_character_range",
            ),
            models.CheckConstraint(
                condition=Q(text_length__gte=0),
                name="text_chunk_text_length_nonnegative",
            ),
        ]

    def __str__(self):
        return f"Chunk {self.chunk_index} for extraction {self.extraction_result_id}"


class ChunkEmbedding(models.Model):
    class Status(models.TextChoices):
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    chunk = models.OneToOneField(
        TextChunk,
        on_delete=models.CASCADE,
        related_name="embedding",
    )
    provider = models.CharField(max_length=64)
    model = models.CharField(max_length=128)
    dimension = models.PositiveIntegerField()
    vector = models.JSONField(default=list, blank=True)
    embedding_vector = VectorField(dimensions=768, null=True, blank=True)
    chunk_sha256 = models.CharField(max_length=64)
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.COMPLETED,
        db_index=True,
    )
    error_code = models.CharField(max_length=64, blank=True)
    error_message = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["chunk__chunk_index"]
        indexes = [
            models.Index(fields=["provider", "model"], name="chunk_emb_provider_model_idx"),
            models.Index(fields=["status", "-updated_at"], name="chunk_emb_status_updated_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=Q(dimension__gt=0),
                name="chunk_emb_dimension_positive",
            ),
            models.CheckConstraint(
                condition=Q(status__in=["completed", "failed"]),
                name="chunk_emb_valid_status",
            ),
        ]

    def __str__(self):
        return f"Embedding for chunk {self.chunk_id}"


class RaptorIndex(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    document = models.OneToOneField(
        Document,
        on_delete=models.CASCADE,
        related_name="raptor_index",
    )
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    build_version = models.CharField(max_length=64, blank=True)
    provider = models.CharField(max_length=64, blank=True)
    embedding_model = models.CharField(max_length=128, blank=True)
    embedding_dimension = models.PositiveIntegerField(default=0)
    level_count = models.PositiveIntegerField(default=0)
    leaf_count = models.PositiveIntegerField(default=0)
    summary_node_count = models.PositiveIntegerField(default=0)
    total_node_count = models.PositiveIntegerField(default=0)
    nodes_per_level = models.JSONField(default=dict, blank=True)
    clusters_per_level = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    error_code = models.CharField(max_length=64, blank=True)
    error_message = models.CharField(max_length=255, blank=True)
    internal_error_detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["status", "-updated_at"], name="raptor_idx_status_updated_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=Q(status__in=["pending", "processing", "completed", "failed"]),
                name="raptor_idx_valid_status",
            ),
            models.CheckConstraint(
                condition=Q(embedding_dimension__gte=0),
                name="raptor_idx_embedding_dimension_nonnegative",
            ),
        ]

    def __str__(self):
        return f"RAPTOR index for document {self.document_id}"


class RaptorNode(models.Model):
    index = models.ForeignKey(
        RaptorIndex,
        on_delete=models.CASCADE,
        related_name="nodes",
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="raptor_nodes",
    )
    source_chunk = models.ForeignKey(
        TextChunk,
        on_delete=models.CASCADE,
        related_name="raptor_leaf_nodes",
        null=True,
        blank=True,
    )
    level = models.PositiveIntegerField()
    node_index = models.PositiveIntegerField()
    text = models.TextField(blank=True)
    text_sha256 = models.CharField(max_length=64)
    embedding_vector = VectorField(dimensions=768, null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["level", "node_index"]
        indexes = [
            models.Index(fields=["document", "level", "node_index"], name="raptor_node_doc_level_idx"),
            models.Index(fields=["index", "level", "node_index"], name="raptor_node_index_level_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["index", "level", "node_index"],
                name="raptor_node_unique_level_index",
            ),
            models.CheckConstraint(
                condition=Q(level__gte=0),
                name="raptor_node_level_nonnegative",
            ),
        ]

    def __str__(self):
        return f"RAPTOR node {self.node_index} at level {self.level}"


class RaptorNodeChild(models.Model):
    parent = models.ForeignKey(
        RaptorNode,
        on_delete=models.CASCADE,
        related_name="child_links",
    )
    child = models.ForeignKey(
        RaptorNode,
        on_delete=models.CASCADE,
        related_name="parent_links",
    )
    child_rank = models.PositiveIntegerField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["parent_id", "child_rank"]
        indexes = [
            models.Index(fields=["parent", "child_rank"], name="raptor_child_parent_rank_idx"),
            models.Index(fields=["child"], name="raptor_child_child_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["parent", "child"],
                name="raptor_child_unique_parent_child",
            ),
            models.UniqueConstraint(
                fields=["parent", "child_rank"],
                name="raptor_child_unique_parent_rank",
            ),
        ]

    def __str__(self):
        return f"RAPTOR child {self.child_id} for parent {self.parent_id}"
