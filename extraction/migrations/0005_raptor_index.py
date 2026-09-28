# Generated manually for RAPTOR hierarchical index persistence.

import django.db.models.deletion
from django.db import migrations, models
from pgvector.django import VectorField


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0001_initial"),
        ("extraction", "0004_chunkembedding_pgvector"),
    ]

    operations = [
        migrations.CreateModel(
            name="RaptorIndex",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("status", models.CharField(choices=[("pending", "Pending"), ("processing", "Processing"), ("completed", "Completed"), ("failed", "Failed")], db_index=True, default="pending", max_length=32)),
                ("build_version", models.CharField(blank=True, max_length=64)),
                ("provider", models.CharField(blank=True, max_length=64)),
                ("embedding_model", models.CharField(blank=True, max_length=128)),
                ("embedding_dimension", models.PositiveIntegerField(default=0)),
                ("level_count", models.PositiveIntegerField(default=0)),
                ("leaf_count", models.PositiveIntegerField(default=0)),
                ("summary_node_count", models.PositiveIntegerField(default=0)),
                ("total_node_count", models.PositiveIntegerField(default=0)),
                ("nodes_per_level", models.JSONField(blank=True, default=dict)),
                ("clusters_per_level", models.JSONField(blank=True, default=dict)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("error_code", models.CharField(blank=True, max_length=64)),
                ("error_message", models.CharField(blank=True, max_length=255)),
                ("internal_error_detail", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "document",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="raptor_index",
                        to="documents.document",
                    ),
                ),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
        migrations.CreateModel(
            name="RaptorNode",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("level", models.PositiveIntegerField()),
                ("node_index", models.PositiveIntegerField()),
                ("text", models.TextField(blank=True)),
                ("text_sha256", models.CharField(max_length=64)),
                ("embedding_vector", VectorField(blank=True, dimensions=768, null=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                (
                    "document",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="raptor_nodes",
                        to="documents.document",
                    ),
                ),
                (
                    "index",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="nodes",
                        to="extraction.raptorindex",
                    ),
                ),
                (
                    "source_chunk",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="raptor_leaf_nodes",
                        to="extraction.textchunk",
                    ),
                ),
            ],
            options={
                "ordering": ["level", "node_index"],
            },
        ),
        migrations.CreateModel(
            name="RaptorNodeChild",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("child_rank", models.PositiveIntegerField()),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                (
                    "child",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="parent_links",
                        to="extraction.raptornode",
                    ),
                ),
                (
                    "parent",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="child_links",
                        to="extraction.raptornode",
                    ),
                ),
            ],
            options={
                "ordering": ["parent_id", "child_rank"],
            },
        ),
        migrations.AddIndex(
            model_name="raptorindex",
            index=models.Index(fields=["status", "-updated_at"], name="raptor_idx_status_updated_idx"),
        ),
        migrations.AddIndex(
            model_name="raptornode",
            index=models.Index(fields=["document", "level", "node_index"], name="raptor_node_doc_level_idx"),
        ),
        migrations.AddIndex(
            model_name="raptornode",
            index=models.Index(fields=["index", "level", "node_index"], name="raptor_node_index_level_idx"),
        ),
        migrations.AddIndex(
            model_name="raptornodechild",
            index=models.Index(fields=["parent", "child_rank"], name="raptor_child_parent_rank_idx"),
        ),
        migrations.AddIndex(
            model_name="raptornodechild",
            index=models.Index(fields=["child"], name="raptor_child_child_idx"),
        ),
        migrations.AddConstraint(
            model_name="raptorindex",
            constraint=models.CheckConstraint(condition=models.Q(("status__in", ["pending", "processing", "completed", "failed"])), name="raptor_idx_valid_status"),
        ),
        migrations.AddConstraint(
            model_name="raptorindex",
            constraint=models.CheckConstraint(condition=models.Q(("embedding_dimension__gte", 0)), name="raptor_idx_embedding_dimension_nonnegative"),
        ),
        migrations.AddConstraint(
            model_name="raptornode",
            constraint=models.UniqueConstraint(fields=("index", "level", "node_index"), name="raptor_node_unique_level_index"),
        ),
        migrations.AddConstraint(
            model_name="raptornode",
            constraint=models.CheckConstraint(condition=models.Q(("level__gte", 0)), name="raptor_node_level_nonnegative"),
        ),
        migrations.AddConstraint(
            model_name="raptornodechild",
            constraint=models.UniqueConstraint(fields=("parent", "child"), name="raptor_child_unique_parent_child"),
        ),
        migrations.AddConstraint(
            model_name="raptornodechild",
            constraint=models.UniqueConstraint(fields=("parent", "child_rank"), name="raptor_child_unique_parent_rank"),
        ),
    ]
