# Generated manually for chunk embedding persistence before pgvector.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("extraction", "0002_textchunk"),
    ]

    operations = [
        migrations.CreateModel(
            name="ChunkEmbedding",
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
                ("provider", models.CharField(max_length=64)),
                ("model", models.CharField(max_length=128)),
                ("dimension", models.PositiveIntegerField()),
                ("vector", models.JSONField(blank=True, default=list)),
                ("chunk_sha256", models.CharField(max_length=64)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                        ],
                        db_index=True,
                        default="completed",
                        max_length=32,
                    ),
                ),
                ("error_code", models.CharField(blank=True, max_length=64)),
                ("error_message", models.CharField(blank=True, max_length=255)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "chunk",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="embedding",
                        to="extraction.textchunk",
                    ),
                ),
            ],
            options={
                "ordering": ["chunk__chunk_index"],
            },
        ),
        migrations.AddIndex(
            model_name="chunkembedding",
            index=models.Index(
                fields=["provider", "model"],
                name="chunk_emb_provider_model_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="chunkembedding",
            index=models.Index(
                fields=["status", "-updated_at"],
                name="chunk_emb_status_updated_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="chunkembedding",
            constraint=models.CheckConstraint(
                condition=models.Q(("dimension__gt", 0)),
                name="chunk_emb_dimension_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="chunkembedding",
            constraint=models.CheckConstraint(
                condition=models.Q(("status__in", ["completed", "failed"])),
                name="chunk_emb_valid_status",
            ),
        ),
    ]
