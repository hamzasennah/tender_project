# Generated manually for deterministic text chunking.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0002_refactor_document_management"),
        ("extraction", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="TextChunk",
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
                ("chunk_index", models.PositiveIntegerField()),
                ("text", models.TextField()),
                ("character_start", models.PositiveIntegerField()),
                ("character_end", models.PositiveIntegerField()),
                ("text_length", models.PositiveIntegerField()),
                ("text_sha256", models.CharField(max_length=64)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                (
                    "document",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="text_chunks",
                        to="documents.document",
                    ),
                ),
                (
                    "extraction_result",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="chunks",
                        to="extraction.textextractionresult",
                    ),
                ),
            ],
            options={
                "ordering": ["chunk_index"],
            },
        ),
        migrations.AddIndex(
            model_name="textchunk",
            index=models.Index(
                fields=["document", "chunk_index"],
                name="text_chunk_document_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="textchunk",
            index=models.Index(
                fields=["extraction_result", "chunk_index"],
                name="text_chunk_extraction_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="textchunk",
            constraint=models.UniqueConstraint(
                fields=("extraction_result", "chunk_index"),
                name="text_chunk_unique_index_per_extraction",
            ),
        ),
        migrations.AddConstraint(
            model_name="textchunk",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    ("character_end__gte", models.F("character_start"))
                ),
                name="text_chunk_valid_character_range",
            ),
        ),
        migrations.AddConstraint(
            model_name="textchunk",
            constraint=models.CheckConstraint(
                condition=models.Q(("text_length__gte", 0)),
                name="text_chunk_text_length_nonnegative",
            ),
        ),
    ]
