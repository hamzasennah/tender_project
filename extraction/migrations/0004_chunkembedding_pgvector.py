# Generated manually for pgvector-backed embedding persistence.

import math

from django.db import migrations
from pgvector.django import VectorField


EMBEDDING_DIMENSION = 768


def _valid_vector(vector):
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
        return False
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector)


def backfill_embedding_vector(apps, schema_editor):
    ChunkEmbedding = apps.get_model("extraction", "ChunkEmbedding")
    completed_embeddings = ChunkEmbedding.objects.filter(status="completed")

    for embedding in completed_embeddings.iterator():
        if not _valid_vector(embedding.vector):
            raise ValueError(
                "Cannot backfill malformed completed embedding vector "
                f"for ChunkEmbedding id={embedding.id}"
            )

        embedding.embedding_vector = [float(value) for value in embedding.vector]
        embedding.save(update_fields=["embedding_vector"])


def clear_embedding_vector(apps, schema_editor):
    ChunkEmbedding = apps.get_model("extraction", "ChunkEmbedding")
    ChunkEmbedding.objects.update(embedding_vector=None)


class Migration(migrations.Migration):

    dependencies = [
        ("extraction", "0003_chunkembedding"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                ) THEN
                    RAISE EXCEPTION
                        'pgvector extension "vector" must be enabled before applying extraction.0004_chunkembedding_pgvector';
                END IF;
            END
            $$;
            """,
            reverse_sql=migrations.RunSQL.noop,
        ),
        migrations.AddField(
            model_name="chunkembedding",
            name="embedding_vector",
            field=VectorField(blank=True, dimensions=768, null=True),
        ),
        migrations.RunPython(backfill_embedding_vector, clear_embedding_vector),
    ]
