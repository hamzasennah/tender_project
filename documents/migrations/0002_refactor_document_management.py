# Generated manually for the production-oriented document management refactor.

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models

import documents.models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("documents", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="document",
            old_name="name",
            new_name="original_filename",
        ),
        migrations.RenameField(
            model_name="document",
            old_name="uploaded_at",
            new_name="created_at",
        ),
        migrations.AlterModelOptions(
            name="document",
            options={"ordering": ["-created_at"]},
        ),
        migrations.AlterField(
            model_name="document",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, db_index=True),
        ),
        migrations.AddField(
            model_name="document",
            name="owner",
            field=models.ForeignKey(
                db_index=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="documents",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="document",
            name="stored_filename",
            field=models.CharField(db_index=True, editable=False, max_length=128),
        ),
        migrations.AddField(
            model_name="document",
            name="mime_type",
            field=models.CharField(max_length=100),
        ),
        migrations.AddField(
            model_name="document",
            name="file_size",
            field=models.PositiveBigIntegerField(),
        ),
        migrations.AddField(
            model_name="document",
            name="processing_metadata",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="document",
            name="error_code",
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name="document",
            name="error_message",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="document",
            name="internal_error_detail",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="document",
            name="updated_at",
            field=models.DateTimeField(
                auto_now=True,
                default=django.utils.timezone.now,
            ),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="document",
            name="file",
            field=models.FileField(
                max_length=512,
                upload_to=documents.models.document_upload_path,
            ),
        ),
        migrations.AlterField(
            model_name="document",
            name="status",
            field=models.CharField(
                choices=[
                    ("uploaded", "Uploaded"),
                    ("processing", "Processing"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                ],
                db_index=True,
                default="uploaded",
                max_length=32,
            ),
        ),
        migrations.AddIndex(
            model_name="document",
            index=models.Index(
                fields=["owner", "-created_at"],
                name="documents_owner_created_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="document",
            index=models.Index(
                fields=["owner", "status"],
                name="documents_owner_status_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="document",
            constraint=models.CheckConstraint(
                condition=models.Q(file_size__gt=0),
                name="documents_file_size_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="document",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    status__in=["uploaded", "processing", "completed", "failed"]
                ),
                name="documents_valid_status",
            ),
        ),
    ]
