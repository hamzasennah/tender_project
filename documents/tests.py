import shutil
import tempfile
from io import BytesIO

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from pypdf import PdfWriter
from rest_framework import status
from rest_framework.test import APITestCase

from documents.models import Document


VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog /Pages 2 0 R >>\n"
    b"endobj\n"
    b"2 0 obj\n"
    b"<< /Type /Pages /Count 0 >>\n"
    b"endobj\n"
    b"xref\n"
    b"0 3\n"
    b"0000000000 65535 f \n"
    b"0000000010 00000 n \n"
    b"0000000060 00000 n \n"
    b"trailer\n"
    b"<< /Root 1 0 R >>\n"
    b"startxref\n"
    b"120\n"
    b"%%EOF\n"
)
VALID_PDF_WITH_ALT_CATALOG_DELIMITER_BYTES = VALID_PDF_BYTES.replace(
    b"/Type /Catalog",
    b"/Type\n/Catalog",
)


def encrypted_pdf_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class DocumentAPITests(APITestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_root = tempfile.mkdtemp(prefix="documents-tests-")
        cls._settings = override_settings(
            ALLOWED_HOSTS=["testserver"],
            MEDIA_ROOT=cls._media_root,
            DOCUMENTS_MAX_UPLOAD_SIZE_BYTES=1024 * 1024,
            DOCUMENTS_ALLOWED_UPLOAD_CONTENT_TYPES=["application/pdf"],
        )
        cls._settings.enable()

    @classmethod
    def tearDownClass(cls):
        cls._settings.disable()
        shutil.rmtree(cls._media_root, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="alice",
            email="alice@example.com",
            password="password-1",
        )
        self.other_user = User.objects.create_user(
            username="bob",
            email="bob@example.com",
            password="password-2",
        )
        self.list_url = reverse("document-list")

    def pdf_file(self, name="contract.pdf", content=VALID_PDF_BYTES, content_type="application/pdf"):
        return SimpleUploadedFile(name, content, content_type=content_type)

    def authenticate(self, user=None):
        self.client.force_authenticate(user=user or self.user)

    def upload_document(self, user=None, **file_kwargs):
        self.authenticate(user)
        return self.client.post(
            self.list_url,
            {"file": self.pdf_file(**file_kwargs)},
            format="multipart",
        )

    def test_authenticated_user_can_upload_pdf(self):
        response = self.upload_document(name="..\\..\\contract.pdf")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Document.objects.count(), 1)

        document = Document.objects.get()
        self.assertEqual(document.owner, self.user)
        self.assertEqual(document.status, Document.Status.UPLOADED)
        self.assertEqual(document.original_filename, "contract.pdf")
        self.assertEqual(document.mime_type, "application/pdf")
        self.assertEqual(document.file_size, len(VALID_PDF_BYTES))
        self.assertTrue(document.file.name.startswith(f"documents/user_{self.user.id}/"))
        self.assertTrue(document.file.name.endswith(".pdf"))

        self.assertEqual(response.data["original_filename"], "contract.pdf")
        self.assertEqual(response.data["status"], Document.Status.UPLOADED)
        self.assertIn("download_url", response.data)
        self.assertNotIn("file", response.data)
        self.assertNotIn("owner", response.data)
        self.assertNotIn("stored_filename", response.data)
        self.assertNotIn("internal_error_detail", response.data)

    def test_unauthenticated_requests_are_rejected(self):
        list_response = self.client.get(self.list_url)
        create_response = self.client.post(
            self.list_url,
            {"file": self.pdf_file()},
            format="multipart",
        )

        self.assertEqual(list_response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(create_response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_cannot_access_another_users_document(self):
        create_response = self.upload_document(user=self.user)
        document_id = create_response.data["id"]

        self.authenticate(self.other_user)
        detail_url = reverse("document-detail", args=[document_id])
        download_url = reverse("document-download", args=[document_id])

        self.assertEqual(self.client.get(detail_url).status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(self.client.get(download_url).status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(self.client.delete(detail_url).status_code, status.HTTP_404_NOT_FOUND)

    def test_list_is_scoped_to_authenticated_owner(self):
        own_response = self.upload_document(user=self.user, name="own.pdf")
        other_response = self.upload_document(user=self.other_user, name="other.pdf")

        self.authenticate(self.user)
        response = self.client.get(self.list_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        ids = [item["id"] for item in response.data]
        self.assertEqual(ids, [own_response.data["id"]])
        self.assertNotIn(other_response.data["id"], ids)

    def test_backend_controls_owner_status_and_metadata_fields(self):
        self.authenticate(self.user)
        response = self.client.post(
            self.list_url,
            {
                "file": self.pdf_file(),
                "owner": self.other_user.id,
                "status": Document.Status.COMPLETED,
                "processing_metadata": '{"trusted": true}',
                "internal_error_detail": "client supplied",
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        document = Document.objects.get()
        self.assertEqual(document.owner, self.user)
        self.assertEqual(document.status, Document.Status.UPLOADED)
        self.assertEqual(document.processing_metadata, {})
        self.assertEqual(document.internal_error_detail, "")

    def test_download_requires_owner_and_returns_pdf_bytes(self):
        create_response = self.upload_document()
        download_url = reverse("document-download", args=[create_response.data["id"]])

        response = self.client.get(download_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertEqual(response["X-Content-Type-Options"], "nosniff")
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertEqual(b"".join(response.streaming_content), VALID_PDF_BYTES)

    def test_delete_removes_document_and_stored_file(self):
        create_response = self.upload_document()
        document = Document.objects.get(pk=create_response.data["id"])
        file_name = document.file.name

        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.delete(reverse("document-detail", args=[document.id]))

        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Document.objects.filter(pk=document.id).exists())
        self.assertFalse(document.file.storage.exists(file_name))

    def test_rejects_non_pdf_extension(self):
        response = self.upload_document(name="contract.txt")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Document.objects.count(), 0)

    def test_rejects_long_non_pdf_filename_before_truncation(self):
        response = self.upload_document(name=f"{'a' * 260}.txt")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Document.objects.count(), 0)

    def test_rejects_invalid_declared_content_type(self):
        response = self.upload_document(content_type="text/plain")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Document.objects.count(), 0)

    def test_rejects_empty_file(self):
        response = self.upload_document(content=b"")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Document.objects.count(), 0)

    def test_rejects_oversized_file(self):
        with override_settings(DOCUMENTS_MAX_UPLOAD_SIZE_BYTES=16):
            response = self.upload_document()

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Document.objects.count(), 0)

    def test_rejects_malformed_pdf(self):
        response = self.upload_document(content=b"%PDF-1.4\nmissing required structure\n%%EOF\n")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Document.objects.count(), 0)

    def test_accepts_valid_pdf_without_literal_catalog_spacing(self):
        self.assertNotIn(b"/Type /Catalog", VALID_PDF_WITH_ALT_CATALOG_DELIMITER_BYTES)

        response = self.upload_document(content=VALID_PDF_WITH_ALT_CATALOG_DELIMITER_BYTES)

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Document.objects.count(), 1)

    def test_rejects_encrypted_pdf(self):
        response = self.upload_document(content=encrypted_pdf_bytes())

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Document.objects.count(), 0)
