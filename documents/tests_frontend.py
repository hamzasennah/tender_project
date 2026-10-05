from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import Document
from extraction.models import ChunkEmbedding, TextChunk, TextExtractionResult


@override_settings(ALLOWED_HOSTS=["testserver"])
class FrontendViewTests(TestCase):
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

    def create_document(self, owner=None, filename="tender.pdf"):
        owner = owner or self.user
        return Document.objects.create(
            owner=owner,
            file=f"documents/user_{owner.id}/{filename}",
            original_filename=filename,
            stored_filename=filename,
            mime_type="application/pdf",
            file_size=2048,
            status=Document.Status.UPLOADED,
        )

    def create_prepared_document(self, owner=None, filename="prepared.pdf"):
        document = self.create_document(owner=owner, filename=filename)
        extraction = TextExtractionResult.objects.create(
            document=document,
            status=TextExtractionResult.Status.COMPLETED,
            extracted_text="Tender content with deadline and guarantee details.",
            has_text=True,
            page_count=2,
            pages_processed=2,
            character_count=51,
        )
        chunk = TextChunk.objects.create(
            document=document,
            extraction_result=extraction,
            chunk_index=0,
            text="Tender content with deadline and guarantee details.",
            character_start=0,
            character_end=51,
            text_length=51,
            text_sha256="a" * 64,
        )
        ChunkEmbedding.objects.create(
            chunk=chunk,
            provider="gemini",
            model="gemini-embedding-2",
            dimension=768,
            vector=[0.0] * 768,
            chunk_sha256=chunk.text_sha256,
            status=ChunkEmbedding.Status.COMPLETED,
        )
        return document

    def login(self, user=None):
        self.client.force_login(user or self.user)

    def test_login_page_renders(self):
        response = self.client.get(reverse("login"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Tender Intelligence")
        self.assertContains(response, "Sign in")
        self.assertTemplateUsed(response, "registration/login.html")

    def test_authenticated_user_is_redirected_from_login(self):
        self.login()

        response = self.client.get(reverse("login"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("dashboard"))

    def test_dashboard_requires_authentication(self):
        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response["Location"])
        self.assertIn("next=/dashboard/", response["Location"])

    def test_dashboard_renders_for_authenticated_user(self):
        self.login()
        self.create_document(filename="own.pdf")

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your documents")
        self.assertContains(response, "own.pdf")
        self.assertNotContains(response, "Document intelligence workspace")
        self.assertTemplateUsed(response, "dashboard/index.html")

    def test_confirm_modal_is_hidden_by_default(self):
        self.login()

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-confirm-modal hidden aria-hidden="true"')

    def test_document_library_is_scoped_to_owner(self):
        self.create_document(filename="own.pdf")
        self.create_document(owner=self.other_user, filename="other.pdf")
        self.login()

        response = self.client.get(reverse("web-document-list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "own.pdf")
        self.assertNotContains(response, "other.pdf")
        self.assertContains(response, reverse("document-list"))
        self.assertContains(response, 'data-prepare-after-upload="true"')
        self.assertContains(response, "Preparing your document")
        self.assertTemplateUsed(response, "documents/list.html")

    def test_document_workspace_requires_owner(self):
        own_document = self.create_prepared_document(filename="own.pdf")
        other_document = self.create_document(owner=self.other_user, filename="other.pdf")
        self.login()

        own_response = self.client.get(reverse("web-document-detail", args=[own_document.id]))
        other_response = self.client.get(reverse("web-document-detail", args=[other_document.id]))

        self.assertEqual(own_response.status_code, 200)
        self.assertEqual(other_response.status_code, 404)
        self.assertContains(own_response, "own.pdf")
        self.assertContains(
            own_response,
            reverse("document-rag-ask", args=[own_document.id]),
        )
        self.assertContains(own_response, "Ask this document")
        self.assertContains(own_response, "evidence from the PDF")
        self.assertContains(own_response, "Ready")
        self.assertTemplateUsed(own_response, "documents/detail.html")

    def test_normal_document_ui_hides_technical_pipeline_actions(self):
        document = self.create_document(filename="preparing.pdf")
        self.login()

        response = self.client.get(reverse("web-document-detail", args=[document.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Preparing your document")
        self.assertNotContains(response, "Generate embeddings")
        self.assertNotContains(response, "Build RAPTOR")
        self.assertNotContains(response, "Document readiness")

    def test_ready_state_uses_business_language(self):
        document = self.create_prepared_document(filename="ready.pdf")
        self.login()

        response = self.client.get(reverse("web-document-detail", args=[document.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ready")
        self.assertContains(response, "This document is ready for grounded questions.")
        self.assertContains(response, "Research comparison")

    def test_frontend_static_translates_provider_errors(self):
        with open("documents/static/js/analysis.js", encoding="utf-8") as script:
            contents = script.read()

        self.assertIn("Temporary processing issue", contents)
        self.assertNotIn("Gemini 429", contents)

    def test_home_redirects_by_authentication_state(self):
        anonymous_response = self.client.get(reverse("home"))
        self.assertEqual(anonymous_response.status_code, 302)
        self.assertEqual(anonymous_response["Location"], reverse("login"))

        self.login()
        authenticated_response = self.client.get(reverse("home"))
        self.assertEqual(authenticated_response.status_code, 302)
        self.assertEqual(authenticated_response["Location"], reverse("dashboard"))
