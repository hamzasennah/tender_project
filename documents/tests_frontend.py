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

    def create_failed_document(self, owner=None, filename="failed.pdf"):
        document = self.create_document(owner=owner, filename=filename)
        document.status = Document.Status.FAILED
        document.save(update_fields=["status"])
        TextExtractionResult.objects.create(
            document=document,
            status=TextExtractionResult.Status.FAILED,
            extracted_text="",
            has_text=False,
            page_count=0,
            pages_processed=0,
            character_count=0,
        )
        return document

    def login(self, user=None):
        self.client.force_login(user or self.user)

    def test_login_page_renders(self):
        response = self.client.get(reverse("login"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Tender Intelligence")
        self.assertContains(response, "Review tender documents with evidence.")
        self.assertNotContains(response, "Ask better questions of complex tender PDFs.")
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
        self.assertContains(response, 'data-upload-panel hidden')
        self.assertContains(response, 'data-open-upload')
        self.assertContains(response, 'data-prepare-after-upload="true"')
        self.assertContains(response, "Preparing your document")
        self.assertNotContains(response, "-&gt;")
        self.assertTemplateUsed(response, "documents/list.html")

    def test_document_library_failed_row_exposes_retry(self):
        document = self.create_failed_document(filename="failed.pdf")
        self.login()

        response = self.client.get(reverse("web-document-list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "failed.pdf")
        self.assertContains(response, 'data-retry-document')
        self.assertContains(response, f'data-document-id="{document.id}"')

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
        self.assertContains(own_response, "evidence from the document")
        self.assertContains(own_response, "Ready")
        self.assertContains(own_response, 'data-document-url')
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

    def test_failed_document_ui_uses_retry_without_jargon(self):
        document = self.create_failed_document(filename="failed.pdf")
        self.login()

        response = self.client.get(reverse("web-document-detail", args=[document.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "We couldn't prepare this document.")
        self.assertContains(response, "Try again in a moment.")
        self.assertContains(response, 'data-retry-document')
        self.assertNotContains(response, "provider")
        self.assertNotContains(response, "Gemini")
        self.assertNotContains(response, "429")
        self.assertNotContains(response, "503")

    def test_ready_state_uses_business_language(self):
        document = self.create_prepared_document(filename="ready.pdf")
        self.login()

        response = self.client.get(reverse("web-document-detail", args=[document.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ready")
        self.assertContains(response, "This document is ready for grounded questions.")
        self.assertContains(response, "Research comparison")

    def test_frontend_static_translates_technical_errors(self):
        with open("documents/static/js/analysis.js", encoding="utf-8") as analysis_script:
            analysis_contents = analysis_script.read()
        with open("documents/static/js/documents.js", encoding="utf-8") as documents_script:
            documents_contents = documents_script.read()
        combined = analysis_contents + documents_contents

        self.assertIn("We couldn't prepare this document.", documents_contents)
        self.assertIn("Open in document", analysis_contents)
        self.assertNotIn("Gemini", combined)
        self.assertNotIn("HTTP 429", combined)
        self.assertNotIn("HTTP 503", combined)
        self.assertNotIn("provider", combined)
        self.assertNotIn("showToast(friendlyProcessingError", documents_contents)

    def test_analysis_static_uses_answer_supporting_evidence(self):
        with open("documents/static/js/analysis.js", encoding="utf-8") as analysis_script:
            analysis_contents = analysis_script.read()

        self.assertIn("supporting_evidence", analysis_contents)
        self.assertIn("evidenceFromAnswer", analysis_contents)
        self.assertNotIn("evidenceFromSearch", analysis_contents)
        self.assertNotIn("workspace.dataset.searchUrl", analysis_contents)
        self.assertIn("#page=", analysis_contents)
        self.assertIn("supported_claims", analysis_contents)
        self.assertIn("Supports:", analysis_contents)

    def test_home_redirects_by_authentication_state(self):
        anonymous_response = self.client.get(reverse("home"))
        self.assertEqual(anonymous_response.status_code, 302)
        self.assertEqual(anonymous_response["Location"], reverse("login"))

        self.login()
        authenticated_response = self.client.get(reverse("home"))
        self.assertEqual(authenticated_response.status_code, 302)
        self.assertEqual(authenticated_response["Location"], reverse("dashboard"))
