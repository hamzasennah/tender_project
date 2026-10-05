from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import Document
from extraction.models import TextExtractionResult


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
        self.assertContains(response, "Document intelligence workspace")
        self.assertContains(response, "own.pdf")
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
        self.assertTemplateUsed(response, "documents/list.html")

    def test_document_workspace_requires_owner(self):
        own_document = self.create_document(filename="own.pdf")
        other_document = self.create_document(owner=self.other_user, filename="other.pdf")
        TextExtractionResult.objects.create(
            document=own_document,
            status=TextExtractionResult.Status.COMPLETED,
            extracted_text="Tender content",
            has_text=True,
            page_count=2,
            pages_processed=2,
            character_count=14,
        )
        self.login()

        own_response = self.client.get(reverse("web-document-detail", args=[own_document.id]))
        other_response = self.client.get(reverse("web-document-detail", args=[other_document.id]))

        self.assertEqual(own_response.status_code, 200)
        self.assertEqual(other_response.status_code, 404)
        self.assertContains(own_response, "own.pdf")
        self.assertContains(
            own_response,
            reverse("document-prompt-engineering-ask", args=[own_document.id]),
        )
        self.assertContains(own_response, "Prompt Engineering")
        self.assertContains(own_response, "RAG")
        self.assertContains(own_response, "RAPTOR")
        self.assertTemplateUsed(own_response, "documents/detail.html")

    def test_home_redirects_by_authentication_state(self):
        anonymous_response = self.client.get(reverse("home"))
        self.assertEqual(anonymous_response.status_code, 302)
        self.assertEqual(anonymous_response["Location"], reverse("login"))

        self.login()
        authenticated_response = self.client.get(reverse("home"))
        self.assertEqual(authenticated_response.status_code, 302)
        self.assertEqual(authenticated_response["Location"], reverse("dashboard"))
