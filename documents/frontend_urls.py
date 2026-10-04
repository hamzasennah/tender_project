from django.urls import path

from documents import frontend_views

urlpatterns = [
    path("dashboard/", frontend_views.dashboard, name="dashboard"),
    path("documents/", frontend_views.document_library, name="web-document-list"),
    path(
        "documents/<int:document_id>/",
        frontend_views.document_workspace,
        name="web-document-detail",
    ),
]
