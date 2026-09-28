from django.urls import path

from extraction.views import (
    DocumentChunkEmbeddingView,
    DocumentPromptEngineeringAskView,
    DocumentRAGAskView,
    DocumentRaptorAskView,
    DocumentRaptorIndexView,
    DocumentSemanticSearchView,
    DocumentTextChunkingView,
    DocumentTextExtractionView,
)

urlpatterns = [
    path(
        "documents/<int:document_id>/extraction/",
        DocumentTextExtractionView.as_view(),
        name="document-text-extraction",
    ),
    path(
        "documents/<int:document_id>/chunks/",
        DocumentTextChunkingView.as_view(),
        name="document-text-chunks",
    ),
    path(
        "documents/<int:document_id>/embeddings/",
        DocumentChunkEmbeddingView.as_view(),
        name="document-chunk-embeddings",
    ),
    path(
        "documents/<int:document_id>/search/",
        DocumentSemanticSearchView.as_view(),
        name="document-semantic-search",
    ),
    path(
        "documents/<int:document_id>/ask/",
        DocumentRAGAskView.as_view(),
        name="document-rag-ask",
    ),
    path(
        "documents/<int:document_id>/prompt-engineering/ask/",
        DocumentPromptEngineeringAskView.as_view(),
        name="document-prompt-engineering-ask",
    ),
    path(
        "documents/<int:document_id>/raptor/",
        DocumentRaptorIndexView.as_view(),
        name="document-raptor-index",
    ),
    path(
        "documents/<int:document_id>/raptor/ask/",
        DocumentRaptorAskView.as_view(),
        name="document-raptor-ask",
    ),
]
