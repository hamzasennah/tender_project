from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from documents.models import Document
from extraction.models import TextExtractionResult
from extraction.serializers import (
    ChunkEmbeddingStatusSerializer,
    TextChunkSerializer,
    TextExtractionResultSerializer,
)
from extraction.services.embeddings import (
    EmbeddingServiceError,
    embed_document_chunks,
    get_document_embeddings_status,
)
from extraction.services.pdf_text_extraction import extract_document_text
from extraction.services.prompt_engineering import (
    PromptEngineeringError,
    answer_prompt_engineering_question,
)
from extraction.services.rag import (
    RAGError,
    answer_document_question,
)
from extraction.services.raptor import (
    RaptorError,
    answer_raptor_question,
    build_raptor_tree,
    get_raptor_index_status,
)
from extraction.services.semantic_search import (
    SemanticSearchError,
    semantic_search_document,
)
from extraction.services.text_chunking import (
    TextChunkingError,
    chunk_document_text,
    get_document_text_chunks,
)


class DocumentTextExtractionView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def get(self, request, document_id):
        document = self.get_document(request, document_id)
        extraction_result = get_object_or_404(
            TextExtractionResult.objects.select_related("document"),
            document=document,
        )
        serializer = TextExtractionResultSerializer(extraction_result)
        return Response(serializer.data)

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        extraction_result = extract_document_text(document)
        serializer = TextExtractionResultSerializer(extraction_result)
        response_status = (
            status.HTTP_422_UNPROCESSABLE_ENTITY
            if extraction_result.status == TextExtractionResult.Status.FAILED
            else status.HTTP_200_OK
        )
        return Response(serializer.data, status=response_status)


class DocumentTextChunkingView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def build_response(self, payload, response_status=status.HTTP_200_OK):
        serializer = TextChunkSerializer(payload["chunks"], many=True)
        return Response(
            {
                "document_id": payload["document_id"],
                "extraction_result_id": payload["extraction_result_id"],
                "chunk_count": payload["chunk_count"],
                "chunks": serializer.data,
                "chunking_metadata": payload["chunking_metadata"],
            },
            status=response_status,
        )

    def error_response(self, exc):
        return Response(
            {
                "error_code": exc.code,
                "error_message": exc.public_message,
            },
            status=exc.response_status,
        )

    def get(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = get_document_text_chunks(document)
        except TextChunkingError as exc:
            return self.error_response(exc)
        return self.build_response(payload)

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = chunk_document_text(document)
        except TextChunkingError as exc:
            return self.error_response(exc)
        return self.build_response(payload)


class DocumentChunkEmbeddingView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def build_response(self, payload, response_status=status.HTTP_200_OK):
        serializer = ChunkEmbeddingStatusSerializer(payload["embeddings"], many=True)
        return Response(
            {
                "document_id": payload["document_id"],
                "chunk_count": payload["chunk_count"],
                "embedding_count": payload["embedding_count"],
                "embeddings": serializer.data,
                "embedding_metadata": payload["embedding_metadata"],
            },
            status=response_status,
        )

    def error_response(self, exc):
        return Response(
            {
                "error_code": exc.code,
                "error_message": exc.public_message,
            },
            status=exc.response_status,
        )

    def get(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = get_document_embeddings_status(document)
        except EmbeddingServiceError as exc:
            return self.error_response(exc)
        return self.build_response(payload)

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = embed_document_chunks(document)
        except EmbeddingServiceError as exc:
            return self.error_response(exc)
        return self.build_response(payload)


class DocumentSemanticSearchView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def error_response(self, exc):
        return Response(
            {
                "error_code": exc.code,
                "error_message": exc.public_message,
            },
            status=exc.response_status,
        )

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = semantic_search_document(
                document,
                request.data.get("query", ""),
                request.data.get("top_k"),
            )
        except SemanticSearchError as exc:
            return self.error_response(exc)
        return Response(payload, status=status.HTTP_200_OK)


class DocumentRAGAskView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def error_response(self, exc):
        return Response(
            {
                "error_code": exc.code,
                "error_message": exc.public_message,
            },
            status=exc.response_status,
        )

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = answer_document_question(
                document,
                request.data.get("question", ""),
                request.data.get("top_k"),
            )
        except RAGError as exc:
            return self.error_response(exc)
        return Response(payload, status=status.HTTP_200_OK)


class DocumentPromptEngineeringAskView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def error_response(self, exc):
        return Response(
            {
                "error_code": exc.code,
                "error_message": exc.public_message,
            },
            status=exc.response_status,
        )

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = answer_prompt_engineering_question(
                document,
                request.data.get("question", ""),
            )
        except PromptEngineeringError as exc:
            return self.error_response(exc)
        return Response(payload, status=status.HTTP_200_OK)


class DocumentRaptorIndexView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def error_response(self, exc):
        return Response(
            {
                "error_code": exc.code,
                "error_message": exc.public_message,
            },
            status=exc.response_status,
        )

    def get(self, request, document_id):
        document = self.get_document(request, document_id)
        return Response(get_raptor_index_status(document), status=status.HTTP_200_OK)

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = build_raptor_tree(document)
        except RaptorError as exc:
            return self.error_response(exc)
        return Response(payload, status=status.HTTP_200_OK)


class DocumentRaptorAskView(APIView):
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get_document(self, request, document_id):
        return get_object_or_404(
            Document.objects.filter(owner=request.user),
            pk=document_id,
        )

    def error_response(self, exc):
        return Response(
            {
                "error_code": exc.code,
                "error_message": exc.public_message,
            },
            status=exc.response_status,
        )

    def post(self, request, document_id):
        document = self.get_document(request, document_id)
        try:
            payload = answer_raptor_question(
                document,
                request.data.get("question", ""),
                request.data.get("top_k"),
            )
        except RaptorError as exc:
            return self.error_response(exc)
        return Response(payload, status=status.HTTP_200_OK)
