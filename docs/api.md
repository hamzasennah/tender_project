# API

L'API est exposee par Django REST Framework. Les endpoints applicatifs utilisent `BasicAuthentication` et `SessionAuthentication` et exigent un utilisateur authentifie.

Les exemples utilisent `42` comme identifiant fictif de document.

## Documents

### Lister les documents

```http
GET /api/documents/
```

Authentification : requise.

Reponse generale :

```json
[
  {
    "id": 42,
    "original_filename": "appel_offres.pdf",
    "mime_type": "application/pdf",
    "file_size": 123456,
    "status": "uploaded",
    "processing_metadata": {},
    "error_code": "",
    "error_message": "",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "download_url": "http://localhost:8000/api/documents/42/download/"
  }
]
```

### Creer un document

```http
POST /api/documents/
Content-Type: multipart/form-data
```

Payload :

```text
file=<PDF>
```

Contraintes importantes :

- un seul fichier nomme `file` ;
- PDF uniquement ;
- taille maximale configuree ;
- PDF non chiffre et structurellement lisible.

Erreurs possibles : `invalid_filename`, `unsupported_extension`, `empty_file`, `file_too_large`, `unsupported_content_type`, `invalid_pdf_header`, `invalid_pdf_structure`, `encrypted_pdf`.

### Lire un document

```http
GET /api/documents/42/
```

Retourne le serializer de lecture du document si l'utilisateur en est proprietaire.

### Telecharger un document

```http
GET /api/documents/42/download/
```

Retourne le PDF avec `X-Content-Type-Options: nosniff`.

Erreur possible : `document_file_unavailable` si le fichier n'est pas accessible.

### Supprimer un document

```http
DELETE /api/documents/42/
```

Supprime le document et tente de nettoyer le fichier stocke.

## Extraction

### Consulter le resultat d'extraction

```http
GET /api/documents/42/extraction/
```

Reponse generale :

```json
{
  "id": 1,
  "document_id": 42,
  "status": "completed",
  "extracted_text": "...",
  "has_text": true,
  "page_count": 10,
  "pages_processed": 10,
  "character_count": 12000,
  "text_sha256": "...",
  "extraction_metadata": {},
  "error_code": "",
  "error_message": "",
  "created_at": "2026-01-01T00:00:00Z",
  "updated_at": "2026-01-01T00:00:00Z"
}
```

### Lancer l'extraction

```http
POST /api/documents/42/extraction/
```

Retourne `200` si l'extraction aboutit, `422` si l'extraction echoue avec un statut `failed`.

## Chunks

### Consulter les chunks

```http
GET /api/documents/42/chunks/
```

### Generer les chunks

```http
POST /api/documents/42/chunks/
```

Reponse generale :

```json
{
  "document_id": 42,
  "extraction_result_id": 1,
  "chunk_count": 3,
  "chunks": [
    {
      "id": 10,
      "document_id": 42,
      "extraction_result_id": 1,
      "chunk_index": 0,
      "text": "...",
      "character_start": 0,
      "character_end": 1200,
      "text_length": 1200,
      "text_sha256": "...",
      "metadata": {},
      "created_at": "2026-01-01T00:00:00Z"
    }
  ],
  "chunking_metadata": {}
}
```

Erreurs importantes : extraction absente, texte non disponible, entree trop grande, trop de chunks, chunk trop grand.

## Embeddings

### Consulter le statut des embeddings

```http
GET /api/documents/42/embeddings/
```

### Generer les embeddings

```http
POST /api/documents/42/embeddings/
```

Reponse generale :

```json
{
  "document_id": 42,
  "chunk_count": 3,
  "embedding_count": 3,
  "embeddings": [
    {
      "id": 20,
      "chunk_id": 10,
      "chunk_index": 0,
      "provider": "gemini",
      "model": "gemini-embedding-2",
      "dimension": 768,
      "chunk_sha256": "...",
      "status": "completed",
      "error_code": "",
      "error_message": "",
      "metadata": {},
      "created_at": "2026-01-01T00:00:00Z",
      "updated_at": "2026-01-01T00:00:00Z"
    }
  ],
  "embedding_metadata": {}
}
```

Erreurs importantes : chunks absents, fournisseur mal configure, timeout, rate limit, erreur fournisseur.

## Search

```http
POST /api/documents/42/search/
Content-Type: application/json
```

Payload :

```json
{
  "query": "Quelle est la date limite ?",
  "top_k": 5
}
```

Retourne les resultats de recherche semantique/hybride avec metadonnees. `top_k` est borne par `SEMANTIC_SEARCH_MAX_TOP_K`.

## RAG Ask

```http
POST /api/documents/42/ask/
Content-Type: application/json
```

Payload :

```json
{
  "question": "Quelle est la date limite ?",
  "top_k": 5
}
```

Reponse generale :

```json
{
  "document_id": 42,
  "question": "Quelle est la date limite ?",
  "answer": "...",
  "sources": [],
  "rag_metadata": {}
}
```

Erreurs importantes : question trop longue, `top_k` invalide, chunks/embeddings absents, erreur fournisseur LLM.

## Prompt Engineering Ask

```http
POST /api/documents/42/prompt-engineering/ask/
Content-Type: application/json
```

Payload :

```json
{
  "question": "Resume le document."
}
```

Cette approche utilise le texte extrait directement, sans retrieval.

## RAPTOR Build

### Statut RAPTOR

```http
GET /api/documents/42/raptor/
```

### Construire l'index RAPTOR

```http
POST /api/documents/42/raptor/
```

Erreurs importantes : chunks absents, embeddings absents, trop de chunks pour RAPTOR, trop de noeuds, erreur fournisseur de resume.

## RAPTOR Ask

```http
POST /api/documents/42/raptor/ask/
Content-Type: application/json
```

Payload :

```json
{
  "question": "Quelles sont les obligations principales ?",
  "top_k": 6
}
```

Reponse generale :

```json
{
  "document_id": 42,
  "question": "Quelles sont les obligations principales ?",
  "answer": "...",
  "sources": [],
  "raptor_metadata": {}
}
```

Erreurs importantes : index RAPTOR absent, question trop longue, `top_k` invalide, erreur fournisseur.
