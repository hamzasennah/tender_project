# Structure du Repository

Arborescence simplifiee de la V1 :

```text
tender_project/
├── README.md
├── .env.example
├── requirements.txt
├── manage.py
├── config/
├── documents/
├── extraction/
├── benchmark/
└── docs/
```

Les dossiers locaux tels que `venv/`, `media/`, `__pycache__/`, `.git/` et les secrets ne font pas partie de la documentation fonctionnelle du repository.

## Racine

### `README.md`

Point d'entree du projet, resume fonctionnel, stack, resultats et liens vers la documentation detaillee.

### `.env.example`

Modele de configuration. Il liste les variables attendues pour Django, PostgreSQL, upload, OCR, chunking, embeddings, recherche, RAG, Prompt Engineering et RAPTOR.

### `requirements.txt`

Dependances Python de la V1.

### `manage.py`

Point d'entree Django pour migrations, serveur local, tests et commandes de gestion.

## `config/`

Configuration Django :

- `settings.py` : apps, base PostgreSQL, chargement `.env`, media, limites et parametres IA.
- `urls.py` : routage racine vers admin, documents et extraction.
- `asgi.py`, `wsgi.py` : points d'entree serveur.

## `documents/`

Gestion des documents PDF :

- `models.py` : modele `Document`.
- `views.py` : `DocumentViewSet`, download et CRUD autorise.
- `serializers.py` : serializers upload/lecture.
- `permissions.py` : ownership.
- `services/file_validation.py` : validation PDF.
- `services/document_service.py` : creation et suppression documentaire.
- `migrations/` : schema documents.
- `tests.py` : tests de l'app documents.

## `extraction/`

Pipeline IA/documentaire :

- `models.py` : `TextExtractionResult`, `TextChunk`, `ChunkEmbedding`, `RaptorIndex`, `RaptorNode`, `RaptorNodeChild`.
- `views.py` : endpoints extraction, chunks, embeddings, search, RAG, Prompt Engineering, RAPTOR.
- `serializers.py` : serializers de resultats extraction/chunks/embeddings.
- `urls.py` : routes API.
- `services/pdf_text_extraction.py` : extraction native et OCR.
- `services/text_chunking.py` : chunking.
- `services/embeddings.py` : embeddings.
- `services/semantic_search.py` : recherche semantique/hybride.
- `services/rag.py` : RAG.
- `services/prompt_engineering/` : Prompt Engineering.
- `services/raptor/` : build, retrieval, answering, clustering, summarization et intents RAPTOR.
- `migrations/` : schema extraction, chunks, embeddings pgvector et RAPTOR.
- `tests.py`, `tests_generalization_evaluation.py` : tests.

## `benchmark/`

Evaluation et benchmark :

- `evaluate_generalization.py` : commande de lancement de l'evaluation synthetique.
- `generalization_evaluation.py` : logique d'evaluation synthetique.
- `evaluation_report.json` : rapport synthetique suivi.
- `questions.json` : questions du benchmark reel.
- `run_benchmark.py` : benchmark final provider-dependent via API.
- `test_run_benchmark.py` : tests unitaires du script benchmark.
- `results_provider_incident_*.json` : artefact d'incident fournisseur historique.
- `results_final.json` : sortie locale du benchmark reel, non suivie par Git.

## `docs/`

Documentation finale V1 :

- `installation.md`
- `architecture.md`
- `workflow.md`
- `api.md`
- `benchmarks.md`
- `security_limits.md`
- `repository_structure.md`
