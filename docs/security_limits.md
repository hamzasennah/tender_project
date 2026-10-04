# Securite et Limites

Cette page documente les protections presentes en V1 et les limites connues. Elle ne constitue pas une garantie de securite absolue.

## Authentication / Authorization

Les endpoints applicatifs utilisent :

- `BasicAuthentication`
- `SessionAuthentication`
- `IsAuthenticated`

Les vues filtrent les documents avec `owner=request.user`. Un utilisateur ne doit voir et manipuler que ses propres documents via l'API applicative.

## Ownership

Le modele `Document` possede un champ `owner`. Les listes, lectures, traitements d'extraction, chunking, embeddings, RAG, Prompt Engineering et RAPTOR recuperent le document dans le scope du proprietaire connecte.

Les URLs contenant un identifiant de document retournent une erreur si le document n'appartient pas a l'utilisateur authentifie.

## Stockage Prive des Documents

Les fichiers sont stockes sous un chemin incluant le proprietaire logique :

```text
media/documents/user_<owner_id>/<uuid>.pdf
```

Le nom stocke est un UUID et le nom original est conserve en base. Le telechargement passe par l'endpoint authentifie.

## Validation PDF

L'upload controle :

- nom de fichier ;
- extension `.pdf` ;
- taille non nulle ;
- taille maximale ;
- content type declare ;
- entete `%PDF-` ;
- structure PDF lisible ;
- rejet des PDF chiffres.

Variables associees :

- `DOCUMENTS_MAX_UPLOAD_SIZE_BYTES`
- `DOCUMENTS_ALLOWED_UPLOAD_CONTENT_TYPES`
- `DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE`
- `DJANGO_FILE_UPLOAD_MAX_MEMORY_SIZE`

## Extraction, OCR et Limites de Ressources

Limites d'extraction :

- `EXTRACTION_MAX_PDF_PAGES`
- `EXTRACTION_MAX_EXTRACTED_TEXT_CHARS`
- `EXTRACTION_MAX_PAGE_CONTENT_BYTES`
- seuils de detection de texte natif.

Limites OCR :

- `EXTRACTION_OCR_MAX_PDF_PAGES`
- `EXTRACTION_OCR_DPI`
- `EXTRACTION_OCR_MAX_IMAGE_PIXELS`
- `EXTRACTION_OCR_TIMEOUT_SECONDS`
- `EXTRACTION_OCR_MAX_TEXT_CHARS_PER_PAGE`

La V1 applique une prevalidation des dimensions raster OCR avant rendu, puis conserve une defense post-rendu sur l'image produite.

La V1 controle aussi les streams PDF avant decompression lorsque les tailles declarees ou brutes sont disponibles. Risque residuel : un stream fortement compresse peut avoir une taille decodee importante qui n'est connue completement qu'au decodage par `pypdf`. Ce risque n'est pas elimine ; il est borne par la verification secondaire apres decodage.

## Chunking, Embeddings, RAG et RAPTOR

Limites chunking :

- `TEXT_CHUNKING_TARGET_CHARS`
- `TEXT_CHUNKING_MAX_CHARS`
- `TEXT_CHUNKING_OVERLAP_CHARS`
- `TEXT_CHUNKING_MIN_CHARS`
- `TEXT_CHUNKING_MAX_CHUNKS`
- `TEXT_CHUNKING_MAX_INPUT_CHARS`

Limites embeddings :

- `EMBEDDING_BATCH_SIZE`
- `EMBEDDING_MAX_CHUNKS_PER_RUN`
- retries et timeout fournisseur.

Limites RAG :

- `RAG_MAX_TOP_K`
- `RAG_MAX_QUESTION_CHARS`
- `RAG_MAX_CONTEXT_CHARS`
- `RAG_MAX_SUBQUERIES`
- timeout, retries, temperature et tokens de sortie LLM.

Limites Prompt Engineering :

- longueur maximale de question ;
- limite de tokens d'entree modele ;
- marge de securite ;
- estimation chars/token.

Limites RAPTOR :

- niveaux ;
- clusters ;
- chunks ;
- noeuds ;
- contexte de resume ;
- taille de resume ;
- top_k ;
- longueur de question ;
- contexte final ;
- profondeur top-down ;
- nombre de sous-intents.

## LLM et Fournisseur

Les contenus documentaires, OCR et questions utilisateur doivent etre consideres comme non fiables. Les prompts demandent des reponses fondees sur le contexte fourni, mais une application utilisant un LLM ne doit pas etre consideree comme infaillible.

Des erreurs fournisseur peuvent se produire :

- rate limit ;
- timeout ;
- modele indisponible ;
- erreur reseau ou API.

Le benchmark reel inclut un preflight fournisseur avant execution.

## Prompt Injection

Les documents et questions peuvent contenir des instructions malveillantes. La V1 inclut des instructions de grounding et des contraintes de contexte, mais ne garantit pas une resistance parfaite aux prompt injections.

Les reponses doivent etre interpretees comme des sorties assistees par IA, a valider pour un usage critique.

## Secrets et Configuration

Les secrets doivent rester dans `.env`, qui est ignore par Git. Utiliser `.env.example` comme modele sans y placer de vraies credentials.

Variables sensibles :

- `DJANGO_SECRET_KEY`
- `DB_PASSWORD`
- `GEMINI_API_KEY`

## Limites Connues

- Prompt Engineering : limite structurelle de fenetre de contexte lorsque l'information pertinente est situee apres une troncature.
- RAG : limites observees sur certaines syntheses ou comparaisons dispersees dans l'evaluation synthetique.
- RAPTOR : syntheses dispersees et pertes observees sur certains cas du benchmark reel.
- PDF streams : risque residuel sur des streams fortement compresses dont la taille decodee complete n'est connue qu'au decodage.

## Deployment Considerations

Avant tout deploiement hors environnement local, revoir au minimum :

- `DJANGO_DEBUG=false`
- `DJANGO_SECRET_KEY` robuste ;
- `DJANGO_ALLOWED_HOSTS` restrictif ;
- HTTPS et configuration proxy ;
- stockage media prive ;
- logs et retention ;
- sauvegardes base et media ;
- supervision des erreurs fournisseur ;
- politique de rotation des secrets.

## Perspectives

Ameliorations possibles non bloquantes pour la V1 :

- throttling/rate limiting pour operations couteuses ;
- durcissement de configuration de deploiement ;
- rationalisation eventuelle du stockage embeddings ;
- tests workflow supplementaires ;
- jobs async pour traitements longs ;
- antivirus ou sandbox PDF ;
- object storage prive ;
- metriques cout/tokens avancees.

Ces points sont des perspectives P2/P3, pas des bugs bloquants de la V1.
