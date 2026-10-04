# Installation Locale

Ce guide decrit une installation locale de Tender Intelligence V1. Les commandes sont donnees pour PowerShell sous Windows, car le projet utilise deja un environnement `venv` Windows.

## Prerequis

- Python compatible avec les dependances de `requirements.txt`.
- PostgreSQL.
- Extension PostgreSQL `vector` fournie par pgvector.
- Tesseract OCR.
- Poppler, utilise par `pdf2image`.
- Une cle API fournisseur pour Gemini si les embeddings, RAG, RAPTOR ou Prompt Engineering doivent appeler le fournisseur reel.

Le projet ne fixe pas de version systeme de PostgreSQL, Tesseract ou Poppler dans le repository. Utiliser des versions compatibles avec les paquets Python installes.

## Environnement Python

Depuis la racine du repository :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Les dependances principales sont listees dans `requirements.txt` : Django, Django REST Framework, PostgreSQL/pgvector, Google GenAI, extraction PDF/OCR, scikit-learn et UMAP.

## PostgreSQL et pgvector

Creer une base et un utilisateur PostgreSQL. Les noms ci-dessous sont des exemples ; adaptez-les a votre environnement.

```sql
CREATE DATABASE tender_project;
CREATE USER tender_user WITH PASSWORD 'YOUR_DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE tender_project TO tender_user;
```

Activer l'extension `vector` dans la base cible avec un compte disposant des droits necessaires :

```sql
\c tender_project
CREATE EXTENSION IF NOT EXISTS vector;
```

Cette etape est obligatoire avant `python manage.py migrate` : la migration `extraction.0004_chunkembedding_pgvector` verifie explicitement que l'extension `vector` existe.

Selon la configuration PostgreSQL locale, il peut etre necessaire d'accorder les droits sur le schema `public` :

```sql
GRANT ALL ON SCHEMA public TO tender_user;
```

## Configuration `.env`

Copier le modele :

```powershell
copy .env.example .env
```

Renseigner au minimum :

```dotenv
DJANGO_SECRET_KEY=YOUR_SECRET_KEY
DJANGO_DEBUG=true
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

DB_NAME=tender_project
DB_USER=tender_user
DB_PASSWORD=YOUR_DB_PASSWORD
DB_HOST=localhost
DB_PORT=5432

GEMINI_API_KEY=YOUR_API_KEY
```

Pour OCR :

```dotenv
TESSERACT_CMD=C:\Path\To\tesseract.exe
POPPLER_PATH=C:\Path\To\poppler\bin
```

Si Tesseract et Poppler sont deja dans le `PATH`, ces variables peuvent rester vides selon l'environnement local.

Les limites de ressources et les parametres IA sont deja enumeres dans `.env.example` : upload PDF, extraction, OCR, chunking, embeddings, recherche hybride, RAG, Prompt Engineering et RAPTOR.

## Migrations

Appliquer les migrations apres creation de la base et activation de pgvector :

```powershell
python manage.py migrate
```

Migrations utiles a connaitre :

- `documents.0002_refactor_document_management` : proprietaire, stockage, statuts et contraintes des documents.
- `extraction.0001_initial` : resultats d'extraction.
- `extraction.0002_textchunk` : chunks.
- `extraction.0003_chunkembedding` : embeddings.
- `extraction.0004_chunkembedding_pgvector` : champ vectoriel pgvector.
- `extraction.0005_raptor_index` : index et noeuds RAPTOR.

## Utilisateur Django

Creer un superutilisateur ou un utilisateur de test :

```powershell
python manage.py createsuperuser
```

L'API utilise `BasicAuthentication` et `SessionAuthentication`. Les endpoints applicatifs exigent un utilisateur authentifie.

## Lancement

Verifier la configuration Django :

```powershell
python manage.py check
```

Lancer le serveur local :

```powershell
python manage.py runserver
```

L'API est ensuite disponible sous :

```text
http://127.0.0.1:8000/api/
```

## Tests

Tests Django :

```powershell
python manage.py test --keepdb
```

Test unitaire du script de benchmark depuis la racine :

```powershell
python -m unittest benchmark.test_run_benchmark
```

Evaluation synthetique :

```powershell
python benchmark/evaluate_generalization.py
```

Le benchmark reel appelle les endpoints et le fournisseur LLM configure. Il necessite une base preparee, un utilisateur existant, un document deja traite, et une cle fournisseur valide :

```powershell
$env:BENCHMARK_DOCUMENT_ID="YOUR_DOCUMENT_ID"
$env:BENCHMARK_USERNAME="YOUR_USERNAME"
python benchmark/run_benchmark.py
```

Ce benchmark genere localement `benchmark/results_final.json`. Ce fichier n'est pas suivi par Git dans la V1.
