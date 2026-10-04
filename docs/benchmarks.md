# Benchmarks et Resultats

Cette page separe strictement deux experiences :

1. evaluation synthetique de generalisation ;
2. benchmark final sur document reel.

Les metriques ci-dessous proviennent des artefacts existants du repository et de la synthese finale V1. Elles ne doivent pas etre melangees.

## A. Evaluation Synthetique de Generalisation

Artefact de reference : `benchmark/evaluation_report.json`.

Objectif : tester la generalisation sur des structures `small`, `medium` et `large`, avec des categories variees, sans dependre du document reel.

Caracteristiques :

- `objective`: `scientific_generalization_evaluation`
- l'artefact source indique que cette evaluation ne depend pas du document reel final ;
- `external_provider_calls`: `false`
- 30 cas retrieval pour RAG.
- 30 cas retrieval pour RAPTOR.
- 4 scenarios specifiques pour Prompt Engineering.

### Resultats Globaux

| Approche | Resultat |
|---|---:|
| RAG | 25/30 succes |
| RAPTOR | 27/30 succes |
| Prompt Engineering | 4/4 scenarios specifiques |

### RAG

| Metrique | Valeur |
|---|---:|
| Succes | 25/30 |
| Recall@1 | 0.7222 |
| Recall@3 | 0.8889 |
| Recall@5 | 0.9136 |
| MRR | 0.8889 |

Par taille :

| Dataset | Resultat |
|---|---:|
| small | 9/10 |
| medium | 8/10 |
| large | 8/10 |

Limites observees :

- `comparison`: 1/3.
- `global_synthesis`: 0/3.
- autres categories : 3/3.

### RAPTOR

| Metrique | Valeur |
|---|---:|
| Succes | 27/30 |
| Recall@k | 0.9259 |
| MRR | 0.9815 |
| Summary fidelity | 1.0 |

Par taille :

| Dataset | Resultat |
|---|---:|
| small | 9/10 |
| medium | 9/10 |
| large | 9/10 |

Limites observees :

- `global_synthesis`: 0/3.
- autres categories : 3/3.

### Prompt Engineering

| Metrique | Valeur |
|---|---:|
| Scenarios | 4/4 |
| full_document_used | 3 |
| truncated | 1 |

Les scenarios Prompt Engineering ne sont pas directement equivalents aux 30 cas retrieval RAG/RAPTOR. La limite structurelle observee concerne les cas ou l'information pertinente se trouve apres une troncature imposee par la fenetre disponible.

## B. Benchmark Final sur Document Reel

Le benchmark final execute 8 questions sur 3 approches, soit 24 executions.

Artefacts associes :

- `benchmark/questions.json` : questions du benchmark reel, suivi par Git.
- `benchmark/results_final.json` : sortie generee localement par `benchmark/run_benchmark.py`, non suivie par Git.

La documentation est autonome : elle ne suppose pas que `benchmark/results_final.json` soit present apres un clone Git.

### Execution

| Indicateur | Valeur |
|---|---:|
| Questions | 8 |
| Approches | 3 |
| Executions | 24 |
| Preflight provider | HTTP 200 |
| HTTP 200 | 24/24 |
| Retries | 0 |
| 429 | 0 |
| 503 | 0 |

Attention : `success_count` dans l'artefact d'execution correspond aux statuts HTTP 200. Ce n'est pas une metrique de qualite scientifique.

### Qualite des Reponses

| Approche | Correct | Partial | Incorrect |
|---|---:|---:|---:|
| Prompt Engineering | 8 | 0 | 0 |
| RAG | 8 | 0 | 0 |
| RAPTOR | 4 | 2 | 2 |

### Detail Question par Question

| Question | Type | PE | RAG | RAPTOR |
|---|---|---|---|---|
| Q1 | factual_date | Correct | Correct | Correct |
| Q2 | duration | Correct | Correct | Correct |
| Q3 | percentage | Correct | Correct | Incorrect |
| Q4 | comparison | Correct | Correct | Incorrect |
| Q5 | global_synthesis | Correct | Correct | Partial |
| Q6 | unanswerable | Correct | Correct | Correct |
| Q7 | multi_query | Correct | Correct | Correct |
| Q8 | synthesis | Correct | Correct | Partial |

### Latences

| Approche | Mean ms | Median ms | Min ms | Max ms |
|---|---:|---:|---:|---:|
| Prompt Engineering | 16771.87 | 11472.65 | 6310.03 | 33623.5 |
| RAG | 12411.71 | 12380.19 | 4369.12 | 24511.39 |
| RAPTOR | 16955.13 | 13956.66 | 4225.71 | 49075.32 |

### Contexte

| Approche | Contexte |
|---|---|
| Prompt Engineering | 131603 chars/appel ; moyenne tokens environ 35133.62 ; document complet 8/8 ; truncated 0/8 |
| RAG | moyenne 5997.88 chars ; min 5984 ; max 6000 ; 5 chunks/appel ; tokens exacts non exposes |
| RAPTOR | moyenne 5472.88 chars ; min 4307 ; max 6000 ; context truncated 2/8 ; tokens exacts non exposes |

## Interpretation

Dans cette execution sur ce document :

- Prompt Engineering offre une forte couverture documentaire, avec un contexte tres volumineux.
- RAG obtient ici un excellent compromis qualite / contexte / latence.
- RAPTOR reussit plusieurs categories, mais montre des pertes de couverture sur certaines questions answerables et certaines syntheses.

Ces observations ne prouvent pas qu'une approche est toujours meilleure. Elles caracterisent cette V1, ce document, ces questions, cette configuration et ce fournisseur.

## Reproductibilite

Evaluation synthetique :

```powershell
python benchmark/evaluate_generalization.py
```

Test unitaire du script de benchmark depuis la racine :

```powershell
python -m unittest benchmark.test_run_benchmark
```

Benchmark reel provider-dependent :

```powershell
$env:BENCHMARK_DOCUMENT_ID="YOUR_DOCUMENT_ID"
$env:BENCHMARK_USERNAME="YOUR_USERNAME"
$env:BENCHMARK_TOP_K="5"
$env:BENCHMARK_DELAY_SECONDS="3"
$env:BENCHMARK_MAX_RETRIES="3"
$env:BENCHMARK_RETRY_BACKOFF_SECONDS="5"
$env:BENCHMARK_PREFLIGHT_MAX_RETRIES="3"
$env:BENCHMARK_PREFLIGHT_BACKOFF_SECONDS="5"
python benchmark/run_benchmark.py
```

Prerequis du benchmark reel :

- serveur Django/configuration Django fonctionnelle ;
- base PostgreSQL preparee ;
- utilisateur existant ;
- document cible deja charge et prepare ;
- extraction/chunking/embeddings/RAPTOR disponibles selon les approches ;
- `GEMINI_API_KEY` et fournisseur LLM fonctionnels.

Le script ecrit `benchmark/results_final.json` localement. Ce fichier doit rester non suivi par Git.
