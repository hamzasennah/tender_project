import json
import os
import sys
import time
from pathlib import Path


# --------------------------------------------------
# Initialisation Django
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "config.settings",
)

import django

django.setup()

from django.conf import settings

if "testserver" not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS.append("testserver")

# --------------------------------------------------
# Imports Django après django.setup()
# --------------------------------------------------

from django.contrib.auth import get_user_model
from rest_framework.test import APIClient


# --------------------------------------------------
# Configuration benchmark
# --------------------------------------------------

DOCUMENT_ID = 12
BENCHMARK_USERNAME = "Hamza"

QUESTIONS_FILE = Path(__file__).parent / "questions.json"

ENHANCED_RAG_URL = f"/api/documents/{DOCUMENT_ID}/ask/"
RAPTOR_URL = f"/api/documents/{DOCUMENT_ID}/raptor/ask/"
PROMPT_ENGINEERING_URL = f"/api/documents/{DOCUMENT_ID}/prompt-engineering/ask/"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_authenticated_client():
    User = get_user_model()

    try:
        user = User.objects.get(username=BENCHMARK_USERNAME)
    except User.DoesNotExist:
        raise RuntimeError(
            f"Utilisateur '{BENCHMARK_USERNAME}' introuvable."
        )

    client = APIClient()

    # Authentification uniquement dans ce script de benchmark local.
    # Aucun changement de sécurité n'est effectué sur l'API.
    client.force_authenticate(user=user)

    return client


def ask_endpoint(client, url, question):
    payload = {
        "question": question,
        "top_k": 5,
    }

    start = time.perf_counter()

    response = client.post(
        url,
        payload,
        format="json",
    )

    elapsed_ms = round(
        (time.perf_counter() - start) * 1000,
        2,
    )

    return response, elapsed_ms


def main():
    questions = load_questions()
    client = get_authenticated_client()

    print("=" * 80)
    print("BENCHMARK ENHANCED RAG vs RAPTOR")
    print(f"Document : {DOCUMENT_ID}")
    print(f"Utilisateur : {BENCHMARK_USERNAME}")
    print("=" * 80)

    for item in questions:
        question_id = item["id"]
        question_type = item["type"]
        question = item["question"]
        expected = item.get("expected")

        print("\n" + "=" * 80)
        print(f"QUESTION {question_id} — {question_type}")
        print("=" * 80)

        print(question)

        if expected:
            print(f"\nAttendu : {expected}")

        # --------------------------------------------------
        # Enhanced RAG
        # --------------------------------------------------

        print("\n--- ENHANCED RAG ---")

        response, elapsed_ms = ask_endpoint(
            client,
            ENHANCED_RAG_URL,
            question,
        )

        print(f"HTTP : {response.status_code}")
        print(f"Temps : {elapsed_ms} ms")

        if response.status_code == 200:
            data = response.json()

            print("\nRéponse :")
            print(data.get("answer"))

            metadata = data.get("rag_metadata", {})
            context = metadata.get("context", {})

            print(
                "\nContexte :",
                context.get("context_char_count"),
            )

            print(
                "Sources :",
                [
                    source.get("chunk_id")
                    for source in data.get("sources", [])
                ],
            )

        else:
            print(response.content.decode("utf-8"))

        # --------------------------------------------------
        # Prompt Engineering
        # --------------------------------------------------

        print("\n--- PROMPT ENGINEERING ---")

        response, elapsed_ms = ask_endpoint(
            client,
            PROMPT_ENGINEERING_URL,
            question,
        )

        print(f"HTTP : {response.status_code}")
        print(f"Temps : {elapsed_ms} ms")

        if response.status_code == 200:
            data = response.json()

            print("\nRÃ©ponse :")
            print(data.get("answer"))

            metadata = data.get("prompt_engineering_metadata", {})

            print("Contexte original chars :", metadata.get("original_char_count"))
            print(
                "Contexte tokens :",
                metadata.get("context_token_count")
                or metadata.get("estimated_context_token_count"),
            )
            print(
                "Contexte original tokens :",
                metadata.get("original_token_count")
                or metadata.get("estimated_original_token_count"),
            )
            print("Document complet utilise :", metadata.get("full_document_used"))

            print(
                "\nContexte :",
                metadata.get("context_char_count"),
            )

            print(
                "TronquÃ© :",
                metadata.get("truncated"),
            )

        else:
            print(response.content.decode("utf-8"))

        # --------------------------------------------------
        # RAPTOR
        # --------------------------------------------------

        print("\n--- RAPTOR ---")

        response, elapsed_ms = ask_endpoint(
            client,
            RAPTOR_URL,
            question,
        )

        print(f"HTTP : {response.status_code}")
        print(f"Temps : {elapsed_ms} ms")

        if response.status_code == 200:
            data = response.json()

            print("\nRéponse :")
            print(data.get("answer"))

            metadata = data.get("raptor_metadata", {})

            print(
                "\nNiveaux sélectionnés :",
                metadata.get("selected_nodes_per_level"),
            )

            print(
                "Contexte :",
                metadata.get("context_char_count"),
            )

            print(
                "Nodes :",
                [
                    source.get("node_id")
                    for source in data.get("sources", [])
                ],
            )

        else:
            print(response.content.decode("utf-8"))


if __name__ == "__main__":
    main()
