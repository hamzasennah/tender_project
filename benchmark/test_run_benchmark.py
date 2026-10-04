import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmark import run_benchmark


class FakeLLMProvider:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def generate_answer(self, _question, _context):
        status = self.statuses.pop(0)
        if status == 200:
            return "OK"
        raise run_benchmark.LLMProviderError(
            "llm_service_unavailable",
            "LLM provider service is unavailable.",
            response_status=status,
        )


class ProviderPreflightTests(unittest.TestCase):
    def test_preflight_retries_transient_statuses_without_network(self):
        sleep_calls = []
        provider = FakeLLMProvider([503, 429, 200])

        with (
            patch.object(run_benchmark, "PREFLIGHT_MAX_ATTEMPTS", 3),
            patch.object(run_benchmark, "PREFLIGHT_BACKOFF_SECONDS", 5),
            patch.object(run_benchmark.time, "sleep", sleep_calls.append),
        ):
            result, attempts = run_benchmark.provider_preflight(
                llm_provider=provider,
            )

        self.assertEqual(result["http_status"], 200)
        self.assertEqual([attempt["http_status"] for attempt in attempts], [503, 429, 200])
        self.assertEqual(sleep_calls, [5, 10])

    def test_main_aborts_before_benchmark_when_preflight_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / "results_final.json"
            output = io.StringIO()

            with (
                patch.object(run_benchmark, "RESULTS_FILE", result_file),
                patch.object(run_benchmark, "load_questions", return_value=[]),
                patch.object(run_benchmark, "get_authenticated_client", return_value=object()),
                patch.object(
                    run_benchmark,
                    "provider_preflight",
                    return_value=(
                        {
                            "http_status": 503,
                            "error_code": "llm_service_unavailable",
                            "error_message": "LLM provider service is unavailable.",
                        },
                        [{"attempt": 1, "http_status": 503}],
                    ),
                ),
                redirect_stdout(output),
            ):
                run_benchmark.main()

            self.assertFalse(result_file.exists())
            self.assertIn("Provider preflight failed: HTTP 503", output.getvalue())
            self.assertIn("Benchmark aborted before execution.", output.getvalue())
            self.assertIn("No scientific result generated.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
