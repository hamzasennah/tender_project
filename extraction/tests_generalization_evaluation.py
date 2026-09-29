import tempfile
import shutil

from django.test import TestCase, override_settings

from benchmark.generalization_evaluation import (
    FACTS,
    build_synthetic_dataset_specs,
    run_evaluation,
)


class ScientificGeneralizationEvaluationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_root = tempfile.mkdtemp(prefix="generalization-eval-tests-")
        cls._settings = override_settings(
            MEDIA_ROOT=cls._media_root,
            RAPTOR_MAX_LEVELS=3,
            RAPTOR_MAX_CLUSTERS=6,
            RAPTOR_MAX_CHUNKS=50,
            RAPTOR_MAX_NODES=200,
            RAPTOR_MAX_SUMMARY_CONTEXT_CHARS=4000,
            RAPTOR_MAX_SUMMARY_CHARS=1200,
            RAG_MAX_CONTEXT_CHARS=5000,
            HYBRID_VECTOR_CANDIDATES=20,
            HYBRID_LEXICAL_CANDIDATES=20,
        )
        cls._settings.enable()

    @classmethod
    def tearDownClass(cls):
        cls._settings.disable()
        shutil.rmtree(cls._media_root, ignore_errors=True)
        super().tearDownClass()

    def test_default_synthetic_dataset_is_multi_structure_and_not_benchmark_specific(self):
        specs = build_synthetic_dataset_specs()

        self.assertEqual([spec.name for spec in specs], ["small", "medium", "large"])
        self.assertGreaterEqual(specs[0].size, 5)
        self.assertGreaterEqual(specs[1].size, 40)
        self.assertGreaterEqual(specs[2].size, 150)
        joined = "\n".join(chunk for spec in specs for chunk in spec.chunks)
        self.assertNotIn("28 septembre 2018", joined)
        self.assertNotIn("60 jours", joined)
        self.assertNotIn("3 %", joined)
        self.assertIn(FACTS["atlas_date"]["value"], joined)
        self.assertIn(FACTS["cygnus_amount"]["value"], joined)

    def test_generalization_evaluation_report_contains_all_approaches(self):
        report = run_evaluation(
            output_path=None,
            sizes={"small": 8, "medium": 10, "large": 12},
        )

        self.assertFalse(report["uses_document_id_12"])
        self.assertFalse(report["external_provider_calls"])
        self.assertEqual(set(report["approaches"]), {"rag", "raptor", "prompt_engineering"})
        self.assertEqual(len(report["approaches"]["rag"]["datasets"]), 3)
        self.assertEqual(len(report["approaches"]["raptor"]["datasets"]), 3)
        self.assertGreaterEqual(
            report["approaches"]["rag"]["metrics"]["case_count"],
            30,
        )
        self.assertGreaterEqual(
            report["approaches"]["raptor"]["metrics"]["case_count"],
            30,
        )
        self.assertGreaterEqual(
            report["approaches"]["prompt_engineering"]["metrics"]["case_count"],
            4,
        )
        self.assertIn("above_window_documents", report["improvement_classification"]["prompt_engineering"])

    def test_prompt_engineering_above_limit_case_is_marked_structural(self):
        report = run_evaluation(
            output_path=None,
            sizes={"small": 8},
        )
        above_limit = [
            item
            for item in report["approaches"]["prompt_engineering"]["cases"]
            if item["question_type"] == "above_limit"
        ][0]

        self.assertTrue(above_limit["truncation_metadata"]["truncated"])
        self.assertTrue(above_limit["structural_limit"])
        self.assertEqual(
            report["improvement_classification"]["prompt_engineering"]["above_window_documents"],
            "STRUCTURAL LIMIT",
        )
