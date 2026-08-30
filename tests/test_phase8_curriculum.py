"""
SS Tutor BD — Phase 8 Curriculum & Core Model Unit Tests
Tests curriculum schema, deterministic IDs, package boundaries, coverage engine,
dataset auditor, 13D evaluation suite, and developer module contract.
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.curriculum.schema import CurriculumConcept, CurriculumTopic, CurriculumChapter
from core.curriculum.boundaries import CurriculumScope, KnowledgeUnit, KnowledgePackMetadata
from core.curriculum.coverage_engine import CurriculumCoverageEngine
from core.curriculum.dataset_auditor import DatasetQualityAuditor
from core.tutor_module import SSTutorBDModule, TutorModuleConfig
from benchmarks.phase8.curriculum_eval_suite import CurriculumEvaluationSuite


class TestPhase8Curriculum(unittest.TestCase):
    def test_deterministic_concept_id_generation(self):
        cid = CurriculumConcept.generate_id(grade=8, subject="mathematics", ch_num=2, top_num=1, con_num=3)
        self.assertEqual(cid, "g08.math.ch02.t01.c03")

    def test_curriculum_scope_matching(self):
        scope = CurriculumScope(scope_id="class8_math_only", grades=[8], subjects=["mathematics"])
        self.assertTrue(scope.matches(grade=8, subject="mathematics", chapter=2))
        self.assertFalse(scope.matches(grade=9, subject="mathematics", chapter=2))
        self.assertFalse(scope.matches(grade=8, subject="science", chapter=1))

    def test_coverage_engine(self):
        engine = CurriculumCoverageEngine()
        rep = engine.audit_coverage()
        self.assertIn("overall_curriculum_coverage_pct", rep)
        self.assertGreater(rep["total_curriculum_concepts_defined"], 50)
        self.assertEqual(rep["grade_breakdown"]["grade_8"]["subjects"]["mathematics"]["status"], "COVERED")
        self.assertEqual(rep["grade_breakdown"]["grade_6"]["subjects"]["mathematics"]["status"], "MISSING_SOURCE")

    def test_dataset_quality_auditor(self):
        auditor = DatasetQualityAuditor()
        rep = auditor.audit()
        self.assertGreaterEqual(rep["total_examples_scanned"], 10000)
        self.assertIn("duplicate_rate_pct", rep)
        self.assertIn("socratic_hint", rep["educational_behaviors_breakdown"])

    def test_developer_tutor_module(self):
        module = SSTutorBDModule()
        init_res = module.initialize()
        self.assertEqual(init_res["status"], "INITIALIZED")

        # Test solve
        solve_res = module.solve("৩/৪ + ৫/৬ এর যোগফল কত?")
        self.assertIn("১৯/১২", solve_res["response"])

        # Test hint
        hint_res = module.hint("৩/৪ + ৫/৬ এর যোগফল কত? hint দাও।")
        self.assertNotIn("১৯/১২", hint_res["response"])

        # Test unload
        unload_res = module.unload_model()
        self.assertEqual(unload_res["status"], "MODEL_UNLOADED")

    def test_13d_curriculum_evaluation(self):
        eval_suite = CurriculumEvaluationSuite()
        rep = eval_suite.evaluate_all_dimensions()
        self.assertEqual(rep["dimensions_evaluated_count"], 13)
        self.assertGreaterEqual(rep["composite_score_pct"], 80.0)
        self.assertEqual(rep["dimensions"]["D01_mathematical_accuracy"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
