import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cyber_utils import NO_CONSENSUS, SEVERITY_ORDER, compute_metrics, make_committee_snapshot


class MetricsTests(unittest.TestCase):
    def _assessment(self, label: str, severity: str, citations=None, accept: bool = True, block_reason=None):
        return {
            "ranked_findings": [
                {"rank": 1, "label": label, "severity": severity, "citations": citations or ["L001"]},
                {"rank": 2, "label": label, "severity": severity, "citations": []},
                {"rank": 3, "label": label, "severity": severity, "citations": []},
            ],
            "decision_summary": "stub",
            "accept": accept,
            "block_reason": block_reason,
        }

    def _snapshot(self, turn_index: int, triples):
        latest = {
            "A": {"assessment": self._assessment(*triples[0])},
            "B": {"assessment": self._assessment(*triples[1])},
            "C": {"assessment": self._assessment(*triples[2])},
        }
        phase = "round0" if turn_index == 0 else "public"
        public_turn_index = None if turn_index == 0 else turn_index - 1
        return make_committee_snapshot(turn_index, phase, latest, public_turn_index=public_turn_index)

    def _history(self, trajectory):
        return {
            "decision_trajectory": trajectory,
            "round0": [],
            "rounds": [],
            "validation_stats": {},
            "condition": {"condition_id": "C1"},
            "run_id": "run-1",
            "run_status": "completed",
        }

    def test_single_agent_baseline_snapshot_is_trivially_unanimous(self):
        snapshot = make_committee_snapshot(
            0,
            "round0",
            {"Solo": {"assessment": self._assessment("XSS_Reflected", "Medium", ["L001"])}},
            public_turn_index=None,
        )
        self.assertTrue(snapshot["unanimous_type"])
        self.assertTrue(snapshot["unanimous_exact"])
        self.assertTrue(snapshot["all_accept"])
        self.assertTrue(snapshot["agreement_exact_with_signoff"])
        self.assertEqual(snapshot["committee_type"], "XSS_Reflected")
        self.assertEqual(snapshot["committee_exact"], {"label": "XSS_Reflected", "severity": "Medium"})

    def test_aborted_run_does_not_get_fabricated_outcome_metrics(self):
        history = self._history([self._snapshot(0, [("XSS_Reflected", "Medium", ["L001"])] * 3)])
        history["run_status"] = "aborted_invalid_output"
        summary = compute_metrics(
            history,
            {"final_label": "XSS_Reflected", "final_severity": "Medium"},
        )
        self.assertIsNone(summary["headline_metrics"]["FinalCorrectExact"])
        self.assertIsNone(summary["headline_metrics"]["AnyAgreementExact"])
        self.assertEqual(summary["appendix_debug"]["RunStatus"], "aborted_invalid_output")

    def test_tie_handling_uses_no_consensus(self):
        snapshot = self._snapshot(
            0,
            [
                ("LabelA", "Low", ["L001"]),
                ("LabelB", "Low", ["L001"]),
                ("LabelC", "Low", ["L001"]),
            ],
        )
        self.assertEqual(snapshot["committee_type"], NO_CONSENSUS)
        self.assertEqual(snapshot["committee_exact"], NO_CONSENSUS)

    def test_wrong_consensus_exact(self):
        trajectory = [
            self._snapshot(0, [("XSS_Reflected", "Medium", ["L001"])] * 3),
            self._snapshot(1, [("XSS_Reflected", "Medium", ["L001"])] * 3),
        ]
        summary = compute_metrics(
            self._history(trajectory),
            {"final_label": "SQLInjection", "final_severity": "High"},
        )
        self.assertEqual(summary["derived_metrics"]["WrongConsensusExact"], 1)
        self.assertEqual(summary["headline_metrics"]["FinalAgreementExact"], 1)
        self.assertEqual(summary["headline_metrics"]["FinalCorrectExact"], 0)

    def test_late_drift_agreement_exact(self):
        trajectory = [
            self._snapshot(0, [("XSS_Reflected", "Medium", ["L001"])] * 3),
            self._snapshot(1, [("XSS_Reflected", "Medium", ["L001"])] * 3),
            self._snapshot(
                2,
                [
                    ("XSS_Reflected", "Medium", ["L001"]),
                    ("XSS_Stored", "Low", ["L001"]),
                    ("NoFinding", "Info", ["L001"]),
                ],
            ),
        ]
        summary = compute_metrics(
            self._history(trajectory),
            {"final_label": "XSS_Reflected", "final_severity": "Medium"},
        )
        self.assertEqual(summary["headline_metrics"]["AnyAgreementExact"], 1)
        self.assertEqual(summary["headline_metrics"]["FinalAgreementExact"], 0)
        self.assertEqual(summary["derived_metrics"]["LateDriftAgreementExact"], 1)

    def test_final_agreement_exact_requires_signoff(self):
        trajectory = [
            self._snapshot(0, [("XSS_Reflected", "Medium", ["L001"])] * 3),
            make_committee_snapshot(
                1,
                "public",
                {
                    "A": {"assessment": self._assessment("XSS_Reflected", "Medium", ["L001"], accept=True)},
                    "B": {
                        "assessment": self._assessment(
                            "XSS_Reflected",
                            "Medium",
                            ["L001"],
                            accept=False,
                            block_reason="Severity not defensible enough.",
                        )
                    },
                    "C": {"assessment": self._assessment("XSS_Reflected", "Medium", ["L001"], accept=True)},
                },
                public_turn_index=0,
            ),
        ]
        summary = compute_metrics(
            self._history(trajectory),
            {"final_label": "XSS_Reflected", "final_severity": "Medium"},
        )
        self.assertEqual(summary["headline_metrics"]["FinalCorrectExact"], 1)
        self.assertEqual(summary["headline_metrics"]["FinalAgreementExact"], 0)
        self.assertEqual(summary["derived_metrics"]["FinalAgreementType"], 0)
        self.assertEqual(summary["derived_metrics"]["AnyAgreementType"], 0)
        self.assertEqual(summary["derived_metrics"]["FalseAgreementWithoutSignoffExact"], 1)

    def test_consensus_latency_exact_uses_stable_start(self):
        trajectory = [
            self._snapshot(0, [("XSS_Reflected", "Low", ["L001"])] * 3),
            self._snapshot(1, [("XSS_Reflected", "Low", ["L001"])] * 3),
            self._snapshot(2, [("XSS_Reflected", "Medium", ["L001"])] * 3),
            self._snapshot(3, [("XSS_Reflected", "Medium", ["L001"])] * 3),
        ]
        summary = compute_metrics(
            self._history(trajectory),
            {"final_label": "XSS_Reflected", "final_severity": "Medium"},
        )
        self.assertEqual(summary["derived_metrics"]["ConsensusLatencyExact"], 2)

    def test_severity_mapping_ordering(self):
        self.assertLess(SEVERITY_ORDER["Compliance"], SEVERITY_ORDER["Info"])
        self.assertLess(SEVERITY_ORDER["Info"], SEVERITY_ORDER["Low"])
        self.assertLess(SEVERITY_ORDER["Low"], SEVERITY_ORDER["Medium"])
        self.assertLess(SEVERITY_ORDER["Medium"], SEVERITY_ORDER["High"])


if __name__ == "__main__":
    unittest.main()
