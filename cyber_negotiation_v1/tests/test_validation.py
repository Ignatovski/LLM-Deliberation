import json
import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cyber_utils import detect_leakage, parse_structured_response, read_condition_config, validate_rank1_citations


LABEL_SET = {
    "labels": ["XSS_Reflected", "XSS_Stored", "NoFinding", "Other"],
    "aliases": {"xss reflected": "XSS_Reflected"},
}
LINE_IDS = ["L001", "L002", "L003"]


class ValidationTests(unittest.TestCase):
    def _response(self, *, citations=None, public_message=None, accept=True, block_reason=None):
        payload = {
            "scratchpad": "<SCRATCHPAD>keep private reasoning separate from the public note</SCRATCHPAD>",
            "answer": (
                "<ANSWER>"
                + (
                    public_message
                    or "I am grounding this claim in L001 and I am keeping the committee focused on the cited evidence rather than expanding scope without support. The current packet still fits the reflected XSS hypothesis best, and I want any objection to explain the same lines more cleanly before we change the top label."
                )
                + "</ANSWER>\n<ASSESSMENT>"
                + json.dumps(
                    {
                        "ranked_findings": [
                            {
                                "rank": 1,
                                "label": "XSS_Reflected",
                                "severity": "Medium",
                                "citations": citations if citations is not None else ["L001"],
                                "rationale": "stub",
                            },
                            {
                                "rank": 2,
                                "label": "XSS_Stored",
                                "severity": "Low",
                                "citations": [],
                                "rationale": "stub",
                            },
                            {
                                "rank": 3,
                                "label": "NoFinding",
                                "severity": "Info",
                                "citations": [],
                                "rationale": "stub",
                            },
                        ],
                        "decision_summary": "stub",
                        "accept": accept,
                        "block_reason": block_reason,
                    }
                )
                + "</ASSESSMENT>"
            ),
            "plan": "<PLAN>ask for direct line-based objections only</PLAN>",
        }
        return json.dumps(payload)

    def test_citation_validator_flags_bad_count_and_id(self):
        result = validate_rank1_citations(["L001", "BAD", "L002"], LINE_IDS)
        self.assertTrue(result["citation_violation"])
        self.assertTrue(result["citation_count_violation"])
        self.assertTrue(result["citation_invalid_id_violation"])

    def test_parse_keeps_citation_violations_non_fatal(self):
        parsed, error, validation = parse_structured_response(
            self._response(citations=["BAD"]),
            line_ids=LINE_IDS,
            label_set=LABEL_SET,
            public_message_min_chars=50,
            public_message_max_chars=500,
            forbidden_public_tokens=["PRIVATE"],
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(error, "")
        self.assertTrue(validation["citation_violation"])
        self.assertTrue(validation["citation_invalid_id_violation"])

    def test_parse_requires_block_reason_when_accept_is_false(self):
        parsed, error, _ = parse_structured_response(
            self._response(accept=False, block_reason=None),
            line_ids=LINE_IDS,
            label_set=LABEL_SET,
            public_message_min_chars=50,
            public_message_max_chars=500,
            forbidden_public_tokens=["PRIVATE"],
        )
        self.assertIsNone(parsed)
        self.assertIn("block_reason", error)

    def test_leakage_heuristic_triggers_on_marker_tokens(self):
        result = detect_leakage(
            "This PUBLIC message accidentally says PRIVATE notes.",
            forbidden_tokens=["PRIVATE"],
        )
        self.assertTrue(result["leakage_flag"])
        self.assertIn("PRIVATE", result["leakage_marker_hits"])

    def test_condition_parser_rejects_missing_required_keys(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8") as fh:
            fh.write("condition_id=C3\nmode=negotiation\nconfig_file=config.txt\n")
            fh.flush()
            with self.assertRaises(ValueError):
                read_condition_config(fh.name)

    def test_condition_parser_rejects_unknown_keys(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8") as fh:
            fh.write(
                "\n".join(
                    [
                        "condition_id=C3",
                        "mode=negotiation",
                        "config_file=config.txt",
                        "public_messages=15",
                        "json_max_retries=5",
                        "prior_round0=",
                        "reminder_text=test",
                        "final_turn_announcement_window=1",
                        "public_message_min_chars=350",
                        "public_message_max_chars=1200",
                        "forbidden_public_tokens=PRIVATE",
                        "typo_field=oops",
                    ]
                )
            )
            fh.flush()
            with self.assertRaises(ValueError):
                read_condition_config(fh.name)


if __name__ == "__main__":
    unittest.main()
