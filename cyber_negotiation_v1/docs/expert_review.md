# Expert Review Exports (Final-Turn Only)

The expert-review CSV is now aligned to the final-turn questionnaire.

## CSV Fields

- `packet_id`
- `condition_id`
- `run_id`
- `final_committee_label`
- `final_committee_severity`
- `final_agreement_exact`
- `history_reference`
- `public_history_reference`
- `q1_finding_type_correctness`
- `q1_reviewer_type`
- `q2_severity_appropriateness`
- `q2_reviewer_severity`
- `q3_evidence_support`
- `q4_report_defensibility`
- `q5_main_issue`
- `q6_minimal_fix_needed`
- `comment`
- `reviewer`

## Intended Values

- `q1_finding_type_correctness`: `Yes`, `No`, `Unclear`
- `q1_reviewer_type`: free text
- `q2_severity_appropriateness`: `Yes`, `Borderline`, `No`
- `q2_reviewer_severity`: `Compliance`, `Info`, `Low`, `Medium`, `High`
- `q3_evidence_support`: `2`, `1`, `0`
- `q4_report_defensibility`: `Yes`, `Borderline`, `No`
- `q5_main_issue`: `None / OK`, `Wrong type`, `Over-severity`, `Under-severity`, `Weak / irrelevant evidence usage`, `Ambiguous packet (not enough evidence)`, `Non-defensible reasoning / missing justification`, `Other`
- `q6_minimal_fix_needed`: `No change`, `Change type`, `Adjust severity`, `Add/replace evidence lines`, `Add clearer justification`, `Needs more data (cannot decide from packet)`, `Reject as false positive`, `Other`
- `comment`: free text
- `reviewer`: free text

## Notes

- Reviewer-answer fields are blank by default for manual completion.
- `history_reference` points to the full run log with private and public content.
- `public_history_reference` points to the public-only transcript export.
