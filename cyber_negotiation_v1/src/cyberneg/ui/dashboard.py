from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in root.rglob("manifest.json"):
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            continue
        if not isinstance(manifest, dict) or "run_id" not in manifest:
            continue
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics" / "run_metrics.json"
        run_metrics = None
        if metrics_path.exists():
            try:
                run_metrics = _load_json(metrics_path)
            except Exception:
                run_metrics = None
        row = {
            "manifest_path": str(manifest_path),
            "run_dir": str(run_dir),
            "run_id": manifest.get("run_id"),
            "experiment_id": manifest.get("experiment_id"),
            "condition_id": manifest.get("condition_id"),
            "scenario_id": manifest.get("scenario_id"),
            "mode": manifest.get("mode"),
            "status": manifest.get("status"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "manifest": manifest,
            "run_metrics": run_metrics,
        }
        if isinstance(run_metrics, dict):
            metrics = run_metrics.get("metrics") or {}
            committee_final = run_metrics.get("committee_final") or {}
            row.update(
                {
                    "FinalCorrectType": metrics.get("FinalCorrectType"),
                    "FinalCorrectExact": metrics.get("FinalCorrectExact"),
                    "FinalAgreementExact": metrics.get("FinalAgreementExact"),
                    "WrongConsensusExact": metrics.get("WrongConsensusExact"),
                    "committee_exact_label": committee_final.get("committee_exact_label"),
                    "committee_exact_severity": committee_final.get("committee_exact_severity"),
                }
            )
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("condition_id")), str(r.get("scenario_id")), str(r.get("run_id"))))
    return rows


def _render_dashboard(root: Path) -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Cyber Negotiation V1 Dashboard", layout="wide")
    st.title("Cyber Negotiation V1 Dashboard")

    with st.sidebar:
        st.header("Data")
        root_input = st.text_input("Outputs root", value=str(root))
        debug_private = st.checkbox("Show private fields (debug)", value=False)
        refresh = st.button("Refresh")

    root = Path(root_input)
    if not root.exists():
        st.error(f"Path does not exist: {root}")
        return

    if refresh:
        st.cache_data.clear()

    @st.cache_data(show_spinner=False)
    def load_rows_cached(root_str: str) -> list[dict[str, Any]]:
        return _discover_runs(Path(root_str))

    rows = load_rows_cached(str(root.resolve()))
    if not rows:
        st.warning("No run manifests found under the selected root.")
        return

    df = pd.DataFrame(
        [
            {
                "run_id": r.get("run_id"),
                "condition_id": r.get("condition_id"),
                "scenario_id": r.get("scenario_id"),
                "mode": r.get("mode"),
                "status": r.get("status"),
                "FinalCorrectType": r.get("FinalCorrectType"),
                "FinalCorrectExact": r.get("FinalCorrectExact"),
                "FinalAgreementExact": r.get("FinalAgreementExact"),
                "WrongConsensusExact": r.get("WrongConsensusExact"),
                "committee_exact_label": r.get("committee_exact_label"),
                "committee_exact_severity": r.get("committee_exact_severity"),
                "run_dir": r.get("run_dir"),
            }
            for r in rows
        ]
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Runs", len(df))
    col2.metric("Conditions", int(df["condition_id"].nunique()) if "condition_id" in df else 0)
    col3.metric("Scenarios", int(df["scenario_id"].nunique()) if "scenario_id" in df else 0)

    st.subheader("Run Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    condition_options = ["(all)"] + sorted({str(r.get("condition_id")) for r in rows})
    selected_condition = st.selectbox("Condition", condition_options, index=0)
    filtered = rows
    if selected_condition != "(all)":
        filtered = [r for r in filtered if str(r.get("condition_id")) == selected_condition]

    scenario_options = ["(all)"] + sorted({str(r.get("scenario_id")) for r in filtered})
    selected_scenario = st.selectbox("Scenario", scenario_options, index=0)
    if selected_scenario != "(all)":
        filtered = [r for r in filtered if str(r.get("scenario_id")) == selected_scenario]

    if not filtered:
        st.warning("No runs match the selected filters.")
        return

    run_labels = [f"{r['condition_id']} / {r['scenario_id']} / {r['run_id']}" for r in filtered]
    selected_idx = st.selectbox("Run", range(len(filtered)), format_func=lambda i: run_labels[i])
    selected = filtered[selected_idx]
    run_dir = Path(selected["run_dir"])

    st.subheader("Selected Run")
    st.code(str(run_dir))

    tabs = st.tabs(["Overview", "Transcript", "Committee", "Provider Attempts", "Raw JSON"])

    with tabs[0]:
        metrics = (selected.get("run_metrics") or {}).get("metrics") if isinstance(selected.get("run_metrics"), dict) else {}
        committee_final = (
            (selected.get("run_metrics") or {}).get("committee_final")
            if isinstance(selected.get("run_metrics"), dict)
            else {}
        )
        st.write("Final Committee")
        st.json(committee_final or {})
        st.write("Metrics")
        st.json(metrics or {})

    with tabs[1]:
        turns_path = run_dir / "transcript" / "turns.json"
        public_path = run_dir / "transcript" / "public_history.json"
        turns = _load_json(turns_path) if turns_path.exists() else []
        public_history = _load_json(public_path) if public_path.exists() else []

        st.write("Public Timeline")
        if public_history:
            for msg in sorted(public_history, key=lambda x: x.get("public_turn_index", 0)):
                st.markdown(
                    f"**Turn {msg.get('public_turn_index')} · {msg.get('role_id')}**\n\n{msg.get('public_message', '')}"
                )
        else:
            st.caption("No public history (baseline or failed early run).")

        if debug_private:
            st.write("Private / Structured Outputs (Debug)")
            for turn in turns:
                with st.expander(
                    f"{turn.get('turn_id')} · {turn.get('phase')} · {turn.get('role_id')} · {turn.get('status')}"
                ):
                    final_output = turn.get("final_output") or {}
                    st.write("private_notes")
                    st.code(final_output.get("private_notes", ""))
                    st.write("private_plan")
                    st.code(final_output.get("private_plan", ""))
                    st.write("assessment")
                    st.json(final_output.get("assessment", {}))

    with tabs[2]:
        snaps_path = run_dir / "transcript" / "committee_snapshots.json"
        snaps = _load_json(snaps_path) if snaps_path.exists() else []
        st.json(snaps)

    with tabs[3]:
        attempts_path = run_dir / "provider" / "attempts.json"
        attempts = _load_json(attempts_path) if attempts_path.exists() else []
        st.json(attempts)

    with tabs[4]:
        st.write("Manifest")
        st.json(selected.get("manifest") or {})
        st.write("Run Metrics JSON")
        st.json(selected.get("run_metrics") or {})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default="outputs", help="Output root directory to scan")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _render_dashboard(Path(args.root))


if __name__ == "__main__":
    main()

