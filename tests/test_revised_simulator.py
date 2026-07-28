import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_diverse_authenticity_experiment.py"
SPEC = importlib.util.spec_from_file_location("audit_simulator", SCRIPT)
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


def test_ordinal_cut_points():
    assert [SIM.ordinalize(x) for x in (-1, 0.49, 0.5, 1.49, 1.5, 2.49, 2.5, 4)] == [
        0, 0, 1, 1, 2, 2, 3, 3
    ]


def test_profile_states():
    states = {p.name: SIM.profile_status(p) for p in SIM.PROFILES}
    assert sum(v == "documentary" for v in states.values()) == 8
    assert sum(v == "placeholder" for v in states.values()) == 2
    assert sum(v == "rejected" for v in states.values()) == 2
    for profile in SIM.PROFILES:
        expected = 7 if SIM.profile_status(profile) == "documentary" else 5 if SIM.profile_status(profile) == "placeholder" else 0
        assert len(SIM.assessable_criteria(profile)) == expected


def test_archived_records_and_method_level_winner():
    result_dir = ROOT / "results" / "exp02"
    detail = pd.read_csv(result_dir / "scenario_detail.csv")
    summary = pd.read_csv(result_dir / "scenario_summary.csv")
    assert set(detail.record_status) == {"assessable", "rejected_profile"}
    assert (detail.record_status == "assessable").sum() == 480000
    assert (detail.record_status == "rejected_profile").sum() == 96000
    score_columns = [f"score_{c}" for c in SIM.CRITERIA]
    numeric = detail.loc[detail.record_status.eq("assessable"), score_columns].stack().dropna()
    assert set(numeric.astype(int).unique()).issubset({0, 1, 2, 3})

    first = detail[(detail.scenario_id == summary.iloc[0].scenario_id) & detail.profile_status.eq("documentary")]
    means = first.groupby("method").audit_score.mean().sort_values(ascending=False)
    assert summary.iloc[0].winner_method == means.index[0]
    assert abs(summary.iloc[0].winner_margin - (means.iloc[0] - means.iloc[1])) < 1e-3
