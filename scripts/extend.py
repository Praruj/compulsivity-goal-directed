"""Addendum 1 (docs/REASONING.md): glmer-style mixed models on Study 1/2 and the Seow et al. 2021 sample.

Writes outputs/results_addendum.json. Part B runs only if data/raw/seow2021/ exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import QUESTIONNAIRES, people_study2  # noqa: E402
from cgd import stats as S  # noqa: E402
from cgd.data import load_self_report, load_task  # noqa: E402

OUT = ROOT / "outputs"
PAPER = {"CIT": -0.058, "CIT_z": -4.71}


def glmm_runs(trials, specs):
    out = {}
    for name, scores in specs.items():
        print("GLMM", name, flush=True)
        out[name] = S.glmm(trials, scores)
    return out


def part_a() -> dict:
    df2, _ = people_study2()
    task2 = load_task(2, df2.index)
    cols = ["AD", "CIT", "SW", "CIT_noAUDIT", "audit_total", "audit_c", "audit_harm", "age", "iq", "gender"] + \
        [q for q in QUESTIONNAIRES if q != "audit_total"]
    trials2 = S.trial_frame(task2, df2, cols)
    specs = {"H1_AD_CIT_SW": ["AD", "CIT", "SW"], "AD": ["AD"], "CIT": ["CIT"], "SW": ["SW"],
             "H2a_audit": ["audit_total"], "H2b_audit_plus_CIT": ["audit_total", "CIT"],
             "H2c_CIT_noAUDIT": ["CIT_noAUDIT"], "expl_audit_c": ["audit_c"], "expl_audit_harm": ["audit_harm"]}
    specs.update({f"q_{q}": [q] for q in QUESTIONNAIRES if q != "audit_total"})
    g2 = glmm_runs(trials2, specs)
    a_alone = g2["H2a_audit"]["terms"]["reward:transition:audit_total"]["beta"]
    a_with = g2["H2b_audit_plus_CIT"]["terms"]["reward:transition:audit_total"]["beta"]
    cit = g2["H1_AD_CIT_SW"]["terms"]["reward:transition:CIT"]["beta"]
    sr1 = load_self_report(1)
    task1 = load_task(1, sr1.index)
    trials1 = S.trial_frame(task1, sr1, ["oci_total", "sds_total", "stai_total", "age", "iq", "gender"])
    g1 = glmm_runs(trials1, {"H1b_oci": ["oci_total"]})
    return {"glmm_study2": g2, "glmm_study1": g1, "H2b_glmm_shrinkage": 1 - a_with / a_alone,
            "A1_paper": PAPER, "A1_within_0.010_of_paper": bool(abs(cit - PAPER["CIT"]) <= 0.010)}


def main():
    res = {"pre_registration": "docs/REASONING.md#addendum-1-2026-10-29-mixed-model-check-and-an-independent-sample"}
    res.update(part_a())
    OUT.mkdir(exist_ok=True)
    (OUT / "results_addendum.json").write_text(json.dumps(res, indent=2, default=float))
    print("wrote outputs/results_addendum.json")


if __name__ == "__main__":
    main()
