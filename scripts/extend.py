"""Addendum 1 (docs/REASONING.md): glmer-style mixed models on Study 1/2 and the Seow et al. 2021 sample.

Writes outputs/results_addendum.json. Part B runs only if data/raw/seow2021/ exists.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import QUESTIONNAIRES, fit_frame, load_fit, people_study2, recovery_summary  # noqa: E402
from cgd import stats as S  # noqa: E402
from cgd.data import SEOW, load_seow, load_self_report, load_task, quality, raw_mb_index, stay_design  # noqa: E402
from cgd.models import MODELS, STAY  # noqa: E402

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


def odd_even_raw(x) -> tuple[float, float]:
    d = stay_design(x)
    out = []
    for keep in (d.trial % 2 == 1, d.trial % 2 == 0):
        h = d[keep]
        p = lambda rr, tt: h.stay[(h.reward == rr) & (h.transition == tt)].mean()  # noqa: E731
        out.append((p(1, 1) - p(1, -1)) - (p(-1, 1) - p(-1, -1)))
    return tuple(out)


def part_b(study2: dict) -> dict:
    task, df = load_seow()
    q = pd.DataFrame({sid: quality(x) for sid, x in task.items()}).T
    df = df.join(q)
    df["raw_mb_index"] = pd.Series({sid: raw_mb_index(x) for sid, x in task.items()})
    cols = ["AD", "CIT", "SW", "CIT_noAUDIT", "audit_total", "oci_total", "age", "iq", "gender"]
    trials = S.trial_frame(task, df, cols)
    specs = {"B1_AD_CIT_SW": ["AD", "CIT", "SW"], "B1_CIT": ["CIT"], "B3_audit": ["audit_total"],
             "B3_audit_plus_CIT": ["audit_total", "CIT"], "expl_CIT_noAUDIT": ["CIT_noAUDIT"], "expl_oci": ["oci_total"]}
    res = {"n_people": int(len(task)), "n_trial_pairs": int(len(trials)),
           "missed_first_mean": float(df.missed_first.mean()),
           "glmm": glmm_runs(trials, specs),
           "gee": {k: S.gee(trials, v) for k, v in specs.items() if k.startswith("B1")}}
    # authors' own model (no covariates), for comparison with their reported estimate
    saved = S.COVARIATES[:]
    S.COVARIATES[:] = []
    res["glmm_no_covariates_AD_CIT_SW"] = S.glmm(trials, ["AD", "CIT", "SW"])
    S.COVARIATES[:] = saved

    # B1 verdict
    t = res["glmm"]["B1_AD_CIT_SW"]["terms"]["reward:transition:CIT"]
    s2 = study2["H1_AD_CIT_SW"]["terms"]["reward:transition:CIT"]
    p_one = float(stats.norm.cdf(t["z"]))
    diff_z = (t["beta"] - s2["beta"]) / np.hypot(t["se"], s2["se"])
    p_diff = float(2 * stats.norm.sf(abs(diff_z)))
    verdict = ("replicated" if p_one < 0.05 else "inconsistent" if p_diff < 0.05 else
               "consistent" if t["ci_low"] <= s2["beta"] <= t["ci_high"] else "not significant; CI excludes Study 2")
    res["B1"] = {"beta": t["beta"], "se": t["se"], "z": t["z"], "p_one_sided": p_one,
                 "study2_beta": s2["beta"], "diff_z": float(diff_z), "p_diff": p_diff, "verdict": verdict}
    # B2 pooled fixed-effect
    w = np.array([1 / t["se"] ** 2, 1 / s2["se"] ** 2])
    b = np.array([t["beta"], s2["beta"]])
    pooled = float((w * b).sum() / w.sum())
    se_p = float(1 / np.sqrt(w.sum()))
    Q = float((w * (b - pooled) ** 2).sum())
    res["B2_pooled"] = {"beta": pooled, "se": se_p, "z": pooled / se_p, "p": float(2 * stats.norm.sf(abs(pooled / se_p))),
                        "Q": Q, "p_Q": float(stats.chi2.sf(Q, 1))}
    # B4 reliability
    halves = pd.DataFrame({sid: odd_even_raw(x) for sid, x in task.items()}, index=["odd", "even"]).T
    rel = {"raw_mb_index": S.split_half(halves.odd, halves.even)}
    so, se_ = fit_frame("stay_s3_odd", STAY), fit_frame("stay_s3_even", STAY)
    if len(so) and len(se_):
        ids = so.index.intersection(se_.index)
        rel["mb_index"] = S.split_half(so.loc[ids, "mb_index"], se_.loc[ids, "mb_index"])
    rel["CIT_items"] = S.split_half(df.CIT_odd_items, df.CIT_even_items)
    res["B4_reliability_odd_even"] = rel
    # exploratory: person-level models, incl. the Study 2 post-hoc beta_MB finding
    st, hy = fit_frame("stay_s3", STAY), fit_frame("hybrid_s3", MODELS["hybrid"])
    d = df.join(st[["mb_index"]]) if len(st) else df
    if len(hy):
        d = d.join(hy.add_prefix("rl_"))
        d["rl_log_beta_mb"], d["rl_log_beta_mf"] = np.log(d.rl_beta_mb), np.log(d.rl_beta_mf)
        res["ibic_hybrid"] = float(load_fit("hybrid_s3").ibic)
        res["recovery_hybrid"] = recovery_summary("recovery_hybrid_s3", MODELS["hybrid"])
    res["expl_person_level"] = {o: S.person_ols(d, o, ["AD", "CIT", "SW"]) for o in
                                ["mb_index", "raw_mb_index", "rl_w", "rl_log_beta_mb", "rl_log_beta_mf"] if o in d}
    res["expl_person_level_ci"] = {o: {s: [v["ci_low"], v["ci_high"]] for s, v in m["terms"].items()}
                                   for o, m in res["expl_person_level"].items()}
    res["r_CIT_raw_mb_index"] = float(d[["CIT", "raw_mb_index"]].corr().iloc[0, 1])
    return res


def main():
    path = OUT / "results_addendum.json"
    res = json.loads(path.read_text()) if path.exists() and "--redo-a" not in sys.argv else {}
    res["pre_registration"] = "docs/REASONING.md#addendum-1-2026-10-29-mixed-model-check-and-an-independent-sample"
    if "glmm_study2" not in res:
        res.update(part_a())
    if SEOW.exists():
        res["seow2021"] = part_b(res["glmm_study2"])
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=2, default=float))
    print("wrote outputs/results_addendum.json")


if __name__ == "__main__":
    main()
