"""Addendum 2 (docs/REASONING.md): transition learning (C1-C4) and second-stage RT slowing (D1-D3).

Needs fits hybrid_tl_s{2,3}, recovery_hybrid_tl_s{2,3}, hybrid_s{2,3}, stay_s{2,3}[_odd,_even] in outputs/fits/.
Writes outputs/results_addendum2.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import fit_frame, load_fit, people_study2, recovery_summary  # noqa: E402
from cgd import stats as S  # noqa: E402
from cgd.data import load_seow, load_task  # noqa: E402
from cgd.models import MODELS, STAY  # noqa: E402

OUT = ROOT / "outputs"


def rt_slowing(x, keep=None, fast_ms=150.0) -> float:
    """Median second-stage RT (ms) after rare minus after common transitions, same trial."""
    rt = np.asarray(x.rt2, float)
    ok = (x.a2 >= 0) & np.isfinite(rt) & (rt > fast_ms)
    if keep is not None:
        ok &= keep
    rare, common = ok & (x.common == 0), ok & (x.common == 1)
    if rare.sum() < 5 or common.sum() < 5:
        return np.nan
    return float(np.median(rt[rare]) - np.median(rt[common]))


def one_sided(t: dict, sign: int) -> float:
    from scipy import stats
    z = t["beta"] / t["se"]
    return float(stats.norm.sf(z) if sign > 0 else stats.norm.cdf(z))


def sample(tag: str, task: dict, df: pd.DataFrame) -> dict:
    s = tag[-1]
    res = {}
    # ---- C: transition learning ----
    tl = fit_frame(f"hybrid_tl_s{s}", MODELS["hybrid_tl"])
    if len(tl):
        d = df.join(tl.add_prefix("tl_"))
        d["logit_gamma"] = np.log(d.tl_gamma / (1 - d.tl_gamma))
        d["log_beta_mb"] = np.log(d.tl_beta_mb)
        c1 = S.person_ols(d, "logit_gamma", ["CIT"])
        res["C1_C2_gamma_CIT"] = {**c1, "p_one_sided_positive": one_sided(c1["terms"]["CIT"], +1)}
        res["C1_gamma_AD_CIT_SW"] = S.person_ols(d, "logit_gamma", ["AD", "CIT", "SW"])
        res["C3_gamma_CIT_plus_log_beta_mb"] = S.person_ols(d, "logit_gamma", ["CIT", "log_beta_mb"])
        res["C3_log_beta_mb_CIT"] = S.person_ols(d, "log_beta_mb", ["CIT"])
        res["gamma_summary"] = {"median": float(d.tl_gamma.median()), "q25": float(d.tl_gamma.quantile(.25)),
                                "q75": float(d.tl_gamma.quantile(.75))}
        res["C4_ibic"] = {"hybrid_tl": float(load_fit(f"hybrid_tl_s{s}").ibic),
                          "hybrid": float(load_fit(f"hybrid_s{s}").ibic)}
        res["C4_recovery"] = recovery_summary(f"recovery_hybrid_tl_s{s}", MODELS["hybrid_tl"])
    # ---- D: RT slowing ----
    df = df.copy()
    df["rt_slowing"] = pd.Series({sid: rt_slowing(x) for sid, x in task.items()})
    halves = {}
    for h, f in (("odd", lambda n: np.arange(n) % 2 == 1), ("even", lambda n: np.arange(n) % 2 == 0)):
        halves[h] = pd.Series({sid: rt_slowing(x, f(len(x.a1))) for sid, x in task.items()})
    st = fit_frame(f"stay_s{s}", STAY)
    df = df.join(st[["mb_index"]])
    d1 = S.person_ols(df, "rt_slowing", ["CIT"])
    res["D1_rt_slowing_CIT"] = {**d1, "p_one_sided_negative": one_sided(d1["terms"]["CIT"], -1),
                                "median_slowing_ms": float(df.rt_slowing.median()),
                                "share_slowing_positive": float((df.rt_slowing > 0).mean())}
    res["D1_rt_slowing_AD_CIT_SW"] = S.person_ols(df, "rt_slowing", ["AD", "CIT", "SW"])
    rel = {"rt_slowing": S.split_half(halves["odd"], halves["even"])}
    so, se_ = fit_frame(f"stay_s{s}_odd", STAY), fit_frame(f"stay_s{s}_even", STAY)
    ids = so.index.intersection(se_.index).intersection(halves["odd"].dropna().index).intersection(
        halves["even"].dropna().index)
    rel["mb_index"] = S.split_half(so.loc[ids, "mb_index"], se_.loc[ids, "mb_index"])
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    comp_odd = (z(so.loc[ids, "mb_index"]) + z(halves["odd"].loc[ids])) / 2
    comp_even = (z(se_.loc[ids, "mb_index"]) + z(halves["even"].loc[ids])) / 2
    rel["composite"] = S.split_half(comp_odd, comp_even)
    rel["r_mb_index_rt_slowing"] = float(df[["mb_index", "rt_slowing"]].corr().iloc[0, 1])
    res["D2_reliability_odd_even"] = rel
    # D3: both measures together; composite
    df["mb_z"], df["rt_z"] = z(df.mb_index), z(df.rt_slowing)
    df["composite"] = (df.mb_z + df.rt_z) / 2
    res["D3_CIT_on_each"] = {o: S.person_ols(df, o, ["CIT"]) for o in ["mb_index", "rt_slowing", "composite"]}
    # CIT ~ both measures (which carries information about compulsivity?)
    d = df.dropna(subset=["CIT", "mb_z", "rt_z", "age", "iq", "gender"])
    import statsmodels.api as sm
    X = sm.add_constant(d[["mb_z", "rt_z", "age", "iq", "gender"]].apply(S.zscore))
    m = sm.OLS(S.zscore(d.CIT), X).fit(cov_type="HC3")
    res["D3_CIT_on_both"] = {k: {"beta": float(m.params[k]), "p": float(m.pvalues[k])} for k in ["mb_z", "rt_z"]}
    res["n"] = int(df.rt_slowing.notna().sum())
    return res


def main():
    df2, _ = people_study2()
    task2 = load_task(2, df2.index)
    task3, df3 = load_seow()
    res = {"pre_registration": "docs/REASONING.md#addendum-2-2026-10-29-transition-learning-and-a-reaction-time-marker",
           "study2": sample("s2", task2, df2), "seow2021": sample("s3", task3, df3)}
    (OUT / "results_addendum2.json").write_text(json.dumps(res, indent=2, default=float))
    print("wrote outputs/results_addendum2.json")


if __name__ == "__main__":
    main()
