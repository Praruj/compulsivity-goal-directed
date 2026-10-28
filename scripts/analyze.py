"""Run the pre-registered analyses (docs/REASONING.md) and write outputs/results.json.

Needs the fits from scripts/fit_models.py in outputs/fits/. Trial-level GEEs are computed here.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgd import stats as S  # noqa: E402
from cgd.data import (factor_scores, load_items, load_self_report, load_task, quality, raw_mb_index,  # noqa: E402
                      split_items, stay_design)
from cgd.models import MODELS, STAY  # noqa: E402

FITS = ROOT / "outputs" / "fits"
OUT = ROOT / "outputs"
SRO_RETEST = {"w": 0.28, "raw_mb_index": 0.32}          # self-control-phenotypes, ICC(3,1), n = 150, ~4 months
QUESTIONNAIRES = ["sds_total", "stai_total", "oci_total", "lsas_total", "bis_total", "scz_total", "aes_total",
                  "eat_total", "audit_total"]


def load_fit(job):
    p = FITS / f"{job}.pkl"
    if not p.exists():
        print("missing fit:", job); return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def fit_frame(job, model) -> pd.DataFrame:
    f = load_fit(job)
    if f is None:
        return pd.DataFrame()
    nat = model.natural(f.theta)
    return pd.DataFrame({k: np.asarray(v) for k, v in nat.items()}, index=f.ids)


def raw_half(x, keep) -> float:
    d = stay_design(x)
    d = d[keep[d.trial.to_numpy()]]
    p = lambda rr, tt: d.stay[(d.reward == rr) & (d.transition == tt)].mean()  # noqa: E731
    return (p(1, 1) - p(1, -1)) - (p(-1, 1) - p(-1, -1))


def people_study2() -> tuple[pd.DataFrame, dict]:
    sr = load_self_report(2)
    items, weights = load_items()
    items = items.loc[sr.index]
    fs = factor_scores(items, weights)
    # which provided Factor column is which factor: decided by correlation (pre-registered)
    corr = pd.DataFrame({f: [np.corrcoef(fs[f], sr[c])[0, 1] for c in ["Factor1", "Factor2", "Factor3"]]
                         for f in fs.columns}, index=["Factor1", "Factor2", "Factor3"])
    match = {f: corr[f].abs().idxmax() for f in fs.columns}
    match_r = {f: float(corr.at[match[f], f]) for f in fs.columns}
    use_recomputed = all(abs(r) > 0.99 for r in match_r.values())
    df = sr.copy()
    for f in fs.columns:
        df[f] = fs[f] if use_recomputed else sr[match[f]]
    df["CIT_noAUDIT"] = factor_scores(items, weights, drop_prefix="AUDIT")["CIT"]
    odd, even = split_items(weights)
    df["CIT_odd_items"] = factor_scores(items, weights, item_subset=odd)["CIT"]
    df["CIT_even_items"] = factor_scores(items, weights, item_subset=even)["CIT"]
    audit = items[[f"AUDIT_{i}" for i in range(1, 11)]].astype(float)
    df["audit_c"] = audit.iloc[:, :3].sum(1)
    df["audit_harm"] = audit.iloc[:, 3:].sum(1)
    df["hazardous"] = (df.audit_total >= 8).astype(float)
    info = {"factor_match": match, "factor_match_r": match_r, "used_recomputed_scores": use_recomputed,
            "audit_items_sum_vs_total_r": float(np.corrcoef(audit.sum(1), df.audit_total)[0, 1]),
            "n_hazardous": int(df.hazardous.sum()),
            "cit_noaudit_vs_cit_r": float(np.corrcoef(df.CIT_noAUDIT, df.CIT)[0, 1])}
    return df, info


def add_task_measures(df, study, task):
    q = pd.DataFrame({sid: quality(x) for sid, x in task.items()}).T
    df = df.join(q)
    df["raw_mb_index"] = pd.Series({sid: raw_mb_index(x) for sid, x in task.items()})
    st = fit_frame(f"stay_s{study}", STAY)
    hy = fit_frame(f"hybrid_s{study}", MODELS["hybrid"])
    df = df.join(st[["stay_bias", "mf_index", "mb_index"]]) if len(st) else df
    df = df.join(hy.add_prefix("rl_")) if len(hy) else df
    df["sensitivity_keep"] = (df.missed_first <= 0.2) & (df.same_first < 0.95)
    return df


def gee_runs(trials, specs):
    out = {}
    for name, scores in specs.items():
        print("GEE", name, flush=True)
        out[name] = S.gee(trials, scores)
    return out


def recovery_summary(job, model):
    rec = load_fit(job)
    if rec is None:
        return None
    t, r = model.natural(rec["true"]), model.natural(rec["recovered"])
    return {k: float(np.corrcoef(t[k], r[k])[0, 1]) for k in t}


def reliability(task, study):
    """Split-half reliability of the model-agnostic and model-based measures (Spearman-Brown corrected)."""
    out = {}
    for a, b in (("odd", "even"), ("first", "second")):
        res = {}
        sa, sb = fit_frame(f"stay_s{study}_{a}", STAY), fit_frame(f"stay_s{study}_{b}", STAY)
        if len(sa) and len(sb):
            for k in ["mb_index", "mf_index", "stay_bias"]:
                res[k] = S.split_half(sa[k], sb.loc[sa.index, k])
        ha, hb = fit_frame(f"hybrid_s{study}_{a}", MODELS["hybrid"]), fit_frame(f"hybrid_s{study}_{b}",
                                                                                  MODELS["hybrid"])
        if len(ha) and len(hb):
            for k in ["w", "beta_mb", "beta_mf", "alpha", "persev"]:
                v = (lambda s: s) if k in ("alpha", "persev", "w") else np.log
                res[f"rl_{k}"] = S.split_half(v(ha[k]), v(hb.loc[ha.index, k]))
        halves = {"odd": lambda n: np.arange(n) % 2 == 1, "even": lambda n: np.arange(n) % 2 == 0,
                  "first": lambda n: np.arange(n) < 100, "second": lambda n: np.arange(n) >= 100}
        ra = [raw_half(x, halves[a](len(x.a1))) for x in task.values()]
        rb = [raw_half(x, halves[b](len(x.a1))) for x in task.values()]
        res["raw_mb_index"] = S.split_half(ra, rb)
        out[f"{a}_{b}"] = res
    return out


def main():
    res = {"pre_registration": "docs/REASONING.md"}

    # ---------- Study 2 ----------
    df2, info = people_study2()
    task2 = load_task(2, df2.index)
    df2 = add_task_measures(df2, 2, task2)
    res["study2_scores"] = info
    res["n"] = {"study1": None, "study2": int(len(task2)),
                "study2_sensitivity": int(df2.sensitivity_keep.sum())}
    cols = ["AD", "CIT", "SW", "CIT_noAUDIT", "audit_total", "audit_c", "audit_harm", "hazardous", "age", "iq",
            "gender"] + [q for q in QUESTIONNAIRES if q != "audit_total"]
    trials2 = S.trial_frame(task2, df2, cols)
    specs = {"H1_AD_CIT_SW": ["AD", "CIT", "SW"], "AD": ["AD"], "CIT": ["CIT"], "SW": ["SW"],
             "H2a_audit": ["audit_total"], "H2b_audit_plus_CIT": ["audit_total", "CIT"],
             "H2c_CIT_noAUDIT": ["CIT_noAUDIT"], "H2c_CIT_noAUDIT_with_AD_SW": ["AD", "CIT_noAUDIT", "SW"],
             "expl_audit_c": ["audit_c"], "expl_audit_harm": ["audit_harm"], "expl_hazardous": ["hazardous"],
             "expl_audit_c_and_harm": ["audit_c", "audit_harm"]}
    specs.update({f"q_{q}": [q] for q in QUESTIONNAIRES if q != "audit_total"})
    res["gee_study2"] = gee_runs(trials2, specs)
    keep = df2.index[df2.sensitivity_keep]
    res["gee_study2_sensitivity"] = gee_runs(trials2[trials2.subj.isin(keep)], {"H1_AD_CIT_SW": ["AD", "CIT", "SW"]})

    g = res["gee_study2"]
    a_alone = g["H2a_audit"]["terms"]["reward:transition:audit_total"]["beta"]
    a_with = g["H2b_audit_plus_CIT"]["terms"]["reward:transition:audit_total"]["beta"]
    res["H2b"] = {"gee_shrinkage": 1 - a_with / a_alone}
    if "mb_index" in df2:
        res["H2b"]["person_level_bootstrap"] = S.shrinkage_boot(df2.dropna(subset=["mb_index"]), "mb_index",
                                                                "audit_total", "CIT")
    p_h2 = [g["H2a_audit"]["terms"]["reward:transition:audit_total"]["p"],
            g["H2c_CIT_noAUDIT"]["terms"]["reward:transition:CIT_noAUDIT"]["p"]]
    res["H2_bh"] = {"tests": ["H2a", "H2c"], "p": p_h2, "q": S.bh(p_h2).tolist()}

    # person-level (two-stage) models
    pl = {}
    for outcome in ["mb_index", "mf_index", "raw_mb_index", "rl_w", "rl_beta_mb", "rl_beta_mf", "rl_persev"]:
        if outcome in df2:
            d = df2.copy()
            if outcome in ("rl_beta_mb", "rl_beta_mf"):
                d[outcome] = np.log(d[outcome])
            pl[outcome] = {"AD_CIT_SW": S.person_ols(d, outcome, ["AD", "CIT", "SW"]),
                           "audit": S.person_ols(d, outcome, ["audit_total"]),
                           "audit_plus_CIT": S.person_ols(d, outcome, ["audit_total", "CIT"]),
                           "CIT_noAUDIT": S.person_ols(d, outcome, ["CIT_noAUDIT"])}
    res["person_level_study2"] = pl

    # recovery, model comparison
    res["recovery"] = {"stay_s2": recovery_summary("recovery_stay_s2", STAY),
                       "hybrid_s2": recovery_summary("recovery_hybrid_s2", MODELS["hybrid"]),
                       "hybrid_s1": recovery_summary("recovery_hybrid_s1", MODELS["hybrid"])}
    res["ibic_study2"] = {m: float(f.ibic) for m in ["hybrid", "model_free", "model_based"]
                          if (f := load_fit(f"{m}_s2")) is not None}

    # reliability, attenuation, power
    rel = reliability(task2, 2)
    cit_rel = S.split_half(df2.CIT_odd_items, df2.CIT_even_items)
    res["reliability_study2"] = {"task": rel, "CIT_items": cit_rel, "sro_retest_icc": SRO_RETEST}
    att = {}
    for meas, key in [("mb_index", "mb_index"), ("raw_mb_index", "raw_mb_index"), ("rl_w", "rl_w")]:
        if meas not in df2:
            continue
        r_obs = float(df2[[meas, "CIT"]].dropna().corr().iloc[0, 1])
        row = {"r_observed": r_obs, "n_needed_observed": S.n_for_power(r_obs)}
        for split in ("odd_even", "first_second"):
            if key in rel[split]:
                rx = rel[split][key]["spearman_brown"]
                rc = S.disattenuate(r_obs, rx, cit_rel["spearman_brown"])
                row[f"r_corrected_{split}"] = rc
                row[f"rel_{split}"] = rx
        sro = SRO_RETEST.get(meas.replace("rl_", ""))
        if sro:
            row["r_if_retest_reliability"] = S.disattenuate(r_obs, sro, cit_rel["spearman_brown"])
        att[meas] = row
    res["attenuation_study2"] = att

    # prediction of CIT and AUDIT from task behaviour (exploratory)
    feats = [c for c in ["mb_index", "mf_index", "stay_bias", "rl_w", "rl_persev", "rl_alpha", "rt1_median",
                         "rt2_median"] if c in df2]
    df2["gender_"] = df2.gender
    res["prediction_study2"] = {o: S.cv_gain(df2, ["age", "iq", "gender_"], feats, o) for o in ["CIT", "audit_total"]}
    res["prediction_features"] = feats

    # ---------- Study 1 ----------
    sr1 = load_self_report(1)
    task1 = load_task(1, sr1.index)
    df1 = add_task_measures(sr1, 1, task1)
    res["n"]["study1"] = int(len(task1))
    trials1 = S.trial_frame(task1, df1, ["oci_total", "sds_total", "stai_total", "age", "iq", "gender"])
    res["gee_study1"] = gee_runs(trials1, {"H1b_oci": ["oci_total"], "sds": ["sds_total"], "stai": ["stai_total"],
                                           "all_three": ["oci_total", "sds_total", "stai_total"]})
    res["person_level_study1"] = {o: S.person_ols(df1, o, ["oci_total"]) for o in ["mb_index", "rl_w"] if o in df1}
    res["reliability_study1"] = {"raw_only": reliability(task1, 1)}

    # group stay probabilities for the figure: bottom vs top CIT tertile (study 2)
    trials2["CIT_tertile"] = pd.qcut(trials2.CIT, 3, labels=["low", "mid", "high"])
    sp = trials2.groupby(["CIT_tertile", "reward", "transition"], observed=True).stay.mean()
    res["stay_by_cit_tertile"] = {f"{t}|{'rew' if r > 0 else 'unrew'}|{'common' if c > 0 else 'rare'}": float(v)
                                  for (t, r, c), v in sp.items()}

    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    keepcols = ["AD", "CIT", "SW", "CIT_noAUDIT", "audit_total", "mb_index", "mf_index", "raw_mb_index", "rl_w"]
    df2[[c for c in keepcols if c in df2]].to_csv(OUT / "fits" / "person_measures_study2.csv")
    print("wrote outputs/results.json")


if __name__ == "__main__":
    main()
