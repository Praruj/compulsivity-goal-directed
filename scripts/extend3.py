"""Addendum 3 (docs/REASONING.md): joint choice + RT model (RL-DDM), E1-E5. Writes outputs/results_addendum3.json."""
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
from extend2 import one_sided, rt_slowing  # noqa: E402
from cgd import stats as S  # noqa: E402
from cgd.data import load_seow, load_task, stay_design  # noqa: E402
from cgd.ddm import RLDDM  # noqa: E402
from cgd.models import MODELS, STAY  # noqa: E402

OUT = ROOT / "outputs"
z = lambda v: (v - v.mean()) / v.std()  # noqa: E731


def ddm_frame(job):
    f = fit_frame(job, RLDDM)
    if len(f):
        f["log_v_mb"] = np.log(f.v_mb)
    return f


def predictive_check(job, task, seed=0) -> dict:
    """Simulate each person once from their fitted parameters; compare group stay pattern and RT slowing."""
    f = load_fit(job)
    rng = np.random.default_rng(seed)
    obs, sim = [], []
    for i, sid in enumerate(f.ids):
        x = task[sid]
        for store, y in ((obs, x), (sim, RLDDM.simulate(f.theta[i], x, rng))):
            d = stay_design(y)
            p = d.groupby(["reward", "transition"]).stay.mean()
            store.append({**{f"stay_{'rew' if r > 0 else 'unrew'}_{'common' if t > 0 else 'rare'}": v
                             for (r, t), v in p.items()}, "rt_slowing": rt_slowing(y),
                          "rt1_median": float(np.nanmedian(y.rt1))})
    o, s = pd.DataFrame(obs).mean(), pd.DataFrame(sim).mean()
    return {"observed": o.to_dict(), "simulated": s.to_dict(),
            "rt_slowing_within_25pct": bool(abs(s.rt_slowing - o.rt_slowing) <= 0.25 * abs(o.rt_slowing))}


def sample(tag, task, df) -> dict:
    s = tag[-1]
    res = {}
    full, odd, even = ddm_frame(f"rlddm_s{s}"), ddm_frame(f"rlddm_s{s}_odd"), ddm_frame(f"rlddm_s{s}_even")
    d = df.join(full)
    res["E1_recovery"] = recovery_summary(f"recovery_rlddm_s{s}", RLDDM)
    res["param_medians"] = {k: float(full[k].median()) for k in full.columns}
    e2 = S.person_ols(d, "log_v_mb", ["CIT"])
    res["E2_log_v_mb_CIT"] = {**e2, "p_one_sided_negative": one_sided(e2["terms"]["CIT"], -1)}
    res["E2_log_v_mb_AD_CIT_SW"] = S.person_ols(d, "log_v_mb", ["AD", "CIT", "SW"])
    e3 = S.person_ols(d, "tau", ["CIT"])
    res["E3_tau_CIT"] = {**e3, "p_one_sided_negative": one_sided(e3["terms"]["CIT"], -1)}
    res["E3_tau_AD_CIT_SW"] = S.person_ols(d, "tau", ["AD", "CIT", "SW"])
    # simple markers for comparison
    d["rt_slowing"] = pd.Series({sid: rt_slowing(x) for sid, x in task.items()})
    d = d.join(fit_frame(f"stay_s{s}", STAY)[["mb_index"]])
    hy = fit_frame(f"hybrid_s{s}", MODELS["hybrid"])
    d["hy_log_beta_mb"] = np.log(hy.beta_mb)
    d["composite"] = (z(d.mb_index) + z(d.rt_slowing)) / 2
    d["ddm_composite"] = (z(d.log_v_mb) + z(d.tau)) / 2   # both: more use of task structure
    # E4 reliability
    halves = {h: pd.Series({sid: rt_slowing(x, (np.arange(len(x.a1)) % 2 == (1 if h == "odd" else 0)))
                            for sid, x in task.items()}) for h in ("odd", "even")}
    so, se_ = fit_frame(f"stay_s{s}_odd", STAY), fit_frame(f"stay_s{s}_even", STAY)
    ho, he = fit_frame(f"hybrid_s{s}_odd", MODELS["hybrid"]), fit_frame(f"hybrid_s{s}_even", MODELS["hybrid"])
    ids = odd.index.intersection(even.index).intersection(so.index).intersection(ho.index).intersection(
        halves["odd"].dropna().index).intersection(halves["even"].dropna().index)
    pairs = {"log_v_mb": (odd.log_v_mb, even.log_v_mb), "tau": (odd.tau, even.tau),
             "ddm_composite": ((z(odd.log_v_mb.loc[ids]) + z(odd.tau.loc[ids])) / 2,
                               (z(even.log_v_mb.loc[ids]) + z(even.tau.loc[ids])) / 2),
             "hybrid_log_beta_mb": (np.log(ho.beta_mb), np.log(he.beta_mb)),
             "mb_index": (so.mb_index, se_.mb_index), "rt_slowing": (halves["odd"], halves["even"]),
             "composite": ((z(so.mb_index.loc[ids]) + z(halves["odd"].loc[ids])) / 2,
                           (z(se_.mb_index.loc[ids]) + z(halves["even"].loc[ids])) / 2)}
    res["E4_reliability_odd_even"] = {k: S.split_half(a.loc[ids], b.loc[ids])["spearman_brown"]
                                      for k, (a, b) in pairs.items()}
    res["E4_CIT_effect"] = {o: S.person_ols(d, o, ["CIT"])["terms"]["CIT"] for o in
                            ["log_v_mb", "tau", "ddm_composite", "hy_log_beta_mb", "mb_index", "rt_slowing",
                             "composite"]}
    res["E4_n_reliability"] = int(len(ids))
    if s == "2":
        eff, rel = res["E4_CIT_effect"], res["E4_reliability_odd_even"]
        best = {k: (eff[k]["beta"], rel[k]) for k in ("log_v_mb", "ddm_composite")}
        res["E4_ddm_wins"] = {k: bool(b <= eff["composite"]["beta"] and r >= rel["composite"] - 0.05)
                              for k, (b, r) in best.items()}
    res["r_log_v_mb_hybrid_log_beta_mb"] = float(d[["log_v_mb", "hy_log_beta_mb"]].corr().iloc[0, 1])
    res["r_tau_rt_slowing"] = float(d[["tau", "rt_slowing"]].corr().iloc[0, 1])
    res["E5_predictive_check"] = predictive_check(f"rlddm_s{s}", task)
    return res


def main():
    df2, _ = people_study2()
    task2 = load_task(2, df2.index)
    task3, df3 = load_seow()
    res = {"pre_registration": "docs/REASONING.md#addendum-3-2026-10-29-a-joint-choice--rt-model-rl-ddm",
           "study2": sample("s2", task2, df2), "seow2021": sample("s3", task3, df3)}
    (OUT / "results_addendum3.json").write_text(json.dumps(res, indent=2, default=float))
    print("wrote outputs/results_addendum3.json")


if __name__ == "__main__":
    main()
