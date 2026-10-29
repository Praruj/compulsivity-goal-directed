"""Fast tests on synthetic data: no downloads needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgd import stats as S  # noqa: E402
from cgd.data import TwoStep, factor_scores, parse_task_file, raw_mb_index, split_items, stay_design  # noqa: E402
from cgd.fit import fit  # noqa: E402
from cgd.models import MODELS, STAY, drifting_probs, stay_data  # noqa: E402

RNG = np.random.default_rng(0)
MB_AGENT = np.array([0.0, np.log(6), np.log(0.01), np.log(4), 0.0])
MF_AGENT = np.array([0.0, np.log(0.01), np.log(6), np.log(4), 0.0])


def blank(n=200):
    z = np.zeros(n, np.int64)
    return TwoStep(z, z, z, z, z, drifting_probs(n, RNG))


def test_parse_task_file_maps_columns_and_missed_trials(tmp_path):
    lines = ["AgeDropdown,22", "", "instructionLoop", "id,INSTRUCT,twostep_instruct_9,1000",
             "1,0.7,0.3,0.6,0.4,left,1,500,true,left,1,2,600,1,1",      # stim 1, common -> state 2
             "2,0.7,0.3,0.6,0.4,right,2,450,false,right,2,2,700,0,1",   # stim 2, rare -> state 2
             "3,0.7,0.3,0.6,0.4,-1,2,2502,-1,-1,-1,-1,-1,-1,1",         # missed first stage
             "4,0.7,0.3,0.6,0.4,left,1,480,true,-1,-1,2,2500,-1,1"]     # missed second stage
    p = tmp_path / "s.csv"
    p.write_text("\n".join(lines))
    x = parse_task_file(p)
    assert list(x.a1) == [0, 1, -1, 0]
    assert list(x.common) == [1, 0, 0, 1]
    assert list(x.s2[[0, 1, 3]]) == [0, 0, 0]
    assert list(x.a2) == [0, 1, -1, -1]
    assert list(x.r) == [1, 0, 0, 0]
    assert x.probs[0, 1, 0] == 0.6
    # only pair 1->2 is valid (pair 2->3 has a missed choice, 3->4 a missed outcome)
    assert len(stay_design(x)) == 1


def test_model_based_agent_shows_interaction_model_free_does_not():
    m = MODELS["hybrid"]
    mb = np.mean([raw_mb_index(m.simulate(MB_AGENT, blank(), RNG)) for _ in range(20)])
    mf = np.mean([raw_mb_index(m.simulate(MF_AGENT, blank(), RNG)) for _ in range(20)])
    assert mb > 0.2 and abs(mf) < 0.1


def test_mask_splits_likelihood_exactly():
    m = MODELS["hybrid"]
    x = m.simulate(MB_AGENT, blank(), RNG)
    th = np.array([0.2, 1.0, 0.5, 1.0, 0.3])
    full = m.nll(th, x)
    odd = np.arange(200) % 2 == 1
    halves = []
    for keep in (odd, ~odd):
        x.mask = keep.astype(float); halves.append(m.nll(th, x))
    x.mask = np.zeros(200)
    assert np.isclose(sum(halves), full) and m.nll(th, x) == 0.0


def test_hierarchical_stay_regression_recovers_mb_index():
    n = 80
    true_mb = RNG.normal(0.4, 0.3, n)
    data = {}
    for i in range(n):
        x = blank()
        x.a1 = RNG.integers(0, 2, 200); x.a2 = RNG.integers(0, 2, 200)
        x.r = RNG.integers(0, 2, 200); x.common = (RNG.random(200) < .7).astype(np.int64)
        s = stay_data(x)
        data[f"s{i:03d}"] = STAY.simulate(np.array([0.8, 0.3, 0.0, true_mb[i]]), s, RNG)
    f = fit(STAY, data, n_samples=100)
    order = [int(s[1:]) for s in f.ids]
    assert np.corrcoef(true_mb[order], f.theta[:, 3])[0, 1] > 0.6


def test_gee_finds_planted_symptom_effect():
    rows, people = [], []
    for i in range(150):
        score = RNG.normal()
        people.append({"subj": f"s{i}", "CIT": score, "age": RNG.normal(35, 10), "iq": RNG.normal(100, 15),
                       "gender": RNG.choice([-1, 1])})
        mb = 0.5 - 0.3 * score
        rew, tr = RNG.choice([-1, 1], 200), RNG.choice([-1, 1], 200)
        stay = RNG.random(200) < 1 / (1 + np.exp(-(0.5 + 0.3 * rew + mb * rew * tr)))
        rows.append(pd.DataFrame({"stay": stay.astype(int), "reward": rew, "transition": tr, "subj": f"s{i}"}))
    people = pd.DataFrame(people).set_index("subj")
    t = pd.concat(rows, ignore_index=True)
    z = people.apply(S.zscore)
    for c in z.columns:
        t[c] = t.subj.map(z[c])
    res = S.gee(t, ["CIT"])["terms"]["reward:transition:CIT"]
    assert res["beta"] < 0 and res["p"] < 0.001


def test_factor_scores_and_item_split():
    items = pd.DataFrame(RNG.normal(size=(50, 4)), columns=["OCI_1", "OCI_2", "AUDIT_1", "AUDIT_2"])
    w = pd.DataFrame({"CIT": [1.0, 1.0, 1.0, 1.0]}, index=items.columns)
    full = factor_scores(items, w)["CIT"]
    no_audit = factor_scores(items, w, drop_prefix="AUDIT")["CIT"]
    z = (items - items.mean()) / items.std(ddof=0)
    assert np.allclose(full, z.sum(1)) and np.allclose(no_audit, z[["OCI_1", "OCI_2"]].sum(1))
    odd, even = split_items(w)
    assert odd == ["OCI_1", "AUDIT_1"] and even == ["OCI_2", "AUDIT_2"]


def test_reliability_helpers():
    assert np.isclose(S.spearman_brown(0.5), 2 / 3)
    assert np.isclose(S.disattenuate(0.1, 0.25, 1.0), 0.2)
    assert 770 <= S.n_for_power(0.1) <= 790        # textbook value ~783
    q = S.bh([0.01, 0.04, 0.03])
    assert np.allclose(q, [0.03, 0.04, 0.04])


def test_glmm_recovers_fixed_and_random_effects():
    rows, people = [], []
    for i in range(400):
        score = RNG.normal()
        people.append({"subj": f"s{i}", "CIT": score, "age": RNG.normal(35, 10), "iq": RNG.normal(100, 15),
                       "gender": RNG.choice([-1, 1])})
        mb = 0.3 - 0.15 * score + RNG.normal(0, 0.25)
        rew, tr = RNG.choice([-1, 1], 200), RNG.choice([-1, 1], 200)
        stay = RNG.random(200) < 1 / (1 + np.exp(-(1 + RNG.normal(0, 0.8) + 0.4 * rew + mb * rew * tr)))
        rows.append(pd.DataFrame({"stay": stay.astype(int), "reward": rew, "transition": tr, "subj": f"s{i}"}))
    people = pd.DataFrame(people).set_index("subj")
    t = pd.concat(rows, ignore_index=True)
    z = people.apply(S.zscore)
    for c in z.columns:
        t[c] = t.subj.map(z[c])
    g = S.glmm(t, ["CIT"])
    planted = -0.15 * people.CIT.std()
    assert g["converged"]
    assert abs(g["terms"]["reward:transition:CIT"]["beta"] - planted) < 3 * g["terms"]["reward:transition:CIT"]["se"]
    assert abs(g["terms"]["reward"]["beta"] - 0.4) < 0.05
    assert abs(g["random_sd"]["Intercept"] - 0.8) < 0.15 and abs(g["random_sd"]["reward:transition"] - 0.25) < 0.1


def test_seow_coding():
    from cgd.data import seow_from_frame
    d = pd.DataFrame({"nTrial": [1, 2, 3, 4], "r1": .7, "r2": .3, "r3": .6, "r4": .4,
                      "S1c": [1, 2, np.nan, 2], "S1rt": [0.5, 0.6, np.nan, 0.1], "trans": [1, 1, 2, 2],
                      "S2s": [2, 3, np.nan, 2], "S2c": [4, 5, 0, 3], "S2rt": [.5, .5, np.nan, .5],
                      "hit": [1, 0, np.nan, 1]})
    x = seow_from_frame(d)
    assert x.a1.tolist() == [0, 1, -1, -1]          # trial 4 is faster than 150 ms
    assert x.s2.tolist()[:2] == [0, 1] and x.a2.tolist() == [1, 0, -1, -1]
    assert x.r.tolist() == [1, 0, 0, 0] and x.common.tolist()[:2] == [1, 1]
    assert x.probs.shape == (4, 2, 2) and x.probs[0, 1, 0] == 0.6


def test_transition_learning_reduces_to_hybrid_and_simulates():
    from cgd.models import MODELS
    rng = np.random.default_rng(3)
    th = np.array([0.0, 1.0, 0.3, 1.0, 0.2])
    x = MODELS["hybrid"].simulate(th, TwoStep(*(np.zeros(200, np.int64) for _ in range(5))), rng)
    frozen = np.append(th, -40.0)                      # gamma ~ 0: beliefs stay at 0.7
    assert abs(MODELS["hybrid_tl"].nll(frozen, x) - MODELS["hybrid"].nll(th, x)) < 1e-8
    assert MODELS["hybrid_tl"].nll(np.append(th, 0.0), x) != MODELS["hybrid"].nll(th, x)
    y = MODELS["hybrid_tl"].simulate(np.append(th, -1.0), x, rng)
    assert set(np.unique(y.a1)) <= {0, 1} and 0.6 < y.common.mean() < 0.8


def test_wiener_density_and_rlddm():
    from scipy.integrate import quad
    from cgd.ddm import RLDDM, wiener_lower
    for v, a in [(0.0, 1.0), (1.5, 1.2), (-2.0, 2.0)]:
        up = quad(lambda t: wiener_lower(t, -v, a, 0.5), 0, 20, limit=400)[0]
        lo = quad(lambda t: wiener_lower(t, v, a, 0.5), 0, 20, limit=400)[0]
        assert abs(up + lo - 1) < 1e-5 and abs(up - 1 / (1 + np.exp(-v * a))) < 1e-5
    rng = np.random.default_rng(4)
    z = np.zeros(200, np.int64)
    x0 = TwoStep(z, z, z, z, z, drifting_probs(200, rng), np.full(200, 400.0), np.full(200, 400.0))
    th = np.array([0.0, 1.0, 0.0, 0.3, 0.2, 0.5, 1.2, 0.2, 0.5, 1.0])     # tau = 0.1 s
    x = RLDDM.simulate(th, x0, rng)
    slow = np.median(x.rt2[x.common == 0]) - np.median(x.rt2[x.common == 1])
    assert 50 < slow < 150
    assert np.isfinite(RLDDM.nll(th, x)) and RLDDM.nll(th, x) < RLDDM.nll(np.append(th[:9], -5.0), x)
