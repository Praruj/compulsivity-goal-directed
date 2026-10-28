"""Choice models (log-likelihood + simulation), parameters on an unbounded scale.

The two-step hybrid model is adapted from C:/dev/self-control-phenotypes (src/scp/models.py), with two additions:
a per-trial likelihood mask (so split halves can be fitted on interleaved trials while the learner still sees
every trial) and simulation on each person's actual reward drifts. The stay-regression model lets the same
hierarchical fitter give empirical-Bayes model-based / model-free indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit

from .data import TwoStep, stay_design


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Model:
    name: str
    params: list[str]
    nll: Callable
    natural: Callable          # theta (n, p) -> dict of natural-unit arrays
    simulate: Callable
    n_obs: Callable            # data -> number of modelled choices


# ---------- Daw et al. 2011 hybrid ----------
@njit(cache=True)
def _nll_twostep(th, a1, s2, a2, r, mask, use_mb, use_mf):
    alpha = 1.0 / (1.0 + np.exp(-th[0]))
    b_mb = np.exp(th[1]) if use_mb else 0.0
    j = 2 if use_mb else 1
    b_mf = np.exp(th[j]) if use_mf else 0.0
    if use_mf:
        j += 1
    b2, pers = np.exp(th[j]), th[j + 1]
    q1 = np.zeros(2)
    q2 = np.full((2, 2), 0.5)
    prev = -1
    nll = 0.0
    for t in range(len(a1)):                    # scalar arithmetic only: no allocation inside the loop
        if a1[t] < 0:
            continue
        m0, m1 = max(q2[0, 0], q2[0, 1]), max(q2[1, 0], q2[1, 1])
        u0 = b_mb * (0.7 * m0 + 0.3 * m1) + b_mf * q1[0]
        u1 = b_mb * (0.7 * m1 + 0.3 * m0) + b_mf * q1[1]
        if prev == 0:
            u0 += pers
        elif prev == 1:
            u1 += pers
        m = max(u0, u1)
        nll -= mask[t] * ((u1 if a1[t] == 1 else u0) - m - np.log(np.exp(u0 - m) + np.exp(u1 - m)))
        prev = a1[t]
        if a2[t] < 0:
            continue
        s = s2[t]
        v0, v1 = b2 * q2[s, 0], b2 * q2[s, 1]
        m = max(v0, v1)
        nll -= mask[t] * ((v1 if a2[t] == 1 else v0) - m - np.log(np.exp(v0 - m) + np.exp(v1 - m)))
        d1 = q2[s, a2[t]] - q1[a1[t]]
        d2 = r[t] - q2[s, a2[t]]
        q1[a1[t]] += alpha * (d1 + d2)          # lambda = 1
        q2[s, a2[t]] += alpha * d2
    return nll


def drifting_probs(n, rng):
    probs = np.empty((n, 2, 2)); p = rng.uniform(.25, .75, (2, 2))
    for t in range(n):
        p = np.clip(p + rng.normal(0, .025, (2, 2)), .25, .75); probs[t] = p
    return probs


def _sim_twostep(th, x: TwoStep, rng, use_mb=True, use_mf=True):
    alpha = sigmoid(th[0])
    b_mb = np.exp(th[1]) if use_mb else 0.0
    j = 2 if use_mb else 1
    b_mf = np.exp(th[j]) if use_mf else 0.0
    j += 1 if use_mf else 0
    b2, pers = np.exp(th[j]), th[j + 1]
    n = len(x.a1)
    probs = x.probs if x.probs is not None else drifting_probs(n, rng)
    q1, q2 = np.zeros(2), np.full((2, 2), 0.5)
    a1, s2, a2, r, c = (np.zeros(n, np.int64) for _ in range(5))
    prev = -1
    for t in range(n):
        mb = np.array([.7 * q2[a].max() + .3 * q2[1 - a].max() for a in range(2)])
        u = b_mb * mb + b_mf * q1
        if prev >= 0:
            u[prev] += pers
        pa = np.exp(u - u.max()); pa /= pa.sum()
        a1[t] = rng.random() < pa[1]
        c[t] = rng.random() < 0.7
        s2[t] = a1[t] if c[t] else 1 - a1[t]
        v = b2 * q2[s2[t]]; pv = np.exp(v - v.max()); pv /= pv.sum()
        a2[t] = rng.random() < pv[1]
        r[t] = rng.random() < probs[t, s2[t], a2[t]]
        d1 = q2[s2[t], a2[t]] - q1[a1[t]]; d2 = r[t] - q2[s2[t], a2[t]]
        q1[a1[t]] += alpha * (d1 + d2); q2[s2[t], a2[t]] += alpha * d2
        prev = a1[t]
    return TwoStep(a1, s2, a2, r, c, probs, mask=x.mask)


def _ts(use_mb, use_mf):
    names = ["alpha"] + (["beta_mb"] if use_mb else []) + (["beta_mf"] if use_mf else []) + ["beta2", "persev"]

    def natural(T):
        out = {"alpha": sigmoid(T[:, 0])}
        j = 1
        if use_mb:
            out["beta_mb"] = np.exp(T[:, j]); j += 1
        if use_mf:
            out["beta_mf"] = np.exp(T[:, j]); j += 1
        out["beta2"], out["persev"] = np.exp(T[:, j]), T[:, j + 1]
        if use_mb and use_mf:
            out["w"] = out["beta_mb"] / (out["beta_mb"] + out["beta_mf"])
        return out

    tag = "hybrid" if use_mb and use_mf else ("model_based" if use_mb else "model_free")
    return Model(tag, names,
                 lambda th, x: _nll_twostep(th, x.a1, x.s2, x.a2, x.r, x.mask.astype(np.float64), use_mb, use_mf),
                 natural, lambda th, x, rng: _sim_twostep(th, x, rng, use_mb, use_mf),
                 lambda x: int(((x.a1 >= 0) * (x.mask > 0)).sum() + ((x.a2 >= 0) * (x.mask > 0)).sum()))


# ---------- hybrid with incremental transition learning (Sharp, Dolan & Eldar 2023) ----------
@njit(cache=True)
def _nll_twostep_tl(th, a1, s2, a2, r, mask):
    """Hybrid whose model-based system learns p_a = P(own state | a) from 0.7 with rate gamma (chosen action only)."""
    alpha = 1.0 / (1.0 + np.exp(-th[0]))
    b_mb, b_mf, b2, pers = np.exp(th[1]), np.exp(th[2]), np.exp(th[3]), th[4]
    gamma = 1.0 / (1.0 + np.exp(-th[5]))
    q1 = np.zeros(2)
    q2 = np.full((2, 2), 0.5)
    p0, p1 = 0.7, 0.7
    prev = -1
    nll = 0.0
    for t in range(len(a1)):
        if a1[t] < 0:
            continue
        m0, m1 = max(q2[0, 0], q2[0, 1]), max(q2[1, 0], q2[1, 1])
        u0 = b_mb * (p0 * m0 + (1 - p0) * m1) + b_mf * q1[0]
        u1 = b_mb * (p1 * m1 + (1 - p1) * m0) + b_mf * q1[1]
        if prev == 0:
            u0 += pers
        elif prev == 1:
            u1 += pers
        m = max(u0, u1)
        nll -= mask[t] * ((u1 if a1[t] == 1 else u0) - m - np.log(np.exp(u0 - m) + np.exp(u1 - m)))
        prev = a1[t]
        own = 1.0 if s2[t] == a1[t] else 0.0
        if a1[t] == 0:
            p0 += gamma * (own - p0)
        else:
            p1 += gamma * (own - p1)
        if a2[t] < 0:
            continue
        s = s2[t]
        v0, v1 = b2 * q2[s, 0], b2 * q2[s, 1]
        m = max(v0, v1)
        nll -= mask[t] * ((v1 if a2[t] == 1 else v0) - m - np.log(np.exp(v0 - m) + np.exp(v1 - m)))
        d1 = q2[s, a2[t]] - q1[a1[t]]
        d2 = r[t] - q2[s, a2[t]]
        q1[a1[t]] += alpha * (d1 + d2)
        q2[s, a2[t]] += alpha * d2
    return nll


def _sim_twostep_tl(th, x: TwoStep, rng):
    alpha, gamma = sigmoid(th[0]), sigmoid(th[5])
    b_mb, b_mf, b2, pers = np.exp(th[1]), np.exp(th[2]), np.exp(th[3]), th[4]
    n = len(x.a1)
    probs = x.probs if x.probs is not None else drifting_probs(n, rng)
    q1, q2, pt = np.zeros(2), np.full((2, 2), 0.5), np.array([0.7, 0.7])
    a1, s2, a2, r, c = (np.zeros(n, np.int64) for _ in range(5))
    prev = -1
    for t in range(n):
        mb = np.array([pt[a] * q2[a].max() + (1 - pt[a]) * q2[1 - a].max() for a in range(2)])
        u = b_mb * mb + b_mf * q1
        if prev >= 0:
            u[prev] += pers
        pa = np.exp(u - u.max()); pa /= pa.sum()
        a1[t] = rng.random() < pa[1]
        c[t] = rng.random() < 0.7
        s2[t] = a1[t] if c[t] else 1 - a1[t]
        pt[a1[t]] += gamma * (c[t] - pt[a1[t]])
        v = b2 * q2[s2[t]]; pv = np.exp(v - v.max()); pv /= pv.sum()
        a2[t] = rng.random() < pv[1]
        r[t] = rng.random() < probs[t, s2[t], a2[t]]
        d1 = q2[s2[t], a2[t]] - q1[a1[t]]; d2 = r[t] - q2[s2[t], a2[t]]
        q1[a1[t]] += alpha * (d1 + d2); q2[s2[t], a2[t]] += alpha * d2
        prev = a1[t]
    return TwoStep(a1, s2, a2, r, c, probs, mask=x.mask)


def _natural_tl(T):
    out = {"alpha": sigmoid(T[:, 0]), "beta_mb": np.exp(T[:, 1]), "beta_mf": np.exp(T[:, 2]),
           "beta2": np.exp(T[:, 3]), "persev": T[:, 4], "gamma": sigmoid(T[:, 5])}
    out["w"] = out["beta_mb"] / (out["beta_mb"] + out["beta_mf"])
    return out


HYBRID_TL = Model("hybrid_tl", ["alpha", "beta_mb", "beta_mf", "beta2", "persev", "gamma"],
                  lambda th, x: _nll_twostep_tl(th, x.a1, x.s2, x.a2, x.r, x.mask.astype(np.float64)),
                  _natural_tl, _sim_twostep_tl,
                  lambda x: int(((x.a1 >= 0) * (x.mask > 0)).sum() + ((x.a2 >= 0) * (x.mask > 0)).sum()))


# ---------- stay regression (model-agnostic) ----------
@dataclass
class Stay:
    X: np.ndarray   # (n, 4): intercept, reward, transition, reward x transition
    y: np.ndarray


def stay_data(x: TwoStep, keep: np.ndarray | None = None) -> Stay:
    d = stay_design(x)
    if keep is not None:
        d = d[keep[d.trial.to_numpy()]]
    X = np.column_stack([np.ones(len(d)), d.reward, d.transition, d.reward * d.transition]).astype(float)
    return Stay(X, d.stay.to_numpy(float))


def _nll_logistic(th, s: Stay):
    z = s.X @ th
    return float(np.sum(np.logaddexp(0, z) - s.y * z))


def _sim_logistic(th, s: Stay, rng):
    return Stay(s.X, (rng.random(len(s.y)) < sigmoid(s.X @ th)).astype(float))


STAY = Model("stay_logistic", ["intercept", "reward", "transition", "mb"], _nll_logistic,
             lambda T: {"stay_bias": T[:, 0], "mf_index": T[:, 1], "transition": T[:, 2], "mb_index": T[:, 3]},
             _sim_logistic, lambda s: len(s.y))

MODELS = {"hybrid": _ts(True, True), "model_free": _ts(False, True), "model_based": _ts(True, False),
          "hybrid_tl": HYBRID_TL}
