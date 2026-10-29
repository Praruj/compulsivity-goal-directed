"""Reinforcement-learning drift-diffusion model (RL-DDM) of the two-step task: choices and RTs together.

Following Shahar et al. (2019), each choice is a Wiener diffusion between two boundaries, with the drift set by the
learned values. Stage 1: drift = v_mb * (MB value difference) + v_mf * (MF value difference) + perseveration,
boundary a1, non-decision time t1. Stage 2: drift = v2 * (Q2 difference), boundary a2, non-decision time t2, plus
tau, an extra delay after a rare transition (the surprise slowing, Seow et al. 2021). A 2% uniform lapse keeps the
likelihood finite for outlying RTs. Values are learned exactly as in the hybrid model (lambda = 1).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .data import TwoStep
from .models import Model, drifting_probs, sigmoid

LAPSE = 0.02
TAU_SCALE = 0.1          # tau = 0.1 * theta seconds, so theta is on a unit-ish scale


@njit(cache=True)
def wiener_lower(t, v, a, w, err=1e-7):
    """First-passage density at the lower boundary (Navarro & Fuss 2009), drift v, boundary a, start w*a."""
    if t <= 0.0:
        return 0.0
    tt = t / (a * a)
    if np.pi * tt * err < 1.0:
        kl = np.sqrt(-2.0 * np.log(np.pi * tt * err) / (np.pi * np.pi * tt))
        kl = max(kl, 1.0 / (np.pi * np.sqrt(tt)))
    else:
        kl = 1.0 / (np.pi * np.sqrt(tt))
    if 2.0 * np.sqrt(2.0 * np.pi * tt) * err < 1.0:
        ks = 2.0 + np.sqrt(-2.0 * tt * np.log(2.0 * np.sqrt(2.0 * np.pi * tt) * err))
        ks = max(ks, np.sqrt(tt) + 1.0)
    else:
        ks = 2.0
    p = 0.0
    if ks < kl:
        K = int(np.ceil(ks))
        for k in range(-((K - 1) // 2), K // 2 + 1):
            p += (w + 2 * k) * np.exp(-((w + 2 * k) ** 2) / (2.0 * tt))
        p /= np.sqrt(2.0 * np.pi * tt ** 3)
    else:
        K = int(np.ceil(kl))
        for k in range(1, K + 1):
            p += k * np.exp(-(k * k) * (np.pi * np.pi) * tt / 2.0) * np.sin(k * np.pi * w)
        p *= np.pi
    return max(p, 0.0) * np.exp(-v * a * w - v * v * t / 2.0) / (a * a)


@njit(cache=True)
def choice_density(choice, rt, v, a, t0, window):
    """Density of choosing option `choice` (1 = upper boundary) at `rt`, mixed with a uniform lapse."""
    dt = rt - t0
    if choice == 1:
        f = wiener_lower(dt, -v, a, 0.5)
    else:
        f = wiener_lower(dt, v, a, 0.5)
    return (1.0 - LAPSE) * f + LAPSE * 0.5 / window


@njit(cache=True)
def _params(th, tmax1, tmax2):
    th = np.minimum(np.maximum(th, -8.0), 8.0)            # keeps exp() and boundaries in a finite range
    alpha = 1.0 / (1.0 + np.exp(-th[0]))
    v_mb, v_mf, pers = np.exp(th[1]), np.exp(th[2]), th[3]
    a1, t1 = np.exp(min(max(th[4], -3.0), 3.0)), tmax1 / (1.0 + np.exp(-th[5]))
    v2, a2, t2 = np.exp(th[6]), np.exp(min(max(th[7], -3.0), 3.0)), tmax2 / (1.0 + np.exp(-th[8]))
    tau = TAU_SCALE * th[9]
    return alpha, v_mb, v_mf, pers, a1, t1, v2, a2, t2, tau


@njit(cache=True)
def _nll_rlddm(th, a1c, s2, a2c, r, common, rt1, rt2, mask, tmax1, tmax2, win1, win2):
    alpha, v_mb, v_mf, pers, b1, t1, v2, b2, t2, tau = _params(th, tmax1, tmax2)
    q1 = np.zeros(2)
    q2 = np.full((2, 2), 0.5)
    prev = -1
    nll = 0.0
    for t in range(len(a1c)):
        if a1c[t] < 0:
            continue
        m0, m1 = max(q2[0, 0], q2[0, 1]), max(q2[1, 0], q2[1, 1])
        mb = (0.7 * m1 + 0.3 * m0) - (0.7 * m0 + 0.3 * m1)
        drift = v_mb * mb + v_mf * (q1[1] - q1[0])
        if prev == 1:
            drift += pers
        elif prev == 0:
            drift -= pers
        if np.isfinite(rt1[t]):
            nll -= mask[t] * np.log(choice_density(a1c[t], rt1[t], drift, b1, t1, win1) + 1e-300)
        else:                                   # no RT: choice probability only
            pu = 1.0 / (1.0 + np.exp(-drift * b1))
            nll -= mask[t] * np.log((pu if a1c[t] == 1 else 1.0 - pu) + 1e-300)
        prev = a1c[t]
        if a2c[t] < 0:
            continue
        s = s2[t]
        d2 = v2 * (q2[s, 1] - q2[s, 0])
        t0 = t2 + (tau if common[t] == 0 else 0.0)
        if t0 < 0.0:
            t0 = 0.0
        if np.isfinite(rt2[t]):
            nll -= mask[t] * np.log(choice_density(a2c[t], rt2[t], d2, b2, t0, win2) + 1e-300)
        else:
            pu = 1.0 / (1.0 + np.exp(-d2 * b2))
            nll -= mask[t] * np.log((pu if a2c[t] == 1 else 1.0 - pu) + 1e-300)
        dq1 = q2[s, a2c[t]] - q1[a1c[t]]
        dq2 = r[t] - q2[s, a2c[t]]
        q1[a1c[t]] += alpha * (dq1 + dq2)
        q2[s, a2c[t]] += alpha * dq2
    return nll


def _rt_info(x: TwoStep):
    """RTs in seconds and per-person constants: the bound for non-decision time and the lapse window."""
    rt1 = np.asarray(x.rt1, float) / 1000.0
    rt2 = np.asarray(x.rt2, float) / 1000.0
    rt1 = np.where((x.a1 >= 0) & (rt1 > 0.15), rt1, np.nan)
    rt2 = np.where((x.a2 >= 0) & (rt2 > 0.15), rt2, np.nan)
    c = (np.nanmin(rt1) * 0.98 if np.isfinite(rt1).any() else 0.15,
         np.nanmin(rt2) * 0.98 if np.isfinite(rt2).any() else 0.15,
         max(np.nanmax(rt1) if np.isfinite(rt1).any() else 2.0, 1.0),
         max(np.nanmax(rt2) if np.isfinite(rt2).any() else 2.0, 1.0))
    return rt1, rt2, c


def nll(th, x: TwoStep) -> float:
    cache = getattr(x, "_rt_cache", None)
    if cache is None:
        cache = _rt_info(x)
        x._rt_cache = cache
    rt1, rt2, (tm1, tm2, w1, w2) = cache
    return _nll_rlddm(np.asarray(th, float), x.a1, x.s2, x.a2, x.r, x.common, rt1, rt2,
                      x.mask.astype(np.float64), tm1, tm2, w1, w2)


@njit(cache=True)
def _diffuse(v, a, seed_u, dt=0.001, max_t=5.0):
    """Euler simulation of one diffusion from a/2: returns (choice 1 = upper, decision time)."""
    x, t = a / 2.0, 0.0
    sd = np.sqrt(dt)
    i = 0
    while 0.0 < x < a and t < max_t:
        x += v * dt + sd * seed_u[i % len(seed_u)]
        t += dt
        i += 1
    return (1 if x >= a else 0), t


def simulate(th, x: TwoStep, rng) -> TwoStep:
    _, _, (tm1, tm2, w1, w2) = _rt_info(x)
    alpha, v_mb, v_mf, pers, b1, t1, v2, b2, t2, tau = _params(np.asarray(th, float), tm1, tm2)
    n = len(x.a1)
    probs = x.probs if x.probs is not None else drifting_probs(n, rng)
    q1, q2 = np.zeros(2), np.full((2, 2), 0.5)
    a1, s2, a2, r, c = (np.zeros(n, np.int64) for _ in range(5))
    rt1, rt2 = np.zeros(n), np.zeros(n)
    prev = -1
    for t in range(n):
        mb = (0.7 * q2[1].max() + 0.3 * q2[0].max()) - (0.7 * q2[0].max() + 0.3 * q2[1].max())
        drift = v_mb * mb + v_mf * (q1[1] - q1[0]) + (pers if prev == 1 else -pers if prev == 0 else 0.0)
        a1[t], dtime = _diffuse(drift, b1, rng.standard_normal(6000))
        rt1[t] = (t1 + dtime) * 1000
        c[t] = rng.random() < 0.7
        s2[t] = a1[t] if c[t] else 1 - a1[t]
        d2 = v2 * (q2[s2[t], 1] - q2[s2[t], 0])
        a2[t], dtime = _diffuse(d2, b2, rng.standard_normal(6000))
        rt2[t] = (max(t2 + (0.0 if c[t] else tau), 0.0) + dtime) * 1000
        r[t] = rng.random() < probs[t, s2[t], a2[t]]
        dq1 = q2[s2[t], a2[t]] - q1[a1[t]]; dq2 = r[t] - q2[s2[t], a2[t]]
        q1[a1[t]] += alpha * (dq1 + dq2); q2[s2[t], a2[t]] += alpha * dq2
        prev = a1[t]
    sim = TwoStep(a1, s2, a2, r, c, probs, rt1, rt2, mask=x.mask)
    s_rt1, s_rt2, _ = _rt_info(sim)
    sim._rt_cache = (s_rt1, s_rt2, (tm1, tm2, w1, w2))      # same non-decision bounds as the source person
    return sim


def natural(T):
    T = np.atleast_2d(T)
    return {"alpha": sigmoid(T[:, 0]), "v_mb": np.exp(T[:, 1]), "v_mf": np.exp(T[:, 2]), "persev": T[:, 3],
            "a1": np.exp(T[:, 4]), "t1_rel": sigmoid(T[:, 5]), "v2": np.exp(T[:, 6]), "a2": np.exp(T[:, 7]),
            "t2_rel": sigmoid(T[:, 8]), "tau": TAU_SCALE * T[:, 9]}


RLDDM = Model("rlddm", ["alpha", "v_mb", "v_mf", "persev", "a1", "t1", "v2", "a2", "t2", "tau"], nll, natural,
              simulate, lambda x: int(((x.a1 >= 0) * (x.mask > 0)).sum() + ((x.a2 >= 0) * (x.mask > 0)).sum()))
