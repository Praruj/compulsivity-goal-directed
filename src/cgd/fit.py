"""Hierarchical empirical-Bayes fitting (EM with Laplace approximation; Huys et al., 2011) and iBIC.

Copied from C:/dev/self-control-phenotypes (src/scp/fit.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .models import Model


@dataclass
class Fit:
    model: str
    ids: list[str]
    theta: np.ndarray       # (n, p) MAP estimates, unbounded scale
    post_var: np.ndarray    # (n, p) Laplace posterior variances
    mu: np.ndarray
    var: np.ndarray
    nll: np.ndarray         # per-person negative log-likelihood at the MAP
    ibic: float
    n_iter: int


def _hess_diag(f, x, h=1e-3):
    f0 = f(x)
    d = np.empty_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        d[i] = (f(x + e) - 2 * f0 + f(x - e)) / h ** 2
    return d


def _map(model: Model, x, mu, var, start):
    obj = lambda th: model.nll(th, x) + 0.5 * np.sum((th - mu) ** 2 / var)  # noqa: E731
    best = None
    for s in (start, mu):
        res = minimize(obj, s, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    h = np.clip(_hess_diag(obj, best.x), 1e-6, None)
    return best.x, 1.0 / h


def fit(model: Model, data: dict, max_iter: int = 40, tol: float = 1e-3, seed: int = 0,
        n_samples: int = 2000, verbose: bool = False) -> Fit:
    ids = sorted(data)
    p = len(model.params)
    mu, var = np.zeros(p), np.full(p, 10.0)     # broad start
    theta = np.zeros((len(ids), p))
    pv = np.ones((len(ids), p))
    for it in range(1, max_iter + 1):
        for i, sid in enumerate(ids):
            theta[i], pv[i] = _map(model, data[sid], mu, var, theta[i])
        new_mu = theta.mean(0)
        new_var = np.maximum((theta ** 2 + pv).mean(0) - new_mu ** 2, 1e-4)
        delta = np.max(np.abs(new_mu - mu)) + np.max(np.abs(np.sqrt(new_var) - np.sqrt(var)))
        mu, var = new_mu, new_var
        if verbose:
            print(f"  {model.name} iter {it}: mu={np.round(mu, 3)} sd={np.round(np.sqrt(var), 3)}", flush=True)
        if delta < tol:
            break
    nll = np.array([model.nll(theta[i], data[sid]) for i, sid in enumerate(ids)])
    # integrated BIC: Monte-Carlo integral of the likelihood over the fitted group prior
    rng = np.random.default_rng(seed)
    samples = mu + rng.standard_normal((n_samples, p)) * np.sqrt(var)
    log_ml = 0.0
    n_obs = 0
    for sid in ids:
        ll = np.array([-model.nll(s, data[sid]) for s in samples])
        log_ml += logsumexp(ll) - np.log(n_samples)
        n_obs += model.n_obs(data[sid])
    ibic = -2 * log_ml + 2 * p * np.log(n_obs)
    return Fit(model.name, ids, theta, pv, mu, var, nll, float(ibic), it)


def recovery(model: Model, f: Fit, data: dict, seed: int = 1, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Simulate everyone with their fitted parameters, refit hierarchically, return (true, recovered)."""
    rng = np.random.default_rng(seed)
    sim = {sid: model.simulate(f.theta[i], data[sid], rng) for i, sid in enumerate(f.ids)}
    g = fit(model, sim, n_samples=200, **kw)
    return f.theta, g.theta
