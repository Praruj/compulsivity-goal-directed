"""Trial-level GEE (the glmer analogue), person-level OLS, split-half reliability, attenuation, power, FDR, CV."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TwoStep, stay_design

COVARIATES = ["age", "iq", "gender"]


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def trial_frame(task: dict[str, TwoStep], people: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Stack every person's stay pairs and attach their z-scored person-level predictors."""
    p = people[cols].apply(zscore)
    frames = []
    for sid, x in task.items():
        if sid not in p.index:
            continue
        d = stay_design(x).drop(columns="trial")
        d["subj"] = sid
        for c in cols:
            d[c] = p.at[sid, c]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def gee(trials: pd.DataFrame, scores: list[str]) -> dict:
    """stay ~ reward * transition * (scores + covariates), exchangeable within person, robust SEs."""
    rhs = " + ".join(scores + COVARIATES)
    m = smf.gee(f"stay ~ reward * transition * ({rhs})", "subj", trials, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    out = {"n_people": int(trials.subj.nunique()), "n_trials": int(len(trials)), "terms": {}}
    ci = m.conf_int()
    for term in ["reward", "transition", "reward:transition"] + [f"reward:transition:{s}" for s in scores]:
        out["terms"][term] = {"beta": float(m.params[term]), "se": float(m.bse[term]), "z": float(m.tvalues[term]),
                              "p": float(m.pvalues[term]), "ci_low": float(ci.loc[term, 0]),
                              "ci_high": float(ci.loc[term, 1])}
    return out


def person_ols(df: pd.DataFrame, outcome: str, scores: list[str]) -> dict:
    """Standardised outcome ~ z-scored scores + covariates, HC3 errors."""
    d = df[[outcome] + scores + COVARIATES].dropna()
    X = sm.add_constant(d[scores + COVARIATES].apply(zscore))
    m = sm.OLS(zscore(d[outcome]), X).fit(cov_type="HC3")
    ci = m.conf_int()
    return {"outcome": outcome, "n": int(len(d)), "terms": {
        s: {"beta": float(m.params[s]), "se": float(m.bse[s]), "p": float(m.pvalues[s]),
            "ci_low": float(ci.loc[s, 0]), "ci_high": float(ci.loc[s, 1])} for s in scores}}


def ols_beta(df: pd.DataFrame, outcome: str, scores: list[str], term: str) -> float:
    d = df[[outcome] + scores + COVARIATES].dropna()
    X = np.column_stack([np.ones(len(d)), d[scores + COVARIATES].apply(zscore).to_numpy()])
    b = np.linalg.lstsq(X, zscore(d[outcome]).to_numpy(), rcond=None)[0]
    return float(b[1 + scores.index(term)])


def shrinkage_boot(df: pd.DataFrame, outcome: str, focal: str, control: str, n_boot: int = 2000, seed: int = 0):
    """Proportion by which the focal person-level beta shrinks when `control` is added; person bootstrap CI."""
    def shrink(d):
        a = ols_beta(d, outcome, [focal], focal)
        b = ols_beta(d, outcome, [focal, control], focal)
        return 1 - b / a
    rng = np.random.default_rng(seed)
    est = shrink(df)
    boots = np.array([shrink(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n_boot)])
    return {"shrinkage": float(est), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n_boot": n_boot}


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r)


def split_half(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return {"r_halves": r, "spearman_brown": spearman_brown(r), "n": int(ok.sum())}


def disattenuate(r: float, rel_x: float, rel_y: float) -> float:
    return float(r / np.sqrt(rel_x * rel_y))


def n_for_power(r: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """People needed to detect correlation r (two-sided), Fisher z approximation."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(((za + zb) / np.arctanh(abs(r))) ** 2 + 3))


def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.minimum(q, 1)
    return out


def cv_gain(df: pd.DataFrame, base: list[str], extra: list[str], outcome: str, reps: int = 20, seed: int = 0):
    """Repeated 10-fold CV ridge: base features vs base + extra, R^2 of the outcome."""
    d = df[base + extra + [outcome]].dropna()
    y = d[outcome].to_numpy()
    res = {"base": [], "full": []}
    for rep in range(reps):
        cv = KFold(10, shuffle=True, random_state=seed + rep)
        for key, cols in (("base", base), ("full", base + extra)):
            X = d[cols].to_numpy()
            pred = np.zeros(len(y))
            for tr, te in cv.split(X):
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            res[key].append(r2_score(y, pred))
    b, f = np.array(res["base"]), np.array(res["full"])
    return {"outcome": outcome, "n": int(len(d)), "base_r2": float(b.mean()), "full_r2": float(f.mean()),
            "gain_mean": float((f - b).mean()), "gain_sd": float((f - b).std()),
            "gain_ci_low": float(np.percentile(f - b, 2.5)), "gain_ci_high": float(np.percentile(f - b, 97.5))}



CELL_Z = np.array([[1, r, t, r * t] for r in (-1, 1) for t in (-1, 1)], float)  # rows: (reward, transition) cells
GLMM_ROWS = ["Intercept", "reward", "transition", "reward:transition"]


def _cells(trials: pd.DataFrame, preds: list[str]):
    """Per person: stay counts and totals in the four reward x transition cells, plus person-level predictors."""
    d = trials.dropna(subset=preds)
    g = d.groupby(["subj", "reward", "transition"]).stay.agg(["sum", "count"]).unstack(["reward", "transition"])
    order = [(r, t) for r in (-1, 1) for t in (-1, 1)]
    s = g["sum"][order].fillna(0).to_numpy()
    n = g["count"][order].fillna(0).to_numpy()
    c = d.groupby("subj")[preds].first().loc[g.index].to_numpy()
    return s, n, np.column_stack([np.ones(len(c)), c])


def _laplace(theta0, s, n, mu, prec, iters=50):
    """Mode of each person's 4 cell-logit coefficients under N(mu, prec^-1) by batched Newton; Laplace terms."""
    th = theta0.copy()
    for _ in range(iters):
        eta = th @ CELL_Z.T
        p = 1 / (1 + np.exp(-eta))
        grad = (s - n * p) @ CELL_Z - (th - mu) @ prec
        H = np.einsum("ic,ck,cl->ikl", n * p * (1 - p), CELL_Z, CELL_Z) + prec
        step = np.linalg.solve(H, grad[..., None])[..., 0]
        th += step
        if np.abs(step).max() < 1e-10:
            break
    eta = th @ CELL_Z.T
    ll = (s * eta - n * np.logaddexp(0, eta)).sum(1)
    q = np.einsum("ik,kl,il->i", th - mu, prec, th - mu)
    p = 1 / (1 + np.exp(-eta))
    H = np.einsum("ic,ck,cl->ikl", n * p * (1 - p), CELL_Z, CELL_Z) + prec
    return th, ll - 0.5 * q - 0.5 * np.linalg.slogdet(H)[1]


def glmm(trials: pd.DataFrame, scores: list[str]) -> dict:
    """glmer analogue: stay ~ reward * transition * (scores + covariates) + (reward * transition | subj), logit link,
    Laplace approximation, full 4x4 random-effect covariance, Wald z from the observed information.
    Each person's trials reduce to four reward x transition binomial cells, so the fit takes seconds."""
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_hess3
    preds = scores + COVARIATES
    s, n, C = _cells(trials, preds)
    k = C.shape[1]
    emp = np.log((s + 0.5) / (n - s + 0.5)) @ np.linalg.inv(CELL_Z).T          # empirical person coefficients
    A0 = np.linalg.lstsq(C, emp, rcond=None)[0].T
    L0 = np.linalg.cholesky(np.cov((emp - C @ A0.T).T) + 1e-3 * np.eye(4))
    tril = np.tril_indices(4)
    x0 = np.concatenate([A0.ravel(), np.where(tril[0] == tril[1], np.log(np.abs(L0[tril])), L0[tril])])
    state = {"th": emp.copy()}

    def unpack(x):
        A = x[:4 * k].reshape(4, k)
        L = np.zeros((4, 4))
        L[tril] = x[4 * k:]
        L[np.diag_indices(4)] = np.exp(np.diag(L))
        return A, L

    def nll(x):
        A, L = unpack(x)
        Sigma = L @ L.T
        th, lp = _laplace(state["th"], s, n, C @ A.T, np.linalg.inv(Sigma))
        state["th"] = th
        return -(lp.sum() - 0.5 * len(s) * np.linalg.slogdet(Sigma)[1])

    r = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 2000, "maxfun": 200000})
    cov = np.linalg.inv(approx_hess3(r.x, nll))
    A, L = unpack(r.x)
    se = np.sqrt(np.diag(cov)[:4 * k]).reshape(4, k)
    cols = ["Intercept"] + preds
    out = {"n_people": int(len(s)), "n_trials": int(n.sum()), "method": "Laplace GLMM (glmer nAGQ=1 analogue)",
           "converged": bool(r.success), "neg_loglik": float(r.fun),
           "random_sd": dict(zip(GLMM_ROWS, np.sqrt(np.diag(L @ L.T)).tolist())), "terms": {}}
    for i, row in enumerate(GLMM_ROWS[1:], 1):
        for j, col in enumerate(cols):
            if col != "Intercept" and col not in scores:
                continue
            term = row if col == "Intercept" else f"{row}:{col}"
            b, e = float(A[i, j]), float(se[i, j])
            out["terms"][term] = {"beta": b, "se": e, "z": b / e, "p": float(2 * stats.norm.sf(abs(b / e))),
                                  "ci_low": b - 1.959964 * e, "ci_high": b + 1.959964 * e}
    return out
