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
