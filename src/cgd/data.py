"""Loaders for the Gillan et al. (2016) OSF data (https://osf.io/usdgt) and the stay-regression design."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "usdgt"
STUDY_DIR = {1: "Experiment 1", 2: "Experiment 2"}


@dataclass
class TwoStep:
    a1: np.ndarray       # first-stage choice 0/1, -1 missed
    s2: np.ndarray       # second-stage state 0/1
    a2: np.ndarray       # second-stage choice 0/1, -1 missed
    r: np.ndarray        # reward 0/1 (0 when a2 missed)
    common: np.ndarray   # 1 if the transition was the common one
    probs: np.ndarray | None = None   # (n, 2, 2) reward probabilities per state/option, from the task file
    rt1: np.ndarray | None = None
    rt2: np.ndarray | None = None
    mask: np.ndarray = field(default=None)  # per-trial likelihood weight (1 = counts); used for split halves

    def __post_init__(self):
        if self.mask is None:
            self.mask = np.ones(len(self.a1))


def parse_task_file(path: Path) -> TwoStep:
    """One person's CSV: task rows have 15 fields and start with the trial number (see the authors' readme)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        f = line.strip().split(",")
        if len(f) == 15 and f[0].isdigit():
            try:
                float(f[1])
            except ValueError:
                continue
            rows.append(f)
    n = len(rows)
    a1, s2, a2, r, c = (np.full(n, -1, np.int64) for _ in range(5))
    probs = np.zeros((n, 2, 2))
    rt1, rt2 = np.full(n, np.nan), np.full(n, np.nan)
    for t, f in enumerate(rows):
        probs[t] = np.array([float(x) for x in f[1:5]]).reshape(2, 2)
        if f[5] == "-1" or f[8] not in ("true", "false"):
            continue
        a1[t] = int(f[6]) - 1
        c[t] = f[8] == "true"
        s2[t] = int(f[11]) - 2
        rt1[t] = float(f[7])
        if f[9] == "-1":
            continue
        a2[t] = int(f[10]) - 1
        r[t] = int(f[13])
        rt2[t] = float(f[12])
    s2 = np.where(s2 < 0, 0, s2)
    r = np.where(r < 0, 0, r)
    c = np.where(c < 0, 0, c)
    return TwoStep(a1, s2, a2, r, c, probs, rt1, rt2)


def load_task(study: int, ids, root: Path = RAW) -> dict[str, TwoStep]:
    d = root / STUDY_DIR[study] / f"twostep_data_study{study}"
    return {sid: parse_task_file(d / f"{sid}.csv") for sid in ids if (d / f"{sid}.csv").exists()}


def load_self_report(study: int, root: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(root / STUDY_DIR[study] / f"self_report_study{study}.csv", index_col=0)
    return df.rename(columns={"subj.x": "subj"}).set_index("subj")


def load_items(root: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_csv(root / STUDY_DIR[2] / "individual_items_study2.csv", index_col=0).set_index("subject")
    weights = pd.read_csv(root / STUDY_DIR[2] / "weights.csv", index_col=0)
    return items, weights


def factor_scores(items: pd.DataFrame, weights: pd.DataFrame, drop_prefix: str | None = None,
                  item_subset: list[str] | None = None) -> pd.DataFrame:
    """Factor scores = z-scored items x factor-score weights. Optionally drop a questionnaire or keep a subset."""
    w = weights.copy()
    if drop_prefix:
        w = w[~w.index.str.startswith(drop_prefix + "_")]
    if item_subset is not None:
        w = w.loc[[i for i in w.index if i in set(item_subset)]]
    z = items[w.index].astype(float)
    z = (z - z.mean()) / z.std(ddof=0)
    return pd.DataFrame(z.to_numpy() @ w.to_numpy(), index=items.index, columns=w.columns)


def split_items(weights: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Odd and even items within each questionnaire (by item number)."""
    odd, even = [], []
    for item in weights.index:
        (odd if int(item.rsplit("_", 1)[1]) % 2 else even).append(item)
    return odd, even


# ---------- model-agnostic stay analysis ----------
def stay_design(x: TwoStep) -> pd.DataFrame:
    """Trial pairs (t-1 -> t) with valid responses: stay, previous reward (+/-1), previous transition (+/-1).

    `trial` is the index of the outcome trial (t-1), which decides which split half a pair belongs to.
    """
    prev, cur = np.arange(len(x.a1) - 1), np.arange(1, len(x.a1))
    ok = (x.a1[prev] >= 0) & (x.a2[prev] >= 0) & (x.a1[cur] >= 0)
    p, c = prev[ok], cur[ok]
    return pd.DataFrame({"trial": p, "stay": (x.a1[c] == x.a1[p]).astype(int),
                         "reward": 2 * x.r[p] - 1, "transition": 2 * x.common[p] - 1})


def stay_probabilities(x: TwoStep) -> dict[str, float]:
    d = stay_design(x)
    out = {}
    for rew, rl in ((1, "rewarded"), (-1, "unrewarded")):
        for tr, tl in ((1, "common"), (-1, "rare")):
            s = d.stay[(d.reward == rew) & (d.transition == tr)]
            out[f"{rl}_{tl}"] = float(s.mean()) if len(s) else np.nan
    return out


def raw_mb_index(x: TwoStep) -> float:
    p = stay_probabilities(x)
    return (p["rewarded_common"] - p["rewarded_rare"]) - (p["unrewarded_common"] - p["unrewarded_rare"])


def quality(x: TwoStep) -> dict[str, float]:
    valid = x.a1 >= 0
    same = max(np.mean(x.a1[valid] == 0), np.mean(x.a1[valid] == 1)) if valid.any() else 1.0
    return {"missed_first": float(1 - valid.mean()), "same_first": float(same),
            "rt1_median": float(np.nanmedian(x.rt1)) if x.rt1 is not None else np.nan,
            "rt2_median": float(np.nanmedian(x.rt2)) if x.rt2 is not None else np.nan}


# ---------- independent sample: Seow et al. 2021 (https://osf.io/mx9kf, CC-BY 4.0) ----------
SEOW = RAW.parent / "seow2021"
SEOW_PREFIX = {"ALCOHOL": "AUDIT", "LSASONE": "LSAS"}


def seow_from_frame(d: pd.DataFrame, fast: float = 0.15) -> TwoStep:
    """One person's rows from SeowEtAl_twostepBehavData.csv. S1c 1/2; S2s 2/3; S2c 3/4 (state 2) or 5/6 (state 3),
    0 = missed; trans 1 = common, 2 = rare; hit 0/1; r1..r4 drifting reward probabilities. First-stage responses
    faster than 150 ms count as missed, as in the authors' script."""
    d = d.sort_values("nTrial")
    ok1 = d.S1c.notna().to_numpy() & (d.S1rt.fillna(0).to_numpy() > fast)
    ok2 = ok1 & (d.S2c.fillna(0).to_numpy() > 0) & d.hit.notna().to_numpy()
    a1 = np.where(ok1, d.S1c.fillna(1).to_numpy() - 1, -1).astype(np.int64)
    s2 = np.where(ok1, d.S2s.fillna(2).to_numpy() - 2, 0).astype(np.int64)
    a2 = np.where(ok2, (d.S2c.fillna(3).to_numpy() - 3) % 2, -1).astype(np.int64)
    r = np.where(ok2, d.hit.fillna(0).to_numpy(), 0).astype(np.int64)
    c = np.where(ok1, d.trans.to_numpy() == 1, 0).astype(np.int64)
    probs = d[["r1", "r2", "r3", "r4"]].to_numpy().reshape(-1, 2, 2)
    return TwoStep(a1, s2, a2, r, c, probs, np.where(ok1, d.S1rt, np.nan), np.where(ok2, d.S2rt, np.nan))


def load_seow(root: Path = SEOW) -> tuple[dict[str, TwoStep], pd.DataFrame]:
    """Task data and a person table with AD/CIT/SW (Gillan 2016 weights on items z-scored in this sample), the
    questionnaire totals, CIT without AUDIT items, odd/even-item CIT, and age/IQ/gender."""
    beh = pd.read_csv(root / "SeowEtAl_twostepBehavData.csv")
    task = {sid: seow_from_frame(g) for sid, g in beh.groupby("subj")}
    q = pd.read_csv(root / "SeowEtAl_twostepQuestData.csv").set_index("qnid")
    weights = pd.read_csv(root / "GillanEtAl2016_itemWeights.csv", index_col=0)
    weights.columns = ["AD", "CIT", "SW"]
    names = {}
    for col in q.columns:
        head = col.rstrip("0123456789")
        if col != head and head != "IQ":
            names[col] = f"{SEOW_PREFIX.get(head, head)}_{col[len(head):]}"
    items = q[list(names)].rename(columns=names)
    df = factor_scores(items, weights)
    df["CIT_noAUDIT"] = factor_scores(items, weights, drop_prefix="AUDIT")["CIT"]
    odd, even = split_items(weights)
    df["CIT_odd_items"] = factor_scores(items, weights, item_subset=odd)["CIT"]
    df["CIT_even_items"] = factor_scores(items, weights, item_subset=even)["CIT"]
    for t in ["alcohol", "aes", "bis", "eat", "lsas", "oci", "scz", "sds", "stai"]:
        df[f"{'audit' if t == 'alcohol' else t}_total"] = q[f"total_{t}"]
    df["age"], df["iq"] = q.age, q.total_iq
    df["gender"] = np.where(q.gender == "Male", 1.0, np.where(q.gender == "Female", -1.0, np.nan))
    return task, df.loc[[s for s in df.index if s in task]]
