"""Hierarchical fits, one job per call, cached to outputs/fits/<job>.pkl so jobs can run in parallel.

Jobs: {stay,hybrid,model_free,model_based}_s{1,2}[_{odd,even,first,second}] and recovery_{stay,hybrid}_s{1,2}.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgd.data import load_self_report, load_task  # noqa: E402
from cgd.fit import fit, recovery  # noqa: E402
from cgd.models import MODELS, STAY, stay_data  # noqa: E402

FITS = ROOT / "outputs" / "fits"
HALVES = {"odd": lambda n: np.arange(n) % 2 == 1, "even": lambda n: np.arange(n) % 2 == 0,
          "first": lambda n: np.arange(n) < 100, "second": lambda n: np.arange(n) >= 100}


def task_for(study: int):
    return load_task(study, load_self_report(study).index)


def build(kind: str, task: dict, half: str | None):
    keep = {sid: (HALVES[half](len(x.a1)) if half else np.ones(len(x.a1), bool)) for sid, x in task.items()}
    if kind == "stay":
        return STAY, {sid: stay_data(x, keep[sid]) for sid, x in task.items()}
    for sid, x in task.items():
        x.mask = keep[sid].astype(float)
    return MODELS[kind], task


def run(job: str) -> None:
    FITS.mkdir(parents=True, exist_ok=True)
    out = FITS / f"{job}.pkl"
    if out.exists():
        print("cached", job); return
    t0 = time.time()
    parts = job.split("_")
    if parts[0] == "recovery":
        kind, study = parts[1], int(parts[2][1])
        with open(FITS / f"{kind}_s{study}.pkl", "rb") as fh:
            f = pickle.load(fh)
        model, data = build(kind, task_for(study), None)
        true, rec = recovery(model, f, data)
        res = {"true": true, "recovered": rec, "ids": f.ids, "params": model.params}
    else:
        i = next(i for i, p in enumerate(parts) if p in ("s1", "s2"))
        kind, study = "_".join(parts[:i]), int(parts[i][1])
        half = parts[i + 1] if len(parts) > i + 1 else None
        model, data = build(kind, task_for(study), half)
        res = fit(model, data, verbose=True)
    with open(out, "wb") as fh:
        pickle.dump(res, fh)
    print(f"done {job} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for j in sys.argv[1:]:
        run(j)
