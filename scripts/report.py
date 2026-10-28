"""Figures from outputs/results.json into outputs/figures/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "outputs" / "results.json").read_text())
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, GOLD, GREY, RED, BLUE = "#1f1f1f", "#b8893f", "#9a9a9a", "#c0504d", "#4f7cac"


def stay_plot():
    sp = RES["stay_by_cit_tertile"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
    for ax, t, title in zip(axes, ["low", "high"], ["Lowest third of CIT", "Highest third of CIT"]):
        x = np.arange(2)
        common = [sp[f"{t}|rew|common"], sp[f"{t}|unrew|common"]]
        rare = [sp[f"{t}|rew|rare"], sp[f"{t}|unrew|rare"]]
        ax.bar(x - 0.18, common, 0.36, color=BLUE, label="common")
        ax.bar(x + 0.18, rare, 0.36, color=GOLD, label="rare")
        ax.set_xticks(x, ["rewarded", "unrewarded"])
        ax.set_ylim(0.5, 0.9)
        ax.set_title(title)
    axes[0].set_ylabel("P(repeat first-stage choice)")
    axes[1].legend(frameon=False, title="previous transition")
    fig.suptitle("Study 2 (n = %d): the model-based signature is the common-vs-rare gap reversing with reward"
                 % RES["n"]["study2"], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "stay_by_cit.png", dpi=160)
    plt.close(fig)


def forest():
    g, g1 = RES["gee_study2"], RES["gee_study1"]
    rows = [("Study 2: AD (joint)", g["H1_AD_CIT_SW"], "AD"), ("Study 2: CIT (joint)", g["H1_AD_CIT_SW"], "CIT"),
            ("Study 2: SW (joint)", g["H1_AD_CIT_SW"], "SW"),
            ("Study 1: OCI-R", g1["H1b_oci"], "oci_total"),
            ("AUDIT total", g["H2a_audit"], "audit_total"),
            ("AUDIT, controlling CIT", g["H2b_audit_plus_CIT"], "audit_total"),
            ("CIT without AUDIT items", g["H2c_CIT_noAUDIT"], "CIT_noAUDIT"),
            ("AUDIT-C (consumption)", g["expl_audit_c"], "audit_c"),
            ("AUDIT 4-10 (dependence/harm)", g["expl_audit_harm"], "audit_harm")]
    labels = {"sds_total": "Depression (SDS)", "stai_total": "Trait anxiety (STAI)", "oci_total": "OCD (OCI-R)",
              "lsas_total": "Social anxiety (LSAS)", "bis_total": "Impulsivity (BIS)",
              "scz_total": "Schizotypy", "aes_total": "Apathy (AES)", "eat_total": "Eating (EAT-26)"}
    rows += [(f"Study 2: {labels[q]}", g[f"q_{q}"], q) for q in labels]
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for i, (lab, m, s) in enumerate(rows[::-1]):
        t = m["terms"][f"reward:transition:{s}"]
        col = RED if t["p"] < 0.05 and t["beta"] < 0 else (BLUE if t["p"] < 0.05 else GREY)
        ax.errorbar(t["beta"], i, xerr=[[t["beta"] - t["ci_low"]], [t["ci_high"] - t["beta"]]], fmt="o",
                    color=col, capsize=2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlabel("reward x transition x score (GEE log-odds per SD, 95% CI)\n< 0 = less model-based")
    fig.tight_layout()
    fig.savefig(FIG / "symptom_effects.png", dpi=160)
    plt.close(fig)


def reliability_plot():
    rel = RES["reliability_study2"]["task"]
    items = [("mb_index", "Model-based index\n(hierarchical regression)"), ("raw_mb_index", "Model-based index\n(raw stay probabilities)"),
             ("rl_w", "Hybrid-model w"), ("mf_index", "Model-free index")]
    items = [(k, lab) for k, lab in items if k in rel["odd_even"]]
    sro = RES["reliability_study2"]["sro_retest_icc"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(items))
    ax.bar(x - 0.27, [rel["odd_even"][k]["spearman_brown"] for k, _ in items], 0.27, color=BLUE,
           label="split-half: odd vs even trials")
    ax.bar(x, [rel["first_second"][k]["spearman_brown"] for k, _ in items], 0.27, color=GOLD,
           label="split-half: first vs second 100 trials")
    ax.bar(x + 0.27, [sro.get(k.replace("rl_", ""), np.nan) for k, _ in items], 0.27, color=GREY,
           label="4-month retest ICC (SRO, other sample)")
    ax.axhline(RES["reliability_study2"]["CIT_items"]["spearman_brown"], color=RED, ls="--", lw=1)
    ax.text(len(items) - 0.5, RES["reliability_study2"]["CIT_items"]["spearman_brown"] + 0.02, "CIT score",
            color=RED, ha="right", fontsize=8)
    ax.set_xticks(x, [lab for _, lab in items], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("reliability (Spearman-Brown)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=160)
    plt.close(fig)


def power_plot():
    att = RES["attenuation_study2"].get("mb_index")
    if not att:
        return
    r = np.linspace(0.04, 0.3, 200)
    n = [((1.959964 + 0.841621) / np.arctanh(v)) ** 2 + 3 for v in r]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(r, n, color=INK)
    pts = [("observed", abs(att["r_observed"]), RED, (-12, -34))]
    if "r_corrected_odd_even" in att:
        pts.append(("corrected for\nsplit-half noise", abs(att["r_corrected_odd_even"]), BLUE, (10, 6)))
    for lab, v, c, off in pts:
        nv = ((1.959964 + 0.841621) / np.arctanh(v)) ** 2 + 3
        ax.scatter([v], [nv], color=c, zorder=3)
        ax.annotate(f"{lab}\n|r| = {v:.2f}, n = {nv:,.0f}", (v, nv), textcoords="offset points", xytext=off,
                    fontsize=8, color=c)
    ax.axhline(RES["n"]["study2"], color=GREY, ls=":", lw=1)
    ax.text(0.3, RES["n"]["study2"] * 1.05, f"this sample ({RES['n']['study2']:,})", ha="right", fontsize=8,
            color=GREY)
    ax.set_yscale("log")
    ax.set_xlabel("|correlation| between CIT and model-based index")
    ax.set_ylabel("people needed (80% power)")
    fig.tight_layout()
    fig.savefig(FIG / "power.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    stay_plot(); forest(); reliability_plot(); power_plot()
    print("figures in", FIG)
