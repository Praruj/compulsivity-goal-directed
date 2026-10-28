# Pre-registration: compulsivity, alcohol and goal-directed control (Gillan et al. 2016, replicated and extended)

Written and committed **before** any task data were modelled and before any symptom score was related to task
behaviour. I have only looked at file formats and column names. Anything decided later is tagged **[post-hoc]**.

## Question

Gillan, Kosinski, Whelan, Phelps & Daw (eLife 2016) showed in two online samples that deficits in goal-directed
(model-based) planning on the two-step task go with a transdiagnostic **"compulsive behaviour and intrusive
thought" (CIT)** dimension, not with anxious-depression (AD) or social withdrawal (SW). This is one of the most
cited results in computational psychiatry and is the template for the "dimensions, not diagnoses" argument.

This project asks three things of their public data:

1. **Replication.** Do I get the same result from the raw trial data, with my own independent code in Python?
2. **Addiction.** Alcohol problems (AUDIT) also go with weaker model-based control in their data. Is that a
   separate signal, or is it carried by the compulsivity dimension? Note the circularity: AUDIT items themselves
   load on CIT, so I also test CIT rebuilt **without** the AUDIT items.
3. **Reliability.** My earlier project (self-control-phenotypes, Self-Regulation Ontology) found that the
   model-based weight of the same task has a 4-month test-retest ICC of only 0.28. How consistent is the
   model-based index **within** a session here, and what does that imply for the size of the true CIT effect and
   for the samples needed to study it?

## Data

Public OSF project https://osf.io/usdgt (Gillan, 2019 upload; no licence is stated on the project, so raw data
are not redistributed: they stay in the git-ignored `data/` folder).

- **Study 1:** n = 548 (the included sample in `self_report_study1.csv`): age, IQ, gender, OCI-R, SDS (depression),
  STAI (trait anxiety).
- **Study 2:** n = 1,413: nine questionnaires with item-level answers (`individual_items_study2.csv`), the
  authors' factor scores (`Factor1-3`), and the factor-score weights (`weights.csv`, items x AD/CIT/SW).
- **Task:** two-step task (Daw et al. 2011), 200 trials, one CSV per person. Trials with a missed response are
  coded -1.

**Sample.** The people in the self-report files are the authors' post-exclusion samples. I use those people
(intersected with available task files). **Sensitivity check:** additionally drop anyone with > 20 % missed
first-stage responses or >= 95 % identical first-stage choices.

**Factor scores.** I recompute AD, CIT and SW as z-scored items x `weights.csv` and check they correlate > .99
with the provided `Factor1-3` (which column is which is decided by that correlation). If they do not, I use the
provided scores and say so. One SDS item was administered with an error (authors' note); they report results are
unchanged without it, and I keep it in.

## Measures of goal-directed control

**Model-agnostic (primary).** Logistic regression of first-stage stay (repeat the previous first-stage choice)
on previous reward (+1/-1), previous transition (common +1 / rare -1) and their product. The **reward x
transition** coefficient is the model-based index; the reward main effect is the model-free index. Only trial
pairs with valid responses on both trials count.

**Computational (secondary).** The Daw hybrid model from self-control-phenotypes (one learning rate, lambda = 1,
beta_MB, beta_MF, second-stage beta, perseveration), fitted by hierarchical empirical Bayes (EM + Laplace; Huys
et al. 2011). w = beta_MB / (beta_MB + beta_MF). Parameter recovery is checked by simulating each person with
their fitted parameters (on that person's actual reward drifts) and refitting; parameters with recovery r < .5
are not interpreted.

## Statistical models

- **Trial-level (primary), mirroring Gillan's glmer:** `stay ~ reward * transition * (score + age + IQ + gender)`,
  scores and covariates z-scored. Python has no glmer with random slopes, so I use a **GEE** (logit link,
  exchangeable working correlation within person, robust standard errors). This estimates the population-averaged
  effect; its coefficients are expected to be somewhat smaller than glmer's subject-specific ones, so I compare
  **signs and p-values**, not raw betas.
- **Person-level (secondary, two-stage):** each person's model-based index from a hierarchical logistic
  regression (the same EM fitter, Gaussian group prior over the four coefficients), then OLS
  `index ~ score + age + IQ + gender` with HC3 errors, reporting standardised betas.

## Hypotheses

**H1 (replication, Study 2; confirmatory).** With AD, CIT and SW entered together (plus covariates), the
reward x transition x CIT term is negative, p < .05. AD's term is smaller in absolute size than CIT's. (Gillan
also report a small *positive* SW term; I check its sign but do not count it as a test.)

**H1b (replication, Study 1; confirmatory).** The reward x transition x OCI-R term is negative, p < .05.

**H2 (alcohol; confirmatory family, Benjamini-Hochberg across H2a-c).**
- H2a: AUDIT total alone (plus covariates) has a negative reward x transition x AUDIT term.
- H2b: When CIT is added, the AUDIT term shrinks by at least half (i.e. the alcohol link is largely carried by
  compulsivity). Tested by the size of the shrinkage with a person-level bootstrap 95 % CI; supported if the CI
  lies above 0 and the point estimate is >= 50 %.
- H2c: CIT recomputed **without** the 10 AUDIT items still has a negative term (the compulsivity link is not just
  the drinking items).
- **Exploratory:** AUDIT-C (items 1-3, consumption) versus items 4-10 (dependence and harm); hazardous drinking
  (AUDIT >= 8) as a group.

**H3 (reliability; descriptive with a directional prediction).**
- Split-half reliability (Pearson r, Spearman-Brown corrected) of the model-based index, model-free index and w:
  (a) odd vs even trials (the pair counts towards the half of the trial whose outcome is evaluated); (b) first vs
  second 100 trials. For w, the RL model is run over the full sequence and the likelihood counts only the choices
  in one half. Each half is fitted with its own hierarchical prior.
- Prediction: odd/even > first/second half, and both above the 4-month test-retest ICC from the Self-Regulation
  Ontology (0.28 for w, 0.32 for the raw index). If within-session reliability is high but retest is low, the
  problem is stability over time, not measurement noise within a session.
- CIT reliability: split-half over items (odd/even items within each questionnaire), Spearman-Brown corrected.
- Then: the CIT-model-based correlation corrected for attenuation, and the n needed for 80 % power (two-sided,
  alpha = .05) at the observed and at the corrected correlation.

**H4 (computational; confirmatory if recovery r >= .5).** w is lower with higher CIT (person-level OLS with
covariates, p < .05), with AD and SW in the same model.

**H5 (prediction; exploratory).** Can task behaviour predict a person's CIT score out of sample? 10-fold CV
repeated 20 times, ridge regression on task features (model-based and model-free index, perseveration, w, median
RTs) plus age/IQ/gender vs covariates alone. Expected gain in R^2 is below 0.02: a reliable group effect does not
make an individual-level predictor.

## What would change my mind / how results will be reported

- If H1 fails with my code, that is the headline, and I check the GEE against the two-stage analysis before
  saying anything about the original.
- If recovery of w is poor (< .5), H4 is reported as "not measurable here".
- If split-half reliability of the model-based index is below .5, the disattenuated effect sizes are flagged as
  unstable (correcting with a small denominator inflates them).
- All tests are reported, including the null ones, in `outputs/results.json`.

---

## Addendum 1 (2026-10-29): mixed-model check and an independent sample

Written after the results above were in `outputs/results.json` and the README, and **before** (a) fitting any
mixed model to the real data and (b) downloading or looking at the independent sample. Everything here is
reported, whatever it shows.

### A. Mixed model (same Study 1 and Study 2 data)

The primary tests above used GEE, whose population-averaged coefficients are smaller than Gillan et al.'s glmer
by construction, and the alcohol result (H2a) sat near p = .05. I now fit the model they fitted:
`stay ~ reward * transition * (scores + age + IQ + gender) + (reward * transition | subj)`, logit link, Laplace
approximation (glmer's default), full 4 x 4 random-effect covariance, Wald z. Implementation: `cgd.stats.glmm`,
checked on synthetic data (`test_glmm_recovers_fixed_and_random_effects`). Python has no lme4; this is the same
model, written by me, not lme4 itself.

- **A1.** H1, H1b, H2a, H2b (descriptive shrinkage), H2c and the single-factor models are re-run with the GLMM.
  Expectation: CIT coefficient close to the paper's (−0.058, z ≈ −4.7) and larger in magnitude than my GEE
  (−0.041). If my GLMM coefficient for CIT in the H1 model lies within ±0.010 of −0.058, I call the replication
  "numerically close"; otherwise I report the gap.
- **A2.** Alcohol decision rule: if GEE and GLMM disagree on H2a at p < .05, the conclusion is "fragile
  (model-dependent)", not "absent". If both are above .05, "not supported". If both are below, "supported but carried
  by CIT" only if the GLMM shrinkage (H2b) is ≥ 50%.

### B. Independent sample: Seow et al. 2021 (*J Neurosci*), OSF mx9kf, CC-BY 4.0

From the paper (I have not opened the data): 192 people after exclusions, recruited in the lab with EEG, 150
two-step trials (70/30 transitions, drifting rewards), the same nine questionnaires as Gillan Study 2 plus I-CAR
IQ, and Gillan 2016's item weights on the OSF page. The authors already report that compulsivity relates to
less model-based control in this sample, so the direction is not blind. What is new is independent code, the same
pipeline as Study 2, the alcohol tests, reliability, and an honest power statement.

- **Scores.** If item-level answers are present, AD, CIT and SW are computed with Gillan 2016's weights applied to
  items z-scored within this sample (as for Study 2); otherwise the authors' provided scores are used, and failing
  that the OCI-R total stands in for CIT. Covariates: age, IQ, gender, as available.
- **Power, stated before looking.** Scaling the Study 2 glmer z (−4.7) by √(192/1413 × 150/200) gives an expected
  z ≈ −1.5: about 45% power one-sided. A null here would not be evidence against the effect.
- **B1 (primary).** CIT × reward × transition < 0 in the GLMM with AD and SW, one-sided α = .05, plus the
  CIT-only model. Verdicts: "replicated" if one-sided p < .05; "consistent" if not significant but the 95% CI
  contains the Study 2 GLMM estimate; "inconsistent" if a two-sided test of the Study 2 vs Seow difference gives
  p < .05.
- **B2.** Pooled estimate: inverse-variance fixed-effect combination of Study 2 and Seow GLMM CIT coefficients,
  with the heterogeneity Q test.
- **B3 (exploratory).** AUDIT alone and AUDIT + CIT (GLMM). No verdict with n = 192; estimate and CI only.
- **B4.** Split-half reliability (odd/even) of the raw model-based index and of hierarchical stay-regression
  `mb_index` in this sample, compared with Study 2.
- GEE versions of B1 are reported alongside, as a robustness check.

---

## Addendum 2 (2026-10-29): transition learning and a reaction-time marker

Written after Addendum 1's results, and **before** fitting any transition-learning model or computing any
reaction-time measure on real data. While planning it I read two published results on these data that my
analyses could have been guided by:

- Sharp, Dolan & Eldar (2023, *Psychological Medicine*) re-analysed Gillan Study 2 and found that compulsivity
  goes with a **faster state-transition learning rate γ** (β = 0.15, p = .003), fully mediated by a lower β_MB
  (β_MB ~ compulsivity β = −0.12, p < .001). My post-hoc Study 2 β_MB result (−0.117) is therefore a
  re-discovery of their result, not a new finding, and the paper will say so.
- Seow et al. (2021) found that compulsive people **slow down less after rare transitions** at the second
  stage, and replicated this in Gillan Study 2.

Neither result has been tested in the other sample with independent code, and the γ result has not been tested
in the Seow sample at all. So:

### C. Transition learning (hybrid model + incremental transition learning)

Model `hybrid_tl`: the Daw hybrid, plus a learned belief p_a = P(the "own" second-stage state | action a),
starting at the instructed 0.7. After each transition p_a += γ (1[common] − p_a) for the chosen action only. The
model-based value uses p_a instead of the fixed 0.7. Parameters: α, β_MB, β_MF, β2, perseveration, γ (logit
scale). Fitted, recovered and compared with the fixed hybrid exactly like the original models.

- **C1 (replication of Sharp et al. in Study 2 with my code).** logit γ ~ CIT + age + IQ + gender, person-level
  OLS with HC3: CIT > 0, α = .05 two-sided. Secondary: with AD and SW in the model.
- **C2 (new test, Seow sample).** The same, one-sided (CIT > 0), α = .05. Power is low (~35% if the Study 2
  standardised effect is 0.15), so a null is not evidence against it.
- **C3 (mediation, both samples).** The CIT coefficient on logit γ when log β_MB is added. Expectation from Sharp et
  al.: it shrinks toward zero.
- **C4.** iBIC of hybrid_tl vs hybrid in both samples. γ is interpreted only if its recovery r ≥ .5.

### D. Reaction-time marker: second-stage slowing after rare transitions

Per person: RT slowing = median second-stage RT after rare transitions − after common transitions (trials with a
valid second-stage response, RT > 150 ms).

- **D1.** RT slowing ~ CIT + age + IQ + gender (person-level OLS, HC3), one-sided CIT < 0, in Study 2 and Seow.
- **D2.** Split-half reliability of RT slowing (odd/even trials) in both samples, compared with the choice-based
  model-based index.
- **D3 (exploratory).** CIT on the choice-based index (mb_index) and RT slowing entered together: does each carry
  separate information? And does a composite (mean of the two z-scores) have higher split-half reliability than
  either alone?
