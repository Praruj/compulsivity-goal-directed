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
