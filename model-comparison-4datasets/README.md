# Does One Algorithm Consistently Outperform the Others?

*A comparison of five classifiers on four datasets with nested cross-validation, corrected significance tests, calibration and conformal prediction.*

Author: Ben Brahim. Machine Learning project: model comparison across healthcare, banking, credit and cybersecurity data. A 4-page PDF report (`model_comparison_report.pdf`) accompanies this README; the numbers below follow that report.

## Summary

The question: **does one classification algorithm consistently outperform the others across different types of problems?**
Five model families (logistic regression, k-NN, MLP, random forest, histogram gradient boosting) are compared on four public binary-classification datasets, using nested repeated cross-validation (25 paired ROC-AUC scores per model and dataset), a corrected resampled t-test, a Friedman / Nemenyi test across datasets and a held-out study.

- The winner changes with the dataset: logistic regression on Blood Transfusion, random forest on Banknote and Credit-G, histogram gradient boosting on Spambase.
- Across the four datasets **no pair of models differs significantly** (Friedman χ² = 4.60, p = 0.33).
- The plain paired t-test flags **30 of 40** pairs as significant; the corrected test only **14**.

## Contents of this folder

| Item | Description |
|---|---|
| `README.md` | This file: all results |
| `model_comparison_report.pdf` | 4-page report: introduction, datasets, methodology, results, discussion, limitations (Figures 1 to 5) |
| `model_comparison.ipynb` | Commented notebook with the full code (loading, nested CV, statistical tests, held-out study) and colored figures: class and feature histograms, AUC heatmap, box plots, fold-AUC histograms, critical-difference plot, ROC curves, calibration and conformal charts. Saved with the outputs of the full run |
| `data/*.csv` | The four datasets (one CSV each, original column names and values) |

## 1. Introduction

"Model B beats model A by 0.8 AUC points" is one of the most common claims in data science and one of the least often justified. A single train/test split, or a plain t-test on cross-validation scores, can easily turn noise into a ranking. This project asks two things of the data: which model wins on each dataset, and whether any model wins consistently across datasets.

The protocol is designed so that tuning never sees the test data, all models are scored on identical splits, and differences are called significant only after correcting for the overlap between cross-validation folds.

## 2. The four datasets

The datasets differ in size (748 to 4,601 rows), dimensionality (4 to 61 features) and class balance (positive rate from 24 % to 45 %).

| Dataset | Domain | OpenML id | Rows | Features | Positive rate | Target |
|---|---|---|---|---|---|---|
| Blood Transfusion | Healthcare | 1464 | 748 | 4 | 0.238 | Will the donor donate in March 2007? (class 2 = donates) |
| Banknote Authentication | Banking / security | 1462 | 1,372 | 4 | 0.445 | Genuine or forged banknote (class 2 = positive) |
| Credit-G | Financial risk | 31 | 1,000 | 20 → 61 after one-hot | 0.300 | Good or bad credit risk (positive = bad) |
| Spambase | Cybersecurity | 44 | 4,601 | 57 | 0.394 | Spam or non-spam (1 = spam) |

- **Blood Transfusion** (`blood-transfusion.csv`): each row is a blood donor. Inputs: recency, frequency, total volume donated and time since first donation (columns `V1`–`V4`).
- **Banknote Authentication** (`banknote.csv`): each row is a banknote. Inputs: four statistics extracted from the banknote image (wavelet variance, skewness, kurtosis, entropy; columns `V1`–`V4`).
- **Credit-G** (`credit-g.csv`): each row is a credit applicant. Inputs: checking account status, credit history, loan duration, credit amount, employment, age and other characteristics (7 numeric and 13 categorical columns).
- **Spambase** (`spambase.csv`): each row is an email. Inputs: 48 word frequencies, 6 character frequencies and 3 capital-letter statistics.

## 3. Methodology

- **Models** (5 families, each tuned by a small grid on the inner 3-fold cross-validation):

  | Model | Tuned hyper-parameter (grid) |
  |---|---|
  | Logistic regression | inverse regularisation `C` in {0.1, 1, 10} |
  | k-NN | neighbours `k` in {15, 31} |
  | MLP (one hidden layer, 64 units) | L2 penalty `alpha` in {1e-4, 1e-2} |
  | Random forest (300 trees) | `min_samples_leaf` in {1, 5} |
  | Histogram gradient boosting | `learning_rate` in {0.05, 0.1}; `max_leaf_nodes` in {15, 31} |

- **Preprocessing**: categorical columns one-hot encoded, missing values median-imputed, features standardised for logistic regression, k-NN and MLP.
- **Nested, repeated cross-validation**: outer loop = 5 × repeated stratified 5-fold (**25 paired scores per model and dataset**); inner loop = 3-fold grid search on ROC AUC run on the outer training part only, so tuning never sees the outer test fold. All models use the same outer splits, so scores are paired. In total 4 × 5 × 25 = 500 nested fits.
- **Within one dataset**: the plain paired t-test (naive) is compared with the Nadeau & Bengio (2003) corrected resampled t-test. Folds share training data, so the plain test understates the variance; the correction adds the test/train size ratio:

  `t = mean(d) / sqrt( (1/n + n_test/n_train) · var(d) )`

- **Across datasets**: Friedman test on the model ranks, then the Nemenyi critical difference (Demšar, 2006).
- **Held-out study** (60 / 20 / 20 split): test AUC with a 1,000-resample bootstrap 95 % confidence interval, ROC curves, calibration error (ECE) raw, after sigmoid (Platt) and after isotonic recalibration, and 90 % split-conformal prediction sets.
- **Metrics and reproducibility**: ROC AUC is the main metric; log loss, Brier score and ECE are recorded for every fold. Seed 0 drives every split, and all results come from one notebook, `model_comparison.ipynb`.

## 4. Results

### 4.1 Model performance

ROC AUC, mean ± sd over the 25 outer folds (bold = best mean):

| Dataset | Logistic reg. | k-NN | MLP | Random forest | Hist. grad. boosting |
|---|---|---|---|---|---|
| Blood Transfusion | **0.752 ± 0.026** | 0.739 ± 0.032 | 0.728 ± 0.049 | 0.724 ± 0.032 | 0.713 ± 0.035 |
| Banknote | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.994 ± 0.002 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| Credit-G | 0.785 ± 0.027 | 0.757 ± 0.033 | 0.756 ± 0.042 | **0.794 ± 0.030** | 0.779 ± 0.026 |
| Spambase | 0.971 ± 0.005 | 0.960 ± 0.005 | 0.974 ± 0.007 | 0.987 ± 0.004 | **0.989 ± 0.003** |

Banknote is saturated: logistic regression, random forest and histogram gradient boosting tie at 1.000 to three decimals, and the bold marks the highest unrounded mean.

Mean AUC over the four datasets: logistic regression 0.877, random forest 0.876, histogram gradient boosting 0.870, k-NN 0.864, MLP 0.863.

The best model by mean AUC changes with the dataset: logistic regression on Blood Transfusion (0.752), random forest on Banknote and Credit-G (0.794), and histogram gradient boosting on Spambase (0.989). Log loss and Brier score give the same winners except on Banknote, where histogram gradient boosting leads instead (log loss 0.019, Brier 0.004). The MLP is clearly weakest on Banknote (0.994) but is statistically tied with the best on Blood Transfusion and Credit-G.

The 25 fold scores overlap heavily on the small datasets: on Blood Transfusion the fold-to-fold standard deviation of each model (0.026 to 0.049) is comparable to the 0.039 gap between the best and the worst mean.

### 4.2 Which models are tied with the best?

A model is tied with the best when the corrected t-test (α = 0.05) cannot separate them.

| Dataset | Best (mean AUC) | Not significantly worse (corrected t-test, α = 0.05) |
|---|---|---|
| Blood Transfusion | Logistic reg. | k-NN, MLP, Random forest |
| Banknote | Random forest | Logistic reg., k-NN, Hist. grad. boosting |
| Credit-G | Random forest | Logistic reg., MLP, Hist. grad. boosting |
| Spambase | Hist. grad. boosting | Random forest |

On three datasets at least three rivals are tied with the winner. On Spambase, the largest dataset, only random forest stays tied and the other three models are clearly behind.

### 4.3 Plain versus corrected t-test

For each dataset the ten model pairs are compared on their 25 paired fold AUCs (α = 0.05).

| Dataset | Pairs | Plain test significant | Corrected test significant |
|---|---|---|---|
| Blood Transfusion | 10 | 6 | 1 |
| Banknote | 10 | 6 | 4 |
| Credit-G | 10 | 8 | 1 |
| Spambase | 10 | 10 | 8 |
| **Total** | 40 | 30 | 14 |

The plain test calls **30 of 40** differences significant; the corrected test only **14**. More than half of the apparent wins disappear once the overlap between training sets is accounted for. Spambase keeps most of its significant pairs (8 of 10): with 4,601 rows the fold scores are precise enough to separate models, whereas on the three smaller datasets only 1 to 4 of 10 pairs survive.

### 4.4 Ranking across the four datasets (Friedman + Nemenyi)

Rank of each model on each dataset (1 = best, by mean AUC):

| Dataset | Logistic reg. | k-NN | MLP | Random forest | Hist. grad. boosting |
|---|---|---|---|---|---|
| Blood Transfusion | 1 | 2 | 3 | 4 | 5 |
| Banknote | 3 | 4 | 5 | 1 | 2 |
| Credit-G | 2 | 4 | 5 | 1 | 3 |
| Spambase | 4 | 5 | 3 | 2 | 1 |

Average ranks:

| Model | Average rank |
|---|---|
| Random forest | 2.00 |
| Logistic reg. | 2.50 |
| Hist. grad. boosting | 2.75 |
| k-NN | 3.75 |
| MLP | 4.00 |

The Friedman test does not reject the hypothesis that all five models are equivalent (χ² = 4.60, p = 0.33). The Nemenyi critical difference (α = 0.05) is 3.05 rank units. The widest gap, random forest against MLP, is 2.00 ranks, so **no pair of models differs significantly across the four datasets**.

### 4.5 Held-out study (60 / 20 / 20 split)

Two models per dataset: random forest and the logistic-regression baseline. Conformal sets target 90 % coverage. ECE is the calibration error; "sigmoid" and "isotonic" are after recalibration.

| Dataset | Model | n test | Test AUC [95 % CI] | ECE raw | ECE sigmoid | ECE isotonic | Conformal coverage | Mean set size | Singleton rate |
|---|---|---|---|---|---|---|---|---|---|
| Blood Transfusion | Logistic reg. | 150 | 0.712 [0.616, 0.803] | 0.028 | 0.050 | 0.088 | 0.887 | 1.30 | 0.70 |
| Blood Transfusion | Random forest | 150 | 0.688 [0.583, 0.793] | 0.078 | 0.054 | 0.068 | 0.887 | 1.34 | 0.66 |
| Banknote | Logistic reg. | 275 | 1.000 [1.000, 1.000] | 0.015 | 0.021 | 0.007 | 0.938 | 0.94 | 0.94 |
| Banknote | Random forest | 275 | 1.000 [1.000, 1.000] | 0.021 | 0.013 | 0.005 | 0.909 | 0.91 | 0.91 |
| Credit-G | Logistic reg. | 200 | 0.800 [0.731, 0.856] | 0.057 | 0.065 | 0.030 | 0.910 | 1.38 | 0.62 |
| Credit-G | Random forest | 200 | 0.831 [0.771, 0.885] | 0.057 | 0.050 | 0.036 | 0.905 | 1.36 | 0.64 |
| Spambase | Logistic reg. | 921 | 0.977 [0.967, 0.986] | 0.043 | 0.065 | 0.025 | 0.938 | 0.99 | 0.99 |
| Spambase | Random forest | 921 | 0.990 [0.982, 0.997] | 0.065 | 0.020 | 0.013 | 0.916 | 0.93 | 0.93 |

- **Test AUC**: 0.69 to 0.71 (Blood Transfusion), 0.80 to 0.83 (Credit-G), 0.98 to 0.99 (Spambase), and 1.000 on Banknote. The 95 % interval is wide on small test sets: [0.62, 0.80] for logistic regression on Blood Transfusion (150 test rows), about five times the 0.039 spread of the cross-validated means on that dataset.
- **Recalibration** helps when the calibration set is large enough (Spambase random forest: ECE 0.065 → 0.013; Credit-G logistic regression: 0.057 → 0.030) but can hurt on small ones (Blood Transfusion logistic regression: 0.028 → 0.088).
- **Conformal prediction** delivers its guarantee: mean empirical coverage over the 8 rows is **0.911** for a 0.90 target. A mean set size below 1 (Banknote, Spambase) means the set is sometimes empty, which should be read as "abstain".
- **ROC curves** (all five models, held-out set): indistinguishable on Banknote; the MLP is lowest on Credit-G (AUC 0.613) and k-NN lowest on Spambase (AUC 0.960); on Blood Transfusion all five curves stay close to each other.

### 4.6 Other metrics (cross-validation, mean ± sd over the 25 outer folds; bold = best, lower is better)

These tables come from the same nested-CV run. They may differ from the report in the last digit (for example the report gives log loss 0.019 and Brier 0.004 for histogram gradient boosting on Banknote); the winners are the same.

**Log loss**

| Dataset | Logistic reg. | k-NN | MLP | Random forest | Hist. grad. boosting |
|---|---|---|---|---|---|
| Blood Transfusion | **0.477 ± 0.015** | 0.523 ± 0.090 | 0.552 ± 0.034 | 0.501 ± 0.035 | 0.525 ± 0.037 |
| Banknote | 0.029 ± 0.010 | 0.031 ± 0.020 | 0.474 ± 0.112 | 0.030 ± 0.007 | **0.018 ± 0.017** |
| Credit-G | 0.506 ± 0.033 | 0.535 ± 0.027 | 0.527 ± 0.036 | **0.499 ± 0.016** | 0.519 ± 0.037 |
| Spambase | 0.231 ± 0.022 | 0.288 ± 0.031 | 0.203 ± 0.029 | 0.153 ± 0.014 | **0.128 ± 0.019** |

**Brier score**

| Dataset | Logistic reg. | k-NN | MLP | Random forest | Hist. grad. boosting |
|---|---|---|---|---|---|
| Blood Transfusion | **0.1548 ± 0.0050** | 0.1550 ± 0.0080 | 0.1840 ± 0.0147 | 0.1582 ± 0.0092 | 0.1651 ± 0.0121 |
| Banknote | 0.0080 ± 0.0032 | 0.0061 ± 0.0030 | 0.1476 ± 0.0448 | 0.0062 ± 0.0027 | **0.0038 ± 0.0031** |
| Credit-G | 0.1666 ± 0.0110 | 0.1778 ± 0.0069 | 0.1757 ± 0.0133 | **0.1652 ± 0.0068** | 0.1689 ± 0.0102 |
| Spambase | 0.0599 ± 0.0044 | 0.0749 ± 0.0044 | 0.0556 ± 0.0086 | 0.0394 ± 0.0033 | **0.0345 ± 0.0043** |

**Expected calibration error (ECE, 10 bins)**

| Dataset | Logistic reg. | k-NN | MLP | Random forest | Hist. grad. boosting |
|---|---|---|---|---|---|
| Blood Transfusion | **0.058 ± 0.020** | 0.075 ± 0.023 | 0.149 ± 0.029 | 0.076 ± 0.020 | 0.093 ± 0.028 |
| Banknote | 0.018 ± 0.006 | 0.016 ± 0.006 | 0.321 ± 0.079 | 0.022 ± 0.004 | **0.005 ± 0.003** |
| Credit-G | 0.071 ± 0.021 | 0.067 ± 0.020 | **0.064 ± 0.020** | 0.069 ± 0.025 | 0.082 ± 0.022 |
| Spambase | 0.033 ± 0.007 | 0.035 ± 0.006 | 0.033 ± 0.017 | 0.046 ± 0.005 | **0.020 ± 0.005** |

### 4.7 Fit time per outer fold (seconds, mean; indicative only, measured while other jobs were running)

| Dataset | Logistic reg. | k-NN | MLP | Random forest | Hist. grad. boosting |
|---|---|---|---|---|---|
| Blood Transfusion | 0.08 | 0.05 | 0.41 | 3.91 | 5.84 |
| Banknote | 0.11 | 0.06 | 0.40 | 7.29 | 69.14 |
| Credit-G | 0.24 | 0.21 | 1.16 | 7.02 | 29.74 |
| Spambase | 0.44 | 0.25 | 3.77 | 6.63 | 8.03 |

## 5. Discussion and conclusion

**No algorithm consistently outperforms the others.** The winner changes from dataset to dataset, and across the four datasets the Friedman test finds no significant difference between the five families (p = 0.33).

- **Size decides how clear the winner is.** On the three smaller datasets (748 to 1,372 rows) at least three rivals tie with the best. Only on Spambase (4,601 rows) do models clearly separate: histogram gradient boosting and random forest lead, and the other three fall behind.
- **Simple baselines are hard to beat.** Logistic regression is best on Blood Transfusion, tied with the best on Credit-G and Banknote, and second in average rank (2.50) behind random forest (2.00).
- **The MLP is never best** and has the worst average rank (4.00), although it is tied with the best on Blood Transfusion and Credit-G.
- **The plain t-test misleads.** It flags 30 of 40 pairs as significant against 14 for the corrected test; comparisons built on repeated cross-validation should use the corrected test.
- **One small split cannot rank models**, and conformal prediction keeps its promise (coverage 0.911 for a 0.90 target).

**In practice:** start from a regularised linear model and a random forest, use a tree ensemble when the data is large or non-linear, and claim a ranking only when a corrected test supports it.

## 6. Limitations and next steps

- **Four datasets only.** The Nemenyi critical difference is large (3.05 ranks), so the cross-dataset test has little power; more datasets are needed to separate the middle of the ranking.
- **Small grids and one seed.** Grids have 2 to 4 points per model; larger searches could change the MLP and boosting results (the most sensitive to tuning), and a second seed would show the stability of the rankings.
- **Held-out calibration and conformal results cover two models per dataset only** (random forest and logistic regression); the ROC curves cover all five.
- **Next steps:** add datasets and a Bayesian comparison of models, and tune the conformal and recalibration choices on larger calibration sets.

## Reproducibility

These numbers come from running the full protocol above on the four datasets (results filtered to these four from an original seven-dataset run, with the Friedman / Nemenyi test, the plain-vs-corrected counts and the mean AUC over datasets recomputed on the four). `model_comparison.ipynb` reruns everything from `data/*.csv`; it has a `QUICK_RUN` switch for a fast reduced run (the full run takes about 20 minutes on 2 CPU cores). Tiny library-version differences can change the last digit (e.g. Blood Transfusion k-NN AUC 0.741 instead of 0.739).

## References

- Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52, 239–281.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Data: OpenML (ids 1464, 1462, 31, 44); original sources UCI Machine Learning Repository.