# Local Explanation Fidelity in Credit Scoring

Do synthetic training data preserve individual explanations?

MSc Financial Technology dissertation project (MSO4992), Middlesex University London.

---

## What this project asks

Lenders are legally required to tell rejected applicants **why** they were rejected. Because real customer data is restricted by privacy law, synthetic data is increasingly used to train credit scoring models instead.

Synthetic data is normally validated by checking that a model trained on it still **predicts** accurately. No study identified in this review had checked whether the model still attributes its decisions to the same features for the same people.

> When a credit scoring model is trained on synthetic data instead of real data, do the same features drive the model's output for a given individual applicant?

Two sub-questions follow. Do the two models refuse the same applicants? And for the applicants refused by both, are the features driving their scores the same?

## What it found

Accuracy largely survives, and so does agreement on who gets refused. Individual explanations do not.

| Comparison | Mean top-3 Jaccard | Mean Kendall tau | Identical top-3 |
|---|---|---|---|
| Real vs Real (noise baseline) | 0.6887 | 0.5838 | **43.64%** |
| Real vs TVAE | 0.3783 | 0.2360 | **7.03%** |
| Real vs CTGAN | 0.1681 | 0.0572 | **1.47%** |

All models are explained against the same background sample of 100 real training records, so the model is the only thing that varies.

The clearest result concerns the applicants both models would refuse, since these are the people who would receive an adverse action notice either way. At a rejection rate matching the real default rate:

| Model | Refused by both | Different top-3 |
|---|---|---|
| TVAE | 1,075 | **88.65%** |
| CTGAN | 900 | **91.44%** |

This holds despite the TVAE-trained model reaching a ROC-AUC within 0.033 of the real-trained model and refusing 81.01% of the same applicants.

Two controls support the finding. Correcting both synthetic sets to the real 22.12% default rate produced no recovery in agreement. And repeating the generation across ten independent runs produced no run whose agreement approached the noise baseline: the highest synthetic run reached 0.3687, against a lowest noise run of 0.6586.

## Repository structure

    ├── stage1_baseline.py            # real-data reference model, frozen test split
    ├── stage2_synthetic.py           # CTGAN + TVAE generation, TSTR evaluation
    ├── stage3_explanations.py        # noise baseline + per-applicant SHAP agreement
    ├── stage4_balance_control.py     # class-balance control experiment
    ├── stage5_robustness.py          # common background, decision agreement, jointly refused
    ├── stage6_seeds.py               # five seeded runs per generator
    ├── stage7_agreement_seeds.py     # explanation agreement across the ten runs
    ├── stage8_deciles.py             # risk deciles on the common background
    ├── stage9_balance_common.py      # class-balance control on the common background
    ├── requirements.txt
    ├── outputs/                      # results written by each stage
    └── data/                         # source dataset (not tracked, see below)

Stages 1 to 4 are the main pipeline. Stages 5 to 9 are robustness checks added after supervision feedback; each reads the files the main pipeline produced and writes to new output files.

## Dataset

Yeh, I. (2009) *Default of Credit Card Clients*. UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H - CC BY 4.0.

30,000 records, 23 features, 22.12% default rate.

The raw file is not tracked in this repository. Retrieve it with:

```bash
curl -L -o data/creditcard.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
unzip -o data/creditcard.zip -d data/
mv "data/default of credit card clients.xls" data/default_of_credit_card_clients.xls
```

**Note:** the programmatic `ucimlrepo` interface returned a read error for dataset 350 during this study. The static archive above was used instead.

## Reproducing the study

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, set these before running any stage that loads both SDV and XGBoost - without them the process segfaults on a duplicate OpenMP runtime:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

Then run the stages in order. Each consumes artefacts written by its predecessor.

```bash
python stage1_baseline.py            # < 1 min
python stage2_synthetic.py           # ~1 min with the saved synthetic data, ~20 min to generate new
python stage3_explanations.py        # 3-5 min
python stage4_balance_control.py     # 3-5 min
python stage5_robustness.py          # ~10 min
python stage6_seeds.py               # ~90 min to generate, ~2 min with the saved datasets
python stage7_agreement_seeds.py     # ~25 min
python stage8_deciles.py             # ~15 min
python stage9_balance_common.py      # ~10 min
```

**Note on the synthetic data.** The synthetic training sets behind the reported results are included in `outputs/stage2/` and `outputs/stage6/`. The stage 2 generators are unseeded, so new synthetic data would differ; `stage2_synthetic.py` reuses the saved files when they exist and only generates new data if they are missing. The stage 6 generators are seeded, and `stage6_seeds.py` likewise reuses any dataset it finds, so an interrupted run can be restarted without repeating work.

Stages 5 to 9 use SHAP's interventional mode, which is slower than the default because every applicant is evaluated against every record in the background sample.

## Method in brief

**Stage 1.** Clean the data, split 80/20 stratified, train XGBoost on the real training set. The 20% test partition is written to disk and never re-split - every later comparison uses the same 6,000 real applicants. Reference ROC-AUC 0.7768, in line with published results for tree-based models on this dataset.

**Stage 2.** Fit CTGAN and TVAE on the training partition only (never the test set), sample 24,000 synthetic records from each, train identical models, evaluate all three on the frozen real test set. Categorical columns were declared explicitly in the SDV metadata so that codes are not treated as continuous.

**Stage 3.** Compute SHAP values via TreeExplainer for each model over the test set. For every applicant, extract the top three features by absolute contribution and measure agreement between the real-trained and synthetic-trained models. A **noise baseline** is built first: five models trained on stratified 80% subsamples of the real training data, compared against the reference, to quantify how much explanations move for reasons unrelated to synthetic data. Without it the main comparison is uninterpretable (Chen et al., 2024).

**Stage 4.** Undersample each synthetic set to the real 22.12% default rate and repeat the comparison, isolating "synthetic" from "wrong class balance".

**Stages 5 to 9.** Repeat the comparison with a **common background** of 100 real training records, so that every model is explained against the same reference distribution rather than its own training data. Measure whether the models refuse the same applicants, and recompute agreement for the applicants refused by both. Rerun both generators five times each with different seeds and repeat the full comparison across the ten resulting datasets. Recompute the risk decile breakdown and the class-balance control on the common background, so that every reported figure rests on the same basis.

## Results summary

Predictive utility, all evaluated on the same 6,000 real applicants:

| Trained on | ROC-AUC | PR-AUC | Default rate in training data |
|---|---|---|---|
| Real | 0.7768 | 0.5566 | 22.12% |
| CTGAN | 0.6933 | 0.4729 | 45.29% |
| TVAE | 0.7443 | 0.4914 | 11.60% |

Across five seeded runs per generator:

| Generator | Mean ROC-AUC | SD | Range | Mean default rate |
|---|---|---|---|---|
| CTGAN | 0.7103 | 0.0189 | 0.6884 - 0.7331 | 32.98% |
| TVAE | 0.7471 | 0.0083 | 0.7364 - 0.7560 | 14.64% |

Both generators distorted the class balance substantially, in opposite directions, in every one of the ten runs. No run came close to the real 22.12%, and none of the distortion is visible in the utility metrics.

Explanation agreement across the same ten runs:

| Source | Runs | Mean Jaccard | SD | Range | Identical top-3 |
|---|---|---|---|---|---|
| Noise baseline | 5 | 0.6887 | 0.0248 | 0.6586 - 0.7158 | 43.64% |
| TVAE | 5 | 0.3350 | 0.0289 | 0.2993 - 0.3687 | 5.08% |
| CTGAN | 5 | 0.1821 | 0.0638 | 0.1480 - 0.2955 | 1.35% |

The two ranges do not overlap. Among applicants refused by both models, the share receiving a different top three ranged from 87.00% to 97.62% across the CTGAN runs and from 79.39% to 93.40% across the TVAE runs.

Class-balance control, on the common background:

| Generator | Condition | Records | Default rate | Mean Jaccard |
|---|---|---|---|---|
| CTGAN | original | 24,000 | 45.29% | 0.1681 |
| CTGAN | rebalanced | 16,861 | 22.12% | 0.1455 |
| TVAE | original | 24,000 | 11.60% | 0.3783 |
| TVAE | rebalanced | 12,581 | 22.12% | 0.3714 |

Agreement did not recover; if anything both generators moved slightly further from the baseline. Class balance is ruled out as the main driver.

## Limitations

- The **mechanism** is unidentified. Class balance was ruled out as the main cause; what property of synthetic data actually causes the divergence was not established.
- **One dataset**, one country, one credit product. Generalisation is untested, particularly to low-default portfolios such as mortgages.
- The **top three features are a proxy** for adverse action reasons. They are not restricted to contributions that pushed an applicant's score towards refusal, so the measure is broader than what a lender is required to disclose.
- The **control reduced training set size** while correcting class balance, so a second variable changed. TVAE lost 47.6% of its rows and agreement moved by less than 0.01, which argues against sample size being influential, but this is indirect evidence.
- **Ten generator runs** establish that the finding does not depend on a single dataset, but they are a small sample for estimating variation, particularly for CTGAN.
- One model family, one explanation method, one value of k. The common background sample was fixed at 100 records.
- Agreement is measured against a real-data reference, not against ground truth.

## Key references

- Chen, Y., Calabrese, R. and Martin-Barragan, B. (2024) 'Interpretable machine learning for imbalanced credit scoring datasets', *European Journal of Operational Research*, 312(1), pp. 357-372.
- Lundberg, S. M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', *NeurIPS* 30, pp. 4765-4774.
- Lundberg, S. M. et al. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*, 2(1), pp. 56-67.
- Xu, L., Skoularidou, M., Cuesta-Infante, A. and Veeramachaneni, K. (2019) 'Modeling tabular data using conditional GAN', *NeurIPS* 32.
- Yu, K., Ishikura, S., Usukura, Y., Shigoku, Y. and Hayashi, T. (2025) 'SHAP Distance: an explainability-aware metric for evaluating the semantic fidelity of synthetic tabular data', arXiv:2511.17590.

## Author

Gauravkumar Raiyani - MSc Financial Technology, Middlesex University London.
Supervisor: Ann-Ngoc Nguyen.