# Do Synthetic Training Data Preserve Adverse Action Reasons?

Local explanation fidelity in credit scoring models trained on generated data.

MSc Financial Technology dissertation project (MSO4992), Middlesex University London.

---

## What this project asks

Lenders are legally required to tell rejected applicants **why** they were rejected. Because real customer data is restricted by privacy law, synthetic data is increasingly used to train credit scoring models instead.

Synthetic data is normally validated by checking that a model trained on it still **predicts** accurately. Nobody had checked whether the model still gives the same **reasons** - and specifically, whether a given individual receives the same reasons.

> When a credit scoring model is trained on synthetic data rather than real data, do individual applicants receive the same principal reasons for the decision made about them?

## What it found

Accuracy largely survives. Individual explanations do not.

| Comparison | Mean top-3 Jaccard | Mean Kendall tau | Identical top-3 |
|---|---|---|---|
| Real vs Real (noise floor) | 0.7368 | 0.5924 | **50.40%** |
| Real vs TVAE | 0.3394 | 0.1634 | **3.67%** |
| Real vs CTGAN | 0.1696 | 0.0826 | **2.63%** |

Retraining on a resampled subset of the *same real data* leaves 50.4% of applicants with an identical top-three reason set. Retraining on TVAE-generated data leaves 3.67% - despite a ROC-AUC drop of only about three points.

A control experiment ruled out class balance distortion as the cause: correcting both synthetic sets to the real 22.12% default rate produced no recovery in agreement.

An earlier run of both generators, whose synthetic data was not kept, showed the same pattern (see Limitations).

## Repository structure

    .
    ├── fetch_data.py                 # retrieve the UCI dataset
    ├── stage1_baseline.py            # real-data baseline, frozen test split
    ├── stage2_synthetic.py           # CTGAN + TVAE generation, TSTR evaluation
    ├── stage3_explanations.py        # noise baseline + per-applicant SHAP agreement
    ├── stage4_balance_control.py     # class-balance control experiment
    ├── requirements.txt
    ├── outputs/                      # results written by each stage
    └── data/                         # source dataset (not tracked, see below)

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
python stage1_baseline.py          # < 1 min
python stage2_synthetic.py         # ~1 min with the saved synthetic data, ~20 min to generate new data
python stage3_explanations.py      # 3-5 min
python stage4_balance_control.py   # 3-5 min
```

**Note on the synthetic data.** The synthetic training sets behind the reported results are included in `outputs/stage2/`. Because the generators are unseeded, new synthetic data would differ, so `stage2_synthetic.py` reuses the saved files when they exist and only generates new data if they are missing. Running all four stages in order reproduces every reported figure exactly.

## Method in brief

**Stage 1.** Clean the data, split 80/20 stratified, train XGBoost on the real training set. The 20% test partition is written to disk and never re-split - every later comparison uses the same 6,000 real applicants. Baseline ROC-AUC 0.7768, within the published range for this dataset.

**Stage 2.** Fit CTGAN and TVAE on the training partition only (never the test set), sample 24,000 synthetic records from each, train identical models, evaluate all three on the frozen real test set. Categorical columns were declared explicitly in the SDV metadata so that codes are not treated as continuous.

**Stage 3.** Compute SHAP values via TreeExplainer for each model over the test set. For every applicant, extract the top three features by absolute contribution and measure agreement between the real-trained and synthetic-trained models. Crucially, a **noise baseline** is built first: five models trained on resampled real data, compared against the reference, to quantify how much explanations move for reasons unrelated to synthetic data. Without it the main comparison is uninterpretable (Chen et al., 2024).

**Stage 4.** Undersample each synthetic set to the real 22.12% default rate and repeat the comparison, isolating "synthetic" from "wrong class balance".

## Results summary

Predictive utility, all evaluated on the same 6,000 real applicants:

| Trained on | ROC-AUC | PR-AUC | Default rate in training data |
|---|---|---|---|
| Real | 0.7768 | 0.5566 | 22.12% |
| CTGAN | 0.6933 | 0.4729 | 45.29% |
| TVAE | 0.7443 | 0.4914 | 11.60% |

Both generators distorted the class balance substantially, in opposite directions, and neither distortion is visible in the utility metrics.

Control experiment:

| Generator | Condition | Records | Default rate | Mean Jaccard |
|---|---|---|---|---|
| CTGAN | original | 24,000 | 45.29% | 0.1696 |
| CTGAN | rebalanced | 16,861 | 22.12% | 0.1793 |
| TVAE | original | 24,000 | 11.60% | 0.3394 |
| TVAE | rebalanced | 12,581 | 22.12% | 0.3297 |

Agreement did not recover. Class balance is ruled out as the driver.

## Limitations

- The **mechanism** is unidentified. Class balance was eliminated; what property of synthetic data actually causes the divergence was not established.
- **One dataset**, one country, one credit product. Generalisation is untested, particularly to low-default portfolios such as mortgages.
- **Unseeded generators, two runs.** The SDV synthesizers were not given a fixed random seed. The reported results come from the second run, whose synthetic data is committed in `outputs/stage2/`, so every reported figure can be reproduced exactly. An earlier run, whose synthetic data was not kept, gave different individual figures (mean Jaccard 0.1724 for CTGAN and 0.3011 for TVAE; identical top-3 0.68% and 2.17%) but the same overall pattern. Two runs are not enough for confidence intervals.
- The **control reduced training set size** while correcting class balance, so a second variable changed. TVAE lost 47.6% of its rows and agreement moved by 0.0097, which argues against sample size being influential, but this is indirect evidence.
- One model family, one explanation method, one value of k.
- Agreement is measured against a real-data reference, not against ground truth.

## Key references

- Chen, Y., Calabrese, R. and Martin-Barragan, B. (2024) 'Interpretable machine learning for imbalanced credit scoring datasets', *European Journal of Operational Research*, 312(1), pp. 357-372.
- Xu, L., Skoularidou, M., Cuesta-Infante, A. and Veeramachaneni, K. (2019) 'Modeling tabular data using conditional GAN', *NeurIPS* 32, pp. 7335-7345.
- Yu, J., Ishikura, T., Usukura, S., Shigoku, R. and Hayashi, K. (2025) 'SHAP Distance: an explainability-aware metric for evaluating the semantic fidelity of synthetic tabular data', arXiv:2511.17590.
- Lundberg, S. M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', *NeurIPS* 30, pp. 4765-4774.

## Author

Gauravkumar Raiyani - MSc Financial Technology, Middlesex University London.
Supervisor: Ann-Ngoc Nguyen.
