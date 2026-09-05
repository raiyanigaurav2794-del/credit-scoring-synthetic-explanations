"""
Stage 4 - the control experiment.

Stage 3 found a big drop in per-applicant explanation agreement when the
model was trained on synthetic data. Problem: the synthetic sets ALSO had
the wrong class balance (CTGAN 38.58%, TVAE 13.45%, real 22.12%). So two
things changed at once and I can't say which one caused the divergence.

This script changes only one thing at a time.

For each generator I build a rebalanced version of its synthetic data by
dropping rows from whichever class is over-represented, until the default
rate matches the real 22.12%. Everything else about the data is untouched -
still fully synthetic, same generator, same rows otherwise.

Then I re-run the exact same SHAP comparison from stage 3.

Reading the result:
  - agreement stays low  -> class balance was NOT the driver. Being
                            synthetic is what breaks the explanations.
                            My original claim holds.
  - agreement jumps up   -> class balance WAS the driver. Synthetic data
                            harms explanations BECAUSE generators distort
                            the class ratio. Different claim, still a real
                            finding, and a more mechanistic one.

Either way I can stop hedging in the discussion chapter.

One honest limitation to write up: rebalancing by undersampling shrinks the
training set, and a smaller training set is itself a change. I report the
row counts so the reader can judge. Chen et al. (2024) undersampled too, so
there's precedent, but it isn't free.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from xgboost import XGBClassifier

S1_DIR = Path("outputs/stage1")
S2_DIR = Path("outputs/stage2")
OUT_DIR = Path("outputs/stage4")

SEED = 42
TARGET = "default"
TOP_K = 3

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    eval_metric="logloss", n_jobs=-1,
)


def rebalance(df: pd.DataFrame, target_rate: float, seed: int) -> pd.DataFrame:
    """
    Drop rows from the over-represented class until the default rate matches
    the real data. Undersampling, not oversampling - I don't want to invent
    duplicate rows on top of already-synthetic rows.
    """
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0]

    current = len(pos) / len(df)

    if current > target_rate:
        # too many defaulters - keep all negatives, cut positives
        n_pos = int(round(target_rate * len(neg) / (1 - target_rate)))
        pos = pos.sample(n=min(n_pos, len(pos)), random_state=seed)
    else:
        # too few defaulters - keep all positives, cut negatives
        n_neg = int(round(len(pos) * (1 - target_rate) / target_rate))
        neg = neg.sample(n=min(n_neg, len(neg)), random_state=seed)

    out = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def shap_matrix(model, X):
    vals = shap.TreeExplainer(model).shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return np.asarray(vals)


def agreement(shap_a, shap_b, k=TOP_K):
    n = shap_a.shape[0]
    jac = np.zeros(n)
    tau = np.zeros(n)
    for i in range(n):
        a = set(np.argsort(np.abs(shap_a[i]))[-k:])
        b = set(np.argsort(np.abs(shap_b[i]))[-k:])
        jac[i] = len(a & b) / len(a | b)
        t, _ = kendalltau(np.abs(shap_a[i]), np.abs(shap_b[i]))
        tau[i] = 0.0 if np.isnan(t) else t
    return {
        "mean_jaccard": jac.mean(),
        "mean_tau": tau.mean(),
        "pct_identical_topk": (jac == 1.0).mean(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_real = pd.read_csv(S1_DIR / "train_real.csv")
    test_real = pd.read_csv(S1_DIR / "test_real.csv")

    X_test = test_real.drop(columns=[TARGET])
    features = list(X_test.columns)

    real_rate = train_real[TARGET].mean()
    print(f"real default rate: {real_rate:.4%}")
    print(f"test set: {len(X_test):,} real applicants\n")

    # reference model - same one stage 3 used
    ref = XGBClassifier(**XGB_PARAMS, random_state=SEED)
    ref.fit(train_real[features], train_real[TARGET])
    shap_ref = shap_matrix(ref, X_test)

    rows = []

    for name in ["ctgan", "tvae"]:
        path = S2_DIR / f"train_{name}.csv"
        if not path.exists():
            print(f"skipping {name.upper()} - {path} missing")
            continue

        syn = pd.read_csv(path)

        if syn[TARGET].nunique() < 2:
            print(f"skipping {name.upper()} - synthetic set has only one class, "
                  "nothing to rebalance\n")
            continue

        syn_bal = rebalance(syn, real_rate, SEED)
        syn_bal.to_csv(OUT_DIR / f"train_{name}_balanced.csv", index=False)

        print(f"{name.upper()}")
        print(f"  original : {len(syn):,} rows, {syn[TARGET].mean():.4%} default")
        print(f"  balanced : {len(syn_bal):,} rows, {syn_bal[TARGET].mean():.4%} default")

        for label, data in [("original", syn), ("balanced", syn_bal)]:
            m = XGBClassifier(**XGB_PARAMS, random_state=SEED)
            m.fit(data[features], data[TARGET])

            res = agreement(shap_ref, shap_matrix(m, X_test))
            res["generator"] = name.upper()
            res["version"] = label
            res["n_train"] = len(data)
            res["default_rate"] = data[TARGET].mean()
            rows.append(res)

            print(f"  {label:<9} -> jaccard {res['mean_jaccard']:.4f}, "
                  f"identical top-3 {res['pct_identical_topk']:.2%}")
        print()

    table = pd.DataFrame(rows)[
        ["generator", "version", "n_train", "default_rate",
         "mean_jaccard", "mean_tau", "pct_identical_topk"]
    ]
    table.to_csv(OUT_DIR / "balance_control.csv", index=False)

    print("=" * 72)
    print("CONTROL: does matching the class balance recover the explanations?")
    print("=" * 72)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCompare against stage 3's noise floor: jaccard 0.7368, "
          "identical top-3 50.40%")
    print("\nIf 'balanced' looks like 'original', class balance was not the")
    print("driver and the divergence is caused by synthetic generation itself.")
    print("If 'balanced' moves up toward the noise floor, the distorted class")
    print("ratio was doing the damage.")
    print(f"\nsaved to {OUT_DIR}/")


if __name__ == "__main__":
    main() 