"""
Stage 2 - synthetic data generation and the TSTR comparison.

What happens here:
  1. Load the REAL training split that stage 1 froze
  2. Fit CTGAN and TVAE on it, sample the same number of rows back out
  3. Train an XGBoost on each synthetic set, using the exact same settings
     as the real baseline
  4. Score everything on the frozen REAL test set from stage 1

Three rules I'm not allowed to break here, or the whole comparison is void:

  - The generators only ever see train_real.csv. If they saw the test set,
    synthetic rows would leak test information and every number after this
    would be garbage.
  - The test set is loaded from disk, never re-split. Same 6,000 real
    applicants that scored the baseline.
  - The XGBoost settings are identical across all three models. The ONLY
    thing that changes is the training data. That's the entire experiment.

Expected outcome based on the literature: the synthetic models land close to
the real baseline on ROC-AUC. That's the "synthetic data works fine" result
everyone reports, and it's the result my project then goes on to complicate
in stage 3 by looking at the explanations instead of the accuracy.
"""

from pathlib import Path

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

IN_DIR = Path("outputs/stage1")
OUT_DIR = Path("outputs/stage2")
SEED = 42
TARGET = "default"

# START WITH EPOCHS = 10. That runs in about two minutes and proves the whole
# script works end to end. Only once it completes cleanly, set this to 300 and
# do the real run. Do NOT go straight to 300 - if something breaks on the last
# line you'll have burned 40 minutes finding out.
EPOCHS = 300

# Columns that are codes, not quantities. If I let SDV treat these as numbers
# the generator will happily produce EDUCATION = 2.37, which is meaningless.
# Getting this right matters more than any hyperparameter here.
CATEGORICAL = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    TARGET,
]


# ----------------------------------------------------------------------

def build_metadata(df: pd.DataFrame) -> Metadata:
    """
    SDV guesses column types automatically, but its guesses on this dataset
    are wrong in the ways that matter - the repayment status codes and the
    demographic codes all look like integers to it. So I detect first, then
    override the ones I know about.
    """
    meta = Metadata.detect_from_dataframe(data=df, table_name="credit")

    for col in CATEGORICAL:
        if col in df.columns:
            meta.update_column(column_name=col, table_name="credit", sdtype="categorical")

    meta.validate()
    return meta


def generate(name: str, synth, train_real: pd.DataFrame) -> pd.DataFrame:
    """Fit a generator and sample the same number of rows the real set has."""
    print(f"\n--- fitting {name} ({EPOCHS} epochs) ---")
    synth.fit(train_real)

    synthetic = synth.sample(num_rows=len(train_real))

    out = OUT_DIR / f"train_{name.lower()}.csv"
    synthetic.to_csv(out, index=False)

    print(f"{name}: generated {len(synthetic):,} rows -> {out}")
    print(f"{name}: default rate {synthetic[TARGET].mean():.4%}  "
          f"(real is {train_real[TARGET].mean():.4%})")

    return synthetic


def train_and_score(name: str, train_df: pd.DataFrame,
                    X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Same hyperparameters as stage 1, no exceptions. If I tuned each model
    separately I'd be measuring my tuning, not the data.
    """
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    # column order has to match the test set or XGBoost will silently
    # misalign features
    X = X[X_test.columns]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X, y)
    model.save_model(OUT_DIR / f"model_{name.lower()}.json")

    proba = model.predict_proba(X_test)[:, 1]

    return {
        "trained_on": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }


# ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_real = pd.read_csv(IN_DIR / "train_real.csv")
    test_real = pd.read_csv(IN_DIR / "test_real.csv")

    X_test = test_real.drop(columns=[TARGET])
    y_test = test_real[TARGET]

    print(f"train_real : {train_real.shape}")
    print(f"test_real  : {test_real.shape}  <- frozen in stage 1, never re-split")

    meta = build_metadata(train_real)

    results = [train_and_score("Real", train_real, X_test, y_test)]

    for name, synth_cls in [("CTGAN", CTGANSynthesizer), ("TVAE", TVAESynthesizer)]:
        synth = synth_cls(meta, epochs=EPOCHS, verbose=True)
        synthetic = generate(name, synth, train_real)
        results.append(train_and_score(name, synthetic, X_test, y_test))

    table = pd.DataFrame(results)
    table.to_csv(OUT_DIR / "tstr_comparison.csv", index=False)

    print("\n" + "=" * 55)
    print("TSTR comparison - all scored on the same real test set")
    print("=" * 55)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsaved to {OUT_DIR}/")
    print("\nIf the synthetic rows are close to Real on ROC-AUC, that's the")
    print("expected result and it sets up stage 3 - accuracy survives, so now")
    print("I go and check whether the explanations survive too.")


if __name__ == "__main__":
    main()