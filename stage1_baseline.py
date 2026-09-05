"""
Stage 1 - real-data baseline.

This is the TRTR run (train real, test real). Everything later in the project
gets compared back to what comes out of here, so this is the file that has to
be right.

Two things this script has to get correct or nothing downstream works:
  1. The test set is carved out here and NEVER changes again. Every model I
     train later (synthetic, resampled, whatever) gets scored on this exact
     same set of real applicants. If the test set moves, the comparison is
     meaningless.
  2. Seeds are fixed and written down, because Chen et al. (2024) show
     explanations wobble just from resampling. I need to be able to
     reproduce a run exactly before I can claim anything about synthetic data.

Dataset: UCI "Default of Credit Card Clients" - 30,000 rows, 22.12% default.
Paper 4 (Japinye & Adedugbe 2025) used the same one, so I can sanity check my
AUC against a published number instead of guessing whether it's reasonable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# config - everything I might want to change lives here, not buried below
# ----------------------------------------------------------------------

DATA_PATH = Path("data/default_of_credit_card_clients.xls")  
OUT_DIR = Path("outputs/stage1")
SEED = 42
TEST_FRAC = 0.20

TARGET = "default"


# ----------------------------------------------------------------------
# load
# ----------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    """
    The UCI file is an .xls with a junk first row - the real column names are
    on row 2, hence header=1. If you've already converted it to CSV this
    handles that too.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Can't find {path}. Download 'default of credit card clients.xls' "
            "from the UCI repository and put it in a data/ folder next to this script."
        )

    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)

    return df


# ----------------------------------------------------------------------
# clean
# ----------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Known problems with this dataset - all documented in the literature, and
    I need to say in the methodology chapter that I handled them:

      - target column has an awkward name with spaces
      - the first repayment column is PAY_0, not PAY_1 (inconsistent naming)
      - EDUCATION has values 0, 5, 6 that aren't in the codebook
      - MARRIAGE has a 0 that isn't in the codebook
      - ID is a row number, not a feature. Leaving it in would let the model
        cheat and would also pollute the SHAP values, which is the whole point
        of the project, so it goes.
    """
    df = df.copy()

    df = df.rename(columns={
        "default payment next month": TARGET,
        "default.payment.next.month": TARGET,
        "PAY_0": "PAY_1",
    })

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # collapse the undocumented categories into the existing "other" bucket
    # rather than dropping the rows - dropping would lose ~1.5% of the data
    # for no good reason
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df


def sanity_check(df: pd.DataFrame) -> None:
    """
    If these don't match the published figures I've loaded the wrong file or
    mangled it. Better to crash here than to find out in chapter 5.
    """
    n_rows = len(df)
    default_rate = df[TARGET].mean()

    print(f"rows loaded          : {n_rows:,}")
    print(f"default rate         : {default_rate:.4%}  (expected ~22.12%)")
    print(f"missing values       : {df.isna().sum().sum()}")
    print(f"features (excl. y)   : {df.shape[1] - 1}")

    if n_rows != 30_000:
        print(f"  WARNING: expected 30,000 rows, got {n_rows:,}")
    if not 0.21 < default_rate < 0.23:
        print(f"  WARNING: default rate {default_rate:.4%} is off - check the file")


# ----------------------------------------------------------------------
# split - this is the important bit
# ----------------------------------------------------------------------

def make_split(df: pd.DataFrame):
    """
    Stratified so the 22% default rate is preserved in both halves. If I let
    this drift, the test set stops being comparable and Chen et al.'s point
    about imbalance affecting explanations starts contaminating my results.

    The test indices get written to disk. Stage 2 loads them rather than
    re-splitting, so there is physically no way for the test set to change
    between experiments.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_FRAC,
        stratify=y,
        random_state=SEED,
    )

    print(f"\ntrain: {len(X_train):,} rows, {y_train.mean():.4%} default")
    print(f"test : {len(X_test):,} rows, {y_test.mean():.4%} default")

    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------

def train_model(X_train, y_train) -> XGBClassifier:
    """
    Deliberately NOT using scale_pos_weight or any resampling here.

    Reason: Chen et al. (2024) showed that how you handle imbalance changes
    the explanations, not just the accuracy. Since my whole project is about
    whether explanations survive a change in training data, I need to hold
    imbalance handling constant and untouched. The 22% imbalance stays as-is
    for every model in the study. I'll say this explicitly in the methodology.

    Hyperparameters are conservative on purpose - I'm not chasing a leaderboard
    score, I need a competent, stable, believable baseline.
    """
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
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    """
    AUC because that's what everyone reports and I need to compare to Paper 4.
    PR-AUC because with a 22% positive rate, AUC alone flatters the model -
    and PR-AUC is the one that actually reflects performance on defaulters,
    who are the people my whole research question is about.
    """
    proba = model.predict_proba(X_test)[:, 1]

    results = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "baseline_pr_auc": y_test.mean(),  # what random guessing would get
    }

    print("\n--- real-data baseline (TRTR) ---")
    print(f"ROC-AUC : {results['roc_auc']:.4f}")
    print(f"PR-AUC  : {results['pr_auc']:.4f}  (random would be {results['baseline_pr_auc']:.4f})")
    print("\nPublished work on this dataset lands roughly in the 0.77-0.79 ROC-AUC")
    print("range. If I'm miles off that, something is wrong with my pipeline.")

    return results


# ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw(DATA_PATH))
    sanity_check(df)

    X_train, X_test, y_train, y_test = make_split(df)

    model = train_model(X_train, y_train)
    results = evaluate(model, X_test, y_test)

    # freeze the test set so stage 2 can't accidentally use a different one
    pd.Series(X_test.index, name="test_index").to_csv(
        OUT_DIR / "test_indices.csv", index=False
    )
    X_test.assign(**{TARGET: y_test}).to_csv(OUT_DIR / "test_real.csv", index=False)
    X_train.assign(**{TARGET: y_train}).to_csv(OUT_DIR / "train_real.csv", index=False)

    model.save_model(OUT_DIR / "model_real.json")
    pd.DataFrame([results]).to_csv(OUT_DIR / "baseline_metrics.csv", index=False)

    print(f"\nsaved everything to {OUT_DIR}/")
    print("train_real.csv is what goes into CTGAN in stage 2.")


if __name__ == "__main__":
    main()          