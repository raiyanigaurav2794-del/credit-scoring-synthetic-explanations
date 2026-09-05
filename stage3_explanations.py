"""
Stage 3 - the actual contribution.

Stage 2 answered "do the models predict the same?". This one answers
"do they give the same reasons to the same person?".

The distinction is the whole project. Paper 2 (Yu et al. 2025) compared
AVERAGE feature attributions - one number per feature across everyone. That
can match perfectly while every individual applicant gets a different
explanation, the same way two schools can have identical class averages with
completely different individual marks.

Credit lending cares about the individual. Under ECOA a rejected applicant
gets told THEIR top reasons, not the portfolio's average reasons. So that's
what I measure.

--- the noise baseline, and why it comes first ---

Chen et al. (2024) showed SHAP explanations wobble on their own under class
imbalance, with no synthetic data involved. So if I just compare real vs
synthetic and find disagreement, I can't tell whether the synthetic data
caused it or whether SHAP is simply noisy at a 22% default rate.

Fix: first train several models on RESAMPLED REAL data with different seeds
and measure how much they disagree with each other. That's the noise floor -
how much explanations move for no reason at all.

Then compare real vs synthetic. Only the amount ABOVE the noise floor counts
as an effect of the synthetic data. Without this, the headline finding is
unfalsifiable and the viva will take it apart.

--- what gets measured ---

For each of the 6,000 real test applicants:
  - top-k overlap (Jaccard): of your top 3 reasons under model A, how many
    also appear in your top 3 under model B?
  - rank correlation (Kendall's tau): do the features line up in the same
    order for you specifically?

Then everything gets split by predicted risk decile, because Paper 2 admitted
their generators smooth out the tails - and in credit scoring the tails are
the high-risk applicants who actually receive adverse action notices.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

S1_DIR = Path("outputs/stage1")
S2_DIR = Path("outputs/stage2")
OUT_DIR = Path("outputs/stage3")

SEED = 42
TARGET = "default"

TOP_K = 3          # adverse action notices typically give 3-4 reasons
N_NOISE_RUNS = 5   # how many resampled-real models to build the noise floor
N_DECILES = 10

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    n_jobs=-1,
)


# ----------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------

def shap_matrix(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    One row per applicant, one column per feature. TreeExplainer is exact for
    tree models, so there's no sampling noise from the explainer itself -
    any wobble I measure comes from the model, which is what I want.
    """
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)

    # some versions hand back a list (one array per class) for binary problems
    if isinstance(vals, list):
        vals = vals[1]

    return np.asarray(vals)


def top_k_features(row: np.ndarray, k: int) -> set:
    """
    Which k features drove THIS person's score the hardest.

    Using absolute value: a feature that pushed the score down by 0.4 is just
    as much "a reason" as one that pushed it up by 0.4. An adverse action
    notice names whatever mattered most, regardless of direction.
    """
    return set(np.argsort(np.abs(row))[-k:])


# ----------------------------------------------------------------------
# comparison metrics
# ----------------------------------------------------------------------

def compare(shap_a: np.ndarray, shap_b: np.ndarray, k: int) -> pd.DataFrame:
    """Per-applicant agreement between two sets of explanations."""
    n = shap_a.shape[0]
    jaccard = np.zeros(n)
    tau = np.zeros(n)

    for i in range(n):
        a_top = top_k_features(shap_a[i], k)
        b_top = top_k_features(shap_b[i], k)

        jaccard[i] = len(a_top & b_top) / len(a_top | b_top)

        # rank agreement across ALL features for this person, not just top k
        t, _ = kendalltau(np.abs(shap_a[i]), np.abs(shap_b[i]))
        tau[i] = 0.0 if np.isnan(t) else t

    return pd.DataFrame({"jaccard": jaccard, "kendall_tau": tau})


def summarise(label: str, cmp_df: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    """Overall figures plus a breakdown by predicted-risk decile."""
    df = cmp_df.copy()
    df["risk"] = risk
    df["decile"] = pd.qcut(risk, N_DECILES, labels=False, duplicates="drop")

    overall = pd.DataFrame([{
        "comparison": label,
        "decile": "all",
        "mean_jaccard": df["jaccard"].mean(),
        "mean_tau": df["kendall_tau"].mean(),
        "pct_identical_topk": (df["jaccard"] == 1.0).mean(),
        "n": len(df),
    }])

    by_decile = (
        df.groupby("decile")
          .agg(mean_jaccard=("jaccard", "mean"),
               mean_tau=("kendall_tau", "mean"),
               pct_identical_topk=("jaccard", lambda s: (s == 1.0).mean()),
               n=("jaccard", "size"))
          .reset_index()
    )
    by_decile["comparison"] = label
    by_decile["decile"] = by_decile["decile"].astype(str)

    return pd.concat([overall, by_decile], ignore_index=True)


# ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_real = pd.read_csv(S1_DIR / "train_real.csv")
    test_real = pd.read_csv(S1_DIR / "test_real.csv")

    X_test = test_real.drop(columns=[TARGET])
    feature_names = list(X_test.columns)

    print(f"test set: {X_test.shape[0]:,} real applicants, "
          f"{X_test.shape[1]} features")

    # ---------- the reference model ----------
    print("\ntraining reference model on real data...")
    ref = XGBClassifier(**XGB_PARAMS, random_state=SEED)
    ref.fit(train_real.drop(columns=[TARGET])[feature_names], train_real[TARGET])

    shap_ref = shap_matrix(ref, X_test)
    risk = ref.predict_proba(X_test)[:, 1]

    all_results = []

    # ---------- noise floor: real vs real ----------
    # Same data, same settings, different random subsample. Any disagreement
    # here is pure noise. This is the number every later comparison has to
    # beat to mean anything.
    print(f"\nbuilding noise baseline ({N_NOISE_RUNS} resampled-real models)...")
    noise_frames = []

    for run in range(N_NOISE_RUNS):
        sub, _ = train_test_split(
            train_real,
            train_size=0.8,
            stratify=train_real[TARGET],
            random_state=1000 + run,
        )
        m = XGBClassifier(**XGB_PARAMS, random_state=1000 + run)
        m.fit(sub.drop(columns=[TARGET])[feature_names], sub[TARGET])

        cmp_df = compare(shap_ref, shap_matrix(m, X_test), TOP_K)
        noise_frames.append(cmp_df)
        print(f"  run {run + 1}: mean jaccard {cmp_df['jaccard'].mean():.4f}, "
              f"mean tau {cmp_df['kendall_tau'].mean():.4f}")

    noise = pd.concat(noise_frames, ignore_index=True)
    all_results.append(summarise("Real vs Real (noise floor)",
                                 noise, np.tile(risk, N_NOISE_RUNS)))

    # ---------- the real comparisons ----------
    for name in ["ctgan", "tvae"]:
        path = S2_DIR / f"train_{name}.csv"
        if not path.exists():
            print(f"\nskipping {name.upper()} - {path} not found. "
                  "Run stage 2 with EPOCHS=300 first.")
            continue

        print(f"\ncomparing Real vs {name.upper()}...")
        syn = pd.read_csv(path)

        m = XGBClassifier(**XGB_PARAMS, random_state=SEED)
        m.fit(syn.drop(columns=[TARGET])[feature_names], syn[TARGET])

        cmp_df = compare(shap_ref, shap_matrix(m, X_test), TOP_K)
        all_results.append(summarise(f"Real vs {name.upper()}", cmp_df, risk))

        cmp_df.assign(risk=risk).to_csv(
            OUT_DIR / f"per_applicant_{name}.csv", index=False)

    table = pd.concat(all_results, ignore_index=True)
    table.to_csv(OUT_DIR / "explanation_agreement.csv", index=False)

    # ---------- report ----------
    overall = table[table["decile"] == "all"]

    print("\n" + "=" * 70)
    print(f"EXPLANATION AGREEMENT - top-{TOP_K} reasons, per applicant")
    print("=" * 70)
    print(overall[["comparison", "mean_jaccard", "mean_tau",
                   "pct_identical_topk", "n"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nHow to read this:")
    print("  mean_jaccard       - overlap of top-3 reasons for the same person")
    print("  mean_tau           - do features rank the same order for them")
    print("  pct_identical_topk - share of applicants whose top 3 match exactly")
    print("\nThe noise floor row is the control. A synthetic comparison only")
    print("means something if it sits clearly BELOW that row.")

    print("\n" + "=" * 70)
    print("BY PREDICTED RISK DECILE (9 = highest risk = most likely rejected)")
    print("=" * 70)
    by_dec = table[table["decile"] != "all"]
    print(by_dec.pivot(index="decile", columns="comparison",
                      values="mean_jaccard")
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nIf agreement falls off in the high deciles, that is the finding:")
    print("explanations diverge most for exactly the applicants who are")
    print("legally owed an explanation.")
    print(f"\nsaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()