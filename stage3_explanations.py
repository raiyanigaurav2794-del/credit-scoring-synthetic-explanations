import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

S1_DIR = "outputs/stage1"
S2_DIR = "outputs/stage2"
OUT_DIR = "outputs/stage3"
# TOP_K = 3 because adverse action notices typically give three or four reasons.
# N_NOISE_RUNS = 5 is how many resampled-real models build the noise floor
SEED = 42
TOP_K = 3
N_NOISE_RUNS = 5

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
)

os.makedirs(OUT_DIR, exist_ok=True)


# SHAP gives each applicant's feature contributions. 
# TreeExplainer is exact for tree models, so any wobble I measure comes from the model itself.

def get_shap(model, X):
    vals = shap.TreeExplainer(model).shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return np.asarray(vals)


# for each person, take their top 3 features from each model and measure how many overlap. 
# Absolute values because direction doesn't matter for a reason code.

def compare(shap_a, shap_b):
    n = len(shap_a)
    jaccard = np.zeros(n)
    tau = np.zeros(n)

    for i in range(n):
        a = set(np.argsort(np.abs(shap_a[i]))[-TOP_K:])
        b = set(np.argsort(np.abs(shap_b[i]))[-TOP_K:])
        jaccard[i] = len(a & b) / len(a | b)

        t, _ = kendalltau(np.abs(shap_a[i]), np.abs(shap_b[i]))
        tau[i] = 0.0 if np.isnan(t) else t

    return pd.DataFrame({"jaccard": jaccard, "kendall_tau": tau})


#split applicants into ten risk groups to see if divergence depends on how risky they are. 
# pct_identical_topk is the share whose top 3 match exactly.

def summarise(label, cmp_df, risk):
    df = cmp_df.copy()
    df["decile"] = pd.qcut(risk, 10, labels=False, duplicates="drop")

    overall = pd.DataFrame([{
        "comparison": label,
        "decile": "all",
        "mean_jaccard": df["jaccard"].mean(),
        "mean_tau": df["kendall_tau"].mean(),
        "pct_identical_topk": (df["jaccard"] == 1.0).mean(),
        "n": len(df),
    }])

    by_decile = df.groupby("decile").agg(
        mean_jaccard=("jaccard", "mean"),
        mean_tau=("kendall_tau", "mean"),
        pct_identical_topk=("jaccard", lambda s: (s == 1.0).mean()),
        n=("jaccard", "size"),
    ).reset_index()

    by_decile["comparison"] = label
    by_decile["decile"] = by_decile["decile"].astype(str)

    return pd.concat([overall, by_decile], ignore_index=True)


# ---- load ----

train_real = pd.read_csv(S1_DIR + "/train_real.csv")
test_real = pd.read_csv(S1_DIR + "/test_real.csv")

X_test = test_real.drop(columns=["default"])
features = list(X_test.columns)

print("test set:", len(X_test), "applicants,", len(features), "features")


# ---- reference model on real data ----

print("training reference model")

ref = XGBClassifier(**XGB_PARAMS, random_state=SEED)
ref.fit(train_real[features], train_real["default"])

shap_ref = get_shap(ref, X_test)
risk = ref.predict_proba(X_test)[:, 1]

all_results = []


# five models on resampled real data, compared to the reference.
# Nothing meaningful changed, so any disagreement here is just noise. 
# This is the bar the synthetic comparisons have to beat.

print("building noise baseline")

noise_frames = []

for run in range(N_NOISE_RUNS):
    sub, _ = train_test_split(
        train_real,
        train_size=0.8,
        stratify=train_real["default"],
        random_state=1000 + run,
    )

    m = XGBClassifier(**XGB_PARAMS, random_state=1000 + run)
    m.fit(sub[features], sub["default"])

    cmp_df = compare(shap_ref, get_shap(m, X_test))
    noise_frames.append(cmp_df)

    print("  run", run + 1, "jaccard", round(cmp_df["jaccard"].mean(), 4))

noise = pd.concat(noise_frames, ignore_index=True)
all_results.append(summarise("Real vs Real (noise floor)", noise,
                             np.tile(risk, N_NOISE_RUNS)))


# ---- real vs each synthetic set ----

for name in ["ctgan", "tvae"]:
    path = S2_DIR + "/train_" + name + ".csv"

    if not os.path.exists(path):
        print("skipping", name, "- file not found")
        continue

    print("comparing real vs", name)

    syn = pd.read_csv(path)

    m = XGBClassifier(**XGB_PARAMS, random_state=SEED)
    m.fit(syn[features], syn["default"])

    cmp_df = compare(shap_ref, get_shap(m, X_test))
    all_results.append(summarise("Real vs " + name.upper(), cmp_df, risk))

    cmp_df.assign(risk=risk).to_csv(
        OUT_DIR + "/per_applicant_" + name + ".csv", index=False)


# ---- save and print ----

table = pd.concat(all_results, ignore_index=True)
table.to_csv(OUT_DIR + "/explanation_agreement.csv", index=False)

print()
print("explanation agreement, top-3 reasons per applicant")
print(table[table["decile"] == "all"].to_string(index=False))

print()
print("by predicted risk decile")
print(table[table["decile"] != "all"].pivot(
    index="decile", columns="comparison", values="mean_jaccard").to_string())

print()
print("saved to", OUT_DIR)