import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from xgboost import XGBClassifier
import os

S1_DIR = "outputs/stage1"
S2_DIR = "outputs/stage2"
OUT_DIR = "outputs/stage4"

SEED = 42
TOP_K = 3

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


# drop rows from the over-represented class until rates match

def rebalance(df, target_rate):
    pos = df[df["default"] == 1]
    neg = df[df["default"] == 0]

    current = len(pos) / len(df)

# cut rows from whichever class is over-represented until the rate matches real.
# Only deleting, never duplicating — I don't want copies of already-synthetic rows.
    if current > target_rate:
        n_pos = int(round(target_rate * len(neg) / (1 - target_rate)))
        pos = pos.sample(n=min(n_pos, len(pos)), random_state=SEED)
    else:
        n_neg = int(round(len(pos) * (1 - target_rate) / target_rate))
        neg = neg.sample(n=min(n_neg, len(neg)), random_state=SEED)

    out = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
    return out.reset_index(drop=True)


# same two helpers as stage 3 

def get_shap(model, X):
    vals = shap.TreeExplainer(model).shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return np.asarray(vals)


def agreement(shap_a, shap_b):
    n = len(shap_a)
    jac = np.zeros(n)
    tau = np.zeros(n)

    for i in range(n):
        a = set(np.argsort(np.abs(shap_a[i]))[-TOP_K:])
        b = set(np.argsort(np.abs(shap_b[i]))[-TOP_K:])
        jac[i] = len(a & b) / len(a | b)

        t, _ = kendalltau(np.abs(shap_a[i]), np.abs(shap_b[i]))
        tau[i] = 0.0 if np.isnan(t) else t

    return {
        "mean_jaccard": jac.mean(),
        "mean_tau": tau.mean(),
        "pct_identical_topk": (jac == 1.0).mean(),
    }


# ---- load ----

train_real = pd.read_csv(S1_DIR + "/train_real.csv")
test_real = pd.read_csv(S1_DIR + "/test_real.csv")

X_test = test_real.drop(columns=["default"])
features = list(X_test.columns)

real_rate = train_real["default"].mean()

print("real default rate:", round(real_rate * 100, 2), "%")
print("test set:", len(X_test), "applicants")


# reference model, same as stage 3 

ref = XGBClassifier(**XGB_PARAMS, random_state=SEED)
ref.fit(train_real[features], train_real["default"])
shap_ref = get_shap(ref, X_test)


# un each generator both ways in the same execution so nothing else can differ.
# Record row counts because undersampling shrinks the data.

rows = []

for name in ["ctgan", "tvae"]:
    path = S2_DIR + "/train_" + name + ".csv"

    if not os.path.exists(path):
        print("skipping", name, "- file not found")
        continue

    syn = pd.read_csv(path)

    if syn["default"].nunique() < 2:
        print("skipping", name, "- only one class in the synthetic data")
        continue

    syn_bal = rebalance(syn, real_rate)
    syn_bal.to_csv(OUT_DIR + "/train_" + name + "_balanced.csv", index=False)

    print()
    print(name.upper())
    print("  original:", len(syn), "rows,",
          round(syn["default"].mean() * 100, 2), "% default")
    print("  balanced:", len(syn_bal), "rows,",
          round(syn_bal["default"].mean() * 100, 2), "% default")

    for label, data in [("original", syn), ("balanced", syn_bal)]:
        m = XGBClassifier(**XGB_PARAMS, random_state=SEED)
        m.fit(data[features], data["default"])

        res = agreement(shap_ref, get_shap(m, X_test))
        res["generator"] = name.upper()
        res["version"] = label
        res["n_train"] = len(data)
        res["default_rate"] = data["default"].mean()
        rows.append(res)

        print("  ", label, "jaccard", round(res["mean_jaccard"], 4),
              "identical top-3", round(res["pct_identical_topk"] * 100, 2), "%")


#  save and print 

table = pd.DataFrame(rows)[["generator", "version", "n_train", "default_rate",
                            "mean_jaccard", "mean_tau", "pct_identical_topk"]]

table.to_csv(OUT_DIR + "/balance_control.csv", index=False)

print()
print("does matching the class balance recover the explanations?")
print(table.to_string(index=False))
print()
print("stage 3 noise floor was jaccard 0.7368, identical top-3 50.40%")
print()
print("saved to", OUT_DIR)