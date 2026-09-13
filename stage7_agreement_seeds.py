import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEEDS = [101, 202, 303, 404, 505]
TOP_K = 3
N_BACKGROUND = 100
N_NOISE_RUNS = 5
REJECTION_RATE = 0.2212
IN_DIR = "outputs/stage6"

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  eval_metric="logloss", enable_categorical=False)

train_real = pd.read_csv("outputs/stage1/train_real.csv")
test_real = pd.read_csv("outputs/stage1/test_real.csv")
X_test = test_real.drop(columns=["default"])
features = list(X_test.columns)

# same shared background as stage 5, so these figures are comparable with it
background = train_real[features].sample(n=N_BACKGROUND, random_state=42)


def shap_values(model):
    explainer = shap.TreeExplainer(model, data=background,
                                   feature_perturbation="interventional")
    values = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


def compare(shap_a, shap_b):
    jaccards = []
    for i in range(len(X_test)):
        top_a = set(np.argsort(np.abs(shap_a[i]))[-TOP_K:])
        top_b = set(np.argsort(np.abs(shap_b[i]))[-TOP_K:])
        jaccards.append(len(top_a & top_b) / len(top_a | top_b))
    return np.array(jaccards)


print("training reference model")
reference = XGBClassifier(**XGB_PARAMS, random_state=42)
reference.fit(train_real[features], train_real["default"])
shap_reference = shap_values(reference)
risk_reference = reference.predict_proba(X_test)[:, 1]

# the applicants the real-data model would refuse, used again for every seed
k = int(round(len(X_test) * REJECTION_RATE))
rejected_reference = set(np.argsort(-risk_reference)[:k])

rows = []

print("building noise baseline")
for run in range(N_NOISE_RUNS):
    subsample, _ = train_test_split(train_real, train_size=0.8,
                                    stratify=train_real["default"],
                                    random_state=1000 + run)
    noise_model = XGBClassifier(**XGB_PARAMS, random_state=1000 + run)
    noise_model.fit(subsample[features], subsample["default"])
    jaccards = compare(shap_reference, shap_values(noise_model))
    rows.append({"source": "Noise baseline", "run": run + 1,
                 "mean_jaccard": round(jaccards.mean(), 4),
                 "pct_identical": round((jaccards == 1.0).mean() * 100, 2),
                 "pct_different_if_rejected": None})
    print("  run", run + 1, "jaccard", round(jaccards.mean(), 4))

# the ten synthetic datasets already exist, so this only trains and explains
for name in ["CTGAN", "TVAE"]:
    for seed in SEEDS:
        print("comparing", name, "seed", seed)
        synthetic = pd.read_csv(IN_DIR + "/train_" + name.lower() + "_seed" + str(seed) + ".csv")
        model = XGBClassifier(**XGB_PARAMS, random_state=42)
        model.fit(synthetic[features], synthetic["default"])
        jaccards = compare(shap_reference, shap_values(model))

        # of the applicants both models would refuse, how many get different reasons
        risk_synthetic = model.predict_proba(X_test)[:, 1]
        rejected_synthetic = set(np.argsort(-risk_synthetic)[:k])
        both = sorted(rejected_reference & rejected_synthetic)

        rows.append({"source": name, "run": seed,
                     "mean_jaccard": round(jaccards.mean(), 4),
                     "pct_identical": round((jaccards == 1.0).mean() * 100, 2),
                     "pct_different_if_rejected": round((jaccards[both] < 1.0).mean() * 100, 2)})

table = pd.DataFrame(rows)
table.to_csv(IN_DIR + "/agreement_runs.csv", index=False)

summary = table.groupby("source").agg(
    runs=("run", "count"),
    jaccard_mean=("mean_jaccard", "mean"),
    jaccard_sd=("mean_jaccard", "std"),
    jaccard_min=("mean_jaccard", "min"),
    jaccard_max=("mean_jaccard", "max"),
    identical_mean=("pct_identical", "mean"),
    different_if_rejected_mean=("pct_different_if_rejected", "mean"),
).round(4).reset_index()
summary.to_csv(IN_DIR + "/agreement_summary.csv", index=False)

print("\nall runs")
print(table.to_string(index=False))
print("\nsummary across runs")
print(summary.to_string(index=False))
print("\nsaved to", IN_DIR)