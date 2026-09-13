import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
TOP_K = 3
N_BACKGROUND = 100
N_NOISE_RUNS = 5
REJECTION_RATES = [0.10, 0.2212, 0.30]

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  eval_metric="logloss", enable_categorical=False)

train_real = pd.read_csv("outputs/stage1/train_real.csv")
test_real = pd.read_csv("outputs/stage1/test_real.csv")
X_test = test_real.drop(columns=["default"])
features = list(X_test.columns)

# the shared comparison point: the same 100 real applicants for every model
background = train_real[features].sample(n=N_BACKGROUND, random_state=SEED)


def shap_values(model):
    explainer = shap.TreeExplainer(model, data=background,
                                   feature_perturbation="interventional")
    values = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


# returns one jaccard and one tau per applicant, so they can be filtered later
def compare(shap_a, shap_b):
    jaccards = []
    taus = []
    for i in range(len(X_test)):
        top_a = set(np.argsort(np.abs(shap_a[i]))[-TOP_K:])
        top_b = set(np.argsort(np.abs(shap_b[i]))[-TOP_K:])
        jaccards.append(len(top_a & top_b) / len(top_a | top_b))
        tau, _ = kendalltau(np.abs(shap_a[i]), np.abs(shap_b[i]))
        taus.append(0.0 if np.isnan(tau) else tau)
    return np.array(jaccards), np.array(taus)


def summarise(label, jaccards, taus, n_applicants):
    return {"comparison": label,
            "mean_jaccard": round(jaccards.mean(), 4),
            "mean_tau": round(taus.mean(), 4),
            "pct_identical_topk": round((jaccards == 1.0).mean() * 100, 2),
            "n": n_applicants}


print("training reference model")
reference = XGBClassifier(**XGB_PARAMS, random_state=SEED)
reference.fit(train_real[features], train_real["default"])
shap_reference = shap_values(reference)
risk_reference = reference.predict_proba(X_test)[:, 1]

# the noise baseline is rebuilt here so it uses the same background as everything else
print("building noise baseline")
noise_jaccards = []
noise_taus = []
for run in range(N_NOISE_RUNS):
    subsample, _ = train_test_split(train_real, train_size=0.8,
                                    stratify=train_real["default"],
                                    random_state=1000 + run)
    noise_model = XGBClassifier(**XGB_PARAMS, random_state=1000 + run)
    noise_model.fit(subsample[features], subsample["default"])
    j, t = compare(shap_reference, shap_values(noise_model))
    noise_jaccards.append(j)
    noise_taus.append(t)
    print("  run", run + 1, "jaccard", round(j.mean(), 4))

noise_jaccards = np.concatenate(noise_jaccards)
noise_taus = np.concatenate(noise_taus)

explanation_rows = [summarise("Real vs Real (noise baseline)",
                              noise_jaccards, noise_taus, len(noise_jaccards))]
decision_rows = []
rejected_rows = []

for name in ["ctgan", "tvae"]:
    print("comparing real vs", name)
    synthetic = pd.read_csv("outputs/stage2/train_" + name + ".csv")
    model = XGBClassifier(**XGB_PARAMS, random_state=SEED)
    model.fit(synthetic[features], synthetic["default"])
    shap_synthetic = shap_values(model)

    jaccards, taus = compare(shap_reference, shap_synthetic)
    explanation_rows.append(summarise("Real vs " + name.upper(),
                                      jaccards, taus, len(X_test)))

    risk_synthetic = model.predict_proba(X_test)[:, 1]
    for rate in REJECTION_RATES:
        k = int(round(len(X_test) * rate))
        rejected_reference = set(np.argsort(-risk_reference)[:k])
        rejected_synthetic = set(np.argsort(-risk_synthetic)[:k])
        both = rejected_reference & rejected_synthetic
        decision_rows.append({"model": name.upper(),
                              "rejection_rate": str(round(rate * 100, 2)) + "%",
                              "n_rejected": k,
                              "overlap_jaccard": round(len(both) / len(rejected_reference | rejected_synthetic), 4),
                              "pct_also_rejected": round(len(both) / k * 100, 2)})

        # the applicants both models refuse: these are the people who would
        # actually get an adverse action notice either way, so this asks
        # whether the notice would name the same three features
        both = sorted(both)
        rejected_rows.append({"model": name.upper(),
                              "rejection_rate": str(round(rate * 100, 2)) + "%",
                              "n_rejected_by_both": len(both),
                              "mean_jaccard": round(jaccards[both].mean(), 4),
                              "pct_identical_topk": round((jaccards[both] == 1.0).mean() * 100, 2),
                              "pct_different_topk": round((jaccards[both] < 1.0).mean() * 100, 2)})

explanation_table = pd.DataFrame(explanation_rows)
decision_table = pd.DataFrame(decision_rows)
rejected_table = pd.DataFrame(rejected_rows)

print("\nA. explanation agreement, common real-data background")
print(explanation_table.to_string(index=False))
print("\nB. decision agreement: do the models reject the same applicants?")
print(decision_table.to_string(index=False))
print("\nC. applicants rejected by BOTH models: do they get the same reasons?")
print(rejected_table.to_string(index=False))

explanation_table.to_csv("outputs/stage5_common_background.csv", index=False)
decision_table.to_csv("outputs/stage5_decision_agreement.csv", index=False)
rejected_table.to_csv("outputs/stage5_rejected_by_both.csv", index=False)
print("\nsaved to outputs/")