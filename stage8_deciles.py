import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TOP_K = 3
N_BACKGROUND = 100
N_NOISE_RUNS = 5
SEED = 42

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  eval_metric="logloss", enable_categorical=False)

train_real = pd.read_csv("outputs/stage1/train_real.csv")
test_real = pd.read_csv("outputs/stage1/test_real.csv")
X_test = test_real.drop(columns=["default"])
features = list(X_test.columns)

# same shared background as stages 5 and 7 so the deciles match those tables
background = train_real[features].sample(n=N_BACKGROUND, random_state=SEED)


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
reference = XGBClassifier(**XGB_PARAMS, random_state=SEED)
reference.fit(train_real[features], train_real["default"])
shap_reference = shap_values(reference)
risk_reference = reference.predict_proba(X_test)[:, 1]

# split the test applicants into ten equal groups by predicted risk
decile = pd.qcut(risk_reference, 10, labels=False)
result = pd.DataFrame({"decile": decile})

print("building noise baseline")
noise = []
for run in range(N_NOISE_RUNS):
    subsample, _ = train_test_split(train_real, train_size=0.8,
                                    stratify=train_real["default"],
                                    random_state=1000 + run)
    noise_model = XGBClassifier(**XGB_PARAMS, random_state=1000 + run)
    noise_model.fit(subsample[features], subsample["default"])
    noise.append(compare(shap_reference, shap_values(noise_model)))
    print("  run", run + 1, "done")
# average the five noise runs applicant by applicant before grouping
result["Real vs Real (noise baseline)"] = np.mean(noise, axis=0)

for name in ["CTGAN", "TVAE"]:
    print("comparing real vs", name)
    synthetic = pd.read_csv("outputs/stage2/train_" + name.lower() + ".csv")
    model = XGBClassifier(**XGB_PARAMS, random_state=SEED)
    model.fit(synthetic[features], synthetic["default"])
    result["Real vs " + name] = compare(shap_reference, shap_values(model))

table = result.groupby("decile").mean().round(4)
table.to_csv("outputs/stage8_deciles.csv")

print("\nmean top-3 jaccard by predicted risk decile")
print(table.to_string())
print("\nsaved to outputs/stage8_deciles.csv")