import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

TOP_K = 3
N_BACKGROUND = 100
SEED = 42
TARGET_RATE = 0.2212

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  eval_metric="logloss", enable_categorical=False)

train_real = pd.read_csv("outputs/stage1/train_real.csv")
test_real = pd.read_csv("outputs/stage1/test_real.csv")
X_test = test_real.drop(columns=["default"])
features = list(X_test.columns)

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


# drop records at random from whichever class is overrepresented until the
# default rate matches the real data. nothing is duplicated.
def undersample(df, rate):
    pos = df[df["default"] == 1]
    neg = df[df["default"] == 0]
    if len(pos) / len(df) > rate:
        keep = int(round(len(neg) * rate / (1 - rate)))
        pos = pos.sample(n=keep, random_state=SEED)
    else:
        keep = int(round(len(pos) * (1 - rate) / rate))
        neg = neg.sample(n=keep, random_state=SEED)
    return pd.concat([pos, neg]).sample(frac=1, random_state=SEED)


print("training reference model")
reference = XGBClassifier(**XGB_PARAMS, random_state=SEED)
reference.fit(train_real[features], train_real["default"])
shap_reference = shap_values(reference)

rows = []
for name in ["CTGAN", "TVAE"]:
    original = pd.read_csv("outputs/stage2/train_" + name.lower() + ".csv")
    balanced = undersample(original, TARGET_RATE)

    for label, data in [("original", original), ("balanced", balanced)]:
        print("comparing", name, label)
        model = XGBClassifier(**XGB_PARAMS, random_state=SEED)
        model.fit(data[features], data["default"])
        jaccards = compare(shap_reference, shap_values(model))

        rows.append({"generator": name,
                     "version": label,
                     "n_train": len(data),
                     "default_rate": round(data["default"].mean(), 4),
                     "mean_jaccard": round(jaccards.mean(), 4),
                     "pct_identical_topk": round((jaccards == 1.0).mean() * 100, 2)})
        print("  ", rows[-1])

table = pd.DataFrame(rows)
table.to_csv("outputs/stage9_balance_common.csv", index=False)

print("\nclass balance control on the common background")
print(table.to_string(index=False))
print("\nsaved to outputs/stage9_balance_common.csv")