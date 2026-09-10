import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import os

# paths and settings kept at the top so I only change them in one place. 
# Seed fixed so the run is repeatable.

DATA_PATH = "data/default_of_credit_card_clients.xls"
OUT_DIR = "outputs/stage1"
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


# skip the junk first row in the UCI file, real column names are on row 2.

if DATA_PATH.endswith(".xls") or DATA_PATH.endswith(".xlsx"):
    df = pd.read_excel(DATA_PATH, header=1)
else:
    df = pd.read_csv(DATA_PATH)


# shorten the target name and fix the inconsistent PAY_0 label.

df = df.rename(columns={
    "default payment next month": "default",
    "default.payment.next.month": "default",
    "PAY_0": "PAY_1",
})
# ID is a row number, not real information.
# Removing it also keeps it out of the SHAP values, which is what this project measures.
if "ID" in df.columns:
    df = df.drop(columns=["ID"])
# these codes aren't in the codebook. 
# Folded them into "other" rather than deleting the rows.
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("rows:", len(df))
print("default rate:", round(df["default"].mean() * 100, 2), "%")
print("missing values:", df.isna().sum().sum())
print("features:", df.shape[1] - 1)


# 80/20 split. Stratified so both halves keep the real 22% default rate.
# Fixed seed so it's always the same 6,000 people in the test set.

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print("train:", len(X_train), "rows")
print("test:", len(X_test), "rows")


# kept the settings conservative.
# No imbalance handling on purpose — that would change the explanations, which is what I'm measuring.

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=SEED,
)

model.fit(X_train, y_train)


# ROC-AUC to compare against published results on this dataset. 
# PR-AUC because it reflects performance on defaulters, who are the people this project is about.

proba = model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, proba)
pr = average_precision_score(y_test, proba)

print()
print("ROC-AUC:", round(roc, 4))
print("PR-AUC:", round(pr, 4))
print("random PR-AUC would be:", round(y_test.mean(), 4))


# train_real.csv feeds the generators in stage 2. 
# test_real.csv is the fixed test set every later stage loads instead of re-splitting.

X_train.assign(default=y_train).to_csv(OUT_DIR + "/train_real.csv", index=False)
X_test.assign(default=y_test).to_csv(OUT_DIR + "/test_real.csv", index=False)

model.save_model(OUT_DIR + "/model_real.json")

pd.DataFrame([{"roc_auc": roc, "pr_auc": pr}]).to_csv(
    OUT_DIR + "/baseline_metrics.csv", index=False
)

print()
print("saved to", OUT_DIR)