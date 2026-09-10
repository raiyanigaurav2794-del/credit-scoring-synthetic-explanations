import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import os

# paths, settings, and the list of columns that are codes rather than numbers.

IN_DIR = "outputs/stage1"
OUT_DIR = "outputs/stage2"
SEED = 42
EPOCHS = 300

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE",
               "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
               "default"]

os.makedirs(OUT_DIR, exist_ok=True)


# ---- load the splits from stage 1 ----

train_real = pd.read_csv(IN_DIR + "/train_real.csv")
test_real = pd.read_csv(IN_DIR + "/test_real.csv")

X_test = test_real.drop(columns=["default"])
y_test = test_real["default"]

print("train:", train_real.shape)
print("test:", test_real.shape)


# ---- tell SDV which columns are categories ----

meta = Metadata.detect_from_dataframe(data=train_real, table_name="credit")

for col in CATEGORICAL:
    meta.update_column(column_name=col, table_name="credit", sdtype="categorical")

meta.validate()


# ---- one function to train and score a model ----

def train_and_score(name, train_df):
    X = train_df.drop(columns=["default"])
    X = X[X_test.columns]
    y = train_df["default"]

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
    model.fit(X, y)
    model.save_model(OUT_DIR + "/model_" + name.lower() + ".json")

    proba = model.predict_proba(X_test)[:, 1]

    return {
        "trained_on": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }


# ---- real baseline first ----

results = [train_and_score("Real", train_real)]


# ---- then each generator ----

for name, synth_class in [("CTGAN", CTGANSynthesizer), ("TVAE", TVAESynthesizer)]:
    print()
    print("fitting", name, "for", EPOCHS, "epochs")

    synth = synth_class(meta, epochs=EPOCHS, verbose=True)
    synth.fit(train_real)

    synthetic = synth.sample(num_rows=len(train_real))
    synthetic.to_csv(OUT_DIR + "/train_" + name.lower() + ".csv", index=False)

    print(name, "rows:", len(synthetic))
    print(name, "default rate:", round(synthetic["default"].mean() * 100, 2), "%")
    print("real default rate:", round(train_real["default"].mean() * 100, 2), "%")

    results.append(train_and_score(name, synthetic))


# ---- results table ----

table = pd.DataFrame(results)
table.to_csv(OUT_DIR + "/tstr_comparison.csv", index=False)

print()
print("TSTR comparison, all scored on the same real test set")
print(table.to_string(index=False))
print()
print("saved to", OUT_DIR)