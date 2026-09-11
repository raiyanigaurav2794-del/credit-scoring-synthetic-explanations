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


# load the split from stage 1 rather than splitting again, so the test set can't change.
# Generators only ever see the training half.

train_real = pd.read_csv(IN_DIR + "/train_real.csv")
test_real = pd.read_csv(IN_DIR + "/test_real.csv")

X_test = test_real.drop(columns=["default"])
y_test = test_real["default"]

print("train:", train_real.shape)
print("test:", test_real.shape)


# SDV reads these code columns as numbers by default.
# Overriding them stops the generator producing values like EDUCATION = 2.37.

meta = Metadata.detect_from_dataframe(data=train_real, table_name="credit")

for col in CATEGORICAL:
    meta.update_column(column_name=col, table_name="credit", sdtype="categorical")

meta.validate()


# same model settings for all three so the only thing changing is the training data. 
# Reorder columns to match the test set — XGBoost goes by position, not name.

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


# Retrains on real data so all three rows in the results table come from the same code path.

results = [train_and_score("Real", train_real)]


# two generators built on different principles, so a finding isn't just a quirk of one method. 
# Sample the same number of rows as the real set so training size stays constant.

for name, synth_class in [("CTGAN", CTGANSynthesizer), ("TVAE", TVAESynthesizer)]:
    print()
    path = OUT_DIR + "/train_" + name.lower() + ".csv"

    # the generators aren't seeded, so a fresh run gives different data.
    # If a saved set exists, reuse it so the reported results can be reproduced.
    if os.path.exists(path):
        print("using saved", name, "data from", path)
        synthetic = pd.read_csv(path)
    else:
        print("fitting", name, "for", EPOCHS, "epochs")
        synth = synth_class(meta, epochs=EPOCHS, verbose=True)
        synth.fit(train_real)
        synthetic = synth.sample(num_rows=len(train_real))
        synthetic.to_csv(path, index=False)

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