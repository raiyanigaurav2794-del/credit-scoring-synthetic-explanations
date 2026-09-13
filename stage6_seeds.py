import os
import random

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

SEEDS = [101, 202, 303, 404, 505]
EPOCHS = 300
OUT_DIR = "outputs/stage6"

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE",
               "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
               "default"]

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  eval_metric="logloss", enable_categorical=False)

os.makedirs(OUT_DIR, exist_ok=True)

train_real = pd.read_csv("outputs/stage1/train_real.csv")
test_real = pd.read_csv("outputs/stage1/test_real.csv")
X_test = test_real.drop(columns=["default"])
y_test = test_real["default"]
features = list(X_test.columns)

# same metadata overrides as stage 2 so the generators behave the same way
meta = Metadata.detect_from_dataframe(data=train_real, table_name="credit")
for col in CATEGORICAL:
    meta.update_column(column_name=col, table_name="credit", sdtype="categorical")
meta.validate()

SYNTHESIZERS = {"CTGAN": CTGANSynthesizer, "TVAE": TVAESynthesizer}

rows = []

for seed in SEEDS:
    for name, synth_class in SYNTHESIZERS.items():
        path = OUT_DIR + "/train_" + name.lower() + "_seed" + str(seed) + ".csv"

        # each run is saved, so the script can be stopped and restarted
        # without losing the runs that already finished
        if os.path.exists(path):
            print("using saved", name, "seed", seed)
            synthetic = pd.read_csv(path)
        else:
            print("fitting", name, "seed", seed)
            # SDV has no seed argument, so seed the libraries it uses instead
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            synth = synth_class(meta, epochs=EPOCHS, verbose=False)
            synth.fit(train_real)
            synthetic = synth.sample(num_rows=len(train_real))
            synthetic.to_csv(path, index=False)

        model = XGBClassifier(**XGB_PARAMS, random_state=42)
        model.fit(synthetic[features], synthetic["default"])
        proba = model.predict_proba(X_test)[:, 1]

        rows.append({"generator": name,
                     "seed": seed,
                     "default_rate": round(synthetic["default"].mean(), 4),
                     "roc_auc": round(roc_auc_score(y_test, proba), 4),
                     "pr_auc": round(average_precision_score(y_test, proba), 4)})
        print("  ", rows[-1])

table = pd.DataFrame(rows)
table.to_csv(OUT_DIR + "/seed_runs.csv", index=False)

# the spread across seeds is the point: sd, min and max say how much
# a single run could have differed just by chance
summary = table.groupby("generator").agg(
    runs=("seed", "count"),
    default_rate_mean=("default_rate", "mean"),
    roc_auc_mean=("roc_auc", "mean"),
    roc_auc_sd=("roc_auc", "std"),
    roc_auc_min=("roc_auc", "min"),
    roc_auc_max=("roc_auc", "max"),
).round(4).reset_index()
summary.to_csv(OUT_DIR + "/seed_summary.csv", index=False)

print("\nall runs")
print(table.to_string(index=False))
print("\nsummary across seeds")
print(summary.to_string(index=False))
print("\nsaved to", OUT_DIR)