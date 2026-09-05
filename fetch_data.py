from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

print("downloading UCI dataset 350...")
ds = fetch_ucirepo(id=350)

X = ds.data.features
y = ds.data.targets
target_col = y.columns[0]
print(f"target column came back as: '{target_col}'")

df = pd.concat([X, y], axis=1)
df = df.rename(columns={target_col: "default payment next month"})

Path("data").mkdir(exist_ok=True)
out = Path("data/default_of_credit_card_clients.csv")
df.to_csv(out, index=False)

print(f"saved -> {out}")
print(f"shape        : {df.shape}")
print(f"default rate : {df['default payment next month'].mean():.4%}  (expect ~22.12%)")
