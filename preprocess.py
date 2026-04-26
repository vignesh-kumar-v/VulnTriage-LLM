import json
import pandas as pd
from sklearn.utils import shuffle

with open("nvd_cves.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df = df[df['severity'] != 'NONE'].reset_index(drop=True)
print(f"After dropping NONE: {len(df)} records")

df['year'] = df['published'].str[:4].astype(int)

train_df = df[df['year'] <= 2021].reset_index(drop=True)
val_df   = df[df['year'] == 2022].reset_index(drop=True)
test_df  = df[df['year'] >= 2023].reset_index(drop=True)

train_df = shuffle(train_df, random_state=42).reset_index(drop=True)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Total accounted for: {len(train_df) + len(val_df) + len(test_df)}")

print(f"\n--- Train severity distribution ---")
print((train_df['severity'].value_counts(normalize=True) * 100).round(2))

print(f"\n--- Val severity distribution ---")
print((val_df['severity'].value_counts(normalize=True) * 100).round(2))

print(f"\n--- Test severity distribution ---")
print((test_df['severity'].value_counts(normalize=True) * 100).round(2))

train_df.to_json("train.json", orient="records", indent=2)
val_df.to_json("val.json",   orient="records", indent=2)
test_df.to_json("test.json", orient="records", indent=2)

print("\nSplits saved: train.json, val.json, test.json")