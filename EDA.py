import json
import pandas as pd
import matplotlib.pyplot as plt

with open("nvd_cves.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(f"Total records: {len(df)}")
print(f"\n--- Missing Values ---")
print(df.isnull().sum())

print(f"\n--- Severity Distribution ---")
severity_counts = df['severity'].value_counts()
print(severity_counts)
print("\nAs percentages:")
print((severity_counts / len(df) * 100).round(2))

print(f"\n--- Attack Vector Distribution ---")
print(df['attack_vector'].value_counts())

print(f"\n--- Description Length Stats ---")
df['desc_len'] = df['description'].str.split().str.len()
print(df['desc_len'].describe().round(2))

print(f"\n--- CVEs per Year ---")
df['year'] = df['published'].str[:4]
print(df['year'].value_counts().sort_index())