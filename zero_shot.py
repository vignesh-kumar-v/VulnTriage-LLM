import json
import requests
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.1:8b"
N_PER_CLASS = 250
VALID_LABELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

test_df = pd.read_json("test.json")

sampled = pd.concat([
    resample(
        test_df[test_df['severity'] == label],
        n_samples=N_PER_CLASS,
        random_state=42,
        replace=False
    )
    for label in VALID_LABELS
]).reset_index(drop=True)

print(f"Sampled {len(sampled)} records ({N_PER_CLASS} per class)")
print(sampled['severity'].value_counts())

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in vulnerability assessment.
Your task is to predict the CVSS v3.1 severity rating of a software vulnerability based on its description.

You must respond with exactly ONE word from this list: CRITICAL, HIGH, MEDIUM, LOW
Do not explain. Do not add punctuation. Just the single severity label."""

def build_prompt(description: str) -> str:
    return f"""Vulnerability description:
{description}

Severity (CRITICAL/HIGH/MEDIUM/LOW):"""

def query_llama(description: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": SYSTEM_PROMPT + "\n\n" + build_prompt(description),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 10,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        raw = response.json()["response"].strip().upper()

        for label in VALID_LABELS:
            if label in raw:
                return label
        return "UNKNOWN"
    except Exception as e:
        print(f"  Request failed: {e}")
        return "UNKNOWN"

print("\nRunning zero-shot inference...")
predictions = []
unknowns    = 0

for i, row in sampled.iterrows():
    pred = query_llama(row['description'])
    predictions.append(pred)

    if pred == "UNKNOWN":
        unknowns += 1

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(sampled)} complete — {unknowns} unparseable so far")

sampled['predicted'] = predictions

sampled.to_json("zeroshot_results.json", orient="records", indent=2)
print(f"\nResults saved to zeroshot_results.json")

clean = sampled[sampled['predicted'] != 'UNKNOWN']
print(f"\nParseable predictions: {len(clean)}/{len(sampled)}")
print(f"Unparseable (UNKNOWN): {unknowns}")

print("\n--- Classification Report ---")
print(classification_report(
    clean['severity'],
    clean['predicted'],
    labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    digits=4
))

print("\n--- Confusion Matrix ---")
print(pd.DataFrame(
    confusion_matrix(
        clean['severity'],
        clean['predicted'],
        labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ),
    index=["True CRITICAL", "True HIGH", "True MEDIUM", "True LOW"],
    columns=["Pred CRITICAL", "Pred HIGH", "Pred MEDIUM", "Pred LOW"]
))

from sklearn.metrics import f1_score
macro_f1 = f1_score(
    clean['severity'],
    clean['predicted'],
    labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    average='macro'
)
print(f"\nMacro F1 (primary metric): {macro_f1:.4f}")
print("Note: Macro F1 is our headline metric — it weights all classes equally,")
print("penalizing the model for ignoring LOW and CRITICAL minorities.")