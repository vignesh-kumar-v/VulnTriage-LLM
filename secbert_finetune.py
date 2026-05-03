import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
import os

# ── Config ─────────────────────────────────────────────────
MODEL_ID   = "jackaduma/SecBERT"
OUTPUT_DIR = "./secbert-vuln"
MAX_LENGTH = 256
BATCH_SIZE = 32       # BERT is small — 32 fits easily on 8GB
EPOCHS     = 5        # BERT needs more epochs than LLMs
LR         = 2e-5     # standard BERT learning rate
SEED       = 42

# ── Label mapping ──────────────────────────────────────────
LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── 1. Load data ───────────────────────────────────────────
print("Loading data...")
train_df = pd.read_json("train.json")
val_df   = pd.read_json("val.json")
test_df  = pd.read_json("test.json")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 2. Add integer labels ──────────────────────────────────
train_df['label'] = train_df['severity'].map(LABEL2ID)
val_df['label']   = val_df['severity'].map(LABEL2ID)
test_df['label']  = test_df['severity'].map(LABEL2ID)

# ── 3. Convert to HuggingFace datasets ────────────────────
train_data = Dataset.from_dict({
    'text':  train_df['description'].tolist(),
    'label': train_df['label'].tolist(),
})
val_data = Dataset.from_dict({
    'text':  val_df['description'].tolist(),
    'label': val_df['label'].tolist(),
})

# ── 4. Tokenizer ───────────────────────────────────────────
print("Loading SecBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,    # DataCollator handles padding dynamically
    )

print("Tokenizing...")
train_data = train_data.map(tokenize, batched=True, remove_columns=['text'])
val_data   = val_data.map(tokenize,   batched=True, remove_columns=['text'])
print("Tokenization complete.")

# ── 5. Class weights for imbalanced dataset ───────────────
# Penalizes the model for ignoring minority classes (LOW, CRITICAL)
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = torch.tensor(
    1.0 / class_counts * class_counts.sum() / len(LABEL2ID),
    dtype=torch.float32
).cuda()
print(f"\nClass weights: {class_weights}")

# ── 6. Model ───────────────────────────────────────────────
print("\nLoading SecBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
print("Model loaded.")

# ── 7. Custom Trainer with weighted loss ───────────────────
# Standard Trainer uses uniform cross-entropy — we override
# compute_loss to apply class weights, helping LOW and CRITICAL
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── 8. Metrics ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {"macro_f1": macro_f1}

# ── 9. Training arguments ──────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    bf16=True,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
    seed=SEED,
)

# ── 10. Train ──────────────────────────────────────────────
print("\nStarting SecBERT fine-tuning...")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ── 11. Save ───────────────────────────────────────────────
print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ── 12. Evaluate on test set ───────────────────────────────
print("\nEvaluating on test set...")
VALID_LABELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

# Same stratified 1000-sample test set as previous models
sampled = pd.concat([
    resample(
        test_df[test_df['severity'] == label],
        n_samples=250,
        random_state=42,
        replace=False
    )
    for label in VALID_LABELS
]).reset_index(drop=True)

# Tokenize test samples
model.eval()
model.cuda()
predictions = []

for i, row in sampled.iterrows():
    inputs = tokenizer(
        row['description'],
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
        predictions.append(ID2LABEL[pred_id])

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(sampled)}")

sampled['predicted'] = predictions
sampled.to_json("secbert_results.json", orient="records", indent=2)

print("\n--- Classification Report ---")
print(classification_report(
    sampled['severity'],
    sampled['predicted'],
    labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    digits=4
))

macro_f1 = f1_score(
    sampled['severity'],
    sampled['predicted'],
    labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    average='macro'
)
print(f"Macro F1: {macro_f1:.4f}")
print("Results saved to secbert_results.json")