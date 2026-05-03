# VulnTriage-LLM

```
 __     __    _    _____    _                ____  __  __
 \ \   / /   | |  / |   |  | |              |  _ \|  \/  |
  \ \ / /   _| | / /| |  _| |_ _ __ _  __ _ | | | | \  / |
   \ V / | | | || | | | |_   _| '__| |/ _` || | | | |\/| |
    \ /| |_| | || | | |   | | | |  | | (_| || |_| | |  | |
     \/  \__,_|_||_| |_|   |_| |_|  |_|\__,_||____/|_|  |_|
        Automated CVE Triage with Large Language Models
```

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.4253-brightgreen.svg)](#results)

> An automated vulnerability triage system that predicts CVSS v3.1 severity ratings (`CRITICAL` / `HIGH` / `MEDIUM` / `LOW`) directly from CVE descriptions, comparing zero-shot LLM inference, fine-tuned LLMs, and a fine-tuned domain-specific encoder. Includes a multi-tool agentic pipeline that produces analyst-ready triage reports.

---

## Table of Contents

- [Motivation](#motivation)
- [Highlights](#highlights)
- [Dataset](#dataset)
- [Models and Approaches](#models-and-approaches)
- [Results](#results)
- [Key Findings](#key-findings)
- [Agentic Triage Pipeline](#agentic-triage-pipeline)
- [Repository Structure](#repository-structure)
- [Hardware](#hardware)
- [Requirements](#requirements)
- [Setup and Usage](#setup-and-usage)
- [Notes and Caveats](#notes-and-caveats)
- [License](#license)
- [Author](#author)

---

## Motivation

Security teams routinely receive **hundreds of new CVEs per week** and must decide which to patch first. Manual triage is slow and inconsistent, while CVSS scoring itself can lag behind disclosure. **VulnTriage-LLM** addresses this operational bottleneck by learning to predict severity directly from the natural-language description of a vulnerability — the same input a human analyst would read first.

The project asks a concrete research question:

> **Given only a CVE description, can a language model match the severity rating that NIST eventually publishes — and how does a small domain-specific encoder compare to a large general-purpose LLM?**

---

## Highlights

- **144,851 CVEs** collected directly from the NIST NVD public API (2015–2024)
- **Three modeling approaches benchmarked end-to-end:** zero-shot Llama 3.1 8B, QLoRA-fine-tuned Llama 3.1 8B, and a weighted-loss-fine-tuned SecBERT
- **+83% Macro F1** improvement from zero-shot baseline → fine-tuned models
- **SecBERT (110M) matches Llama (8B)** on overall Macro F1 — a 75× parameter reduction with no performance loss
- **Temporal train/val/test split** (2015–2021 / 2022 / 2023–2024) — no data leakage, simulates real deployment
- **Agentic triage pipeline** with 5 tools producing structured analyst-ready reports

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [NIST National Vulnerability Database (NVD) API](https://nvd.nist.gov/developers) |
| **Total records** | 144,851 CVEs with CVSS v3.1 labels |
| **Date range** | January 2015 – December 2024 |
| **Train split** | 2015–2021 (52,052 records) |
| **Validation split** | 2022 (24,976 records) |
| **Test split** | 2023–2024 (67,823 records) |

### Fields per record

```
cve_id, description, published, severity, cvss_score, cvss_vector,
attack_vector, attack_complexity, privileges_required, user_interaction
```

### Class distribution

| Severity | Share | Patch SLA (recommended) |
|---|---|---|
| MEDIUM    | 45.9% | 30 days |
| HIGH      | 37.8% | 7 days |
| CRITICAL  | 12.6% | 24 hours |
| LOW       |  3.6% | Next maintenance window |

> **Why a temporal split?** A random shuffle would let the model see CVEs from the same disclosure week in both train and test, leaking style and vocabulary cues. Splitting strictly by publication year mirrors how the system would actually be deployed: train on the past, predict the future.

---

## Models and Approaches

### 1. Zero-Shot Baseline — Llama 3.1 8B

A pure prompt-engineering baseline using `meta-llama/Llama-3.1-8B`. The model is asked to reply with a single severity label. Evaluated on a stratified 1,000-sample subset of the test set (250 per class) for tractability.

- **Macro F1: 0.2289**

### 2. Fine-Tuned Llama 3.1 8B (QLoRA)

Parameter-efficient fine-tuning using **QLoRA** (4-bit NF4 base + LoRA adapters).

| Hyperparameter | Value |
|---|---|
| Quantization      | 4-bit (bitsandbytes NF4) |
| LoRA rank         | 8 |
| Epochs            | 3 |
| Effective batch   | 32 (per-device 8 × grad-accum 4) |
| Learning rate     | 2e-4 (cosine schedule) |
| Best checkpoint   | step 1500 (early-stopped — overfitting after step 2000) |
| Hardware          | Google Colab Pro · NVIDIA A100 40 GB |

- **Macro F1: 0.4184**

### 3. Fine-Tuned SecBERT

`jackaduma/SecBERT` is a BERT model pretrained on cybersecurity corpora. Fine-tuned with **class-weighted cross-entropy loss** to combat the heavy imbalance toward MEDIUM/HIGH.

| Hyperparameter | Value |
|---|---|
| Epochs            | 5 |
| Batch size        | 32 |
| Learning rate     | 2e-5 |
| Loss              | Weighted cross-entropy (inverse class frequency) |
| Hardware          | NVIDIA RTX 5060 8 GB · Fedora Linux |

- **Macro F1: 0.4253**

---

## Results

| Model | Macro F1 | CRITICAL F1 | HIGH F1 | MEDIUM F1 | LOW F1 |
|---|---|---|---|---|---|
| Llama 3.1 8B — Zero-Shot   | 0.2289 | 0.2249 | 0.3672 | 0.3156 | 0.0080 |
| Llama 3.1 8B — Fine-Tuned  | 0.4184 | **0.6157** | **0.5061** | **0.4906** | 0.0613 |
| SecBERT — Fine-Tuned       | **0.4253** | 0.5880 | 0.4635 | 0.4884 | **0.1614** |

> **Macro F1 is the headline metric, not accuracy.** Macro F1 weights every class equally, so a model that ignores LOW (3.6% of data) and CRITICAL (12.6%) cannot hide behind a strong MEDIUM/HIGH score. With a class-imbalanced label space, accuracy is misleading.

---

## Key Findings

- **Fine-tuning is decisive.** Macro F1 jumps from `0.2289` → `0.4184` for Llama after fine-tuning — an **83% relative improvement** with only LoRA-adapter weights trained.
- **Small models can match large ones.** SecBERT (110M params) edges out fine-tuned Llama (8B params) on Macro F1 despite being **~75× smaller**. The cybersecurity-domain pretraining of SecBERT closes the parameter gap.
- **Each model has a class it owns.** Fine-tuned Llama is strongest on `CRITICAL` (`0.6157` vs `0.5880`), while SecBERT is meaningfully better on `LOW` (`0.1614` vs `0.0613`) thanks to weighted loss.
- **`LOW` is the universally hardest class.** Only 3.6% of training data and linguistically very close to `MEDIUM` — the ceiling is set by the data, not the model.
- **Description quality is the upper bound.** Vague NVD descriptions (e.g. *CVE-2023-23397: "Microsoft Outlook Elevation of Privilege Vulnerability"*) cause mispredictions across every model. No amount of capacity fixes a missing input signal.

---

## Agentic Triage Pipeline

`agent.py` wraps the fine-tuned Llama model in a **5-tool pipeline** that produces a structured triage report from either a CVE ID or a raw description.

| # | Tool | Purpose |
|---|---|---|
| 1 | **Severity Predictor**       | Fine-tuned Llama inference on the description |
| 2 | **NVD CVE Lookup**           | Fetches official metadata, CVSS vector, publication date |
| 3 | **Related CVE Finder**       | Keyword search for similar vulnerabilities |
| 4 | **Patch Status Checker**     | Detects patch availability and known exploits via NVD reference tags |
| 5 | **Triage Report Generator**  | Synthesizes everything into an analyst-ready document |

**Sample output** (see [`triage_reports.txt`](triage_reports.txt) for full reports):

```
╔══════════════════════════════════════════════════════════════╗
║           VULNERABILITY TRIAGE REPORT                       ║
╚══════════════════════════════════════════════════════════════╝

CVE ID:      CVE-2021-44228
─── SEVERITY ASSESSMENT ───────────────────────────────────────
  Model Prediction:    CRITICAL
  Official CVSS:       CRITICAL (Score: 10.0)
  Assessment:          ✓ MATCHES official severity
  Recommended Action:  Patch within 24 hours
─── PATCH & EXPLOIT STATUS ────────────────────────────────────
  Patch Available: YES
  ⚠ Known Exploit: YES
  Urgency Note: EXPLOIT EXISTS — prioritize immediately
```

---

## Repository Structure

```
VulnTriage-LLM/
├── data.py                         # NVD API data collection (quarterly chunking + checkpointing)
├── EDA.py                          # Exploratory analysis and class distribution
├── preprocess.py                   # Temporal train/val/test splitting
├── zero_shot.py                    # Zero-shot Llama 3.1 8B baseline evaluation
├── secbert_finetune.py             # SecBERT fine-tuning with weighted loss
├── VulnTriage_FineTuning.ipynb     # Llama 3.1 8B QLoRA fine-tuning (Colab)
├── agent.py                        # 5-tool agentic triage pipeline
│
├── zeroshot_results.json           # Zero-shot predictions on test set
├── finetuned_results.json          # Fine-tuned Llama predictions on test set
├── secbert_results.json            # SecBERT predictions on test set
├── triage_reports.txt              # Sample agent output for 3 real-world CVEs
│
├── LICENSE
└── README.md
```

---

## Hardware

| Stage | Hardware |
|---|---|
| Llama QLoRA fine-tuning  | Google Colab Pro · NVIDIA A100 40 GB |
| SecBERT fine-tuning      | NVIDIA RTX 5060 8 GB · Fedora Linux |
| Inference and agent      | Apple M5 Pro · 24 GB unified memory (Metal / MPS) |

---

## Requirements

- **Python 3.11+**
- `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`
- `datasets`, `scikit-learn`, `pandas`, `numpy`
- `nvdlib`, `requests`, `torch`

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/vignesh-kumar-v/VulnTriage-LLM.git
cd VulnTriage-LLM
```

### 2. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\activate         # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install torch transformers peft trl bitsandbytes accelerate \
            datasets scikit-learn pandas numpy nvdlib requests
```

### 4. Get an NVD API key

Request a free key from [https://nvd.nist.gov/developers/request-an-api-key](https://nvd.nist.gov/developers/request-an-api-key). It arrives by email within minutes.

Export it as an environment variable (used by both `data.py` and `agent.py`):

```bash
export NVD_API_KEY="your-key-here"
```

### 5. Collect the dataset

This pulls all CVSS v3.1-labeled CVEs from 2015–2024 in quarterly chunks with automatic resume support. Expect ~1–2 hours on first run.

```bash
python data.py
```

Output: `nvd_cves.json` (~98 MB).

### 6. Preprocess into temporal splits

```bash
python preprocess.py
```

Output: `train.json`, `val.json`, `test.json`.

### 7. Run the zero-shot baseline

```bash
python zero_shot.py
```

Output: `zeroshot_results.json`.

### 8. Fine-tune Llama 3.1 8B (QLoRA)

Open [`VulnTriage_FineTuning.ipynb`](VulnTriage_FineTuning.ipynb) in **Google Colab** with an A100 runtime and execute cells top-to-bottom. The trained adapter is saved to `llama-vuln-qlora/`. Download it locally to `./llama-vuln-qlora/` for use with `agent.py`.

### 9. Fine-tune SecBERT

```bash
python secbert_finetune.py
```

Output: `secbert-vuln/` checkpoints and `secbert_results.json`.

### 10. Run the agentic triage pipeline

```bash
python agent.py
```

By default it triages `CVE-2021-44228`, `CVE-2022-22965`, and `CVE-2023-23397` and writes the structured reports to `triage_reports.txt`.

---

## Notes and Caveats

- **Dataset JSON files are not committed** (they exceed reasonable repo size). Regenerate them via `data.py` with your NVD API key.
- **Model weights are not committed.** The fine-tuned Llama adapter and SecBERT checkpoints must be reproduced via the provided scripts/notebook.
- **Macro F1, not accuracy**, is the primary metric throughout — class imbalance makes accuracy uninformative.
- **Temporal split, not random split.** The model is evaluated on CVEs published *after* its training cutoff to simulate real-world deployment.
- **The NVD API has rate limits.** With a key you get 50 requests / 30 seconds; without one you get 5 / 30 seconds. `data.py` uses the documented `delay=0.6` to stay under the limit.

---

## License

Released under the [MIT License](LICENSE).

---

## Author

**Vigneshkumar V.** · [@vignesh-kumar-v](https://github.com/vignesh-kumar-v)
