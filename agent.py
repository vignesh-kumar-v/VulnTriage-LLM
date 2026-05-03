import os
import torch
import json
import requests
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime

# ── Config ─────────────────────────────────────────────────
BASE_MODEL_ID  = "meta-llama/Llama-3.1-8B"
ADAPTER_PATH   = "./llama-vuln-qlora/checkpoint-1500"
NVD_API_BASE   = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY    = os.environ.get("NVD_API_KEY", "")
VALID_LABELS   = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

SYSTEM = (
    "You are a cybersecurity expert. Given a CVE vulnerability description, "
    "predict its CVSS v3.1 severity. Reply with exactly one word: "
    "CRITICAL, HIGH, MEDIUM, or LOW."
)

# ── Detect device ──────────────────────────────────────────
if torch.cuda.is_available():
    device     = "cuda"
    device_map = "auto"
elif torch.backends.mps.is_available():
    device     = "mps"
    device_map = None   # MPS doesn't support device_map="auto" with PEFT
else:
    device     = "cpu"
    device_map = None

# ── Load model once at startup ─────────────────────────────
print(f"Loading fine-tuned model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    is_trainable=False,
)
if device_map is None:
    model = model.to(device)
model.eval()
print(f"Model ready on {device}.\n")


# ══════════════════════════════════════════════════════════
# TOOL 1: Severity Predictor
# ══════════════════════════════════════════════════════════
def predict_severity(description: str) -> dict:
    """
    Tool 1: Predict CVSS v3.1 severity from CVE description
    using the fine-tuned Llama model.
    """
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Vulnerability: {description}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().upper()

    severity = "UNKNOWN"
    for label in VALID_LABELS:
        if label in decoded:
            severity = label
            break

    priority_map = {
        "CRITICAL": "Patch within 24 hours",
        "HIGH":     "Patch within 7 days",
        "MEDIUM":   "Patch within 30 days",
        "LOW":      "Patch at next scheduled maintenance",
        "UNKNOWN":  "Manual review required",
    }
    return {
        "predicted_severity": severity,
        "recommended_action": priority_map[severity],
    }


# ══════════════════════════════════════════════════════════
# TOOL 2: NVD CVE Lookup
# ══════════════════════════════════════════════════════════
def lookup_cve(cve_id: str) -> dict:
    """
    Tool 2: Fetch official CVE metadata from NVD API.
    Returns ground truth severity, CVSS score, and vector.
    """
    try:
        headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
        resp = requests.get(
            NVD_API_BASE,
            params={"cveId": cve_id},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {"error": f"CVE {cve_id} not found in NVD"}

        cve = vulns[0]["cve"]

        descriptions = cve.get("descriptions", [])
        description  = next(
            (d["value"] for d in descriptions if d["lang"] == "en"), "N/A"
        )

        metrics   = cve.get("metrics", {})
        cvss_data = metrics.get("cvssMetricV31", [{}])[0].get("cvssData", {})

        return {
            "cve_id":              cve_id,
            "description":         description,
            "published":           cve.get("published", "N/A")[:10],
            "official_severity":   cvss_data.get("baseSeverity", "N/A"),
            "cvss_score":          cvss_data.get("baseScore", "N/A"),
            "cvss_vector":         cvss_data.get("vectorString", "N/A"),
            "attack_vector":       cvss_data.get("attackVector", "N/A"),
            "attack_complexity":   cvss_data.get("attackComplexity", "N/A"),
            "privileges_required": cvss_data.get("privilegesRequired", "N/A"),
            "user_interaction":    cvss_data.get("userInteraction", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
# TOOL 3: Related CVE Finder
# ══════════════════════════════════════════════════════════
def find_related_cves(description: str, severity: str, max_results: int = 3) -> list:
    """
    Tool 3: Search NVD for related CVEs based on keywords
    extracted from the vulnerability description.
    """
    stopwords = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and",
        "or", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "via", "with", "this", "that", "which",
        "when", "where", "how", "allow", "allows", "attacker", "remote",
        "local", "user", "system", "version", "before", "after", "through",
    }
    words    = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
    keywords = [w for w in words if w not in stopwords]
    keyword_query = " ".join(list(dict.fromkeys(keywords))[:3])

    try:
        headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
        resp = requests.get(
            NVD_API_BASE,
            params={
                "keywordSearch":     keyword_query,
                "keywordExactMatch": False,
                "resultsPerPage":    10,
            },
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        related = []
        for vuln in data.get("vulnerabilities", [])[:max_results]:
            cve       = vuln["cve"]
            metrics   = cve.get("metrics", {})
            cvss_data = metrics.get("cvssMetricV31", [{}])[0].get("cvssData", {})
            desc_list = cve.get("descriptions", [])
            desc      = next((d["value"] for d in desc_list if d["lang"] == "en"), "N/A")

            related.append({
                "cve_id":   cve["id"],
                "severity": cvss_data.get("baseSeverity", "N/A"),
                "score":    cvss_data.get("baseScore", "N/A"),
                "summary":  desc[:150] + "..." if len(desc) > 150 else desc,
            })
        return related

    except Exception as e:
        return [{"error": str(e)}]


# ══════════════════════════════════════════════════════════
# TOOL 4: Patch Status Checker
# ══════════════════════════════════════════════════════════
def check_patch_status(cve_id: str) -> dict:
    """
    Tool 4: Check patch and exploit status for a CVE
    using NVD reference tags.
    """
    try:
        headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
        resp = requests.get(
            NVD_API_BASE,
            params={"cveId": cve_id},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {"error": f"{cve_id} not found"}

        cve      = vulns[0]["cve"]
        refs     = cve.get("references", [])
        tags_all = [tag for r in refs for tag in r.get("tags", [])]

        has_patch   = any(t in tags_all for t in ["Patch", "Vendor Advisory", "Mitigation"])
        has_exploit = any(t in tags_all for t in ["Exploit", "Third Party Advisory"])

        patch_links = [
            r["url"] for r in refs
            if any(t in r.get("tags", []) for t in ["Patch", "Vendor Advisory"])
        ][:3]

        return {
            "cve_id":          cve_id,
            "patch_available": has_patch,
            "exploit_known":   has_exploit,
            "patch_links":     patch_links,
            "urgency_note": (
                "EXPLOIT EXISTS — prioritize immediately"      if has_exploit and has_patch
                else "Exploit known, no patch yet — consider mitigations" if has_exploit
                else "Patch available"                         if has_patch
                else "No patch or exploit information found"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
# TOOL 5: Triage Report Generator
# ══════════════════════════════════════════════════════════
def generate_triage_report(
    cve_id: str,
    cve_data: dict,
    prediction: dict,
    patch_status: dict,
    related_cves: list,
) -> str:
    """
    Tool 5: Generate a structured triage report combining
    all tool outputs into an analyst-ready document.
    """
    predicted = prediction["predicted_severity"]
    official  = cve_data.get("official_severity", "N/A")
    match_str = (
        "✓ MATCHES official severity"
        if predicted == official
        else f"✗ DIFFERS from official severity ({official})"
    )

    related_str = ""
    for r in related_cves:
        if "error" not in r:
            related_str += f"  • {r['cve_id']} [{r['severity']} / {r['score']}] — {r['summary']}\n"

    patch_str = ""
    if patch_status.get("patch_available"):
        patch_str += "  Patch Available: YES\n"
        for link in patch_status.get("patch_links", []):
            patch_str += f"    → {link}\n"
    else:
        patch_str += "  Patch Available: NO\n"
    if patch_status.get("exploit_known"):
        patch_str += "  ⚠ Known Exploit: YES\n"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           VULNERABILITY TRIAGE REPORT                       ║
║           Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                    ║
╚══════════════════════════════════════════════════════════════╝

CVE ID:      {cve_id}
Published:   {cve_data.get('published', 'N/A')}

─── DESCRIPTION ───────────────────────────────────────────────
{cve_data.get('description', 'N/A')}

─── SEVERITY ASSESSMENT ───────────────────────────────────────
  Model Prediction:    {predicted}
  Official CVSS:       {official} (Score: {cve_data.get('cvss_score', 'N/A')})
  Assessment:          {match_str}
  Recommended Action:  {prediction['recommended_action']}

─── CVSS VECTOR ───────────────────────────────────────────────
  {cve_data.get('cvss_vector', 'N/A')}
  Attack Vector:       {cve_data.get('attack_vector', 'N/A')}
  Attack Complexity:   {cve_data.get('attack_complexity', 'N/A')}
  Privileges Required: {cve_data.get('privileges_required', 'N/A')}
  User Interaction:    {cve_data.get('user_interaction', 'N/A')}

─── PATCH & EXPLOIT STATUS ────────────────────────────────────
{patch_str}  Urgency Note: {patch_status.get('urgency_note', 'N/A')}

─── RELATED VULNERABILITIES ───────────────────────────────────
{related_str if related_str else '  No related CVEs found.'}
══════════════════════════════════════════════════════════════
"""
    return report


# ══════════════════════════════════════════════════════════
# AGENT ORCHESTRATOR
# ══════════════════════════════════════════════════════════
def run_triage_agent(cve_id: str = None, description: str = None) -> str:
    """
    Main agent entry point. Accepts either:
    - A CVE ID  → fetches description from NVD then runs pipeline
    - A raw description string → predicts severity directly
    Chains: lookup → predict → patch_check → related → report
    """
    print(f"\n{'='*60}")
    print(f"AGENT: Starting triage pipeline...")

    # Step 1: Get CVE data
    if cve_id:
        print(f"TOOL 2: Looking up {cve_id} in NVD...")
        cve_data = lookup_cve(cve_id)
        if "error" in cve_data:
            return f"Error fetching CVE data: {cve_data['error']}"
        description = cve_data["description"]
    else:
        cve_id   = "CUSTOM"
        cve_data = {
            "cve_id":              "CUSTOM",
            "description":         description,
            "published":           datetime.now().strftime("%Y-%m-%d"),
            "official_severity":   "N/A",
            "cvss_score":          "N/A",
            "cvss_vector":         "N/A",
            "attack_vector":       "N/A",
            "attack_complexity":   "N/A",
            "privileges_required": "N/A",
            "user_interaction":    "N/A",
        }

    # Step 2: Predict severity
    print(f"TOOL 1: Predicting severity...")
    prediction = predict_severity(description)
    print(f"  → Predicted: {prediction['predicted_severity']}")

    # Step 3: Check patch status
    if cve_id != "CUSTOM":
        print(f"TOOL 4: Checking patch/exploit status...")
        patch_status = check_patch_status(cve_id)
    else:
        patch_status = {
            "patch_available": False,
            "exploit_known":   False,
            "patch_links":     [],
            "urgency_note":    "N/A — custom description, no CVE ID provided",
        }

    # Step 4: Find related CVEs
    print(f"TOOL 3: Finding related CVEs...")
    related = find_related_cves(description, prediction["predicted_severity"])

    # Step 5: Generate report
    print(f"TOOL 5: Generating triage report...")
    report = generate_triage_report(cve_id, cve_data, prediction, patch_status, related)

    return report


# ══════════════════════════════════════════════════════════
# DEMO — Run agent on 3 real CVEs of different severities
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    test_cves = [
        "CVE-2021-44228",   # Log4Shell  — CRITICAL
        "CVE-2022-22965",   # Spring4Shell — CRITICAL
        "CVE-2023-23397",   # Outlook NTLM — CRITICAL
    ]

    all_reports = []
    for cve_id in test_cves:
        report = run_triage_agent(cve_id=cve_id)
        print(report)
        all_reports.append(report)

    with open("triage_reports.txt", "w") as f:
        f.write("\n\n".join(all_reports))
    print("\nAll reports saved to triage_reports.txt")