import nvdlib
import json
import os
import datetime

API_KEY = "1fa9aa52-2d8b-4757-9b6b-45b67fd883b5"
OUTPUT_FILE = "nvd_cves.json"
CHECKPOINT_FILE = "checkpoint.txt"
START_YEAR = 2015
END_YEAR = 2024

def extract_cve(cve):
    try:
        description = cve.descriptions[0].value if cve.descriptions else None
        if not description:
            return None
        if not hasattr(cve, 'v31severity') or cve.v31severity is None:
            return None
        return {
            "cve_id":              cve.id,
            "description":         description,
            "published":           str(cve.published),
            "severity":            cve.v31severity,
            "cvss_score":          cve.v31score,
            "cvss_vector":         cve.v31vector,
            "attack_vector":       _parse_vector_component(cve.v31vector, "AV"),
            "attack_complexity":   _parse_vector_component(cve.v31vector, "AC"),
            "privileges_required": _parse_vector_component(cve.v31vector, "PR"),
            "user_interaction":    _parse_vector_component(cve.v31vector, "UI"),
        }
    except Exception:
        return None

def _parse_vector_component(vector, prefix):
    if not vector:
        return None
    for part in vector.split("/"):
        if part.startswith(prefix + ":"):
            return part.split(":")[1]
    return None

def get_quarters(year):
    return [
        (datetime.datetime(year, 1,  1,  0, 0), datetime.datetime(year, 3,  31, 23, 59)),
        (datetime.datetime(year, 4,  1,  0, 0), datetime.datetime(year, 6,  30, 23, 59)),
        (datetime.datetime(year, 7,  1,  0, 0), datetime.datetime(year, 9,  30, 23, 59)),
        (datetime.datetime(year, 10, 1,  0, 0), datetime.datetime(year, 12, 31, 23, 59)),
    ]

def collect(start_year, end_year):
    all_cves = []

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_cves = json.load(f)
        print(f"Resuming — {len(all_cves)} CVEs already collected.")

    completed_chunks = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            completed_chunks = set(line.strip() for line in f if line.strip())

    for year in range(start_year, end_year + 1):
        year_count = 0
        for i, (start, end) in enumerate(get_quarters(year)):
            chunk_id = f"{year}_Q{i+1}"

            if chunk_id in completed_chunks:
                print(f"Skipping {chunk_id} — already done.")
                continue

            print(f"Fetching {chunk_id}: {start.date()} → {end.date()}")

            try:
                results = nvdlib.searchCVE(
                    pubStartDate=start,
                    pubEndDate=end,
                    key=API_KEY,
                    delay=0.6
                )

                chunk_count = 0
                for cve in results:
                    record = extract_cve(cve)
                    if record:
                        all_cves.append(record)
                        chunk_count += 1
                year_count += chunk_count
                print(f"  → {chunk_count} valid CVEs")
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(all_cves, f, indent=2)

                with open(CHECKPOINT_FILE, "a") as f:
                    f.write(f"{chunk_id}\n")

            except Exception as e:
                print(f"Error on {chunk_id}: {e}")
                print("Progress saved. Re-run to resume.")
                return

        print(f"Year {year} complete — {year_count} CVEs collected this year.")

    print(f"\nDone. Total CVEs: {len(all_cves)}")

if __name__ == "__main__":
    collect(START_YEAR, END_YEAR)