"""Find Phase 2/3 trial results with positive outcomes in I&I since 2021."""
import os, time, json, glob
from datetime import datetime, date, timezone
from decimal import Decimal
from pathlib import Path
from parallel import Parallel
from openai import OpenAI
from deal_finder.models import Deal
from deal_finder.output import ExcelWriter
from dotenv import load_dotenv

# Config
load_dotenv()
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY_TRIALS", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OUTPUT_DIR = Path("output/trial_results_ph23")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create FindAll run
client = Parallel(api_key=PARALLEL_API_KEY)
print("Creating FindAll run for Phase 2/3 trial results...")

findall_run = client.beta.findall.create(
    objective="Find all Phase 2 or Phase 3 clinical trial results in Immunology & Inflammation with positive or successful outcomes announced since 2021-01-01",
    entity_type="clinical_trials",
    match_conditions=[
        {"name": "ii_check", "description": "Trial must be in Immunology & Inflammation indication (Oncology is ok if I&I related)."},
        {"name": "phase_check", "description": "Trial must be Phase 2 or Phase 3 clinical trial. Do not include preclinical or phase 1."},
        {"name": "outcome_check", "description": "Trial results must be positive, successful, or show good outcomes. Include phrases like 'met primary endpoint', 'statistically significant', 'positive results', 'successful'. Exclude negative or failed trials."},
        {"name": "since_2021_check", "description": "Trial results must have been announced or published since 2021-01-01."}
    ],
    generator="core",
    match_limit=500,
    betas=["findall-2025-09-15"]
)

print(f"✓ Run: {findall_run.findall_id}")

# Poll for completion
print("Waiting for results...")
max_wait = 1800  # 30 min timeout
start = time.time()
while True:
    run = client.beta.findall.retrieve(findall_run.findall_id, betas=["findall-2025-09-15"])
    status = run.status.status

    if status == 'completed':
        print(f"\n✓ Completed! Matched: {run.status.metrics.matched_candidates_count}")
        break
    elif status == 'failed':
        print(f"\n✗ Run failed: {run.status}")
        exit(1)

    elapsed = int(time.time() - start)
    metrics = getattr(run.status, 'metrics', None)
    gen = metrics.generated_candidates_count if metrics else 0
    print(f"  [{elapsed}s] Status: {status} | Generated: {gen}", end='\r')

    if elapsed > max_wait:
        print(f"\n✗ Timeout after {max_wait}s")
        exit(1)

    time.sleep(10)

# Retrieve results
result = client.beta.findall.result(findall_run.findall_id, betas=["findall-2025-09-15"])
matched = [c for c in result.model_dump().get('candidates', []) if c.get('match_status') == 'matched']
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save raw (checkpoint 1)
with open(OUTPUT_DIR / f"raw_{timestamp}.json", 'w') as f:
    json.dump({'run_id': findall_run.findall_id, 'matched': len(matched), 'candidates': matched}, f, indent=2, default=str)
print(f"✓ Checkpoint: Saved raw to {OUTPUT_DIR / f'raw_{timestamp}.json'}")

# Parse with OpenAI - check for existing checkpoint
openai_client = OpenAI(api_key=OPENAI_API_KEY)
existing = sorted(glob.glob(str(OUTPUT_DIR / "openai_responses_*.json")), reverse=True)
if existing:
    responses_file = Path(existing[0])
    with open(responses_file, 'r') as f:
        openai_responses = json.load(f)
    print(f"✓ Checkpoint: Loaded {len(openai_responses)} cached responses from {responses_file.name}")
else:
    responses_file = OUTPUT_DIR / f"openai_responses_{timestamp}.json"
    openai_responses = {}

def parse_with_openai(candidate):
    prompt = f"""Extract fields from clinical trial results data. Return JSON with null for missing data.
Fields: announcement_date (YYYY-MM-DD), company_name, asset_name, trial_phase (phase 2/phase 3), indication, mechanism_of_action, outcome_description (brief summary of results), primary_endpoint_met (true/false), patient_count (number)
Data: {json.dumps(candidate, indent=2)}"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Parse error: {e}")
        return {}

def parse_date(s):
    if not s: return date.today()
    try: return datetime.fromisoformat(str(s)).date()
    except: return date.today()

def safe_decimal(v):
    try: return Decimal(str(v)) if v else None
    except: return None

# Parse all (using Deal model as generic container - adapt fields as needed)
trials = []
print(f"Parsing {len(matched)} trials...")
for i, c in enumerate(matched):
    try:
        cid = c.get('candidate_id', f'candidate_{i}')
        print(f"[{i+1}/{len(matched)}] {c.get('name', 'Unknown')[:50]}...", end='\r')

        # Check checkpoint (checkpoint 2)
        if cid in openai_responses:
            p = openai_responses[cid]
        else:
            p = parse_with_openai(c)
            openai_responses[cid] = p
            with open(responses_file, 'w') as f:
                json.dump(openai_responses, f, indent=2)
        trials.append(Deal(
            date_announced=parse_date(p.get('announcement_date')),
            target=(p.get('company_name') or c.get('name', 'Unknown'))[:200],
            acquirer='N/A',  # Not applicable for trials
            stage=(p.get('trial_phase') or 'unknown')[:50],
            therapeutic_area=(p.get('indication') or 'Immunology & Inflammation')[:100],
            asset_focus=(p.get('asset_name') or c.get('name', 'Unknown'))[:200],
            deal_type_detailed=None,  # Not applicable
            source_url=c.get('url', 'https://example.com'),
            upfront_value_usd=safe_decimal(p.get('patient_count')),  # Repurpose for patient count
            contingent_payment_usd=None,
            total_deal_value_usd=None,
            geography=None,
            key_evidence=(p.get('outcome_description') or c.get('description', ''))[:500],
            confidence=Decimal("0.5"),
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        ))
    except Exception as e:
        print(f"\nFailed: {c.get('name')} - {e}")

print(f"\n✓ Parsed {len(trials)} trials")

# Export
if trials:
    ExcelWriter().write(trials, str(OUTPUT_DIR / f"trial_results_ph23_{timestamp}.xlsx"))
    print(f"✓ Saved: {OUTPUT_DIR / f'trial_results_ph23_{timestamp}.xlsx'}")
    for i, t in enumerate(trials[:3], 1):
        print(f"{i}. {t.target} | {t.asset_focus} | {t.stage} | {t.date_announced}")
        print(f"   {t.key_evidence[:100]}...")
else:
    print("✗ No trials found")
