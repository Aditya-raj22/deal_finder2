"""Find late-stage (Phase 2/3) I&I biotech deals since 2021."""
import os, time, json, glob
from datetime import datetime, date, timezone
from decimal import Decimal
from pathlib import Path
from parallel import Parallel
from openai import OpenAI
from deal_finder.models import Deal, DealTypeDetailed
from deal_finder.output import ExcelWriter

# Config
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY_LATE_STAGE", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OUTPUT_DIR = Path("output/late_stage_deals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create FindAll run
client = Parallel(api_key=PARALLEL_API_KEY)
print("Creating FindAll run for late-stage deals...")

findall_run = client.beta.findall.create(
    objective="Find all late-stage (lead asset is in phase 2 or phase 3) Immunology & Inflammation biotech deals (M&A, partnerships, licensing, option-to-license) that have happened since 2021-01-01",
    entity_type="deals",
    match_conditions=[
        {"name": "ii_check", "description": "Deal must be in Immunology & Inflammation (Oncology is ok if I&I related)."},
        {"name": "late_stage_check", "description": "The lead asset at the time of the deal must be in phase 2 or phase 3 clinical trials. Do not include preclinical or phase 1 assets."},
        {"name": "biotech_check", "description": "Deal must be a biotech deal (M&A, partnerships, licensing, option-to-license)."},
        {"name": "since_2021_check", "description": "Deal must have happened since 2021-01-01."}
    ],
    generator="core",
    match_limit=500,
    betas=["findall-2025-09-15"]
)

print(f"✓ Run: {findall_run.findall_id}")

# Poll for completion
print("Waiting for results...")
while True:
    run = client.beta.findall.retrieve(findall_run.findall_id, betas=["findall-2025-09-15"])
    if run.status.status == 'completed':
        print(f"✓ Completed! Matched: {run.status.metrics.matched_candidates_count}")
        break
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
    prompt = f"""Extract fields from biotech deal data. Return JSON with null for missing data.
Fields: date_announced (YYYY-MM-DD), target_company, acquirer_company, development_stage, asset_name, mechanism_of_action, deal_type (M&A/licensing/option-to-license/partnership), upfront_value_usd (millions), milestone_value_usd (millions)
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

def parse_deal_type(s):
    if not s: return DealTypeDetailed.PARTNERSHIP
    s = s.lower()
    if 'm&a' in s or 'acquisition' in s: return DealTypeDetailed.MA
    if 'option' in s and 'licens' in s: return DealTypeDetailed.OPTION_TO_LICENSE
    if 'licens' in s: return DealTypeDetailed.LICENSING
    return DealTypeDetailed.PARTNERSHIP

def parse_date(s):
    if not s: return date.today()
    try: return datetime.fromisoformat(str(s)).date()
    except: return date.today()

def safe_decimal(v):
    try: return Decimal(str(v)) if v else None
    except: return None

# Parse all
deals = []
print(f"Parsing {len(matched)} deals...")
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
        deals.append(Deal(
            date_announced=parse_date(p.get('date_announced')),
            target=(p.get('target_company') or c.get('name', 'Unknown'))[:200],
            acquirer=(p.get('acquirer_company') or 'Partner')[:200],
            stage=(p.get('development_stage') or 'unknown')[:50],
            therapeutic_area='Immunology & Inflammation',
            asset_focus=(p.get('asset_name') or c.get('name', 'Unknown'))[:200],
            deal_type_detailed=parse_deal_type(p.get('deal_type')),
            source_url=c.get('url', 'https://example.com'),
            upfront_value_usd=safe_decimal(p.get('upfront_value_usd')),
            contingent_payment_usd=safe_decimal(p.get('milestone_value_usd')),
            total_deal_value_usd=safe_decimal(p.get('total_deal_value_usd')),
            geography=None,
            key_evidence=c.get('description', '')[:500],
            confidence=Decimal("0.5"),
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        ))
    except Exception as e:
        print(f"\nFailed: {c.get('name')} - {e}")

print(f"\n✓ Parsed {len(deals)} deals")

# Export
if deals:
    ExcelWriter().write(deals, str(OUTPUT_DIR / f"late_stage_deals_{timestamp}.xlsx"))
    print(f"✓ Saved: {OUTPUT_DIR / f'late_stage_deals_{timestamp}.xlsx'}")
    for i, d in enumerate(deals[:3], 1):
        print(f"{i}. {d.target} / {d.acquirer} | {d.stage} | {d.date_announced}")
else:
    print("✗ No deals found")
