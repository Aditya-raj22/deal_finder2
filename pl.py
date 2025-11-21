"""
Run Parallel FindAll API to find early-stage I&I biotech deals.
Clean version - just put your API key below and run.
"""
import os
import time
import json
from datetime import datetime, date, timezone
from decimal import Decimal
from pathlib import Path
from parallel import Parallel
from openai import OpenAI

from deal_finder.models import Deal, DealTypeDetailed
from deal_finder.output import ExcelWriter

# =============================================================================
# CONFIGURATION - PUT YOUR API KEY HERE
# =============================================================================
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")  # Set via environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set via environment variable
FINDALL_BETA = "findall-2025-09-15"
GENERATOR = "core"  # Options: "base" ($0.25 + $0.03/match), "core" ($2 + $0.15/match), "pro" ($10 + $1/match)

# =============================================================================
# CREATE FINDALL RUN
# =============================================================================
client = Parallel(api_key=PARALLEL_API_KEY)

print("Creating FindAll run...")

findall_run = client.beta.findall.create(
    objective="Find all early-stage (lead asset is preclinical or phase 1 but hasn't started phase 2) Immunology & Inflammation (Oncology is ok if I&I related) biotech deals (M&A, partnerships, licensing, option-to-license) that have happened since 2021-01-01",
    entity_type="deals",

    # Match conditions - what qualifies as a match
    match_conditions=[
        {
            "name": "immunology_inflammation_check",
            "description": "Deal must be in Immunology & Inflammation (Oncology is ok if I&I related)."
        },
        {
            "name": "early_stage_check",
            "description": "The lead asset at the time of the deal announcement must be in preclinical development OR phase 1 clinical trials. The asset must NOT have started phase 2 trials yet. Do not include deals where the asset is described as 'clinical-stage' without specific phase information - only include if explicitly stated as preclinical or phase 1."
        },
        {
            "name": "biotech_check",
            "description": "Deal must be a biotech deal (M&A, partnerships, licensing, option-to-license)."
        },
        {
            "name": "since_2021_01_01_check",
            "description": "Deal must have happened since 2021-01-01."
        },
        {
            "name": "deal_value_check",
            "description": "Deal must mention financial terms (upfront value, milestones, total deal value) OR be described as 'undisclosed', 'not disclosed', or 'terms not disclosed'. This should match nearly all deals."
        }
    ],

    generator=GENERATOR,
    match_limit=500,
    betas=[FINDALL_BETA]
)

print(f"✓ Created FindAll run: {findall_run.findall_id}")
print(f"  Generator: {GENERATOR}")

# =============================================================================
# POLL FOR COMPLETION
# =============================================================================
print("\nWaiting for results (this may take several minutes)...")

while True:
    run_status = client.beta.findall.retrieve(findall_run.findall_id, betas=[FINDALL_BETA])

    if run_status.status.status == 'completed':
        print(f"\n✓ Run completed!")
        print(f"  Generated candidates: {run_status.status.metrics.generated_candidates_count}")
        print(f"  Matched candidates: {run_status.status.metrics.matched_candidates_count}")
        break

    print(f"  Status: {run_status.status.status}...", end='\r')
    time.sleep(10)

# =============================================================================
# RETRIEVE AND PARSE RESULTS
# =============================================================================
print("\nRetrieving results...")
result = client.beta.findall.result(findall_run.findall_id, betas=[FINDALL_BETA])
result_dict = result.model_dump()

candidates = result_dict.get('candidates', [])
matched = [c for c in candidates if c.get('match_status') == 'matched']

print(f"✓ Retrieved {len(matched)} matched deals\n")

# Save raw Parallel output
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
raw_output_file = Path("output") / f"parallel_raw_{timestamp}.json"
raw_output_file.parent.mkdir(exist_ok=True)

with open(raw_output_file, 'w') as f:
    json.dump({
        'run_id': findall_run.findall_id,
        'timestamp': timestamp,
        'total_candidates': len(candidates),
        'matched_candidates': len(matched),
        'candidates': result_dict.get('candidates', [])
    }, f, indent=2, default=str)

print(f"✓ Saved raw Parallel output to: {raw_output_file}\n")

# =============================================================================
# OPENAI PARSING
# =============================================================================
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai_client = OpenAI()

def parse_candidate_with_openai(candidate: dict) -> dict:
    """Use OpenAI to parse the candidate into structured fields."""

    prompt = f"""You are parsing biotech deal data. Extract the following fields from the data below.
If a field is not mentioned or cannot be determined, return null for that field.

Required fields to extract:
- date_announced: Date in YYYY-MM-DD format
- target_company: Name of target/partner company being acquired/licensed
- acquirer_company: Name of acquiring/partnering company
- development_stage: Exact stage (preclinical, phase 1, phase 1b, etc.) - do NOT use generic terms
- asset_name: Drug/asset/technology name
- mechanism_of_action: Exact name of gene of pathway target 
- deal_type: One of: M&A, licensing, option-to-license, partnership
- upfront_value_usd: Upfront payment in millions USD (number only, or null)
- milestone_value_usd: Total milestone/contingent payments in millions USD (number only, or null)

Deal data:
{json.dumps(candidate, indent=2)}

Return ONLY a JSON object with the extracted fields. Use null for missing data."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"OpenAI parsing error: {e}")
        return {}

# Helper functions
def parse_deal_type(deal_type_str: str) -> DealTypeDetailed:
    """Parse deal type string to enum."""
    if not deal_type_str:
        return DealTypeDetailed.PARTNERSHIP
    lower = deal_type_str.lower()
    if 'm&a' in lower or 'acquisition' in lower or 'merger' in lower:
        return DealTypeDetailed.MA
    elif 'option' in lower and 'licens' in lower:
        return DealTypeDetailed.OPTION_TO_LICENSE
    elif 'licens' in lower:
        return DealTypeDetailed.LICENSING
    else:
        return DealTypeDetailed.PARTNERSHIP

def parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    if not date_str:
        return date.today()
    try:
        return datetime.fromisoformat(str(date_str)).date()
    except:
        try:
            for fmt in ["%Y-%m-%d", "%B %d, %Y", "%Y"]:
                try:
                    return datetime.strptime(str(date_str), fmt).date()
                except:
                    continue
        except:
            pass
    return date.today()

def safe_decimal(value) -> Decimal:
    """Convert to Decimal safely."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except:
        return None

def get_enriched_value(field_data):
    """Extract value from enrichment field (handles both dict and direct values)."""
    if isinstance(field_data, dict):
        return field_data.get('value')
    return field_data

# Convert to Deal objects using OpenAI
deals = []
failed = []

print("Parsing deals with OpenAI (this may take a few minutes)...")

for i, candidate in enumerate(matched):
    try:
        print(f"  [{i+1}/{len(matched)}] Parsing: {candidate.get('name', 'Unknown')[:60]}...", end='\r')

        # Parse with OpenAI
        parsed = parse_candidate_with_openai(candidate)

        # Fallbacks if OpenAI returns empty/null
        url = candidate.get('url', 'https://example.com')
        description = candidate.get('description', '')

        deal = Deal(
            date_announced=parse_date(parsed.get('date_announced') or ''),
            target=(parsed.get('target_company') or candidate.get('name', 'Unknown'))[:200],
            acquirer=(parsed.get('acquirer_company') or 'Partner')[:200],
            stage=(parsed.get('development_stage') or 'unknown')[:50],
            therapeutic_area=(parsed.get('therapeutic_area') or 'Immunology & Inflammation')[:100],
            asset_focus=(parsed.get('asset_name') or candidate.get('name', 'Unknown'))[:200],
            deal_type_detailed=parse_deal_type(parsed.get('deal_type') or 'partnership'),
            source_url=url,
            upfront_value_usd=safe_decimal(parsed.get('upfront_value_usd')),
            contingent_payment_usd=safe_decimal(parsed.get('milestone_value_usd')),
            total_deal_value_usd=safe_decimal(parsed.get('total_deal_value_usd')),
            geography=None,
            key_evidence=description[:500],
            confidence=Decimal("0.5"),
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        )
        deals.append(deal)

    except Exception as e:
        failed.append({'index': i, 'name': candidate.get('name'), 'error': str(e)})
        print(f"\n✗ Failed to parse: {candidate.get('name')} - {e}")

print(f"\n✓ Successfully parsed {len(deals)} deals")
if failed:
    print(f"✗ Failed to parse {len(failed)} deals")

# Save OpenAI parsed results as JSON backup
openai_parsed_file = Path("output") / f"openai_parsed_{timestamp}.json"
with open(openai_parsed_file, 'w') as f:
    json.dump({
        'timestamp': timestamp,
        'total_deals': len(deals),
        'failed_parses': failed,
        'deals': [
            {
                'date_announced': str(d.date_announced),
                'target': d.target,
                'acquirer': d.acquirer,
                'stage': d.stage,
                'therapeutic_area': d.therapeutic_area,
                'asset_focus': d.asset_focus,
                'deal_type': d.deal_type_detailed.value,
                'source_url': str(d.source_url),
                'upfront_value_usd': str(d.upfront_value_usd) if d.upfront_value_usd else None,
                'contingent_payment_usd': str(d.contingent_payment_usd) if d.contingent_payment_usd else None,
                'total_deal_value_usd': str(d.total_deal_value_usd) if d.total_deal_value_usd else None,
                'key_evidence': d.key_evidence
            }
            for d in deals
        ]
    }, f, indent=2)

print(f"✓ Saved OpenAI parsed JSON to: {openai_parsed_file}\n")

# =============================================================================
# EXPORT TO EXCEL
# =============================================================================
if deals:
    output_file = Path("output") / f"deals_parallel_II_{timestamp}.xlsx"
    output_file.parent.mkdir(exist_ok=True)

    ExcelWriter().write(deals, str(output_file))
    print(f"\n✓ Saved to: {output_file}")

    # Show samples
    print(f"\nSample deals:")
    for i, d in enumerate(deals[:5], 1):
        print(f"{i}. {d.target} / {d.acquirer}")
        print(f"   {d.stage} | {d.therapeutic_area} | {d.date_announced}")
        if d.total_deal_value_usd:
            print(f"   Total value: ${d.total_deal_value_usd}M")
else:
    print("\n✗ No deals to export!")

print("\nDone!")