"""
Parse saved Parallel AI results with OpenAI.
Use this if the main script failed during OpenAI parsing.
"""
import os
import json
from datetime import datetime, date, timezone
from decimal import Decimal
from pathlib import Path
from openai import OpenAI

from deal_finder.models import Deal, DealTypeDetailed
from deal_finder.output import ExcelWriter

# =============================================================================
# CONFIGURATION
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set via environment variable

# Point to your saved Parallel raw output file
PARALLEL_RAW_FILE = "output/parallel_raw_20251118_184341.json"

# =============================================================================
# LOAD PARALLEL DATA
# =============================================================================
print(f"Loading Parallel data from: {PARALLEL_RAW_FILE}")

with open(PARALLEL_RAW_FILE, 'r') as f:
    parallel_data = json.load(f)

candidates = parallel_data.get('candidates', [])
matched = [c for c in candidates if c.get('match_status') == 'matched']

print(f"✓ Loaded {len(matched)} matched deals\n")

# =============================================================================
# OPENAI PARSING
# =============================================================================
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

try:
    openai_client = OpenAI()
    print("✓ OpenAI client initialized\n")
except Exception as e:
    print(f"✗ OpenAI client initialization failed: {e}")
    print("Trying alternative initialization...")
    try:
        from openai import OpenAI as OpenAIClient
        openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
        print("✓ OpenAI client initialized with alternative method\n")
    except Exception as e2:
        print(f"✗ Failed: {e2}")
        print("\nTry running: pip install --upgrade openai")
        exit(1)

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

# =============================================================================
# PARSE WITH OPENAI
# =============================================================================
# Check for existing OpenAI responses file
import glob
existing_files = sorted(glob.glob("output/openai_responses_*.json"), reverse=True)

if existing_files:
    openai_responses_file = Path(existing_files[0])
    print(f"✓ Found existing OpenAI responses: {openai_responses_file}")
    with open(openai_responses_file, 'r') as f:
        openai_responses = json.load(f)
    print(f"  Loaded {len(openai_responses)} cached responses\n")
    # Extract timestamp from filename
    timestamp = openai_responses_file.stem.replace('openai_responses_', '')
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    openai_responses_file = Path("output") / f"openai_responses_{timestamp}.json"
    openai_responses = {}
    print(f"No existing responses found, creating new file: {openai_responses_file}\n")

deals = []
failed = []

print("Parsing deals with OpenAI...")

for i, candidate in enumerate(matched):
    try:
        candidate_id = candidate.get('candidate_id', f'candidate_{i}')
        print(f"  [{i+1}/{len(matched)}] {candidate.get('name', 'Unknown')[:60]}...", end='\r')

        # Check if already parsed
        if candidate_id in openai_responses:
            parsed = openai_responses[candidate_id]
        else:
            # Parse with OpenAI
            parsed = parse_candidate_with_openai(candidate)

            # Save immediately
            openai_responses[candidate_id] = parsed
            with open(openai_responses_file, 'w') as f:
                json.dump(openai_responses, f, indent=2)


        # Fallbacks if OpenAI returns empty/null
        url = candidate.get('url', 'https://example.com')
        description = candidate.get('description', '')

        deal = Deal(
            date_announced=parse_date(parsed.get('date_announced') or ''),
            target=(parsed.get('target_company') or candidate.get('name', 'Unknown'))[:200],
            acquirer=(parsed.get('acquirer_company') or 'Partner')[:200],
            stage=(parsed.get('development_stage') or 'unknown')[:50],
            therapeutic_area='Immunology & Inflammation',  # From match conditions
            asset_focus=(parsed.get('asset_name') or candidate.get('name', 'Unknown'))[:200],
            deal_type_detailed=parse_deal_type(parsed.get('deal_type') or 'partnership'),
            source_url=url,
            upfront_value_usd=safe_decimal(parsed.get('upfront_value_usd')),
            contingent_payment_usd=safe_decimal(parsed.get('milestone_value_usd')),
            total_deal_value_usd=safe_decimal(parsed.get('total_deal_value_usd')),
            geography=None,
            # Note: Deal model uses 'evidence' field (FieldEvidence object), not 'key_evidence'
            # We'll just pass description in the JSON output below
            confidence=Decimal("0.5"),
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        )
        deals.append(deal)

    except Exception as e:
        failed.append({'index': i, 'name': candidate.get('name'), 'error': str(e)})
        print(f"\n✗ Failed: {candidate.get('name')} - {e}")

print(f"\n\n✓ Successfully parsed {len(deals)} deals")
if failed:
    print(f"✗ Failed to parse {len(failed)} deals")

# =============================================================================
# SAVE RESULTS
# =============================================================================
# Save JSON
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
                'deal_type': d.deal_type_detailed.value if hasattr(d.deal_type_detailed, 'value') else str(d.deal_type_detailed),
                'source_url': str(d.source_url),
                'upfront_value_usd': str(d.upfront_value_usd) if d.upfront_value_usd else None,
                'contingent_payment_usd': str(d.contingent_payment_usd) if d.contingent_payment_usd else None,
                'total_deal_value_usd': str(d.total_deal_value_usd) if d.total_deal_value_usd else None
            }
            for d in deals
        ]
    }, f, indent=2)

print(f"\n✓ Saved JSON to: {openai_parsed_file}")

# Save Excel
if deals:
    output_file = Path("output") / f"deals_parallel_II_{timestamp}.xlsx"

    ExcelWriter().write(deals, str(output_file))
    print(f"✓ Saved Excel to: {output_file}")

    # Show samples
    print(f"\nSample deals:")
    for i, d in enumerate(deals[:5], 1):
        print(f"{i}. {d.target} / {d.acquirer}")
        print(f"   {d.stage} | {d.date_announced}")
        if d.upfront_value_usd:
            print(f"   Upfront: ${d.upfront_value_usd}M")
        if d.contingent_payment_usd:
            print(f"   Milestones: ${d.contingent_payment_usd}M")
else:
    print("\n✗ No deals to export!")

print("\nDone!")
