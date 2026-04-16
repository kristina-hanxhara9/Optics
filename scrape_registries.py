#!/usr/bin/env python3
"""
Nordic Optics Business Registry Lookup
=======================================
Reads an Excel file with optics shop names (sheets: Denmark, Norway, Sweden)
and enriches each entry with official business registry data.

Registries used:
  - Norway:  Bronnøysundregistrene (BRREG) — Free REST API, no auth
  - Denmark: CVR via cvrapi.dk — Free REST API (User-Agent required)
  - Sweden:  OpenCorporates — Free tier REST API (500 req/month, no auth)

Usage:
  python scrape_registries.py optics_shops.xlsx
  python scrape_registries.py optics_shops.xlsx -o enriched.xlsx --delay 1.0
  python scrape_registries.py optics_shops.xlsx --name-column "Business Name"
  python scrape_registries.py optics_shops.xlsx --countries norway denmark
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optics")

# ---------------------------------------------------------------------------
# Standardised result
# ---------------------------------------------------------------------------

@dataclass
class RegistryResult:
    """One row of enriched data coming back from any registry."""
    matched_name: str = ""
    org_number: str = ""
    legal_form: str = ""
    address: str = ""
    postal_code: str = ""
    city: str = ""
    country: str = ""
    industry_code: str = ""
    industry_desc: str = ""
    status: str = ""
    employees: str = ""
    website: str = ""
    phone: str = ""
    email: str = ""
    registration_date: str = ""
    match_score: float = 0.0
    raw_json: dict = field(default_factory=dict, repr=False)

    def to_flat_dict(self) -> dict:
        """Return a dict suitable for a DataFrame row (no nested objects)."""
        d = asdict(self)
        d.pop("raw_json", None)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


def _get_json(url: str, params: dict = None, headers: dict = None) -> dict | None:
    """GET with retries and exponential back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = RETRY_BACKOFF ** attempt
                log.warning("Rate-limited (429). Waiting %ds…", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            wait = RETRY_BACKOFF ** attempt
            log.warning("Connection error (attempt %d/%d): %s — retrying in %ds",
                        attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF ** attempt
            log.warning("Timeout (attempt %d/%d) — retrying in %ds",
                        attempt, MAX_RETRIES, wait)
            time.sleep(wait)
        except Exception as exc:
            log.error("Request failed: %s", exc)
            return None
    log.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# =========================================================================
#  NORWAY  —  Brønnøysundregistrene (BRREG)
# =========================================================================
class NorwayBRREG:
    """
    Docs: https://data.brreg.no/enhetsregisteret/api/docs/index.html
    No authentication required.
    """
    BASE = "https://data.brreg.no/enhetsregisteret/api/enheter"

    def search(self, name: str) -> RegistryResult | None:
        data = _get_json(self.BASE, params={"navn": name, "size": 5})
        if not data:
            return None

        units = data.get("_embedded", {}).get("enheter", [])
        if not units:
            return None

        best = self._pick_best(name, units)
        return self._parse(name, best)

    # ------------------------------------------------------------------
    def _pick_best(self, query: str, units: list[dict]) -> dict:
        scored = [(u, _similarity(query, u.get("navn", ""))) for u in units]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]

    def _parse(self, query: str, unit: dict) -> RegistryResult:
        addr = unit.get("forretningsadresse") or unit.get("postadresse") or {}
        nace = unit.get("naeringskode1") or {}
        org_form = unit.get("organisasjonsform") or {}

        return RegistryResult(
            matched_name=unit.get("navn", ""),
            org_number=str(unit.get("organisasjonsnummer", "")),
            legal_form=org_form.get("beskrivelse", ""),
            address=", ".join(addr.get("adresse", [])),
            postal_code=addr.get("postnummer", ""),
            city=addr.get("poststed", ""),
            country="Norway",
            industry_code=nace.get("kode", ""),
            industry_desc=nace.get("beskrivelse", ""),
            status="Active" if not unit.get("slettedato") else "Deleted",
            employees=str(unit.get("antallAnsatte", "")),
            website=unit.get("hjemmeside", "") or "",
            registration_date=unit.get("registreringsdatoEnhetsregisteret", ""),
            match_score=round(_similarity(query, unit.get("navn", "")), 3),
            raw_json=unit,
        )


# =========================================================================
#  DENMARK  —  CVR via cvrapi.dk
# =========================================================================
class DenmarkCVR:
    """
    Docs: https://cvrapi.dk/documentation
    Requires a descriptive User-Agent header (name + contact).
    Free, no API key.
    """
    BASE = "https://cvrapi.dk/api"

    def __init__(self, user_agent: str = "OpticsScraper/1.0 (contact@example.com)"):
        self.ua = user_agent

    def search(self, name: str) -> RegistryResult | None:
        data = _get_json(
            self.BASE,
            params={"search": name, "country": "dk"},
            headers={"User-Agent": self.ua},
        )
        if not data or "error" in data:
            return None
        return self._parse(name, data)

    def _parse(self, query: str, d: dict) -> RegistryResult:
        end = d.get("enddate")
        return RegistryResult(
            matched_name=d.get("name", ""),
            org_number=str(d.get("vat", "")),
            legal_form=d.get("companydesc", ""),
            address=d.get("address", ""),
            postal_code=str(d.get("zipcode", "")),
            city=d.get("city", ""),
            country="Denmark",
            industry_code=str(d.get("industrycode", "")),
            industry_desc=d.get("industrydesc", ""),
            status="Closed" if end else "Active",
            employees=str(d.get("employees", "")),
            phone=str(d.get("phone", "") or ""),
            email=d.get("email", "") or "",
            website="",
            registration_date=d.get("startdate", ""),
            match_score=round(_similarity(query, d.get("name", "")), 3),
            raw_json=d,
        )


# =========================================================================
#  SWEDEN  —  OpenCorporates (free tier)
# =========================================================================
class SwedenOpenCorporates:
    """
    Docs: https://api.opencorporates.com/documentation/API-Reference
    Free tier: anonymous access, rate-limited.
    Covers Bolagsverket data.
    """
    BASE = "https://api.opencorporates.com/v0.4/companies/search"

    def search(self, name: str) -> RegistryResult | None:
        data = _get_json(
            self.BASE,
            params={
                "q": name,
                "jurisdiction_code": "se",
                "per_page": 5,
                "order": "score",
            },
        )
        if not data:
            return None

        companies = data.get("results", {}).get("companies", [])
        if not companies:
            return None

        best = self._pick_best(name, companies)
        return self._parse(name, best)

    # ------------------------------------------------------------------
    def _pick_best(self, query: str, items: list[dict]) -> dict:
        scored = [
            (it, _similarity(query, it.get("company", {}).get("name", "")))
            for it in items
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]

    def _parse(self, query: str, item: dict) -> RegistryResult:
        c = item.get("company", {})
        addr = c.get("registered_address") or {}

        ind_codes = c.get("industry_codes") or []
        ind_code = ind_codes[0].get("industry_code", {}).get("code", "") if ind_codes else ""
        ind_desc = ind_codes[0].get("industry_code", {}).get("description", "") if ind_codes else ""

        return RegistryResult(
            matched_name=c.get("name", ""),
            org_number=c.get("company_number", ""),
            legal_form=c.get("company_type", ""),
            address=addr.get("street_address", "") or "",
            postal_code=addr.get("postal_code", "") or "",
            city=addr.get("locality", "") or "",
            country="Sweden",
            industry_code=ind_code,
            industry_desc=ind_desc,
            status=c.get("current_status", "") or "",
            registration_date=c.get("incorporation_date", "") or "",
            match_score=round(_similarity(query, c.get("name", "")), 3),
            raw_json=c,
        )


# =========================================================================
#  Excel I/O
# =========================================================================
COUNTRY_KEYWORDS = {
    "denmark": "denmark", "danmark": "denmark", "dk": "denmark",
    "norway": "norway",  "norge": "norway",    "no": "norway",
    "sweden": "sweden",  "sverige": "sweden",  "se": "sweden",
}

NAME_HINTS = [
    "name", "business", "company", "firma", "virksomhed", "foretak",
    "företag", "butik", "optik", "shop", "store", "navn", "namn",
    "forretning", "selskap", "bolag", "kæde", "chain",
]


def _detect_country(sheet_name: str, index: int) -> str:
    """Map a sheet name (or position) to a country key."""
    low = sheet_name.lower().strip()
    for kw, country in COUNTRY_KEYWORDS.items():
        if kw in low:
            return country
    fallback = {0: "denmark", 1: "norway", 2: "sweden"}
    return fallback.get(index, f"unknown_{index}")


def _detect_name_column(df: pd.DataFrame) -> str:
    """Guess which column holds the business names."""
    for col in df.columns:
        for hint in NAME_HINTS:
            if hint in str(col).lower():
                return col
    return df.columns[0]


def read_input(path: str) -> dict[str, dict]:
    """Return {country: {"df": DataFrame, "sheet": str}}."""
    xls = pd.ExcelFile(path)
    log.info("Sheets found: %s", xls.sheet_names)
    out = {}
    for idx, sn in enumerate(xls.sheet_names):
        df = pd.read_excel(path, sheet_name=sn)
        country = _detect_country(sn, idx)
        log.info("  Sheet '%s' → %s  (%d rows)", sn, country, len(df))
        out[country] = {"df": df, "sheet": sn}
    return out


# =========================================================================
#  Processing pipeline
# =========================================================================
def process_sheet(
    df: pd.DataFrame,
    registry,
    country: str,
    name_col: str,
    delay: float,
    save_json: bool = False,
) -> pd.DataFrame:
    """Look up every business in *df* and return a merged DataFrame."""

    enriched_rows: list[dict] = []
    raw_jsons: list[dict] = []

    names = df[name_col].astype(str).str.strip()
    pbar = tqdm(names, desc=country.upper(), unit="biz", leave=True)

    for biz_name in pbar:
        pbar.set_postfix_str(biz_name[:30])

        if not biz_name or biz_name.lower() == "nan":
            enriched_rows.append(RegistryResult().to_flat_dict())
            raw_jsons.append({})
            continue

        result = registry.search(biz_name)
        if result:
            enriched_rows.append(result.to_flat_dict())
            raw_jsons.append(result.raw_json)
            status = f"✓ {result.matched_name[:25]} ({result.match_score:.0%})"
        else:
            empty = RegistryResult(matched_name="NOT FOUND")
            enriched_rows.append(empty.to_flat_dict())
            raw_jsons.append({})
            status = "✗ not found"

        pbar.set_postfix_str(status)
        time.sleep(delay)

    result_df = pd.DataFrame(enriched_rows)
    merged = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    if save_json:
        json_path = Path(f"{country}_raw_responses.json")
        json_path.write_text(json.dumps(raw_jsons, ensure_ascii=False, indent=2))
        log.info("Raw JSON saved → %s", json_path)

    return merged


# =========================================================================
#  Main
# =========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nordic Optics — Business Registry Lookup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  python scrape_registries.py optics_shops.xlsx
  python scrape_registries.py optics_shops.xlsx -o enriched.xlsx
  python scrape_registries.py optics_shops.xlsx --name-column "Butik"
  python scrape_registries.py optics_shops.xlsx --countries norway sweden
  python scrape_registries.py optics_shops.xlsx --delay 1.5 --json

REGISTRIES
  Norway   BRREG            data.brreg.no        Free REST, no auth
  Denmark  CVR / cvrapi.dk  cvrapi.dk            Free REST, User-Agent
  Sweden   OpenCorporates   api.opencorporates   Free tier (rate-limited)
""",
    )
    ap.add_argument("input_file", help="Excel file (.xlsx) with optics shop names")
    ap.add_argument("-o", "--output", help="Output Excel path  [default: <input>_enriched.xlsx]")
    ap.add_argument("-n", "--name-column", help="Column with business names (auto-detected if omitted)")
    ap.add_argument("-d", "--delay", type=float, default=0.5,
                    help="Seconds between API calls  [default: 0.5]")
    ap.add_argument("--countries", nargs="+", choices=["denmark", "norway", "sweden"],
                    help="Only process these countries")
    ap.add_argument("--json", action="store_true", help="Also dump raw JSON per country")
    ap.add_argument("--user-agent", default="OpticsScraper/1.0 (contact@example.com)",
                    help="User-Agent for cvrapi.dk  [default: generic]")
    args = ap.parse_args()

    src = Path(args.input_file)
    if not src.exists():
        log.error("File not found: %s", src)
        sys.exit(1)

    dest = Path(args.output) if args.output else src.with_name(f"{src.stem}_enriched.xlsx")

    # --- registries -------------------------------------------------------
    registries = {
        "norway":  NorwayBRREG(),
        "denmark": DenmarkCVR(user_agent=args.user_agent),
        "sweden":  SwedenOpenCorporates(),
    }

    # --- read input -------------------------------------------------------
    sheets = read_input(str(src))

    # --- process ----------------------------------------------------------
    results: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for country, info in sheets.items():
        if args.countries and country not in args.countries:
            log.info("Skipping %s (filtered out)", country)
            continue
        if country not in registries:
            log.warning("No registry for '%s' — skipping", country)
            continue

        df = info["df"]
        sheet = info["sheet"]
        name_col = args.name_column if args.name_column else _detect_name_column(df)

        if args.name_column and args.name_column not in df.columns:
            log.error("Column '%s' not in sheet '%s'. Available: %s",
                      args.name_column, sheet, list(df.columns))
            continue

        log.info("")
        log.info("=" * 60)
        log.info("  %s  |  sheet '%s'  |  %d rows  |  col '%s'",
                 country.upper(), sheet, len(df), name_col)
        log.info("  Registry: %s", type(registries[country]).__name__)
        log.info("=" * 60)

        enriched = process_sheet(
            df, registries[country], country, name_col, args.delay, save_json=args.json
        )
        results[sheet] = enriched

        total = len(enriched)
        found = int((enriched["matched_name"] != "NOT FOUND").sum())
        high = int((enriched["match_score"] >= 0.8).sum()) if "match_score" in enriched else 0
        summary_rows.append({
            "Country": sheet,
            "Total": total,
            "Found": found,
            "Not Found": total - found,
            "Match Rate": f"{found / total * 100:.1f}%" if total else "N/A",
            "High Confidence (≥80%)": high,
        })

    if not results:
        log.warning("Nothing to write.")
        sys.exit(0)

    # --- write output -----------------------------------------------------
    log.info("\nWriting → %s", dest)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        for sheet_name, edf in results.items():
            edf.to_excel(writer, sheet_name=sheet_name, index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

    # --- console summary --------------------------------------------------
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for r in summary_rows:
        print(f"  {r['Country']:20s}  {r['Found']}/{r['Total']} found  ({r['Match Rate']})")
    print(f"\n  Output → {dest}")
    print("=" * 60)


if __name__ == "__main__":
    main()
