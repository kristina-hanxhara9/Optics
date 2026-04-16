#!/usr/bin/env python3
"""
Nordic Optics Business Registry Lookup
=======================================
Reads an Excel file with optics shop names (sheets: Denmark, Norway, Sweden)
and enriches each entry with official business registry data.

Registries used:
  - Norway:  Bronnøysundregistrene (BRREG) — Free REST API, no auth
  - Denmark: CVR via cvrapi.dk — Free REST API (User-Agent required)
  - Sweden:  Bolagsverket bulk CSV — downloaded from
             https://bolagsverket.se/omoss/oppnadata  (Näringslivsregistret)
             Loaded locally and matched by company name.

Usage:
  python scrape_registries.py optics_shops.xlsx
  python scrape_registries.py optics_shops.xlsx --sweden-csv swedish_companies.csv
  python scrape_registries.py optics_shops.xlsx -o enriched.xlsx --delay 1.0
  python scrape_registries.py optics_shops.xlsx --countries norway denmark
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path

# --- Check dependencies early with a clear message -----------------------
_MISSING = []
for _pkg in ["pandas", "openpyxl", "requests", "tqdm"]:
    try:
        __import__(_pkg)
    except ImportError:
        _MISSING.append(_pkg)
if _MISSING:
    print(f"ERROR: Missing packages: {', '.join(_MISSING)}")
    print(f"       Run:  pip install -r requirements.txt")
    sys.exit(1)

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging  –  send to BOTH stderr and stdout so output is always visible
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
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


def _normalise(name: str) -> str:
    """Lower-case, strip common suffixes (AB, AS, ApS, HB …) for matching."""
    n = name.lower().strip()
    n = re.sub(r"\b(ab|hb|kb|ef|as|asa|aps|a/s|i/s)\s*$", "", n).strip()
    n = re.sub(r"[^\w\s]", " ", n)           # punctuation → space
    return re.sub(r"\s+", " ", n).strip()     # collapse whitespace


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
#  SWEDEN  —  Bolagsverket bulk data (local CSV / XLSX)
# =========================================================================
class SwedenBulkCSV:
    """
    Matches Swedish optics shop names against a locally downloaded bulk file
    from Bolagsverket (Swedish Companies Registration Office).

    Where to get the data (FREE, no account needed)
    ------------------------------------------------
    Go to Bolagsverket's download page:
      https://bolagsverket.se/apierochoppnadata/nedladdningsbarafiler.2517.html

    Download one of:
      - bolagsverket_bulkfil.zip  (company data from Bolagsverket)
      - scb_bulkfil.zip           (company data from SCB / Statistics Sweden)

    Unzip → .txt file.  Pass it with  --sweden-csv <file.txt>

    These are "Värdefulla datamängder" (EU High-Value Datasets), released
    Feb 2025.  Updated weekly, completely free, no contract.

    Contents: org number, company name, legal form, address, SNI codes,
    business description, and more.

    Supported formats: .csv  .tsv  .txt  .xlsx  .xls
    """

    # Common Swedish column names → our canonical field
    _COL_MAP = {
        # name
        "företagsnamn": "name", "foretagsnamn": "name",
        "juridiskt namn": "name", "juridiskt_namn": "name",
        "namn": "name", "name": "name", "company_name": "name",
        "firma": "name", "bolagsnamn": "name",
        # org number
        "organisationsnummer": "org_number", "orgnr": "org_number",
        "org.nr": "org_number", "org_nr": "org_number",
        "organisationsnr": "org_number", "org_number": "org_number",
        # legal form
        "företagsform": "legal_form", "foretagsform": "legal_form",
        "juridisk form": "legal_form", "bolagsform": "legal_form",
        "legal_form": "legal_form", "company_type": "legal_form",
        # SNI / industry
        "sni": "industry_code", "sni_kod": "industry_code",
        "sni-kod": "industry_code", "branschkod": "industry_code",
        "industry_code": "industry_code", "nace": "industry_code",
        "bransch": "industry_desc", "sni_beskrivning": "industry_desc",
        "industry_desc": "industry_desc", "industry_description": "industry_desc",
        # address
        "adress": "address", "gatuadress": "address",
        "address": "address", "utdelningsadress": "address",
        # postal code
        "postnummer": "postal_code", "postnr": "postal_code",
        "postal_code": "postal_code", "zipcode": "postal_code",
        # city
        "postort": "city", "ort": "city", "stad": "city",
        "city": "city", "kommun": "city",
        # status
        "status": "status", "företagsstatus": "status",
        # employees
        "anställda": "employees", "antal_anstallda": "employees",
        "employees": "employees",
    }

    def __init__(self, file_path: str):
        self.path = Path(file_path)
        log.info("Loading Swedish bulk data from %s …", self.path)
        self.df = self._load()
        self.col_mapping = self._map_columns()
        self.name_col = self.col_mapping.get("name")
        if not self.name_col:
            raise ValueError(
                f"Cannot find a company-name column in {self.path}. "
                f"Columns found: {list(self.df.columns)}"
            )
        # Pre-compute normalised names for fast matching
        self.df["_norm"] = self.df[self.name_col].astype(str).apply(_normalise)
        log.info("  Loaded %d Swedish companies (name col: '%s')", len(self.df), self.name_col)

    # ------------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        suffix = self.path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(self.path, dtype=str)

        # Try as delimited text regardless of extension (.csv .tsv .txt or anything else)
        for enc in ["utf-8", "latin-1", "cp1252"]:
            for sep in [";", ",", "\t", "|"]:
                try:
                    df = pd.read_csv(self.path, sep=sep, dtype=str,
                                     encoding=enc, on_bad_lines="skip")
                    if len(df.columns) > 1:
                        log.info("  Parsed with sep=%r  encoding=%s  cols=%d",
                                 sep, enc, len(df.columns))
                        return df
                except Exception:
                    continue
        # last resort: auto-detect
        return pd.read_csv(self.path, dtype=str, encoding="latin-1",
                           on_bad_lines="skip")

    def _map_columns(self) -> dict[str, str]:
        """Map bulk-file columns to canonical field names."""
        mapping: dict[str, str] = {}   # canonical → original col name
        for col in self.df.columns:
            key = col.lower().strip().replace(" ", "_")
            if key in self._COL_MAP:
                canonical = self._COL_MAP[key]
                if canonical not in mapping:
                    mapping[canonical] = col
        return mapping

    # ------------------------------------------------------------------
    def search(self, name: str) -> RegistryResult | None:
        query_norm = _normalise(name)
        if not query_norm:
            return None

        # 1) Exact normalised match
        exact = self.df[self.df["_norm"] == query_norm]
        if not exact.empty:
            return self._build(name, exact.iloc[0], 1.0)

        # 2) Substring: bulk name contains the query (or vice-versa)
        mask_contains = self.df["_norm"].str.contains(
            re.escape(query_norm), na=False
        )
        subset = self.df[mask_contains]
        if not subset.empty:
            # pick the shortest name (most specific match)
            best_idx = subset[self.name_col].str.len().idxmin()
            row = subset.loc[best_idx]
            score = _similarity(name, str(row[self.name_col]))
            return self._build(name, row, score)

        # 3) Token overlap — keep rows sharing ≥1 significant word
        words = [w for w in query_norm.split() if len(w) > 2]
        if words:
            pattern = "|".join(re.escape(w) for w in words)
            mask_tok = self.df["_norm"].str.contains(pattern, na=False)
            candidates = self.df[mask_tok]
        else:
            candidates = self.df

        if candidates.empty:
            return None

        # 4) Fuzzy-score the candidates (cap at 2000 to stay fast)
        if len(candidates) > 2000:
            candidates = candidates.head(2000)

        scores = candidates[self.name_col].apply(
            lambda x: _similarity(name, str(x))
        )
        best_idx = scores.idxmax()
        best_score = scores[best_idx]
        if best_score < 0.45:
            return None
        return self._build(name, candidates.loc[best_idx], best_score)

    # ------------------------------------------------------------------
    def _build(self, query: str, row: pd.Series, score: float) -> RegistryResult:
        def _g(canonical: str) -> str:
            col = self.col_mapping.get(canonical)
            if col is None:
                return ""
            val = row.get(col, "")
            return "" if pd.isna(val) else str(val).strip()

        return RegistryResult(
            matched_name=_g("name"),
            org_number=_g("org_number"),
            legal_form=_g("legal_form"),
            address=_g("address"),
            postal_code=_g("postal_code"),
            city=_g("city"),
            country="Sweden",
            industry_code=_g("industry_code"),
            industry_desc=_g("industry_desc"),
            status=_g("status") or "Unknown",
            employees=_g("employees"),
            match_score=round(score, 3),
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
    """Guess which column holds the business names.
    If the sheet has only one column, just return it."""
    if len(df.columns) == 1:
        return df.columns[0]
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
#  Post-lookup analysis
# =========================================================================
# Words to ignore when counting name keywords
_STOPWORDS = {
    # Legal suffixes
    "ab", "as", "asa", "aps", "a/s", "i/s", "hb", "kb", "ef", "oy", "ltd",
    "gmbh", "inc", "co", "sa", "nv",
    # Common filler
    "og", "och", "and", "i", "the", "de", "van", "von", "af", "av",
    # Country / generic
    "danmark", "denmark", "norge", "norway", "sverige", "sweden",
    "nordic", "scandinavia", "scandinavian", "europe",
}


def analyse_nace(all_enriched: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Frequency table of NACE / industry codes across all countries."""
    rows = []
    for sheet, df in all_enriched.items():
        found = df[df["matched_name"] != "NOT FOUND"]
        for _, r in found.iterrows():
            code = str(r.get("industry_code", "")).strip()
            desc = str(r.get("industry_desc", "")).strip()
            if code and code.lower() != "nan":
                rows.append({
                    "country": sheet,
                    "industry_code": code,
                    "industry_desc": desc,
                })

    if not rows:
        return pd.DataFrame(columns=["industry_code", "industry_desc",
                                      "count", "pct", "countries"])

    raw = pd.DataFrame(rows)

    # Aggregate per unique code
    grouped = (
        raw.groupby("industry_code")
        .agg(
            industry_desc=("industry_desc", lambda s: s.mode().iloc[0] if len(s) else ""),
            count=("industry_code", "size"),
            countries=("country", lambda s: ", ".join(sorted(s.unique()))),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    total = grouped["count"].sum()
    grouped["pct"] = (grouped["count"] / total * 100).round(1)
    return grouped


def analyse_keywords(all_enriched: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Frequency table of significant words in matched business names."""
    word_counter: Counter = Counter()
    word_by_country: dict[str, set] = {}

    for sheet, df in all_enriched.items():
        found = df[df["matched_name"] != "NOT FOUND"]
        for _, r in found.iterrows():
            name = str(r.get("matched_name", ""))
            tokens = re.findall(r"[a-zæøåäöü]+", name.lower())
            for tok in tokens:
                if len(tok) <= 1 or tok in _STOPWORDS:
                    continue
                word_counter[tok] += 1
                word_by_country.setdefault(tok, set()).add(sheet)

    if not word_counter:
        return pd.DataFrame(columns=["keyword", "count", "pct", "countries"])

    total = sum(word_counter.values())
    rows = [
        {
            "keyword": w,
            "count": c,
            "pct": round(c / total * 100, 1),
            "countries": ", ".join(sorted(word_by_country[w])),
        }
        for w, c in word_counter.most_common()
    ]
    return pd.DataFrame(rows)


def print_analysis(nace_df: pd.DataFrame, kw_df: pd.DataFrame) -> None:
    """Print analysis tables to the console."""

    print()
    print("=" * 60)
    print("  NACE / INDUSTRY CODE ANALYSIS")
    print("=" * 60)
    if nace_df.empty:
        print("  No industry codes found.")
    else:
        top = nace_df.head(15)
        for _, r in top.iterrows():
            desc = r["industry_desc"][:40] if r["industry_desc"] else ""
            print(f"  {r['industry_code']:>8s}  {r['count']:4d}  ({r['pct']:5.1f}%)  "
                  f"{desc:40s}  [{r['countries']}]")
        if len(nace_df) > 15:
            print(f"  … and {len(nace_df) - 15} more codes (see output Excel)")

    print()
    print("=" * 60)
    print("  NAME KEYWORD ANALYSIS")
    print("=" * 60)
    if kw_df.empty:
        print("  No keywords found.")
    else:
        top = kw_df.head(25)
        for _, r in top.iterrows():
            print(f"  {r['keyword']:25s}  {r['count']:4d}  ({r['pct']:5.1f}%)  "
                  f"[{r['countries']}]")
        if len(kw_df) > 25:
            print(f"  … and {len(kw_df) - 25} more keywords (see output Excel)")


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
  python scrape_registries.py optics_shops.xlsx --sweden-csv swedish_companies.csv
  python scrape_registries.py optics_shops.xlsx -o enriched.xlsx
  python scrape_registries.py optics_shops.xlsx --countries norway denmark
  python scrape_registries.py optics_shops.xlsx --delay 1.5 --json

REGISTRIES
  Norway   BRREG             data.brreg.no                Free REST, no auth
  Denmark  CVR / cvrapi.dk   cvrapi.dk                    Free REST, User-Agent
  Sweden   Bolagsverket CSV  bolagsverket.se/oppnadata    Bulk download, local match

GETTING THE SWEDISH BULK DATA (FREE, no account needed)
  1. Go to: https://bolagsverket.se/apierochoppnadata/nedladdningsbarafiler.2517.html
  2. Download "bolagsverket_bulkfil.zip" or "scb_bulkfil.zip"
  3. Unzip → .txt file
  4. Pass with --sweden-csv <file.txt>
  These are EU "High-Value Datasets", free since Feb 2025, updated weekly.
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
    ap.add_argument("--sweden-csv", metavar="PATH",
                    help="Path to Swedish bulk company data (CSV/XLSX) from Bolagsverket. "
                         "Required for Sweden lookups.")
    args = ap.parse_args()

    print()
    print("Nordic Optics — Business Registry Lookup")
    print("=" * 42)

    src = Path(args.input_file)
    if not src.exists():
        print(f"ERROR: File not found: {src}")
        print(f"       Make sure the Excel file is in the current directory,")
        print(f"       or provide the full path.")
        sys.exit(1)

    dest = Path(args.output) if args.output else src.with_name(f"{src.stem}_enriched.xlsx")

    # --- registries -------------------------------------------------------
    registries: dict = {
        "norway":  NorwayBRREG(),
        "denmark": DenmarkCVR(user_agent=args.user_agent),
    }

    if args.sweden_csv:
        csv_path = Path(args.sweden_csv)
        if not csv_path.exists():
            print(f"ERROR: Swedish data file not found: {csv_path}")
            sys.exit(1)
        registries["sweden"] = SwedenBulkCSV(str(csv_path))
    else:
        log.info(
            "No --sweden-csv provided. Sweden lookups will be skipped.\n"
            "  To enable Sweden:\n"
            "  1. Download from: https://bolagsverket.se/apierochoppnadata/nedladdningsbarafiler.2517.html\n"
            "     → bolagsverket_bulkfil.zip  (free, no account needed)\n"
            "  2. Unzip and rerun with:  --sweden-csv bolagsverket_bulkfil.txt"
        )

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
            log.warning("No registry for '%s' — skipping (see --help)", country)
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

    # --- analysis ---------------------------------------------------------
    log.info("\nRunning post-lookup analysis …")
    nace_df = analyse_nace(results)
    kw_df = analyse_keywords(results)

    # --- write output -----------------------------------------------------
    log.info("Writing → %s", dest)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        for sheet_name, edf in results.items():
            edf.to_excel(writer, sheet_name=sheet_name, index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        if not nace_df.empty:
            nace_df.to_excel(writer, sheet_name="NACE Analysis", index=False)
        if not kw_df.empty:
            kw_df.to_excel(writer, sheet_name="Keyword Analysis", index=False)

    # --- console summary --------------------------------------------------
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for r in summary_rows:
        print(f"  {r['Country']:20s}  {r['Found']}/{r['Total']} found  ({r['Match Rate']})")
    print(f"\n  Output → {dest}")
    print("=" * 60)

    # --- console analysis -------------------------------------------------
    print_analysis(nace_df, kw_df)


if __name__ == "__main__":
    main()
