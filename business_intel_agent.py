#!/usr/bin/env python3
"""
Optics Business Intelligence Agent
===================================
Uses AI + web search to research each optics shop and find:
  - What they sell (glasses, lenses, eye exams, hearing aids…)
  - Chain / group affiliation (Specsavers, Synoptik, independent…)
  - Phone, email, website, address, opening hours

Supports both OpenAI (GPT) and Anthropic (Claude) as providers.

Usage:
  # OpenAI (cheapest — gpt-4o-mini)
  export OPENAI_API_KEY=sk-...
  python business_intel_agent.py optics_shops.xlsx --provider openai

  # Anthropic (Claude)
  export ANTHROPIC_API_KEY=sk-ant-...
  python business_intel_agent.py optics_shops.xlsx --provider anthropic
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check  (provider-specific packages checked later)
# ---------------------------------------------------------------------------
for _pkg in ["pandas", "openpyxl", "tqdm"]:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"ERROR: Missing package: {_pkg}")
        print(f"       Run:  pip install -r requirements.txt")
        sys.exit(1)

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# System prompt (shared across providers)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a business research agent specialising in the Nordic optics / eyewear
industry (Denmark, Norway, Sweden, Finland).

When given a business name and country, use web search to find detailed
information about that specific business.

ALWAYS respond with ONLY a valid JSON object — no markdown fences, no
explanation, no preamble.  The JSON must have exactly these keys:

{
  "official_name": "the registered / trading name",
  "what_they_sell": "products & services, e.g. prescription glasses, sunglasses, contact lenses, eye exams, hearing aids",
  "chain": "retail chain or buying group (Specsavers, Synoptik, Synsam, Louis Nielsen, Profil Optik, Smarteyes, Brilleland, Interoptik, or 'Independent')",
  "phone": "main phone number, or empty string",
  "email": "contact email, or empty string",
  "website": "website URL, or empty string",
  "address": "street address + city, or empty string",
  "opening_hours": "e.g. Mon-Fri 10-18, Sat 10-15, or empty string",
  "description": "1–2 sentence description of the business"
}

If you truly cannot find any information, return the JSON with empty strings
and set description to "No information found online"."""

RESULT_COLS = [
    "official_name", "what_they_sell", "chain", "phone", "email",
    "website", "address", "opening_hours", "description",
]

# ---------------------------------------------------------------------------
# Country / column helpers
# ---------------------------------------------------------------------------
_COUNTRY_KW = {
    "denmark": "Denmark", "danmark": "Denmark", "dk": "Denmark",
    "norway": "Norway",  "norge": "Norway",    "no": "Norway",
    "sweden": "Sweden",  "sverige": "Sweden",  "se": "Sweden",
    "finland": "Finland", "suomi": "Finland",  "fi": "Finland",
}

_NAME_HINTS = [
    "name", "business", "company", "firma", "virksomhed", "foretak",
    "företag", "butik", "optik", "shop", "store", "navn", "namn",
]


def detect_country(sheet: str, idx: int) -> str:
    low = sheet.lower().strip()
    for kw, country in _COUNTRY_KW.items():
        if kw in low:
            return country
    return {0: "Denmark", 1: "Norway", 2: "Sweden"}.get(idx, f"Country {idx}")


def detect_name_col(df: pd.DataFrame) -> str:
    if len(df.columns) == 1:
        return df.columns[0]
    for col in df.columns:
        for hint in _NAME_HINTS:
            if hint in str(col).lower():
                return col
    return df.columns[0]


# ---------------------------------------------------------------------------
# JSON parser (shared)
# ---------------------------------------------------------------------------
def _parse_json(text: str, fallback_name: str) -> dict:
    """Best-effort JSON extraction from model response."""
    # 1) Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) Extract from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Find the largest JSON object in the text
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

    return _empty_result(fallback_name, f"Could not parse: {text[:200]}")


def _empty_result(name: str, desc: str = "") -> dict:
    d = {k: "" for k in RESULT_COLS}
    d["official_name"] = name
    d["description"] = desc
    return d


# =========================================================================
#  OPENAI provider
# =========================================================================
class OpenAIProvider:
    """Uses GPT + web search via the OpenAI Responses API."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: Missing package: openai")
            print("       Run:  pip install openai")
            sys.exit(1)

        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set.")
            print("       export OPENAI_API_KEY=sk-...")
            sys.exit(1)

        self.client = OpenAI()
        self.model = model or self.DEFAULT_MODEL

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def research(self, biz_name: str, country: str) -> dict:
        from openai import RateLimitError, APIError

        user_msg = (
            f'Research this optics / eyewear business: "{biz_name}" '
            f'in {country}.  Return JSON only.'
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search_preview"}],
                instructions=SYSTEM_PROMPT,
                input=user_msg,
            )
        except RateLimitError:
            print("  Rate-limited — waiting 30 s …")
            time.sleep(30)
            return self.research(biz_name, country)
        except APIError as exc:
            return _empty_result(biz_name, f"API error: {exc}")

        text = response.output_text or ""
        return _parse_json(text.strip(), biz_name)


# =========================================================================
#  ANTHROPIC provider
# =========================================================================
class AnthropicProvider:
    """Uses Claude + web search via the Anthropic Messages API."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, model: str | None = None):
        try:
            import anthropic as _anth
            self._anthropic = _anth
        except ImportError:
            print("ERROR: Missing package: anthropic")
            print("       Run:  pip install anthropic")
            sys.exit(1)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set.")
            print("       export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)

        self.client = _anth.Anthropic()
        self.model = model or self.DEFAULT_MODEL

    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"

    def research(self, biz_name: str, country: str) -> dict:
        user_msg = (
            f'Research this optics / eyewear business: "{biz_name}" '
            f'in {country}.  Return JSON only.'
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
        except self._anthropic.RateLimitError:
            print("  Rate-limited — waiting 30 s …")
            time.sleep(30)
            return self.research(biz_name, country)
        except self._anthropic.APIError as exc:
            return _empty_result(biz_name, f"API error: {exc}")

        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        full_text = "\n".join(text_parts).strip()
        return _parse_json(full_text, biz_name)


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------
def save_progress(all_results: dict[str, pd.DataFrame], dest: Path):
    with pd.ExcelWriter(dest, engine="openpyxl") as w:
        for sheet, df in all_results.items():
            df.to_excel(w, sheet_name=sheet, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Optics Business Intelligence Agent — AI + Web Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
EXAMPLES
  # OpenAI  (cheapest — gpt-4o-mini ≈ $0.15/1M input tokens)
  export OPENAI_API_KEY=sk-...
  python business_intel_agent.py optics_shops.xlsx --provider openai

  # OpenAI with GPT-4o for better quality
  python business_intel_agent.py optics_shops.xlsx --provider openai --model gpt-4o

  # Anthropic Claude
  export ANTHROPIC_API_KEY=sk-ant-...
  python business_intel_agent.py optics_shops.xlsx --provider anthropic

  # Anthropic Haiku (cheaper)
  python business_intel_agent.py optics_shops.xlsx --provider anthropic --model claude-haiku-4-5-20251001

COST COMPARISON (approx. per 100 shops)
  gpt-4o-mini + web search     ~$0.50       (cheapest)
  claude-haiku + web search    ~$1-2
  gpt-4o + web search          ~$3-5
  claude-sonnet + web search   ~$5-8

ENVIRONMENT
  OPENAI_API_KEY      For --provider openai
  ANTHROPIC_API_KEY   For --provider anthropic
""",
    )
    ap.add_argument("input_file", help="Excel file with optics shop names")
    ap.add_argument("-o", "--output", help="Output Excel  [default: <input>_intel.xlsx]")
    ap.add_argument("-n", "--name-column", help="Column with business names")
    ap.add_argument("-d", "--delay", type=float, default=1.0,
                    help="Seconds between API calls  [default: 1.0]")
    ap.add_argument("--countries", nargs="+",
                    help="Only process these countries (e.g. norway denmark)")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                    help="AI provider  [default: openai]")
    ap.add_argument("--model",
                    help="Model override (default: gpt-4o-mini for openai, "
                         "claude-sonnet-4-6 for anthropic)")
    args = ap.parse_args()

    # --- Pre-flight -------------------------------------------------------
    print()
    print("Optics Business Intelligence Agent")
    print("=" * 35)

    src = Path(args.input_file)
    if not src.exists():
        print(f"ERROR: File not found: {src}")
        sys.exit(1)

    dest = Path(args.output) if args.output else src.with_name(f"{src.stem}_intel.xlsx")

    # --- Init provider ----------------------------------------------------
    if args.provider == "openai":
        provider = OpenAIProvider(model=args.model)
    else:
        provider = AnthropicProvider(model=args.model)

    print(f"  Provider: {provider.name}")
    print(f"  Input:    {src}")
    print(f"  Output:   {dest}")
    print()

    # --- Read Excel -------------------------------------------------------
    xls = pd.ExcelFile(src)
    print(f"  Sheets: {xls.sheet_names}")

    sheets: list[tuple[str, str, pd.DataFrame]] = []
    for idx, sn in enumerate(xls.sheet_names):
        df = pd.read_excel(src, sheet_name=sn)
        country = detect_country(sn, idx)
        if args.countries:
            if country.lower() not in [c.lower() for c in args.countries]:
                print(f"  Skipping {sn} ({country})")
                continue
        print(f"  {sn} → {country}  ({len(df)} shops)")
        sheets.append((country, sn, df))
    print()

    # --- Research each business -------------------------------------------
    all_results: dict[str, pd.DataFrame] = {}
    total_calls = 0

    for country, sheet_name, df in sheets:
        name_col = args.name_column or detect_name_col(df)
        names = df[name_col].dropna().astype(str).str.strip().tolist()
        names = [n for n in names if n and n.lower() != "nan"]

        print(f"{'=' * 60}")
        print(f"  {country.upper()}  |  {len(names)} businesses  |  col '{name_col}'")
        print(f"{'=' * 60}")

        rows = []
        pbar = tqdm(names, desc=country, unit="biz", leave=True)
        for biz_name in pbar:
            pbar.set_postfix_str(biz_name[:30])

            result = provider.research(biz_name, country)
            result["_input_name"] = biz_name
            result["_country"] = country
            rows.append(result)
            total_calls += 1

            chain = result.get("chain", "")
            status = f"→ {chain}" if chain else "→ ?"
            pbar.set_postfix_str(f"{biz_name[:20]} {status}")

            time.sleep(args.delay)

        result_df = pd.DataFrame(rows)

        col_order = ["_input_name", "_country"] + RESULT_COLS
        for c in col_order:
            if c not in result_df.columns:
                result_df[c] = ""
        extra = [c for c in result_df.columns if c not in col_order]
        result_df = result_df[col_order + extra]

        all_results[sheet_name] = result_df

        save_progress(all_results, dest)
        print(f"  Saved progress → {dest}\n")

    # --- Summary ----------------------------------------------------------
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for sn, df in all_results.items():
        found = len(df[df["description"] != "No information found online"])
        print(f"  {sn:20s}  {found}/{len(df)} researched")
        if "chain" in df.columns:
            chains = df["chain"].value_counts()
            for chain, count in chains.head(5).items():
                if chain:
                    print(f"    {chain}: {count}")
    print(f"\n  Provider: {provider.name}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Output: {dest}")
    print("=" * 60)


if __name__ == "__main__":
    main()
