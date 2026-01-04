import json
import os
import sys
from pathlib import Path

from pypdf import PdfReader
from langfuse.openai import OpenAI
from config import OPENAI_API_KEY
from langfuse import observe


# ---------------------------
# 1. CONFIG
# ---------------------------

MODEL_NAME = "gpt-4o-mini"   # fast + cheap + reliable
MAX_PDF_CHARS = 30000       # avoid token explosion

EXPECTED_SCHEMA = {
    "invoice_number": "string or null",
    "invoice_date": "string (YYYY-MM-DD) or null",
    "due_date": "string (YYYY-MM-DD) or null",
    "seller_name": "string or null",
    "buyer_name": "string or null",
    "total_amount": "number or null",
    "currency": "string or null",
    "tax_amount": "number or null"
}


# ---------------------------
# 2. PDF TEXT EXTRACTION
# ---------------------------
@observe(name="pdf-processing")
def extract_text_from_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    all_text = []

    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"Warning: page {idx+1} failed: {e}")
            text = ""
        all_text.append(text)

    full_text = "\n".join(all_text)

    if len(full_text) > MAX_PDF_CHARS:
        full_text = full_text[:MAX_PDF_CHARS]

    return full_text.strip()


# ---------------------------
# 3. PROMPT ENGINEERING
# ---------------------------

SYSTEM_PROMPT = f"""
You are an expert document information extraction engine.

Rules:
- Extract ONLY the requested fields
- Output STRICT JSON (no explanation, no markdown)
- If a value is missing or uncertain, use null
- Numbers must be numeric (no currency symbols)
- Dates must be ISO format YYYY-MM-DD
- if currency is not mentioned  then use country  mentioned in text to get the  accurate currency code of that country, use city, country mention in section to whome invoice is sent

JSON schema:
{json.dumps(EXPECTED_SCHEMA, indent=2)}
"""

USER_PROMPT_TEMPLATE = """
Extract the required fields from the following invoice text.

Invoice text:
\"\"\"
{invoice_text}
\"\"\"
"""


# ---------------------------
# 4. OPENAI CALL
# ---------------------------
@observe(name="open-ai-generation")
def extract_fields_with_openai(invoice_text: str) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    invoice_text=invoice_text
                )
            }
        ]
    )

    raw_output = response.choices[0].message.content.strip()

    # Safe JSON parsing
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw_output[start:end+1])
        raise ValueError("Model did not return valid JSON")


# ---------------------------
# 5. MAIN
# ---------------------------

def main():
    
    pdf_path ="invoice1.pdf"

    print("📄 Reading PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)

    print("🤖 Calling OpenAI GPT...")
    extracted_data = extract_fields_with_openai(pdf_text)

    print("\n✅ Extracted JSON:")
    print(json.dumps(extracted_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()