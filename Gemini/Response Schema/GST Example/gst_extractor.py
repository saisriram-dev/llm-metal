import os
import gst_basemodel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv()
client = genai.Client(api_key=os.getenv("llm-metal-gemini"))

SYSTEM_INSTRUCTION = (
    "You extract structured GST invoice data from messy, unstructured text "
    "(OCR output, pasted emails, scanned-invoice dumps, etc.). "
    "Always return your best-effort extraction even if some fields are unclear -- "
    "use sensible defaults (e.g. empty string) rather than omitting a field. "
    "Dates must be in YYYY-MM-DD format. "
    "'amount' on the top-level object is the total invoice amount. "
    "Each line item's 'amount' is quantity * unit_price for that line."
)


messy_text = """
Invoice Receipt
Vendor: TechCorp Solutions Pvt Ltd
Date of Issue: 2023-10-15
GSTIN: 29ABCDE1234F1Z5
Total Amount: 12500.00

Items purchased:
1. Premium Widgets - 2.0 qty @ 5000.00/unit = 10000.00
2. Express Shipping - 1.0 qty @ 2500.00/unit = 2500.00
"""


def extractor(messy_text: str, max_retries: int = 3):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=gst_basemodel.Details,
    )

    prompt = f"Extract the GST invoice details from this text:\n\n{messy_text}"
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        raw_json = response.text

        try:
            details = gst_basemodel.Details.model_validate_json(raw_json)
            return details
        except ValidationError as exc:
            last_error = exc
            print("Schema Validation failed: \n{exc}\n")

            prompt = (
                f"Your previous JSON output failed schema validation with this error:\n"
                f"{exc}\n\n"
                f"Previous output was:\n{raw_json}\n\n"
                f"Re-extract the GST invoice details from the ORIGINAL text below, "
                f"fixing the schema issue. Return ONLY valid JSON matching the schema.\n\n"
                f"Original text:\n{messy_text}"
            )
        except Exception as exc:  # e.g. response.text wasn't even valid JSON
            last_error = exc
            print(
                f"[attempt {attempt}/{max_retries}] could not parse response as JSON: {exc}\n"
            )

            prompt = (
                f"Your previous response was not valid JSON. Return ONLY valid JSON "
                f"matching the required schema, with no markdown fences or commentary.\n\n"
                f"Original text:\n{messy_text}"
            )

    raise last_error


if __name__ == "__main__":
    sample_messy_text = """
    TAX INVOICE
    M/s Sharma Traders & Co.
    GSTIN: 29ABCDE1234F1Z5
    Invoice Date: 14/03/2024
 
    Sl  Item                  Qty   Rate     Amt
    1   A4 Paper Ream         10    320.00   3200.00
    2   Stapler (Heavy Duty)  3     150.50   451.50
    3   Blue Ink Pens (Box)   5     90.00    450.00
 
    Sub total : 4101.50
    CGST 9%   : 369.14
    SGST 9%   : 369.14
    -----------------------------------------
    Grand Total: 4839.78
    Thank you for your business!
    """

    result = extractor(sample_messy_text)
    print(result.model_dump_json(indent=2))
