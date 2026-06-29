# Standard library imports
import os

# Local imports
import gst_basemodel

# Google Gemini API client and configuration types
from google import genai
from google.genai import types

# Loads environment variables from .env file for secure API key management
from dotenv import load_dotenv

# Exception class for handling Pydantic validation errors
from pydantic import ValidationError

# Load environment variables from .env file
# This allows us to keep the API key out of source code
load_dotenv()

# Initialize Gemini API client with the API key from environment variable
# The key "llm-metal-gemini" should be set in your .env file
client = genai.Client(api_key=os.getenv("llm-metal-gemini"))

# System instruction tells Gemini how to behave when extracting data
# This acts as the "role" or "context" for the AI model
# It ensures consistent extraction logic and format across all API calls
SYSTEM_INSTRUCTION = (
    "You extract structured GST invoice data from messy, unstructured text "
    "(OCR output, pasted emails, scanned-invoice dumps, etc.). "
    "Always return your best-effort extraction even if some fields are unclear -- "
    "use sensible defaults (e.g. empty string) rather than omitting a field. "
    "Dates must be in YYYY-MM-DD format. "
    "'amount' on the top-level object is the total invoice amount. "
    "Each line item's 'amount' is quantity * unit_price for that line."
)


# Sample messy invoice text for testing
# This simulates real OCR output or pasted invoice data that needs parsing
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
    """
    Extracts structured GST invoice data from unstructured text using Gemini.
    
    Args:
        messy_text: Raw invoice text (OCR, pasted email, etc.)
        max_retries: Number of retry attempts if validation fails (default 3)
    
    Returns:
        Details: A validated Pydantic model instance with extracted invoice data
    
    Raises:
        ValidationError: If all retry attempts fail to produce valid JSON matching the schema
    """
    # Configure Gemini to return JSON that matches the Details pydantic model schema
    # response_mime_type="application/json" tells Gemini to output JSON only
    # response_schema=gst_basemodel.Details constrains the output to match our schema
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=gst_basemodel.Details,
    )

    # Initial prompt to Gemini - ask it to extract invoice details
    prompt = f"Extract the GST invoice details from this text:\n\n{messy_text}"
    
    # Variable to store the last exception in case all retries fail
    last_error: Exception | None = None

    # Retry loop: attempt up to max_retries times to get valid JSON
    for attempt in range(1, max_retries + 1):
        # Call Gemini API with the configured schema constraint
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        # Extract the raw JSON string from Gemini's response
        raw_json = response.text

        try:
            # Attempt to parse the JSON string and validate it against the Details schema
            # model_validate_json() does two things:
            #   1. Parses the JSON string
            #   2. Validates all fields match the Pydantic model definition
            # If successful, returns a Details model instance with all fields properly typed
            details = gst_basemodel.Details.model_validate_json(raw_json)
            
            # Success! Return the validated model instance
            return details
            
        except ValidationError as exc:
            # Gemini returned valid JSON, but it doesn't match our schema
            # Store the error and retry with an error-aware prompt
            last_error = exc
            print("Schema Validation failed: \n{exc}\n")

            # Build a new prompt that tells Gemini what went wrong
            # This helps Gemini self-correct and avoid the same mistake
            prompt = (
                f"Your previous JSON output failed schema validation with this error:\n"
                f"{exc}\n\n"
                f"Previous output was:\n{raw_json}\n\n"
                f"Re-extract the GST invoice details from the ORIGINAL text below, "
                f"fixing the schema issue. Return ONLY valid JSON matching the schema.\n\n"
                f"Original text:\n{messy_text}"
            )
            
        except Exception as exc:
            # Gemini's response wasn't even valid JSON
            # Store the error and retry asking for properly formatted JSON
            last_error = exc
            print(
                f"[attempt {attempt}/{max_retries}] could not parse response as JSON: {exc}\n"
            )

            # Tell Gemini to return valid JSON only (no markdown, no extra text)
            prompt = (
                f"Your previous response was not valid JSON. Return ONLY valid JSON "
                f"matching the required schema, with no markdown fences or commentary.\n\n"
                f"Original text:\n{messy_text}"
            )

    # All retries exhausted - raise the last error encountered
    raise last_error


if __name__ == "__main__":
    # Alternative sample invoice for testing (different format from messy_text above)
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

    # Call the extractor function
    # Returns a Details pydantic model instance with all fields validated and typed
    result = extractor(sample_messy_text)

    # Convert the pydantic model instance to a JSON string with pretty formatting
    # indent=2 makes the JSON human-readable with 2-space indentation
    # This is useful for logging, debugging, or sending to other systems
    print(result.model_dump_json(indent=2))
